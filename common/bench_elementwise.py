# -*- coding: utf-8 -*-
"""纯带宽型算子（elementwise）的对标：**达标率 = 实达带宽 / D2D memcpy 上限**。

为什么这类算子的基线不是 cuBLAS：
  它们的理论上限就是显存带宽 —— 读写这么多字节，最快就是 memcpy 那么快。
  所以唯一有意义的指标是「达到了 memcpy 带宽的百分之几」。

⚠️ 关键前提：**数据量必须远大于 L2**，否则测的是 L2 带宽而不是 DRAM 带宽。
   RTX 5090 D 的 L2 = 96 MB，所以：
     - N=2^24（fp32 单向 64MB）→ 基本够
     - N=2^26（256MB）→ 稳
     - N<=10^6 → 全部落在 L2 里，达标率会虚高（>100%），**别当 DRAM 结论**
   下面每行都标了"数据量/L2 比"，比值 < 1 的读数不要用来下 DRAM 结论。

用法: python bench_elementwise.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from localrun import Runner, run_vec_then_tail, run_simple   # noqa: E402

L2_BYTES = 96 << 20          # RTX 5090 D 的 L2
BANDWIDTH_GBPS = 1514.9      # vendor_baseline.py 实测的 D2D memcpy 上限
HOST_FLOOR_MS = 6.5e-3       # ctypes 单次 launch 的 host 固定开销

# ⚠️ 本类算子的两个"不可信陷阱"（都必须在报告里标出来，别只看达标率）：
#   陷阱1：数据量 < L2（5090 的 L2 有 96MB！）→ 测到的是 L2 带宽（~3.8TB/s），
#          达标率会虚高到 200~330%，完全不是 DRAM 结论。
#   陷阱2：耗时贴着 host launch floor → 测到的是 host 开销，与 kernel 无关。
#          elementwise 的 kernel 极快，实测 0.008~0.010ms 就已经撞上 6.5us 的地板了。

# (标签, 目录, 分派类型, N 列表, 每元素字节数, 说明)
# 每元素字节数：读+写的总字节 / 元素数
CASES = [
    ("#001 向量加  c=a+b", "001-Vector-Addition", "vtt", "vec_add4", "vec_add1",
     [1 << 20, 1 << 22, 1 << 24, 1 << 26], 12, "读 2 写 1（fp32）"),
    ("#021 ReLU    y=max(x,0)", "021-ReLU", "vtt", "relu4", "relu1",
     [1 << 20, 1 << 22, 1 << 24, 1 << 26], 8, "读 1 写 1（fp32）"),
    ("#023 LeakyReLU", "023-Leaky-ReLU", "vtt", "lrelu4", "lrelu1",
     [1 << 20, 1 << 22, 1 << 24, 1 << 26], 8, "读 1 写 1（fp32）"),
    ("#031 矩阵拷贝 y=x", "031-Matrix-Copy", "vtt", "copy4", "copy1",
     [512 * 512, 1024 * 1024, 2048 * 2048], 8, "读 1 写 1（N<=4096，偏小）"),
    ("#008 矩阵加  C=A+B", "008-Matrix-Addition", "vtt", "add4", "add1",
     [512 * 512, 1024 * 1024, 2048 * 2048], 12, "读 2 写 1"),
    # 下面两题的 N 上限很小（10^4 / 10^5），数据全在 L2 里 —— 只会被标成不可信，
    # 保留是为了展示"规模不够就别谈带宽"这件事。
    ("#052 SiLU    y=x*sigmoid(x)", "052-Sigmoid-Linear-Unit", "simple", "silu_kernel", None,
     [1 << 12, 1 << 13], 8, "读 1 写 1（N<=1e4，必落 L2）"),
    ("#054 SwiGLU  gate*up", "054-Swish-Gated-Linear-Unit", "simple", "swiglu_kernel", None,
     [1 << 14, 1 << 16], 6, "读 N 写 N/2（N<=1e5，必落 L2）"),
]


def n_sizes(sq):
    return [n * n for n in sq]


def run_case(r, label, subdir, kind, k4, k1, Ns, bytes_per, note):
    path = os.path.join(os.path.dirname(HERE), subdir, "solution.cu")
    if not os.path.isfile(path):
        print("  %-26s 跳过（找不到 solution.cu）" % label)
        return
    rr = Runner.from_file(path)
    print("  %s   %s" % (label, note))
    for N in Ns:
        rng = np.random.default_rng(3)
        A = rng.random(N, dtype=np.float32)
        B = rng.random(N, dtype=np.float32)
        dA, dB = rr.to_device(A), rr.to_device(B)
        dC = rr.alloc(N * 4)
        bytes_moved = N * bytes_per
        if kind == "vtt":
            ins = [dA, dB] if bytes_per == 12 else [dA]
            fn = lambda: run_vec_then_tail(rr, k4, k1, ins, dC, N)
        else:
            fn = lambda: run_simple(rr, k4, [dA], dC, N)
        try:
            fn(); rr.sync()
            ms = rr.timeit(fn, repeat=20, warmup=5)
        except Exception as e:
            print("     N=%-9d 失败: %r" % (N, e))
            for p in (dA, dB, dC):
                rr.free(p)
            continue
        bw = bytes_moved / (ms * 1e-3) / 1e9
        util = bw / BANDWIDTH_GBPS * 100
        ratio_l2 = bytes_moved / L2_BYTES
        floor_x = ms / HOST_FLOOR_MS
        if floor_x < 4:
            trust = "❌host绑定(%.1fx)" % floor_x
        elif ratio_l2 >= 4:
            trust = "OK"
        elif ratio_l2 >= 1:
            trust = "偏小(L2)"
        else:
            trust = "❌落在L2"
        print("     N=%-9d %7.3fms  %8.1f GB/s  **%5.1f%%**  数据量/L2=%-5.1f 实测/host=%-5.1f %s"
              % (N, ms, bw, util, ratio_l2, floor_x, trust))
        for p in (dA, dB, dC):
            rr.free(p)
    print()


def main():
    ptx = r"""
__global__ void noop_probe(float* p) { if (p && threadIdx.x > 1024) p[0] = 1.f; }
"""
    from localrun import nvrtc_compile
    ptxc, arch, _ = nvrtc_compile(ptx, name="base.cu")
    Runner(ptxc, arch, "")
    print("设备: NVIDIA RTX 5090 D | 带宽上限 = %.1f GB/s | L2 = %d MB"
          % (BANDWIDTH_GBPS, L2_BYTES >> 20))
    print("指标：**达标率 = 实达带宽 / memcpy 上限**；「数据量/L2」< 1 的读数别当 DRAM 结论")
    print()
    print("=== 纯带宽型算子：达到 D2D 上限的百分之几 ===")
    for (label, subdir, kind, k4, k1, Ns, bpe, note) in CASES:
        run_case(None, label, subdir, kind, k4, k1, Ns, bpe, note)


if __name__ == "__main__":
    sys.exit(main())
