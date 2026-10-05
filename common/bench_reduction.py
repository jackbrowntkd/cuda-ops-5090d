# -*- coding: utf-8 -*-
"""归约 / 扫描族的对标：基线是「读这么多字节的理论下限」。

为什么这类算子的分母不是 cuBLAS：
  归约只读不写（输出只有 1 个数），所以理论上限就是
     t_min = 读入字节数 / D2D 带宽
  达标率 = t_min / 实测耗时。这是它唯一有意义的对标方式。

⚠️ 与 elementwise 同样的两个可信度判据（每行都打印）：
  · 数据量 / L2 >= 4 才谈 DRAM 带宽（5090 的 L2 有 96MB，小规模根本不碰 DRAM）
  · 实测 / host floor >= 4 才可信（归约的 stage2 只有 1 个 block，极快）

用法: python bench_reduction.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from localrun import Runner, nvrtc_compile   # noqa: E402

L2_BYTES = 96 << 20
BANDWIDTH_GBPS = 1514.9
HOST_FLOOR_MS = 6.5e-3
RBLOCK = 256
RMAXG = 1024


def run_reduce_sub(r, dIn, dOut, N, S, E):
    """#047 专用：stage1 的签名是 (dIn, part, S, E)，不是通用的 (ins..., part, N)。"""
    length = E - S + 1
    vec = ((S & 3) == 0) and (length >= 4)
    units = (length >> 2) if vec else length
    grid = min(65535, max(1, (units + RBLOCK - 1) // RBLOCK))
    part = r.alloc(grid * 4)
    r.launch("sub_stage1", grid, RBLOCK, [dIn, part, r.i(S), r.i(E)])
    r.launch("sub_stage2", 1, RBLOCK, [part, dOut, r.i(grid)])
    r.sync()
    r.free(part)


def run_reduce(r, stage1, ins, dOut, N, scale, grid_cap=RMAXG):
    """镜像各题 check.py 里的 run_reduce：stage1 出部分和 -> stage2 归约。
    注意 grid 被 capped（实测平台题的写法），所以 stage1 内部必须有 grid-stride 循环。"""
    grid = min(grid_cap, max(1, (N + RBLOCK - 1) // RBLOCK))
    part = r.alloc(grid * 4)
    r.launch(stage1, grid, RBLOCK, list(ins) + [part, r.i(N)])
    r.launch("stage2", 1, RBLOCK, [part, dOut, r.i(grid), r.f(scale)])
    r.sync()
    r.free(part)


def bench(r, label, subdir, stage1, make_ins, N, bytes_per_elem, note):
    path = os.path.join(os.path.dirname(HERE), subdir, "solution.cu")
    if not os.path.isfile(path):
        print("  %-28s 跳过（没有 solution.cu）" % label)
        return
    rr = Runner.from_file(path)
    dOut = rr.alloc(4)
    rng = np.random.default_rng(5)
    ins = make_ins(rr, rng, N)
    read_bytes = N * bytes_per_elem
    fn = lambda: run_reduce(rr, stage1, ins, dOut, N, 1.0 / max(1, N))
    try:
        fn(); rr.sync()
        ms = rr.timeit(fn, repeat=20, warmup=5)
    except Exception as e:
        print("  %-28s 失败: %r" % (label, e))
        return
    t_min = read_bytes / BANDWIDTH_GBPS / 1e9          # 秒
    util = t_min / (ms * 1e-3) * 100
    ratio_l2 = read_bytes / L2_BYTES
    floor_x = ms / HOST_FLOOR_MS
    if floor_x < 4:
        trust = "❌host绑定(%.1fx)" % floor_x
    elif ratio_l2 >= 4:
        trust = "OK"
    else:
        trust = "❌偏小(L2)"
    print("  %-28s N=%-11d %8.3fms  读 %6.1f MB  理论下限 %7.3fms  **达标 %5.1f%%**"
          "  L2比=%-4.1f abs/host=%-5.1f %s"
          % (label, N, ms, read_bytes / 1e6, t_min * 1e3, util, ratio_l2, floor_x, trust))
    print("      %s" % note)
    for p in ins:
        rr.free(p)
    rr.free(dOut)


def main():
    ptx, arch, _ = nvrtc_compile(
        "__global__ void noop(float* p){ if(p && threadIdx.x>1024) p[0]=1.f; }", name="b.cu")
    Runner(ptx, arch, "")
    print("设备: NVIDIA RTX 5090 D | 带宽上限 = %.1f GB/s | L2 = %d MB"
          % (BANDWIDTH_GBPS, L2_BYTES >> 20))
    print("达标率 = 理论下限(读入字节/带宽) / 实测耗时")
    print()
    print("=== 归约 / 扫描族 ===")
    bench(None, "#027 MSE", "027-Mean-Squared-Error", "stage1_mse",
          lambda rr, rng, N: [rr.to_device(rng.random(N, dtype=np.float32)),
                              rr.to_device(rng.random(N, dtype=np.float32))],
          50_000_000, 8, "读两个数组（predictions/targets），题面性能规模 N=5e7")
    path = os.path.join(os.path.dirname(HERE), "047-Subarray-Sum", "solution.cu")
    if os.path.isfile(path):
        rr = Runner.from_file(path)
        N = 100_000_000
        dIn = rr.to_device(np.random.default_rng(5).random(N, dtype=np.float32))
        dOut = rr.alloc(4)
        fn = lambda: run_reduce_sub(rr, dIn, dOut, N, 0, N - 1)
        fn(); rr.sync()
        ms = rr.timeit(fn, repeat=20, warmup=5)
        t_min = N * 4 / BANDWIDTH_GBPS / 1e9
        print("  %-28s N=%-11d %8.3fms  读 %6.1f MB  理论下限 %7.3fms  **达标 %5.1f%%**"
              "  L2比=%-4.1f abs/host=%-5.1f %s"
              % ("#047 Subarray-Sum", N, ms, N * 4 / 1e6, t_min * 1e3,
                 t_min / (ms * 1e-3) * 100, N * 4 / L2_BYTES, ms / HOST_FLOOR_MS,
                 "OK" if ms / HOST_FLOOR_MS >= 4 else "❌host绑定"))
        print("      读一个数组；stage1 签名是 (S,E) 而非 (N)，所以单独跑")
        rr.free(dIn); rr.free(dOut)


if __name__ == "__main__":
    sys.exit(main())
