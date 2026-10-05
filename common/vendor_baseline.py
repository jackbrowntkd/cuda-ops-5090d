# -*- coding: utf-8 -*-
"""厂商基线批量测量：给一族算子量「库实现的天花板」，并给出差距参照。

设计原则（这三点决定了结果可不可信）：
1. **规模取自各题题面的 Constraints 里写明的性能测试规模**，不自己另选 ——
   最优配置随规模变化，换个规模就换结论。
2. **报告原始耗时**，同时给出 host launch floor 与比值；比值 < 4 的标注为不可信。
3. 区分两类基线：
   - **计算密集**（GEMM 家族）→ cuBLAS
   - **纯带宽**（elementwise）→ D2D memcpy 带宽 = 理论下限

用法: python vendor_baseline.py            # 全部
      python vendor_baseline.py gemm       # 只跑 GEMM 家族
      python vendor_baseline.py bw         # 只量带宽上限
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from localrun import Runner, _bind, cu_check, nvrtc_compile  # noqa: E402
from cublas import Handle  # noqa: E402

import ctypes  # noqa: E402
from ctypes import c_size_t, c_void_p, byref  # noqa: E402

cuMemcpyDtoD = _bind("cuMemcpyDtoD_v2", [c_void_p, c_void_p, c_size_t])

# 空 kernel：量 host launch floor（ctypes + cuLaunchKernel 的固定成本）
NOOP_SRC = r"""
__global__ void noop_probe(float* p) { if (p && threadIdx.x > 1024) p[0] = 1.f; }
"""

# 各题题面写明的性能测试规模
GEMM_CASES = [
    # (标签, 题号, 家族, M, N, K, batch)
    ("#002 fp32 GEMM",        2, "f32",      8192, 6144, 4096, 1),
    ("#022 fp16 GEMM",       22, "f16",      1024, 1024, 1024, 1),
    ("#030 fp32 batched",    30, "f32b",      256,  256,  256, 128),
    ("#057 fp16 batched",    57, "f16b",      256,  256,  256, 128),
    ("#032 int8 GEMM",       32, "int8",     8192, 4096, 2048, 1),
]


def host_floor_ms(r, repeat=20, warmup=5):
    ptx, arch, _ = nvrtc_compile(NOOP_SRC, name="noop.cu")
    R = Runner(ptx, arch, "")
    d = R.alloc(4)
    args = [d]
    R.launch("noop_probe", 1, 1, args)
    R.sync()
    ms = R.timeit(lambda: R.launch("noop_probe", 1, 1, args),
                  repeat=repeat, warmup=warmup)
    R.free(d)
    return ms


def memcpy_bandwidth(r, nbytes=512 << 20, repeat=20):
    """D2D memcpy 带宽（GB/s）。纯带宽型算子的理论上限就看它。"""
    a = r.alloc(nbytes)
    b = r.alloc(nbytes)
    cu_check(cuMemcpyDtoD(b, a, c_size_t(nbytes)), "d2d")
    r.sync()
    ms = r.timeit(lambda: cuMemcpyDtoD(b, a, c_size_t(nbytes)),
                  repeat=repeat, warmup=3)
    r.free(a)
    r.free(b)
    return 2.0 * nbytes / (ms * 1e-3) / 1e9, ms


def make_inputs(kind, M, N, K, batch, rng):
    """按家族造输入。返回 (dA, dB, dC, flop, 说明)。"""
    if kind in ("f32", "f32b"):
        A = (rng.standard_normal((batch, M, K)) * 0.5).astype(np.float32)
        B = (rng.standard_normal((batch, K, N)) * 0.5).astype(np.float32)
        C = np.zeros((batch, M, N), np.float32)
    elif kind in ("f16", "f16b"):
        A = (rng.standard_normal((batch, M, K)) * 0.5).astype(np.float16)
        B = (rng.standard_normal((batch, K, N)) * 0.5).astype(np.float16)
        C = np.zeros((batch, M, N), np.float16)
    elif kind == "int8":
        A = rng.integers(-128, 128, (batch, M, K), dtype=np.int8)
        B = rng.integers(-128, 128, (batch, K, N), dtype=np.int8)
        C = np.zeros((batch, M, N), np.int32)
    else:
        raise ValueError(kind)
    return A, B, C


def bench_case(r, blas, label, cid, kind, M, N, K, batch, oh):
    rng = np.random.default_rng(22)
    A, B, C = make_inputs(kind, M, N, K, batch, rng)
    dA, dB, dC = r.to_device(A), r.to_device(B), r.to_device(C)
    flop = 2.0 * M * N * K * batch

    if kind == "f32":
        call = lambda: blas.gemm_f32(dA, dB, dC, M, N, K, 1.0, 0.0)
    elif kind == "f16":
        call = lambda: blas.gemm_f16_f32acc(dA, dB, dC, M, N, K, 1.0, 0.0)
    elif kind == "f32b":
        call = lambda: blas.gemm_f32_batched(dA, dB, dC, M, N, K, batch, 1.0, 0.0)
    elif kind == "f16b":
        call = lambda: blas.gemm_f16_batched(dA, dB, dC, M, N, K, batch, 1.0, 0.0)
    elif kind == "int8":
        call = lambda: blas.gemm_int8(dA, dB, dC, M, N, K, 1, 0)

    try:
        call()
        r.sync()
    except Exception as e:
        print("%-20s  调用失败: %r" % (label, e))
        for p in (dA, dB, dC):
            r.free(p)
        return None

    ms = r.timeit(call, repeat=20, warmup=5)
    gf = flop / (ms * 1e-3) / 1e9
    ratio = ms / oh if oh else 0
    flag = "OK" if ratio >= 4 else "~host(%.1fx)" % ratio
    print("%-20s M=%-5d N=%-5d K=%-5d B=%-4d %8.3fms %11.1f GFLOPS  %-12s %s"
          % (label, M, N, K, batch, ms, gf, flag,
             "算力 %.2f TFLOP" % (flop / 1e12)))
    for p in (dA, dB, dC):
        r.free(p)
    return gf


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    ptx, arch, _ = nvrtc_compile(NOOP_SRC, name="base.cu")
    r = Runner(ptx, arch, "")
    print("设备:", Runner.devicename(), "| PTX arch:", arch)

    oh = host_floor_ms(r)
    print("host launch floor = %.3f us（实测耗时/它 < 4 的数字不可信）" % (oh * 1000))
    print()

    if what in ("all", "bw"):
        bw, ms = memcpy_bandwidth(r)
        print("=== 纯带宽型算子的理论上限（elementwise 家族的分母）===")
        print("  D2D memcpy 实测带宽 = %.1f GB/s（512MB×2, %.3f ms/次）" % (bw, ms))
        print("  → 某算子理论最短时间 = 读写总字节数 / %.1f GB/s" % bw)
        print("  → 例如 elementwise `C = f(A,B)`（fp32, N 个元素）：")
        for N in (1 << 24, 1 << 26):
            need = N * 4 * 3
            print("       N=2^%d: %d MB 读写 → 理论下限 %.3f ms"
                  % (int(np.log2(N)), need >> 20, need / bw / 1e6))
        print()

    if what in ("all", "gemm"):
        blas = Handle()
        print("=== GEMM 家族：cuBLAS 基线（规模取自各题题面的性能测试规模）===")
        print("%-20s %-31s %10s %14s %-12s %s"
              % ("题", "形状", "耗时", "吞吐", "可信度", "计算量"))
        print("-" * 100)
        for (label, cid, kind, M, N, K, batch) in GEMM_CASES:
            bench_case(r, blas, label, cid, kind, M, N, K, batch, oh)
        print()
        print("注：")
        print("  · #032 的 cuBLAS 调用是**纯 int8 GEMM**，不含 zero_point/scale 校正，")
        print("    只能当忽略零点校正的理论上界参照，不是逐位等价实现。")
        print("  · #030/#057 的 batch 题面只写了 K=M=N=256，未明示 batch 上限（约束 B ≤ 128），")
        print("    这里按 B=128（最大）取，以便给出吞吐上界。")
        blas.destroy()


if __name__ == "__main__":
    sys.exit(main())
