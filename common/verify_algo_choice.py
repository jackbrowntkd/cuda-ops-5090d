# -*- coding: utf-8 -*-
"""验证「我们比 cuBLAS 快」是真赢，还是因为调用时没选到 cuBLAS 的最优 algorithm。

背景：cuBLAS 的 GemmEx / GemmStridedBatchedEx 接受一个 `cublasGemmAlgo_t` 参数。
传 CUBLAS_GEMM_DEFAULT(-1) 时交给启发式选；也可以显式指定 ALGO0..ALGO15(0..15)
或 CUBLAS_GEMM_DFALT_TENSOR_OP(99)。

如果某个 algo 能把 cuBLAS 从 39.9 抬到 60 TFLOPS，那"我们 1.16x"就是调用方式的
artifact，必须如实纠正而不是当成战绩。

用法: python verify_algo_choice.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from localrun import Runner, nvrtc_compile   # noqa: E402
from cublas import Handle                    # noqa: E402

_NOOP = r"""
__global__ void noop_probe(float* p) { if (p && threadIdx.x > 1024) p[0] = 1.f; }
"""

ALGOS = [("DEFAULT", -1), ("TENSOR_OP", 99)] + [("ALGO%d" % i, i) for i in range(1)]


def sweep_batched(r, blas, kind, M, N, K, batch):
    rng = np.random.default_rng(22)
    if kind == "f32":
        A = (rng.standard_normal((batch, M, K)) * 0.5).astype(np.float32)
        B = (rng.standard_normal((batch, K, N)) * 0.5).astype(np.float32)
        C = np.zeros((batch, M, N), np.float32)
        mk = lambda a: blas.gemm_f32_batched(a[0], a[1], a[2], M, N, K, batch, 1.0, 0.0, algo=a[3])
    else:
        A = (rng.standard_normal((batch, M, K)) * 0.5).astype(np.float16)
        B = (rng.standard_normal((batch, K, N)) * 0.5).astype(np.float16)
        C = np.zeros((batch, M, N), np.float16)
        mk = lambda a: blas.gemm_f16_batched(a[0], a[1], a[2], M, N, K, batch, 1.0, 0.0, algo=a[3])
    dA, dB, dC = r.to_device(A), r.to_device(B), r.to_device(C)
    flop = 2.0 * M * N * K * batch
    print("  %s  batched  M=%d N=%d K=%d B=%d" % (kind, M, N, K, batch))
    best = None
    for name, algo in ALGOS:
        try:
            mk((dA, dB, dC, algo))
            r.sync()
        except Exception:
            print("    %-10s  不支持" % name)
            continue
        ms = r.timeit(lambda: mk((dA, dB, dC, algo)), repeat=10, warmup=3)
        gf = flop / (ms * 1e-3) / 1e9
        mark = ""
        if best is None or gf > best[1]:
            best = (name, gf); mark = "  <= 目前最好"
        print("    %-10s %7.3fms %10.1f GFLOPS%s" % (name, ms, gf, mark))
    print("    → 最优 algo = %s (%.1f GFLOPS)" % (best[0], best[1]))
    for p in (dA, dB, dC):
        r.free(p)
    return best


def main():
    ptx, arch, _ = nvrtc_compile(_NOOP, name="base.cu")
    r = Runner(ptx, arch, "")
    blas = Handle()
    print("设备:", Runner.devicename())
    print()
    print("=== #030 fp32 batched（我们 46 176 GFLOPS）===")
    sweep_batched(r, blas, "f32", 256, 256, 256, 128)
    print()
    print("=== #057 fp16 batched（我们 115 367 GFLOPS）===")
    sweep_batched(r, blas, "f16", 256, 256, 256, 128)
    print()
    print("=== 对照：大尺寸单发（看 cuBLAS 在充分规模下的实力）===")
    print("  fp32 8192x6144x4096 单发 ≈ 61 200 GFLOPS")
    print("  fp16 4096^3 单发      ≈ 212 600 GFLOPS")
    blas.destroy()


if __name__ == "__main__":
    sys.exit(main())
