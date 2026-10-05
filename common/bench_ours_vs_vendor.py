# -*- coding: utf-8 -*-
"""我们的实现 vs 厂商库：逐题量化差距（各题都用**题面写明的性能规模**）。

为什么单独一个脚本：每题 `check.py` 里的 kernel 名 / grid / block / 形参约定都不一样
（#002 的 "N" 是收缩维，#030 的 batch 在 gridDim.z，……），
自动反射不靠谱，这里用**显式声明**把它们记下来 —— 顺带也是分派方式的文档。

用法: python bench_ours_vs_vendor.py            # 全部
      python bench_ours_vs_vendor.py 2 30       # 只跑指定题号
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from localrun import Runner, nvrtc_compile  # noqa: E402
from cublas import Handle  # noqa: E402

WORK = os.path.abspath(os.path.join(HERE, ".."))
HOST_FLOOR_MS = 6.5e-3      # vendor_baseline.py 实测的 host launch floor（ms）

_NOOP = r"""
__global__ void noop_probe(float* p) { if (p && threadIdx.x > 1024) p[0] = 1.f; }
"""


def flag(ms):
    r = ms / HOST_FLOOR_MS
    return "OK" if r >= 4 else "~host(%.1fx)" % r


def run_ours(r, kernel, grid, block, args, repeat=20, warmup=5, fn=None):
    if fn is None:
        def fn():
            r.launch(kernel, grid, block, args)
    fn()
    r.sync()
    return r.timeit(fn, repeat=repeat, warmup=warmup)


# --------------------------------------------------------------- 各题
# 注意 #002 的形参约定：A(M×N) @ B(N×K) = C(M×K)，**收缩维是 N**。
# 而 _tools/cublas.py 的接口是 C(M'×N') = A(M'×K') × B(K'×N')，
# 所以 #002 要对齐成 (M'=M, N'=K, K'=N) —— 这里最容易写错，专门注明。


def case_002(r, blas):
    """#002 fp32 Matrix-Multiplication（A(M,N) @ B(N,K)，收缩维 N）"""
    M, N, K = 8192, 6144, 4096
    path = os.path.join(WORK, "002-Matrix-Multiplication", "solution.cu")
    rr = Runner.from_file(path)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((M, N)).astype(np.float32)
    B = rng.standard_normal((N, K)).astype(np.float32)
    dA, dB = rr.to_device(A), rr.to_device(B)
    dC = rr.alloc(M * K * 4)
    flop = 2.0 * M * N * K
    ours_ms = run_ours(rr, "matmul_v10",
                       ((K + 127) // 128, (M + 127) // 128, 1), (256, 1, 1),
                       [dA, dB, dC, rr.i(M), rr.i(N), rr.i(K)])
    # vendor：C(M×K) = A(M×N) × B(N×K) -> 传 (M, K, N)
    v = rr.timeit(lambda: blas.gemm_f32(dA, dB, dC, M, K, N, 1.0, 0.0),
                  repeat=20, warmup=5)
    for p in (dA, dB, dC):
        rr.free(p)
    return flop, ours_ms, v, "语义等价（同 fp32 GEMM）"


def case_030(r, blas):
    """#030 fp32 batched GEMM（B 在 gridDim.z）"""
    BATCH, M, N, K = 128, 256, 256, 256
    path = os.path.join(WORK, "030-Batched-Matrix-Multiplication", "solution.cu")
    rr = Runner.from_file(path)
    rng = np.random.default_rng(39)
    A = rng.standard_normal((BATCH, M, K)).astype(np.float32)
    Bm = rng.standard_normal((BATCH, K, N)).astype(np.float32)
    dA, dB = rr.to_device(A), rr.to_device(Bm)
    dC = rr.alloc(BATCH * M * N * 4)
    flop = 2.0 * M * N * K * BATCH
    ours_ms = run_ours(rr, "bmm_v10",
                       ((N + 127) // 128, (M + 127) // 128, BATCH), (256, 1, 1),
                       [dA, dB, dC, rr.i(M), rr.i(N), rr.i(K)])
    v = rr.timeit(lambda: blas.gemm_f32_batched(dA, dB, dC, M, N, K, BATCH, 1.0, 0.0),
                  repeat=20, warmup=5)
    for p in (dA, dB, dC):
        rr.free(p)
    return flop, ours_ms, v, "语义等价（同 fp32 batched GEMM）"


def case_032(r, blas):
    """#032 INT8 Quantized MatMul（带 scale/zero_point）"""
    M, N, K = 8192, 4096, 2048
    path = os.path.join(WORK, "032-INT8-Quantized-MatMul", "solution.cu")
    rr = Runner.from_file(path)
    rng = np.random.default_rng(22)
    A = rng.integers(-128, 128, (M, K), dtype=np.int8)
    Bm = rng.integers(-128, 128, (K, N), dtype=np.int8)
    dA, dB = rr.to_device(A), rr.to_device(Bm)
    dC = rr.alloc(M * N)          # int8 输出
    dC32 = rr.alloc(M * N * 4)    # cuBLAS 的 int32 输出
    dSumA, dSumB = rr.alloc(M * 4), rr.alloc(N * 4)
    flop = 2.0 * M * N * K

    def ours_call():
        rr.launch("i8_rowsum", ((M + 255) // 256,), (256,),
                  [dA, dSumA, rr.i(M), rr.i(K)])
        rr.launch("i8_colsum", (N,), (256,), [dB, dSumB, rr.i(K), rr.i(N)])
        rr.launch("i8_gemm_dp4a", ((N + 63) // 64, (M + 63) // 64), (256,),
                  [dA, dB, dC, dSumA, dSumB, rr.i(M), rr.i(N), rr.i(K),
                   rr.f(1.0), rr.f(1.0), rr.f(1.0), rr.i(0), rr.i(0), rr.i(0)])

    ours_ms = run_ours(rr, "__fn__", None, None, None, fn=ours_call)
    v = rr.timeit(lambda: blas.gemm_int8(dA, dB, dC32, M, N, K, 1, 0),
                  repeat=20, warmup=5)
    for p in (dA, dB, dC, dC32, dSumA, dSumB):
        rr.free(p)
    return flop, ours_ms, v, "⚠️ 非严格等价：cuBLAS 不含 zero_point/scale 校正"


def case_057(r, blas):
    """#057 FP16 batched GEMM（批在 gridDim.z）"""
    BATCH, M, N, K = 128, 256, 256, 256
    path = os.path.join(WORK, "057-FP16-Batched-Matrix-Multiplication", "solution.cu")
    rr = Runner.from_file(path)
    rng = np.random.default_rng(22)
    A = (rng.standard_normal((BATCH, M, K)) * 0.5).astype(np.float16)
    Bm = (rng.standard_normal((BATCH, K, N)) * 0.5).astype(np.float16)
    dA, dB = rr.to_device(A), rr.to_device(Bm)
    dC = rr.alloc(BATCH * M * N * 2)
    flop = 2.0 * M * N * K * BATCH
    ours_ms = run_ours(rr, "fp16_bmm_wmma",
                       ((N + 63) // 64, (M + 63) // 64, BATCH), (128, 1, 1),
                       [dA, dB, dC, rr.i(M), rr.i(N), rr.i(K)])
    v = rr.timeit(lambda: blas.gemm_f16_batched(dA, dB, dC, M, N, K, BATCH, 1.0, 0.0),
                  repeat=20, warmup=5)
    for p in (dA, dB, dC):
        rr.free(p)
    return flop, ours_ms, v, "语义等价（同 fp16 batched，fp32 累加）"


CASES = {
    2: ("#002 fp32 GEMM", case_002),
    30: ("#030 fp32 batched", case_030),
    32: ("#032 int8 MatMul", case_032),
    57: ("#057 fp16 batched", case_057),
}


def main():
    want = [int(x) for x in sys.argv[1:]] or sorted(CASES)
    # ⚠️ 顺序不能反：必须先建 CUDA context（第一个 Runner 会 cuCtxCreate，并把它设为
    # 当前线程的 current context），再 cublasCreate —— 否则 handle 绑到错误的 context，
    # 后面报 `CUBLAS_STATUS_INTERNAL_ERROR(14)`。这条写在 _tools/cublas.py 的 docstring 里。
    ptx, arch, _ = nvrtc_compile(_NOOP, name="base.cu")
    Runner(ptx, arch, "")
    blas = Handle()
    print("设备: NVIDIA RTX 5090 D | host launch floor = %.1f us" % (HOST_FLOOR_MS * 1000))
    print()
    print("%-20s %12s %12s %10s %10s %9s %s"
          % ("题", "我们", "厂商库", "GFLOPS我们", "GFLOPS厂商", "差距", "可信度"))
    print("-" * 108)
    rows = []
    for cid in want:
        if cid == 22:
            continue
        label, fn = CASES[cid]
        try:
            flop, ours, vend, note = fn(None, blas)
        except Exception as e:
            print("%-20s 失败: %r" % (label, e))
            continue
        go = flop / (ours * 1e-3) / 1e9
        gv = flop / (vend * 1e-3) / 1e9
        print("%-20s %10.3fms %10.3fms %10.1f %10.1f %8.2fx %s"
              % (label, ours, vend, go, gv, go / gv, flag(ours)))
        rows.append((label, go, gv, note))
    print()
    print("=== 差距小结（我们 / 厂商库）===")
    for label, go, gv, note in rows:
        print("  %-20s %.2fx   %s" % (label, go / gv, note))
    print()
    print("注：#022 GEMM 的差距见 `_debug/_gemm_ladder.py`（fp16 → V8 约 0.75x cuBLAS）")
    print("    与 `_debug/_gemm_sgemm.py`（fp32 → v10d 约 0.85x）。")
    blas.destroy()


if __name__ == "__main__":
    sys.exit(main())
