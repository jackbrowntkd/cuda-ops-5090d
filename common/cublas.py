# -*- coding: utf-8 -*-
"""cuBLAS via ctypes —— 给本机测速提供「厂商基线」。

本地 cl.exe 被安全策略拉黑、nvcc 不可用，但 cublas64_*.dll 可以直接 ctypes 调。

⚠️ 两个必须记住的点：

1. **cuBLAS 是列主序（column-major）**，而题目里的 A/B/C 全是行主序。
   row-major 的 `C = alpha*A*B + beta*C`（A:M×K, B:K×N, C:M×N）要**交换 A 与 B**
   再调用，同时把 m/n 也交换：

       cublasGemmEx(OP_N, OP_N,
                    m = N, n = M, k = K,
                    alpha, B, CUDA_R_16F, lda=N,   // B 当作列主序 N×K
                           A, CUDA_R_16F, ldb=K,   // A 当作列主序 K×M
                    beta,  C, CUDA_R_16F, ldc=N,   // C 当作列主序 N×M
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)

   之所以成立：行主序 M×N 的那段内存，按列主序读就是 N×M，恰好等于转置，
   于是 `C^T = B^T A^T` 与 `C = A B` 共享同一块内存布局，不需要真的转置。

2. **handle 必须在 CUDA context 已经 current 之后创建。**
   `localrun.Runner` 的 `__init__` 里会 `cuCtxCreate`（该调用会把新 context
   设为当前线程的 current context），所以顺序必须是：先建 Runner，再 create handle。

走 NULL stream，与 localrun 的 `cuLaunchKernel(hStream=None)` 是同一条流，
因此和自研 kernel 的计时可以直接横比。
"""
import ctypes
import os
from ctypes import POINTER, byref, c_float, c_int, c_longlong, c_void_p

# ---------------------------------------------------------------- 常量
CUDA_R_16F = 2
CUDA_R_32F = 0
CUDA_R_8I = 3
CUDA_R_32I = 10
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_COMPUTE_16F = 64
CUBLAS_COMPUTE_32F = 68
CUBLAS_COMPUTE_32I = 73
CUBLAS_GEMM_DEFAULT = -1
CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
CUBLAS_STATUS_SUCCESS = 0

_CUDA_BIN_CANDIDATES = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
]


def _find_cublas():
    for d in _CUDA_BIN_CANDIDATES:
        if not os.path.isdir(d):
            continue
        cands = [f for f in sorted(os.listdir(d))
                 if f.startswith("cublas64_") and f.endswith(".dll")]
        if cands:
            return os.path.join(d, cands[0]), d
    raise RuntimeError("找不到 cublas64_*.dll（检查 CUDA Toolkit 是否安装）")


CUBLAS_DLL, CUDA_BIN = _find_cublas()
# 依赖 cublasLt64_*.dll，Py3.8+ 需要显式加 DLL 搜索目录
try:
    os.add_dll_directory(CUDA_BIN)
except Exception:
    pass
os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")

_lib = ctypes.WinDLL(CUBLAS_DLL)


def _bind(name, argtypes):
    """同 localrun._bind：把配好 argtypes 的对象缓存回实例字典。

    ctypes 的 CDLL.__getattr__ 每次访问都会新建 _FuncPtr，不缓存的话
    `lib.cublasCreate_v2` 这种写法拿到的是没设过 argtypes 的新对象。
    """
    f = getattr(_lib, name)
    f.argtypes = argtypes
    f.restype = c_int
    try:
        setattr(_lib, name, f)
    except Exception:
        pass
    return f


cublasCreate_v2 = _bind("cublasCreate_v2", [POINTER(c_void_p)])
cublasDestroy_v2 = _bind("cublasDestroy_v2", [c_void_p])
cublasSetStream_v2 = _bind("cublasSetStream_v2", [c_void_p, c_void_p])
cublasSetMathMode = _bind("cublasSetMathMode", [c_void_p, c_int])
cublasGetVersion_v2 = _bind("cublasGetVersion_v2", [c_void_p, POINTER(c_int)])
cublasGemmEx = _bind("cublasGemmEx", [
    c_void_p,                      # handle
    c_int, c_int,                  # transa, transb
    c_int, c_int, c_int,           # m, n, k
    c_void_p,                      # alpha (host void*)
    c_void_p, c_int, c_int,        # A, Atype, lda
    c_void_p, c_int, c_int,        # B, Btype, ldb
    c_void_p,                      # beta (host void*)
    c_void_p, c_int, c_int,        # C, Ctype, ldc
    c_int, c_int,                  # computeType, algo
])

cublasGemmStridedBatchedEx = _bind("cublasGemmStridedBatchedEx", [
    c_void_p,                      # handle
    c_int, c_int,                  # transa, transb
    c_int, c_int, c_int,           # m, n, k
    c_void_p,                      # alpha
    c_void_p, c_int, c_int, c_longlong,   # A, Atype, lda, strideA
    c_void_p, c_int, c_int, c_longlong,   # B, Btype, ldb, strideB
    c_void_p,                      # beta
    c_void_p, c_int, c_int, c_longlong,   # C, Ctype, ldc, strideC
    c_int,                         # batchCount
    c_int, c_int,                  # computeType, algo
])


def _ok(st, what):
    if st != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError("cuBLAS error @ %s: status=%d" % (what, st))
    return st


class Handle:
    """cuBLAS handle。必须在 Runner 之后创建（context 得先 current）。"""

    def __init__(self):
        h = c_void_p()
        _ok(cublasCreate_v2(byref(h)), "cublasCreate_v2")
        self.h = h
        self._v = c_int()
        cublasGetVersion_v2(h, byref(self._v))

    @property
    def version(self):
        return self._v.value

    def set_tensor_op(self, on=True):
        """允许 Tensor Core 路径（fp32 输入时才有意义的 TF32 开关）。"""
        _ok(cublasSetMathMode(self.h, CUBLAS_GEMM_DEFAULT_TENSOR_OP if on
                              else CUBLAS_GEMM_DEFAULT), "setMathMode")

    def gemm_f16_f32acc(self, dA, dB, dC, M, N, K, alpha=1.0, beta=0.0,
                        algo=CUBLAS_GEMM_DEFAULT):
        """行主序 C(M×N) = alpha*A(M×K)*B(K×N) + beta*C，fp16 入 fp32 累加。

        见模块 docstring 的列主序映射推导。
        """
        return self._gemm(dA, dB, dC, M, N, K, alpha, beta,
                          CUDA_R_16F, CUBLAS_COMPUTE_32F, algo, "cublasGemmEx")

    def gemm_f32(self, dA, dB, dC, M, N, K, alpha=1.0, beta=0.0,
                 algo=CUBLAS_GEMM_DEFAULT):
        """行主序 **fp32 (SGEMM)**：C = alpha*A*B + beta*C，全 fp32。

        ⚠️ 这是和 T600 那类卡「手写 GEMM 打 cuBLAS」唯一可比的赛道：
        computeType 用 CUBLAS_COMPUTE_32F（不是 FAST_TF32），
        所以 cuBLAS 也被限制在 **CUDA core 的 FFMA** 路径上，双方同一起跑线。
        用 fp16 入参时 cuBLAS 会切张量核，CUDA core 手写实现永远追不上，那是另一回事。
        """
        return self._gemm(dA, dB, dC, M, N, K, alpha, beta,
                          CUDA_R_32F, CUBLAS_COMPUTE_32F, algo, "cublasSgemmEx")

    def _gemm(self, dA, dB, dC, M, N, K, alpha, beta, dtype, ctype, algo, what):
        # 列主序映射：m/n 与 A/B 都交换（详见模块 docstring）
        a = c_float(float(alpha))
        b = c_float(float(beta))
        _ok(cublasGemmEx(
            self.h, CUBLAS_OP_N, CUBLAS_OP_N,
            int(N), int(M), int(K),
            byref(a),
            dB, dtype, int(N),             # 实参 A' = B
            dA, dtype, int(K),             # 实参 B' = A
            byref(b),
            dC, dtype, int(N),
            int(ctype), int(algo),
        ), what)

    # ------------------------------------------------------------ batched
    def gemm_f32_batched(self, dA, dB, dC, M, N, K, batch, alpha=1.0, beta=0.0,
                         algo=CUBLAS_GEMM_DEFAULT):
        """行主序 batched fp32：C[i] = alpha*A[i]*B[i] + beta*C[i]，i < batch。

        每个 batch 的 stride 就是矩阵元素数（A: M*K, B: K*N, C: M*N）—— 题面要求
        A/B/C 都是连续排布的 batch，正好对应 cuBLAS 的 strided batched。
        """
        return self._gemm_batched(dA, dB, dC, M, N, K, batch, alpha, beta,
                                  CUDA_R_32F, CUBLAS_COMPUTE_32F, algo,
                                  "cublasGemmStridedBatchedEx")

    def gemm_f16_batched(self, dA, dB, dC, M, N, K, batch, alpha=1.0, beta=0.0,
                         algo=CUBLAS_GEMM_DEFAULT):
        """行主序 batched fp16 入 fp32 累加。"""
        return self._gemm_batched(dA, dB, dC, M, N, K, batch, alpha, beta,
                                  CUDA_R_16F, CUBLAS_COMPUTE_32F, algo,
                                  "cublasGemmStridedBatchedEx")

    def _gemm_batched(self, dA, dB, dC, M, N, K, batch, alpha, beta,
                      dtype, ctype, algo, what):
        # 列主序映射同单发版；stride 用**元素个数**（cuBLAS 的 stride 是元素单位）
        a = c_float(float(alpha))
        b = c_float(float(beta))
        _ok(cublasGemmStridedBatchedEx(
            self.h, CUBLAS_OP_N, CUBLAS_OP_N,
            int(N), int(M), int(K),                 # m=N, n=M, k=K
            byref(a),
            dB, dtype, int(N), c_longlong(K * N),   # A' = B[i]（列主序 N×K）
            dA, dtype, int(K), c_longlong(M * K),   # B' = A[i]（列主序 K×M）
            byref(b),
            dC, dtype, int(N), c_longlong(M * N),
            int(batch), int(ctype), int(algo),
        ), what)

    # ------------------------------------------------------------ int8
    def gemm_int8(self, dA, dB, dC, M, N, K, alpha=1, beta=0,
                  algo=CUBLAS_GEMM_DEFAULT):
        """行主序 int8 GEMM，int32 累加：C = alpha*A*B + beta*C（alpha/beta 是 int32）。

        ⚠️ 这是**纯 int8 GEMM**，不含 zero_point / scale 校正。
        对 #032 这类"量化 MatMul"题，它只能当作**忽略零点校正的理论上界参照**，
        不是逐位可比的等价实现 —— 报差距时必须写清这一点。
        """
        a = c_int(int(alpha))
        b = c_int(int(beta))
        _ok(cublasGemmEx(
            self.h, CUBLAS_OP_N, CUBLAS_OP_N,
            int(N), int(M), int(K),
            byref(a),
            dB, CUDA_R_8I, int(N),
            dA, CUDA_R_8I, int(K),
            byref(b),
            dC, CUDA_R_32I, int(N),
            CUBLAS_COMPUTE_32I, int(algo),
        ), "cublasGemmEx(int8)")
        return None

    def destroy(self):
        if self.h:
            cublasDestroy_v2(self.h)
            self.h = None
