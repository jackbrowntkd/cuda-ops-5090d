# -*- coding: utf-8 -*-
"""fp32 (SGEMM) 手写实现 vs cuBLAS —— 验证「tiling + reg + float4 + warp 能超过 cuBLAS」。

为什么单独做这个：
- `_gemm_ladder.py` 测的是 **fp16 入 / fp32 累加**，那种场景 cuBLAS 直接切**张量核**，
  CUDA core 手写实现永远追不上 —— 拿它当"手写打不过 cuBLAS"的证据是**错的**。
- fp32 的 SGEMM 里 cuBLAS 也被限制在 **CUDA core 的 FFMA** 上，双方同一起跑线。
  这才是 T600 那类卡（40 SM、无 TF32）上「手写超过 cuBLAS」的场景。

把配方逐项拆开，看每一项各自值多少：
  F1 = tiling + register blocking（标量全局加载）
  F2 = F1 + float4 向量化全局加载（合并访存 / 128bit）
  F3 = 更大 tile + 更多寄存器（128x128，每线程 8x8）
  F4 = warp 级分块（每 warp 32x32，每线程 8x4）+ float4
并对每一档都跑 cuBLAS SGEMM（CUBLAS_COMPUTE_32F，非 TF32）。

用法: python _debug/_gemm_sgemm.py
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from localrun import Runner, nvrtc_compile, cuda, cuDeviceGetAttribute  # noqa: E402

CU_DEV_MULTIPROCESSOR_COUNT = 16
CU_DEV_CLOCK_RATE = 13          # kHz

SRC = r"""
// ============ F0: naive，一线程一元素 ============
__global__ void sgemm_f0_naive(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               float alpha, float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = alpha * acc + beta * C[(size_t)row * N + col];
}

// ============ F1: tiling + register blocking，**标量**全局加载 ============
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define NTH (16 * 16)

__global__ void sgemm_f1_tiled(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int t = 0; t < (BM * BK) / NTH; ++t) {
            const int idx = tid + t * NTH, r = idx / BK, c = idx % BK;
            As[r][c] = A[(size_t)(row0 + r) * K + k0 + c];
        }
#pragma unroll
        for (int t = 0; t < (BK * BN) / NTH; ++t) {
            const int idx = tid + t * NTH, r = idx / BN, c = idx % BN;
            Bs[r][c] = B[(size_t)(k0 + r) * N + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[threadIdx.y * TM + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][threadIdx.x * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + threadIdx.y * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + threadIdx.x * TN + j;
            const size_t o = (size_t)gr * N + gc;
            C[o] = alpha * acc[i][j] + beta * C[o];
        }
    }
}

// ============ F2: F1 + float4 向量化全局加载（合并访存）============
// 共享行距取 BK+4 / BN+4：既是 4 的倍数（float4 存入需 16B 对齐），
// 又避开 16 words 的幂次（消 bank conflict）。
#define AP2 (BK + 4)
#define BP2 (BN + 4)

__global__ void sgemm_f2_float4(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K,
                                float alpha, float beta) {
    __shared__ __align__(16) float As[BM][AP2];
    __shared__ __align__(16) float Bs[BK][BP2];
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    for (int k0 = 0; k0 < K; k0 += BK) {
        {   // A 片 64x16：每行 4 个 float4，共 256 个 = 256 线程各 1 个
            const int r = tid / (BK / 4), c4 = tid % (BK / 4);
            *reinterpret_cast<float4*>(&As[r][c4 * 4]) =
                A4[((size_t)(row0 + r) * K + k0) / 4 + c4];
        }
        {   // B 片 16x64：每行 16 个 float4，共 256 个
            const int r = tid / (BN / 4), c4 = tid % (BN / 4);
            *reinterpret_cast<float4*>(&Bs[r][c4 * 4]) =
                B4[((size_t)(k0 + r) * N + col0) / 4 + c4];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[threadIdx.y * TM + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][threadIdx.x * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + threadIdx.y * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + threadIdx.x * TN + j;
            const size_t o = (size_t)gr * N + gc;
            C[o] = alpha * acc[i][j] + beta * C[o];
        }
    }
}

// ============ F3: 128x128 tile / 每线程 8x8（64 个累加器）+ float4 ============
#define BM3 128
#define BN3 128
#define BK3 16
#define TM3 8
#define TN3 8
#define AP3 (BK3 + 4)
#define BP3 (BN3 + 4)

__global__ void sgemm_f3_128(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K,
                             float alpha, float beta) {
    __shared__ __align__(16) float As[BM3][AP3];
    __shared__ __align__(16) float Bs[BK3][BP3];
    const int tid = threadIdx.y * 16 + threadIdx.x;      // 256
    const int row0 = blockIdx.y * BM3, col0 = blockIdx.x * BN3;
    float acc[TM3][TN3];
#pragma unroll
    for (int i = 0; i < TM3; ++i)
#pragma unroll
        for (int j = 0; j < TN3; ++j) acc[i][j] = 0.f;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    for (int k0 = 0; k0 < K; k0 += BK3) {
#pragma unroll
        for (int t = 0; t < (BM3 * BK3 / 4) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / (BK3 / 4), c4 = idx % (BK3 / 4);
            *reinterpret_cast<float4*>(&As[r][c4 * 4]) =
                A4[((size_t)(row0 + r) * K + k0) / 4 + c4];
        }
#pragma unroll
        for (int t = 0; t < (BK3 * BN3 / 4) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / (BN3 / 4), c4 = idx % (BN3 / 4);
            *reinterpret_cast<float4*>(&Bs[r][c4 * 4]) =
                B4[((size_t)(k0 + r) * N + col0) / 4 + c4];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK3; ++k) {
            float a[TM3], b[TN3];
#pragma unroll
            for (int i = 0; i < TM3; ++i) a[i] = As[threadIdx.y * TM3 + i][k];
#pragma unroll
            for (int j = 0; j < TN3; ++j) b[j] = Bs[k][threadIdx.x * TN3 + j];
#pragma unroll
            for (int i = 0; i < TM3; ++i)
#pragma unroll
                for (int j = 0; j < TN3; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM3; ++i) {
        const int gr = row0 + threadIdx.y * TM3 + i;
#pragma unroll
        for (int j = 0; j < TN3; ++j) {
            const int gc = col0 + threadIdx.x * TN3 + j;
            const size_t o = (size_t)gr * N + gc;
            C[o] = alpha * acc[i][j] + beta * C[o];
        }
    }
}

// ============ F4: warp 级分块（block 64x64 / 4 warp 2x2 / warp 32x32 /
//              每线程 8x4）+ float4 ============
#define WM 64
#define WN 64
#define WK 16
#define WAP (WK + 4)
#define WBP (WN + 4)

__global__ void sgemm_f4_warp(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K,
                              float alpha, float beta) {
    __shared__ __align__(16) float As[WM][WAP];
    __shared__ __align__(16) float Bs[WK][WBP];
    const int tid = threadIdx.x;                 // 128 = 4 warp
    const int lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;
    const int lm = lane & 3, ln = lane >> 2;     // warp 内 4x8 线程网格
    const int row0 = blockIdx.y * WM, col0 = blockIdx.x * WN;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    for (int k0 = 0; k0 < K; k0 += WK) {
#pragma unroll
        for (int t = 0; t < (WM * WK / 4) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / (WK / 4), c4 = idx % (WK / 4);
            *reinterpret_cast<float4*>(&As[r][c4 * 4]) =
                A4[((size_t)(row0 + r) * K + k0) / 4 + c4];
        }
#pragma unroll
        for (int t = 0; t < (WK * WN / 4) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / (WN / 4), c4 = idx % (WN / 4);
            *reinterpret_cast<float4*>(&Bs[r][c4 * 4]) =
                B4[((size_t)(k0 + r) * N + col0) / 4 + c4];
        }
        __syncthreads();

        const int rb = wm * 32 + lm * 8;
        const int cb = wn * 32 + ln * 4;
#pragma unroll
        for (int k = 0; k < WK; ++k) {
            float a[8], b[4];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[rb + i][k];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][cb + j];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gr = row0 + wm * 32 + lm * 8 + i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gc = col0 + wn * 32 + ln * 4 + j;
            const size_t o = (size_t)gr * N + gc;
            C[o] = alpha * acc[i][j] + beta * C[o];
        }
    }
}
// ============ 空 kernel：用来量「单次 launch 的固定开销」============
__global__ void noop_probe(float* p) { if (p && threadIdx.x > 1024) p[0] = 1.f; }
"""

# ---------------------------------------------------------------------------
# v9（用户提供）：warp 级分块 —— block -> warp -> thread 三级分块的中间一级。
# 生成 TS = 8 / 16 / 32 三个实例，用来单独观察「k 方向步长 / 同步次数」的影响。
#
# 核心论点：一个 warp 读一次共享内存的代价，取决于这 32 个线程一共落在几个不同的
# 地址上。块 tile 128x128、微块 8x8：
#   reg8（大块直接切给线程，线程排 2 ty x 16 tx）:
#       A 地址数 = 2(ty) x 2(a0/a1) = 4 ; B = 16(tx) x 2 = 32 ; 合计 36
#   warp 级（每 warp 领一块 64x32，warp 内线程排 8x4）:
#       A 地址数 = 8(tyw) x 2 = 16 ; B = 4(tx) x 2 = 8 ; 合计 24   (-33%)
# 因为 64/8 + 32/8 = 12 < 16/8 + 128/8 = 18 —— 让 WR/TM 与 WC/TN 尽量相等。
#
# ⚠️ 相对原文的唯一改动：APAD 必须是 4 的倍数。
# 原文对共享内存做 float4 读（&As[k][aBase]），行距是 WBM+APAD = 128+APAD；
# 「pad 1 个 float 消 bank conflict」这个经典做法会让奇数 k 的行首落在 4 字节而非
# 16 字节边界 -> float4 读未对齐 -> CUDA misaligned address 直接崩。
# 所以这里 APAD 固定 4（保住 16B 对齐）。
_V9_TPL = r"""
#define WBM 128
#define WBN 128
#define WTM 8
#define WTN 8
#define APAD 4

__global__ void __launch_bounds__(256, 2)
@NAME@(const float* __restrict__ A, const float* __restrict__ B,
       float* __restrict__ C, int N) {
    __shared__ float As[@TS@][WBM + APAD];
    __shared__ float Bs[@TS@][WBN + APAD];

    int tx = threadIdx.x;               // 0..3
    int ty = threadIdx.y;               // 0..63
    int tid = ty * 4 + tx;
    int rowStart = blockIdx.y * WBM;
    int colStart = blockIdx.x * WBN;

    int warp = ty >> 3;                 // 0..7
    int warpRow = warp & 1;             // 0..1
    int warpCol = warp >> 1;            // 0..3
    int tyw = ty & 7;                   // warp 内行号 0..7

    float acc[WTM][WTN];
#pragma unroll
    for (int i = 0; i < WTM; ++i)
#pragma unroll
        for (int j = 0; j < WTN; ++j) acc[i][j] = 0.f;

    for (int t = 0; t < N; t += @TS@) {
        // A 转置存成 As[k][row]，这样计算侧的 float4 读才是连续的
        for (int l = 0; l < (WBM * @TS@) / 256; ++l) {
            int idx = tid + l * 256;
            int r = idx / @TS@;
            int c = idx % @TS@;
            As[c][r] = A[(size_t)(rowStart + r) * N + t + c];
        }
        for (int l = 0; l < (@TS@ * WBN) / 256; ++l) {
            int idx = tid + l * 256;
            Bs[idx / WBN][idx % WBN] = B[(size_t)(t + idx / WBN) * N + colStart + idx % WBN];
        }
        __syncthreads();

        const int aBase = warpRow * 64 + tyw * WTM;
        const int bBase = warpCol * 32 + tx * WTN;
#pragma unroll @UNROLL@
        for (int k = 0; k < @TS@; ++k) {
            float4 a0 = *reinterpret_cast<const float4*>(&As[k][aBase]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[k][aBase + 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][bBase]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][bBase + 4]);
            float aReg[WTM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float bReg[WTN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < WTM; ++i)
#pragma unroll
                for (int j = 0; j < WTN; ++j) acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    const int row0 = rowStart + warpRow * 64 + tyw * WTM;
    const int col0 = colStart + warpCol * 32 + tx * WTN;
#pragma unroll
    for (int i = 0; i < WTM; ++i)
        for (int j = 0; j < WTN; j += 4) {
            float4 v = make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
            *reinterpret_cast<float4*>(&C[(size_t)(row0 + i) * N + col0 + j]) = v;
        }
}
"""

# (TS, k 循环展开度, kernel 名)。展开度 = TS 表示全展开。
# TS x 展开度 组成二维矩阵：TS 决定 __syncthreads 次数，展开度决定寄存器压力。
V9_TS = (8, 16, 32)
V9_INST = [
    (8, 8, "sgemm_v9_ts8"), (8, 4, "sgemm_v9_ts8_u4"),
    (16, 16, "sgemm_v9_ts16"), (16, 8, "sgemm_v9_ts16_u8"),
    (16, 4, "sgemm_v9_ts16_u4"),
    (32, 32, "sgemm_v9_ts32"), (32, 8, "sgemm_v9_ts32_u8"),
    (32, 4, "sgemm_v9_ts32_u4"),
]
SRC = SRC + "\n".join(
    _V9_TPL.replace("@TS@", str(ts)).replace("@NAME@", nm)
           .replace("@UNROLL@", str(u))
    for (ts, u, nm) in V9_INST)

# ---------------------------------------------------------------------------
# v10：移植 siboehm/SGEMM_CUDA 的 kernel 10（warptiling，A6000 上 93.7% cuBLAS）。
# 与用户 v9 的三处关键差别 —— 这就是要验证的东西：
#   1. **warp tile 从 64x32 扩到 64x64**，线程微块 8x4，引入 WNITER 子块分列
#      -> 每 dotIdx 的共享载入从 16 次/64 FMA = 0.25 降到 24 次/128 FMA = **0.1875**（-25%）
#   2. **完全不用 padding**：A 转置存成 As[k*BM + row]，行距恰好 = BM（4 的倍数）
#      -> float4 读天然 16B 对齐，绕开 v9 那个「APAD 必须是 4 的倍数」硬约束
#   3. 每线程 **128 个累加器**（v9 是 64），**128 线程/block**（v9 是 256）
# 注意 siboehm 的 As 读同样是 threadRowInWarp*TM 步长 8 -> **同样存在 2 路 bank 冲突**，
# 它照样 93.7%，再次说明冲突不是瓶颈。
#
# 模板占位：@NAME@ @BM@ @BN@ @BK@ @WM@ @WN@ @TM@ @TN@ @WNITER@ @NT@ @MREAD@ @NREAD@
# ⚠️ 向量化的共享读（@MREAD_VEC@/@NREAD_VEC@）要求 WMITER==1 且 TN==4。
_MREAD_SCALAR = r"""
#pragma unroll
            for (int a = 0; a < WMITER; ++a)
#pragma unroll
                for (int i = 0; i < XTM; ++i)
                    regM[a * XTM + i] =
                        As[k * XBM + warpRow * XWM + a * WSUBM + threadRowInWarp * XTM + i];"""

_MREAD_VEC = r"""
#pragma unroll
            for (int q = 0; q < WMITER * XTM; q += 4)
                *reinterpret_cast<float4*>(&regM[q]) =
                    *reinterpret_cast<const float4*>(
                        &As[k * XBM + warpRow * XWM + threadRowInWarp * XTM + q]);"""

_NREAD_SCALAR = r"""
#pragma unroll
            for (int b = 0; b < WNITER; ++b)
#pragma unroll
                for (int j = 0; j < XTN; ++j)
                    regN[b * XTN + j] =
                        Bs[k * XBN + warpCol * XWN + b * WSUBN + threadColInWarp * XTN + j];"""

_NREAD_VEC = r"""
#pragma unroll
            for (int b = 0; b < WNITER; ++b)
#pragma unroll
                for (int q = 0; q < XTN; q += 4)
                    *reinterpret_cast<float4*>(&regN[b * XTN + q]) =
                        *reinterpret_cast<const float4*>(
                            &Bs[k * XBN + warpCol * XWN + b * WSUBN
                                + threadColInWarp * XTN + q]);"""

# ⚠️ 常量名统一加 X 前缀：本文件前面的 F1/F3/F4 已经有 #define BM/BN/BK/TM/TN/WM/WN，
# 不加前缀会被预处理器替换掉（`constexpr int BM = 128` -> `constexpr int 64 = 128`）。
_V10_TPL = r"""
__global__ void __launch_bounds__(@NT@)
@NAME@(const float* __restrict__ A, const float* __restrict__ B,
       float* __restrict__ C, int N) {
    constexpr int XBM = @BM@, XBN = @BN@, XBK = @BK@;
    constexpr int XWM = @WM@, XWN = @WN@, XTM = @TM@, XTN = @TN@;
    constexpr int WNITER = @WNITER@, NTHR = @NT@;
    constexpr int WARPS_N = XBN / XWN;
    constexpr int WMITER = (XWM * XWN) / (32 * XTM * XTN * WNITER);
    constexpr int WSUBM = XWM / WMITER;
    constexpr int WSUBN = XWN / WNITER;

    __shared__ float As[XBK * XBM];   // 转置存: As[k*XBM + row]  —— 无 padding
    __shared__ float Bs[XBK * XBN];   // 正常:   Bs[k*XBN + col]

    const int tid = threadIdx.x;
    const int warpIdx = tid >> 5;
    const int warpCol = warpIdx % WARPS_N;
    const int warpRow = warpIdx / WARPS_N;
    const int lane = tid & 31;
    const int threadColInWarp = lane % (WSUBN / XTN);
    const int threadRowInWarp = lane / (WSUBN / XTN);

    const int rowStart = blockIdx.y * XBM;
    const int colStart = blockIdx.x * XBN;

    const int innerRowA = tid / (XBK / 4);
    const int innerColA = tid % (XBK / 4);
    constexpr int rowStrideA = (NTHR * 4) / XBK;
    const int innerRowB = tid / (XBN / 4);
    const int innerColB = tid % (XBN / 4);
    constexpr int rowStrideB = NTHR / (XBN / 4);

    float acc[WMITER][XTM][WNITER][XTN];
#pragma unroll
    for (int a = 0; a < WMITER; ++a)
#pragma unroll
        for (int i = 0; i < XTM; ++i)
#pragma unroll
            for (int b = 0; b < WNITER; ++b)
#pragma unroll
                for (int j = 0; j < XTN; ++j) acc[a][i][b][j] = 0.f;

    for (int t = 0; t < N; t += XBK) {
        for (int off = 0; off + rowStrideA <= XBM; off += rowStrideA) {
            const float4 v = *reinterpret_cast<const float4*>(
                &A[(size_t)(rowStart + innerRowA + off) * N + t + innerColA * 4]);
            As[(innerColA * 4 + 0) * XBM + innerRowA + off] = v.x;
            As[(innerColA * 4 + 1) * XBM + innerRowA + off] = v.y;
            As[(innerColA * 4 + 2) * XBM + innerRowA + off] = v.z;
            As[(innerColA * 4 + 3) * XBM + innerRowA + off] = v.w;
        }
        for (int off = 0; off + rowStrideB <= XBK; off += rowStrideB) {
            *reinterpret_cast<float4*>(&Bs[(innerRowB + off) * XBN + innerColB * 4]) =
                *reinterpret_cast<const float4*>(
                    &B[(size_t)(t + innerRowB + off) * N + colStart + innerColB * 4]);
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < XBK; ++k) {
            float regM[WMITER * XTM];
            float regN[WNITER * XTN];@MREAD@@NREAD@
#pragma unroll
            for (int a = 0; a < WMITER; ++a)
#pragma unroll
                for (int b = 0; b < WNITER; ++b)
#pragma unroll
                    for (int i = 0; i < XTM; ++i)
#pragma unroll
                        for (int j = 0; j < XTN; ++j)
                            acc[a][i][b][j] += regM[a * XTM + i] * regN[b * XTN + j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < WMITER; ++a)
#pragma unroll
        for (int i = 0; i < XTM; ++i)
#pragma unroll
            for (int b = 0; b < WNITER; ++b) {
                const int r = rowStart + warpRow * XWM + a * WSUBM + threadRowInWarp * XTM + i;
                const int c = colStart + warpCol * XWN + b * WSUBN + threadColInWarp * XTN;
#pragma unroll
                for (int j = 0; j < XTN; j += 4)
                    *reinterpret_cast<float4*>(&C[(size_t)r * N + c + j]) =
                        make_float4(acc[a][i][b][j], acc[a][i][b][j + 1],
                                    acc[a][i][b][j + 2], acc[a][i][b][j + 3]);
            }
}
"""

# (名, BM, BN, BK, WM, WN, TM, TN, WNITER, NT, 共享读是否向量化, 标签)
V10_INST = [
    ("sgemm_v10_k10", 128, 128, 16, 64, 64, 8, 4, 4, 128, False,
     "v10 = siboehmK10 原样(标量共享读)"),
    ("sgemm_v10_k10_v4", 128, 128, 16, 64, 64, 8, 4, 4, 128, True,
     "v10b = K10 + float4 共享读"),
    ("sgemm_v10_bk32", 128, 128, 32, 64, 64, 8, 4, 4, 128, True,
     "v10c = v10b 但 BK32(同步减半)"),
    ("sgemm_v10_w8", 128, 128, 16, 64, 32, 8, 4, 2, 256, True,
     "v10d = 8warp/半宽warp tile(当前最好)"),
    # ---- 2026-10-05 参数空间扫：三条轴 ----
    # 轴1：v10d 形状上把 BK 16->32（v10b->v10c 时 BK32 值 +11%）
    ("sgemm_v10_e_bk32", 128, 128, 32, 64, 32, 8, 4, 2, 256, True,
     "v10e = v10d + BK32"),
    # 轴2：占用率再翻倍 —— NT=512，warp tile 降到 32x32（acc/线程=32）
    ("sgemm_v10_g_nt512", 128, 128, 16, 32, 32, 8, 4, 1, 512, True,
     "v10g = NT512/32x32 warp tile(占用率轴)"),
    # 轴3：换个形状的同面积 warp tile —— WM=32 WN=64，转向 N 方向
    ("sgemm_v10_h_wn64", 128, 128, 16, 32, 64, 8, 8, 1, 256, True,
     "v10h = WM32/WN64 TN8 (N向warp tile)"),
]


def _v10_src(name, bm, bn, bk, wm, wn, tm, tn, wniter, nt, vec):
    return (_V10_TPL
            .replace("@NAME@", name).replace("@BM@", str(bm)).replace("@BN@", str(bn))
            .replace("@BK@", str(bk)).replace("@WM@", str(wm)).replace("@WN@", str(wn))
            .replace("@TM@", str(tm)).replace("@TN@", str(tn))
            .replace("@WNITER@", str(wniter)).replace("@NT@", str(nt))
            .replace("@MREAD@", _MREAD_VEC if vec else _MREAD_SCALAR)
            .replace("@NREAD@", _NREAD_VEC if vec else _NREAD_SCALAR))


SRC = SRC + "\n".join(_v10_src(*inst[:11]) for inst in V10_INST)


VARIANTS = [
    ("F0 naive 一线程/元素", "sgemm_f0_naive",
     lambda M, N: ((N + 15) // 16, (M + 15) // 16), (16, 16)),
    ("F1 tiling+reg 64x64/4x4 标量", "sgemm_f1_tiled",
     lambda M, N: (N // 64, M // 64), (16, 16)),
    ("F2 =F1 + float4 合并访存", "sgemm_f2_float4",
     lambda M, N: (N // 64, M // 64), (16, 16)),
    ("F3 128x128/8x8 + float4", "sgemm_f3_128",
     lambda M, N: (N // 128, M // 128), (16, 16)),
    ("F4 warp分块 64x64/8x4 + float4", "sgemm_f4_warp",
     lambda M, N: (N // 64, M // 64), 128),
    # v9：block 128x128 / 8 warp 排 2x4 / 每 warp 领 64x32 / 线程 8x4 -> 微块 8x8
    #     TS 是 k 方向步长（同步次数 = 2K/TS）；U 是 k 循环展开度。
    #     ⚠️ 不能只测 TS 全展开的做法 —— 见 PERF.md，全展开会把寄存器压力抬爆。
    ("v9 TS8  unroll8 (全展开)", "sgemm_v9_ts8",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    ("v9 TS8  unroll4", "sgemm_v9_ts8_u4",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    ("v9 TS16 unroll16(全展开)", "sgemm_v9_ts16",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    ("v9 TS16 unroll8", "sgemm_v9_ts16_u8",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    ("v9 TS16 unroll4", "sgemm_v9_ts16_u4",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    ("v9 TS32 unroll32(全展开)", "sgemm_v9_ts32",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    ("v9 TS32 unroll8", "sgemm_v9_ts32_u8",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    ("v9 TS32 unroll4", "sgemm_v9_ts32_u4",
     lambda M, N: (N // 128, M // 128), (4, 64)),
    # v10：siboehm/SGEMM_CUDA kernel10 的移植（block 128x128，无 padding）
    ("v10 = siboehmK10 原样", "sgemm_v10_k10",
     lambda M, N: (N // 128, M // 128), 128),
    ("v10b = K10 + float4 共享读", "sgemm_v10_k10_v4",
     lambda M, N: (N // 128, M // 128), 128),
    ("v10c = v10b 但 BK32", "sgemm_v10_bk32",
     lambda M, N: (N // 128, M // 128), 128),
    ("v10d = 8warp 半宽warp tile", "sgemm_v10_w8",
     lambda M, N: (N // 128, M // 128), 256),
    ("v10e = v10d + BK32", "sgemm_v10_e_bk32",
     lambda M, N: (N // 128, M // 128), 256),
    ("v10g = NT512 / 32x32 warp tile", "sgemm_v10_g_nt512",
     lambda M, N: (N // 128, M // 128), 512),
    ("v10h = WM32/WN64 TN8", "sgemm_v10_h_wn64",
     lambda M, N: (N // 128, M // 128), 256),
]

# v9 / v10 的签名都是 (A, B, C, N)（方阵、无 alpha/beta），需要单独的实参通道
NONSTD_ARGS = {nm for (_, _, nm) in V9_INST} | {inst[0] for inst in V10_INST}


def make_args(r, kname, dA, dB, dC, M, N, K, al, be):
    if kname in NONSTD_ARGS:
        return [dA, dB, dC, r.i(N)]
    return [dA, dB, dC, r.i(M), r.i(N), r.i(K), r.f(al), r.f(be)]


def launch_overhead_ms(r, repeat=20, warmup=5):
    """量「单次 launch」的固定开销（ctypes 编组 + cuLaunchKernel + 空 kernel）。

    用途是给出**可信度基准**，不是用来做减法：
    实测每次迭代 = max(host成本, GPU时间)，`实测 / HOST ≥ 4` 才说明 GPU 时间
    确实压过了 host 开销，数字可横向比较。（详见 PERF.md 的方法论一节）
    """
    d = r.alloc(4)
    args = [d]
    r.launch("noop_probe", 1, 1, args)
    r.sync()
    ms = r.timeit(lambda: r.launch("noop_probe", 1, 1, args),
                  repeat=repeat, warmup=warmup)
    r.free(d)
    return ms


def device_peak_fp32():
    """fp32 FFMA 理论峰值 = SM 数 x 128 FMA/cycle x 2 flop x 频率。"""
    sm = ctypes.c_int()
    clk = ctypes.c_int()
    cuDeviceGetAttribute(ctypes.byref(sm), CU_DEV_MULTIPROCESSOR_COUNT, 0)
    cuDeviceGetAttribute(ctypes.byref(clk), CU_DEV_CLOCK_RATE, 0)
    return sm.value, clk.value, sm.value * 128 * 2 * clk.value * 1e3


def main():
    ptx, arch, log = nvrtc_compile(SRC, name="sgemm_ladder.cu")
    r = Runner(ptx, arch, log)
    sm, clk, peak = device_peak_fp32()
    print("设备:", Runner.devicename(), "| PTX arch:", arch)
    print("SM 数 = %d | 频率上限 = %.2f GHz | fp32 FFMA 理论峰值 ≈ %.1f TFLOPS"
          % (sm, clk / 1e6, peak / 1e12))
    print()

    try:
        from cublas import Handle
        blas = Handle()
        print("cuBLAS 版本:", blas.version)
    except Exception as e:
        blas = None
        print("!! cuBLAS 不可用:", repr(e))

    oh = launch_overhead_ms(r)
    print("单次 launch 的 host 固定开销（空 kernel 实测）= %.3f us" % (oh * 1000))
    print("→ 报告**原始**实测值；末列「可信度」= 实测/host 开销，<4 说明该规模数字被 host 污染")
    print()

    M = N = K = 1024
    flop = 2.0 * M * N * K
    rng = np.random.default_rng(22)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    C0 = rng.standard_normal((M, N)).astype(np.float32)
    ref = (A.astype(np.float64) @ B.astype(np.float64))
    dA, dB = r.to_device(A), r.to_device(B)

    print("== fp32 SGEMM 1024³（cuBLAS 走 CUBLAS_COMPUTE_32F，即 FFMA 路径）==")
    print("%-32s %9s %10s %7s %10s %5s %s"
          % ("变体", "耗时", "GFLOPS", "%峰值", "vs cuBLAS", "正确", "可信度"))
    print("-" * 104)

    def flag(raw):
        if oh <= 0:
            return ""
        rr = raw / oh
        return "OK" if rr >= 4.0 else "~host(%.1fx)" % rr

    blas_gf = None
    if blas is not None:
        dC = r.to_device(C0)
        blas.gemm_f32(dA, dB, dC, M, N, K, 1.0, 0.0)
        r.sync()
        got = r.from_device(dC, (M, N), np.float32).astype(np.float64)
        rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
        okb = rel < 1e-4
        raw = r.timeit(lambda: blas.gemm_f32(dA, dB, dC, M, N, K, 1.0, 0.0),
                       repeat=20, warmup=5)
        blas_gf = flop / (raw * 1e-3) / 1e9
        r.free(dC)
        print("%-32s %7.3fms %10.1f %6.1f%% %10s %5s %s"
              % ("cuBLAS SGEMM", raw, blas_gf, 100 * blas_gf * 1e9 / peak,
                 "1.00x", "OK" if okb else "FAIL", flag(raw)))

    base = None
    results = []
    for label, kname, gridfn, blk in VARIANTS:
        # 正确性：alpha=1/beta=0 与 alpha=1.5/beta=0.5 各一次
        # （v9 的签名不含 alpha/beta，只跑 (1,0)）
        ab_cases = ((1.0, 0.0),) if kname in NONSTD_ARGS else ((1.0, 0.0), (1.5, 0.5))
        oks = []
        for (al, be) in ab_cases:
            dC = r.to_device(C0)
            r.launch(kname, gridfn(M, N), blk,
                     make_args(r, kname, dA, dB, dC, M, N, K, al, be))
            r.sync()
            got = r.from_device(dC, (M, N), np.float32).astype(np.float64)
            r.free(dC)
            want = al * ref + be * C0.astype(np.float64)
            oks.append(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30) < 1e-4)
        dC = r.to_device(C0)
        args = make_args(r, kname, dA, dB, dC, M, N, K, 1.0, 0.0)
        r.launch(kname, gridfn(M, N), blk, args)
        r.sync()
        raw = r.timeit(lambda: r.launch(kname, gridfn(M, N), blk, args),
                       repeat=20, warmup=5)
        gf = flop / (raw * 1e-3) / 1e9
        base = gf if base is None else base
        r.free(dC)
        results.append((label, kname, gridfn, blk, raw, gf))
        print("%-32s %7.3fms %10.1f %6.1f%% %10s %5s %s"
              % (label, raw, gf, 100 * gf * 1e9 / peak,
                 ("%.2fx" % (gf / blas_gf)) if blas_gf else "-",
                 "OK" if all(oks) else "FAIL", flag(raw)))

    print()
    print("== 规模扫描（GFLOPS，原始实测值）==")
    sizes = (1024, 2048, 4096)
    sweep = [(l, k, g, b) for (l, k, g, b, _, _) in results if k != "sgemm_f0_naive"]
    if blas is not None:
        sweep.insert(0, ("cuBLAS SGEMM", "__cublas__", None, None))
    gflops = {}
    for label, kname, gridfn, blk in sweep:
        row = []
        for sz in sizes:
            Asz = (rng.standard_normal((sz, sz)) * 0.5).astype(np.float32)
            Bsz = (rng.standard_normal((sz, sz)) * 0.5).astype(np.float32)
            Csz = np.zeros((sz, sz), np.float32)
            xA, xB, xC = r.to_device(Asz), r.to_device(Bsz), r.to_device(Csz)
            if kname == "__cublas__":
                raw = r.timeit(lambda: blas.gemm_f32(xA, xB, xC, sz, sz, sz, 1.0, 0.0),
                               repeat=5, warmup=2)
            else:
                xx = make_args(r, kname, xA, xB, xC, sz, sz, sz, 1.0, 0.0)
                raw = r.timeit(lambda: r.launch(kname, gridfn(sz, sz), blk, xx),
                               repeat=5, warmup=2)
            row.append(2.0 * sz ** 3 / (raw * 1e-3) / 1e9)
            for p in (xA, xB, xC):
                r.free(p)
        gflops[label] = row
        print("%-34s %13.1f %13.1f %13.1f"
              % (label, row[0], row[1], row[2]))

    if blas is not None and "cuBLAS SGEMM" in gflops:
        ref_row = gflops["cuBLAS SGEMM"]
        print()
        print("== 手写 / cuBLAS 比值（>1.00 = 超过厂商库）==")
        print("%-34s %13s %13s %13s" % ("变体", "1024^3", "2048^3", "4096^3"))
        print("-" * 78)
        for label, row in gflops.items():
            if label == "cuBLAS SGEMM":
                continue
            print("%-34s %12.2fx %12.2fx %12.2fx"
                  % (label, row[0] / ref_row[0], row[1] / ref_row[1],
                     row[2] / ref_row[2]))

    if blas is not None:
        blas.destroy()
    print()
    print("注：本文件所有变体都只覆盖「M/N/K 为 tile 整数倍」的情形（性能规模 1024³ 满足），")
    print("    不做边界 guard —— 目的只是量性能台阶，不是当提交答案。")
    print("    F1->F2 的差值 = float4 合并访存单独贡献；F2->F4 = warp 分块单独贡献。")


if __name__ == "__main__":
    sys.exit(main())
