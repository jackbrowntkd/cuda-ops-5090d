# -*- coding: utf-8 -*-
"""#022 GEMM 优化阶梯实测（本机 RTX 5090 D）。

目的：把「naive -> shared tiling -> 消 bank conflict -> 大 tile + 128bit 向量化
      -> WMMA 张量核」这几个台阶的真实 GFLOPS 量出来，而不是靠印象讲故事。

约束：本机 cl.exe 被安全策略拉黑，nvcc 不可用；走 _tools/localrun.py 的
      NVRTC + Driver API 路线。kernel 只覆盖 M/N/K 均为 tile 整数倍的情形
      （性能规模 1024^3 满足）。非对齐的完整 guard 版见各题 solution.cu。

用法: python _debug/_gemm_ladder.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from localrun import Runner, nvrtc_compile  # noqa: E402

SRC = r"""
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// ============ V0: naive，一线程一元素，零复用 ============
__global__ void gemm_v0_naive(const half* __restrict__ A, const half* __restrict__ B,
                              half* __restrict__ C, int M, int N, int K,
                              float alpha, float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(A[(size_t)row * K + k]) * __half2float(B[(size_t)k * N + col]);
    C[(size_t)row * N + col] =
        __float2half(alpha * acc + beta * __half2float(C[(size_t)row * N + col]));
}

// ============ 64x64 tile / 4x4 per thread / BK=16 ============
#define BM1 64
#define BN1 64
#define BK1 16
#define TM1 4
#define TN1 4

// V1: 无 pad —— 就是当前 solution.cu 的形态
__global__ void gemm_v1_tiled(const half* __restrict__ A, const half* __restrict__ B,
                              half* __restrict__ C, int M, int N, int K,
                              float alpha, float beta) {
    __shared__ half As[BM1][BK1];
    __shared__ half Bs[BK1][BN1];
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int row0 = blockIdx.y * BM1, col0 = blockIdx.x * BN1;
    float acc[TM1][TN1];
#pragma unroll
    for (int i = 0; i < TM1; ++i)
#pragma unroll
        for (int j = 0; j < TN1; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK1) {
#pragma unroll
        for (int t = 0; t < (BM1 * BK1) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / BK1, c = idx % BK1;
            As[r][c] = A[(size_t)(row0 + r) * K + k0 + c];
        }
#pragma unroll
        for (int t = 0; t < (BK1 * BN1) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / BN1, c = idx % BN1;
            Bs[r][c] = B[(size_t)(k0 + r) * N + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK1; ++k) {
            half a[TM1], b[TN1];
#pragma unroll
            for (int i = 0; i < TM1; ++i) a[i] = As[threadIdx.y * TM1 + i][k];
#pragma unroll
            for (int j = 0; j < TN1; ++j) b[j] = Bs[k][threadIdx.x * TN1 + j];
#pragma unroll
            for (int i = 0; i < TM1; ++i) {
                const float af = __half2float(a[i]);
#pragma unroll
                for (int j = 0; j < TN1; ++j) acc[i][j] += af * __half2float(b[j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM1; ++i) {
        const int gr = row0 + threadIdx.y * TM1 + i;
#pragma unroll
        for (int j = 0; j < TN1; ++j) {
            const int gc = col0 + threadIdx.x * TN1 + j;
            const size_t o = (size_t)gr * N + gc;
            C[o] = __float2half(alpha * acc[i][j] + beta * __half2float(C[o]));
        }
    }
}

// V2: V1 + 共享内存 padding，消 16 路 bank conflict
__global__ void gemm_v2_pad(const half* __restrict__ A, const half* __restrict__ B,
                            half* __restrict__ C, int M, int N, int K,
                            float alpha, float beta) {
    __shared__ half As[BM1][BK1 + 2];      // 行距 18 half = 36B = 9 word（奇数）
    __shared__ half Bs[BK1][BN1 + 2];
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int row0 = blockIdx.y * BM1, col0 = blockIdx.x * BN1;
    float acc[TM1][TN1];
#pragma unroll
    for (int i = 0; i < TM1; ++i)
#pragma unroll
        for (int j = 0; j < TN1; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK1) {
#pragma unroll
        for (int t = 0; t < (BM1 * BK1) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / BK1, c = idx % BK1;
            As[r][c] = A[(size_t)(row0 + r) * K + k0 + c];
        }
#pragma unroll
        for (int t = 0; t < (BK1 * BN1) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / BN1, c = idx % BN1;
            Bs[r][c] = B[(size_t)(k0 + r) * N + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK1; ++k) {
            half a[TM1], b[TN1];
#pragma unroll
            for (int i = 0; i < TM1; ++i) a[i] = As[threadIdx.y * TM1 + i][k];
#pragma unroll
            for (int j = 0; j < TN1; ++j) b[j] = Bs[k][threadIdx.x * TN1 + j];
#pragma unroll
            for (int i = 0; i < TM1; ++i) {
                const float af = __half2float(a[i]);
#pragma unroll
                for (int j = 0; j < TN1; ++j) acc[i][j] += af * __half2float(b[j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM1; ++i) {
        const int gr = row0 + threadIdx.y * TM1 + i;
#pragma unroll
        for (int j = 0; j < TN1; ++j) {
            const int gc = col0 + threadIdx.x * TN1 + j;
            const size_t o = (size_t)gr * N + gc;
            C[o] = __float2half(alpha * acc[i][j] + beta * __half2float(C[o]));
        }
    }
}

// ============ 128x128 tile / 8x8 per thread / BK=32 ============
#define BM3 128
#define BN3 128
#define BK3 32
#define TM3 8
#define TN3 8

// V3: pad=8（16B 对齐）-> 全局 uint4 向量加载 + 共享 uint4 存储
__global__ void gemm_v3_big_vec(const half* __restrict__ A, const half* __restrict__ B,
                                half* __restrict__ C, int M, int N, int K,
                                float alpha, float beta) {
    __shared__ half As[BM3][BK3 + 8];
    __shared__ half Bs[BK3][BN3 + 8];
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int row0 = blockIdx.y * BM3, col0 = blockIdx.x * BN3;
    float acc[TM3][TN3];
#pragma unroll
    for (int i = 0; i < TM3; ++i)
#pragma unroll
        for (int j = 0; j < TN3; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK3) {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
#pragma unroll
        for (int t = 0; t < (BM3 * BK3 / 8) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / (BK3 / 8), c4 = idx % (BK3 / 8);
            *reinterpret_cast<uint4*>(&As[r][c4 * 8]) =
                A4[((size_t)(row0 + r) * K + k0) / 8 + c4];
        }
        const uint4* B4 = reinterpret_cast<const uint4*>(B);
#pragma unroll
        for (int t = 0; t < (BK3 * BN3 / 8) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / (BN3 / 8), c4 = idx % (BN3 / 8);
            *reinterpret_cast<uint4*>(&Bs[r][c4 * 8]) =
                B4[((size_t)(k0 + r) * N + col0) / 8 + c4];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK3; ++k) {
            half a[TM3], b[TN3];
#pragma unroll
            for (int i = 0; i < TM3; ++i) a[i] = As[threadIdx.y * TM3 + i][k];
#pragma unroll
            for (int j = 0; j < TN3; ++j) b[j] = Bs[k][threadIdx.x * TN3 + j];
#pragma unroll
            for (int i = 0; i < TM3; ++i) {
                const float af = __half2float(a[i]);
#pragma unroll
                for (int j = 0; j < TN3; ++j) acc[i][j] += af * __half2float(b[j]);
            }
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
            C[o] = __float2half(alpha * acc[i][j] + beta * __half2float(C[o]));
        }
    }
}

// V4: pad=2（消冲突更彻底，但行距 68B 非 16B 对齐）-> 共享只能标量存储
__global__ void gemm_v4_big_pad2(const half* __restrict__ A, const half* __restrict__ B,
                                 half* __restrict__ C, int M, int N, int K,
                                 float alpha, float beta) {
    __shared__ half As[BM3][BK3 + 2];
    __shared__ half Bs[BK3][BN3 + 2];
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int row0 = blockIdx.y * BM3, col0 = blockIdx.x * BN3;
    float acc[TM3][TN3];
#pragma unroll
    for (int i = 0; i < TM3; ++i)
#pragma unroll
        for (int j = 0; j < TN3; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK3) {
#pragma unroll
        for (int t = 0; t < (BM3 * BK3) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / BK3, c = idx % BK3;
            As[r][c] = A[(size_t)(row0 + r) * K + k0 + c];
        }
#pragma unroll
        for (int t = 0; t < (BK3 * BN3) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / BN3, c = idx % BN3;
            Bs[r][c] = B[(size_t)(k0 + r) * N + col0 + c];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK3; ++k) {
            half a[TM3], b[TN3];
#pragma unroll
            for (int i = 0; i < TM3; ++i) a[i] = As[threadIdx.y * TM3 + i][k];
#pragma unroll
            for (int j = 0; j < TN3; ++j) b[j] = Bs[k][threadIdx.x * TN3 + j];
#pragma unroll
            for (int i = 0; i < TM3; ++i) {
                const float af = __half2float(a[i]);
#pragma unroll
                for (int j = 0; j < TN3; ++j) acc[i][j] += af * __half2float(b[j]);
            }
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
            C[o] = __float2half(alpha * acc[i][j] + beta * __half2float(C[o]));
        }
    }
}

// ============ V6: WMMA 张量核（题目明确允许 WMMA）============
// 64x64 tile / BK=32 / 128 线程（4 warp，2x2 排布，每 warp 32x32）
// 小 tile 的原因：1024^2 规模下总 block 数要够多才能喂满 170 个 SM
#define BW_M 64
#define BW_N 64
#define BW_K 32
#define BW_LDA (BW_K + 8)
#define BW_LDB (BW_N + 8)

__global__ void gemm_wmma64(const half* __restrict__ A, const half* __restrict__ B,
                            half* __restrict__ C, int M, int N, int K,
                            float alpha, float beta) {
    __shared__ half As[BW_M][BW_LDA];
    __shared__ half Bs[BW_K][BW_LDB];
    __shared__ float stg[BW_M][BW_N + 4];      // 行距 68：fp32 累加器要求 ldm%4==0

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BW_M, col0 = blockIdx.x * BW_N;
    const int warp = tid / 32;
    const int wm = warp % 2, wn = warp / 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BW_K) {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
#pragma unroll
        for (int t = 0; t < (BW_M * BW_K / 8) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / (BW_K / 8), c4 = idx % (BW_K / 8);
            *reinterpret_cast<uint4*>(&As[r][c4 * 8]) =
                A4[((size_t)(row0 + r) * K + k0) / 8 + c4];
        }
        const uint4* B4 = reinterpret_cast<const uint4*>(B);
#pragma unroll
        for (int t = 0; t < (BW_K * BW_N / 8) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / (BW_N / 8), c4 = idx % (BW_N / 8);
            *reinterpret_cast<uint4*>(&Bs[r][c4 * 8]) =
                B4[((size_t)(k0 + r) * N + col0) / 8 + c4];
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BW_K; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wm * 32 + i * 16][kk], BW_LDA);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wn * 32 + j * 16], BW_LDB);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&stg[wm * 32 + i * 16][wn * 32 + j * 16],
                                    acc[i][j], BW_N + 4, wmma::mem_row_major);
    __syncthreads();

#pragma unroll 4
    for (int idx = tid; idx < BW_M * BW_N; idx += 128) {
        const int r = idx / BW_N, c = idx % BW_N;
        const size_t o = (size_t)(row0 + r) * N + col0 + c;
        C[o] = __float2half(alpha * stg[r][c] + beta * __half2float(C[o]));
    }
}

// ============ V6: warp 级分块（CUDA core，非张量核）============
// block 64(M)x64(N) / BK=16 / 4 warps 排 2x2 / 每 warp 32x32 / 每线程 8x4 = 32 累加器
// 与 V1（256 线程、每线程 4x4）对比的核心差异：
//   每个 k 的共享读 = TM+TN = 12 次，服务 8*4 = 32 次 FMA -> 0.375 读/FMA
//   V1 是 4+4 = 8 次服务 16 次 FMA -> 0.5 读/FMA
// 即共享读压力降 25%，代价是 block 只有 128 线程，要靠更多 block 填 SM。
#define W6_M 64
#define W6_N 64
#define W6_K 16
#define W6_APAD (W6_K + 2)
#define W6_BPAD (W6_N + 2)

__global__ void gemm_warp64(const half* __restrict__ A, const half* __restrict__ B,
                            half* __restrict__ C, int M, int N, int K,
                            float alpha, float beta) {
    __shared__ half As[W6_M][W6_APAD];
    __shared__ half Bs[W6_K][W6_BPAD];

    const int tid  = threadIdx.x;                    // 128 线程 = 4 warp
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;         // 2x2 warp 网格
    const int lm = lane & 3, ln = lane >> 2;         // warp 内 4x8 线程网格
    const int row0 = blockIdx.y * W6_M, col0 = blockIdx.x * W6_N;

    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += W6_K) {
#pragma unroll
        for (int t = 0; t < (W6_M * W6_K) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / W6_K, c = idx % W6_K;
            As[r][c] = A[(size_t)(row0 + r) * K + k0 + c];
        }
#pragma unroll
        for (int t = 0; t < (W6_K * W6_N) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / W6_N, c = idx % W6_N;
            Bs[r][c] = B[(size_t)(k0 + r) * N + col0 + c];
        }
        __syncthreads();

        const int rb = wm * 32 + lm * 8;             // 本线程负责的 8 行起点
        const int cb = wn * 32 + ln * 4;             // 本线程负责的 4 列起点
#pragma unroll
        for (int k = 0; k < W6_K; ++k) {
            half a[8], b[4];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = As[rb + i][k];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][cb + j];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float af = __half2float(a[i]);
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += af * __half2float(b[j]);
            }
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
            C[o] = __float2half(alpha * acc[i][j] + beta * __half2float(C[o]));
        }
    }
}

// ============ V7: warp 级分块 + 更大 block tile（张量核）============
// block 128(M)x64(N) / BK=32 / 8 warps 排 4x2 / 每 warp 32x32（2x2 fragment）
// 与 V5 的差别只有 block tile 形状：
//   全局载入字节/输出 = (128*32 + 32*64)*2 / (128*64) = 1.5 B/out   （V5 是 2.0）
//   但 block 数从 256 掉到 128（5090 有 170 个 SM）-> 这是在赌哪边赢
#define W7_M 128
#define W7_N 64
#define W7_K 32
// As/Bs 不 pad（40/72 会让 stg 超 48KB 静态上限）；stg 行距必须 %4==0
__global__ void gemm_wmma_warp(const half* __restrict__ A, const half* __restrict__ B,
                               half* __restrict__ C, int M, int N, int K,
                               float alpha, float beta) {
    __shared__ half  As[W7_M][W7_K];
    __shared__ half  Bs[W7_K][W7_N];
    __shared__ float stg[W7_M][W7_N + 4];

    const int tid = threadIdx.x;                     // 256 线程 = 8 warp
    const int row0 = blockIdx.y * W7_M, col0 = blockIdx.x * W7_N;
    const int warp = tid / 32;
    const int wm = warp % 4, wn = warp / 4;          // 4x2 warp 网格

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += W7_K) {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
#pragma unroll
        for (int t = 0; t < (W7_M * W7_K / 8) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / (W7_K / 8), c4 = idx % (W7_K / 8);
            *reinterpret_cast<uint4*>(&As[r][c4 * 8]) =
                A4[((size_t)(row0 + r) * K + k0) / 8 + c4];
        }
        const uint4* B4 = reinterpret_cast<const uint4*>(B);
#pragma unroll
        for (int t = 0; t < (W7_K * W7_N / 8) / 256; ++t) {
            const int idx = tid + t * 256, r = idx / (W7_N / 8), c4 = idx % (W7_N / 8);
            *reinterpret_cast<uint4*>(&Bs[r][c4 * 8]) =
                B4[((size_t)(k0 + r) * N + col0) / 8 + c4];
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < W7_K; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wm * 32 + i * 16][kk], W7_K);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wn * 32 + j * 16], W7_N);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&stg[wm * 32 + i * 16][wn * 32 + j * 16],
                                    acc[i][j], W7_N + 4, wmma::mem_row_major);
    __syncthreads();

#pragma unroll 4
    for (int idx = tid; idx < W7_M * W7_N; idx += 256) {
        const int r = idx / W7_N, c = idx % W7_N;
        const size_t o = (size_t)(row0 + r) * N + col0 + c;
        C[o] = __float2half(alpha * stg[r][c] + beta * __half2float(C[o]));
    }
}

// ============ V8: V5 但 BK 32 -> 64（把 __syncthreads 次数砍一半）============
// BK=32 时 K=1024 要走 32 轮，每轮 2 次 __syncthreads = 64 次；
// BK=64 只要 16 轮 = 32 次。共享占用 64x72 + 64x72 half ≈ 18KB，仍然宽松。
#define W8_M 64
#define W8_N 64
#define W8_K 64
#define W8_LDA (W8_K + 8)
#define W8_LDB (W8_N + 8)

__global__ void gemm_wmma64_bk64(const half* __restrict__ A, const half* __restrict__ B,
                                 half* __restrict__ C, int M, int N, int K,
                                 float alpha, float beta) {
    __shared__ half  As[W8_M][W8_LDA];
    __shared__ half  Bs[W8_K][W8_LDB];
    __shared__ float stg[W8_M][W8_N + 4];

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * W8_M, col0 = blockIdx.x * W8_N;
    const int warp = tid / 32;
    const int wm = warp % 2, wn = warp / 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += W8_K) {
        const uint4* A4 = reinterpret_cast<const uint4*>(A);
#pragma unroll
        for (int t = 0; t < (W8_M * W8_K / 8) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / (W8_K / 8), c4 = idx % (W8_K / 8);
            *reinterpret_cast<uint4*>(&As[r][c4 * 8]) =
                A4[((size_t)(row0 + r) * K + k0) / 8 + c4];
        }
        const uint4* B4 = reinterpret_cast<const uint4*>(B);
#pragma unroll
        for (int t = 0; t < (W8_K * W8_N / 8) / 128; ++t) {
            const int idx = tid + t * 128, r = idx / (W8_N / 8), c4 = idx % (W8_N / 8);
            *reinterpret_cast<uint4*>(&Bs[r][c4 * 8]) =
                B4[((size_t)(k0 + r) * N + col0) / 8 + c4];
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < W8_K; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wm * 32 + i * 16][kk], W8_LDA);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wn * 32 + j * 16], W8_LDB);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&stg[wm * 32 + i * 16][wn * 32 + j * 16],
                                    acc[i][j], W8_N + 4, wmma::mem_row_major);
    __syncthreads();

#pragma unroll 4
    for (int idx = tid; idx < W8_M * W8_N; idx += 128) {
        const int r = idx / W8_N, c = idx % W8_N;
        const size_t o = (size_t)(row0 + r) * N + col0 + c;
        C[o] = __float2half(alpha * stg[r][c] + beta * __half2float(C[o]));
    }
}

// ============ 空 kernel：量「单次 launch 固定开销」============
// ctypes 每次 cuLaunchKernel 约 7us。1024^3 的 cuBLAS 只要 ~17us，
// 不减去这个常数项，所有比值都会被向 1 压缩（高估慢的、低估快的）。
__global__ void noop_probe(float* p) { if (p && threadIdx.x > 1024) p[0] = 1.f; }
"""

# (标签, kernel, grid 计算, block)
VARIANTS = [
    ("V0 naive 1线程/元素", "gemm_v0_naive",
     lambda M, N: ((N + 15) // 16, (M + 15) // 16), (16, 16)),
    ("V1 tiled 64x64/4x4/BK16 无pad", "gemm_v1_tiled",
     lambda M, N: (N // 64, M // 64), (16, 16)),
    ("V2 V1+pad2 消bank冲突", "gemm_v2_pad",
     lambda M, N: (N // 64, M // 64), (16, 16)),
    ("V3 128x128/8x8/BK32 pad8+uint4", "gemm_v3_big_vec",
     lambda M, N: (N // 128, M // 128), (16, 16)),
    ("V4 128x128/8x8/BK32 pad2+标量", "gemm_v4_big_pad2",
     lambda M, N: (N // 128, M // 128), (16, 16)),
    ("V5 WMMA 64x64/BK32 张量核", "gemm_wmma64",
     lambda M, N: (N // 64, M // 64), 128),
    ("V6 warp分块 CUDAcore 64x64/8x4", "gemm_warp64",
     lambda M, N: (N // 64, M // 64), 128),
    ("V7 warp分块 WMMA 128x64/4x2", "gemm_wmma_warp",
     lambda M, N: (N // 64, M // 128), 256),
    ("V8 V5+BK64 同步次数减半", "gemm_wmma64_bk64",
     lambda M, N: (N // 64, M // 64), 128),
]

WARP_NEW = {"gemm_warp64", "gemm_wmma_warp", "gemm_wmma64_bk64"}


OH_MS = 0.0     # 单次 launch 的 host 固定开销（ms），main 里实测后填


def flag(ms):
    """标记测量可信度：实测值 vs host 固定开销。

    比值 >= 4 说明 GPU 时间明显压过 host 开销，数字可信；
    否则这个规模下测得的是 host 抖动，不能用于横向比较。
    """
    if OH_MS <= 0:
        return ""
    r = ms / OH_MS
    return "OK" if r >= 4.0 else "~host(%.1fx)" % r


def launch_overhead_ms(r, repeat=20, warmup=5):
    """量单次 launch 的固定开销：空 kernel 走同一条 timeit 路径。"""
    d = r.alloc(4)
    args = [d]
    r.launch("noop_probe", 1, 1, args)
    r.sync()
    ms = r.timeit(lambda: r.launch("noop_probe", 1, 1, args),
                  repeat=repeat, warmup=warmup)
    r.free(d)
    return ms


def bench(r, dA, dB, dC, M, N, K, kname, gridfn, blk, repeat=20, warmup=5,
          alpha=1.0, beta=0.0):
    """返回**实测原始**单次耗时（ms）。

    ⚠️ 不要无条件减去 launch 开销：
    - host 每次调用的固定成本 = HOST_MS；GPU 时间 = G。
    - 若 HOST_MS < G，GPU 流水线填满，实测值**就是** G，再减就低估；
    - 若 HOST_MS > G，实测值被 host 顶住，等于 HOST_MS（此时 G 测不出来）。
    所以正确做法是报告原始值 + HOST_MS，并用 `raw / HOST_MS` 判断可信度：
    比值越大越可信；比值 < 4 就说明这个规模下数字被 host 污染，不该拿来下结论。
    """
    args = [dA, dB, dC, r.i(M), r.i(N), r.i(K), r.f(alpha), r.f(beta)]
    r.launch(kname, gridfn(M, N), blk, args)
    r.sync()
    return r.timeit(lambda: r.launch(kname, gridfn(M, N), blk, args),
                    repeat=repeat, warmup=warmup)


def check(r, kname, gridfn, blk, dA, dB, C0, M, N, K, alpha, beta):
    """用指定的 alpha/beta 跑一次并和 numpy float64 参考比。

    **必须在 beta != 0 时也验一次** —— 只测 alpha=1/beta=0 会漏掉整个
    beta*C_initial 路径（V5/V7/V8 的落地暂存、V6 的读改写都在那一段）。
    """
    dC = r.to_device(C0)
    args = [dA, dB, dC, r.i(M), r.i(N), r.i(K), r.f(alpha), r.f(beta)]
    r.launch(kname, gridfn(M, N), blk, args)
    r.sync()
    got = r.from_device(dC, (M, N), np.float16).astype(np.float64)
    r.free(dC)
    ref = (alpha * (A64 @ B64) + beta * C0.astype(np.float64)).astype(np.float16)
    return bool(np.allclose(got, ref.astype(np.float64), rtol=3e-3, atol=1e-3))


A64 = B64 = None          # 在 main 里填，check() 用


def main():
    global A64, B64
    ptx, arch, log = nvrtc_compile(SRC, name="gemm_ladder.cu")
    r = Runner(ptx, arch, log)
    print("设备:", Runner.devicename(), "| PTX arch:", arch)
    if log.strip():
        print("编译日志:", log.strip()[:400])
    print()

    M = N = K = 1024
    flop = 2.0 * M * N * K
    rng = np.random.default_rng(22)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    C0 = rng.standard_normal((M, N)).astype(np.float16)
    A64, B64 = A.astype(np.float64), B.astype(np.float64)
    dA, dB = r.to_device(A), r.to_device(B)

    # ---- cuBLAS 基线（必须在 Runner 建好 context 之后创建 handle）----
    try:
        from cublas import Handle
        blas = Handle()
        print("cuBLAS 版本: %s" % blas.version)
    except Exception as e:
        blas = None
        print("!! cuBLAS 不可用，跳过基线: %r" % (e,))

    global OH_MS
    OH_MS = launch_overhead_ms(r)
    print("单次 launch 的 host 固定开销 = %.3f us"
          "（实测值 / 它 < 4 的规模，数字不可信）" % (OH_MS * 1000))

    print("== tile 与 block 数的张力：128x128 只有 %d 个 block，64x64 有 %d 个；"
          "本机 SM 数见上 ==" % ((M // 128) ** 2, (M // 64) ** 2))
    print()

    print("%-34s %6s %10s %10s %8s %8s %6s"
          % ("变体", "block", "耗时", "GFLOPS", "vs naive", "vs cuBLAS", "正确"))
    print("-" * 92)
    base = None
    blas_gf = None
    results = []

    if blas is not None:
        dC = r.to_device(C0)
        blas.gemm_f16_f32acc(dA, dB, dC, M, N, K, 1.0, 0.0)
        r.sync()
        got = r.from_device(dC, (M, N), np.float16).astype(np.float64)
        ref = (A64 @ B64).astype(np.float16).astype(np.float64)
        okb = bool(np.allclose(got, ref, rtol=3e-3, atol=1e-3))
        msb = r.timeit(lambda: blas.gemm_f16_f32acc(dA, dB, dC, M, N, K, 1.0, 0.0),
                       repeat=20, warmup=5)
        blas_gf = flop / (msb * 1e-3) / 1e9
        r.free(dC)
        print("%-34s %6s %8.3fms %10.1f %8s %8s %6s %s"
              % ("cuBLAS GemmEx(fp16/fp32acc)", "-", msb, blas_gf,
                 "-", "1.00x", "OK" if okb else "FAIL", flag(msb)))

    for label, kname, gridfn, blk in VARIANTS:
        g = gridfn(M, N)
        nb = g[0] * g[1]
        ok0 = check(r, kname, gridfn, blk, dA, dB, C0, M, N, K, 1.0, 0.0)
        ok1 = check(r, kname, gridfn, blk, dA, dB, C0, M, N, K, 1.5, 0.5)
        ms = bench(r, dA, dB, r.to_device(C0), M, N, K, kname, gridfn, blk)
        gf = flop / (ms * 1e-3) / 1e9
        base = gf if base is None else base
        results.append((label, kname, gridfn, blk, ms, gf))
        print("%-34s %6d %8.3fms %10.1f %7.2fx %7s %6s %s"
              % (label, nb, ms, gf, gf / base,
                 ("%.2fx" % (gf / blas_gf)) if blas_gf else "-",
                 "OK" if (ok0 and ok1) else ("FAIL(a%b)" % (1 if not ok0 else 0)),
                 flag(ms)))

    print()
    print("== 规模扫描：tile 大小 vs 机器规模（GFLOPS）==")
    print("%-34s %12s %12s %12s" % ("变体", "1024^3", "2048^3", "4096^3"))
    print("-" * 76)
    sweep = [(l, k, g, b) for (l, k, g, b, _, _) in results if k != "gemm_v0_naive"]
    if blas is not None:
        sweep.insert(0, ("cuBLAS GemmEx", "__cublas__", None, None))
    for label, kname, gridfn, blk in sweep:
        row = []
        for sz in (1024, 2048, 4096):
            Asz = (rng.standard_normal((sz, sz)) * 0.5).astype(np.float16)
            Bsz = (rng.standard_normal((sz, sz)) * 0.5).astype(np.float16)
            Csz = np.zeros((sz, sz), np.float16)
            xA, xB, xC = r.to_device(Asz), r.to_device(Bsz), r.to_device(Csz)
            if kname == "__cublas__":
                ms = r.timeit(
                    lambda: blas.gemm_f16_f32acc(xA, xB, xC, sz, sz, sz, 1.0, 0.0),
                    repeat=5, warmup=2)
            else:
                ms = bench(r, xA, xB, xC, sz, sz, sz, kname, gridfn, blk,
                           repeat=5, warmup=2)
            row.append(2.0 * sz ** 3 / (ms * 1e-3) / 1e9)
            for p in (xA, xB, xC):
                r.free(p)
        print("%-34s %12.1f %12.1f %12.1f"
              % (label, row[0], row[1], row[2]))

    if blas is not None:
        blas.destroy()
    print()
    print("注：加速比基准 = V0 naive。tile 越大共享复用越多，但 block 数变少；")
    print("    170 SM 的机器在 1024^3 下更吃「block 数够不够」而不是单块效率。")
    print("    cuBLAS 是 fp16 入 / fp32 累加 / fp16 出的同一语义，行主序靠交换 A/B 映射。")


if __name__ == "__main__":
    sys.exit(main())
