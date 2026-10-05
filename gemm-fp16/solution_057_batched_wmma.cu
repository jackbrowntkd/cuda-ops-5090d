// LeetGPU #057 FP16 Batched Matrix Multiplication (medium)
// 签名: extern "C" void solve(const half* A, const half* B, half* C,
//                            int BATCH, int M, int N, int K)
// A: (BATCH, M, K), B: (BATCH, K, N), C: (BATCH, M, N)    —— 收缩维是 K
// 规模: B<=128, M,N,K<=1024, 性能测试 K=M=N=256
//
// = #022 上已验证的 WMMA 设计（BK=64 是关键收益点）+ batch 维度（放在 gridDim.z）。
//   #022 实测：WMMA BK=32 ≈64.5 TFLOPS -> BK=64 ≈86.8，是本设计最大的单项收益
//   （K 方向的 __syncthreads 次数减半）。
// 本版相对 #022 的差异：
//   - 去掉 alpha/beta（#057 是纯 C = A*B），写回不需要读 C_initial
//   - batch 基址由 blockIdx.z 决定
//   - 仍保留共享暂存，因为 accumulator fragment 的元素->(m,n) 映射不可移植，
//     且 N 非 8 倍数时不能直接 store_matrix_sync 到全局
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 64
#define NT 128                 // 4 warp
#define LDA (BK + 8)           // 72 half = 144B，16B 对齐
#define LDB (BN + 8)           // 72 half
#define LDG (BN + 4)           // 68 float

__global__ void __launch_bounds__(NT)
fp16_bmm_wmma(const half* __restrict__ A, const half* __restrict__ B,
              half* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(16) half As[BM][LDA];
    __shared__ __align__(16) half Bs[BK][LDB];
    __shared__ __align__(16) float stg[BM][LDG];

    // 本 batch 的基址
    A += (size_t)blockIdx.z * M * K;
    B += (size_t)blockIdx.z * K * N;
    C += (size_t)blockIdx.z * M * N;

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const int warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;

    const bool vec_a = (K % 8 == 0);
    const bool vec_b = (N % 8 == 0);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- A 片 (BM x BK) ----
        if (vec_a) {
            const uint4* A4 = reinterpret_cast<const uint4*>(A);
#pragma unroll
            for (int t = 0; t < (BM * BK / 8) / NT; ++t) {
                const int idx = tid + t * NT, r = idx / (BK / 8), c8 = idx % (BK / 8);
                const int gr = row0 + r, gc = k0 + c8 * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (gr < M && gc < K) v = A4[((size_t)gr * K + gc) / 8];
                *reinterpret_cast<uint4*>(&As[r][c8 * 8]) = v;
            }
        } else {
#pragma unroll
            for (int t = 0; t < (BM * BK) / NT; ++t) {
                const int idx = tid + t * NT, r = idx / BK, c = idx % BK;
                const int gr = row0 + r, gc = k0 + c;
                As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc]
                                              : __float2half(0.f);
            }
        }
        // ---- B 片 (BK x BN) ----
        if (vec_b) {
            const uint4* B4 = reinterpret_cast<const uint4*>(B);
#pragma unroll
            for (int t = 0; t < (BK * BN / 8) / NT; ++t) {
                const int idx = tid + t * NT, r = idx / (BN / 8), c8 = idx % (BN / 8);
                const int gr = k0 + r, gc = col0 + c8 * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (gr < K && gc < N) v = B4[((size_t)gr * N + gc) / 8];
                *reinterpret_cast<uint4*>(&Bs[r][c8 * 8]) = v;
            }
        } else {
#pragma unroll
            for (int t = 0; t < (BK * BN) / NT; ++t) {
                const int idx = tid + t * NT, r = idx / BN, c = idx % BN;
                const int gr = k0 + r, gc = col0 + c;
                Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc]
                                              : __float2half(0.f);
            }
        }
        __syncthreads();

        // ---- 张量核 ----
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wm * 32 + i * 16][kk], LDA);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wn * 32 + j * 16], LDB);
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
                                    acc[i][j], LDG, wmma::mem_row_major);
    __syncthreads();

#pragma unroll 4
    for (int idx = tid; idx < BM * BN; idx += NT) {
        const int r = idx / BN, c = idx % BN;
        const int gr = row0 + r, gc = col0 + c;
        if (gr >= M || gc >= N) continue;
        C[(size_t)gr * N + gc] = __float2half(stg[r][c]);
    }
}

extern "C" void solve(const half* A, const half* B, half* C,
                      int BATCH, int M, int N, int K) {
    if (!A || !B || !C) return;
    if (BATCH <= 0 || M <= 0 || N <= 0 || K <= 0) return;
    dim3 blk(NT);
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    fp16_bmm_wmma<<<grd, blk>>>(A, B, C, M, N, K);
}
