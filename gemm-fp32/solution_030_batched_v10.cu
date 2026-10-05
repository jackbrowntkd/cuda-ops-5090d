// LeetGPU #030 Batched Matrix Multiplication (medium)
// 签名: extern "C" void solve(const float* A, const float* B, float* C,
//                            int BATCH, int M, int N, int K)
// A: (BATCH, M, K), B: (BATCH, K, N), C: (BATCH, M, N)   —— 收缩维是 K
// 规模: B<=128, M,N,K<=1024, 性能测试 K=M=N=256
//
// = #002 的 v10d 设计 + batch 维度（放在 gridDim.z）。
// 配置：block 128x128 / BK=16 / warp tile 64x32 / 线程微块 8x4 / WNITER=2 / NT=256
//      A 转置存在共享内存，行距 = BM，float4 读写天然 16B 对齐，无需 padding。

#include <cuda_runtime.h>

#define XBM 128
#define XBN 128
#define XBK 16
#define XWM 64
#define XWN 32
#define XTM 8
#define XTN 4
#define XWNITER 2
#define NTHR 256

constexpr int WARPS_N = XBN / XWN;                                // 4
constexpr int WMITER = (XWM * XWN) / (32 * XTM * XTN * XWNITER);  // 1
constexpr int WSUBM = XWM / WMITER;                               // 64
constexpr int WSUBN = XWN / XWNITER;                              // 16

__global__ void __launch_bounds__(NTHR)
bmm_v10(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int N, int K) {
    // 本 batch 的基址（batch 走 gridDim.z）
    const size_t aBaseG = (size_t)blockIdx.z * M * K;
    const size_t bBaseG = (size_t)blockIdx.z * K * N;
    const size_t cBaseG = (size_t)blockIdx.z * M * N;
    A += aBaseG;  B += bBaseG;  C += cBaseG;

    __shared__ float As[XBK * XBM];   // 转置存: As[k*XBM + row]
    __shared__ float Bs[XBK * XBN];

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
    constexpr int rowStrideA = (NTHR * 4) / XBK;   // 64
    const int innerRowB = tid / (XBN / 4);
    const int innerColB = tid % (XBN / 4);
    constexpr int rowStrideB = NTHR / (XBN / 4);   // 8

    const bool a4 = (K % 4 == 0);   // A 的行距是 K
    const bool b4 = (N % 4 == 0);   // B/C 的行距是 N

    float acc[WMITER][XTM][XWNITER][XTN];
#pragma unroll
    for (int a = 0; a < WMITER; ++a)
#pragma unroll
        for (int i = 0; i < XTM; ++i)
#pragma unroll
            for (int b = 0; b < XWNITER; ++b)
#pragma unroll
                for (int j = 0; j < XTN; ++j) acc[a][i][b][j] = 0.f;

    // 收缩维 = K
    for (int t = 0; t < K; t += XBK) {
#pragma unroll
        for (int off = 0; off + rowStrideA <= XBM; off += rowStrideA) {
            const int r = rowStart + innerRowA + off;
            const int c = t + innerColA * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < M) {
                if (a4 && c + 4 <= K) {
                    v = *reinterpret_cast<const float4*>(&A[(size_t)r * K + c]);
                } else {
                    if (c + 0 < K) v.x = A[(size_t)r * K + c + 0];
                    if (c + 1 < K) v.y = A[(size_t)r * K + c + 1];
                    if (c + 2 < K) v.z = A[(size_t)r * K + c + 2];
                    if (c + 3 < K) v.w = A[(size_t)r * K + c + 3];
                }
            }
            As[(innerColA * 4 + 0) * XBM + innerRowA + off] = v.x;
            As[(innerColA * 4 + 1) * XBM + innerRowA + off] = v.y;
            As[(innerColA * 4 + 2) * XBM + innerRowA + off] = v.z;
            As[(innerColA * 4 + 3) * XBM + innerRowA + off] = v.w;
        }
#pragma unroll
        for (int off = 0; off + rowStrideB <= XBK; off += rowStrideB) {
            const int r = t + innerRowB + off;
            const int c = colStart + innerColB * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < K) {
                if (b4 && c + 4 <= N) {
                    v = *reinterpret_cast<const float4*>(&B[(size_t)r * N + c]);
                } else {
                    if (c + 0 < N) v.x = B[(size_t)r * N + c + 0];
                    if (c + 1 < N) v.y = B[(size_t)r * N + c + 1];
                    if (c + 2 < N) v.z = B[(size_t)r * N + c + 2];
                    if (c + 3 < N) v.w = B[(size_t)r * N + c + 3];
                }
            }
            *reinterpret_cast<float4*>(&Bs[(innerRowB + off) * XBN + innerColB * 4]) = v;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < XBK; ++k) {
            float regM[WMITER * XTM];
            float regN[XWNITER * XTN];
#pragma unroll
            for (int q = 0; q < WMITER * XTM; q += 4)
                *reinterpret_cast<float4*>(&regM[q]) =
                    *reinterpret_cast<const float4*>(
                        &As[k * XBM + warpRow * XWM + threadRowInWarp * XTM + q]);
#pragma unroll
            for (int b = 0; b < XWNITER; ++b)
#pragma unroll
                for (int q = 0; q < XTN; q += 4)
                    *reinterpret_cast<float4*>(&regN[b * XTN + q]) =
                        *reinterpret_cast<const float4*>(
                            &Bs[k * XBN + warpCol * XWN + b * WSUBN
                                + threadColInWarp * XTN + q]);
#pragma unroll
            for (int a = 0; a < WMITER; ++a)
#pragma unroll
                for (int b = 0; b < XWNITER; ++b)
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
            for (int b = 0; b < XWNITER; ++b) {
                const int r = rowStart + warpRow * XWM + a * WSUBM
                              + threadRowInWarp * XTM + i;
                const int c = colStart + warpCol * XWN + b * WSUBN
                              + threadColInWarp * XTN;
                if (r >= M) continue;
                if (b4 && c + 4 <= N) {
                    *reinterpret_cast<float4*>(&C[(size_t)r * N + c]) =
                        make_float4(acc[a][i][b][0], acc[a][i][b][1],
                                    acc[a][i][b][2], acc[a][i][b][3]);
                } else {
#pragma unroll
                    for (int j = 0; j < XTN; ++j)
                        if (c + j < N) C[(size_t)r * N + c + j] = acc[a][i][b][j];
                }
            }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int BATCH, int M, int N, int K) {
    if (BATCH <= 0 || M <= 0 || N <= 0 || K <= 0) return;
    dim3 block(NTHR);
    dim3 grid((N + XBN - 1) / XBN, (M + XBM - 1) / XBM, BATCH);
    bmm_v10<<<grid, block>>>(A, B, C, M, N, K);
}
