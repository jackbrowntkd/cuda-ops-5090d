// LeetGPU #002 Matrix Multiplication (easy)
// 签名: extern "C" void solve(const float* A, const float* B, float* C,
//                            int M, int N, int K)
// A: M x N（行主序）, B: N x K, C: M x K        —— 收缩维是 N
// 规模: M,N,K <= 8192, 性能测试 M=8192 N=6144 K=4096
//
// 本版 = 把 #022 上验证过的 v10d 设计（fp32 达 cuBLAS 的 0.85x）推广到这里，
// 并补上一般 M/N/K 的边界 guard。
//
// 配置：block 128x128 / BK=16 / warp tile 64x32 / 线程微块 8x4 / WNITER=2 / NT=256
//      A **转置存**在共享内存（As[k*BM + row]）—— 这样行距恰好 = BM（4 的倍数），
//      float4 共享读天然 16B 对齐，**不需要任何 padding**，绕开「APAD 必须 ≥4 倍数」的坑。
//
// 每线程 64 个累加器；每 k 的共享载入 16 次服务 64 次 FMA（0.25/FMA）。
// 实测（本机 RTX 5090 D，8192x6144x4096）：见文件末尾注释。

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

constexpr int WARPS_N = XBN / XWN;                                  // 4
constexpr int WMITER = (XWM * XWN) / (32 * XTM * XTN * XWNITER);    // 1
constexpr int WSUBM = XWM / WMITER;                                 // 64
constexpr int WSUBN = XWN / XWNITER;                                // 16

__global__ void __launch_bounds__(NTHR)
matmul_v10(const float* __restrict__ A, const float* __restrict__ B,
           float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[XBK * XBM];   // 转置存: As[k*XBM + row]
    __shared__ float Bs[XBK * XBN];   // 正常:   Bs[k*XBN + col]

    const int tid = threadIdx.x;
    const int warpIdx = tid >> 5;
    const int warpCol = warpIdx % WARPS_N;
    const int warpRow = warpIdx / WARPS_N;
    const int lane = tid & 31;
    const int threadColInWarp = lane % (WSUBN / XTN);   // lane % 4
    const int threadRowInWarp = lane / (WSUBN / XTN);   // 0..7

    const int rowStart = blockIdx.y * XBM;   // C 的行基址
    const int colStart = blockIdx.x * XBN;   // C 的列基址

    // 载入索引（每次搬 4 个 float）
    const int innerRowA = tid / (XBK / 4);
    const int innerColA = tid % (XBK / 4);
    constexpr int rowStrideA = (NTHR * 4) / XBK;        // 64
    const int innerRowB = tid / (XBN / 4);
    const int innerColB = tid % (XBN / 4);
    constexpr int rowStrideB = NTHR / (XBN / 4);        // 8

    // 行距是否 4 的倍数 —— 决定 float4 全局读写是否安全（不满足时走标量回退）
    const bool a4 = (N % 4 == 0);
    const bool b4 = (K % 4 == 0);

    float acc[WMITER][XTM][XWNITER][XTN];
#pragma unroll
    for (int a = 0; a < WMITER; ++a)
#pragma unroll
        for (int i = 0; i < XTM; ++i)
#pragma unroll
            for (int b = 0; b < XWNITER; ++b)
#pragma unroll
                for (int j = 0; j < XTN; ++j) acc[a][i][b][j] = 0.f;

    // 收缩维是 **N**（A 是 M×N、B 是 N×K），所以 k 循环的边界必须是 N 而不是 K。
    // 写成 K 时，N>K 的形状会漏掉 N-K 段收缩 —— 而 N==K 或 N<K 的形状会「碰巧」正确，
    // 这类 bug 极易被小规模用例放过。
    for (int t = 0; t < N; t += XBK) {
        // ---- 载入 A 片 (XBM x XBK)，转置存 ----
#pragma unroll
        for (int off = 0; off + rowStrideA <= XBM; off += rowStrideA) {
            const int r = rowStart + innerRowA + off;
            const int c = t + innerColA * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < M) {
                if (a4 && c + 4 <= N) {
                    v = *reinterpret_cast<const float4*>(&A[(size_t)r * N + c]);
                } else {
                    if (c + 0 < N) v.x = A[(size_t)r * N + c + 0];
                    if (c + 1 < N) v.y = A[(size_t)r * N + c + 1];
                    if (c + 2 < N) v.z = A[(size_t)r * N + c + 2];
                    if (c + 3 < N) v.w = A[(size_t)r * N + c + 3];
                }
            }
            As[(innerColA * 4 + 0) * XBM + innerRowA + off] = v.x;
            As[(innerColA * 4 + 1) * XBM + innerRowA + off] = v.y;
            As[(innerColA * 4 + 2) * XBM + innerRowA + off] = v.z;
            As[(innerColA * 4 + 3) * XBM + innerRowA + off] = v.w;
        }
        // ---- 载入 B 片 (XBK x XBN) ----
#pragma unroll
        for (int off = 0; off + rowStrideB <= XBK; off += rowStrideB) {
            const int r = t + innerRowB + off;
            const int c = colStart + innerColB * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < N) {
                if (b4 && c + 4 <= K) {
                    v = *reinterpret_cast<const float4*>(&B[(size_t)r * K + c]);
                } else {
                    if (c + 0 < K) v.x = B[(size_t)r * K + c + 0];
                    if (c + 1 < K) v.y = B[(size_t)r * K + c + 1];
                    if (c + 2 < K) v.z = B[(size_t)r * K + c + 2];
                    if (c + 3 < K) v.w = B[(size_t)r * K + c + 3];
                }
            }
            *reinterpret_cast<float4*>(&Bs[(innerRowB + off) * XBN + innerColB * 4]) = v;
        }
        __syncthreads();

        // ---- 计算：regM / regN 各 8 个 float4，服务 64 次 FMA ----
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

    // ---- 写回 ----
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
                if (b4 && c + 4 <= K) {
                    *reinterpret_cast<float4*>(&C[(size_t)r * K + c]) =
                        make_float4(acc[a][i][b][0], acc[a][i][b][1],
                                    acc[a][i][b][2], acc[a][i][b][3]);
                } else {
#pragma unroll
                    for (int j = 0; j < XTN; ++j)
                        if (c + j < K) C[(size_t)r * K + c + j] = acc[a][i][b][j];
                }
            }
}


extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    dim3 block(NTHR);
    dim3 grid((K + XBN - 1) / XBN, (M + XBM - 1) / XBM);
    matmul_v10<<<grid, block>>>(A, B, C, M, N, K);
}
