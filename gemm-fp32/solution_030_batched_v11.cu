// LeetGPU #030 Batched Matrix Multiplication (medium)
// 签名: extern "C" void solve(const float* A, const float* B, float* C,
//                            int BATCH, int M, int N, int K)
// A: (BATCH, M, K), B: (BATCH, K, N), C: (BATCH, M, N)   —— 收缩维是 K
// 规模: B<=128, M,N,K<=1024, 性能测试 K=M=N=256
//
// 结构 = #002 的 v11 自适应设计 + batch 维度（放在 gridDim.z）。
// 两条正交的优化，都已由多形状扫描验证：
//
// 1) **tile 分档**：判据「128 tile 的 block 数（含 batch）能否 >= SM 数」。
//    小尺寸 batched 用 128 tile 会喂不满 GPU；但反过来 batch 大时 block 数就够了。
//    所以判据必须把 batch 乘进去 —— 这正是 batched 场景和单发的区别。
//
// 2) ⚠️ **边界 guard 编译期化**：逐元素的 `if (r < M)` + 标量回退分支本身就要
//    2.8 倍代价（受控实验：同一模块/同一 grid，512³ 无 guard 13 592 vs 有 guard 4 805）。
//    机理是它让编译器无法把 float4 载入当成无条件操作来调度。
//    => `template <..., bool GUARD>`，对齐形状走零代价快路径。
#include <cuda_runtime.h>

#define XBK 16
#define XTM 8
#define XTN 4

template <int XBM, int XBN, int XWM, int XWN, int XWNITER, int NTHR, bool GUARD>
__device__ __forceinline__ void bmm_body(const float* __restrict__ A, const float* __restrict__ B,
                                         float* __restrict__ C, int M, int N, int K) {
    constexpr int WARPS_N = XBN / XWN;
    constexpr int WMITER = (XWM * XWN) / (32 * XTM * XTN * XWNITER);
    constexpr int WSUBM = XWM / WMITER;
    constexpr int WSUBN = XWN / XWNITER;

    // 本 batch 的基址（batch 走 gridDim.z）
    A += (size_t)blockIdx.z * M * K;
    B += (size_t)blockIdx.z * K * N;
    C += (size_t)blockIdx.z * M * N;

    __shared__ float As[XBK * XBM];   // 转置存: As[k*XBM + row] —— 无 padding，
    __shared__ float Bs[XBK * XBN];   //  行距恰为 XBM（4 的倍数），float4 读天然对齐

    const int tid = threadIdx.x;
    const int warpIdx = tid >> 5;
    const int warpCol = warpIdx % WARPS_N;
    const int warpRow = warpIdx / WARPS_N;
    const int lane = tid & 31;
    const int threadColInWarp = lane % (WSUBN / XTN);
    const int threadRowInWarp = lane / (WSUBN / XTN);

    const int rowStart = blockIdx.y * XBM;   // C 的行
    const int colStart = blockIdx.x * XBN;   // C 的列

    const int innerRowA = tid / (XBK / 4);
    const int innerColA = tid % (XBK / 4);
    constexpr int rowStrideA = (NTHR * 4) / XBK;
    const int innerRowB = tid / (XBN / 4);
    const int innerColB = tid % (XBN / 4);
    constexpr int rowStrideB = NTHR / (XBN / 4);

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
            float4 v;
            if (!GUARD) {
                v = *reinterpret_cast<const float4*>(&A[(size_t)r * K + c]);
            } else {
                v = make_float4(0.f, 0.f, 0.f, 0.f);
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
            float4 v;
            if (!GUARD) {
                v = *reinterpret_cast<const float4*>(&B[(size_t)r * N + c]);
            } else {
                v = make_float4(0.f, 0.f, 0.f, 0.f);
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
                if (!GUARD) {
                    *reinterpret_cast<float4*>(&C[(size_t)r * N + c]) =
                        make_float4(acc[a][i][b][0], acc[a][i][b][1],
                                    acc[a][i][b][2], acc[a][i][b][3]);
                } else {
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
}

// 4 个实例化：{大,小} tile × {无 guard 快路径, 带 guard 安全路径}
__global__ void __launch_bounds__(256)
bmm_big(const float* __restrict__ A, const float* __restrict__ B,
        float* __restrict__ C, int M, int N, int K) {
    bmm_body<128, 128, 64, 32, 2, 256, false>(A, B, C, M, N, K);
}
__global__ void __launch_bounds__(256)
bmm_big_safe(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int M, int N, int K) {
    bmm_body<128, 128, 64, 32, 2, 256, true>(A, B, C, M, N, K);
}
__global__ void __launch_bounds__(32)
bmm_small(const float* __restrict__ A, const float* __restrict__ B,
          float* __restrict__ C, int M, int N, int K) {
    bmm_body<32, 32, 32, 32, 1, 32, false>(A, B, C, M, N, K);
}
__global__ void __launch_bounds__(32)
bmm_small_safe(const float* __restrict__ A, const float* __restrict__ B,
               float* __restrict__ C, int M, int N, int K) {
    bmm_body<32, 32, 32, 32, 1, 32, true>(A, B, C, M, N, K);
}

static int g_sm_count = 0;

static inline bool bmm_aligned(int M, int N, int K, int BM, int BN) {
    return (M % BM == 0) && (N % BN == 0) && (K % XBK == 0) && (N % 4 == 0) && (K % 4 == 0);
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int BATCH, int M, int N, int K) {
    if (BATCH <= 0 || M <= 0 || N <= 0 || K <= 0) return;

    if (g_sm_count == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess)
            cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (g_sm_count <= 0) g_sm_count = 1;
    }

    // ⚠️ batched 的判据必须把 batch 乘进去 —— 256³ 单发只有 4 个 block 会喂不满，
    //    但 256³ B=128 就是 512 个 block，够用了。这正是 batched 与单发的区别。
    const long long nblk_big = (long long)((M + 127) / 128) * ((N + 127) / 128)
                               * (long long)BATCH;
    if (nblk_big >= (long long)g_sm_count) {
        dim3 grd((N + 127) / 128, (M + 127) / 128, BATCH);
        if (bmm_aligned(M, N, K, 128, 128))
            bmm_big<<<grd, dim3(256)>>>(A, B, C, M, N, K);
        else
            bmm_big_safe<<<grd, dim3(256)>>>(A, B, C, M, N, K);
    } else {
        dim3 grd((N + 31) / 32, (M + 31) / 32, BATCH);
        if (bmm_aligned(M, N, K, 32, 32))
            bmm_small<<<grd, dim3(32)>>>(A, B, C, M, N, K);
        else
            bmm_small_safe<<<grd, dim3(32)>>>(A, B, C, M, N, K);
    }
}
