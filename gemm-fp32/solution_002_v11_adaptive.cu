// LeetGPU #002 Matrix Multiplication (easy)
// 签名: extern "C" void solve(const float* A, const float* B, float* C,
//                            int M, int N, int K)
// A: M x N（行主序）, B: N x K, C: M x K        —— 收缩维是 N
// 规模: M,N,K <= 8192, 性能测试 M=8192 N=6144 K=4096
//
// ——— v11：按总 block 数自适应选 tile ———
//
// 实测教训：固定 128x128 tile 在**小规模上会崩**，因为喂不满 170 个 SM：
//     256³ -> 4 个 block / 170 SM -> 0.44x cuBLAS
//     512³ -> 16 个 block          -> 0.53x
//     1024³ -> 64 个 block         -> 0.52x
// 换成 32x32 tile 后 block 数是原来的 16 倍，小规模直接翻倍：
//     256³ -> 256 个 block -> 0.92x ；512³ -> 1024 个 -> 1.11x（反超）
// 但大尺寸反过来，小 tile 的数据复用率差：
//     4096³ -> 128 tile 55 661 GFLOPS vs 32 tile 39 477
//
// => 判据就是「**128 tile 的 block 数能否填满 SM**」，交叉点正好在 SM 数附近。
//    这也是一条通用规则：tile 尺寸必须随 `总block数 / SM数` 分档，不能写死。
#include <cuda_runtime.h>

#define XBK 16
#define XTM 8
#define XTN 4

// 用「模板 __device__ 函数 + 两个普通名字的 __global__ 包装」而不是直接模板化 kernel：
// 模板 kernel 的名字会被 mangle（_ZN13matmul_tiledILi128E...），
// 而本地验证框架是按**普通符号名**找 kernel 的，mangle 后就调不到了。
template <int XBM, int XBN, int XWM, int XWN, int XWNITER, int NTHR, bool GUARD>
__device__ __forceinline__ void gemm_body(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    constexpr int WARPS_N = XBN / XWN;
    constexpr int WMITER = (XWM * XWN) / (32 * XTM * XTN * XWNITER);
    constexpr int WSUBM = XWM / WMITER;
    constexpr int WSUBN = XWN / XWNITER;

    __shared__ float As[XBK * XBM];   // 转置存: As[k*XBM + row] —— 无 padding，
    __shared__ float Bs[XBK * XBN];   //  行距恰为 XBM（4 的倍数），float4 读天然 16B 对齐

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

    // ⚠️ GUARD 必须是**编译期**参数：实测逐元素的 `if (r < M)` + 标量回退分支，
    // 在 32 线程的小 block 上会把载入路径拖慢 **2.8 倍**
    //   （同一模块/同一 grid：无 guard 13 592 GFLOPS vs 有 guard 4 805，512³）
    // 所以形状对齐时走 GUARD=false 的无分支快路径，只在非对齐时付 guard 的代价。
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

    // 收缩维是 N（A 是 M×N、B 是 N×K）—— 写成 K 会让 N>K 的形状漏掉一段收缩，
    // 而 N==K / N<K 会「碰巧」正确，这类 bug 极难被小用例发现。
    for (int t = 0; t < N; t += XBK) {
#pragma unroll
        for (int off = 0; off + rowStrideA <= XBM; off += rowStrideA) {
            const int r = rowStart + innerRowA + off;
            const int c = t + innerColA * 4;
            float4 v;
            if (!GUARD) {
                v = *reinterpret_cast<const float4*>(&A[(size_t)r * N + c]);
            } else {
                v = make_float4(0.f, 0.f, 0.f, 0.f);
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
                v = *reinterpret_cast<const float4*>(&B[(size_t)r * K + c]);
            } else {
                v = make_float4(0.f, 0.f, 0.f, 0.f);
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
                    *reinterpret_cast<float4*>(&C[(size_t)r * K + c]) =
                        make_float4(acc[a][i][b][0], acc[a][i][b][1],
                                    acc[a][i][b][2], acc[a][i][b][3]);
                } else {
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
}

// 四个实例化：{大,小} tile × {无 guard 快路径, 带 guard 安全路径}
// 形状对齐（M/K 是 tile 倍数、N 是 BK 倍数、行距 %4）时用 GUARD=false —— 实测快 2.8 倍。
__global__ void __launch_bounds__(256)
matmul_big(const float* __restrict__ A, const float* __restrict__ B,
           float* __restrict__ C, int M, int N, int K) {
    gemm_body<128, 128, 64, 32, 2, 256, false>(A, B, C, M, N, K);
}
__global__ void __launch_bounds__(256)
matmul_big_safe(const float* __restrict__ A, const float* __restrict__ B,
                float* __restrict__ C, int M, int N, int K) {
    gemm_body<128, 128, 64, 32, 2, 256, true>(A, B, C, M, N, K);
}
__global__ void __launch_bounds__(32)
matmul_small(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int M, int N, int K) {
    gemm_body<32, 32, 32, 32, 1, 32, false>(A, B, C, M, N, K);
}
__global__ void __launch_bounds__(32)
matmul_small_safe(const float* __restrict__ A, const float* __restrict__ B,
                  float* __restrict__ C, int M, int N, int K) {
    gemm_body<32, 32, 32, 32, 1, 32, true>(A, B, C, M, N, K);
}

static int g_sm_count = 0;

// 无 guard 快路径的适用条件：所有 tile 都不会跨越边界，且 float4 行距对齐。
static inline bool tile_aligned(int M, int N, int K, int BM, int BN) {
    return (M % BM == 0) && (K % BN == 0) && (N % XBK == 0) && (K % 4 == 0);
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    if (g_sm_count == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess)
            cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (g_sm_count <= 0) g_sm_count = 1;
    }

    // 判据 1：128 tile 的 block 数能否填满所有 SM。能 -> 大 tile（数据复用率高）；
    //         不能 -> 小 tile（block 数 ×16）。交叉点实测就在 SM 数附近。
    const long long nblk_big =
        (long long)((M + 127) / 128) * ((K + 127) / 128);
    // ⚠️ 窄长输出（K/M >= 4）仍是**未解决的短板**（实测 0.51x）。
    //    试过加一档 32x128 的非方形 tile（理论上把共享载入/计算比从 0.125 降到 0.039），
    //    但**结果算错了**（M256N256K256 上 diff=50 而期望 0.13），没能在预算内定位，
    //    于是回退 —— 发布一个算错的 kernel 比不做更糟。
    //    下一步应该走 **split-K**（沿收缩维切分再加归约），而不是指望换 tile 形状。

    if (nblk_big >= (long long)g_sm_count) {
        if (tile_aligned(M, N, K, 128, 128)) {
            matmul_big<<<dim3((K + 127) / 128, (M + 127) / 128), dim3(256)>>>(A, B, C, M, N, K);
        } else {
            matmul_big_safe<<<dim3((K + 127) / 128, (M + 127) / 128), dim3(256)>>>(A, B, C, M, N, K);
        }
    } else {
        // 判据 2：形状对齐与否 -> 无 guard 快路径 / 带 guard 安全路径
        if (tile_aligned(M, N, K, 32, 32)) {
            matmul_small<<<dim3((K + 31) / 32, (M + 31) / 32), dim3(32)>>>(A, B, C, M, N, K);
        } else {
            matmul_small_safe<<<dim3((K + 31) / 32, (M + 31) / 32), dim3(32)>>>(A, B, C, M, N, K);
        }
    }
}
