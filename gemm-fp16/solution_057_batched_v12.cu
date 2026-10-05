// LeetGPU #057 FP16 Batched Matrix Multiplication (medium)
// 签名: extern "C" void solve(const half* A, const half* B, half* C,
//                            int BATCH, int M, int N, int K)
// A: (BATCH, M, K), B: (BATCH, K, N), C: (BATCH, M, N)    —— 收缩维是 K
// 规模: B<=128, M,N,K<=1024, 性能测试 K=M=N=256
//
// ——— v12：tile 尺寸按「总 block 数」两档自适应 ———
//
// 多形状扫描暴露：固定的 64x64 tile 在极小规模上喂不满 170 个 SM
//   （256³ B=1 -> grid 只有 16 个 block -> 0.77x）
// ⇒ 加一档 32x32（block 数 x16）。
//
// ⚠️ 试过并**否掉**的方案：给大规模再加一档 128x128 tile —— 实测反而更差
//    （512³B32 0.56x vs t64 的 1.20x、1024³B16 0.65x vs 0.86x、非方 0.58x vs 0.73x）。
//    原因：128 tile 只能配 BK=32（共享内存吃紧），而 **BK=64 才是 #022 上学到的
//    关键收益点**（把 K 方向的 __syncthreads 次数减半，实测 +35%）。
//    tile 放大但 k 步长变小，净效果为负 ⇒ **规模大时不要放大 tile，保住 BK=64 更重要。**
//
// ⚠️ t32 的阈值要**收得很紧**：实测 256³ B=8（t64 有 128 个 block）时，
//    t32 的 512 个 block 反而更差（0.69x vs 0.83x）。只有 block 数少到 SM/4 以下才值得换。
//
// 实现要点（沿用 #022 上验证过的设计）：
//   - WMMA 16x16x16，累加器 fp32
//   - 全局→共享走 uint4（16B = 8 个 half），共享行距取 BK+8 保持 16B 对齐
//   - 累加器 fragment 的元素→(m,n) 映射不可移植，所以先 store 到共享暂存再写回
//   - #057 没有 alpha/beta（纯 C = A*B），写回不做 C_initial 读取
//   - K 方向的尾部靠零填充（GEMM 对 0 免疫），M/N 边界靠写回逐元素 guard
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// (BM, BN, BK, WARP_M, WARP_N, NT)
template <int BM, int BN, int BK, int WM, int WN, int NT>
__device__ __forceinline__ void bmm_wmma_body(const half* __restrict__ A,
                                              const half* __restrict__ B,
                                              half* __restrict__ C,
                                              int M, int N, int K) {
    constexpr int NM = WM / 16;
    constexpr int NN = WN / 16;
    constexpr int WA = BM / WM;
    constexpr int WB = BN / WN;

    constexpr int LDA = BK + 8;
    constexpr int LDB = BN + 8;
    constexpr int LDG = BN + 4;          // float 暂存行距须 %4==0

    A += (size_t)blockIdx.z * M * K;
    B += (size_t)blockIdx.z * K * N;
    C += (size_t)blockIdx.z * M * N;

    __shared__ __align__(16) half As[BM][LDA];
    __shared__ __align__(16) half Bs[BK][LDB];
    __shared__ __align__(16) float stg[BM][LDG];

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const int warp = tid >> 5;
    const int wm = warp % WA, wn = warp / WA;

    const bool vec_a = (K % 8 == 0);
    const bool vec_b = (N % 8 == 0);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NM][NN];
#pragma unroll
    for (int i = 0; i < NM; ++i)
#pragma unroll
        for (int j = 0; j < NN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
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
                As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : __float2half(0.f);
            }
        }
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
                Bs[r][c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : __float2half(0.f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[NM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[NN];
#pragma unroll
            for (int i = 0; i < NM; ++i)
                wmma::load_matrix_sync(af[i], &As[wm * WM + i * 16][kk], LDA);
#pragma unroll
            for (int j = 0; j < NN; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wn * WN + j * 16], LDB);
#pragma unroll
            for (int i = 0; i < NM; ++i)
#pragma unroll
                for (int j = 0; j < NN; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < NM; ++i)
#pragma unroll
        for (int j = 0; j < NN; ++j)
            wmma::store_matrix_sync(&stg[wm * WM + i * 16][wn * WN + j * 16],
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

__global__ void __launch_bounds__(128)
fp16_bmm_t64(const half* __restrict__ A, const half* __restrict__ B,
             half* __restrict__ C, int M, int N, int K) {
    bmm_wmma_body<64, 64, 64, 32, 32, 128>(A, B, C, M, N, K);
}
__global__ void __launch_bounds__(32)
fp16_bmm_t32(const half* __restrict__ A, const half* __restrict__ B,
             half* __restrict__ C, int M, int N, int K) {
    bmm_wmma_body<32, 32, 64, 32, 32, 32>(A, B, C, M, N, K);
}

static int g_sm_count = 0;

extern "C" void solve(const half* A, const half* B, half* C,
                      int BATCH, int M, int N, int K) {
    if (!A || !B || !C) return;
    if (BATCH <= 0 || M <= 0 || N <= 0 || K <= 0) return;

    if (g_sm_count == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess)
            cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (g_sm_count <= 0) g_sm_count = 1;
    }

    // 按「总 block 数（含 batch）/ SM 数」选 tile。阈值要收得紧（见文件头说明）。
    const long long n64 = (long long)((M + 63) / 64) * ((N + 63) / 64) * BATCH;

    if (n64 >= (long long)g_sm_count / 4) {
        fp16_bmm_t64<<<dim3((N + 63) / 64, (M + 63) / 64, BATCH), dim3(128)>>>(
            A, B, C, M, N, K);
    } else {
        fp16_bmm_t32<<<dim3((N + 31) / 32, (M + 31) / 32, BATCH), dim3(32)>>>(
            A, B, C, M, N, K);
    }
}
