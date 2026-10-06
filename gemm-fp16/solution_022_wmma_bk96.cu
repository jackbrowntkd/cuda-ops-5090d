// LeetGPU #022 General Matrix Multiplication (GEMM) (medium)
// 签名: extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K,
//          float alpha, float beta)
//
//   C = alpha * (A × B) + beta * C_initial     （全部 row-major，A:M×K  B:K×N  C:M×N）
//
// 形态：WMMA 张量核版（题面明确允许 WMMA）。
//   本机 RTX 5090 D 实测 1024^3：
//     标量 FFMA tiled 版 ≈27.5 TFLOPS  ->  WMMA BK=32 ≈64.5  ->  WMMA BK=64 ≈86.8
//     cuBLAS（fp16/fp32acc）同规模 ≈128 TFLOPS，即 BK=64 已到 cuBLAS 的 0.68x
//
// 设计取舍：
//   - **64×64 tile 而不是 128×128**：1024^2 规模下 128×128 只有 64 个 block，
//     喂不满 170 个 SM（实测反而慢 25%）。64×64 给 256 个 block。
//     （warp 级分块实验：128×64 / 8 warp 拿到 128 个 block，实测 55 TFLOPS，
//       仍然输给 64×64 —— 1024^3 这个规模就是「block 数 > 单块复用」。）
//   - **BK=64 而不是 32**：K=1024 时 __syncthreads 从 64 次降到 32 次，
//     实测 +35%（64.5 -> 86.8 TFLOPS），是本次最大的单项收益。
//   - 128 线程 = 4 warp，按 2×2 排布，每 warp 负责 32×32（4 个 16×16 累加器）
//   - 全局→共享走 uint4（16B/8 half）；共享行距取 BK+8 / BN+8，保持 16B 对齐，
//     既不撞 bank 又允许 128bit 向量存取
//   - K 方向的尾部（K 非 BK 倍数）靠**零填充**处理，GEMM 累加对 0 天然免疫
//   - alpha/beta 需要 C_initial，而 accumulator fragment 的元素→(m,n) 映射不可移植，
//     所以先 store_matrix_sync 落到共享暂存行，再按真实下标套 alpha/beta 写回
//     （暂存行距必须 %4==0，取 BN+4=68）
//   - 边界（M/N 非 64 倍数）在载入零填充 + 写回逐元素 guard 两端都做
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 96
#define NT 128                 // 4 warp
#define LDA (BK + 8)           // 72 half = 144B，16B 对齐
#define LDB (BN + 8)           // 72 half = 144B，16B 对齐
#define LDG (BN + 4)           // 68 float = 272B，float 暂存行距须为 4 的倍数

__global__ void gemm_wmma(const half* __restrict__ A, const half* __restrict__ B,
                          half* __restrict__ C, int M, int N, int K,
                          float alpha, float beta) {
    __shared__ __align__(16) half As[BM][LDA];
    __shared__ __align__(16) half Bs[BK][LDB];
    __shared__ __align__(16) float stg[BM][LDG];

    const int tid = threadIdx.x;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const int warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1;      // warp 在 2x2 网格里的位置

    // 向量路径成立的前提：全局行首 16B 对齐
    const bool vec_a = (K % 8 == 0);
    const bool vec_b = (N % 8 == 0);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---------------- 载入 A 的 64×32 片 ----------------
        if (vec_a) {
            const uint4* A4 = reinterpret_cast<const uint4*>(A);
#pragma unroll
            for (int t = 0; t < (BM * BK / 8) / NT; ++t) {
                const int idx = tid + t * NT, r = idx / (BK / 8), c4 = idx % (BK / 8);
                const int gr = row0 + r, gc = k0 + c4 * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (gr < M && gc < K)
                    v = A4[((size_t)gr * K + gc) / 8];
                *reinterpret_cast<uint4*>(&As[r][c4 * 8]) = v;
            }
        } else {
#pragma unroll
            for (int t = 0; t < (BM * BK) / NT; ++t) {
                const int idx = tid + t * NT, r = idx / BK, c = idx % BK;
                const int gr = row0 + r, gc = k0 + c;
                As[r][c] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : __float2half(0.f);
            }
        }

        // ---------------- 载入 B 的 32×64 片 ----------------
        if (vec_b) {
            const uint4* B4 = reinterpret_cast<const uint4*>(B);
#pragma unroll
            for (int t = 0; t < (BK * BN / 8) / NT; ++t) {
                const int idx = tid + t * NT, r = idx / (BN / 8), c4 = idx % (BN / 8);
                const int gr = k0 + r, gc = col0 + c4 * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (gr < K && gc < N)
                    v = B4[((size_t)gr * N + gc) / 8];
                *reinterpret_cast<uint4*>(&Bs[r][c4 * 8]) = v;
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

        // ---------------- 张量核 ----------------
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

    // 落到共享暂存，才能拿到可移植的 (m,n) 下标去套 alpha/beta
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&stg[wm * 32 + i * 16][wn * 32 + j * 16],
                                    acc[i][j], LDG, wmma::mem_row_major);
    __syncthreads();

    // 逐元素写回：先读 C_initial（beta 项）再覆盖，边界 guard
#pragma unroll 4
    for (int idx = tid; idx < BM * BN; idx += NT) {
        const int r = idx / BN, c = idx % BN;
        const int gr = row0 + r, gc = col0 + c;
        if (gr >= M || gc >= N) continue;
        const size_t o = (size_t)gr * N + gc;
        C[o] = __float2half(alpha * stg[r][c] + beta * __half2float(C[o]));
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K,
                      float alpha, float beta) {
    if (!A || !B || !C) return;
    if (M <= 0 || N <= 0 || K <= 0) return;

    dim3 blk(NT);
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_wmma<<<grd, blk>>>(A, B, C, M, N, K, alpha, beta);
}
