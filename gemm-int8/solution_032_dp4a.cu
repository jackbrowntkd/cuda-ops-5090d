// LeetGPU #032 INT8 Quantized MatMul
// 签名: extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N,
//          int K, float scale_A, float scale_B, float scale_C,
//          int zero_point_A, int zero_point_B, int zero_point_C)
//
//   C(i,j) = clamp( round( (Σ_k (A_ik - zA)(B_kj - zB) * sA * sB) / sC ) + zC, -128, 127 )
//
// ——— 关键难点 ———
// 朴素实现（载入 shared 时就减 zero_point）在内层是 int32 标量乘加，慢。
// 但**不能**直接把 (A-zA) 喂给 int8 张量核 / __dp4a —— (A-zA) ∈ [-255,255]，
// 超出 int8 范围。必须用代数恒等式把零点校正挪到 epilogue：
//
//   Σ_k (A-zA)(B-zB) = Σ_k AB  -  zA·Σ_k B  -  zB·Σ_k A  +  K·zA·zB
//                      ^^^^^^     ^^^^^^^     ^^^^^^^     ^^^^^^^^
//                      纯 int8      B 的列和     A 的行和     常数
//
// 全程整数运算，**结果与原式逐位相同**（不是近似），所以可以放心用 int8 硬件路径。
// 代价是要先算 A 的行和、B 的列和（各一趟 O(MK)/O(KN) 的廉价 kernel）。
//
// ——— 本版实现 ———
//   __dp4a（4 路 int8 MAC，sm_61+） + 寄存器分块 64x64 块 / 每线程 4x4。
//   A 按行存 As[row][k]、B **转置**存 BsT[col][k] —— 这样 4 个连续 k 正好是一个 int32，
//   dp4a 可以直接吃。共享行距取 BK+4（4 的倍数保证 int32 对齐，且 9 words 步长
//   在 32 bank 上互质 → 无冲突）。
//
// ⚠️ 题面样例 1 与公式自相矛盾（见原实现注释），此处以题面公式为准。
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 32
#define NT 256
#define APAD 4                  // BK + APAD = 36：4 的倍数 + 9-word 步长
#define TM 4
#define TN 4

// ---------------- 预计算：A 的行和 ----------------
__global__ void i8_rowsum(const signed char* __restrict__ A, int* __restrict__ sumA,
                          int M, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const signed char* p = A + (size_t)row * K;
    int s = 0;
    const bool k4 = (K % 4 == 0);
    if (k4) {
        const int n4 = K / 4;
        for (int i = 0; i < n4; ++i) {
            const int v = reinterpret_cast<const int*>(p)[i];
            s += (signed char)(v & 0xFF) + (signed char)((v >> 8) & 0xFF)
               + (signed char)((v >> 16) & 0xFF) + (signed char)((v >> 24) & 0xFF);
        }
    } else {
        for (int k = 0; k < K; ++k) s += (int)p[k];
    }
    sumA[row] = s;
}

// ---------------- 预计算：B 的列和 ----------------
// 每 block 负责一组列，块内沿 K 归约
__global__ void i8_colsum(const signed char* __restrict__ B, int* __restrict__ sumB,
                          int K, int N) {
    const int col = blockIdx.x;
    if (col >= N) return;
    int s = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        s += (int)B[(size_t)k * N + col];
    // block 内归约
    __shared__ int sm[256];
    sm[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) sm[threadIdx.x] += sm[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) sumB[col] = sm[0];
}

// ---------------- 主 GEMM：dp4a + 寄存器分块 ----------------
__global__ void __launch_bounds__(NT)
i8_gemm_dp4a(const signed char* __restrict__ A, const signed char* __restrict__ B,
             signed char* __restrict__ C, const int* __restrict__ sumA,
             const int* __restrict__ sumB, int M, int N, int K,
             float sA, float sB, float sC, int zA, int zB, int zC) {
    __shared__ signed char As[BM][BK + APAD];    // As[row][k]
    __shared__ signed char BsT[BN][BK + APAD];   // BsT[col][k]  —— 转置存，方便 dp4a

    const int tid = threadIdx.x;                 // 0..255
    const int tx = tid & 15, ty = tid >> 4;      // 16x16 线程网格
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    int acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;

    const bool a4 = (K % 4 == 0);
    const int nk = (K + BK - 1) / BK;

    for (int kb = 0; kb < nk; ++kb) {
        const int k0 = kb * BK;
        // ---- A 片：BM x BK，每次 4 个 int8 ----
#pragma unroll
        for (int t = 0; t < (BM * BK / 4) / NT; ++t) {
            const int idx = tid + t * NT;
            const int r = idx / (BK / 4), c4 = idx % (BK / 4);
            const int gr = row0 + r, gc = k0 + c4 * 4;
            int v = 0;
            if (gr < M) {
                if (a4 && gc + 4 <= K) {
                    v = *reinterpret_cast<const int*>(&A[(size_t)gr * K + gc]);
                } else {
                    unsigned b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                    if (gc + 0 < K) b0 = (unsigned char)A[(size_t)gr * K + gc + 0];
                    if (gc + 1 < K) b1 = (unsigned char)A[(size_t)gr * K + gc + 1];
                    if (gc + 2 < K) b2 = (unsigned char)A[(size_t)gr * K + gc + 2];
                    if (gc + 3 < K) b3 = (unsigned char)A[(size_t)gr * K + gc + 3];
                    v = (int)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
                }
            }
            *reinterpret_cast<int*>(&As[r][c4 * 4]) = v;
        }
        // ---- B 片：BK x BN，**转置**存进 BsT[col][k] ----
#pragma unroll
        for (int t = 0; t < (BK * BN) / NT; ++t) {
            const int idx = tid + t * NT;
            const int k = idx / BN, c = idx % BN;
            const int gk = k0 + k, gc = col0 + c;
            signed char v = 0;
            if (gk < K && gc < N) v = B[(size_t)gk * N + gc];
            BsT[c][k] = v;
        }
        __syncthreads();

        // ---- dp4a：k 每次推进 4 ----
#pragma unroll
        for (int k = 0; k < BK; k += 4) {
            int ap[TM], bp[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                ap[i] = *reinterpret_cast<const int*>(&As[ty * TM + i][k]);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                bp[j] = *reinterpret_cast<const int*>(&BsT[tx * TN + j][k]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = __dp4a(ap[i], bp[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- epilogue：代数校正 + 缩放 + 舍入 + clamp ----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + ty * TM + i;
        if (gr >= M) continue;
        const int sa = sumA[gr];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + tx * TN + j;
            if (gc >= N) continue;
            // Σ(A-zA)(B-zB) = ΣAB - zA·ΣB - zB·ΣA + K·zA·zB   （纯整数，逐位精确）
            const long long corr = (long long)acc[i][j]
                                 - (long long)zA * (long long)sumB[gc]
                                 - (long long)zB * (long long)sa
                                 + (long long)K * (long long)zA * (long long)zB;
            const double v = (double)corr * (double)sA * (double)sB / (double)sC;
            long o = (long)rint(v) + (long)zC;
            if (o < -128) o = -128;
            if (o > 127) o = 127;
            C[(size_t)gr * N + gc] = (signed char)o;
        }
    }
}

extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    if (!A || !B || !C) return;
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (scale_C == 0.0f) return;

    const signed char* pA = (const signed char*)A;
    const signed char* pB = (const signed char*)B;
    signed char* pC = (signed char*)C;

    // ⚠️ 行和/列和**总是**要算并传进去：即使 zA=zB=0（恒等式退化成 raw），
    // epilogue 里仍会读这两个指针 —— 传 nullptr 会直接段错误。
    // 代价是 O(MK + KN) 的额外读，相对 GEMM 的 O(MNK) 可忽略。
    int *dSumA = nullptr, *dSumB = nullptr;
    if (cudaMalloc(&dSumA, sizeof(int) * (size_t)M) != cudaSuccess) return;
    if (cudaMalloc(&dSumB, sizeof(int) * (size_t)N) != cudaSuccess) {
        cudaFree(dSumA);
        return;
    }
    const int thr = 256;
    i8_rowsum<<<(M + thr - 1) / thr, thr>>>(pA, dSumA, M, K);
    i8_colsum<<<N, thr>>>(pB, dSumB, K, N);

    dim3 blk(NT);
    dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM);
    i8_gemm_dp4a<<<grd, blk>>>(pA, pB, pC, dSumA, dSumB,
                               M, N, K, scale_A, scale_B, scale_C,
                               zero_point_A, zero_point_B, zero_point_C);

    cudaFree(dSumA);
    cudaFree(dSumB);
}
