// LeetGPU #105 Group Normalization (medium)  —— v2 优化版
// 签名: extern "C" void solve(const float* X, const float* gamma, const float* beta,
//                            float* Y, int N, int C, int H, int W, int G, float eps)
// X: [N,C,H,W] 行主序；通道切成 G 组，每组 C/G 个通道。
// 对每个 (n, g)：在 (C/G)*H*W 个元素上求 mean/var，归一化后再按**通道**乘 gamma/beta。
// 规模: N<=32, C<=1024, H,W<=128；性能测试 N=8, C=512, G=32, H=W=64
//
// ——— v2 相对 v1 的三处改动（都是 pass 2 里的浪费）———
//
// 1. ⚠️ **去掉了逐元素的 64 位整数除法。** v1 是
//        for (i = tid; i < cnt; i += 256) { c_local = (int)(i / HW); ... }
//    其中 `i` 和 `HW` 都是 long long ⇒ 每个元素一次**变量除法**（约 30 周期），
//    cnt=262144 时每线程要算 1024 次。改成「外层遍历通道、内层遍历 HW」后就没了。
//
// 2. **gamma/beta 提到循环外**：同一个通道内 gamma[c]/beta[c] 是常量，
//    v1 每个元素都从 global 取一次。
//
// 3. **两趟都用 float4**：标量访存换成 16B/次，指令数降到 1/4。
//
// ⚠️ 注意：这里**没能**做到 #083 那种"单遍读"。原因是 GroupNorm 一组有
//    (C/G)*H*W 个元素 —— 性能规模下是 256KB，远超 48KB 的 shared 上限，
//    放不下整组数据。所以两趟读是这一族的固有代价，
//    理论上限是 2/(2+1) = 66.7%（读两遍 + 写一遍）。
//    LayerNorm(#113) 之所以没这个问题，是因为它一行只有 C 个元素（C=512 时才 2KB，
//    第二趟天然命中缓存）。
#include <cuda_runtime.h>

#define GBLOCK 256

__global__ void groupnorm_kernel(const float* __restrict__ X,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ Y,
                                 int N, int C, int H, int W, int G, float eps) {
    const int n = blockIdx.x / G;
    const int g = blockIdx.x % G;
    if (n >= N || g >= G) return;

    const int Cg = C / G;                            // 每组的通道数
    const long long HW = (long long)H * W;
    const float* Xg = X + ((long long)n * C + (long long)g * Cg) * HW;
    float* Yg = Y + ((long long)n * C + (long long)g * Cg) * HW;
    const long long cnt = (long long)Cg * HW;
    const bool v4 = ((HW & 3) == 0);                 // 通道行能否用 float4
    const int hw4 = (int)(HW >> 2);

    // ---- 趟 1：mean / var（外层通道、内层 HW ⇒ 无除法）----
    float s = 0.0f, s2 = 0.0f;
    for (int cl = 0; cl < Cg; ++cl) {
        const float* row = Xg + (long long)cl * HW;
        if (v4) {
            const float4* r4 = reinterpret_cast<const float4*>(row);
            for (int j = threadIdx.x; j < hw4; j += GBLOCK) {
                const float4 v = r4[j];
                s += (v.x + v.y) + (v.z + v.w);
                s2 += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
            }
            for (int j = (hw4 << 2) + threadIdx.x; j < HW; j += GBLOCK) {
                const float v = row[j];
                s += v; s2 += v * v;
            }
        } else {
            for (long long j = threadIdx.x; j < HW; j += GBLOCK) {
                const float v = row[j];
                s += v; s2 += v * v;
            }
        }
    }

    // ---- 两路归约（s 与 s2）：warp shuffle 拿到 8 个 warp 的部分和，再由 0 号线程合并 ----
    __shared__ float wp[2][GBLOCK / 32 + 1];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s  += __shfl_down_sync(0xffffffffu, s,  o);
        s2 += __shfl_down_sync(0xffffffffu, s2, o);
    }
    if ((threadIdx.x & 31) == 0) {
        wp[0][threadIdx.x >> 5] = s;
        wp[1][threadIdx.x >> 5] = s2;
    }
    __shared__ float sh_mean, sh_inv;
    __syncthreads();
    if (threadIdx.x == 0) {
        float ts = 0.0f, ts2 = 0.0f;
#pragma unroll
        for (int i = 0; i < GBLOCK / 32; ++i) {
            ts  += wp[0][i];
            ts2 += wp[1][i];
        }
        const float invn = 1.0f / (float)cnt;
        const float mean = ts * invn;
        float var = ts2 * invn - mean * mean;
        if (var < 0.0f) var = 0.0f;
        sh_mean = mean;
        sh_inv = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = sh_mean, inv = sh_inv;

    // ---- 趟 2：归一化 + 仿射（gamma/beta 按通道提到循环外）----
    for (int cl = 0; cl < Cg; ++cl) {
        const int c = g * Cg + cl;
        const float gg = gamma[c], bb = beta[c];
        const float* xr = Xg + (long long)cl * HW;
        float* yr = Yg + (long long)cl * HW;
        if (v4) {
            const float4* x4 = reinterpret_cast<const float4*>(xr);
            float4* y4 = reinterpret_cast<float4*>(yr);
            for (int j = threadIdx.x; j < hw4; j += GBLOCK) {
                const float4 v = x4[j];
                y4[j] = make_float4(gg * (v.x - mean) * inv + bb,
                                    gg * (v.y - mean) * inv + bb,
                                    gg * (v.z - mean) * inv + bb,
                                    gg * (v.w - mean) * inv + bb);
            }
            for (int j = (hw4 << 2) + threadIdx.x; j < HW; j += GBLOCK)
                yr[j] = gg * (xr[j] - mean) * inv + bb;
        } else {
            for (long long j = threadIdx.x; j < HW; j += GBLOCK)
                yr[j] = gg * (xr[j] - mean) * inv + bb;
        }
    }
}

extern "C" void solve(const float* X, const float* gamma, const float* beta,
                      float* Y, int N, int C, int H, int W, int G, float eps) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || G <= 0) return;
    if (C % G != 0) return;
    groupnorm_kernel<<<N * G, GBLOCK>>>(X, gamma, beta, Y, N, C, H, W, G, eps);
}
