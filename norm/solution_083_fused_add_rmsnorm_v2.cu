// LeetGPU #083 Fused Residual Add + RMS Norm (medium)  —— v2 优化版
// 签名: extern "C" void solve(const float* x, const float* residual, const float* weight,
//                            float* out, int N, int C, float eps)
// 逐行: z = x + residual ; out = weight * z / sqrt(mean(z^2) + eps)
// 规模: N,C <= 65536；性能测试 N=4096, C=4096
//
// ——— v2 相对 v1 的两处改动 ———
//
// 这个算子是**纯带宽型**：读 x、读 residual、写 out = 3 个 NC 字节。
// v1 的实现有两处浪费：
//
// 1. **标量访存**。每次只搬 4B，load 指令数是 float4 的 4 倍，
//    memory-level parallelism 也差。改成 float4 后主路径指令数降到 1/4。
//
// 2. ⭐ **读了两遍 global**。v1 第一遍读 x/res 求平方和，第二遍**再读一遍** x/res
//    才算输出 —— 等于 2N C 次读，理论上限被压到 3/(2+2+1) = 60%。
//    v2 把 `z = x + residual` **缓存进 shared**，第二遍直接从 shared 读 z，
//    global 只读一遍 ⇒ 回到 3 个 NC 字节的上限。
//
// ⚠️ 缓存的代价是 shared 要放 C 个 float。C ≤ SCACHE_MAX(=24576, 96KB) 时可行；
// 超过就走 v1 的两遍路径（保留作为回退），否则一行的数据根本放不下。
#include <cuda_runtime.h>

#define FBLOCK 256
// ⚠️ 动态 shared 默认上限是 **48KB**，超过就必须先
//    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, N)，
//    而本地的验证框架（Driver API 直调）不会替我们设这个属性 ——
//    所以这里把阈值压到 48KB 以内（(11000+4)*4 = 44016B），保证本地和平台都能跑。
//    想覆盖更大的 C 就把阈值抬到 24000 并在 solve() 里加上 cudaFuncSetAttribute。
#define SCACHE_MAX 11000

// ---------------- 快路径：float4 + shared 缓存 z，global 只读一遍 ----------------
__global__ void fused_add_rmsnorm_cached(const float* __restrict__ x,
                                         const float* __restrict__ res,
                                         const float* __restrict__ w,
                                         float* __restrict__ out,
                                         int N, int C, float eps) {
    extern __shared__ float zsh[];   // [0, C) 存 z；[C, C+8) 归约暂存
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + (size_t)row * C;
    const float* rr = res + (size_t)row * C;
    float* orow = out + (size_t)row * C;

    const int t = threadIdx.x;
    const int n4 = C >> 2;                      // float4 单元数

    // ---- 第一遍：算 z、存 shared、累平方和（global 只读这一遍）----
    float s = 0.0f;
    if ((C & 3) == 0) {
        for (int j = t; j < n4; j += FBLOCK) {
            const float4 a = reinterpret_cast<const float4*>(xr)[j];
            const float4 b = reinterpret_cast<const float4*>(rr)[j];
            const float4 z = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            reinterpret_cast<float4*>(zsh)[j] = z;
            s += (z.x * z.x + z.y * z.y) + (z.z * z.z + z.w * z.w);
        }
        for (int j = (n4 << 2) + t; j < C; j += FBLOCK) {   // 尾部 <4 个
            const float z = xr[j] + rr[j];
            zsh[j] = z;
            s += z * z;
        }
    } else {
        for (int j = t; j < C; j += FBLOCK) {
            const float z = xr[j] + rr[j];
            zsh[j] = z;
            s += z * z;
        }
    }

    // ---- block 归约（warp shuffle + 8 个 warp 的部分和）----
    __shared__ float wpart[FBLOCK / 32 + 1];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((t & 31) == 0) wpart[t >> 5] = s;
    __syncthreads();
    if (t == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int i = 0; i < FBLOCK / 32; ++i) tot += wpart[i];
        wpart[FBLOCK / 32] = rsqrtf(tot / (float)C + eps);
    }
    __syncthreads();
    const float inv = wpart[FBLOCK / 32];

    // ---- 第二遍：从 shared 读 z（不再碰 global），写出 ----
    if ((C & 3) == 0) {
        for (int j = t; j < n4; j += FBLOCK) {
            const float4 z = reinterpret_cast<const float4*>(zsh)[j];
            const float4 wv = reinterpret_cast<const float4*>(w)[j];
            reinterpret_cast<float4*>(orow)[j] = make_float4(
                wv.x * z.x * inv, wv.y * z.y * inv,
                wv.z * z.z * inv, wv.w * z.w * inv);
        }
        for (int j = (n4 << 2) + t; j < C; j += FBLOCK)
            orow[j] = w[j] * zsh[j] * inv;
    } else {
        for (int j = t; j < C; j += FBLOCK)
            orow[j] = w[j] * zsh[j] * inv;
    }
}

// ---------------- 回退路径：C 太大，shared 放不下一整行 ----------------
// 保持 v1 的两遍结构（第二遍重新读 global），但加上 float4 向量化。
__global__ void fused_add_rmsnorm_big(const float* __restrict__ x,
                                      const float* __restrict__ res,
                                      const float* __restrict__ w,
                                      float* __restrict__ out,
                                      int N, int C, float eps) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + (size_t)row * C;
    const float* rr = res + (size_t)row * C;
    float* orow = out + (size_t)row * C;
    const int t = threadIdx.x;
    const int n4 = C >> 2;

    __shared__ float wpart[FBLOCK / 32 + 1];
    float s = 0.0f;
    if ((C & 3) == 0) {
        for (int j = t; j < n4; j += FBLOCK) {
            const float4 a = reinterpret_cast<const float4*>(xr)[j];
            const float4 b = reinterpret_cast<const float4*>(rr)[j];
            const float zx = a.x + b.x, zy = a.y + b.y, zz = a.z + b.z, zw = a.w + b.w;
            s += (zx * zx + zy * zy) + (zz * zz + zw * zw);
        }
        for (int j = (n4 << 2) + t; j < C; j += FBLOCK) {
            const float z = xr[j] + rr[j];
            s += z * z;
        }
    } else {
        for (int j = t; j < C; j += FBLOCK) {
            const float z = xr[j] + rr[j];
            s += z * z;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((t & 31) == 0) wpart[t >> 5] = s;
    __syncthreads();
    if (t == 0) {
        float tot = 0.0f;
#pragma unroll
        for (int i = 0; i < FBLOCK / 32; ++i) tot += wpart[i];
        wpart[FBLOCK / 32] = rsqrtf(tot / (float)C + eps);
    }
    __syncthreads();
    const float inv = wpart[FBLOCK / 32];

    if ((C & 3) == 0) {
        for (int j = t; j < n4; j += FBLOCK) {
            const float4 a = reinterpret_cast<const float4*>(xr)[j];
            const float4 b = reinterpret_cast<const float4*>(rr)[j];
            const float4 wv = reinterpret_cast<const float4*>(w)[j];
            reinterpret_cast<float4*>(orow)[j] = make_float4(
                wv.x * (a.x + b.x) * inv, wv.y * (a.y + b.y) * inv,
                wv.z * (a.z + b.z) * inv, wv.w * (a.w + b.w) * inv);
        }
        for (int j = (n4 << 2) + t; j < C; j += FBLOCK)
            orow[j] = w[j] * (xr[j] + rr[j]) * inv;
    } else {
        for (int j = t; j < C; j += FBLOCK)
            orow[j] = w[j] * (xr[j] + rr[j]) * inv;
    }
}

extern "C" void solve(const float* x, const float* residual, const float* weight,
                      float* out, int N, int C, float eps) {
    if (N <= 0 || C <= 0) return;
    if (C <= SCACHE_MAX) {
        // 需要 C 个 float 存 z；行首 16B 对齐时 float4 路径成立
        const size_t sh = ((size_t)C + 4) * sizeof(float);
        fused_add_rmsnorm_cached<<<N, FBLOCK, sh>>>(x, residual, weight, out, N, C, eps);
    } else {
        fused_add_rmsnorm_big<<<N, FBLOCK>>>(x, residual, weight, out, N, C, eps);
    }
}
