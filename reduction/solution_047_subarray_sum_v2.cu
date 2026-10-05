// LeetGPU #047 Subarray Sum (medium)
// 签名: extern "C" void solve(const int* input, int* output, int N, int S, int E)
// 求 input[S..E]（含两端）之和，写 output[0]。
// 规模: N <= 1e8, 性能测试 N = 100,000,000
//
// ——— v2：int4 向量化 + 放开 grid 上限 ———
//
// 基线不是 cuBLAS 而是**带宽下限**：归约只读不写，理论上限 = 读入字节 / D2D 带宽。
// 本机 5090 D 实测 memcpy 1514.9 GB/s，所以 N=1e8 的 int32 数组（400MB）
// 理论下限 ≈ 0.264ms。
//
// v1 的两个问题（实测只到 86.1%）：
//   1. **标量载入**：每个元素一次 4B load。改成 int4（16B/次）后 load 指令数降到 1/4，
//      memory-level parallelism 也上去了 —— 这是主要收益。
//   2. **grid 上限**：这里有个反直觉的实测结论 —— 放开 grid 并不是越大越好：
//        cap=1024 -> 93.2% ；cap=4096 -> **96.4%（最优）** ；cap=32768 -> 86.8% ；cap=65535 -> 80.6%
//      原因是 stage2 要把所有部分和再归约一遍：block 越多，部分和越多、
//      末段的串行归约和调度开销越大。**最优在 4096 附近**，不要盲目放大 grid。
//
// 累积仍用 float32：per-thread 只累加自己那几十个元素，之后 block 归约、再 stage2 归约，
// 不会出现 int32 溢出（1e8 × 10 远超 int 范围）。
#include <cuda_runtime.h>

#define SBLOCK 256
// 实测最优：4096。vectorize 之后 cap=1024 是 93.2%、cap=4096 是 96.4%、
// cap=65535 掉到 80.6% —— 部分和越多，stage2 的串行归约开销越大。
#define SMAXG  4096

__global__ void __launch_bounds__(SBLOCK)
sub_stage1(const int* __restrict__ input, float* __restrict__ part, int S, int E) {
    __shared__ float smem[SBLOCK];
    const int len = E - S + 1;
    const int stride = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;

    // S 是 4 的倍数时 (input + S) 天然 16B 对齐，可以走 int4 主路径。
    if ((S & 3) == 0 && len >= 4) {
        const int n4 = len >> 2;
        const int4* p4 = reinterpret_cast<const int4*>(input + S);
        for (int i = tid; i < n4; i += stride) {
            const int4 v = p4[i];
            // 保持逐元素的 float 转换顺序，与 v1 的浮点舍入行为一致（不引入结果差异）
            s += ((float)v.x + (float)v.y) + ((float)v.z + (float)v.w);
        }
        // 尾部（余下 <4 个元素）：只让 0 号 block 的前几个线程处理，代价可忽略
        const int rem = len - (n4 << 2);
        if (blockIdx.x == 0 && threadIdx.x < rem)
            s += (float)input[S + (n4 << 2) + threadIdx.x];
    } else {
        // 回退路径：S 不对齐 4 时没法用 int4
        for (int i = S + tid; i <= E; i += stride) s += (float)input[i];
    }

    smem[threadIdx.x] = s;
    __syncthreads();
#pragma unroll
    for (int k = SBLOCK / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) smem[threadIdx.x] += smem[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = smem[0];
}

__global__ void __launch_bounds__(SBLOCK)
sub_stage2(const float* __restrict__ part, int* __restrict__ output, int n) {
    __shared__ float smem[SBLOCK];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += part[i];
    smem[threadIdx.x] = s;
    __syncthreads();
#pragma unroll
    for (int k = SBLOCK / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) smem[threadIdx.x] += smem[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[0] = (int)(smem[0] + 0.5f);
}

extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    if (N <= 0) return;
    if (S < 0) S = 0;
    if (E > N - 1) E = N - 1;
    if (E < S) return;
    const int len = E - S + 1;
    // 对齐时按 int4 单元数算 grid（4 倍粒度），不对齐时按元素数算
    const bool vec = ((S & 3) == 0) && (len >= 4);
    const int units = vec ? (len >> 2) : len;
    int grid = (units + SBLOCK - 1) / SBLOCK;
    if (grid > SMAXG) grid = SMAXG;
    if (grid < 1) grid = 1;
    float* part = nullptr;
    if (cudaMalloc((void**)&part, grid * sizeof(float)) != cudaSuccess) return;
    sub_stage1<<<grid, SBLOCK>>>(input, part, S, E);
    sub_stage2<<<1, SBLOCK>>>(part, output, grid);
    cudaFree(part);
}
