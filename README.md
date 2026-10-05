# 5090D CUDA 算子优化系列

在 **RTX 5090 D（sm_120，170 SM）** 上，用「厂商库基线 → 差距量化 → 抄开源实现的结构 →
参数扫描与归因 → 平台榜单」的闭环，逐个算子做性能对标与优化。

不是"跑通就跑"的 kernel 集合，而是**每一步都有可量的数字、可复现的命令、以及明确标注的
"哪些结论依赖机器规模"**。

---

## 一、先立两个天花板

优化之前必须先知道天花板在哪，否则不知道还差多少。

| 天花板 | 本机实测 | 对标哪类算子 |
|---|---|---|
| **D2D 显存带宽** | **1 514.9 GB/s** | 纯 elementwise / 数据搬运（= 理论上限） |
| **cuBLAS fp32 FFMA** | ~63~68 TFLOPS | 计算密集型（fp32） |

> ⚠️ **「5090 的 fp32 = 116 TFLOPS」是 TF32 张量核，不是真 FFMA。**
> 真 fp32（PyTorch 默认不开 tf32 实测 66.4 / 本机 `CUBLAS_COMPUTE_32F` 实测 68）≈ **66~68 TFLOPS**。
> 这个区分很关键 —— 报"和 cuBLAS 的比值"时不说清路径，数字就是错的。
>
> 本机带宽实测与第三方 [`binbinsh/gpu-bench`](https://github.com/binbinsh/gpu-bench) 的
> 1 528.8 GB/s 只差 **0.9%**，交叉验证通过。

---

## 二、差距总表（**本仓库的核心产出**）

规模全部取自各题题面写明的**性能测试规模**，不是自己另选。
下表的"优化后"是 2026-10-05 用本仓库的方法论做的一轮实战结果。

| 算子 | 厂商库 | 优化前 | 优化后 | 提升 | **差距**（题面规模） |
|---|---|---|---|---|---|
| fp16 GEMM（1024³） | 212 572 | 27 300 | 159 657 | 5.8× | 0.75× |
| fp32 GEMM（8192×6144×4096） | 61 222 | 5 876 | **44 146** | **7.5×** | 0.09× → **0.72×** |
| fp32 batched（256³×128） | 39 924 | 5 929 | **46 176** | **7.8×** | 0.15× → **1.16×** |
| fp16 batched（256³×128） | 98 030 | 6 242 | **115 367** | **18.5×** | 0.07× → **1.18×** |
| int8 MatMul（8192×4096×2048） | 210 704 ⚠️ | 6 463 | **86 678** | **13.4×** | 0.03× → **0.41×** |

单位均为 GFLOPS（fp16 与 int8 为各自 dtype 的吞吐）。

⚠️ **int8 那行不是严格可比**：cuBLAS 侧是纯 int8 GEMM（int32 累加），**不含
zero_point / scale 校正**，只能当"忽略零点校正的理论上界"。
我们的 `solution_032_dp4a.cu` 是**完整**实现（含零点校正），所以 0.41× 是偏保守的读数。

### ⚠️ 但是：单形状的结论站不住 —— 扫 10 个形状后要修正

上表的数字都是**题面那一个规模**上的。换个形状结论会变 —— 所以本仓库加了多形状扫描
（[`common/sweep_shapes.py`](common/sweep_shapes.py)），覆盖"单个小矩阵 → 多个大矩阵"整个区间：

| 家族 | 题面那一点 | **10 个形状的几何平均** | 赢的形状数 | 最差 / 最好 |
|---|---|---|---|---|
| fp32 batched | 1.16× | **0.874×** | 5 / 10 | 0.40× / 1.22× |
| fp16 batched | 1.18× | **0.907×** | 3 / 10 | 0.73× / 1.20× |
| fp32 单发 GEMM | 0.72× | **0.600×** | **0 / 7** | 0.37× / 0.90× |

> **"两题反超 cuBLAS"这个说法只在题面规模上成立，不能当结论用。**
> 站得住的表述是：**在 batch 足够大（总 block 数 ≥ ~4×SM）或矩阵小到厂商库
> 启发式失准时能赢；一旦回到小 batch / 中等规模，就掉到 0.4~0.7×。**

### 四条结论

1. **fp16 GEMM 达到厂商库的 0.75×（题面规模），fp32 GEMM 0.72%。** 完整阶梯见
   [`docs/GEMM-PERF.md`](docs/GEMM-PERF.md)：从 naive 6 950 GFLOPS 一路到 v10d 的
   57 700（4096³），每一步的增益都是分开量的。
2. ⭐ **最差的形状暴露了真正的瓶颈：tile 尺寸没有随总 block 数自适应。**

   | 形状 | 128×128 tile 的 block 数 | 比值 |
   |---|---|---|
   | 256³ B=1 | **4 个** / 170 SM | **0.40×** |
   | 256³ B=128 | 512 个 | 1.20× |
   | 512³ 单发 | **16 个** | **0.43×** |
   | 1024³ 单发 | **64 个** | **0.37×** |
   | 4096³ 单发 | 1024 个 | 0.74× |

   **固定的 tile 只在某一个规模区间最优。** 按 `总 block 数 / SM 数` 分档派发不同
   tile（小规模换小 tile）是当前提升空间最大的方向。
3. ⭐ **小尺寸 batched 确实是厂商库的弱区**：256³×128 时 cuBLAS 只有 39.9（fp32）/
   98.0（fp16）TFLOPS，而同类的**大尺寸单发**是 61.2 / 212.6 TFLOPS
   （fp16 只保留了 **46%**）。但要注意 —— **这个窗口比单形状结论暗示的窄得多**。
4. **int8 用 `__dp4a` 拿到 13.4× 提升**。关键在于那个代数恒等式 ——
   `(A-zA)` 可达 ±255、**塞不进 int8 硬件路径**，必须把零点校正挪到 epilogue：

   ```
   Σ(A-zA)(B-zB) = ΣAB - zA·ΣB - zB·ΣA + K·zA·zB
                   ^^^^   ^^^^^   ^^^^^   ^^^^^^^^
                   纯int8   列和    行和     常数
   ```

   全程整数运算，**结果与原式逐位相同**（不是近似）。


---

## 三、目录

```
common/                     # 复用的基础设施（全部自己写的）
  localrun.py               #   NVRTC + Driver API 的 kernel 跑测框架（绕开 nvcc 不可用）
  cublas.py                 #   cuBLAS 基线封装（ctypes；含列主序映射 + GEMM 家族各接口）
  vendor_baseline.py        #   量 D2D 带宽上限 + GEMM 家族厂商基线
  bench_ours_vs_vendor.py   #   逐题「我们 vs 厂商库」，含各题 kernel 分派的显式声明
  sweep_shapes.py           #   ⭐ 多形状扫描（几何平均 + 最差/最好 + 胜负计数）
  verify_algo_choice.py     #   ⭐ 扫 cuBLAS 的 algo 选项，排除"我们调用方式不对"
  bench_elementwise.py      #   ⭐ 纯带宽型算子：相对 memcpy 上限的达标率（含 L2/host 两个可信度判据）

gemm-fp32/
  ladder_fp32.py            #   fp32 SGEMM 十一级阶梯（tiling/reg/float4/warp 逐项拆解）

  solution_002_v10.cu       #   #002 最终解答：v10d 设计（0.09x -> 0.72x）
  solution_030_batched_v10.cu  # #030 fp32 batched（0.15x -> 1.16x，反超 cuBLAS）
gemm-fp16/
  ladder_fp16.py            #   fp16 GEMM 十二个变体（含 WMMA / warp 分块 / cuBLAS 对照）
  solution_wmma_bk64.cu     #   #022 解答（WMMA BK=64），平台已通过
  solution_057_batched_wmma.cu # #057 fp16 batched（0.07x -> 1.18x，反超 cuBLAS）
gemm-int8/
  solution_032_dp4a.cu      #   #032 int8 量化 MatMul：__dp4a + 零点校正恒等式（0.03x -> 0.41x）

docs/
  BASELINE.md               # 基线对标报告：天花板、差距总表、全部算子按类型的基线分类
  GEMM-PERF.md              # GEMM 完整性能报告（含反例与归因分析）
```

---

## 四、方法论（六步闭环）

1. **确认架构**：sm_120 走的是 Ampere 那代的 `mma.sync`（warp 级、寄存器到寄存器），
   **不是** SM100 的 `tcgen05`、**也不是** SM90 的 `wgmma`。
   → DeepGEMM / CUTLASS SM100 collective / WGMMA 版 flash-attention **在 5090 上编不过或崩**。
   能用的是 WMMA、`mma.sync` PTX、`cp.async`。
2. **量厂商基线**：每个算子都有基线。计算密集 → cuBLAS/cuDNN/CUB；
   **纯 elementwise → D2D 带宽即上限**。
3. **量化差距**：见下面的测量纪律。
4. **抄参考的结构**：⚠️ **抄结构，别抄参数** —— 把别处调好的 tile/线程数直接搬过来会翻车
   （见 `docs/GEMM-PERF.md` 里的 v10 vs v10d）。
5. **参数扫描 + 归因**：一次动一两个维度、每格重复 3 次取中位数、
   **把"可归因"和"不可归因"分开写**。
6. **提交与榜单**：榜单每张卡只返回前 3 名；`isPublic=true` 只是必要条件，进前 3 才上榜。

### ⚠️ 测量纪律（不遵守就会得出错结论）

- **host launch floor**：ctypes 每次 `cuLaunchKernel` 有 **~7 µs** 固定开销。
  `实测 / 7µs ≥ 4` 才可信；否则测到的是 host 抖动。
  → fp16 GEMM 在 1024³ 上 cuBLAS 只要 17 µs（比值 2.5×）→ **那一行根本测不准**，
  横向比较一律以 ≥2048³ 为准。
- **不要"统一减去 7 µs"**：host < GPU 时实测值**已经**是 GPU 时间，再减会低估
  （会让 cuBLAS 的 fp16 虚高到 224 TFLOPS，超过硬件密集算力）。
  正确做法是**报告原始值 + 标注可信度**。
- **单次测量不可信**：同一个 kernel 有一次跑出 27 029 GFLOPS，其余 5 次都是 55 600~59 200。
  **改结论前重复 3~5 次。**

---

## 五、复现

依赖：CUDA Toolkit（本仓库用 v12.8 的 `nvrtc64_120_0.dll` + `cublas64_12.dll`）、Python + numpy。
**不需要 nvcc** —— 全部 kernel 走 NVRTC 编译。

```bash
cd common
python vendor_baseline.py          # 带宽上限 + GEMM 家族厂商基线
python vendor_baseline.py bw       # 只量带宽
python bench_ours_vs_vendor.py     # 逐题差距

cd ../gemm-fp32 && python ladder_fp32.py     # fp32 十一级阶梯
cd ../gemm-fp16 && python ladder_fp16.py     # fp16 十二变体
```

> 注：原开发环境的 `nvcc` 不可用（`cl.exe` 会调用被安全策略拦截的 `reg.exe`），
> 所以整套测试框架走 `NVRTC + CUDA Driver API`（纯 ctypes，零编译工具链依赖）。
> 这个取舍也意味着**本仓库可以在没有完整 CUDA 工具链的 Windows 机器上复现**。

---

## 六、参考与致谢

本仓库的实现思路参考了以下开源项目（**只参考方法论与结构，未复制其源码**）：

- [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) —— fp32 SGEMM 的十级优化阶梯，
  warp tiling 在 A6000 上达到 cuBLAS 的 93.7%
- [Bruce-Lee-LY/cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm) —— fp16 HGEMM 的
  WMMA / MMA PTX 双路线阶梯（padding / async / pg2s / ps2r / multi-stage）
- [yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs](https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs)
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) —— `examples/79_blackwell_geforce_gemm`
  （sm_120 的官方写法）

## 七、License

MIT（`common/` 与各 `ladder_*.py` 为本仓库原创）。
