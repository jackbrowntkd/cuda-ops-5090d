# 算子基线对标报告（RTX 5090 D）

> 方法论见 skill `cuda-operator-optimization`。本文件是「5090D CUDA 算子优化系列」的地基：
> 先量厂商基线 → 再量自己的差距 → 排序找 headroom → 逐个攻克。

- 目标 GPU：**NVIDIA GeForce RTX 5090 D**（sm_120，170 SM）
- 复现：`python leetgpu-work/_tools/vendor_baseline.py`（厂商基线 + 带宽）
  / `python leetgpu-work/_tools/bench_ours_vs_vendor.py`（逐题差距）
- 测量口径：原始耗时 + host launch floor（实测 **6.2~7.3 µs**）；
  `实测 / floor ≥ 4` 才可信。所有 GPU 计时走 cuEvent，warmup 后取 3 轮 min。

---

## 一、本机两个"天花板"

| 天花板 | 实测值 | 用于对标哪类算子 |
|---|---|---|
| **D2D 显存带宽** | **1 514.9 GB/s** | 纯 elementwise / 数据搬运型（理论上限） |
| **cuBLAS fp32 FFMA** | ~63~68 TFLOPS | 计算密集型（fp32） |

> ⚠️ **「5090 的 fp32 = 116 TFLOPS」是 TF32 张量核，不是真 FFMA。**
> 真 fp32（torch 默认不开 tf32 实测 66.4，本机 `CUBLAS_COMPUTE_32F` 实测 68）≈ **66~68 TFLOPS**。
> 本机带宽实测 1 514.9 GB/s 与第三方 `binbinsh/gpu-bench` 的 1 528.8 只差 **0.9%**，交叉验证通过。

---

## 二、GEMM 家族：厂商基线（规模取自各题**题面写明的性能测试规模**）

| 题 | 形状 | cuBLAS 耗时 | 吞吐 | 可信度 |
|---|---|---|---|---|
| #002 fp32 GEMM | M=8192 N=6144 K=4096 | 6.28 ms | **65 609 GFLOPS** | OK |
| #022 fp16 GEMM | 1024³ | 0.017 ms | 127 693 | ⚠️ ~host(2.6×) **不可信** |
| #030 fp32 batched | 256³ × B=128 | 0.109 ms | **38 599 GFLOPS** | OK |
| #032 int8 GEMM | M=8192 N=4096 K=2048 | 0.652 ms | **217 487 GFLOPS** | OK |
| #057 fp16 batched | 256³ × B=128 | 0.044 ms | **95 546 GFLOPS** | OK |

⭐ **两个立即有用的观察**：

1. **小尺寸 batched 是 cuBLAS 的弱区**：#030 只有 38.6 TFLOPS、#057 只有 95.5 TFLOPS，
   而同类的**大尺寸单发**分别是 65.6（fp32）和 236（fp16）TFLOPS。
   → **小尺寸 + 多 batch 正是手写容易追平甚至反超的窗口**（与 T600 时代"手写超过 cuBLAS"同源）。
2. **#022 的 fp16 在 1024³ 上测不准**（0.017 ms 只有 host floor 的 2.6 倍）——
   那个规模下 GPU 时间被 host 开销淹没，横向比较一律以 ≥2048³ 为准。

---

## 三、差距总表：**我们的实现 vs 厂商库**（各题真实性能规模）

| 题 | 算子 | 我们 | 厂商库 | **差距** | headroom |
|---|---|---|---|---|---|
| #022 | fp16 GEMM | 159 657 GFLOPS（V8 @4096³） | 212 572 | **0.75×** | 1.3× |
| #002 | fp32 GEMM | 5 876 GFLOPS | 62 986 | **0.09×** | **~11×** |
| #030 | fp32 batched | 5 929 GFLOPS | 38 599 | **0.15×** | **6.5×** |
| #057 | fp16 batched | 6 242 GFLOPS | 95 546 | **0.07×** | **15×** |
| #032 | int8 MatMul | 6 463 GFLOPS | 217 487 | **0.03×** ⚠️ | ~34× |

⚠️ **#032 的差距不是严格可比的**：cuBLAS 的调用是纯 int8 GEMM（int32 累加），
**不含 zero_point / scale 校正**；我们的 `i8_gemm` 要做量化反量化。
所以 0.03× 只能当"忽略零点校正的理论上界参照"，别当真实差距报。

### ⭐ 这张表最重要的一条结论

**#002 / #030 / #032 / #057 四个实现全部卡在 ~5 900~6 500 GFLOPS**（≈ 6 TFLOPS）。
这不是巧合 —— 它们是早期为了**通过平台**写的朴素 tiled 版本（32×32 / 16×16 线程块），
压根没做过性能优化。作为对照，我们在 #022 上已经把同类 fp32 GEMM 优化到 **0.85× cuBLAS**
（v10d，见 `022-.../PERF.md`）。

> **⇒ 最高性价比的下一步：把 #022 上已经验证过的 v10d 设计（128×128 block / BM=BN=128 /
> BK=16 / WM=64 WN=32 / WNITER=2 / NT=256 / 无 padding 转置 A）推广到 #002，
> 预计 5 876 → ~50 000 GFLOPS，约 8~9 倍提升。**

---

## 四、全部已通过算子的基线分类（53 题）

不是每个算子都有厂商库对标，但**每个都有可量的基线**。按类型分：

| 类型 | 题号 | 基线来源 | 基线值 |
|---|---|---|---|
| **稠密 GEMM** | #002 #022 | cuBLAS `GemmEx` | ✅ 已量（见上表） |
| **Batched GEMM** | #030 #057 | cuBLAS `GemmStridedBatchedEx` | ✅ 已量 |
| **量化 MatMul** | #032 | cuBLAS int8（不含零点校正） | ✅ 已量（带警告） |
| **Dot / GEMV** | #017 #058 | cuBLAS `dot` / `gemv` | 待接 |
| **Sparse** | #018 #075 | cuSPARSE `csrmv` / SpMM | 待接 |
| **Attention** | #006 #012 #026 #055 #056 | PyTorch SDPA / flash-attn（WSL 里有 torch） | 待接 |
| **归约 / 扫描** | #004 #016 #047 #048 #049 #051 | CUB `DeviceReduce` / `DeviceScan` | 待接 |
| **Softmax / Norm** | #005 #050 | PyTorch / cuDNN | 待接 |
| **排序** | #015 #036 | CUB `DeviceRadixSort` | 待接 |
| **纯 elementwise** | #001 #007 #008 #019 #021 #023 #031 #052 #054 | ⭐ **D2D 带宽 1 514.9 GB/s** | ✅ 上限已量 |
| **其他（无对标）** | #003 #009 #010 #011 #013 #014 #024 #025 #027 #028 #029 #033 #035 #037 #038 #040 #042 #043 #044 #045 #058 #075 | 只记"自己的历史最好"，靠迭代改进 | – |

**纯带宽型怎么算达标率**：理论最短时间 = `读写总字节数 / 1514.9 GB/s`。
例如 elementwise `C = f(A,B)`（fp32，N 个元素）：`N=2^24` → 192 MB 读写 → 下限 **0.133 ms**。
达到这个下限的百分比就是这类算子的唯一评分标准。

---

## 五、优先级（按 headroom 排）

| 优先级 | 目标 | 预期收益 | 依据 |
|---|---|---|---|
| **P0** | #002 换 v10d 设计（fp32 GEMM） | **8~9×** | 设计已验证到 0.85× cuBLAS |
| **P0** | #057 fp16 batched 换 WMMA 设计 | **~10×** | #022 的 WMMA BK=64 已验证 |
| **P1** | #030 fp32 batched 上 tiling + float4 | **~5×** | fp32 阶梯已验证 float4/BK 手法 |
| **P1** | #032 int8 换 int8 tensor core（`mma.sync` s8） | 大 | 需先确认 sm_120 的 s8 mma 路径 |
| **P2** | elementwise 家族量"带宽达标率" | 定量诊断 | 带宽上限已量 |
| **P2** | Attention 家族接 PyTorch SDPA 基线 | 定量诊断 | WSL 有 torch |

---

## 六、复现命令

```bash
cd C:/Users/admin/Desktop/CUDA算子/leetgpu-work/_tools
PY="C:/Users/admin/AppData/Local/Programs/Python/Python313/python.exe"

$PY vendor_baseline.py          # 带宽上限 + GEMM 家族厂商基线
$PY vendor_baseline.py bw       # 只量带宽
$PY bench_ours_vs_vendor.py     # 逐题差距（2 30 32 57 可指定题号）
```

⚠️ 必须用 `Programs/Python/Python313/python.exe`（numpy 2.5.0）；
managed 的 3.13.12 没有 numpy。
