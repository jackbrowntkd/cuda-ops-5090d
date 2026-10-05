# -*- coding: utf-8 -*-
"""多形状综合对比：把「我们 vs 厂商库」从"一个形状的结论"升级成"一条曲线 + 一个综合分"。

为什么需要它：
  单形状对比可能是**挑过的** —— 一个 kernel 在 256³ 上赢、在 1024³ 上可能就输。
  cuBLAS 的启发式又是随形状变化的，所以"赢/输"本身就是形状的函数。
  必须扫一组形状，才能说出一句站得住的话。

报告口径（这三个数一起看）：
  · 各形状的逐点比值
  · **几何平均**（比算术平均更能反映"典型倍数"，不会被单个大比值带偏）
  · **最差/最好**比值（说明结论的稳健性 —— 最差那个才代表"下限"）
  · 胜负计数（赢在哪些形状上）

用法: python sweep_shapes.py              # 全部家族
      python sweep_shapes.py f32b f16b    # 只跑指定家族
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from localrun import Runner, nvrtc_compile   # noqa: E402
from cublas import Handle                    # noqa: E402

_NOOP = r"""
__global__ void noop_probe(float* p) { if (p && threadIdx.x > 1024) p[0] = 1.f; }
"""
HOST_FLOOR_MS = 6.5e-3

SOLUTIONS = os.environ.get(
    "SOLUTIONS_DIR",
    os.path.abspath(os.path.join(HERE, "..")))


def find_solution(subdir):
    p = os.path.join(SOLUTIONS, subdir, "solution.cu")
    if not os.path.isfile(p):
        raise FileNotFoundError(
            "找不到 %s\n  当前 SOLUTIONS_DIR = %s" % (p, SOLUTIONS))
    return p


def geom_mean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ------------------------------------------------------------------ 形状组
# 每组都在同一份 solution.cu 上跑，覆盖"能从一个小矩阵到多个大矩阵"的整个区间。
BATCHED_F32 = [
    (128, 128, 128, 128, "128³  B=128"),
    (256, 256, 256, 1, "256³  B=1"),
    (256, 256, 256, 8, "256³  B=8"),
    (256, 256, 256, 32, "256³  B=32"),
    (256, 256, 256, 128, "256³  B=128"),
    (512, 512, 512, 8, "512³  B=8"),
    (512, 512, 512, 32, "512³  B=32"),
    (1024, 1024, 1024, 4, "1024³ B=4"),
    (1024, 1024, 1024, 16, "1024³ B=16"),
    (1024, 256, 256, 32, "非方 1024x256x256 B=32"),
]
BATCHED_F16 = BATCHED_F32   # 同一组形状，换 dtype

SINGLE_F32 = [
    (512, 512, 512, 1, "512³"),
    (1024, 1024, 1024, 1, "1024³"),
    (2048, 2048, 2048, 1, "2048³"),
    (4096, 4096, 4096, 1, "4096³"),
    (8192, 6144, 4096, 1, "8192x6144x4096  (题面规模)"),
    (512, 4096, 4096, 1, "窄长 512x4096x4096"),
    (4096, 512, 4096, 1, "扁宽 4096x512x4096"),
]


def bench_batched(r, blas, kind, shapes):
    """kind: 'f32b' 或 'f16b'。返回逐形状比值列表。"""
    path = find_solution("030-Batched-Matrix-Multiplication" if kind == "f32b"
                         else "057-FP16-Batched-Matrix-Multiplication")
    rr = Runner.from_file(path)
    dt = np.float32 if kind == "f32b" else np.float16
    ratios = []
    print("  %-6s %-28s %11s %11s %10s"
          % ("dtype", "形状", "我们", "厂商库", "比值"))
    print("  " + "-" * 72)
    for (M, N, K, B, label) in shapes:
        rng = np.random.default_rng(7)
        A = (rng.standard_normal((B, M, K)) * 0.5).astype(dt)
        Bm = (rng.standard_normal((B, K, N)) * 0.5).astype(dt)
        C = np.zeros((B, M, N), dt)
        dA, dB, dC = rr.to_device(A), rr.to_device(Bm), rr.to_device(C)
        flop = 2.0 * M * N * K * B

        if kind == "f32b":
            ours = lambda: rr.launch("bmm_v10",
                                     ((N + 127) // 128, (M + 127) // 128, B),
                                     (256, 1, 1),
                                     [dA, dB, dC, rr.i(M), rr.i(N), rr.i(K)])
            vend = lambda: blas.gemm_f32_batched(dA, dB, dC, M, N, K, B, 1.0, 0.0)
        else:
            ours = lambda: rr.launch("fp16_bmm_wmma",
                                     ((N + 63) // 64, (M + 63) // 64, B),
                                     (128, 1, 1),
                                     [dA, dB, dC, rr.i(M), rr.i(N), rr.i(K)])
            vend = lambda: blas.gemm_f16_batched(dA, dB, dC, M, N, K, B, 1.0, 0.0)

        try:
            ours(); rr.sync()
            om = rr.timeit(ours, repeat=10, warmup=3)
            vm = rr.timeit(vend, repeat=10, warmup=3)
        except Exception as e:
            print("  %-6s %-28s  失败: %r" % (kind, label, e))
            for p in (dA, dB, dC):
                rr.free(p)
            continue
        go = flop / (om * 1e-3) / 1e9
        gv = flop / (vm * 1e-3) / 1e9
        rat = go / gv
        ratios.append(rat)
        mark = " 🏆" if rat > 1.0 else ""
        print("  %-6s %-28s %9.1f G %9.1f G %8.2fx%s"
              % (kind, label, go, gv, rat, mark))
        for p in (dA, dB, dC):
            rr.free(p)
    return ratios


def bench_single_f32(r, blas, shapes):
    """#002 fp32 单发 GEMM（注意 A 是 M×N、B 是 N×K，收缩维 N）。"""
    rr = Runner.from_file(find_solution("002-Matrix-Multiplication"))
    ratios = []
    print("  %-6s %-28s %11s %11s %10s"
          % ("dtype", "形状", "我们", "厂商库", "比值"))
    print("  " + "-" * 72)
    for (M, N, K, _b, label) in shapes:
        rng = np.random.default_rng(7)
        A = rng.standard_normal((M, N)).astype(np.float32)
        Bm = rng.standard_normal((N, K)).astype(np.float32)
        dA, dB = rr.to_device(A), rr.to_device(Bm)
        dC = rr.alloc(M * K * 4)
        flop = 2.0 * M * N * K
        ours = lambda: rr.launch("matmul_v10",
                                 ((K + 127) // 128, (M + 127) // 128, 1), (256, 1, 1),
                                 [dA, dB, dC, rr.i(M), rr.i(N), rr.i(K)])
        vend = lambda: blas.gemm_f32(dA, dB, dC, M, K, N, 1.0, 0.0)
        try:
            ours(); rr.sync()
            om = rr.timeit(ours, repeat=10, warmup=3)
            vm = rr.timeit(vend, repeat=10, warmup=3)
        except Exception as e:
            print("  %-6s %-28s  失败: %r" % ("f32", label, e))
            for p in (dA, dB, dC):
                rr.free(p)
            continue
        go = flop / (om * 1e-3) / 1e9
        gv = flop / (vm * 1e-3) / 1e9
        rat = go / gv
        ratios.append(rat)
        mark = " 🏆" if rat > 1.0 else ""
        print("  %-6s %-28s %9.1f G %9.1f G %8.2fx%s"
              % ("f32", label, go, gv, rat, mark))
        for p in (dA, dB, dC):
            rr.free(p)
    return ratios


def summarize(name, ratios):
    if not ratios:
        print("  %s: 无数据" % name)
        return
    win = sum(1 for x in ratios if x > 1.0)
    print()
    print("  ══ %s 综合 ══" % name)
    print("     形状数        : %d" % len(ratios))
    print("     **几何平均**  : **%.3fx**" % geom_mean(ratios))
    print("     最差 / 最好   : %.2fx / %.2fx" % (min(ratios), max(ratios)))
    print("     赢的形状数    : %d / %d%s"
          % (win, len(ratios), "  （全部赢）" if win == len(ratios) else ""))


def main():
    want = sys.argv[1:] or ["f32b", "f16b", "f32"]
    ptx, arch, _ = nvrtc_compile(_NOOP, name="base.cu")
    r = Runner(ptx, arch, "")
    blas = Handle()
    print("设备:", Runner.devicename(),
          "| host launch floor = %.1f us" % (HOST_FLOOR_MS * 1000))
    print()
    if "f32b" in want:
        print("=== fp32 batched（#030 的 bmm_v10）vs cuBLAS ===")
        summarize("fp32 batched", bench_batched(r, blas, "f32b", BATCHED_F32))
        print()
    if "f16b" in want:
        print("=== fp16 batched（#057 的 fp16_bmm_wmma）vs cuBLAS ===")
        summarize("fp16 batched", bench_batched(r, blas, "f16b", BATCHED_F16))
        print()
    if "f32" in want:
        print("=== fp32 单发 GEMM（#002 的 matmul_v10）vs cuBLAS ===")
        summarize("fp32 GEMM", bench_single_f32(r, blas, SINGLE_F32))
        print()
    blas.destroy()


if __name__ == "__main__":
    sys.exit(main())
