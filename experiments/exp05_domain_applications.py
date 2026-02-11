#!/usr/bin/env python
"""
实验 5：领域科学计算应用对比

对比在科学计算典型应用场景中的性能：
  A. 二维卷积 (Conv2D) — 图像处理 / 信号处理 / CNN 基础
  B. Stencil 2D        — 有限差分法（热传导 / 波动方程）
  C. Jacobi 迭代       — 椭圆型 PDE（泊松方程）

对比方案：
  A 任务:
    ① SciPy (correlate2d) — CPU 科学计算
    ② PyTorch (F.conv2d)  — 深度学习框架 (单 GPU)
    ③ Ours (distributed_conv2d) — 本框架分布式卷积
  B/C 任务:
    ① NumPy 单机串行     — CPU 基线
    ② PyTorch 单 GPU     — GPU 加速基线
    ③ Ours (distributed_stencil_2d / jacobi_2d) — 本框架分布式

运行: mpirun --oversubscribe --allow-run-as-root -n 4 python experiments/exp05_domain_applications.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

from mpi4py import MPI
import torch, numpy as np
import torch.nn.functional as F

from distributed_gpu.algorithms.convolution import distributed_conv2d
from distributed_gpu.algorithms.stencil import distributed_stencil_2d, distributed_jacobi_2d

CONV_SIZES  = [(64, 64), (128, 128), (256, 256), (512, 512)]
KERNEL_SIZE = 5
STENCIL_SIZES = [256, 512, 1024, 2048]
JACOBI_SIZES  = [128, 256, 512, 1024]
JACOBI_ITERS  = 20
REPEATS = 3


def run():
    mpi, dist, gpu = setup_framework()
    warmup(mpi, dist)
    rank, ws = mpi.get_rank(), mpi.get_size()
    device = torch.device(f"cuda:{mpi.get_gpu_id()}")

    banner(f"实验5: 领域科学计算应用对比 ({ws} GPUs)", rank)
    R = {"gpu": gpu_name(), "num_gpus": ws,
         "conv2d": [], "stencil": [], "jacobi": []}

    # ==================== A. Conv2D ====================
    log("--- A. Conv2D ---", rank)
    for H, W in CONV_SIZES:
        entry = {"H": H, "W": W}

        # ① SciPy
        if HAS_SCIPY and rank == 0:
            from scipy.signal import correlate2d
            x_np = np.random.randn(H, W).astype(np.float32)
            k_np = np.random.randn(KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)
            s, _ = timed(lambda: correlate2d(x_np, k_np, mode="same"),
                         repeats=REPEATS, sync_cuda=False)
            entry["scipy"] = s["mean"] * 1000
        mpi.synchronize()

        # ② PyTorch 单 GPU
        if rank == 0:
            x_t = torch.randn(1, 1, H, W, device=device)
            k_t = torch.randn(1, 1, KERNEL_SIZE, KERNEL_SIZE, device=device)
            pad = KERNEL_SIZE // 2
            s, _ = timed(lambda: F.conv2d(x_t, k_t, padding=pad), repeats=REPEATS)
            entry["pytorch_1gpu"] = s["mean"] * 1000
        mpi.synchronize()

        # ③ Ours
        def _ours_conv(h=H, w=W):
            x = torch.randn(1, 1, h, w, device=device) if rank == 0 else None
            k = torch.randn(1, 1, KERNEL_SIZE, KERNEL_SIZE, device=device) \
                if rank == 0 else None
            return distributed_conv2d(x, k, mpi, dist, padding=KERNEL_SIZE // 2)
        s, _ = timed_mpi(_ours_conv, mpi, repeats=REPEATS)
        if rank == 0:
            entry["ours"] = s["mean"] * 1000

        R["conv2d"].append(entry)
        if rank == 0:
            log(f"  {H}×{W}: scipy={entry.get('scipy',0):.2f}  "
                f"pt_1gpu={entry.get('pytorch_1gpu',0):.2f}  "
                f"ours={entry.get('ours',0):.2f} ms", rank)
        mpi.synchronize()

    # ==================== B. Stencil 2D ====================
    log("--- B. Stencil 2D (5-point Laplacian) ---", rank)
    for N in STENCIL_SIZES:
        entry = {"N": N}

        # ① NumPy 串行
        if rank == 0:
            def _np_stencil(n=N):
                g = np.random.randn(n, n).astype(np.float32)
                out = np.zeros_like(g)
                out[1:-1, 1:-1] = (g[:-2, 1:-1] + g[2:, 1:-1] +
                                   g[1:-1, :-2] + g[1:-1, 2:] -
                                   4 * g[1:-1, 1:-1])
                return out
            s, _ = timed(_np_stencil, repeats=REPEATS, sync_cuda=False)
            entry["numpy"] = s["mean"] * 1000
        mpi.synchronize()

        # ② PyTorch 单 GPU (用 F.conv2d 实现 5 点 stencil)
        if rank == 0:
            lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                      dtype=torch.float32, device=device) \
                         .reshape(1, 1, 3, 3)
            g_t = torch.randn(1, 1, N, N, device=device)
            s, _ = timed(lambda: F.conv2d(g_t, lap_kernel, padding=1),
                         repeats=REPEATS)
            entry["pytorch_1gpu"] = s["mean"] * 1000
        mpi.synchronize()

        # ③ Ours
        def _ours_stencil(n=N):
            g = torch.randn(n, n, device=device) if rank == 0 else None
            return distributed_stencil_2d(g, mpi, dist)
        s, _ = timed_mpi(_ours_stencil, mpi, repeats=REPEATS)
        if rank == 0:
            entry["ours"] = s["mean"] * 1000

        R["stencil"].append(entry)
        if rank == 0:
            log(f"  N={N}: numpy={entry.get('numpy',0):.2f}  "
                f"pt_1gpu={entry.get('pytorch_1gpu',0):.2f}  "
                f"ours={entry.get('ours',0):.2f} ms", rank)
        mpi.synchronize()

    # ==================== C. Jacobi 迭代 ====================
    log(f"--- C. Jacobi 2D ({JACOBI_ITERS} iterations) ---", rank)
    for N in JACOBI_SIZES:
        entry = {"N": N, "iters": JACOBI_ITERS}

        # ① NumPy 串行
        if rank == 0:
            def _np_jacobi(n=N, it=JACOBI_ITERS):
                g = np.random.randn(n, n).astype(np.float32)
                b = np.random.randn(n, n).astype(np.float32)
                for _ in range(it):
                    g_new = np.copy(g)
                    g_new[1:-1, 1:-1] = 0.25 * (
                        g[:-2, 1:-1] + g[2:, 1:-1] +
                        g[1:-1, :-2] + g[1:-1, 2:] - b[1:-1, 1:-1])
                    g = g_new
                return g
            s, _ = timed(_np_jacobi, repeats=REPEATS, sync_cuda=False)
            entry["numpy"] = s["mean"] * 1000
        mpi.synchronize()

        # ② PyTorch 单 GPU
        if rank == 0:
            def _pt_jacobi(n=N, it=JACOBI_ITERS):
                g = torch.randn(n, n, device=device)
                b = torch.randn(n, n, device=device)
                for _ in range(it):
                    g_new = g.clone()
                    g_new[1:-1, 1:-1] = 0.25 * (
                        g[:-2, 1:-1] + g[2:, 1:-1] +
                        g[1:-1, :-2] + g[1:-1, 2:] - b[1:-1, 1:-1])
                    g = g_new
                return g
            s, _ = timed(_pt_jacobi, repeats=REPEATS)
            entry["pytorch_1gpu"] = s["mean"] * 1000
        mpi.synchronize()

        # ③ Ours
        def _ours_jacobi(n=N, it=JACOBI_ITERS):
            g = torch.randn(n, n, device=device) if rank == 0 else None
            b = torch.randn(n, n, device=device) if rank == 0 else None
            return distributed_jacobi_2d(g, b, mpi, dist,
                                          iterations=it)
        s, _ = timed_mpi(_ours_jacobi, mpi, repeats=REPEATS)
        if rank == 0:
            entry["ours"] = s["mean"] * 1000

        R["jacobi"].append(entry)
        if rank == 0:
            log(f"  N={N}: numpy={entry.get('numpy',0):.1f}  "
                f"pt_1gpu={entry.get('pytorch_1gpu',0):.1f}  "
                f"ours={entry.get('ours',0):.1f} ms", rank)
        mpi.synchronize()

    if rank == 0:
        save_json("exp05_domain_applications", R)
        plot(R)
    log("Done.", rank)


def plot(data=None):
    if data is None:
        data = load_json("exp05_domain_applications")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    tasks = [("conv2d",  "Conv2D", [("scipy", "SciPy"),
                                     ("pytorch_1gpu", "PyTorch 1 GPU"),
                                     ("ours", "Ours")]),
             ("stencil", "Stencil 2D", [("numpy", "NumPy"),
                                         ("pytorch_1gpu", "PyTorch 1 GPU"),
                                         ("ours", "Ours")]),
             ("jacobi",  f"Jacobi 2D ({JACOBI_ITERS} iters)",
              [("numpy", "NumPy"),
               ("pytorch_1gpu", "PyTorch 1 GPU"),
               ("ours", "Ours")])]

    for ax, (tkey, title, fws) in zip(axes, tasks):
        records = data[tkey]
        if tkey == "conv2d":
            xs = [f"{r['H']}×{r['W']}" for r in records]
        else:
            xs = [str(r["N"]) for r in records]
        x_pos = np.arange(len(xs))
        nf = len(fws)
        w = 0.8 / nf
        for i, (k, lbl) in enumerate(fws):
            vals = [r.get(k, 0) for r in records]
            ax.bar(x_pos + (i - nf / 2 + 0.5) * w, vals, w,
                   label=lbl, color=COLORS[i * 2], edgecolor="white")
        ax.set_xticks(x_pos); ax.set_xticklabels(xs)
        ax.set_xlabel("Grid Size"); ax.set_ylabel("Time (ms)")
        ax.set_title(title); ax.legend(fontsize=9)
        ax.set_yscale("log")

    fig.suptitle(f"Exp-5: Domain Applications ({data['num_gpus']} GPUs, {data['gpu']})",
                 fontsize=14, y=1.02)
    save_fig(fig, "exp05_domain_applications")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
