#!/usr/bin/env python
"""
实验 4：弱扩展性 (Weak Scaling)

每 GPU 负载固定，增加 GPU 数（总数据量线性增长）。
理想情况下执行时间应保持恒定。
  Efficiency(P) = T(1) / T(P)

对比方案：
  ① mpi4py + GPU (手写)
  ② Ours

任务: MatMul (per_gpu=1024 rows) / FFT2D (per_gpu=1 batch 512×512)

运行: mpirun --oversubscribe --allow-run-as-root -n {1,2,4} python experiments/exp04_weak_scaling.py
绘图: python experiments/exp04_weak_scaling.py --plot
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

from mpi4py import MPI
import torch, numpy as np

from distributed_gpu.algorithms.matrix_ops import distributed_matmul
from distributed_gpu.algorithms.fft import distributed_fft2d

PER_GPU_ROWS = 1024
FFT_SIZE     = 512
REPEATS      = 5


def _raw_mpi_matmul(comm, rank, ws, N, device):
    N = comm.bcast(N, root=0)
    chunk = (N + ws - 1) // ws
    pad_N = chunk * ws
    if rank == 0:
        A_np = np.random.randn(N, N).astype(np.float32)
        A_pad = np.vstack([A_np, np.zeros((pad_N - N, N), dtype=np.float32)]) \
                if pad_N > N else A_np.copy()
        B_np = np.random.randn(N, N).astype(np.float32)
    else:
        A_pad, B_np = None, np.empty((N, N), dtype=np.float32)
    A_loc = np.empty((chunk, N), dtype=np.float32)
    comm.Scatter(A_pad, A_loc, root=0)
    comm.Bcast(B_np, root=0)
    C = torch.matmul(torch.from_numpy(A_loc).to(device),
                     torch.from_numpy(B_np).to(device)).cpu().numpy()
    C_all = np.empty((pad_N, N), dtype=np.float32) if rank == 0 else None
    comm.Gather(C, C_all, root=0)


def run():
    mpi, dist, gpu = setup_framework()
    warmup(mpi, dist)
    comm = MPI.COMM_WORLD
    rank, ws = mpi.get_rank(), mpi.get_size()
    device = torch.device(f"cuda:{mpi.get_gpu_id()}")
    cost_model = setup_cost_model(mpi)

    N_mm = PER_GPU_ROWS * ws
    banner(f"实验4: 弱扩展性 (n={ws}, matmul={N_mm}×{N_mm}, "
           f"fft batch={ws}×{FFT_SIZE})", rank)
    R = {"num_gpus": ws, "gpu": gpu_name()}

    # ---- MatMul ----
    # mpi4py + GPU
    s_raw, _ = timed_mpi(lambda: _raw_mpi_matmul(comm, rank, ws, N_mm, device),
                         mpi, repeats=REPEATS)
    # Ours
    def _mm():
        A = torch.randn(N_mm, N_mm, device=device) if rank == 0 else None
        B = torch.randn(N_mm, N_mm, device=device) if rank == 0 else None
        return distributed_matmul(A, B, mpi, dist, cost_model)
    s_ours, _ = timed_mpi(_mm, mpi, repeats=REPEATS)

    if rank == 0:
        R["matmul_raw"] = s_raw["mean"] * 1000
        R["matmul_ours"] = s_ours["mean"] * 1000
        log(f"MatMul [{N_mm}×{N_mm}]: mpi_gpu={s_raw['mean']*1000:.1f}ms  "
            f"ours={s_ours['mean']*1000:.1f}ms", rank)

    # ---- FFT 2D ----
    def _fft():
        x = torch.randn(ws, FFT_SIZE, FFT_SIZE, device=device) if rank == 0 else None
        return distributed_fft2d(x, mpi, dist)
    s_fft, _ = timed_mpi(_fft, mpi, repeats=REPEATS)

    if rank == 0:
        R["fft_ours"] = s_fft["mean"] * 1000
        log(f"FFT2D [{ws}×{FFT_SIZE}×{FFT_SIZE}]: ours={s_fft['mean']*1000:.1f}ms", rank)
        save_json(f"exp04_weak_scaling_n{ws}", R)

    log("Done.", rank)


def plot():
    all_data = {}
    for n in [1, 2, 4, 8]:
        try:
            all_data[n] = load_json(f"exp04_weak_scaling_n{n}")
        except FileNotFoundError:
            pass
    if not all_data:
        print("No data found."); return

    gpus = sorted(all_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (key_raw, key_ours, title) in zip(axes, [
            ("matmul_raw", "matmul_ours",
             f"MatMul (per-GPU {PER_GPU_ROWS} rows)"),
            (None, "fft_ours",
             f"FFT2D (per-GPU 1×{FFT_SIZE}×{FFT_SIZE})")]):
        times_ours = [all_data[g].get(key_ours, 0) for g in gpus]
        base_ours = times_ours[0] if times_ours[0] > 0 else 1
        eff_ours = [base_ours / t if t > 0 else 0 for t in times_ours]

        ax2 = ax.twinx()
        ax.bar([i - 0.15 for i in range(len(gpus))], times_ours, 0.3,
               color=COLORS[0], alpha=0.7, label="Ours Time (ms)")

        if key_raw:
            times_raw = [all_data[g].get(key_raw, 0) for g in gpus]
            base_raw = times_raw[0] if times_raw[0] > 0 else 1
            eff_raw = [base_raw / t if t > 0 else 0 for t in times_raw]
            ax.bar([i + 0.15 for i in range(len(gpus))], times_raw, 0.3,
                   color=COLORS[3], alpha=0.7, label="mpi4py+GPU Time (ms)")
            ax2.plot(range(len(gpus)), eff_raw, "-s",
                     color=COLORS[3], label="mpi4py+GPU Efficiency")

        ax2.plot(range(len(gpus)), eff_ours, "-o",
                 color=COLORS[0], label="Ours Efficiency")
        ax2.axhline(1.0, linestyle="--", color="gray", alpha=0.5)
        ax2.set_ylim(0, 1.5); ax2.set_ylabel("Parallel Efficiency")

        ax.set_xticks(range(len(gpus)))
        ax.set_xticklabels([str(g) for g in gpus])
        ax.set_xlabel("Number of GPUs"); ax.set_ylabel("Time (ms)")
        ax.set_title(f"{title} — Weak Scaling")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.suptitle("Exp-4: Weak Scaling", fontsize=14, y=1.02)
    save_fig(fig, "exp04_weak_scaling")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
