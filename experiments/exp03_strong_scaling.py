#!/usr/bin/env python
"""
实验 3：强扩展性 (Strong Scaling)

固定问题规模，增加 GPU 数，观察加速比和并行效率。
  SpeedUp(P) = T(1) / T(P)
  Efficiency(P) = SpeedUp(P) / P

对比方案：
  ① mpi4py + GPU (手写)
  ② PyTorch Distributed
  ③ Ours

任务: MatMul 4096×4096 / FFT2D 1024×1024

运行: mpirun --oversubscribe --allow-run-as-root -n {1,2,4} python experiments/exp03_strong_scaling.py
绘图: python experiments/exp03_strong_scaling.py --plot
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

from mpi4py import MPI
import torch, numpy as np

from distributed_gpu.algorithms.matrix_ops import distributed_matmul
from distributed_gpu.algorithms.fft import distributed_fft2d

MATMUL_N = 4096
FFT_N    = 1024
FFT_BATCH = 8
REPEATS  = 5


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
    pt_ok = init_torch_distributed(rank, ws, mpi.get_gpu_id())

    banner(f"实验3: 强扩展性 (n={ws})", rank)
    R = {"num_gpus": ws, "gpu": gpu_name()}

    # ---- MatMul ----
    # mpi4py + GPU
    s_raw, _ = timed_mpi(lambda: _raw_mpi_matmul(comm, rank, ws, MATMUL_N, device),
                         mpi, repeats=REPEATS)
    # PyTorch Distributed
    mm_pt = None
    if pt_ok:
        import torch.distributed as tdist
        def _pt_mm():
            chunk = (MATMUL_N + ws - 1) // ws
            A_loc = torch.randn(chunk, MATMUL_N, device=device)
            B = torch.randn(MATMUL_N, MATMUL_N, device=device)
            tdist.broadcast(B, src=0)
            C_loc = torch.matmul(A_loc, B)
            gl = [torch.zeros_like(C_loc) for _ in range(ws)] if rank == 0 else None
            tdist.gather(C_loc, gl, dst=0)
        s_pt, _ = timed_mpi(_pt_mm, mpi, repeats=REPEATS)
        mm_pt = s_pt["mean"] * 1000

    # Ours
    def _ours_mm():
        A = torch.randn(MATMUL_N, MATMUL_N, device=device) if rank == 0 else None
        B = torch.randn(MATMUL_N, MATMUL_N, device=device) if rank == 0 else None
        return distributed_matmul(A, B, mpi, dist, cost_model)
    s_ours_mm, _ = timed_mpi(_ours_mm, mpi, repeats=REPEATS)

    if rank == 0:
        R["matmul"] = {"mpi_gpu": s_raw["mean"] * 1000,
                       "ours": s_ours_mm["mean"] * 1000}
        if mm_pt is not None:
            R["matmul"]["pytorch_dist"] = mm_pt
        log(f"MatMul {MATMUL_N}: mpi_gpu={s_raw['mean']*1000:.1f}ms "
            f"pt={mm_pt or 0:.1f}ms ours={s_ours_mm['mean']*1000:.1f}ms", rank)

    # ---- FFT 2D ----
    def _ours_fft():
        x = torch.randn(FFT_BATCH, FFT_N, FFT_N, device=device) if rank == 0 else None
        return distributed_fft2d(x, mpi, dist)
    s_ours_fft, _ = timed_mpi(_ours_fft, mpi, repeats=REPEATS)

    if rank == 0:
        R["fft"] = {"ours": s_ours_fft["mean"] * 1000}
        log(f"FFT2D {FFT_BATCH}×{FFT_N}: ours={s_ours_fft['mean']*1000:.1f}ms", rank)
        save_json(f"exp03_strong_scaling_n{ws}", R)

    cleanup_torch_distributed()
    log("Done.", rank)


def plot():
    all_data = {}
    for n in [1, 2, 4, 8]:
        try:
            all_data[n] = load_json(f"exp03_strong_scaling_n{n}")
        except FileNotFoundError:
            pass
    if not all_data:
        print("No data found."); return

    gpus = sorted(all_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MatMul Speedup
    ax = axes[0]
    for key, lbl, ci in [("ours", "Ours", 0),
                          ("mpi_gpu", "mpi4py+GPU", 3),
                          ("pytorch_dist", "PyTorch Dist.", 4)]:
        base_data = all_data[gpus[0]]["matmul"]
        if key not in base_data:
            continue
        base = base_data[key]
        sp = [base / all_data[g]["matmul"].get(key, base) for g in gpus]
        ax.plot(gpus, sp, f"-{MARKERS[ci]}", label=lbl, color=COLORS[ci])
    ax.plot(gpus, gpus, "--", color="gray", label="Ideal", alpha=0.6)
    ax.set_xlabel("Number of GPUs"); ax.set_ylabel("Speedup")
    ax.set_title(f"MatMul {MATMUL_N}×{MATMUL_N} — Strong Scaling")
    ax.set_xticks(gpus); ax.legend()

    # FFT Speedup
    ax = axes[1]
    base_fft = all_data[gpus[0]]["fft"]["ours"]
    fft_sp = [base_fft / all_data[g]["fft"]["ours"] for g in gpus]
    ax.plot(gpus, fft_sp, "-o", label="Ours", color=COLORS[0])
    ax.plot(gpus, gpus, "--", color="gray", label="Ideal", alpha=0.6)
    ax.set_xlabel("Number of GPUs"); ax.set_ylabel("Speedup")
    ax.set_title(f"FFT2D {FFT_BATCH}×{FFT_N} — Strong Scaling")
    ax.set_xticks(gpus); ax.legend()

    fig.suptitle("Exp-3: Strong Scaling", fontsize=14, y=1.02)
    save_fig(fig, "exp03_strong_scaling")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
