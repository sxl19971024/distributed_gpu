#!/usr/bin/env python
"""
实验 1：核心计算性能综合对比

对比 7 个框架在三大核心科学计算任务上的性能：
  ① NumPy                — CPU 单机基线（通用科学计算）
  ② SciPy                — CPU 科学计算库（优化 FFT / 卷积）
  ③ CuPy                 — 单 GPU 科学计算库（NumPy 兼容）
  ④ Dask                 — 分布式 CPU 科学计算框架
  ⑤ mpi4py + GPU (手写)  — 传统 HPC 手工分布式
  ⑥ PyTorch Distributed  — 深度学习框架 (NCCL)
  ⑦ Ours                 — 本框架（代价模型 + 分布式算子）

任务：
  A. 矩阵乘法 (MatMul)   — 密集线性代数核心
  B. 二维 FFT (FFT2D)     — 频谱分析核心
  C. 全局归约 (Sum)       — 统计/能量计算核心

指标：执行时间 (ms)、相对 NumPy 加速比

运行: mpirun --oversubscribe --allow-run-as-root -n 4 python experiments/exp01_compute_performance.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

from mpi4py import MPI
import torch, numpy as np

from distributed_gpu.algorithms.matrix_ops import distributed_matmul
from distributed_gpu.algorithms.fft import distributed_fft2d
from distributed_gpu.algorithms.reduction import distributed_sum

# ---------- mpi4py + GPU 手写对比实现 ----------

def _raw_mpi_matmul(comm, rank, ws, N, device):
    """手写 MPI 分布式 MatMul: Scatter A 行 → Bcast B → 本地 gemm → Gather C"""
    if rank == 0:
        A_np = np.random.randn(N, N).astype(np.float32)
        B_np = np.random.randn(N, N).astype(np.float32)
    else:
        A_np = B_np = None
    N = comm.bcast(N, root=0)
    chunk = (N + ws - 1) // ws
    pad_N = chunk * ws
    if rank == 0:
        A_pad = np.vstack([A_np, np.zeros((pad_N - N, N), dtype=np.float32)]) \
                if pad_N > N else A_np.copy()
    else:
        A_pad = None
    A_loc = np.empty((chunk, N), dtype=np.float32)
    comm.Scatter(A_pad, A_loc, root=0)
    if rank != 0:
        B_np = np.empty((N, N), dtype=np.float32)
    comm.Bcast(B_np, root=0)
    C_g = torch.matmul(torch.from_numpy(A_loc).to(device),
                       torch.from_numpy(B_np).to(device))
    C_loc = C_g.cpu().numpy()
    C_all = np.empty((pad_N, N), dtype=np.float32) if rank == 0 else None
    comm.Gather(C_loc, C_all, root=0)
    return C_all[:N] if rank == 0 else None


def _raw_mpi_fft2d(comm, rank, ws, N, device):
    """手写 MPI 分布式 FFT2D: 按 batch 维 Scatter → 本地 fft2 → Gather"""
    batch = ws
    if rank == 0:
        x_np = np.random.randn(batch, N, N).astype(np.float32)
    else:
        x_np = None
    loc = np.empty((1, N, N), dtype=np.float32)
    comm.Scatter(x_np, loc, root=0)
    r = torch.fft.fft2(torch.from_numpy(loc).to(device))
    r_np = r.cpu().numpy()
    out = np.empty((batch, N, N), dtype=np.complex64) if rank == 0 else None
    comm.Gather(r_np, out, root=0)
    return out


def _raw_mpi_reduce(comm, rank, ws, numel, device):
    """手写 MPI 归约: 每 rank 本地 sum → MPI reduce"""
    per = numel // ws
    local = torch.randn(per, dtype=torch.float32, device=device)
    s = float(local.sum().cpu())
    return comm.reduce(s, op=MPI.SUM, root=0)


# ---------- 实验参数 ----------

MATMUL_SIZES  = [512, 1024, 2048, 4096, 8192]
FFT_SIZES     = [128, 256, 512, 1024]
REDUCE_ELEMS  = [1_000_000, 4_000_000, 16_000_000, 64_000_000]
REPEATS = 5


def run():
    mpi, dist, gpu = setup_framework()
    warmup(mpi, dist)
    comm = MPI.COMM_WORLD
    rank, ws = mpi.get_rank(), mpi.get_size()
    device = torch.device(f"cuda:{mpi.get_gpu_id()}")
    cost_model = setup_cost_model(mpi)
    pt_ok = init_torch_distributed(rank, ws, mpi.get_gpu_id())

    banner(f"实验1: 核心计算性能综合对比 ({ws} GPUs, {gpu_name()})", rank)
    R = {"gpu": gpu_name(), "num_gpus": ws,
         "matmul": [], "fft": [], "reduce": []}

    # ==================== A. MatMul ====================
    log("--- A. MatMul (C = A × B) ---", rank)
    for N in MATMUL_SIZES:
        entry = {"N": N}

        # ① NumPy
        if rank == 0:
            s, _ = timed(lambda: np.matmul(
                np.random.randn(N, N).astype(np.float32),
                np.random.randn(N, N).astype(np.float32)),
                repeats=REPEATS, sync_cuda=False)
            entry["numpy"] = s["mean"] * 1000
        mpi.synchronize()

        # ② SciPy (scipy.linalg.blas — 对 matmul 无明显优势，跳过)

        # ③ CuPy
        if HAS_CUPY and rank == 0:
            cp.cuda.Device(mpi.get_gpu_id()).use()
            s, _ = timed(lambda: cp.matmul(
                cp.random.randn(N, N, dtype=cp.float32),
                cp.random.randn(N, N, dtype=cp.float32)),
                repeats=REPEATS)
            entry["cupy"] = s["mean"] * 1000
        mpi.synchronize()

        # ④ Dask
        if HAS_DASK and rank == 0:
            dask.config.set(scheduler="threads", num_workers=ws)
            A_np = np.random.randn(N, N).astype(np.float32)
            B_np = np.random.randn(N, N).astype(np.float32)
            ch = max(N // ws, 1)
            s, _ = timed(lambda: da.matmul(
                da.from_array(A_np, chunks=(ch, N)),
                da.from_array(B_np, chunks=(N, ch))).compute(),
                repeats=REPEATS, sync_cuda=False)
            entry["dask"] = s["mean"] * 1000
        mpi.synchronize()

        # ⑤ mpi4py + GPU 手写
        s, _ = timed_mpi(lambda: _raw_mpi_matmul(comm, rank, ws, N, device),
                         mpi, repeats=REPEATS)
        if rank == 0:
            entry["mpi_gpu"] = s["mean"] * 1000

        # ⑥ PyTorch Distributed (NCCL)
        if pt_ok:
            import torch.distributed as tdist
            def _pt_mm(n=N):
                chunk = (n + ws - 1) // ws
                A_loc = torch.randn(chunk, n, device=device)
                B = torch.randn(n, n, device=device)
                tdist.broadcast(B, src=0)
                C_loc = torch.matmul(A_loc, B)
                gl = [torch.zeros_like(C_loc) for _ in range(ws)] if rank == 0 else None
                tdist.gather(C_loc, gl, dst=0)
            s, _ = timed_mpi(lambda: _pt_mm(N), mpi, repeats=REPEATS)
            if rank == 0:
                entry["pytorch_dist"] = s["mean"] * 1000

        # ⑦ Ours
        def _ours(n=N):
            A = torch.randn(n, n, device=device) if rank == 0 else None
            B = torch.randn(n, n, device=device) if rank == 0 else None
            return distributed_matmul(A, B, mpi, dist, cost_model)
        s, _ = timed_mpi(_ours, mpi, repeats=REPEATS)
        if rank == 0:
            entry["ours"] = s["mean"] * 1000

        R["matmul"].append(entry)
        if rank == 0:
            parts = [f"N={N:>5}"]
            for k in ["numpy", "cupy", "dask", "mpi_gpu", "pytorch_dist", "ours"]:
                if k in entry:
                    parts.append(f"{k}={entry[k]:.1f}")
            log(" | ".join(parts), rank)
        mpi.synchronize()

    # ==================== B. FFT 2D ====================
    log("--- B. FFT 2D ---", rank)
    for N in FFT_SIZES:
        batch = ws
        entry = {"N": N, "batch": batch}

        # ① NumPy
        if rank == 0:
            s, _ = timed(lambda: np.fft.fft2(
                np.random.randn(batch, N, N).astype(np.float32)),
                repeats=REPEATS, sync_cuda=False)
            entry["numpy"] = s["mean"] * 1000
        mpi.synchronize()

        # ② SciPy
        if HAS_SCIPY and rank == 0:
            s, _ = timed(lambda: scipy.fft.fft2(
                np.random.randn(batch, N, N).astype(np.float32)),
                repeats=REPEATS, sync_cuda=False)
            entry["scipy"] = s["mean"] * 1000
        mpi.synchronize()

        # ③ CuPy
        if HAS_CUPY and rank == 0:
            s, _ = timed(lambda: cp.fft.fft2(
                cp.random.randn(batch, N, N, dtype=cp.float32)),
                repeats=REPEATS)
            entry["cupy"] = s["mean"] * 1000
        mpi.synchronize()

        # ④ Dask
        if HAS_DASK and rank == 0:
            dask.config.set(scheduler="threads", num_workers=ws)
            x_np = np.random.randn(batch, N, N).astype(np.float32)
            s, _ = timed(lambda: da.fft.fft2(
                da.from_array(x_np, chunks=(1, N, N))).compute(),
                repeats=REPEATS, sync_cuda=False)
            entry["dask"] = s["mean"] * 1000
        mpi.synchronize()

        # ⑤ mpi4py + GPU 手写
        s, _ = timed_mpi(lambda: _raw_mpi_fft2d(comm, rank, ws, N, device),
                         mpi, repeats=REPEATS)
        if rank == 0:
            entry["mpi_gpu"] = s["mean"] * 1000

        # ⑦ Ours
        def _ours_fft(n=N, b=batch):
            x = torch.randn(b, n, n, device=device) if rank == 0 else None
            return distributed_fft2d(x, mpi, dist)
        s, _ = timed_mpi(_ours_fft, mpi, repeats=REPEATS)
        if rank == 0:
            entry["ours"] = s["mean"] * 1000

        R["fft"].append(entry)
        if rank == 0:
            parts = [f"N={N:>5}"]
            for k in ["numpy", "scipy", "cupy", "dask", "mpi_gpu", "ours"]:
                if k in entry:
                    parts.append(f"{k}={entry[k]:.1f}")
            log(" | ".join(parts), rank)
        mpi.synchronize()

    # ==================== C. Reduction (Sum) ====================
    log("--- C. Reduction (Sum) ---", rank)
    for numel in REDUCE_ELEMS:
        entry = {"numel": numel}

        # ① NumPy
        if rank == 0:
            s, _ = timed(lambda: np.sum(
                np.random.randn(numel).astype(np.float32)),
                repeats=REPEATS, sync_cuda=False)
            entry["numpy"] = s["mean"] * 1000
        mpi.synchronize()

        # ③ CuPy
        if HAS_CUPY and rank == 0:
            s, _ = timed(lambda: float(cp.sum(
                cp.random.randn(numel, dtype=cp.float32))),
                repeats=REPEATS)
            entry["cupy"] = s["mean"] * 1000
        mpi.synchronize()

        # ④ Dask
        if HAS_DASK and rank == 0:
            dask.config.set(scheduler="threads", num_workers=ws)
            x_np = np.random.randn(numel).astype(np.float32)
            ch = max(numel // ws, 1)
            s, _ = timed(lambda: float(
                da.from_array(x_np, chunks=ch).sum().compute()),
                repeats=REPEATS, sync_cuda=False)
            entry["dask"] = s["mean"] * 1000
        mpi.synchronize()

        # ⑤ mpi4py + GPU 手写
        s, _ = timed_mpi(lambda: _raw_mpi_reduce(comm, rank, ws, numel, device),
                         mpi, repeats=REPEATS)
        if rank == 0:
            entry["mpi_gpu"] = s["mean"] * 1000

        # ⑥ PyTorch Distributed
        if pt_ok:
            import torch.distributed as tdist
            def _pt_red(n=numel):
                x = torch.randn(n // ws, device=device)
                s_loc = x.sum()
                tdist.all_reduce(s_loc, op=tdist.ReduceOp.SUM)
                return float(s_loc.cpu())
            s, _ = timed_mpi(lambda: _pt_red(numel), mpi, repeats=REPEATS)
            if rank == 0:
                entry["pytorch_dist"] = s["mean"] * 1000

        # ⑦ Ours
        def _ours_red(n=numel):
            x = torch.randn(n, device=device) if rank == 0 else None
            return distributed_sum(x, mpi, dist)
        s, _ = timed_mpi(_ours_red, mpi, repeats=REPEATS)
        if rank == 0:
            entry["ours"] = s["mean"] * 1000

        R["reduce"].append(entry)
        if rank == 0:
            parts = [f"n={numel:>12,}"]
            for k in ["numpy", "cupy", "dask", "mpi_gpu", "pytorch_dist", "ours"]:
                if k in entry:
                    parts.append(f"{k}={entry[k]:.1f}")
            log(" | ".join(parts), rank)
        mpi.synchronize()

    cleanup_torch_distributed()
    if rank == 0:
        save_json("exp01_compute_performance", R)
        plot(R)


# ============================================================
#  绘图
# ============================================================

# 统一框架名称与颜色
FW = [("numpy",       "NumPy (CPU)"),
      ("scipy",        "SciPy (CPU)"),
      ("cupy",         "CuPy (1 GPU)"),
      ("dask",         "Dask (CPU dist.)"),
      ("mpi_gpu",      "mpi4py+GPU"),
      ("pytorch_dist", "PyTorch Dist."),
      ("ours",         "Ours")]


def plot(data=None):
    if data is None:
        data = load_json("exp01_compute_performance")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    tasks = [("matmul", "N", "MatMul (C = A × B)", "Matrix Size N"),
             ("fft",    "N", "FFT 2D",              "Grid Size N"),
             ("reduce", "numel", "Sum Reduction",    "Elements")]

    for ax, (task, xkey, title, xlabel) in zip(axes, tasks):
        records = data[task]
        xs = [r[xkey] for r in records]
        x_pos = np.arange(len(xs))
        avail = [(k, lbl) for k, lbl in FW if k in records[0]]
        nf = len(avail)
        w = 0.8 / max(nf, 1)
        for i, (k, lbl) in enumerate(avail):
            vals = [r.get(k, 0) for r in records]
            ax.bar(x_pos + (i - nf / 2 + 0.5) * w, vals, w,
                   label=lbl, color=COLORS[i % len(COLORS)], edgecolor="white")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{x:,}" if isinstance(x, int) and x > 9999
                            else str(x) for x in xs])
        ax.set_xlabel(xlabel); ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2); ax.set_yscale("log")

    fig.suptitle(f"Exp-1: Core Computation Performance ({data['num_gpus']} GPUs, {data['gpu']})",
                 fontsize=14, y=1.02)
    save_fig(fig, "exp01_compute_performance")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
