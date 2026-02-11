#!/usr/bin/env python
"""
实验 2：通信原语开销对比

对比分布式框架的底层通信原语性能（Broadcast · AllReduce）。
这是分布式计算的基石，决定了框架的通信效率上限。

对比方案（仅分布式框架）：
  ① mpi4py (CPU buffer)     — 传统 HPC CPU 通信
  ② mpi4py (GPU→CPU→GPU)    — HPC + GPU 间接传输
  ③ PyTorch Distributed NCCL — 深度学习框架 GPU Direct
  ④ Ours (MPI GPU 通信层)   — 本框架通信层

指标：延迟 (ms)、有效带宽 (GB/s)

运行: mpirun --oversubscribe --allow-run-as-root -n 4 python experiments/exp02_communication_overhead.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

from mpi4py import MPI
import torch, numpy as np

MSG_SIZES_KB = [1, 16, 256, 1024, 4096, 16384]
REPEATS = 10


def run():
    mpi, dist, gpu = setup_framework()
    warmup(mpi, dist)
    comm = MPI.COMM_WORLD
    rank, ws = mpi.get_rank(), mpi.get_size()
    device = torch.device(f"cuda:{mpi.get_gpu_id()}")
    pt_ok = init_torch_distributed(rank, ws, mpi.get_gpu_id())

    banner(f"实验2: 通信原语开销对比 ({ws} GPUs)", rank)
    R = {"gpu": gpu_name(), "num_gpus": ws,
         "broadcast": [], "allreduce": []}

    for op_name, rkey in [("Broadcast", "broadcast"), ("AllReduce", "allreduce")]:
        log(f"--- {op_name} ---", rank)
        for size_kb in MSG_SIZES_KB:
            n_elem = size_kb * 256  # float32 → 4B/elem → 256 elem/KB
            size_bytes = n_elem * 4
            entry = {"size_kb": size_kb, "n_elem": n_elem}

            # ① mpi4py CPU buffer
            buf = np.random.randn(n_elem).astype(np.float32)
            def _raw_cpu():
                if op_name == "Broadcast":
                    comm.Bcast(buf, root=0)
                else:
                    out = np.empty_like(buf)
                    comm.Allreduce(buf, out, op=MPI.SUM)
            mpi.synchronize()
            s, _ = timed_mpi(_raw_cpu, mpi, repeats=REPEATS)
            if rank == 0:
                entry["raw_cpu_ms"] = s["mean"] * 1000
                entry["raw_cpu_bw"] = size_bytes / max(s["mean"], 1e-12) / 1e9

            # ② mpi4py GPU→CPU→GPU
            t_gpu = torch.randn(n_elem, device=device)
            def _raw_gpu():
                np_buf = t_gpu.cpu().numpy()
                if op_name == "Broadcast":
                    comm.Bcast(np_buf, root=0)
                else:
                    out = np.empty_like(np_buf)
                    comm.Allreduce(np_buf, out, op=MPI.SUM)
                    np_buf[:] = out
                t_gpu.copy_(torch.from_numpy(np_buf))
            mpi.synchronize()
            s, _ = timed_mpi(_raw_gpu, mpi, repeats=REPEATS)
            if rank == 0:
                entry["raw_gpu_ms"] = s["mean"] * 1000
                entry["raw_gpu_bw"] = size_bytes / max(s["mean"], 1e-12) / 1e9

            # ③ PyTorch Distributed NCCL
            if pt_ok:
                import torch.distributed as tdist
                t_nccl = torch.randn(n_elem, device=device)
                def _pt():
                    if op_name == "Broadcast":
                        tdist.broadcast(t_nccl, src=0)
                    else:
                        tdist.all_reduce(t_nccl, op=tdist.ReduceOp.SUM)
                mpi.synchronize()
                s, _ = timed_mpi(_pt, mpi, repeats=REPEATS)
                if rank == 0:
                    entry["nccl_ms"] = s["mean"] * 1000
                    entry["nccl_bw"] = size_bytes / max(s["mean"], 1e-12) / 1e9

            # ④ Ours
            t_ours = torch.randn(n_elem, device=device)
            def _ours():
                if op_name == "Broadcast":
                    mpi.broadcast_tensor(t_ours)
                else:
                    mpi.allreduce_tensor(t_ours)
            mpi.synchronize()
            s, _ = timed_mpi(_ours, mpi, repeats=REPEATS)
            if rank == 0:
                entry["ours_ms"] = s["mean"] * 1000
                entry["ours_bw"] = size_bytes / max(s["mean"], 1e-12) / 1e9

            R[rkey].append(entry)
            if rank == 0:
                log(f"  {size_kb:>6} KB | "
                    f"cpu={entry.get('raw_cpu_ms',0):.3f} "
                    f"gpu={entry.get('raw_gpu_ms',0):.3f} "
                    f"nccl={entry.get('nccl_ms',0):.3f} "
                    f"ours={entry.get('ours_ms',0):.3f} ms", rank)
            mpi.synchronize()

    cleanup_torch_distributed()
    if rank == 0:
        save_json("exp02_communication_overhead", R)
        plot(R)


def plot(data=None):
    if data is None:
        data = load_json("exp02_communication_overhead")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fw = [("raw_cpu", "mpi4py (CPU)"),
          ("raw_gpu", "mpi4py (GPU→CPU→GPU)"),
          ("nccl",    "PyTorch NCCL"),
          ("ours",    "Ours")]

    for row, (op, title) in enumerate([("broadcast", "Broadcast"),
                                        ("allreduce", "AllReduce")]):
        records = data[op]
        sizes = [r["size_kb"] for r in records]

        # 左图: 延迟
        ax = axes[row][0]
        for i, (key, lbl) in enumerate(fw):
            vals = [r.get(f"{key}_ms", None) for r in records]
            if any(v is not None for v in vals):
                ax.plot(sizes, [v or 0 for v in vals],
                        marker=MARKERS[i], label=lbl, color=COLORS[i])
        ax.set_xlabel("Message Size (KB)"); ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{title} — Latency")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=9)

        # 右图: 带宽
        ax = axes[row][1]
        for i, (key, lbl) in enumerate(fw):
            vals = [r.get(f"{key}_bw", None) for r in records]
            if any(v is not None for v in vals):
                ax.plot(sizes, [v or 0 for v in vals],
                        marker=MARKERS[i], label=lbl, color=COLORS[i])
        ax.set_xlabel("Message Size (KB)"); ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title(f"{title} — Bandwidth")
        ax.set_xscale("log"); ax.legend(fontsize=9)

    fig.suptitle(f"Exp-2: Communication Overhead ({data['num_gpus']} GPUs)",
                 fontsize=14, y=1.02)
    save_fig(fig, "exp02_communication_overhead")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    else:
        run()
