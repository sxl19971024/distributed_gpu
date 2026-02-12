#!/usr/bin/env python3
"""
跨框架性能对比实验 (自适应规模版)

自动检测可用GPU和显存，以2倍递增的规模运行实验，
超过可用显存时自动终止当前实验进入下一个。

运行方式 (不要用 mpirun，脚本内部管理多进程):
  python experiments/benchmark_comparison.py                # 自动检测GPU，全部实验
  python experiments/benchmark_comparison.py --gpus 4       # 指定GPU数
  python experiments/benchmark_comparison.py --exp matmul   # 只跑某个实验
  python experiments/benchmark_comparison.py --list         # 查看可用实验

对比框架:
  1. PyTorch Distributed (NCCL)  — GPU直通通信基线
  2. PETSc (petsc4py)            — 科学计算金标准
  3. Dask-CUDA + CuPy            — Python通用分布式GPU框架
  4. 本框架 (distributed_gpu)    — MPI+GPU

每个实验的数据规模按2倍递增 (如 1024→2048→4096→...)，
直到预估显存超出GPU可用显存为止。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

REPEATS = 5       # 每个规模重复测量次数
WARMUP = 2        # 预热次数
PER_SIZE_TIMEOUT = 120  # 单个规模的最大耗时 (秒)，超时则跳过更大规模


# ═══════════════════════════════════════════════════════════════
#  通用工具
# ═══════════════════════════════════════════════════════════════

def detect_free_gpus(min_free_mb: int = 10000) -> List[int]:
    """检测空闲GPU (逻辑ID)"""
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            f, _ = torch.cuda.mem_get_info(i)
            if f / 1024**2 >= min_free_mb:
                free.append(i)
        except Exception:
            pass
    return free


def get_physical_gpu_ids(logical_ids: List[int]) -> List[int]:
    """逻辑ID → 物理ID (考虑 CUDA_VISIBLE_DEVICES)"""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd:
        physical = [int(x.strip()) for x in cvd.split(",")]
        return [physical[i] for i in logical_ids if i < len(physical)]
    return logical_ids


def get_min_free_gb(gpu_ids: List[int]) -> float:
    """获取指定GPU中最小可用显存 (GB)"""
    min_free = float("inf")
    for gid in gpu_ids:
        try:
            f, _ = torch.cuda.mem_get_info(gid)
            min_free = min(min_free, f / 1024**3)
        except Exception:
            pass
    return min_free


def make_doubling_sizes(start: int, max_mem_gb: float, mem_per_element_bytes: int,
                        num_elements_fn) -> List[int]:
    """生成2倍递增的规模列表，直到超出显存
    Args:
        start: 起始规模
        max_mem_gb: 可用显存上限 (GB)
        mem_per_element_bytes: 每个元素字节数 (float32=4)
        num_elements_fn: 给定规模N返回总元素数的函数
    """
    sizes = []
    N = start
    safety = 0.8  # 留20%安全边际
    while True:
        mem_gb = num_elements_fn(N) * mem_per_element_bytes / 1024**3
        if mem_gb > max_mem_gb * safety:
            break
        sizes.append(N)
        N *= 2
    return sizes


def benchmark_fn(fn, repeats=REPEATS, warmup=WARMUP, device=None):
    """计时工具"""
    for _ in range(warmup):
        fn()
        if device is not None:
            torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        if device is not None:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if device is not None:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  💾 保存: {filepath}")


def make_env(gpu_ids):
    """构建子进程环境变量"""
    phys_ids = get_physical_gpu_ids(gpu_ids)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in phys_ids)
    return env, phys_ids


def run_subprocess(cmd, env, timeout=600, label=""):
    """运行子进程并捕获输出"""
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            print(f"  ⚠ {label} 失败 (exit={proc.returncode})")
            stderr = proc.stderr[-300:] if proc.stderr else ""
            if stderr:
                print(f"     {stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠ {label} 超时 ({timeout}s)，跳过")
        return False


def load_tmp_json(filepath):
    """加载并删除临时JSON"""
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
        os.remove(filepath)
        return data
    return []


def remove_if_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)


# ═══════════════════════════════════════════════════════════════
#  实验 1: 矩阵乘法 — 本框架 vs NCCL vs 单GPU
# ═══════════════════════════════════════════════════════════════

def exp_matmul(gpu_ids, output_dir):
    print("\n" + "=" * 70)
    print("  实验: 矩阵乘法 — 本框架 vs PyTorch Distributed (NCCL) vs 单GPU")
    print("  规模: NxN, N从1024开始2倍递增, 显存不足时自动停止")
    print("=" * 70)

    world_size = len(gpu_ids)
    dev = torch.device(f"cuda:{gpu_ids[0]}")
    free_gb = get_min_free_gb(gpu_ids)
    # matmul: A[N,N] + B[N,N] + C[N,N] = 3*N^2 elements
    sizes = make_doubling_sizes(1024, free_gb, 4, lambda N: 3 * N * N)
    print(f"  GPU: {gpu_ids} ({world_size}卡) | 可用显存: {free_gb:.1f}GB")
    print(f"  规模序列: {sizes}")
    if not sizes:
        print("  ⚠ 显存不足，跳过"); return

    # ── 单GPU ──
    print(f"\n  ── 单GPU基线 ──")
    from torch.cuda import empty_cache
    single = []
    for N in sizes:
        A = torch.randn(N, N, device=dev); B = torch.randn(N, N, device=dev)
        m, s = benchmark_fn(lambda: torch.matmul(A, B), device=dev)
        single.append({"size": N, "mean_ms": round(m, 2), "std_ms": round(s, 2)})
        gflops = 2*N**3 / (m/1000) / 1e9
        print(f"    {N:>6}x{N}: {m:>10.2f} ± {s:.2f} ms  ({gflops:.0f} GFLOPS)")
        del A, B; clean()

    # ── NCCL ──
    print(f"\n  ── PyTorch Distributed (NCCL) {world_size}卡 ──")
    nccl_file = os.path.join(RESULTS_ROOT, "_tmp_nccl_mm.json")
    nccl_script = os.path.join(RESULTS_ROOT, "_tmp_nccl_mm.py")
    with open(nccl_script, "w") as f:
        f.write(f'''
import os,json,time; import numpy as np; import torch,torch.distributed
rank=int(os.environ["RANK"]); ws=int(os.environ["WORLD_SIZE"])
lr=int(os.environ.get("LOCAL_RANK",rank)); torch.cuda.set_device(lr)
torch.distributed.init_process_group("nccl")
sizes={sizes}; rep={REPEATS}; wu={WARMUP}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
a=torch.randn(1000,1000,device=f"cuda:{{lr}}")
for _ in range(10): torch.matmul(a,a)
torch.cuda.synchronize(lr); del a; torch.cuda.empty_cache()
for N in sizes:
    sz_t0=time.perf_counter()
    ck=N//ws
    Al=torch.randn(ck,N,device=f"cuda:{{lr}}"); B=torch.randn(N,N,device=f"cuda:{{lr}}")
    for _ in range(wu):
        torch.distributed.broadcast(B,src=0); C=torch.matmul(Al,B)
        cl=[torch.empty_like(C) for _ in range(ws)] if rank==0 else None
        torch.distributed.gather(C,gather_list=cl,dst=0); torch.cuda.synchronize(lr)
    ts=[]
    for _ in range(rep):
        torch.cuda.synchronize(lr); torch.distributed.barrier(); t0=time.perf_counter()
        torch.distributed.broadcast(B,src=0); C=torch.matmul(Al,B)
        cl=[torch.empty_like(C) for _ in range(ws)] if rank==0 else None
        torch.distributed.gather(C,gather_list=cl,dst=0)
        torch.cuda.synchronize(lr); torch.distributed.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000)
    if rank==0: results.append({{"size":N,"mean_ms":round(float(np.mean(ts)),2),"std_ms":round(float(np.std(ts)),2)}})
    del Al,B; torch.cuda.empty_cache()
    if time.perf_counter()-sz_t0>TLIMIT: break
if rank==0:
    with open("{nccl_file}","w") as ff: json.dump(results,ff)
torch.distributed.destroy_process_group()
''')
    env, phys = make_env(gpu_ids)
    cmd = ["torchrun", f"--nproc_per_node={world_size}", "--master_port=29500", nccl_script]
    if run_subprocess(cmd, env, label="NCCL MatMul"):
        nccl = load_tmp_json(nccl_file)
        for r in nccl:
            print(f"    {r['size']:>6}x{r['size']}: {r['mean_ms']:>10.2f} ± {r['std_ms']:.2f} ms")
    else:
        nccl = []
    remove_if_exists(nccl_script)

    # ── 本框架 ──
    print(f"\n  ── 本框架 (mpi4py) {world_size}卡 ──")
    ours_file = os.path.join(RESULTS_ROOT, "_tmp_ours_mm.json")
    ours_script = os.path.join(RESULTS_ROOT, "_tmp_ours_mm.py")
    with open(ours_script, "w") as f:
        f.write(f'''
import sys,os,json,time,gc; sys.path.insert(0,"{PROJECT_ROOT}")
import numpy as np; import torch
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.algorithms.matrix_ops import distributed_matmul
mpi=MPIManager(); dist=TensorDistributor(mpi); gid=mpi.get_gpu_id()
sizes={sizes}; rep={REPEATS}; wu={WARMUP}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
a=torch.randn(1000,1000,device=f"cuda:{{gid}}")
for _ in range(10): torch.matmul(a,a)
torch.cuda.synchronize(gid); del a; torch.cuda.empty_cache()
for N in sizes:
    sz_t0=time.perf_counter()
    A=B=None
    if mpi.is_master_process():
        A=torch.randn(N,N,device=f"cuda:{{gid}}"); B=torch.randn(N,N,device=f"cuda:{{gid}}")
    for _ in range(wu):
        mpi.barrier(); distributed_matmul(A,B,mpi,dist); torch.cuda.synchronize(gid); mpi.barrier()
        gc.collect(); torch.cuda.empty_cache()
    ts=[]
    for _ in range(rep):
        mpi.barrier(); torch.cuda.synchronize(gid); t0=time.perf_counter()
        distributed_matmul(A,B,mpi,dist)
        torch.cuda.synchronize(gid); mpi.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000); gc.collect(); torch.cuda.empty_cache()
    if mpi.is_master_process():
        results.append({{"size":N,"mean_ms":round(float(np.mean(ts)),2),"std_ms":round(float(np.std(ts)),2)}})
        del A,B
    gc.collect(); torch.cuda.empty_cache(); mpi.barrier()
    skip=time.perf_counter()-sz_t0>TLIMIT
    skip=mpi.broadcast(skip if mpi.is_master_process() else None)
    if skip: break
if mpi.is_master_process():
    with open("{ours_file}","w") as ff: json.dump(results,ff)
''')
    cmd = ["mpirun", "-n", str(world_size), "--allow-run-as-root", "--oversubscribe",
           sys.executable, ours_script]
    if run_subprocess(cmd, env, label="本框架 MatMul"):
        ours = load_tmp_json(ours_file)
        for r in ours:
            print(f"    {r['size']:>6}x{r['size']}: {r['mean_ms']:>10.2f} ± {r['std_ms']:.2f} ms")
    else:
        ours = []
    remove_if_exists(ours_script)

    # ── 汇总 ──
    comp = []
    for s, n, o in zip(single, nccl or single, ours or single):
        sp_n = s["mean_ms"]/n["mean_ms"] if n["mean_ms"]>0 else 0
        sp_o = s["mean_ms"]/o["mean_ms"] if o["mean_ms"]>0 else 0
        comp.append({"size":s["size"], "data_gb": round(3*s["size"]**2*4/1024**3,3),
                     "single_ms":s["mean_ms"], "nccl_ms":n["mean_ms"], "ours_ms":o["mean_ms"],
                     "nccl_speedup":round(sp_n,3), "ours_speedup":round(sp_o,3),
                     "ours_div_nccl":round(o["mean_ms"]/n["mean_ms"],2) if n["mean_ms"]>0 else 0})
    data = {"experiment":"矩阵乘法跨框架对比","gpu_count":world_size,
            "single_gpu":single,"torch_distributed_nccl":nccl,
            "distributed_gpu_framework":ours,"comparison":comp}

    print(f"\n  {'Size':>6} {'数据量':>8} {'单GPU':>10} {'NCCL':>10} {'本框架':>10} {'NCCL加速':>9} {'本框架加速':>9} {'本/NCCL':>8}")
    for c in comp:
        print(f"  {c['size']:>6} {c['data_gb']:>6.3f}GB {c['single_ms']:>8.2f}ms "
              f"{c['nccl_ms']:>8.2f}ms {c['ours_ms']:>8.2f}ms "
              f"{c['nccl_speedup']:>8.3f}x {c['ours_speedup']:>8.3f}x {c['ours_div_nccl']:>7.2f}x")
    save_json(data, os.path.join(output_dir, "comparison_matmul.json"))


# ═══════════════════════════════════════════════════════════════
#  实验 2: AllReduce — 本框架 vs NCCL
# ═══════════════════════════════════════════════════════════════

def exp_allreduce(gpu_ids, output_dir):
    print("\n" + "=" * 70)
    print("  实验: AllReduce — 本框架 vs PyTorch Distributed (NCCL)")
    print("  规模: 从1MB开始2倍递增, 显存不足时自动停止")
    print("=" * 70)

    world_size = len(gpu_ids)
    free_gb = get_min_free_gb(gpu_ids)
    # allreduce: 1个张量 + 1个接收缓冲 ≈ 2x
    sizes_mb = []
    mb = 1
    while mb * 2 / 1024 < free_gb * 0.8:
        sizes_mb.append(mb)
        mb *= 2
    print(f"  GPU: {gpu_ids} ({world_size}卡) | 可用显存: {free_gb:.1f}GB")
    print(f"  规模序列 (MB): {sizes_mb}")
    if not sizes_mb:
        print("  ⚠ 显存不足，跳过"); return

    rep = 10  # AllReduce 波动小，多测几次

    # ── NCCL ──
    nccl_file = os.path.join(RESULTS_ROOT, "_tmp_nccl_ar.json")
    nccl_script = os.path.join(RESULTS_ROOT, "_tmp_nccl_ar.py")
    with open(nccl_script, "w") as f:
        f.write(f'''
import os,json,time; import numpy as np; import torch,torch.distributed
rank=int(os.environ["RANK"]); lr=int(os.environ.get("LOCAL_RANK",rank))
torch.cuda.set_device(lr); torch.distributed.init_process_group("nccl")
sizes_mb={sizes_mb}; rep={rep}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
for mb in sizes_mb:
    sz_t0=time.perf_counter()
    n=mb*1024*1024//4; d=torch.randn(n,device=f"cuda:{{lr}}")
    for _ in range(3): torch.distributed.all_reduce(d); torch.cuda.synchronize(lr)
    ts=[]
    for _ in range(rep):
        torch.cuda.synchronize(lr); torch.distributed.barrier(); t0=time.perf_counter()
        torch.distributed.all_reduce(d)
        torch.cuda.synchronize(lr); torch.distributed.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000)
    if rank==0: results.append({{"size_mb":mb,"mean_ms":round(float(np.mean(ts)),3),"std_ms":round(float(np.std(ts)),3)}})
    del d; torch.cuda.empty_cache()
    if time.perf_counter()-sz_t0>TLIMIT: break
if rank==0:
    with open("{nccl_file}","w") as ff: json.dump(results,ff)
torch.distributed.destroy_process_group()
''')
    env, _ = make_env(gpu_ids)
    print(f"\n  ── NCCL AllReduce {world_size}卡 ──")
    run_subprocess(["torchrun",f"--nproc_per_node={world_size}","--master_port=29501",nccl_script],
                   env, label="NCCL AllReduce")
    nccl = load_tmp_json(nccl_file)
    for r in nccl: print(f"    {r['size_mb']:>6}MB: {r['mean_ms']:>10.3f} ± {r['std_ms']:.3f} ms")
    remove_if_exists(nccl_script)

    # ── 本框架 ──
    ours_file = os.path.join(RESULTS_ROOT, "_tmp_ours_ar.json")
    ours_script = os.path.join(RESULTS_ROOT, "_tmp_ours_ar.py")
    with open(ours_script, "w") as f:
        f.write(f'''
import sys,os,json,time; sys.path.insert(0,"{PROJECT_ROOT}")
import numpy as np; import torch
from distributed_gpu.mpi_manager import MPIManager
mpi=MPIManager(); gid=mpi.get_gpu_id()
sizes_mb={sizes_mb}; rep={rep}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
for mb in sizes_mb:
    sz_t0=time.perf_counter()
    n=mb*1024*1024//4; d=torch.randn(n,device=f"cuda:{{gid}}")
    for _ in range(3): mpi.allreduce_tensor(d); torch.cuda.synchronize(gid)
    ts=[]
    for _ in range(rep):
        torch.cuda.synchronize(gid); mpi.barrier(); t0=time.perf_counter()
        mpi.allreduce_tensor(d)
        torch.cuda.synchronize(gid); mpi.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000)
    if mpi.is_master_process():
        results.append({{"size_mb":mb,"mean_ms":round(float(np.mean(ts)),3),"std_ms":round(float(np.std(ts)),3)}})
    del d; torch.cuda.empty_cache()
    skip=time.perf_counter()-sz_t0>TLIMIT
    skip=mpi.broadcast(skip if mpi.is_master_process() else None)
    if skip: break
if mpi.is_master_process():
    with open("{ours_file}","w") as ff: json.dump(results,ff)
''')
    print(f"\n  ── 本框架 AllReduce {world_size}卡 ──")
    run_subprocess(["mpirun","-n",str(world_size),"--allow-run-as-root","--oversubscribe",
                    sys.executable,ours_script], env, label="本框架 AllReduce")
    ours = load_tmp_json(ours_file)
    for r in ours: print(f"    {r['size_mb']:>6}MB: {r['mean_ms']:>10.3f} ± {r['std_ms']:.3f} ms")
    remove_if_exists(ours_script)

    # ── 汇总 ──
    comp = []
    for n, o in zip(nccl, ours):
        ratio = o["mean_ms"]/n["mean_ms"] if n["mean_ms"]>0 else 0
        bw_n = n["size_mb"]/n["mean_ms"]*1000/1000 if n["mean_ms"]>0 else 0
        bw_o = o["size_mb"]/o["mean_ms"]*1000/1000 if o["mean_ms"]>0 else 0
        comp.append({"size_mb":n["size_mb"],"nccl_ms":n["mean_ms"],"ours_ms":o["mean_ms"],
                     "ours_div_nccl":round(ratio,2),"nccl_bw_gbps":round(bw_n,2),"ours_bw_gbps":round(bw_o,2)})
    data = {"experiment":"AllReduce跨框架对比","gpu_count":world_size,
            "torch_distributed_nccl":nccl,"distributed_gpu_framework":ours,"comparison":comp}

    print(f"\n  {'Size':>6} {'NCCL':>10} {'本框架':>10} {'本/NCCL':>8} {'NCCL BW':>10} {'本框架BW':>10}")
    for c in comp:
        print(f"  {c['size_mb']:>4}MB {c['nccl_ms']:>8.3f}ms {c['ours_ms']:>8.3f}ms "
              f"{c['ours_div_nccl']:>7.2f}x {c['nccl_bw_gbps']:>8.2f}GB/s {c['ours_bw_gbps']:>8.2f}GB/s")
    save_json(data, os.path.join(output_dir, "comparison_allreduce.json"))


# ═══════════════════════════════════════════════════════════════
#  实验 3: Jacobi 迭代 — 本框架 GPU vs PETSc/NumPy CPU
# ═══════════════════════════════════════════════════════════════

def exp_stencil(gpu_ids, output_dir):
    print("\n" + "=" * 70)
    print("  实验: Jacobi 迭代 — 本框架 GPU vs PETSc/NumPy CPU")
    print("  规模: 网格从128开始2倍递增, 迭代100/500次, 显存不足时自动停止")
    print("=" * 70)

    world_size = len(gpu_ids)
    free_gb = get_min_free_gb(gpu_ids)
    # jacobi: grid + rhs + padded + dx2_f ≈ 4*N^2
    grid_sizes = make_doubling_sizes(128, free_gb, 4, lambda N: 4 * N * N)
    iters_list = [100, 500]
    configs = [(g, it) for g in grid_sizes for it in iters_list]
    print(f"  GPU: {gpu_ids} ({world_size}卡) | 可用显存: {free_gb:.1f}GB")
    print(f"  网格规模: {grid_sizes}, 迭代: {iters_list}")
    if not configs:
        print("  ⚠ 显存不足，跳过"); return

    # ── CPU 基线 (PETSc or NumPy) ──
    print(f"\n  ── CPU 基线 (NumPy Jacobi) ──")
    cpu_results = []
    for grid_sz, iters in configs:
        u = np.random.randn(grid_sz, grid_sz).astype(np.float32)
        f_rhs = np.random.randn(grid_sz, grid_sz).astype(np.float32)
        def numpy_jacobi():
            uu = u.copy()
            for _ in range(iters):
                uu[1:-1,1:-1] = (uu[:-2,1:-1]+uu[2:,1:-1]+uu[1:-1,:-2]+uu[1:-1,2:]-f_rhs[1:-1,1:-1])/4.0
        m, s = benchmark_fn(numpy_jacobi, repeats=3, warmup=1)
        cpu_results.append({"grid":grid_sz,"iterations":iters,"mean_ms":round(m,2),"std_ms":round(s,2)})
        print(f"    {grid_sz:>6}x{grid_sz} x{iters:>3}iter: {m:>10.2f} ± {s:.2f} ms")

    # PETSc 如果可用
    petsc_available = False
    try:
        import petsc4py
        petsc_available = True
        print(f"\n  ── PETSc (petsc4py {petsc4py.__version__}) Jacobi CPU ──")
        petsc4py.init(sys.argv[:1])
        from petsc4py import PETSc as _P
        petsc_results = []
        for grid_sz, iters in configs:
            da = _P.DMDA().create(dim=2, sizes=[grid_sz,grid_sz], stencil_width=1,
                                  stencil_type=_P.DMDA.StencilType.STAR)
            u_v=da.createGlobalVec(); f_v=da.createGlobalVec()
            ua=da.getVecArray(u_v); fa=da.getVecArray(f_v)
            rng=np.random.default_rng(42); ua[:]=rng.standard_normal(ua.shape); fa[:]=rng.standard_normal(fa.shape)
            def petsc_jacobi():
                mat=da.createMatrix(); mat.setType("aij"); mat.setFromOptions(); mat.setUp()
                (xs,xe),(ys,ye)=da.getRanges()
                for j in range(ys,ye):
                    for i in range(xs,xe):
                        row=j*grid_sz+i; mat.setValue(row,row,-4.0)
                        if i>0: mat.setValue(row,row-1,1.0)
                        if i<grid_sz-1: mat.setValue(row,row+1,1.0)
                        if j>0: mat.setValue(row,row-grid_sz,1.0)
                        if j<grid_sz-1: mat.setValue(row,row+grid_sz,1.0)
                mat.assemblyBegin(); mat.assemblyEnd()
                ksp=_P.KSP().create(); ksp.setOperators(mat)
                ksp.setType("richardson"); ksp.getPC().setType("jacobi")
                ksp.setTolerances(rtol=1e-10,max_it=iters); ksp.setFromOptions()
                ksp.solve(f_v,u_v); ksp.destroy(); mat.destroy()
            m, s = benchmark_fn(petsc_jacobi, repeats=3, warmup=1)
            petsc_results.append({"grid":grid_sz,"iterations":iters,"mean_ms":round(m,2),"std_ms":round(s,2)})
            print(f"    {grid_sz:>6}x{grid_sz} x{iters:>3}iter: {m:>10.2f} ± {s:.2f} ms")
            da.destroy(); u_v.destroy(); f_v.destroy()
    except Exception as e:
        print(f"  ⚠ PETSc 不可用: {e}")
        petsc_results = []

    # ── 本框架 GPU ──
    ours_file = os.path.join(RESULTS_ROOT, "_tmp_ours_jac.json")
    ours_script = os.path.join(RESULTS_ROOT, "_tmp_ours_jac.py")
    with open(ours_script, "w") as f:
        f.write(f'''
import sys,os,json,time,gc; sys.path.insert(0,"{PROJECT_ROOT}")
import numpy as np; import torch
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.algorithms.stencil import distributed_jacobi_2d
mpi=MPIManager(); dist=TensorDistributor(mpi); gid=mpi.get_gpu_id()
configs={configs}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
for gsz,iters in configs:
    sz_t0=time.perf_counter()
    g=r=None
    if mpi.is_master_process():
        g=torch.randn(gsz,gsz,device=f"cuda:{{gid}}"); r=torch.randn(gsz,gsz,device=f"cuda:{{gid}}")
    for _ in range(1):
        mpi.barrier(); distributed_jacobi_2d(g,r,mpi,dist,iterations=iters,tol=1e-10)
        torch.cuda.synchronize(gid); mpi.barrier(); gc.collect(); torch.cuda.empty_cache()
    ts=[]
    for _ in range(3):
        mpi.barrier(); torch.cuda.synchronize(gid); t0=time.perf_counter()
        distributed_jacobi_2d(g,r,mpi,dist,iterations=iters,tol=1e-10)
        torch.cuda.synchronize(gid); mpi.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000); gc.collect(); torch.cuda.empty_cache()
    if mpi.is_master_process():
        results.append({{"grid":gsz,"iterations":iters,"mean_ms":round(float(np.mean(ts)),2),"std_ms":round(float(np.std(ts)),2)}})
        del g,r
    gc.collect(); torch.cuda.empty_cache(); mpi.barrier()
    skip=time.perf_counter()-sz_t0>TLIMIT
    skip=mpi.broadcast(skip if mpi.is_master_process() else None)
    if skip: break
if mpi.is_master_process():
    with open("{ours_file}","w") as ff: json.dump(results,ff)
''')
    env, _ = make_env(gpu_ids)
    print(f"\n  ── 本框架 分布式Jacobi {world_size}卡 GPU ──")
    run_subprocess(["mpirun","-n",str(world_size),"--allow-run-as-root","--oversubscribe",
                    sys.executable,ours_script], env, label="本框架 Jacobi")
    ours = load_tmp_json(ours_file)
    for r in ours: print(f"    {r['grid']:>6}x{r['grid']} x{r['iterations']:>3}iter: {r['mean_ms']:>10.2f} ± {r['std_ms']:.2f} ms")
    remove_if_exists(ours_script)

    # ── 汇总 ──
    comp = []
    cpu_ref = petsc_results if petsc_results else cpu_results
    cpu_label = "petsc" if petsc_results else "numpy"
    for c, o in zip(cpu_ref, ours):
        sp = c["mean_ms"]/o["mean_ms"] if o["mean_ms"]>0 else 0
        comp.append({"grid":c["grid"],"iterations":c["iterations"],
                     f"{cpu_label}_cpu_ms":c["mean_ms"],"gpu_dist_ms":o["mean_ms"],
                     "gpu_speedup_vs_cpu":round(sp,2)})
    data = {"experiment":"Jacobi迭代跨框架对比","gpu_count":world_size,
            "cpu_baseline":cpu_ref,"cpu_backend":cpu_label,
            "petsc_results":petsc_results,"numpy_results":cpu_results,
            "distributed_gpu_framework":ours,"comparison":comp}

    print(f"\n  {'Grid':>6} {'Iter':>5} {cpu_label.upper()+' CPU':>12} {'本框架GPU':>12} {'GPU/CPU加速':>12}")
    for c in comp:
        cpu_k = f"{cpu_label}_cpu_ms"
        print(f"  {c['grid']:>6} {c['iterations']:>5} {c[cpu_k]:>10.2f}ms {c['gpu_dist_ms']:>10.2f}ms {c['gpu_speedup_vs_cpu']:>10.2f}x")
    save_json(data, os.path.join(output_dir, "comparison_jacobi.json"))


# ═══════════════════════════════════════════════════════════════
#  实验 4: FFT2D — 本框架 vs Dask-CUDA vs 单GPU
# ═══════════════════════════════════════════════════════════════

def exp_fft(gpu_ids, output_dir):
    print("\n" + "=" * 70)
    print("  实验: FFT2D — 本框架 vs Dask-CUDA + CuPy vs 单GPU")
    print("  规模: batch从4开始2倍递增, grid从256开始2倍递增, 显存不足时自动停止")
    print("=" * 70)

    world_size = len(gpu_ids)
    dev = torch.device(f"cuda:{gpu_ids[0]}")
    free_gb = get_min_free_gb(gpu_ids)
    # fft2d: input(complex128 output doubles size) ≈ batch*grid*grid * 16 bytes
    configs = []
    for grid in [256, 512, 1024, 2048, 4096]:
        batch = 4
        while batch * grid * grid * 16 / 1024**3 < free_gb * 0.6:
            configs.append((batch, grid))
            batch *= 2
    print(f"  GPU: {gpu_ids} ({world_size}卡) | 可用显存: {free_gb:.1f}GB")
    print(f"  配置数: {len(configs)} 组")
    for b, g in configs:
        print(f"    batch={b:>4}, grid={g:>5}  ({b*g*g*4/1024**2:.1f} MB)")
    if not configs:
        print("  ⚠ 显存不足，跳过"); return

    # ── 单GPU ──
    print(f"\n  ── 单GPU PyTorch FFT2D ──")
    single = []
    for batch, grid in configs:
        d = torch.randn(batch, grid, grid, device=dev)
        m, s = benchmark_fn(lambda: torch.fft.fft2(d), device=dev)
        single.append({"batch":batch,"grid":grid,"mean_ms":round(m,2),"std_ms":round(s,2)})
        print(f"    batch={batch:>4} grid={grid:>5}: {m:>10.2f} ± {s:.2f} ms")
        del d; clean()

    # ── Dask-CUDA ──
    print(f"\n  ── Dask-CUDA + CuPy FFT2D {world_size}卡 ──")
    dask_results = []
    try:
        import cupy as cp
        import dask.array as da
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        phys = get_physical_gpu_ids(gpu_ids)
        cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=phys, n_workers=world_size)
        client = Client(cluster)
        for batch, grid in configs:
            x = np.random.randn(batch, grid, grid).astype(np.float32)
            x_da = da.from_array(x, chunks=(max(1, batch//world_size), grid, grid))
            def dask_fft(): da.fft.fft2(x_da).compute()
            for _ in range(2): dask_fft()
            ts = []
            for _ in range(REPEATS):
                t0=time.perf_counter(); dask_fft(); t1=time.perf_counter()
                ts.append((t1-t0)*1000)
            m,s = float(np.mean(ts)), float(np.std(ts))
            dask_results.append({"batch":batch,"grid":grid,"mean_ms":round(m,2),"std_ms":round(s,2)})
            print(f"    batch={batch:>4} grid={grid:>5}: {m:>10.2f} ± {s:.2f} ms")
        client.close(); cluster.close()
    except Exception as e:
        print(f"  ⚠ Dask-CUDA 失败: {e}")
        print(f"  [回退: CuPy 单GPU FFT2D]")
        try:
            import cupy as cp
            for batch, grid in configs:
                d = cp.random.randn(batch, grid, grid, dtype=cp.float32)
                def cupy_fft(): cp.fft.fft2(d); cp.cuda.Stream.null.synchronize()
                m, s = benchmark_fn(cupy_fft)
                dask_results.append({"batch":batch,"grid":grid,"mean_ms":round(m,2),"std_ms":round(s,2),"backend":"cupy_single"})
                print(f"    batch={batch:>4} grid={grid:>5}: {m:>10.2f} ± {s:.2f} ms")
                del d
        except Exception as e2:
            print(f"  ⚠ CuPy 也失败: {e2}")

    # ── 本框架 ──
    ours_file = os.path.join(RESULTS_ROOT, "_tmp_ours_fft.json")
    ours_script = os.path.join(RESULTS_ROOT, "_tmp_ours_fft.py")
    with open(ours_script, "w") as f:
        f.write(f'''
import sys,os,json,time,gc; sys.path.insert(0,"{PROJECT_ROOT}")
import numpy as np; import torch
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.algorithms.fft import distributed_fft2d
mpi=MPIManager(); dist=TensorDistributor(mpi); gid=mpi.get_gpu_id()
configs={configs}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
for batch,grid in configs:
    sz_t0=time.perf_counter()
    d=None
    if mpi.is_master_process(): d=torch.randn(batch,grid,grid,device=f"cuda:{{gid}}")
    for _ in range(2):
        mpi.barrier(); distributed_fft2d(d,mpi,dist); torch.cuda.synchronize(gid); mpi.barrier()
        gc.collect(); torch.cuda.empty_cache()
    ts=[]
    for _ in range({REPEATS}):
        mpi.barrier(); torch.cuda.synchronize(gid); t0=time.perf_counter()
        distributed_fft2d(d,mpi,dist)
        torch.cuda.synchronize(gid); mpi.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000); gc.collect(); torch.cuda.empty_cache()
    if mpi.is_master_process():
        results.append({{"batch":batch,"grid":grid,"mean_ms":round(float(np.mean(ts)),2),"std_ms":round(float(np.std(ts)),2)}})
        del d
    gc.collect(); torch.cuda.empty_cache(); mpi.barrier()
    skip=time.perf_counter()-sz_t0>TLIMIT
    skip=mpi.broadcast(skip if mpi.is_master_process() else None)
    if skip: break
if mpi.is_master_process():
    with open("{ours_file}","w") as ff: json.dump(results,ff)
''')
    env, _ = make_env(gpu_ids)
    print(f"\n  ── 本框架 分布式FFT2D {world_size}卡 ──")
    run_subprocess(["mpirun","-n",str(world_size),"--allow-run-as-root","--oversubscribe",
                    sys.executable,ours_script], env, label="本框架 FFT2D")
    ours = load_tmp_json(ours_file)
    for r in ours: print(f"    batch={r['batch']:>4} grid={r['grid']:>5}: {r['mean_ms']:>10.2f} ± {r['std_ms']:.2f} ms")
    remove_if_exists(ours_script)

    # ── 汇总 ──
    comp = []
    for i, (s, o) in enumerate(zip(single, ours)):
        d = dask_results[i] if i < len(dask_results) else {"mean_ms":0}
        sp_o = s["mean_ms"]/o["mean_ms"] if o["mean_ms"]>0 else 0
        sp_d = s["mean_ms"]/d["mean_ms"] if d["mean_ms"]>0 else 0
        comp.append({"batch":s["batch"],"grid":s["grid"],"data_mb":round(s["batch"]*s["grid"]**2*4/1024**2,1),
                     "single_ms":s["mean_ms"],"dask_ms":d["mean_ms"],"ours_ms":o["mean_ms"],
                     "ours_vs_single":round(sp_o,3),"dask_vs_single":round(sp_d,3)})
    data = {"experiment":"FFT2D跨框架对比","gpu_count":world_size,
            "single_gpu":single,"dask_cuda":dask_results,
            "distributed_gpu_framework":ours,"comparison":comp}

    print(f"\n  {'Batch':>5} {'Grid':>5} {'Data':>7} {'单GPU':>10} {'Dask':>10} {'本框架':>10} {'本/单':>8} {'Dask/单':>8}")
    for c in comp:
        print(f"  {c['batch']:>5} {c['grid']:>5} {c['data_mb']:>5.1f}MB {c['single_ms']:>8.2f}ms "
              f"{c['dask_ms']:>8.2f}ms {c['ours_ms']:>8.2f}ms {c['ours_vs_single']:>7.3f}x {c['dask_vs_single']:>7.3f}x")
    save_json(data, os.path.join(output_dir, "comparison_fft.json"))


# ═══════════════════════════════════════════════════════════════
#  实验 5: 归约 (Sum) — 本框架 vs NCCL vs 单GPU
# ═══════════════════════════════════════════════════════════════

def exp_reduction(gpu_ids, output_dir):
    print("\n" + "=" * 70)
    print("  实验: 全局求和 (Sum) — 本框架 vs NCCL vs 单GPU")
    print("  规模: NxN, N从1024开始2倍递增, 显存不足时自动停止")
    print("=" * 70)

    world_size = len(gpu_ids)
    dev = torch.device(f"cuda:{gpu_ids[0]}")
    free_gb = get_min_free_gb(gpu_ids)
    sizes = make_doubling_sizes(1024, free_gb, 4, lambda N: N * N * 2)
    print(f"  GPU: {gpu_ids} ({world_size}卡) | 可用显存: {free_gb:.1f}GB")
    print(f"  规模序列: {sizes}")
    if not sizes:
        print("  ⚠ 显存不足，跳过"); return

    # ── 单GPU ──
    print(f"\n  ── 单GPU torch.sum ──")
    single = []
    for N in sizes:
        d = torch.randn(N, N, device=dev)
        m, s = benchmark_fn(lambda: torch.sum(d), device=dev)
        single.append({"size":N,"mean_ms":round(m,2),"std_ms":round(s,2)})
        print(f"    {N:>6}x{N}: {m:>10.2f} ± {s:.2f} ms")
        del d; clean()

    # ── NCCL reduce ──
    nccl_file = os.path.join(RESULTS_ROOT, "_tmp_nccl_sum.json")
    nccl_script = os.path.join(RESULTS_ROOT, "_tmp_nccl_sum.py")
    with open(nccl_script, "w") as f:
        f.write(f'''
import os,json,time; import numpy as np; import torch,torch.distributed
rank=int(os.environ["RANK"]); lr=int(os.environ.get("LOCAL_RANK",rank))
torch.cuda.set_device(lr); torch.distributed.init_process_group("nccl")
ws=int(os.environ["WORLD_SIZE"]); sizes={sizes}; rep={REPEATS}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
for N in sizes:
    sz_t0=time.perf_counter()
    ck=N//ws; d=torch.randn(ck,N,device=f"cuda:{{lr}}")
    for _ in range(2):
        s=d.sum(); torch.distributed.all_reduce(s); torch.cuda.synchronize(lr)
    ts=[]
    for _ in range(rep):
        torch.cuda.synchronize(lr); torch.distributed.barrier(); t0=time.perf_counter()
        s=d.sum(); torch.distributed.all_reduce(s)
        torch.cuda.synchronize(lr); torch.distributed.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000)
    if rank==0: results.append({{"size":N,"mean_ms":round(float(np.mean(ts)),2),"std_ms":round(float(np.std(ts)),2)}})
    del d; torch.cuda.empty_cache()
    if time.perf_counter()-sz_t0>TLIMIT: break
if rank==0:
    with open("{nccl_file}","w") as ff: json.dump(results,ff)
torch.distributed.destroy_process_group()
''')
    env, _ = make_env(gpu_ids)
    print(f"\n  ── NCCL Reduce {world_size}卡 ──")
    run_subprocess(["torchrun",f"--nproc_per_node={world_size}","--master_port=29502",nccl_script],
                   env, label="NCCL Sum")
    nccl = load_tmp_json(nccl_file)
    for r in nccl: print(f"    {r['size']:>6}x{r['size']}: {r['mean_ms']:>10.2f} ± {r['std_ms']:.2f} ms")
    remove_if_exists(nccl_script)

    # ── 本框架 ──
    ours_file = os.path.join(RESULTS_ROOT, "_tmp_ours_sum.json")
    ours_script = os.path.join(RESULTS_ROOT, "_tmp_ours_sum.py")
    with open(ours_script, "w") as f:
        f.write(f'''
import sys,os,json,time,gc; sys.path.insert(0,"{PROJECT_ROOT}")
import numpy as np; import torch
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.algorithms.reduction import distributed_sum
mpi=MPIManager(); dist=TensorDistributor(mpi); gid=mpi.get_gpu_id()
sizes={sizes}; rep={REPEATS}; results=[]; TLIMIT={PER_SIZE_TIMEOUT}
for N in sizes:
    sz_t0=time.perf_counter()
    d=None
    if mpi.is_master_process(): d=torch.randn(N,N,device=f"cuda:{{gid}}")
    for _ in range(2):
        mpi.barrier(); distributed_sum(d,mpi,dist); torch.cuda.synchronize(gid); mpi.barrier()
        gc.collect(); torch.cuda.empty_cache()
    ts=[]
    for _ in range(rep):
        mpi.barrier(); torch.cuda.synchronize(gid); t0=time.perf_counter()
        distributed_sum(d,mpi,dist)
        torch.cuda.synchronize(gid); mpi.barrier(); t1=time.perf_counter()
        ts.append((t1-t0)*1000); gc.collect(); torch.cuda.empty_cache()
    if mpi.is_master_process():
        results.append({{"size":N,"mean_ms":round(float(np.mean(ts)),2),"std_ms":round(float(np.std(ts)),2)}})
        del d
    gc.collect(); torch.cuda.empty_cache(); mpi.barrier()
    skip=time.perf_counter()-sz_t0>TLIMIT
    skip=mpi.broadcast(skip if mpi.is_master_process() else None)
    if skip: break
if mpi.is_master_process():
    with open("{ours_file}","w") as ff: json.dump(results,ff)
''')
    print(f"\n  ── 本框架 分布式Sum {world_size}卡 ──")
    run_subprocess(["mpirun","-n",str(world_size),"--allow-run-as-root","--oversubscribe",
                    sys.executable,ours_script], env, label="本框架 Sum")
    ours = load_tmp_json(ours_file)
    for r in ours: print(f"    {r['size']:>6}x{r['size']}: {r['mean_ms']:>10.2f} ± {r['std_ms']:.2f} ms")
    remove_if_exists(ours_script)

    # ── 汇总 ──
    comp = []
    for s, n, o in zip(single, nccl or single, ours or single):
        sp_n = s["mean_ms"]/n["mean_ms"] if n["mean_ms"]>0 else 0
        sp_o = s["mean_ms"]/o["mean_ms"] if o["mean_ms"]>0 else 0
        comp.append({"size":s["size"],"data_gb":round(s["size"]**2*4/1024**3,3),
                     "single_ms":s["mean_ms"],"nccl_ms":n["mean_ms"],"ours_ms":o["mean_ms"],
                     "nccl_speedup":round(sp_n,3),"ours_speedup":round(sp_o,3)})
    data = {"experiment":"全局求和跨框架对比","gpu_count":world_size,
            "single_gpu":single,"torch_distributed_nccl":nccl,
            "distributed_gpu_framework":ours,"comparison":comp}
    save_json(data, os.path.join(output_dir, "comparison_reduction.json"))


# ═══════════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "matmul":    ("矩阵乘法: 本框架 vs NCCL vs 单GPU (2倍递增)", exp_matmul),
    "allreduce": ("AllReduce: 本框架 vs NCCL (2倍递增)", exp_allreduce),
    "stencil":   ("Jacobi迭代: 本框架GPU vs PETSc/NumPy CPU (2倍递增)", exp_stencil),
    "fft":       ("FFT2D: 本框架 vs Dask-CUDA vs 单GPU (2倍递增)", exp_fft),
    "reduction": ("全局求和: 本框架 vs NCCL vs 单GPU (2倍递增)", exp_reduction),
}


def main():
    parser = argparse.ArgumentParser(
        description="跨框架性能对比实验 (自适应规模，2倍递增，超显存自动停止)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python experiments/benchmark_comparison.py                    # 全部实验
  python experiments/benchmark_comparison.py --gpus 4           # 4卡
  python experiments/benchmark_comparison.py --exp matmul       # 只跑矩阵乘法
  python experiments/benchmark_comparison.py --exp matmul,fft   # 跑两个
  python experiments/benchmark_comparison.py --list             # 查看列表

注意: 不要用 mpirun 启动本脚本！脚本内部会自行管理 MPI/torchrun 子进程。
""")
    parser.add_argument("--gpus", "-g", type=int, default=None,
                        help="使用的GPU数 (默认: 自动检测全部空闲GPU)")
    parser.add_argument("--exp", "-e", type=str, default="all",
                        help="实验名 (逗号分隔) 或 all")
    parser.add_argument("--list", "-l", action="store_true", help="查看可用实验")
    args = parser.parse_args()

    if args.list:
        print("\n可用对比实验:")
        print("-" * 65)
        for k, (desc, _) in EXPERIMENTS.items():
            print(f"  {k:<12} {desc}")
        print(f"  {'all':<12} 运行全部 ({len(EXPERIMENTS)} 个实验)")
        print("-" * 65)
        return

    free_gpus = detect_free_gpus(min_free_mb=10000)
    if args.gpus:
        free_gpus = free_gpus[:args.gpus]
    if len(free_gpus) < 2:
        print(f"❌ 需要至少 2 个空闲 GPU (≥10GB空闲)，当前可用: {free_gpus}")
        print("   提示: 设置 CUDA_VISIBLE_DEVICES 指定空闲GPU")
        print("   例: CUDA_VISIBLE_DEVICES=1,3,5 python experiments/benchmark_comparison.py")
        return

    free_gb = get_min_free_gb(free_gpus)
    print(f"\n🚀 跨框架对比实验")
    print(f"   GPU: {free_gpus} ({len(free_gpus)} 卡)")
    print(f"   最小可用显存: {free_gb:.1f} GB")
    print(f"   规模策略: 2倍递增, 超显存自动停止")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_ROOT, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"   结果目录: {output_dir}")

    if args.exp == "all":
        run_list = list(EXPERIMENTS.keys())
    else:
        run_list = [x.strip() for x in args.exp.split(",")]

    t_start = time.time()
    for exp_name in run_list:
        if exp_name not in EXPERIMENTS:
            print(f"\n  ⚠ 未知实验: {exp_name}, 可用: {list(EXPERIMENTS.keys())}")
            continue
        _, fn = EXPERIMENTS[exp_name]
        try:
            fn(free_gpus, output_dir)
        except Exception as e:
            print(f"\n  ❌ 实验 {exp_name} 异常: {e}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n✅ 实验完成! 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"   结果目录: {output_dir}")
    print(f"   文件列表:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        print(f"     {f} ({os.path.getsize(fpath)/1024:.1f} KB)")

    # ── 自动生成图表 ──
    print(f"\n📊 生成跨框架对比图表...")
    try:
        fig_script = os.path.join(SCRIPT_DIR, "generate_thesis_figures_enhanced.py")
        if os.path.exists(fig_script):
            subprocess.run([sys.executable, fig_script, "--data-dir", output_dir],
                           timeout=120)
            fig_dir = os.path.join(output_dir, "figures")
            if os.path.isdir(fig_dir):
                pngs = [f for f in os.listdir(fig_dir) if f.endswith('.png')]
                print(f"   图表生成完成: {len(pngs)} 张 → {fig_dir}")
    except Exception as e:
        print(f"   ⚠ 图表生成失败: {e}")


if __name__ == "__main__":
    main()
