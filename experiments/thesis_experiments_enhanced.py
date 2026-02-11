#!/usr/bin/env python
"""
增强版硕士论文实验脚本 — 每组实验大幅扩充数据量
- 每组实验重复5次，记录均值和标准差
- 增加更多矩阵规模
- 强/弱扩展性增加多种矩阵尺寸
- 创新算子对比增加更多规模点

运行方式:
  python run_experiments.py --gpus 4 --exp all          (推荐)
  python run_experiments.py --gpus 8 --exp 1            (指定GPU数和实验)
  mpirun -n 4 python experiments/thesis_experiments_enhanced.py all   (直接运行)
  
结果自动保存在 results/n{GPU数}_{时间戳}/ 目录下，多次运行互不覆盖。
"""

import sys, os, json, time, math, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.gpu_manager import GPUManager
from distributed_gpu.cost_model import CostModel, ClusterConfig, SplitStrategy

# RESULTS_DIR 在 main() 中根据 GPU 数和时间戳动态设定
RESULTS_DIR = None

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_result(exp_name, data, mpi):
    if mpi.is_master_process():
        ensure_dir(RESULTS_DIR)
        fname = os.path.join(RESULTS_DIR, f"{exp_name}.json")
        with open(fname, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [保存] {fname}")

def warm_up_gpu(gpu_id):
    """GPU 预热"""
    a = torch.randn(1000, 1000, device=f'cuda:{gpu_id}')
    b = torch.randn(1000, 1000, device=f'cuda:{gpu_id}')
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(gpu_id)
    del a, b
    torch.cuda.empty_cache()

def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def benchmark_fn(fn, repeats=5, warmup=2):
    """计时工具, 返回 (mean_ms, std_ms, all_times)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), times

def dist_benchmark(fn, mpi, repeats=5, warmup=2):
    """分布式计时, 加barrier确保同步"""
    for _ in range(warmup):
        mpi.barrier()
        torch.cuda.synchronize()
        result = fn()
        torch.cuda.synchronize()
        mpi.barrier()
        if result is not None:
            del result
        clean_gpu()

    times = []
    for _ in range(repeats):
        mpi.barrier()
        torch.cuda.synchronize()
        t0 = time.time()
        result = fn()
        torch.cuda.synchronize()
        mpi.barrier()
        t1 = time.time()
        times.append((t1 - t0) * 1000)
        if result is not None:
            del result
        clean_gpu()
    return float(np.mean(times)), float(np.std(times)), times

def check_gpu_mem(mpi, required_gb=2.0):
    """检查GPU可用显存"""
    gpu_id = mpi.get_gpu_id()
    free = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
    if free < required_gb:
        mpi.print_master(f"  ⚠ GPU {gpu_id} 剩余 {free:.1f}GB < 需求 {required_gb:.1f}GB, 跳过")
        return False
    return True


# ==============================================================
#  实验 1: 不同矩阵规模的计算性能 (增强版)
# ==============================================================
def exp1_compute_performance(mpi, distributor):
    """不同矩阵规模下分布式matmul vs 单GPU的性能对比 — 12个规模点, 5次重复"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    sizes = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336]
    results = {"experiment": "计算性能对比(增强)", "gpu_count": mpi.get_size(),
               "repeats": 5, "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for N in sizes:
        # 估算显存需求 (A + B + C) * float32
        mem_gb = 3 * N * N * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        mpi.print_master(f"  矩阵规模: {N}x{N} (预估显存 {mem_gb:.1f}GB)")

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 单GPU计时 (仅master)
        single_mean = single_std = 0
        if mpi.is_master_process():
            def single_fn():
                torch.matmul(A, B)
            single_mean, single_std, _ = benchmark_fn(single_fn, repeats=5, warmup=2)

        # 分布式计时
        dist_mean, dist_std, dist_times = dist_benchmark(
            lambda: distributed_matmul(A, B, mpi, distributor),
            mpi, repeats=5, warmup=2
        )

        if mpi.is_master_process():
            speedup = single_mean / dist_mean if dist_mean > 0 else 0
            efficiency = speedup / mpi.get_size() * 100
            gflops = 2 * N**3 / (dist_mean / 1000) / 1e9 if dist_mean > 0 else 0

            results["data"].append({
                "matrix_size": N,
                "single_gpu_mean_ms": round(single_mean, 2),
                "single_gpu_std_ms": round(single_std, 2),
                "distributed_mean_ms": round(dist_mean, 2),
                "distributed_std_ms": round(dist_std, 2),
                "speedup": round(speedup, 3),
                "efficiency_pct": round(efficiency, 1),
                "distributed_gflops": round(gflops, 1),
                "all_dist_times": [round(t, 2) for t in dist_times],
            })
            print(f"    单GPU: {single_mean:.2f}±{single_std:.2f}ms, "
                  f"分布式: {dist_mean:.2f}±{dist_std:.2f}ms, "
                  f"加速比: {speedup:.2f}x, 效率: {efficiency:.1f}%, GFLOPS: {gflops:.0f}")

        if mpi.is_master_process():
            del A, B
        clean_gpu()
        mpi.barrier()

    save_result("exp1_compute_performance", results, mpi)
    return results


# ==============================================================
#  实验 2: 通信开销分析 (增强版)
# ==============================================================
def exp2_communication_overhead(mpi, distributor):
    """分析不同数据量下通信占比 — 10个规模点, 5次重复"""
    sizes = [256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288]
    results = {"experiment": "通信开销分析(增强)", "gpu_count": mpi.get_size(),
               "repeats": 5, "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for N in sizes:
        mem_gb = 3 * N * N * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        mpi.print_master(f"  矩阵规模: {N}x{N}")

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        all_scatter = []
        all_compute = []
        all_gather = []
        all_total = []

        for rep in range(5):
            mpi.barrier()
            torch.cuda.synchronize()

            t_total_start = time.time()

            # Scatter
            t0 = time.time()
            A_local, B_local = distributor.distribute_with_broadcast(A, B, split_dim=0)
            torch.cuda.synchronize()
            mpi.barrier()
            t1 = time.time()
            all_scatter.append((t1 - t0) * 1000)

            # Compute
            t2 = time.time()
            C_local = torch.matmul(A_local, B_local)
            torch.cuda.synchronize()
            t3 = time.time()
            all_compute.append((t3 - t2) * 1000)

            # Gather
            t4 = time.time()
            C = distributor.gather(C_local, dim=0)
            torch.cuda.synchronize()
            mpi.barrier()
            t5 = time.time()
            all_gather.append((t5 - t4) * 1000)

            all_total.append((t5 - t_total_start) * 1000)

            del A_local, B_local, C_local
            if C is not None: del C
            clean_gpu()

        if mpi.is_master_process():
            scatter_mean = float(np.mean(all_scatter))
            scatter_std = float(np.std(all_scatter))
            compute_mean = float(np.mean(all_compute))
            compute_std = float(np.std(all_compute))
            gather_mean = float(np.mean(all_gather))
            gather_std = float(np.std(all_gather))
            total_mean = float(np.mean(all_total))
            total_std = float(np.std(all_total))
            comm_mean = scatter_mean + gather_mean
            comm_pct = comm_mean / total_mean * 100 if total_mean > 0 else 0
            data_gb = N * N * 4 * 2 / 1024**3

            results["data"].append({
                "matrix_size": N,
                "data_size_gb": round(data_gb, 4),
                "scatter_mean_ms": round(scatter_mean, 2),
                "scatter_std_ms": round(scatter_std, 2),
                "compute_mean_ms": round(compute_mean, 2),
                "compute_std_ms": round(compute_std, 2),
                "gather_mean_ms": round(gather_mean, 2),
                "gather_std_ms": round(gather_std, 2),
                "total_mean_ms": round(total_mean, 2),
                "total_std_ms": round(total_std, 2),
                "comm_mean_ms": round(comm_mean, 2),
                "comm_ratio_pct": round(comm_pct, 1),
                "compute_ratio_pct": round(100 - comm_pct, 1),
            })
            print(f"    scatter={scatter_mean:.1f}±{scatter_std:.1f}ms "
                  f"compute={compute_mean:.1f}±{compute_std:.1f}ms "
                  f"gather={gather_mean:.1f}±{gather_std:.1f}ms "
                  f"通信占比={comm_pct:.1f}%")

        if mpi.is_master_process():
            del A, B
        clean_gpu()
        mpi.barrier()

    save_result("exp2_comm_overhead", results, mpi)
    return results


# ==============================================================
#  实验 3: 强可扩展性 (增强版 — 多种矩阵规模)
# ==============================================================
def exp3_strong_scaling(mpi, distributor):
    """固定问题规模, 当前GPU数下的性能 — 多种矩阵规模, 5次重复"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    matrix_sizes = [2048, 4096, 6144, 8192, 10240]
    results = {"experiment": "强可扩展性(增强)", "gpu_count": mpi.get_size(),
               "repeats": 5, "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for N in matrix_sizes:
        mem_gb = 3 * N * N * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        mpi.print_master(f"  强可扩展性: 矩阵 {N}x{N}, GPU数={mpi.get_size()}")

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 单GPU基线
        single_mean = single_std = 0
        if mpi.is_master_process():
            def single_fn():
                torch.matmul(A, B)
            single_mean, single_std, _ = benchmark_fn(single_fn, repeats=5, warmup=2)

        # 分布式
        dist_mean, dist_std, dist_times = dist_benchmark(
            lambda: distributed_matmul(A, B, mpi, distributor),
            mpi, repeats=5, warmup=2
        )

        if mpi.is_master_process():
            speedup = single_mean / dist_mean if dist_mean > 0 else 0
            efficiency = speedup / mpi.get_size() * 100

            results["data"].append({
                "matrix_size": N,
                "num_gpus": mpi.get_size(),
                "single_gpu_mean_ms": round(single_mean, 2),
                "single_gpu_std_ms": round(single_std, 2),
                "distributed_mean_ms": round(dist_mean, 2),
                "distributed_std_ms": round(dist_std, 2),
                "speedup": round(speedup, 3),
                "efficiency_pct": round(efficiency, 1),
                "all_times": [round(t, 2) for t in dist_times],
            })
            print(f"    N={N}: 单GPU={single_mean:.2f}±{single_std:.2f}ms, "
                  f"分布式={dist_mean:.2f}±{dist_std:.2f}ms, "
                  f"加速比={speedup:.2f}x, 效率={efficiency:.1f}%")

        if mpi.is_master_process():
            del A, B
        clean_gpu()
        mpi.barrier()

    save_result("exp3_strong_scaling", results, mpi)
    return results


# ==============================================================
#  实验 4: 弱可扩展性 (增强版 — 多种每GPU工作量)
# ==============================================================
def exp4_weak_scaling(mpi, distributor):
    """每GPU固定工作量, 当前GPU数下的性能 — 多种per_gpu_rows"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    per_gpu_rows_list = [1024, 2048, 3072, 4096]
    K = 2048
    N_col = 2048

    results = {"experiment": "弱可扩展性(增强)", "gpu_count": mpi.get_size(),
               "K": K, "N_col": N_col, "repeats": 5, "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for per_gpu_rows in per_gpu_rows_list:
        M = per_gpu_rows * mpi.get_size()
        mem_gb = (M * K + K * N_col + M * N_col) * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        mpi.print_master(f"  弱可扩展性: 每GPU {per_gpu_rows}行, 总M={M}")

        if mpi.is_master_process():
            A = torch.randn(M, K, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(K, N_col, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 单GPU基线 (仅 per_gpu_rows 的工作量)
        single_mean = single_std = 0
        if mpi.is_master_process():
            A_small = torch.randn(per_gpu_rows, K, device=f'cuda:{mpi.get_gpu_id()}')
            B_small = torch.randn(K, N_col, device=f'cuda:{mpi.get_gpu_id()}')
            single_mean, single_std, _ = benchmark_fn(
                lambda: torch.matmul(A_small, B_small), repeats=5, warmup=2)
            del A_small, B_small
            clean_gpu()

        # 分布式
        dist_mean, dist_std, dist_times = dist_benchmark(
            lambda: distributed_matmul(A, B, mpi, distributor),
            mpi, repeats=5, warmup=2
        )

        if mpi.is_master_process():
            weak_eff = single_mean / dist_mean * 100 if dist_mean > 0 else 0

            results["data"].append({
                "per_gpu_rows": per_gpu_rows,
                "num_gpus": mpi.get_size(),
                "total_M": M,
                "single_gpu_mean_ms": round(single_mean, 2),
                "single_gpu_std_ms": round(single_std, 2),
                "distributed_mean_ms": round(dist_mean, 2),
                "distributed_std_ms": round(dist_std, 2),
                "weak_efficiency_pct": round(weak_eff, 1),
            })
            print(f"    per_gpu_rows={per_gpu_rows}: "
                  f"单GPU(单份)={single_mean:.2f}±{single_std:.2f}ms, "
                  f"分布式(p份)={dist_mean:.2f}±{dist_std:.2f}ms, 弱效率={weak_eff:.1f}%")

        if mpi.is_master_process():
            del A, B
        clean_gpu()
        mpi.barrier()

    save_result("exp4_weak_scaling", results, mpi)
    return results


# ==============================================================
#  实验 5: 创新算子对比 (增强版)
# ==============================================================
def exp5_innovation_comparison(mpi, distributor):
    """混合精度/稀疏感知/Kahan/Pencil FFT 对比 — 更多规模点"""
    from distributed_gpu.algorithms.matrix_ops import (
        distributed_matmul, distributed_matmul_mixed_precision,
        distributed_matmul_sparse_aware
    )
    from distributed_gpu.algorithms.reduction import distributed_sum, distributed_sum_kahan
    from distributed_gpu.algorithms.fft import distributed_fft2d, distributed_fft2d_pencil

    results = {"experiment": "创新算子对比(增强)", "gpu_count": mpi.get_size(),
               "repeats": 5, "data": {}}

    warm_up_gpu(mpi.get_gpu_id())

    # --- 5a: 混合精度 vs 标准矩阵乘法 (8个规模点) ---
    mpi.print_master("  [5a] 混合精度 vs 标准 matmul")
    mp_results = []
    for N in [1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240]:
        mem_gb = 3 * N * N * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 标准
        std_mean, std_std, _ = dist_benchmark(
            lambda: distributed_matmul(A, B, mpi, distributor),
            mpi, repeats=5, warmup=2
        )

        # 混合精度
        mp_mean, mp_std, _ = dist_benchmark(
            lambda: distributed_matmul_mixed_precision(A, B, mpi, distributor),
            mpi, repeats=5, warmup=2
        )

        # 精度对比
        if mpi.is_master_process():
            C_ref = torch.matmul(A, B)

        C_std = distributed_matmul(A, B, mpi, distributor)
        C_mp = distributed_matmul_mixed_precision(A, B, mpi, distributor)

        if mpi.is_master_process():
            err_std = torch.mean(torch.abs(C_std - C_ref)).item()
            err_mp = torch.mean(torch.abs(C_mp - C_ref)).item()
            rel_err_std = err_std / torch.mean(torch.abs(C_ref)).item()
            rel_err_mp = err_mp / torch.mean(torch.abs(C_ref)).item()

            mp_results.append({
                "N": N,
                "standard_mean_ms": round(std_mean, 2),
                "standard_std_ms": round(std_std, 2),
                "mixed_precision_mean_ms": round(mp_mean, 2),
                "mixed_precision_std_ms": round(mp_std, 2),
                "speedup": round(std_mean / mp_mean if mp_mean > 0 else 0, 3),
                "std_rel_error": float(f"{rel_err_std:.2e}"),
                "mp_rel_error": float(f"{rel_err_mp:.2e}"),
            })
            print(f"    N={N}: std={std_mean:.1f}±{std_std:.1f}ms mp={mp_mean:.1f}±{mp_std:.1f}ms "
                  f"speedup={std_mean/mp_mean:.2f}x")
            del C_std, C_mp, C_ref, A, B
            clean_gpu()
        mpi.barrier()

    results["data"]["mixed_precision"] = mp_results

    # --- 5b: 稀疏感知 (多种矩阵规模 x 多种稀疏度) ---
    mpi.print_master("  [5b] 稀疏感知 matmul")
    sparse_results = []
    for N in [2048, 4096, 8192]:
        mem_gb = 3 * N * N * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        for sparsity in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            if mpi.is_master_process():
                A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
                B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
                mask = torch.rand(N, N, device=f'cuda:{mpi.get_gpu_id()}') > sparsity
                B = B * mask.float()
                del mask
            else:
                A = B = None

            # 标准
            std_mean, std_std, _ = dist_benchmark(
                lambda: distributed_matmul(A, B, mpi, distributor),
                mpi, repeats=5, warmup=2
            )

            # 稀疏感知
            sp_mean, sp_std, _ = dist_benchmark(
                lambda: distributed_matmul_sparse_aware(A, B, mpi, distributor, sparsity_threshold=0.5),
                mpi, repeats=5, warmup=2
            )

            if mpi.is_master_process():
                sparse_results.append({
                    "N": N,
                    "sparsity": sparsity,
                    "standard_mean_ms": round(std_mean, 2),
                    "standard_std_ms": round(std_std, 2),
                    "sparse_aware_mean_ms": round(sp_mean, 2),
                    "sparse_aware_std_ms": round(sp_std, 2),
                    "speedup": round(std_mean / max(sp_mean, 0.01), 3),
                })
                print(f"    N={N} sparsity={sparsity}: "
                      f"std={std_mean:.1f}±{std_std:.1f}ms "
                      f"sparse={sp_mean:.1f}±{sp_std:.1f}ms")
                del A, B
                clean_gpu()
            mpi.barrier()

    results["data"]["sparse_aware"] = sparse_results

    # --- 5c: Kahan 补偿求和精度 (更多数据点) ---
    mpi.print_master("  [5c] Kahan 补偿求和精度")
    kahan_results = []
    for N in [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000]:
        if mpi.is_master_process():
            data = torch.ones(N, device=f'cuda:{mpi.get_gpu_id()}') * 1.0
            data[0] = 1e8
            data[-1] = -1e8
            exact = float(N - 2)
        else:
            data = None
            exact = 0

        # 重复5次取平均
        std_errors = []
        kahan_errors = []
        for _ in range(5):
            std_result = distributed_sum(data, mpi, distributor)
            kahan_result = distributed_sum_kahan(data, mpi, distributor)
            if mpi.is_master_process():
                exact_val = float(N - 2)
                std_errors.append(abs(std_result.item() - exact_val))
                kahan_errors.append(abs(kahan_result.item() - exact_val))

        if mpi.is_master_process():
            std_err_mean = float(np.mean(std_errors))
            kahan_err_mean = float(np.mean(kahan_errors))
            kahan_results.append({
                "N": N,
                "exact_value": exact_val,
                "standard_abs_error_mean": float(f"{std_err_mean:.6e}"),
                "kahan_abs_error_mean": float(f"{kahan_err_mean:.6e}"),
                "improvement_factor": round(std_err_mean / max(kahan_err_mean, 1e-30), 1),
                "all_std_errors": [float(f"{e:.6e}") for e in std_errors],
                "all_kahan_errors": [float(f"{e:.6e}") for e in kahan_errors],
            })
            print(f"    N={N}: std_err={std_err_mean:.2e}, kahan_err={kahan_err_mean:.2e}")
            del data
            clean_gpu()
        mpi.barrier()

    results["data"]["kahan_sum"] = kahan_results

    # --- 5d: Pencil FFT vs Batch FFT (更多网格大小) ---
    mpi.print_master("  [5d] Pencil FFT vs Batch FFT")
    fft_results = []
    P = mpi.get_size()
    for grid_size in [128, 256, 512, 1024, 2048, 4096]:
        if grid_size % P != 0:
            continue
        mem_gb = grid_size * grid_size * 8 * 2 / 1024**3  # complex
        if not check_gpu_mem(mpi, required_gb=mem_gb + 1.0):
            continue

        if mpi.is_master_process():
            data_batch = torch.randn(P, grid_size, grid_size,
                                      device=f'cuda:{mpi.get_gpu_id()}')
            data_pencil = torch.randn(grid_size, grid_size,
                                       device=f'cuda:{mpi.get_gpu_id()}')
        else:
            data_batch = data_pencil = None

        # Batch FFT
        batch_mean, batch_std, _ = dist_benchmark(
            lambda: distributed_fft2d(data_batch, mpi, distributor),
            mpi, repeats=5, warmup=2
        )

        # Pencil FFT
        pencil_mean, pencil_std, _ = dist_benchmark(
            lambda: distributed_fft2d_pencil(data_pencil, mpi, distributor),
            mpi, repeats=5, warmup=2
        )

        if mpi.is_master_process():
            fft_results.append({
                "grid_size": grid_size,
                "batch_fft_mean_ms": round(batch_mean, 2),
                "batch_fft_std_ms": round(batch_std, 2),
                "pencil_fft_mean_ms": round(pencil_mean, 2),
                "pencil_fft_std_ms": round(pencil_std, 2),
            })
            print(f"    grid={grid_size}: batch={batch_mean:.1f}±{batch_std:.1f}ms "
                  f"pencil={pencil_mean:.1f}±{pencil_std:.1f}ms")
            del data_batch, data_pencil
            clean_gpu()
        mpi.barrier()

    results["data"]["pencil_fft"] = fft_results

    save_result("exp5_innovation", results, mpi)
    return results


# ==============================================================
#  实验 6: 流水线优化效果 (增强版)
# ==============================================================
def exp6_pipeline(mpi, distributor):
    """流水线(计算-通信重叠) vs 非流水线 — 更多规模, 更多chunk数"""
    from distributed_gpu.pipeline_optimizer import PipelineOptimizer, PipelineConfig

    pipe = PipelineOptimizer(mpi, PipelineConfig(num_chunks=4))
    results = {"experiment": "流水线优化(增强)", "gpu_count": mpi.get_size(),
               "repeats": 5, "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for N in [1024, 2048, 3072, 4096, 5120, 6144, 8192, 10240]:
        mem_gb = 3 * N * N * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        mpi.print_master(f"  矩阵规模: {N}x{N}")

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 非流水线
        bl_mean, bl_std, _ = dist_benchmark(
            lambda: pipe.baseline_matmul(A, B),
            mpi, repeats=5, warmup=2
        )

        # 流水线 (不同chunk数)
        for num_chunks in [2, 4, 8, 16]:
            pl_mean, pl_std, _ = dist_benchmark(
                lambda: pipe.pipelined_matmul(A, B, num_chunks=num_chunks),
                mpi, repeats=5, warmup=2
            )

            if mpi.is_master_process():
                results["data"].append({
                    "matrix_size": N,
                    "num_chunks": num_chunks,
                    "baseline_mean_ms": round(bl_mean, 2),
                    "baseline_std_ms": round(bl_std, 2),
                    "pipeline_mean_ms": round(pl_mean, 2),
                    "pipeline_std_ms": round(pl_std, 2),
                    "speedup": round(bl_mean / pl_mean if pl_mean > 0 else 0, 3),
                })
                print(f"    N={N} chunks={num_chunks}: "
                      f"baseline={bl_mean:.1f}±{bl_std:.1f}ms "
                      f"pipeline={pl_mean:.1f}±{pl_std:.1f}ms "
                      f"加速={bl_mean/pl_mean:.2f}x")

        if mpi.is_master_process():
            del A, B
        clean_gpu()
        mpi.barrier()

    save_result("exp6_pipeline", results, mpi)
    return results


# ==============================================================
#  实验 7: 代价模型策略选择 (增强版 — 更多矩阵形状)
# ==============================================================
def exp7_cost_model(mpi, distributor):
    """代价模型 vs 固定策略对比 — 10种矩阵形状"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    config = ClusterConfig.from_auto_detect(mpi.get_size())
    cost_model = CostModel(config)

    results = {"experiment": "代价模型策略(增强)", "gpu_count": mpi.get_size(),
               "repeats": 5, "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    # 更多形状矩阵
    shapes = [
        (8192, 2048, 2048, "M>>K,N (4:1:1)"),
        (2048, 2048, 8192, "N>>M,K (1:1:4)"),
        (4096, 4096, 4096, "M=K=N (1:1:1)"),
        (10240, 1024, 1024, "M>>>K,N (10:1:1)"),
        (1024, 1024, 10240, "N>>>M,K (1:1:10)"),
        (8192, 1024, 8192, "M=N>>K (8:1:8)"),
        (2048, 8192, 2048, "K>>M,N (1:4:1)"),
        (4096, 2048, 8192, "N>M>K (2:1:4)"),
        (6144, 6144, 2048, "M=K>>N (3:3:1)"),
        (2048, 6144, 6144, "K=N>>M (1:3:3)"),
    ]

    for M, K, N, desc in shapes:
        mem_gb = (M * K + K * N + M * N) * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        mpi.print_master(f"  [{desc}] A[{M},{K}] @ B[{K},{N}]")

        if mpi.is_master_process():
            A = torch.randn(M, K, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(K, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 代价模型自动选择
        auto_mean, auto_std, _ = dist_benchmark(
            lambda: distributed_matmul(A, B, mpi, distributor, cost_model=cost_model),
            mpi, repeats=5, warmup=2
        )

        # 固定行分割
        row_mean, row_std, _ = dist_benchmark(
            lambda: distributed_matmul(A, B, mpi, distributor, strategy=SplitStrategy.ROW_SPLIT),
            mpi, repeats=5, warmup=2
        )

        # 固定列分割
        col_mean, col_std, _ = dist_benchmark(
            lambda: distributed_matmul(A, B, mpi, distributor, strategy=SplitStrategy.COLUMN_SPLIT),
            mpi, repeats=5, warmup=2
        )

        if mpi.is_master_process():
            plan = cost_model.find_optimal_strategy(M, K, N)
            best_fixed = min(row_mean, col_mean)
            auto_is_best = auto_mean <= best_fixed * 1.05  # 5%容差

            results["data"].append({
                "shape": f"[{M},{K}]@[{K},{N}]",
                "description": desc,
                "auto_strategy": plan.strategy.value,
                "auto_mean_ms": round(auto_mean, 2),
                "auto_std_ms": round(auto_std, 2),
                "row_split_mean_ms": round(row_mean, 2),
                "row_split_std_ms": round(row_std, 2),
                "col_split_mean_ms": round(col_mean, 2),
                "col_split_std_ms": round(col_std, 2),
                "auto_is_optimal": auto_is_best,
            })
            print(f"    auto({plan.strategy.value})={auto_mean:.1f}±{auto_std:.1f}ms, "
                  f"row={row_mean:.1f}±{row_std:.1f}ms, "
                  f"col={col_mean:.1f}±{col_std:.1f}ms "
                  f"{'✓' if auto_is_best else '✗'}")
            del A, B
            clean_gpu()
        mpi.barrier()

    save_result("exp7_cost_model", results, mpi)
    return results


# ==============================================================
#  实验 8: 科学计算应用场景 (增强版)
# ==============================================================
def exp8_applications(mpi, distributor):
    """Stencil/Jacobi/Conv2d/Einsum 应用 — 更多参数组合"""
    from distributed_gpu.algorithms.stencil import distributed_stencil_2d, distributed_jacobi_2d
    from distributed_gpu.algorithms.convolution import distributed_conv2d
    from distributed_gpu.algorithms.einsum import distributed_einsum

    results = {"experiment": "科学计算应用(增强)", "gpu_count": mpi.get_size(),
               "repeats": 5, "data": {}}

    warm_up_gpu(mpi.get_gpu_id())
    P = mpi.get_size()

    # --- 8a: Stencil 热传导模拟 (更多网格+迭代次数) ---
    mpi.print_master("  [8a] Stencil 热传导")
    stencil_results = []
    for grid_size in [128, 256, 512, 1024, 2048, 4096]:
        if grid_size % P != 0:
            continue
        mem_gb = grid_size * grid_size * 4 * 2 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 1.0):
            continue

        for iters in [10, 50, 100, 200]:
            stencil_times = []
            for rep in range(3):
                if mpi.is_master_process():
                    grid = torch.zeros(grid_size, grid_size, device=f'cuda:{mpi.get_gpu_id()}')
                    cx, cy = grid_size // 2, grid_size // 2
                    grid[cx-5:cx+5, cy-5:cy+5] = 100.0
                else:
                    grid = None

                mpi.barrier(); torch.cuda.synchronize()
                t0 = time.time()
                result = distributed_stencil_2d(grid, mpi, distributor, iterations=iters)
                torch.cuda.synchronize(); mpi.barrier()
                stencil_times.append((time.time() - t0) * 1000)

                if mpi.is_master_process():
                    del grid, result
                    clean_gpu()
                mpi.barrier()

            if mpi.is_master_process():
                mean_t = float(np.mean(stencil_times))
                std_t = float(np.std(stencil_times))
                stencil_results.append({
                    "grid_size": grid_size,
                    "iterations": iters,
                    "mean_ms": round(mean_t, 2),
                    "std_ms": round(std_t, 2),
                    "iter_per_sec": round(iters / mean_t * 1000, 1),
                    "throughput_cells_per_sec": round(grid_size * grid_size * iters / mean_t * 1000, 0),
                })
                print(f"    grid={grid_size} iters={iters}: {mean_t:.1f}±{std_t:.1f}ms")

    results["data"]["stencil"] = stencil_results

    # --- 8b: Jacobi 泊松方程 (多种网格, 记录收敛) ---
    mpi.print_master("  [8b] Jacobi 泊松方程")
    jacobi_results = []
    for grid_size in [128, 256, 512, 1024, 2048]:
        if grid_size % P != 0:
            continue
        mem_gb = grid_size * grid_size * 4 * 3 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 1.0):
            continue

        for max_iters in [100, 200, 500]:
            jacobi_times = []
            for rep in range(3):
                if mpi.is_master_process():
                    u = torch.zeros(grid_size, grid_size, device=f'cuda:{mpi.get_gpu_id()}')
                    rhs = torch.zeros(grid_size, grid_size, device=f'cuda:{mpi.get_gpu_id()}')
                    rhs[grid_size//2, grid_size//2] = 1.0
                else:
                    u = rhs = None

                mpi.barrier(); torch.cuda.synchronize()
                t0 = time.time()
                result = distributed_jacobi_2d(u, rhs, mpi, distributor,
                                                iterations=max_iters, tol=1e-6)
                torch.cuda.synchronize(); mpi.barrier()
                jacobi_times.append((time.time() - t0) * 1000)

                if mpi.is_master_process():
                    del u, rhs, result
                    clean_gpu()
                mpi.barrier()

            if mpi.is_master_process():
                mean_t = float(np.mean(jacobi_times))
                std_t = float(np.std(jacobi_times))
                jacobi_results.append({
                    "grid_size": grid_size,
                    "max_iterations": max_iters,
                    "mean_ms": round(mean_t, 2),
                    "std_ms": round(std_t, 2),
                    "iter_per_sec": round(max_iters / mean_t * 1000, 1),
                })
                print(f"    grid={grid_size} iters={max_iters}: {mean_t:.1f}±{std_t:.1f}ms")

    results["data"]["jacobi"] = jacobi_results

    # --- 8c: 分布式卷积 (更多batch+通道) ---
    mpi.print_master("  [8c] 分布式 Conv2d")
    conv_results = []
    conv_configs = [
        (8, 32, 64, 56, "small"),
        (16, 64, 128, 56, "medium"),
        (32, 64, 128, 56, "medium-batch"),
        (64, 64, 128, 56, "large-batch"),
        (128, 64, 128, 56, "xlarge-batch"),
        (32, 128, 256, 28, "deep-small"),
        (64, 128, 256, 28, "deep-large"),
        (32, 256, 512, 14, "very-deep"),
    ]
    for batch, in_ch, out_ch, hw, desc in conv_configs:
        if batch < P:
            continue
        mem_gb = (batch * in_ch * hw * hw + out_ch * in_ch * 9 + batch * out_ch * hw * hw) * 4 / 1024**3
        if not check_gpu_mem(mpi, required_gb=mem_gb + 2.0):
            continue

        if mpi.is_master_process():
            inp = torch.randn(batch, in_ch, hw, hw, device=f'cuda:{mpi.get_gpu_id()}')
            weight = torch.randn(out_ch, in_ch, 3, 3, device=f'cuda:{mpi.get_gpu_id()}')
            bias = torch.randn(out_ch, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            inp = weight = bias = None

        # 分布式
        dist_mean, dist_std, _ = dist_benchmark(
            lambda: distributed_conv2d(inp, weight, mpi, distributor, bias=bias, padding=(1, 1)),
            mpi, repeats=5, warmup=2
        )

        # 单GPU
        single_mean = single_std = 0
        if mpi.is_master_process():
            import torch.nn.functional as F
            single_mean, single_std, _ = benchmark_fn(
                lambda: F.conv2d(inp, weight, bias, padding=(1, 1)),
                repeats=5, warmup=2
            )

        if mpi.is_master_process():
            conv_results.append({
                "config": desc,
                "batch_size": batch,
                "in_channels": in_ch,
                "out_channels": out_ch,
                "spatial_size": hw,
                "single_gpu_mean_ms": round(single_mean, 2),
                "single_gpu_std_ms": round(single_std, 2),
                "distributed_mean_ms": round(dist_mean, 2),
                "distributed_std_ms": round(dist_std, 2),
                "speedup": round(single_mean / dist_mean if dist_mean > 0 else 0, 3),
            })
            print(f"    [{desc}] batch={batch} ch={in_ch}->{out_ch} hw={hw}: "
                  f"single={single_mean:.1f}ms dist={dist_mean:.1f}ms "
                  f"speedup={single_mean/dist_mean:.2f}x")
            del inp, weight, bias
            clean_gpu()
        mpi.barrier()

    results["data"]["conv2d"] = conv_results

    # --- 8d: Einsum 张量收缩 (更多表达式) ---
    mpi.print_master("  [8d] Einsum 张量收缩")
    einsum_results = []
    equations = [
        ("ij,jk->ik", [(2048, 1024), (1024, 2048)], "矩阵乘法"),
        ("ij,jk->ik", [(4096, 2048), (2048, 4096)], "大矩阵乘法"),
        ("bij,bjk->bik", [(32, 256, 128), (32, 128, 256)], "批量矩阵乘法(small)"),
        ("bij,bjk->bik", [(64, 512, 256), (64, 256, 512)], "批量矩阵乘法(large)"),
        ("ijkl,klmn->ijmn", [(16, 16, 32, 32), (32, 32, 16, 16)], "4阶张量收缩"),
        ("ijk,ikl->ijl", [(64, 128, 64), (64, 64, 128)], "3阶收缩"),
        ("ij,jk,kl->il", [(1024, 512), (512, 1024), (1024, 512)], "链式矩阵乘法"),
    ]
    for eq, shapes, desc in equations:
        if mpi.is_master_process():
            ops = tuple(torch.randn(*s, device=f'cuda:{mpi.get_gpu_id()}') for s in shapes)
        else:
            ops = tuple(None for _ in shapes)

        dist_mean, dist_std, _ = dist_benchmark(
            lambda: distributed_einsum(eq, *ops, mpi=mpi, distributor=distributor),
            mpi, repeats=5, warmup=2
        )

        # 单GPU
        single_mean = single_std = 0
        if mpi.is_master_process():
            single_mean, single_std, _ = benchmark_fn(
                lambda: torch.einsum(eq, *ops), repeats=5, warmup=2
            )

        if mpi.is_master_process():
            einsum_results.append({
                "equation": eq,
                "shapes": [list(s) for s in shapes],
                "description": desc,
                "single_gpu_mean_ms": round(single_mean, 2),
                "single_gpu_std_ms": round(single_std, 2),
                "distributed_mean_ms": round(dist_mean, 2),
                "distributed_std_ms": round(dist_std, 2),
                "speedup": round(single_mean / dist_mean if dist_mean > 0 else 0, 3),
            })
            print(f"    {eq} ({desc}): single={single_mean:.1f}ms dist={dist_mean:.1f}ms")
            for op in ops:
                if op is not None: del op
            clean_gpu()
        mpi.barrier()

    results["data"]["einsum"] = einsum_results

    save_result("exp8_applications", results, mpi)
    return results


# ==============================================================
#  Main
# ==============================================================
def main():
    global RESULTS_DIR
    from datetime import datetime

    mpi = MPIManager()
    distributor = TensorDistributor(mpi)

    # 解析命令行参数
    exp_id = "all"
    output_dir = None
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            exp_id = arg
            i += 1

    # 生成输出目录: results/n{gpu}_{timestamp}/
    if output_dir:
        RESULTS_DIR = output_dir
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"n{mpi.get_size()}_{timestamp}"
        RESULTS_DIR = os.path.join(project_root, "results", run_name)

    if mpi.is_master_process():
        ensure_dir(RESULTS_DIR)

    mpi.print_master(f"\n结果输出目录: {RESULTS_DIR}")

    experiments = {
        "1": ("实验1: 计算性能对比(增强)", exp1_compute_performance),
        "2": ("实验2: 通信开销分析(增强)", exp2_communication_overhead),
        "3": ("实验3: 强可扩展性(增强)", exp3_strong_scaling),
        "4": ("实验4: 弱可扩展性(增强)", exp4_weak_scaling),
        "5": ("实验5: 创新算子对比(增强)", exp5_innovation_comparison),
        "6": ("实验6: 流水线优化(增强)", exp6_pipeline),
        "7": ("实验7: 代价模型策略(增强)", exp7_cost_model),
        "8": ("实验8: 科学计算应用(增强)", exp8_applications),
    }

    if exp_id == "all":
        run_list = sorted(experiments.keys())
    else:
        run_list = [exp_id]

    for eid in run_list:
        if eid not in experiments:
            if mpi.is_master_process():
                print(f"未知实验ID: {eid}")
            continue
        name, fn = experiments[eid]
        mpi.print_master(f"\n{'='*60}")
        mpi.print_master(f"  {name}  (GPU数: {mpi.get_size()})")
        mpi.print_master(f"{'='*60}")
        fn(mpi, distributor)

    mpi.print_master(f"\n全部实验完成! 结果保存在: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
