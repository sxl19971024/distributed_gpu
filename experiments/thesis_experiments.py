#!/usr/bin/env python
"""
硕士论文实验脚本 — 全部实验数据生成
涵盖:
  实验1: 不同矩阵规模的计算性能 (单GPU vs 多GPU, 不同分割策略)
  实验2: 通信开销分析 (数据量 vs 通信时间占比)
  实验3: 强可扩展性 (固定问题规模, 增加GPU数)
  实验4: 弱可扩展性 (每GPU问题规模固定, 增加GPU数)
  实验5: 创新算子对比 (混合精度/稀疏感知/Kahan/Pencil FFT)
  实验6: 流水线优化效果 (重叠 vs 非重叠)
  实验7: 代价模型策略选择准确性
  实验8: 科学计算应用 (Stencil/Jacobi/FFT)

运行方式:
  mpirun -n <N> python experiments/thesis_experiments.py <exp_id>
  例: mpirun -n 4 python experiments/thesis_experiments.py 1
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.gpu_manager import GPUManager
from distributed_gpu.cost_model import CostModel, ClusterConfig, SplitStrategy

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "thesis")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_result(exp_name, data, mpi):
    if mpi.is_master_process():
        ensure_dir(RESULTS_DIR)
        fname = os.path.join(RESULTS_DIR, f"{exp_name}_n{mpi.get_size()}.json")
        with open(fname, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [保存] {fname}")

def warm_up_gpu(gpu_id):
    """GPU 预热"""
    a = torch.randn(1000, 1000, device=f'cuda:{gpu_id}')
    b = torch.randn(1000, 1000, device=f'cuda:{gpu_id}')
    for _ in range(5):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(gpu_id)
    del a, b
    torch.cuda.empty_cache()

def benchmark_fn(fn, repeats=3, warmup=1):
    """计时工具, 返回平均耗时(ms)"""
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
    return float(np.median(times))


# ==============================================================
#  实验 1: 不同矩阵规模的计算性能
# ==============================================================
def exp1_compute_performance(mpi, distributor):
    """不同矩阵规模下分布式matmul vs 单GPU的性能对比"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    sizes = [1000, 2000, 4000, 6000, 8000, 10000, 12000]
    results = {"experiment": "计算性能对比", "gpu_count": mpi.get_size(), "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for N in sizes:
        mpi.print_master(f"  矩阵规模: {N}x{N}")

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 单GPU计时 (仅master)
        single_time = 0
        if mpi.is_master_process():
            def single_fn():
                torch.matmul(A, B)
            single_time = benchmark_fn(single_fn, repeats=3)

        # 分布式计时
        mpi.barrier()
        dist_times = []
        for _ in range(3):
            mpi.barrier()
            torch.cuda.synchronize()
            t0 = time.time()
            C = distributed_matmul(A, B, mpi, distributor)
            torch.cuda.synchronize()
            mpi.barrier()
            t1 = time.time()
            dist_times.append((t1 - t0) * 1000)

        dist_time = float(np.median(dist_times))

        if mpi.is_master_process():
            speedup = single_time / dist_time if dist_time > 0 else 0
            efficiency = speedup / mpi.get_size() * 100
            results["data"].append({
                "matrix_size": N,
                "single_gpu_ms": round(single_time, 2),
                "distributed_ms": round(dist_time, 2),
                "speedup": round(speedup, 3),
                "efficiency_pct": round(efficiency, 1),
            })
            print(f"    单GPU: {single_time:.2f}ms, 分布式: {dist_time:.2f}ms, "
                  f"加速比: {speedup:.2f}x, 效率: {efficiency:.1f}%")

        if mpi.is_master_process():
            del A, B
            if C is not None: del C
            torch.cuda.empty_cache()
        mpi.barrier()

    save_result("exp1_compute_performance", results, mpi)
    return results


# ==============================================================
#  实验 2: 通信开销分析
# ==============================================================
def exp2_communication_overhead(mpi, distributor):
    """分析不同数据量下通信占比"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    sizes = [500, 1000, 2000, 4000, 6000, 8000, 10000]
    results = {"experiment": "通信开销分析", "gpu_count": mpi.get_size(), "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for N in sizes:
        mpi.print_master(f"  矩阵规模: {N}x{N}")

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 分离计时: scatter + compute + gather
        scatter_times = []
        compute_times = []
        gather_times = []
        total_times = []

        for _ in range(3):
            mpi.barrier()
            torch.cuda.synchronize()

            t_total_start = time.time()

            # Scatter
            t0 = time.time()
            A_local, B_local = distributor.distribute_with_broadcast(A, B, split_dim=0)
            torch.cuda.synchronize()
            mpi.barrier()
            t1 = time.time()
            scatter_times.append((t1 - t0) * 1000)

            # Compute
            t2 = time.time()
            C_local = torch.matmul(A_local, B_local)
            torch.cuda.synchronize()
            t3 = time.time()
            compute_times.append((t3 - t2) * 1000)

            # Gather
            t4 = time.time()
            C = distributor.gather(C_local, dim=0)
            torch.cuda.synchronize()
            mpi.barrier()
            t5 = time.time()
            gather_times.append((t5 - t4) * 1000)

            total_times.append((t5 - t_total_start) * 1000)

            del A_local, B_local, C_local
            if C is not None: del C
            torch.cuda.empty_cache()

        if mpi.is_master_process():
            scatter_ms = float(np.median(scatter_times))
            compute_ms = float(np.median(compute_times))
            gather_ms = float(np.median(gather_times))
            total_ms = float(np.median(total_times))
            comm_ms = scatter_ms + gather_ms
            comm_pct = comm_ms / total_ms * 100 if total_ms > 0 else 0
            data_gb = N * N * 4 * 2 / 1024**3  # A+B

            results["data"].append({
                "matrix_size": N,
                "data_size_gb": round(data_gb, 3),
                "scatter_ms": round(scatter_ms, 2),
                "compute_ms": round(compute_ms, 2),
                "gather_ms": round(gather_ms, 2),
                "total_ms": round(total_ms, 2),
                "comm_ms": round(comm_ms, 2),
                "comm_ratio_pct": round(comm_pct, 1),
            })
            print(f"    scatter={scatter_ms:.1f}ms compute={compute_ms:.1f}ms "
                  f"gather={gather_ms:.1f}ms total={total_ms:.1f}ms 通信占比={comm_pct:.1f}%")

        if mpi.is_master_process():
            del A, B
            torch.cuda.empty_cache()
        mpi.barrier()

    save_result("exp2_comm_overhead", results, mpi)
    return results


# ==============================================================
#  实验 3: 强可扩展性 (单进程运行不同N来模拟)
# ==============================================================
def exp3_strong_scaling(mpi, distributor):
    """固定问题规模, 当前GPU数下的性能"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    N = 8000  # 固定问题规模
    results = {"experiment": "强可扩展性", "matrix_size": N,
               "gpu_count": mpi.get_size(), "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    if mpi.is_master_process():
        A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
    else:
        A = B = None

    # 单GPU基线
    single_time = 0
    if mpi.is_master_process():
        def single_fn():
            torch.matmul(A, B)
        single_time = benchmark_fn(single_fn, repeats=3)

    # 分布式
    mpi.barrier()
    dist_times = []
    for _ in range(5):
        mpi.barrier()
        torch.cuda.synchronize()
        t0 = time.time()
        C = distributed_matmul(A, B, mpi, distributor)
        torch.cuda.synchronize()
        mpi.barrier()
        t1 = time.time()
        dist_times.append((t1 - t0) * 1000)
        if C is not None: del C
        torch.cuda.empty_cache()

    dist_time = float(np.median(dist_times))

    if mpi.is_master_process():
        speedup = single_time / dist_time if dist_time > 0 else 0
        efficiency = speedup / mpi.get_size() * 100

        results["data"].append({
            "num_gpus": mpi.get_size(),
            "single_gpu_ms": round(single_time, 2),
            "distributed_ms": round(dist_time, 2),
            "speedup": round(speedup, 3),
            "efficiency_pct": round(efficiency, 1),
        })
        print(f"  GPU数={mpi.get_size()}: 单GPU={single_time:.2f}ms, "
              f"分布式={dist_time:.2f}ms, 加速比={speedup:.2f}x, 效率={efficiency:.1f}%")

    if mpi.is_master_process():
        del A, B
        torch.cuda.empty_cache()
    mpi.barrier()

    save_result("exp3_strong_scaling", results, mpi)
    return results


# ==============================================================
#  实验 4: 弱可扩展性
# ==============================================================
def exp4_weak_scaling(mpi, distributor):
    """每GPU固定工作量, 当前GPU数下的性能"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    per_gpu_rows = 2000
    K = 2000
    N = 2000
    M = per_gpu_rows * mpi.get_size()

    results = {"experiment": "弱可扩展性", "per_gpu_rows": per_gpu_rows,
               "K": K, "N_col": N, "gpu_count": mpi.get_size(), "data": []}

    warm_up_gpu(mpi.get_gpu_id())
    mpi.print_master(f"  弱可扩展性: 每GPU {per_gpu_rows}行, 总M={M}")

    if mpi.is_master_process():
        A = torch.randn(M, K, device=f'cuda:{mpi.get_gpu_id()}')
        B = torch.randn(K, N, device=f'cuda:{mpi.get_gpu_id()}')
    else:
        A = B = None

    # 单GPU基线 (仅 per_gpu_rows 的工作量)
    single_time = 0
    if mpi.is_master_process():
        A_small = torch.randn(per_gpu_rows, K, device=f'cuda:{mpi.get_gpu_id()}')
        B_small = torch.randn(K, N, device=f'cuda:{mpi.get_gpu_id()}')
        def single_fn():
            torch.matmul(A_small, B_small)
        single_time = benchmark_fn(single_fn, repeats=3)
        del A_small, B_small
        torch.cuda.empty_cache()

    # 分布式
    mpi.barrier()
    dist_times = []
    for _ in range(5):
        mpi.barrier()
        torch.cuda.synchronize()
        t0 = time.time()
        C = distributed_matmul(A, B, mpi, distributor)
        torch.cuda.synchronize()
        mpi.barrier()
        t1 = time.time()
        dist_times.append((t1 - t0) * 1000)
        if C is not None: del C
        torch.cuda.empty_cache()

    dist_time = float(np.median(dist_times))

    if mpi.is_master_process():
        # 弱可扩展性效率 = T_1(单GPU单份工作) / T_p(p个GPU, p份工作)
        weak_eff = single_time / dist_time * 100 if dist_time > 0 else 0
        results["data"].append({
            "num_gpus": mpi.get_size(),
            "total_M": M,
            "single_gpu_ms": round(single_time, 2),
            "distributed_ms": round(dist_time, 2),
            "weak_efficiency_pct": round(weak_eff, 1),
        })
        print(f"  GPU数={mpi.get_size()}: 单GPU(单份)={single_time:.2f}ms, "
              f"分布式(p份)={dist_time:.2f}ms, 弱效率={weak_eff:.1f}%")

    if mpi.is_master_process():
        del A, B
        torch.cuda.empty_cache()
    mpi.barrier()

    save_result("exp4_weak_scaling", results, mpi)
    return results


# ==============================================================
#  实验 5: 创新算子对比
# ==============================================================
def exp5_innovation_comparison(mpi, distributor):
    """混合精度/稀疏感知/Kahan/Pencil FFT 对比"""
    from distributed_gpu.algorithms.matrix_ops import (
        distributed_matmul, distributed_matmul_mixed_precision,
        distributed_matmul_sparse_aware
    )
    from distributed_gpu.algorithms.reduction import distributed_sum, distributed_sum_kahan
    from distributed_gpu.algorithms.fft import distributed_fft2d, distributed_fft2d_pencil

    results = {"experiment": "创新算子对比", "gpu_count": mpi.get_size(), "data": {}}

    warm_up_gpu(mpi.get_gpu_id())

    # --- 5a: 混合精度 vs 标准矩阵乘法 ---
    mpi.print_master("  [5a] 混合精度 vs 标准 matmul")
    mp_results = []
    for N in [2000, 4000, 6000, 8000]:
        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 标准
        mpi.barrier()
        std_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C1 = distributed_matmul(A, B, mpi, distributor)
            torch.cuda.synchronize(); mpi.barrier()
            std_times.append((time.time() - t0) * 1000)
            if C1 is not None: del C1
            torch.cuda.empty_cache()

        # 混合精度
        mp_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C2 = distributed_matmul_mixed_precision(A, B, mpi, distributor)
            torch.cuda.synchronize(); mpi.barrier()
            mp_times.append((time.time() - t0) * 1000)
            if C2 is not None: del C2
            torch.cuda.empty_cache()

        # 精度对比
        error = 0
        if mpi.is_master_process():
            C_std = distributed_matmul(A, B, mpi, distributor)
        else:
            C_std = distributed_matmul(A, B, mpi, distributor)
        if mpi.is_master_process():
            C_mp = distributed_matmul_mixed_precision(A, B, mpi, distributor)
        else:
            C_mp = distributed_matmul_mixed_precision(A, B, mpi, distributor)

        if mpi.is_master_process():
            C_ref = torch.matmul(A, B)
            err_std = torch.mean(torch.abs(C_std - C_ref)).item()
            err_mp = torch.mean(torch.abs(C_mp - C_ref)).item()
            rel_err_std = err_std / torch.mean(torch.abs(C_ref)).item()
            rel_err_mp = err_mp / torch.mean(torch.abs(C_ref)).item()

            mp_results.append({
                "N": N,
                "standard_ms": round(float(np.median(std_times)), 2),
                "mixed_precision_ms": round(float(np.median(mp_times)), 2),
                "speedup": round(float(np.median(std_times)) / float(np.median(mp_times)), 3),
                "std_rel_error": float(f"{rel_err_std:.2e}"),
                "mp_rel_error": float(f"{rel_err_mp:.2e}"),
            })
            print(f"    N={N}: std={np.median(std_times):.1f}ms mp={np.median(mp_times):.1f}ms "
                  f"speedup={np.median(std_times)/np.median(mp_times):.2f}x")
            del C_std, C_mp, C_ref, A, B
            torch.cuda.empty_cache()
        mpi.barrier()

    results["data"]["mixed_precision"] = mp_results

    # --- 5b: 稀疏感知 ---
    mpi.print_master("  [5b] 稀疏感知 matmul")
    sparse_results = []
    N = 4000
    for sparsity in [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]:
        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            mask = torch.rand(N, N, device=f'cuda:{mpi.get_gpu_id()}') > sparsity
            B = B * mask.float()
        else:
            A = B = None

        # 标准
        mpi.barrier()
        std_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C1 = distributed_matmul(A, B, mpi, distributor)
            torch.cuda.synchronize(); mpi.barrier()
            std_times.append((time.time() - t0) * 1000)
            if C1 is not None: del C1
            torch.cuda.empty_cache()

        # 稀疏感知
        sp_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C2 = distributed_matmul_sparse_aware(A, B, mpi, distributor, sparsity_threshold=0.5)
            torch.cuda.synchronize(); mpi.barrier()
            sp_times.append((time.time() - t0) * 1000)
            if C2 is not None: del C2
            torch.cuda.empty_cache()

        if mpi.is_master_process():
            sparse_results.append({
                "sparsity": sparsity,
                "standard_ms": round(float(np.median(std_times)), 2),
                "sparse_aware_ms": round(float(np.median(sp_times)), 2),
                "speedup": round(float(np.median(std_times)) / max(float(np.median(sp_times)), 0.01), 3),
            })
            print(f"    sparsity={sparsity}: std={np.median(std_times):.1f}ms "
                  f"sparse={np.median(sp_times):.1f}ms")
            del A, B
            torch.cuda.empty_cache()
        mpi.barrier()

    results["data"]["sparse_aware"] = sparse_results

    # --- 5c: Kahan 补偿求和精度 ---
    mpi.print_master("  [5c] Kahan 补偿求和精度")
    kahan_results = []
    for N in [100000, 1000000, 10000000, 50000000]:
        if mpi.is_master_process():
            # 创建含小数值的数组, 测试数值稳定性
            data = torch.ones(N, device=f'cuda:{mpi.get_gpu_id()}') * 1.0
            data[0] = 1e8
            data[-1] = -1e8
            # 精确结果应为 (N-2) * 1.0
            exact = float(N - 2)
        else:
            data = None
            exact = 0

        # 标准求和
        std_result = distributed_sum(data, mpi, distributor)
        # Kahan 求和
        kahan_result = distributed_sum_kahan(data, mpi, distributor)

        if mpi.is_master_process():
            exact = float(N - 2)
            std_err = abs(std_result.item() - exact)
            kahan_err = abs(kahan_result.item() - exact)
            kahan_results.append({
                "N": N,
                "exact_value": exact,
                "standard_sum": float(std_result.item()),
                "kahan_sum": float(kahan_result.item()),
                "standard_abs_error": float(f"{std_err:.6e}"),
                "kahan_abs_error": float(f"{kahan_err:.6e}"),
                "improvement_factor": round(std_err / max(kahan_err, 1e-30), 1),
            })
            print(f"    N={N}: std_err={std_err:.2e}, kahan_err={kahan_err:.2e}")
            del data
            torch.cuda.empty_cache()
        mpi.barrier()

    results["data"]["kahan_sum"] = kahan_results

    # --- 5d: Pencil FFT vs Batch FFT ---
    mpi.print_master("  [5d] Pencil FFT vs Batch FFT")
    fft_results = []
    P = mpi.get_size()
    for grid_size in [256, 512, 1024, 2048]:
        # 要求 grid_size 能被 P 整除
        if grid_size % P != 0:
            continue
        if mpi.is_master_process():
            data_batch = torch.randn(P, grid_size, grid_size,
                                      device=f'cuda:{mpi.get_gpu_id()}')
            data_pencil = torch.randn(grid_size, grid_size,
                                       device=f'cuda:{mpi.get_gpu_id()}')
        else:
            data_batch = data_pencil = None

        # Batch FFT
        mpi.barrier()
        batch_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            r1 = distributed_fft2d(data_batch, mpi, distributor)
            torch.cuda.synchronize(); mpi.barrier()
            batch_times.append((time.time() - t0) * 1000)
            if r1 is not None: del r1
            torch.cuda.empty_cache()

        # Pencil FFT
        pencil_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            r2 = distributed_fft2d_pencil(data_pencil, mpi, distributor)
            torch.cuda.synchronize(); mpi.barrier()
            pencil_times.append((time.time() - t0) * 1000)
            if r2 is not None: del r2
            torch.cuda.empty_cache()

        if mpi.is_master_process():
            fft_results.append({
                "grid_size": grid_size,
                "batch_fft_ms": round(float(np.median(batch_times)), 2),
                "pencil_fft_ms": round(float(np.median(pencil_times)), 2),
            })
            print(f"    grid={grid_size}: batch={np.median(batch_times):.1f}ms "
                  f"pencil={np.median(pencil_times):.1f}ms")
            del data_batch, data_pencil
            torch.cuda.empty_cache()
        mpi.barrier()

    results["data"]["pencil_fft"] = fft_results

    save_result("exp5_innovation", results, mpi)
    return results


# ==============================================================
#  实验 6: 流水线优化效果
# ==============================================================
def exp6_pipeline(mpi, distributor):
    """流水线(计算-通信重叠) vs 非流水线"""
    from distributed_gpu.pipeline_optimizer import PipelineOptimizer, PipelineConfig

    pipe = PipelineOptimizer(mpi, PipelineConfig(num_chunks=4))
    results = {"experiment": "流水线优化", "gpu_count": mpi.get_size(), "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    for N in [2000, 4000, 6000, 8000, 10000]:
        mpi.print_master(f"  矩阵规模: {N}x{N}")

        if mpi.is_master_process():
            A = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(N, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        # 非流水线
        mpi.barrier()
        baseline_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C1 = pipe.baseline_matmul(A, B)
            torch.cuda.synchronize(); mpi.barrier()
            baseline_times.append((time.time() - t0) * 1000)
            if C1 is not None: del C1
            torch.cuda.empty_cache()

        # 流水线 (不同chunk数)
        for num_chunks in [2, 4, 8]:
            pipe_times = []
            for _ in range(3):
                mpi.barrier(); torch.cuda.synchronize()
                t0 = time.time()
                C2 = pipe.pipelined_matmul(A, B, num_chunks=num_chunks)
                torch.cuda.synchronize(); mpi.barrier()
                pipe_times.append((time.time() - t0) * 1000)
                if C2 is not None: del C2
                torch.cuda.empty_cache()

            if mpi.is_master_process():
                bl = float(np.median(baseline_times))
                pl = float(np.median(pipe_times))
                results["data"].append({
                    "matrix_size": N,
                    "num_chunks": num_chunks,
                    "baseline_ms": round(bl, 2),
                    "pipeline_ms": round(pl, 2),
                    "speedup": round(bl / pl if pl > 0 else 0, 3),
                })
                print(f"    chunks={num_chunks}: baseline={bl:.1f}ms pipeline={pl:.1f}ms "
                      f"加速={bl/pl:.2f}x")

        if mpi.is_master_process():
            del A, B
            torch.cuda.empty_cache()
        mpi.barrier()

    save_result("exp6_pipeline", results, mpi)
    return results


# ==============================================================
#  实验 7: 代价模型策略选择
# ==============================================================
def exp7_cost_model(mpi, distributor):
    """代价模型 vs 固定策略对比"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul

    config = ClusterConfig.from_auto_detect(mpi.get_size())
    cost_model = CostModel(config)

    results = {"experiment": "代价模型策略", "gpu_count": mpi.get_size(), "data": []}

    warm_up_gpu(mpi.get_gpu_id())

    # 不同形状矩阵
    shapes = [
        (8000, 2000, 2000, "M>>N"),
        (2000, 2000, 8000, "N>>M"),
        (4000, 4000, 4000, "M≈N"),
        (10000, 1000, 1000, "M>>>N"),
        (1000, 1000, 10000, "N>>>M"),
    ]

    for M, K, N, desc in shapes:
        mpi.print_master(f"  [{desc}] A[{M},{K}] @ B[{K},{N}]")

        if mpi.is_master_process():
            A = torch.randn(M, K, device=f'cuda:{mpi.get_gpu_id()}')
            B = torch.randn(K, N, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            A = B = None

        strategy_results = {}

        # 代价模型自动选择
        mpi.barrier()
        auto_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C = distributed_matmul(A, B, mpi, distributor, cost_model=cost_model)
            torch.cuda.synchronize(); mpi.barrier()
            auto_times.append((time.time() - t0) * 1000)
            if C is not None: del C
            torch.cuda.empty_cache()

        if mpi.is_master_process():
            plan = cost_model.find_optimal_strategy(M, K, N)
            strategy_results["auto"] = {
                "strategy": plan.strategy.value,
                "time_ms": round(float(np.median(auto_times)), 2),
            }

        # 固定行分割
        mpi.barrier()
        row_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C = distributed_matmul(A, B, mpi, distributor, strategy=SplitStrategy.ROW_SPLIT)
            torch.cuda.synchronize(); mpi.barrier()
            row_times.append((time.time() - t0) * 1000)
            if C is not None: del C
            torch.cuda.empty_cache()

        if mpi.is_master_process():
            strategy_results["row_split"] = round(float(np.median(row_times)), 2)

        # 固定列分割
        mpi.barrier()
        col_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            C = distributed_matmul(A, B, mpi, distributor, strategy=SplitStrategy.COLUMN_SPLIT)
            torch.cuda.synchronize(); mpi.barrier()
            col_times.append((time.time() - t0) * 1000)
            if C is not None: del C
            torch.cuda.empty_cache()

        if mpi.is_master_process():
            strategy_results["col_split"] = round(float(np.median(col_times)), 2)

            results["data"].append({
                "shape": f"[{M},{K}]@[{K},{N}]",
                "description": desc,
                "auto_strategy": strategy_results["auto"]["strategy"],
                "auto_time_ms": strategy_results["auto"]["time_ms"],
                "row_split_ms": strategy_results["row_split"],
                "col_split_ms": strategy_results["col_split"],
            })
            print(f"    auto({strategy_results['auto']['strategy']})="
                  f"{strategy_results['auto']['time_ms']:.1f}ms, "
                  f"row={strategy_results['row_split']:.1f}ms, "
                  f"col={strategy_results['col_split']:.1f}ms")
            del A, B
            torch.cuda.empty_cache()
        mpi.barrier()

    save_result("exp7_cost_model", results, mpi)
    return results


# ==============================================================
#  实验 8: 科学计算应用场景
# ==============================================================
def exp8_applications(mpi, distributor):
    """Stencil/Jacobi/Conv2d/Einsum 应用"""
    from distributed_gpu.algorithms.stencil import distributed_stencil_2d, distributed_jacobi_2d
    from distributed_gpu.algorithms.convolution import distributed_conv2d
    from distributed_gpu.algorithms.einsum import distributed_einsum

    results = {"experiment": "科学计算应用", "gpu_count": mpi.get_size(), "data": {}}

    warm_up_gpu(mpi.get_gpu_id())
    P = mpi.get_size()

    # --- 8a: Stencil 热传导模拟 ---
    mpi.print_master("  [8a] Stencil 热传导")
    stencil_results = []
    for grid_size in [256, 512, 1024, 2048]:
        if grid_size % P != 0:
            continue
        for iters in [10, 50, 100]:
            if mpi.is_master_process():
                grid = torch.zeros(grid_size, grid_size, device=f'cuda:{mpi.get_gpu_id()}')
                # 初始热源: 中心点
                cx, cy = grid_size // 2, grid_size // 2
                grid[cx-5:cx+5, cy-5:cy+5] = 100.0
            else:
                grid = None

            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            result = distributed_stencil_2d(grid, mpi, distributor, iterations=iters)
            torch.cuda.synchronize(); mpi.barrier()
            elapsed = (time.time() - t0) * 1000

            if mpi.is_master_process():
                stencil_results.append({
                    "grid_size": grid_size,
                    "iterations": iters,
                    "time_ms": round(elapsed, 2),
                    "iter_per_sec": round(iters / elapsed * 1000, 1),
                })
                print(f"    grid={grid_size} iters={iters}: {elapsed:.1f}ms")
                del grid, result
                torch.cuda.empty_cache()
            mpi.barrier()

    results["data"]["stencil"] = stencil_results

    # --- 8b: Jacobi 泊松方程 ---
    mpi.print_master("  [8b] Jacobi 泊松方程")
    jacobi_results = []
    for grid_size in [256, 512, 1024]:
        if grid_size % P != 0:
            continue
        if mpi.is_master_process():
            u = torch.zeros(grid_size, grid_size, device=f'cuda:{mpi.get_gpu_id()}')
            rhs = torch.zeros(grid_size, grid_size, device=f'cuda:{mpi.get_gpu_id()}')
            rhs[grid_size//2, grid_size//2] = 1.0
        else:
            u = rhs = None

        mpi.barrier(); torch.cuda.synchronize()
        t0 = time.time()
        result = distributed_jacobi_2d(u, rhs, mpi, distributor,
                                        iterations=200, tol=1e-6)
        torch.cuda.synchronize(); mpi.barrier()
        elapsed = (time.time() - t0) * 1000

        if mpi.is_master_process():
            jacobi_results.append({
                "grid_size": grid_size,
                "max_iters": 200,
                "time_ms": round(elapsed, 2),
            })
            print(f"    grid={grid_size}: {elapsed:.1f}ms")
            del u, rhs, result
            torch.cuda.empty_cache()
        mpi.barrier()

    results["data"]["jacobi"] = jacobi_results

    # --- 8c: 分布式卷积 (深度学习风格) ---
    mpi.print_master("  [8c] 分布式 Conv2d")
    conv_results = []
    for batch in [16, 32, 64, 128]:
        if batch < P:
            continue
        if mpi.is_master_process():
            inp = torch.randn(batch, 64, 56, 56, device=f'cuda:{mpi.get_gpu_id()}')
            weight = torch.randn(128, 64, 3, 3, device=f'cuda:{mpi.get_gpu_id()}')
            bias = torch.randn(128, device=f'cuda:{mpi.get_gpu_id()}')
        else:
            inp = weight = bias = None

        # 分布式
        mpi.barrier()
        dist_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            out = distributed_conv2d(inp, weight, mpi, distributor, bias=bias, padding=(1, 1))
            torch.cuda.synchronize(); mpi.barrier()
            dist_times.append((time.time() - t0) * 1000)
            if out is not None: del out
            torch.cuda.empty_cache()

        # 单GPU
        single_time = 0
        if mpi.is_master_process():
            import torch.nn.functional as F
            def single_fn():
                F.conv2d(inp, weight, bias, padding=(1, 1))
            single_time = benchmark_fn(single_fn, repeats=3)

        if mpi.is_master_process():
            dt = float(np.median(dist_times))
            conv_results.append({
                "batch_size": batch,
                "single_gpu_ms": round(single_time, 2),
                "distributed_ms": round(dt, 2),
                "speedup": round(single_time / dt if dt > 0 else 0, 3),
            })
            print(f"    batch={batch}: single={single_time:.1f}ms dist={dt:.1f}ms")
            del inp, weight, bias
            torch.cuda.empty_cache()
        mpi.barrier()

    results["data"]["conv2d"] = conv_results

    # --- 8d: Einsum 张量收缩 ---
    mpi.print_master("  [8d] Einsum 张量收缩")
    einsum_results = []
    equations = [
        ("ij,jk->ik", [(2000, 1000), (1000, 2000)], "矩阵乘法"),
        ("bij,bjk->bik", [(32, 256, 128), (32, 128, 256)], "批量矩阵乘法"),
        ("ijkl,klmn->ijmn", [(16, 16, 32, 32), (32, 32, 16, 16)], "4阶张量收缩"),
    ]
    for eq, shapes, desc in equations:
        if mpi.is_master_process():
            ops = tuple(torch.randn(*s, device=f'cuda:{mpi.get_gpu_id()}') for s in shapes)
        else:
            ops = tuple(None for _ in shapes)

        mpi.barrier()
        dist_times = []
        for _ in range(3):
            mpi.barrier(); torch.cuda.synchronize()
            t0 = time.time()
            r = distributed_einsum(eq, *ops, mpi=mpi, distributor=distributor)
            torch.cuda.synchronize(); mpi.barrier()
            dist_times.append((time.time() - t0) * 1000)
            if r is not None: del r
            torch.cuda.empty_cache()

        if mpi.is_master_process():
            einsum_results.append({
                "equation": eq,
                "shapes": [list(s) for s in shapes],
                "description": desc,
                "time_ms": round(float(np.median(dist_times)), 2),
            })
            print(f"    {eq} ({desc}): {np.median(dist_times):.1f}ms")
            for op in ops:
                if op is not None: del op
            torch.cuda.empty_cache()
        mpi.barrier()

    results["data"]["einsum"] = einsum_results

    save_result("exp8_applications", results, mpi)
    return results


# ==============================================================
#  Main
# ==============================================================
def main():
    mpi = MPIManager()
    distributor = TensorDistributor(mpi)

    exp_id = sys.argv[1] if len(sys.argv) > 1 else "all"

    experiments = {
        "1": ("实验1: 计算性能对比", exp1_compute_performance),
        "2": ("实验2: 通信开销分析", exp2_communication_overhead),
        "3": ("实验3: 强可扩展性", exp3_strong_scaling),
        "4": ("实验4: 弱可扩展性", exp4_weak_scaling),
        "5": ("实验5: 创新算子对比", exp5_innovation_comparison),
        "6": ("实验6: 流水线优化", exp6_pipeline),
        "7": ("实验7: 代价模型策略", exp7_cost_model),
        "8": ("实验8: 科学计算应用", exp8_applications),
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

    mpi.print_master("\n全部实验完成!")


if __name__ == "__main__":
    main()
