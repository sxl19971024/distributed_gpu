#!/usr/bin/env python
"""
增强版 — 生成硕士论文所需的全部图表 (基于增强版实验数据)
包含误差棒, 更多子图, 更丰富的对比
"""
import os, json, glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "thesis_enhanced")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "thesis_enhanced", "figures")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_json(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    results = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            base = os.path.basename(f)
            results[base] = data
    return results


# ================================================================
#  图1: 计算性能对比 (含误差棒)
# ================================================================
def fig1_compute_performance():
    files = load_json("exp1_compute_performance_n*.json")
    if not files:
        print("跳过: exp1 数据不存在"); return

    for fname, data in files.items():
        d = data["data"]
        sizes = [x["matrix_size"] for x in d]
        single = [x["single_gpu_mean_ms"] for x in d]
        single_err = [x["single_gpu_std_ms"] for x in d]
        dist = [x["distributed_mean_ms"] for x in d]
        dist_err = [x["distributed_std_ms"] for x in d]
        speedup = [x["speedup"] for x in d]
        efficiency = [x["efficiency_pct"] for x in d]
        gflops = [x.get("distributed_gflops", 0) for x in d]
        n_gpu = data["gpu_count"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (0,0) 时间对比柱状图(含误差棒)
        ax = axes[0, 0]
        x = np.arange(len(sizes))
        width = 0.35
        ax.bar(x - width/2, single, width, yerr=single_err, capsize=3,
               label='Single GPU', color='#4472C4', alpha=0.85)
        ax.bar(x + width/2, dist, width, yerr=dist_err, capsize=3,
               label=f'Distributed ({n_gpu} GPUs)', color='#ED7D31', alpha=0.85)
        ax.set_xlabel('Matrix Size (N x N)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Computation Time Comparison ({n_gpu} GPUs)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # (0,1) 加速比
        ax = axes[0, 1]
        ax.plot(sizes, speedup, 'o-', color='#70AD47', linewidth=2, markersize=7)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1x)')
        ax.axhline(y=n_gpu, color='blue', linestyle='--', alpha=0.5, label=f'Ideal ({n_gpu}x)')
        ax.set_xlabel('Matrix Size (N x N)')
        ax.set_ylabel('Speedup')
        ax.set_title(f'Speedup Ratio ({n_gpu} GPUs)')
        ax.legend()
        ax.grid(alpha=0.3)

        # (1,0) 并行效率
        ax = axes[1, 0]
        colors_eff = ['#70AD47' if e > 20 else '#FFC000' if e > 10 else '#C00000' for e in efficiency]
        ax.bar(range(len(sizes)), efficiency, color=colors_eff)
        ax.set_xlabel('Matrix Size (N x N)')
        ax.set_ylabel('Parallel Efficiency (%)')
        ax.set_title('Parallel Efficiency vs Matrix Size')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # (1,1) GFLOPS
        ax = axes[1, 1]
        ax.plot(sizes, gflops, 's-', color='#7030A0', linewidth=2, markersize=7)
        ax.set_xlabel('Matrix Size (N x N)')
        ax.set_ylabel('GFLOPS')
        ax.set_title('Distributed Computation Throughput')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"fig1_compute_perf_n{n_gpu}.png")
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f"  生成: {out}")


# ================================================================
#  图2: 通信开销分析 (含误差棒)
# ================================================================
def fig2_comm_overhead():
    files = load_json("exp2_comm_overhead_n*.json")
    if not files:
        print("跳过: exp2 数据不存在"); return

    for fname, data in files.items():
        d = data["data"]
        sizes = [x["matrix_size"] for x in d]
        scatter = [x["scatter_mean_ms"] for x in d]
        compute = [x["compute_mean_ms"] for x in d]
        gather = [x["gather_mean_ms"] for x in d]
        comm_pct = [x["comm_ratio_pct"] for x in d]
        comp_pct = [x["compute_ratio_pct"] for x in d]
        n_gpu = data["gpu_count"]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 堆叠柱状图
        ax = axes[0]
        x = np.arange(len(sizes))
        ax.bar(x, scatter, label='Scatter', color='#4472C4')
        ax.bar(x, compute, bottom=scatter, label='Compute', color='#70AD47')
        ax.bar(x, gather, bottom=[s+c for s,c in zip(scatter, compute)],
                label='Gather', color='#ED7D31')
        ax.set_xlabel('Matrix Size (N x N)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Time Breakdown ({n_gpu} GPUs)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

        # 通信 vs 计算占比
        ax = axes[1]
        ax.stackplot(range(len(sizes)),
                     [comm_pct, comp_pct],
                     labels=['Communication', 'Computation'],
                     colors=['#C00000', '#70AD47'], alpha=0.7)
        ax.set_xlabel('Matrix Size (N x N)')
        ax.set_ylabel('Ratio (%)')
        ax.set_title('Communication vs Computation Ratio')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
        ax.legend(loc='center right')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

        # 通信占比折线
        ax = axes[2]
        ax.plot(sizes, comm_pct, 's-', color='#C00000', linewidth=2, markersize=8)
        ax.set_xlabel('Matrix Size (N x N)')
        ax.set_ylabel('Communication Ratio (%)')
        ax.set_title('Communication Overhead Trend')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"fig2_comm_overhead_n{n_gpu}.png")
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f"  生成: {out}")


# ================================================================
#  图3: 强可扩展性 (多矩阵规模)
# ================================================================
def fig3_strong_scaling():
    files = load_json("exp3_strong_scaling_n*.json")
    if not files:
        print("跳过: exp3 数据不存在"); return

    # 按GPU数分组, 每组有多个矩阵规模
    all_data = {}  # {gpu_count: [{matrix_size, speedup, efficiency, ...}]}
    for fname in sorted(files.keys()):
        data = files[fname]
        gpu_count = data["gpu_count"]
        all_data[gpu_count] = data["data"]

    gpu_counts = sorted(all_data.keys())
    # 提取所有矩阵规模
    matrix_sizes = sorted(set(item["matrix_size"] for items in all_data.values() for item in items))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (0) 加速比 vs GPU数 (每条线一个矩阵规模)
    ax = axes[0]
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#C00000', '#7030A0']
    for i, ms in enumerate(matrix_sizes):
        gcs = []
        sps = []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["matrix_size"] == ms]
            if match:
                gcs.append(gc)
                sps.append(match[0]["speedup"])
        ax.plot(gcs, sps, 'o-', color=colors[i % len(colors)], linewidth=2,
                markersize=8, label=f'N={ms}')
    ax.plot(gpu_counts, gpu_counts, '--', color='gray', alpha=0.5, label='Ideal')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Speedup')
    ax.set_title('Strong Scaling: Speedup')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(gpu_counts)

    # (1) 并行效率 vs GPU数
    ax = axes[1]
    for i, ms in enumerate(matrix_sizes):
        gcs = []
        effs = []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["matrix_size"] == ms]
            if match:
                gcs.append(gc)
                effs.append(match[0]["efficiency_pct"])
        ax.plot(gcs, effs, 's-', color=colors[i % len(colors)], linewidth=2,
                markersize=8, label=f'N={ms}')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('Strong Scaling: Parallel Efficiency')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(gpu_counts)

    # (2) 执行时间 vs GPU数
    ax = axes[2]
    for i, ms in enumerate(matrix_sizes):
        gcs = []
        times = []
        stds = []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["matrix_size"] == ms]
            if match:
                gcs.append(gc)
                times.append(match[0]["distributed_mean_ms"])
                stds.append(match[0]["distributed_std_ms"])
        ax.errorbar(gcs, times, yerr=stds, fmt='o-', color=colors[i % len(colors)],
                    linewidth=2, markersize=8, capsize=3, label=f'N={ms}')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Strong Scaling: Execution Time')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(gpu_counts)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3_strong_scaling.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图4: 弱可扩展性 (多per_gpu_rows)
# ================================================================
def fig4_weak_scaling():
    files = load_json("exp4_weak_scaling_n*.json")
    if not files:
        print("跳过: exp4 数据不存在"); return

    all_data = {}
    for fname in sorted(files.keys()):
        data = files[fname]
        gpu_count = data["gpu_count"]
        all_data[gpu_count] = data["data"]

    gpu_counts = sorted(all_data.keys())
    per_gpu_rows_list = sorted(set(item["per_gpu_rows"] for items in all_data.values() for item in items))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = ['#4472C4', '#ED7D31', '#70AD47', '#C00000']

    # (0) 弱效率 vs GPU数
    ax = axes[0]
    for i, pgr in enumerate(per_gpu_rows_list):
        gcs = []
        effs = []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["per_gpu_rows"] == pgr]
            if match:
                gcs.append(gc)
                effs.append(match[0]["weak_efficiency_pct"])
        ax.plot(gcs, effs, 'o-', color=colors[i % len(colors)], linewidth=2,
                markersize=8, label=f'{pgr} rows/GPU')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Weak Scaling Efficiency (%)')
    ax.set_title('Weak Scaling: Efficiency')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(gpu_counts)

    # (1) 执行时间 vs GPU数
    ax = axes[1]
    for i, pgr in enumerate(per_gpu_rows_list):
        gcs = []
        times = []
        stds = []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["per_gpu_rows"] == pgr]
            if match:
                gcs.append(gc)
                times.append(match[0]["distributed_mean_ms"])
                stds.append(match[0]["distributed_std_ms"])
        ax.errorbar(gcs, times, yerr=stds, fmt='s-', color=colors[i % len(colors)],
                    linewidth=2, markersize=8, capsize=3, label=f'{pgr} rows/GPU')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Weak Scaling: Execution Time')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(gpu_counts)

    # (2) 热力图: 效率矩阵
    ax = axes[2]
    eff_matrix = np.zeros((len(per_gpu_rows_list), len(gpu_counts)))
    for i, pgr in enumerate(per_gpu_rows_list):
        for j, gc in enumerate(gpu_counts):
            match = [item for item in all_data[gc] if item["per_gpu_rows"] == pgr]
            if match:
                eff_matrix[i, j] = match[0]["weak_efficiency_pct"]
    im = ax.imshow(eff_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(gpu_counts)))
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_yticks(range(len(per_gpu_rows_list)))
    ax.set_yticklabels([str(p) for p in per_gpu_rows_list])
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Rows per GPU')
    ax.set_title('Weak Efficiency Heatmap (%)')
    for i in range(len(per_gpu_rows_list)):
        for j in range(len(gpu_counts)):
            ax.text(j, i, f'{eff_matrix[i,j]:.1f}', ha='center', va='center',
                    fontsize=10, color='black' if eff_matrix[i,j] > 30 else 'white')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig4_weak_scaling.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图5: 混合精度 (含误差棒)
# ================================================================
def fig5_mixed_precision():
    files = load_json("exp5_innovation_n*.json")
    if not files:
        print("跳过: exp5 数据不存在"); return

    data = list(files.values())[0]
    mp_data = data["data"].get("mixed_precision", [])
    if not mp_data:
        return

    sizes = [x["N"] for x in mp_data]
    std_t = [x["standard_mean_ms"] for x in mp_data]
    std_err = [x["standard_std_ms"] for x in mp_data]
    mp_t = [x["mixed_precision_mean_ms"] for x in mp_data]
    mp_err = [x["mixed_precision_std_ms"] for x in mp_data]
    speedups = [x["speedup"] for x in mp_data]
    rel_err_std = [x.get("std_rel_error", 0) for x in mp_data]
    rel_err_mp = [x.get("mp_rel_error", 0) for x in mp_data]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 时间对比
    ax = axes[0]
    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width/2, std_t, width, yerr=std_err, capsize=3,
           label='Standard (FP32)', color='#4472C4')
    ax.bar(x + width/2, mp_t, width, yerr=mp_err, capsize=3,
           label='Mixed Precision', color='#ED7D31')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Mixed Precision vs Standard MatMul')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 加速比
    ax = axes[1]
    ax.bar(range(len(sizes)), speedups, color='#70AD47')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Speedup')
    ax.set_title('Mixed Precision Speedup')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # 精度对比
    ax = axes[2]
    x = np.arange(len(sizes))
    width = 0.35
    ax.bar(x - width/2, rel_err_std, width, label='Standard', color='#4472C4')
    ax.bar(x + width/2, rel_err_mp, width, label='Mixed Precision', color='#ED7D31')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Relative Error')
    ax.set_title('Numerical Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig5_mixed_precision.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图6: Kahan 补偿求和精度
# ================================================================
def fig6_kahan():
    files = load_json("exp5_innovation_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    kahan_data = data["data"].get("kahan_sum", [])
    if not kahan_data:
        return

    ns = [x["N"] for x in kahan_data]
    std_err = [x["standard_abs_error_mean"] for x in kahan_data]
    kahan_err = [max(x["kahan_abs_error_mean"], 1e-15) for x in kahan_data]
    improvement = [x["improvement_factor"] for x in kahan_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 误差对比
    x = np.arange(len(ns))
    width = 0.35
    ax1.bar(x - width/2, std_err, width, label='Standard Sum', color='#C00000')
    ax1.bar(x + width/2, kahan_err, width, label='Kahan Sum', color='#70AD47')
    ax1.set_xlabel('Number of Elements')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Numerical Accuracy: Standard vs Kahan')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{n:.0e}' for n in ns], rotation=45, fontsize=9)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 改善倍数
    ax2.bar(range(len(ns)), improvement, color='#7030A0')
    ax2.set_xlabel('Number of Elements')
    ax2.set_ylabel('Improvement Factor (x)')
    ax2.set_title('Kahan Sum Improvement over Standard')
    ax2.set_xticks(range(len(ns)))
    ax2.set_xticklabels([f'{n:.0e}' for n in ns], rotation=45, fontsize=9)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig6_kahan_accuracy.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图7: 稀疏感知 (多规模)
# ================================================================
def fig7_sparse():
    files = load_json("exp5_innovation_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    sparse = data["data"].get("sparse_aware", [])
    if not sparse:
        return

    # 按矩阵规模分组
    Ns = sorted(set(x["N"] for x in sparse))

    fig, axes = plt.subplots(1, len(Ns), figsize=(7*len(Ns), 6))
    if len(Ns) == 1:
        axes = [axes]

    colors_std = '#4472C4'
    colors_sp = '#ED7D31'

    for idx, N in enumerate(Ns):
        ax = axes[idx]
        items = [x for x in sparse if x["N"] == N]
        sparsity = [x["sparsity"] for x in items]
        std_t = [x["standard_mean_ms"] for x in items]
        std_err = [x["standard_std_ms"] for x in items]
        sp_t = [x["sparse_aware_mean_ms"] for x in items]
        sp_err = [x["sparse_aware_std_ms"] for x in items]

        ax.errorbar(sparsity, std_t, yerr=std_err, fmt='o-', color=colors_std,
                    linewidth=2, markersize=6, capsize=3, label='Standard Dense')
        ax.errorbar(sparsity, sp_t, yerr=sp_err, fmt='s-', color=colors_sp,
                    linewidth=2, markersize=6, capsize=3, label='Sparse-Aware')
        ax.set_xlabel('Sparsity Ratio')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Sparse-Aware MatMul (N={N})')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig7_sparse_aware.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图8: Pencil FFT
# ================================================================
def fig8_pencil_fft():
    files = load_json("exp5_innovation_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    fft_data = data["data"].get("pencil_fft", [])
    if not fft_data:
        return

    grids = [x["grid_size"] for x in fft_data]
    batch_t = [x["batch_fft_mean_ms"] for x in fft_data]
    batch_err = [x["batch_fft_std_ms"] for x in fft_data]
    pencil_t = [x["pencil_fft_mean_ms"] for x in fft_data]
    pencil_err = [x["pencil_fft_std_ms"] for x in fft_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(grids))
    width = 0.35
    ax1.bar(x - width/2, batch_t, width, yerr=batch_err, capsize=3,
            label='Batch FFT', color='#4472C4')
    ax1.bar(x + width/2, pencil_t, width, yerr=pencil_err, capsize=3,
            label='Pencil FFT', color='#ED7D31')
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Batch FFT vs Pencil FFT')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(g) for g in grids])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 折线对比
    ax2.plot(grids, batch_t, 'o-', color='#4472C4', linewidth=2, markersize=8, label='Batch FFT')
    ax2.plot(grids, pencil_t, 's-', color='#ED7D31', linewidth=2, markersize=8, label='Pencil FFT')
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('FFT Performance Trend')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig8_pencil_fft.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图9: 流水线优化 (含误差棒, 热力图)
# ================================================================
def fig9_pipeline():
    files = load_json("exp6_pipeline_n*.json")
    if not files:
        print("跳过: exp6 数据不存在"); return

    data = list(files.values())[0]
    d = data["data"]

    sizes = sorted(set(x["matrix_size"] for x in d))
    chunks_list = sorted(set(x["num_chunks"] for x in d))

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # (0) 柱状图对比
    ax = axes[0]
    baselines = {}
    for item in d:
        baselines[item["matrix_size"]] = item["baseline_mean_ms"]

    x = np.arange(len(sizes))
    width = 0.18
    ax.bar(x - 2*width, [baselines[s] for s in sizes], width,
           label='Baseline', color='#A5A5A5')
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#7030A0']
    for i, nc in enumerate(chunks_list):
        vals = []
        errs = []
        for s in sizes:
            match = [item for item in d if item["matrix_size"]==s and item["num_chunks"]==nc]
            vals.append(match[0]["pipeline_mean_ms"] if match else 0)
            errs.append(match[0]["pipeline_std_ms"] if match else 0)
        ax.bar(x + (i-1)*width, vals, width, yerr=errs, capsize=2,
               label=f'Pipeline (chunks={nc})', color=colors[i % len(colors)])
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Pipeline Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # (1) 加速比折线
    ax = axes[1]
    for i, nc in enumerate(chunks_list):
        ss = []
        sps = []
        for s in sizes:
            match = [item for item in d if item["matrix_size"]==s and item["num_chunks"]==nc]
            if match:
                ss.append(s)
                sps.append(match[0]["speedup"])
        ax.plot(ss, sps, 'o-', color=colors[i % len(colors)], linewidth=2,
                markersize=7, label=f'chunks={nc}')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Speedup')
    ax.set_title('Pipeline Speedup')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (2) 加速比热力图
    ax = axes[2]
    speedup_matrix = np.zeros((len(chunks_list), len(sizes)))
    for i, nc in enumerate(chunks_list):
        for j, s in enumerate(sizes):
            match = [item for item in d if item["matrix_size"]==s and item["num_chunks"]==nc]
            if match:
                speedup_matrix[i, j] = match[0]["speedup"]
    im = ax.imshow(speedup_matrix, cmap='YlGn', aspect='auto', vmin=0.5, vmax=2.0)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
    ax.set_yticks(range(len(chunks_list)))
    ax.set_yticklabels([str(c) for c in chunks_list])
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Num Chunks')
    ax.set_title('Pipeline Speedup Heatmap')
    for i in range(len(chunks_list)):
        for j in range(len(sizes)):
            ax.text(j, i, f'{speedup_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=9)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig9_pipeline.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图10: 代价模型策略选择 (含最优标记)
# ================================================================
def fig10_cost_model():
    files = load_json("exp7_cost_model_n*.json")
    if not files:
        print("跳过: exp7 数据不存在"); return

    data = list(files.values())[0]
    d = data["data"]

    labels = [x["description"] for x in d]
    auto_t = [x["auto_mean_ms"] for x in d]
    auto_err = [x["auto_std_ms"] for x in d]
    row_t = [x["row_split_mean_ms"] for x in d]
    row_err = [x["row_split_std_ms"] for x in d]
    col_t = [x["col_split_mean_ms"] for x in d]
    col_err = [x["col_split_std_ms"] for x in d]
    optimal = [x.get("auto_is_optimal", False) for x in d]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 柱状图
    x = np.arange(len(labels))
    width = 0.25
    ax1.bar(x - width, row_t, width, yerr=row_err, capsize=3,
            label='Row Split', color='#4472C4')
    ax1.bar(x, col_t, width, yerr=col_err, capsize=3,
            label='Column Split', color='#ED7D31')
    ax1.bar(x + width, auto_t, width, yerr=auto_err, capsize=3,
            label='Auto (Cost Model)', color='#70AD47')
    # 标记最优
    for i, opt in enumerate(optimal):
        if opt:
            ax1.annotate('✓', xy=(i + width, auto_t[i]), fontsize=14,
                        ha='center', va='bottom', color='green')
        else:
            ax1.annotate('✗', xy=(i + width, auto_t[i]), fontsize=14,
                        ha='center', va='bottom', color='red')
    ax1.set_xlabel('Matrix Shape Type')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Cost Model Strategy Selection')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 策略选择准确率
    accuracy = sum(optimal) / len(optimal) * 100
    strategies = [x.get("auto_strategy", "?") for x in d]
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    ax2.pie(strategy_counts.values(), labels=strategy_counts.keys(),
            autopct='%1.0f%%', colors=['#4472C4', '#ED7D31', '#70AD47'])
    ax2.set_title(f'Strategy Distribution (Accuracy: {accuracy:.0f}%)')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig10_cost_model.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图11: Stencil 应用
# ================================================================
def fig11_stencil():
    files = load_json("exp8_applications_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    stencil = data["data"].get("stencil", [])
    if not stencil:
        return

    grid_sizes = sorted(set(x["grid_size"] for x in stencil))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#4472C4', '#ED7D31', '#70AD47', '#C00000', '#7030A0', '#FFC000']
    for i, gs in enumerate(grid_sizes):
        items = sorted([x for x in stencil if x["grid_size"] == gs], key=lambda x: x["iterations"])
        iters = [x["iterations"] for x in items]
        times = [x["mean_ms"] for x in items]
        stds = [x["std_ms"] for x in items]
        throughput = [x.get("throughput_cells_per_sec", 0) for x in items]

        ax1.errorbar(iters, times, yerr=stds, fmt='o-', color=colors[i % len(colors)],
                     linewidth=2, markersize=7, capsize=3, label=f'Grid {gs}x{gs}')

        ax2.plot(iters, [t/1e9 for t in throughput], 's-', color=colors[i % len(colors)],
                 linewidth=2, markersize=7, label=f'Grid {gs}x{gs}')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Stencil 2D: Execution Time')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Throughput (GCells/s)')
    ax2.set_title('Stencil 2D: Throughput')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig11_stencil.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图12: Jacobi 应用
# ================================================================
def fig12_jacobi():
    files = load_json("exp8_applications_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    jacobi = data["data"].get("jacobi", [])
    if not jacobi:
        return

    grid_sizes = sorted(set(x["grid_size"] for x in jacobi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#4472C4', '#ED7D31', '#70AD47', '#C00000', '#7030A0']
    for i, gs in enumerate(grid_sizes):
        items = sorted([x for x in jacobi if x["grid_size"] == gs], key=lambda x: x["max_iterations"])
        iters = [x["max_iterations"] for x in items]
        times = [x["mean_ms"] for x in items]
        stds = [x["std_ms"] for x in items]

        ax1.errorbar(iters, times, yerr=stds, fmt='o-', color=colors[i % len(colors)],
                     linewidth=2, markersize=7, capsize=3, label=f'Grid {gs}x{gs}')

        ips = [x["iter_per_sec"] for x in items]
        ax2.plot(iters, ips, 's-', color=colors[i % len(colors)],
                 linewidth=2, markersize=7, label=f'Grid {gs}x{gs}')

    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Jacobi 2D: Execution Time')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Max Iterations')
    ax2.set_ylabel('Iterations/sec')
    ax2.set_title('Jacobi 2D: Iteration Throughput')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig12_jacobi.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图13: Conv2d 应用
# ================================================================
def fig13_conv():
    files = load_json("exp8_applications_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    conv = data["data"].get("conv2d", [])
    if not conv:
        return

    configs = [x["config"] for x in conv]
    single = [x["single_gpu_mean_ms"] for x in conv]
    single_err = [x["single_gpu_std_ms"] for x in conv]
    dist = [x["distributed_mean_ms"] for x in conv]
    dist_err = [x["distributed_std_ms"] for x in conv]
    speedups = [x["speedup"] for x in conv]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(configs))
    width = 0.35
    ax1.bar(x - width/2, single, width, yerr=single_err, capsize=3,
            label='Single GPU', color='#4472C4')
    ax1.bar(x + width/2, dist, width, yerr=dist_err, capsize=3,
            label='Distributed', color='#ED7D31')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Distributed Conv2d Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(range(len(configs)), speedups, color='#70AD47')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Conv2d Speedup')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig13_conv2d.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图14: Einsum 应用
# ================================================================
def fig14_einsum():
    files = load_json("exp8_applications_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    einsum = data["data"].get("einsum", [])
    if not einsum:
        return

    descs = [x["description"] for x in einsum]
    single = [x["single_gpu_mean_ms"] for x in einsum]
    dist = [x["distributed_mean_ms"] for x in einsum]
    speedups = [x["speedup"] for x in einsum]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(descs))
    width = 0.35
    ax1.bar(x - width/2, single, width, label='Single GPU', color='#4472C4')
    ax1.bar(x + width/2, dist, width, label='Distributed', color='#ED7D31')
    ax1.set_xlabel('Operation')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Distributed Einsum Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(descs, rotation=30, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.barh(range(len(descs)), speedups, color='#70AD47')
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Speedup')
    ax2.set_yticks(range(len(descs)))
    ax2.set_yticklabels(descs, fontsize=9)
    ax2.set_title('Einsum Speedup')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig14_einsum.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  Main
# ================================================================
def main():
    ensure_dir(FIGURES_DIR)
    print("生成增强版论文图表...")

    fig1_compute_performance()
    fig2_comm_overhead()
    fig3_strong_scaling()
    fig4_weak_scaling()
    fig5_mixed_precision()
    fig6_kahan()
    fig7_sparse()
    fig8_pencil_fft()
    fig9_pipeline()
    fig10_cost_model()
    fig11_stencil()
    fig12_jacobi()
    fig13_conv()
    fig14_einsum()

    print(f"\n全部图表已保存至: {FIGURES_DIR}")
    file_count = len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
    print(f"共 {file_count} 张图")


if __name__ == "__main__":
    main()
