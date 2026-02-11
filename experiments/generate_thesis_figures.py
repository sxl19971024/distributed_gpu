#!/usr/bin/env python
"""
生成硕士论文所需的全部图表
"""
import os, json, glob
import numpy as np

# 尝试使用非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "thesis")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "thesis", "figures")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_json(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    results = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            # 提取 n 数
            base = os.path.basename(f)
            results[base] = data
    return results

# ================================================================
#  图1: 计算性能对比 (不同矩阵规模)
# ================================================================
def fig1_compute_performance():
    files = load_json("exp1_compute_performance_n*.json")
    if not files:
        print("跳过: exp1 数据不存在"); return

    for fname, data in files.items():
        d = data["data"]
        sizes = [x["matrix_size"] for x in d]
        single = [x["single_gpu_ms"] for x in d]
        dist = [x["distributed_ms"] for x in d]
        speedup = [x["speedup"] for x in d]
        n_gpu = data["gpu_count"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: 时间对比
        x = np.arange(len(sizes))
        width = 0.35
        ax1.bar(x - width/2, single, width, label='Single GPU', color='#4472C4')
        ax1.bar(x + width/2, dist, width, label=f'Distributed ({n_gpu} GPUs)', color='#ED7D31')
        ax1.set_xlabel('Matrix Size (N x N)')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'Computation Time: Single GPU vs Distributed ({n_gpu} GPUs)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(s) for s in sizes], rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 右图: 加速比
        ax2.plot(sizes, speedup, 'o-', color='#70AD47', linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1x)')
        ax2.axhline(y=n_gpu, color='blue', linestyle='--', alpha=0.5, label=f'Ideal ({n_gpu}x)')
        ax2.set_xlabel('Matrix Size (N x N)')
        ax2.set_ylabel('Speedup')
        ax2.set_title(f'Speedup Ratio ({n_gpu} GPUs)')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"fig1_compute_perf_n{n_gpu}.png")
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f"  生成: {out}")


# ================================================================
#  图2: 通信开销分析
# ================================================================
def fig2_comm_overhead():
    files = load_json("exp2_comm_overhead_n*.json")
    if not files:
        print("跳过: exp2 数据不存在"); return

    for fname, data in files.items():
        d = data["data"]
        sizes = [x["matrix_size"] for x in d]
        scatter = [x["scatter_ms"] for x in d]
        compute = [x["compute_ms"] for x in d]
        gather = [x["gather_ms"] for x in d]
        comm_pct = [x["comm_ratio_pct"] for x in d]
        n_gpu = data["gpu_count"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: 堆叠柱状图
        x = np.arange(len(sizes))
        ax1.bar(x, scatter, label='Scatter (comm)', color='#4472C4')
        ax1.bar(x, compute, bottom=scatter, label='Compute', color='#70AD47')
        ax1.bar(x, gather, bottom=[s+c for s,c in zip(scatter, compute)],
                label='Gather (comm)', color='#ED7D31')
        ax1.set_xlabel('Matrix Size (N x N)')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'Time Breakdown ({n_gpu} GPUs)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(s) for s in sizes], rotation=45)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3)

        # 右图: 通信占比
        ax2.plot(sizes, comm_pct, 's-', color='#C00000', linewidth=2, markersize=8)
        ax2.set_xlabel('Matrix Size (N x N)')
        ax2.set_ylabel('Communication Ratio (%)')
        ax2.set_title(f'Communication Overhead Ratio ({n_gpu} GPUs)')
        ax2.set_ylim(0, 105)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"fig2_comm_overhead_n{n_gpu}.png")
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f"  生成: {out}")


# ================================================================
#  图3: 强可扩展性
# ================================================================
def fig3_strong_scaling():
    files = load_json("exp3_strong_scaling_n*.json")
    if not files:
        print("跳过: exp3 数据不存在"); return

    gpu_counts = []
    speedups = []
    efficiencies = []
    times = []

    for fname in sorted(files.keys()):
        data = files[fname]
        d = data["data"][0]
        gpu_counts.append(d["num_gpus"])
        speedups.append(d["speedup"])
        efficiencies.append(d["efficiency_pct"])
        times.append(d["distributed_ms"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 加速比
    ax1.plot(gpu_counts, speedups, 'o-', color='#4472C4', linewidth=2, markersize=10, label='Actual')
    ax1.plot(gpu_counts, gpu_counts, '--', color='red', alpha=0.5, label='Ideal (linear)')
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Strong Scaling: Speedup (Matrix 8000x8000)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(gpu_counts)

    # 右图: 并行效率
    ax2.bar(range(len(gpu_counts)), efficiencies, color='#70AD47')
    ax2.set_xlabel('Number of GPUs')
    ax2.set_ylabel('Parallel Efficiency (%)')
    ax2.set_title('Strong Scaling: Parallel Efficiency')
    ax2.set_xticks(range(len(gpu_counts)))
    ax2.set_xticklabels([str(g) for g in gpu_counts])
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig3_strong_scaling.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图4: 弱可扩展性
# ================================================================
def fig4_weak_scaling():
    files = load_json("exp4_weak_scaling_n*.json")
    if not files:
        print("跳过: exp4 数据不存在"); return

    gpu_counts = []
    efficiencies = []
    times = []

    for fname in sorted(files.keys()):
        data = files[fname]
        d = data["data"][0]
        gpu_counts.append(d["num_gpus"])
        efficiencies.append(d["weak_efficiency_pct"])
        times.append(d["distributed_ms"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 执行时间
    ax1.plot(gpu_counts, times, 's-', color='#4472C4', linewidth=2, markersize=10)
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Weak Scaling: Execution Time (2000 rows/GPU)')
    ax1.grid(alpha=0.3)
    ax1.set_xticks(gpu_counts)

    # 右图: 弱可扩展效率
    ax2.bar(range(len(gpu_counts)), efficiencies, color='#ED7D31')
    ax2.set_xlabel('Number of GPUs')
    ax2.set_ylabel('Weak Scaling Efficiency (%)')
    ax2.set_title('Weak Scaling: Efficiency')
    ax2.set_xticks(range(len(gpu_counts)))
    ax2.set_xticklabels([str(g) for g in gpu_counts])
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig4_weak_scaling.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图5: 创新算子 - 混合精度
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
    std_times = [x["standard_ms"] for x in mp_data]
    mp_times = [x["mixed_precision_ms"] for x in mp_data]
    speedups = [x["speedup"] for x in mp_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(sizes))
    width = 0.35
    ax1.bar(x - width/2, std_times, width, label='Standard (FP32)', color='#4472C4')
    ax1.bar(x + width/2, mp_times, width, label='Mixed Precision (FP16 comm)', color='#ED7D31')
    ax1.set_xlabel('Matrix Size (N x N)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Mixed Precision vs Standard MatMul')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(range(len(sizes)), speedups, color='#70AD47')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Matrix Size (N x N)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Mixed Precision Speedup')
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig5_mixed_precision.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图6: 创新算子 - Kahan 精度
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
    std_err = [x["standard_abs_error"] for x in kahan_data]
    kahan_err = [x["kahan_abs_error"] for x in kahan_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ns))
    width = 0.35
    ax.bar(x - width/2, std_err, width, label='Standard Sum', color='#C00000')
    ax.bar(x + width/2, [max(e, 1e-15) for e in kahan_err], width,
           label='Kahan Compensated Sum', color='#70AD47')
    ax.set_xlabel('Number of Elements')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Numerical Accuracy: Standard Sum vs Kahan Compensated Sum')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n:.0e}' for n in ns])
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig6_kahan_accuracy.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图7: 流水线优化
# ================================================================
def fig7_pipeline():
    files = load_json("exp6_pipeline_n*.json")
    if not files:
        print("跳过: exp6 数据不存在"); return

    data = list(files.values())[0]
    d = data["data"]

    # 按矩阵规模分组
    sizes = sorted(set(x["matrix_size"] for x in d))
    chunks_list = sorted(set(x["num_chunks"] for x in d))

    fig, ax = plt.subplots(figsize=(12, 6))

    # baseline (第一条)
    baselines = {}
    for item in d:
        baselines[item["matrix_size"]] = item["baseline_ms"]

    x = np.arange(len(sizes))
    width = 0.18
    ax.bar(x - 1.5*width, [baselines[s] for s in sizes], width,
           label='Baseline (no overlap)', color='#A5A5A5')

    colors = ['#4472C4', '#ED7D31', '#70AD47']
    for i, nc in enumerate(chunks_list):
        vals = []
        for s in sizes:
            match = [item for item in d if item["matrix_size"]==s and item["num_chunks"]==nc]
            vals.append(match[0]["pipeline_ms"] if match else 0)
        ax.bar(x + (i-0.5)*width, vals, width,
               label=f'Pipeline (chunks={nc})', color=colors[i % len(colors)])

    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Pipeline Optimization: Compute-Communication Overlap')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig7_pipeline.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图8: 代价模型策略对比
# ================================================================
def fig8_cost_model():
    files = load_json("exp7_cost_model_n*.json")
    if not files:
        print("跳过: exp7 数据不存在"); return

    data = list(files.values())[0]
    d = data["data"]

    labels = [x["description"] for x in d]
    auto_times = [x["auto_time_ms"] for x in d]
    row_times = [x["row_split_ms"] for x in d]
    col_times = [x["col_split_ms"] for x in d]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, row_times, width, label='Row Split', color='#4472C4')
    ax.bar(x, col_times, width, label='Column Split', color='#ED7D31')
    ax.bar(x + width, auto_times, width, label='Auto (Cost Model)', color='#70AD47')

    ax.set_xlabel('Matrix Shape Type')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Cost Model Strategy Selection')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig8_cost_model.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图9: 科学计算应用 - Stencil
# ================================================================
def fig9_stencil():
    files = load_json("exp8_applications_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    stencil = data["data"].get("stencil", [])
    if not stencil:
        return

    # 按 grid_size 分组, x轴为 iterations
    grid_sizes = sorted(set(x["grid_size"] for x in stencil))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#C00000']
    for i, gs in enumerate(grid_sizes):
        items = [x for x in stencil if x["grid_size"] == gs]
        iters = [x["iterations"] for x in items]
        times = [x["time_ms"] for x in items]
        ax.plot(iters, times, 'o-', color=colors[i % len(colors)],
                linewidth=2, markersize=8, label=f'Grid {gs}x{gs}')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Distributed Stencil 2D: Heat Diffusion')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig9_stencil.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图10: Conv2d 性能
# ================================================================
def fig10_conv():
    files = load_json("exp8_applications_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    conv = data["data"].get("conv2d", [])
    if not conv:
        return

    batches = [x["batch_size"] for x in conv]
    single = [x["single_gpu_ms"] for x in conv]
    dist = [x["distributed_ms"] for x in conv]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batches))
    width = 0.35
    ax.bar(x - width/2, single, width, label='Single GPU', color='#4472C4')
    ax.bar(x + width/2, dist, width, label='Distributed', color='#ED7D31')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Distributed Conv2d Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig10_conv2d.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


# ================================================================
#  图11: 稀疏感知
# ================================================================
def fig11_sparse():
    files = load_json("exp5_innovation_n*.json")
    if not files:
        return
    data = list(files.values())[0]
    sparse = data["data"].get("sparse_aware", [])
    if not sparse:
        return

    sparsity = [x["sparsity"] for x in sparse]
    std_times = [x["standard_ms"] for x in sparse]
    sp_times = [x["sparse_aware_ms"] for x in sparse]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sparsity, std_times, 'o-', color='#4472C4', linewidth=2, markersize=8, label='Standard Dense')
    ax.plot(sparsity, sp_times, 's-', color='#ED7D31', linewidth=2, markersize=8, label='Sparse-Aware')
    ax.set_xlabel('Sparsity Ratio')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Sparse-Aware MatMul: Impact of Sparsity (N=4000)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig11_sparse_aware.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  生成: {out}")


def main():
    ensure_dir(FIGURES_DIR)
    print("生成论文图表...")

    fig1_compute_performance()
    fig2_comm_overhead()
    fig3_strong_scaling()
    fig4_weak_scaling()
    fig5_mixed_precision()
    fig6_kahan()
    fig7_pipeline()
    fig8_cost_model()
    fig9_stencil()
    fig10_conv()
    fig11_sparse()

    print(f"\n全部图表已保存至: {FIGURES_DIR}")
    print(f"共 {len(os.listdir(FIGURES_DIR))} 张图")


if __name__ == "__main__":
    main()
