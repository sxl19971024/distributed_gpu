#!/usr/bin/env python
"""
增强版 — 生成硕士论文所需的全部图表 (基于增强版实验数据)
包含误差棒, 更多子图, 更丰富的对比, seaborn风格, 统一配色, 高DPI

用法:
  python experiments/generate_thesis_figures_enhanced.py --data-dir results/n4_20260211_143025
  python experiments/generate_thesis_figures_enhanced.py             # 自动使用最新一次运行
"""
from __future__ import annotations

import os
import sys
import json
import glob
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ── 尝试导入 seaborn (可选) ──────────────────────────────────
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ================================================================
#  全局样式配置
# ================================================================
def setup_style() -> None:
    """统一配置 matplotlib 全局样式

    说明：集群环境常缺少系统级中文字体。
    本脚本优先尝试加载项目内置的 NotoSansCJKsc 字体文件，
    以避免图中中文出现缺字/乱码。
    """
    # ── 尝试加载内置中文字体 ──────────────────────────────────
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        font_candidates = [
            # 兼容: 用户已有 thesis_mpi_tensor_parallel 项目
            os.path.join(project_root, '..', 'thesis_mpi_tensor_parallel', 'fonts', 'NotoSansCJKsc-Regular.otf'),
            # 兼容: 若未来在 distributed_gpu_framework/thesis/fonts 放置字体
            os.path.join(project_root, 'thesis', 'fonts', 'NotoSansCJKsc-Regular.otf'),
        ]
        for fp in font_candidates:
            fp = os.path.abspath(fp)
            if os.path.isfile(fp):
                fm.fontManager.addfont(fp)
                font_name = fm.FontProperties(fname=fp).get_name()
                # 将内置字体放在最前，确保中文可用
                font_list = [font_name, 'DejaVu Sans', 'Arial Unicode MS']
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = font_list
                break
    except Exception:
        pass

    # 若未成功加载内置字体，仍保持默认 sans-serif 列表
    font_list = plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        sns.set_palette("deep")

    plt.rcParams.update({
        # 字体
        'font.family': 'sans-serif',
        'font.sans-serif': font_list,
        'axes.unicode_minus': False,
        'font.size': 12,
        # DPI
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        # 线条和标记
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        # 网格
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        # 图例
        'legend.framealpha': 0.85,
        'legend.edgecolor': '0.8',
        'legend.fontsize': 10,
        # 边框
        'axes.spines.top': False,
        'axes.spines.right': False,
        # 布局
        'figure.constrained_layout.use': False,
    })


# ── 统一配色方案 ────────────────────────────────────────────────
# 主色调 (最多8种, 适合对比)
COLORS = {
    'blue':     '#2563EB',
    'orange':   '#EA580C',
    'green':    '#16A34A',
    'red':      '#DC2626',
    'purple':   '#9333EA',
    'gold':     '#CA8A04',
    'cyan':     '#0891B2',
    'pink':     '#DB2777',
}
PALETTE = list(COLORS.values())

# 语义色
C_SINGLE_GPU = COLORS['blue']
C_DISTRIBUTED = COLORS['orange']
C_SPEEDUP = COLORS['green']
C_BASELINE = '#9CA3AF'  # gray-400
C_IDEAL = '#6366F1'     # indigo-500

# ── 全局变量 ────────────────────────────────────────────────────
RESULTS_DIR: Optional[str] = None
FIGURES_DIR: Optional[str] = None


# ================================================================
#  工具函数
# ================================================================
def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def load_json(pattern: str) -> Dict[str, Any]:
    """加载匹配 pattern 的所有 JSON 文件"""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    results: Dict[str, Any] = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            base = os.path.basename(f)
            results[base] = data
    return results


def find_latest_run_dir() -> Optional[str]:
    """自动查找 results/ 下最新的运行目录 (按修改时间)"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(project_root, "results")
    if not os.path.isdir(results_root):
        return None
    candidates: List[Tuple[float, str, str]] = []
    for d in os.listdir(results_root):
        full = os.path.join(results_root, d)
        if os.path.isdir(full) and d.startswith("n") and "_" in d:
            jsons = glob.glob(os.path.join(full, "*.json"))
            if jsons:
                candidates.append((os.path.getmtime(full), full, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _save_fig(fig: plt.Figure, filename: str) -> None:
    """统一保存图表: 高DPI PNG + 可选 PDF"""
    out_png = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_png, bbox_inches='tight', facecolor='white')
    # 同时保存 PDF (论文排版用)
    out_pdf = out_png.replace('.png', '.pdf')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {filename}  (+PDF)")


def _add_value_labels(ax: plt.Axes, bars, fmt: str = '{:.1f}',
                      fontsize: int = 8, offset: float = 0.02) -> None:
    """在柱状图上方添加数值标注"""
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height + ymax * offset,
                    fmt.format(height), ha='center', va='bottom', fontsize=fontsize)


# ================================================================
#  图1: 计算性能对比 (含误差棒)
# ================================================================
def fig1_compute_performance() -> None:
    """图1：计算性能对比（兼容两种 exp1 数据结构）

    - 旧结构: data["data"] 为 list, 每项包含 matrix_size/single_gpu_mean_ms 等
    - 新结构(增强): data["data"] 为 dict, 含 matmul/mixed_precision_matmul/fft2d 等
      matmul 每项包含 size/single_ms/dist_ms/speedup
    """
    files = load_json("exp1_compute_performance*.json")
    if not files:
        print("  ⏭ 跳过: exp1 数据不存在"); return

    for fname, data in files.items():
        d = data.get("data")
        n_gpu = data.get("gpu_count", 1)

        # ── 解析数据 ──────────────────────────────────────────
        if isinstance(d, list):
            sizes = [x.get("matrix_size") for x in d]
            single = [x.get("single_gpu_mean_ms", 0.0) for x in d]
            dist = [x.get("distributed_mean_ms", 0.0) for x in d]
            single_err = [x.get("single_gpu_std_ms", 0.0) for x in d]
            dist_err = [x.get("distributed_std_ms", 0.0) for x in d]
            speedup = [x.get("speedup", 0.0) for x in d]
            efficiency = [x.get("efficiency_pct", 0.0) for x in d]
            gflops = [x.get("distributed_gflops", 0.0) for x in d]
            title_suffix = f"({n_gpu} GPUs)"
        elif isinstance(d, dict):
            mm = d.get("matmul", [])
            if not mm:
                print("  ⏭ 跳过: exp1(matmul) 数据不足");
                return
            sizes = [x.get("size") for x in mm]
            single = [x.get("single_ms", 0.0) for x in mm]
            dist = [x.get("dist_ms", 0.0) for x in mm]
            single_err = [0.0 for _ in mm]
            dist_err = [0.0 for _ in mm]
            speedup = [x.get("speedup", (s/t if t else 0.0)) for x, s, t in zip(mm, single, dist)]
            efficiency = [(sp / n_gpu * 100.0) if n_gpu else 0.0 for sp in speedup]
            # 估算分布式GFLOPS: 2*N^3 / time
            gflops = []
            for N, t_ms in zip(sizes, dist):
                if not N or not t_ms or t_ms <= 0:
                    gflops.append(0.0)
                else:
                    flops = 2.0 * (float(N) ** 3)
                    gflops.append(flops / (t_ms / 1000.0) / 1e9)
            title_suffix = f"(MatMul, {n_gpu} GPUs)"
        else:
            print("  ⏭ 跳过: exp1 数据结构未知");
            return

        # ── 绘图 ──────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (0,0) 时间对比
        ax = axes[0, 0]
        x = np.arange(len(sizes))
        w = 0.35
        ax.bar(x - w / 2, single, w, yerr=single_err, capsize=3,
               label='Single GPU', color=C_SINGLE_GPU, alpha=0.85, edgecolor='white')
        ax.bar(x + w / 2, dist, w, yerr=dist_err, capsize=3,
               label=f'Distributed ({n_gpu} GPUs)', color=C_DISTRIBUTED, alpha=0.85, edgecolor='white')
        ax.set_xlabel('Matrix Size (N×N)')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Computation Time', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
        ax.legend(loc='upper left')

        # (0,1) 加速比
        ax = axes[0, 1]
        ax.plot(sizes, speedup, 'o-', color=C_SPEEDUP, linewidth=2.5, markersize=8,
                label='Measured', zorder=5)
        ax.axhline(y=1.0, color=C_BASELINE, linestyle='--', alpha=0.6, label='Baseline (1×)')
        ax.axhline(y=n_gpu, color=C_IDEAL, linestyle='--', alpha=0.6, label=f'Ideal ({n_gpu}×)')
        ax.fill_between(sizes, 1.0, speedup, alpha=0.1, color=C_SPEEDUP)
        ax.set_xlabel('Matrix Size (N×N)')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup', fontweight='bold')
        ax.legend()

        # (1,0) 并行效率
        ax = axes[1, 0]
        colors_eff = [COLORS['green'] if e > 50 else COLORS['gold'] if e > 20 else COLORS['red']
                      for e in efficiency]
        bars = ax.bar(range(len(sizes)), efficiency, color=colors_eff, edgecolor='white')
        _add_value_labels(ax, bars, fmt='{:.1f}%')
        ax.set_xlabel('Matrix Size (N×N)')
        ax.set_ylabel('Parallel Efficiency (%)')
        ax.set_title('Parallel Efficiency', fontweight='bold')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)

        # (1,1) GFLOPS
        ax = axes[1, 1]
        ax.plot(sizes, gflops, 's-', color=COLORS['purple'], linewidth=2.5, markersize=8)
        ax.fill_between(sizes, 0, gflops, alpha=0.1, color=COLORS['purple'])
        ax.set_xlabel('Matrix Size (N×N)')
        ax.set_ylabel('GFLOPS')
        ax.set_title('Throughput (Estimated)', fontweight='bold')

        fig.suptitle(f'Experiment 1: Compute Performance {title_suffix}',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        _save_fig(fig, f"fig1_compute_perf_n{n_gpu}.png")



# ================================================================
#  图2: 通信开销分析 (含误差棒)
# ================================================================
def fig2_comm_overhead() -> None:
    files = load_json("exp2_comm_overhead*.json")
    if not files:
        print("  ⏭ 跳过: exp2 数据不存在"); return

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

        # (0) 堆叠柱状图
        ax = axes[0]
        x = np.arange(len(sizes))
        ax.bar(x, scatter, label='Scatter', color=COLORS['blue'], edgecolor='white')
        ax.bar(x, compute, bottom=scatter, label='Compute', color=COLORS['green'], edgecolor='white')
        ax.bar(x, gather, bottom=[s + c for s, c in zip(scatter, compute)],
               label='Gather', color=COLORS['orange'], edgecolor='white')
        ax.set_xlabel('Matrix Size (N×N)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Time Breakdown ({n_gpu} GPUs)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
        ax.legend()
        ax.set_yscale('log')

        # (1) 通信 vs 计算占比
        ax = axes[1]
        ax.stackplot(range(len(sizes)),
                     [comm_pct, comp_pct],
                     labels=['Communication', 'Computation'],
                     colors=[COLORS['red'], COLORS['green']], alpha=0.7)
        ax.set_xlabel('Matrix Size (N×N)')
        ax.set_ylabel('Ratio (%)')
        ax.set_title('Communication vs Computation Ratio', fontweight='bold')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
        ax.legend(loc='center right')
        ax.set_ylim(0, 105)

        # (2) 通信占比折线
        ax = axes[2]
        ax.plot(sizes, comm_pct, 's-', color=COLORS['red'], linewidth=2.5, markersize=8)
        ax.fill_between(sizes, 0, comm_pct, alpha=0.12, color=COLORS['red'])
        ax.set_xlabel('Matrix Size (N×N)')
        ax.set_ylabel('Communication Ratio (%)')
        ax.set_title('Communication Overhead Trend', fontweight='bold')
        ax.set_ylim(0, 105)

        fig.suptitle(f'Experiment 2: Communication Overhead ({n_gpu} GPUs)',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        _save_fig(fig, f"fig2_comm_overhead_n{n_gpu}.png")


# ================================================================
#  图3: 强可扩展性 (多矩阵规模)
# ================================================================
def fig3_strong_scaling() -> None:
    files = load_json("exp3_strong_scaling*.json")
    if not files:
        print("  ⏭ 跳过: exp3 数据不存在"); return

    all_data: Dict[int, list] = {}
    for fname in sorted(files.keys()):
        data = files[fname]
        gpu_count = data["gpu_count"]
        all_data[gpu_count] = data["data"]

    gpu_counts = sorted(all_data.keys())
    matrix_sizes = sorted(set(item["matrix_size"]
                               for items in all_data.values() for item in items))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (0) 加速比 vs GPU数
    ax = axes[0]
    for i, ms in enumerate(matrix_sizes):
        gcs, sps = [], []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["matrix_size"] == ms]
            if match:
                gcs.append(gc)
                sps.append(match[0]["speedup"])
        ax.plot(gcs, sps, 'o-', color=PALETTE[i % len(PALETTE)], linewidth=2,
                markersize=8, label=f'N={ms}')
    ax.plot(gpu_counts, gpu_counts, '--', color=C_BASELINE, alpha=0.6, label='Ideal')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Speedup')
    ax.set_title('Strong Scaling: Speedup', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(gpu_counts)

    # (1) 并行效率 vs GPU数
    ax = axes[1]
    for i, ms in enumerate(matrix_sizes):
        gcs, effs = [], []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["matrix_size"] == ms]
            if match:
                gcs.append(gc)
                effs.append(match[0]["efficiency_pct"])
        ax.plot(gcs, effs, 's-', color=PALETTE[i % len(PALETTE)], linewidth=2,
                markersize=8, label=f'N={ms}')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('Strong Scaling: Parallel Efficiency', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(gpu_counts)

    # (2) 执行时间 vs GPU数
    ax = axes[2]
    for i, ms in enumerate(matrix_sizes):
        gcs, times, stds = [], [], []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["matrix_size"] == ms]
            if match:
                gcs.append(gc)
                times.append(match[0]["distributed_mean_ms"])
                stds.append(match[0]["distributed_std_ms"])
        ax.errorbar(gcs, times, yerr=stds, fmt='o-', color=PALETTE[i % len(PALETTE)],
                    linewidth=2, markersize=8, capsize=3, label=f'N={ms}')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Strong Scaling: Execution Time', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(gpu_counts)

    fig.suptitle('Experiment 3: Strong Scaling', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig3_strong_scaling.png")


# ================================================================
#  图4: 弱可扩展性 (多per_gpu_rows)
# ================================================================
def fig4_weak_scaling() -> None:
    files = load_json("exp4_weak_scaling*.json")
    if not files:
        print("  ⏭ 跳过: exp4 数据不存在"); return

    all_data: Dict[int, list] = {}
    for fname in sorted(files.keys()):
        data = files[fname]
        gpu_count = data["gpu_count"]
        all_data[gpu_count] = data["data"]

    gpu_counts = sorted(all_data.keys())
    per_gpu_rows_list = sorted(set(item["per_gpu_rows"]
                                    for items in all_data.values() for item in items))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (0) 弱效率 vs GPU数
    ax = axes[0]
    for i, pgr in enumerate(per_gpu_rows_list):
        gcs, effs = [], []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["per_gpu_rows"] == pgr]
            if match:
                gcs.append(gc)
                effs.append(match[0]["weak_efficiency_pct"])
        ax.plot(gcs, effs, 'o-', color=PALETTE[i % len(PALETTE)], linewidth=2,
                markersize=8, label=f'{pgr} rows/GPU')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Weak Scaling Efficiency (%)')
    ax.set_title('Weak Scaling: Efficiency', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(gpu_counts)

    # (1) 执行时间 vs GPU数
    ax = axes[1]
    for i, pgr in enumerate(per_gpu_rows_list):
        gcs, times, stds = [], [], []
        for gc in gpu_counts:
            match = [item for item in all_data[gc] if item["per_gpu_rows"] == pgr]
            if match:
                gcs.append(gc)
                times.append(match[0]["distributed_mean_ms"])
                stds.append(match[0]["distributed_std_ms"])
        ax.errorbar(gcs, times, yerr=stds, fmt='s-', color=PALETTE[i % len(PALETTE)],
                    linewidth=2, markersize=8, capsize=3, label=f'{pgr} rows/GPU')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Weak Scaling: Execution Time', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticks(gpu_counts)

    # (2) 热力图
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
    ax.set_title('Weak Efficiency Heatmap (%)', fontweight='bold')
    for i in range(len(per_gpu_rows_list)):
        for j in range(len(gpu_counts)):
            val = eff_matrix[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='black' if val > 30 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Experiment 4: Weak Scaling', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig4_weak_scaling.png")


# ================================================================
#  图5: 混合精度 (含误差棒)
# ================================================================
def fig5_mixed_precision() -> None:
    files = load_json("exp5_innovation*.json")
    if not files:
        print("  ⏭ 跳过: exp5 数据不存在"); return

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
    w = 0.35
    ax.bar(x - w / 2, std_t, w, yerr=std_err, capsize=3,
           label='Standard (FP32)', color=C_SINGLE_GPU, edgecolor='white')
    ax.bar(x + w / 2, mp_t, w, yerr=mp_err, capsize=3,
           label='Mixed Precision', color=C_DISTRIBUTED, edgecolor='white')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Mixed Precision vs Standard MatMul', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.legend()

    # 加速比
    ax = axes[1]
    bars = ax.bar(range(len(sizes)), speedups, color=C_SPEEDUP, edgecolor='white')
    ax.axhline(y=1.0, color=COLORS['red'], linestyle='--', alpha=0.5)
    _add_value_labels(ax, bars, fmt='{:.2f}×')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Speedup')
    ax.set_title('Mixed Precision Speedup', fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)

    # 精度对比
    ax = axes[2]
    x = np.arange(len(sizes))
    ax.bar(x - w / 2, rel_err_std, w, label='Standard', color=C_SINGLE_GPU, edgecolor='white')
    ax.bar(x + w / 2, rel_err_mp, w, label='Mixed Precision', color=C_DISTRIBUTED, edgecolor='white')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Relative Error')
    ax.set_title('Numerical Accuracy Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.legend()
    ax.set_yscale('log')

    fig.suptitle('Experiment 5a: Mixed Precision MatMul', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig5_mixed_precision.png")


# ================================================================
#  图6: Kahan 补偿求和精度
# ================================================================
def fig6_kahan() -> None:
    files = load_json("exp5_innovation*.json")
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
    w = 0.35
    ax1.bar(x - w / 2, std_err, w, label='Standard Sum', color=COLORS['red'], edgecolor='white')
    ax1.bar(x + w / 2, kahan_err, w, label='Kahan Sum', color=COLORS['green'], edgecolor='white')
    ax1.set_xlabel('Number of Elements')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Numerical Accuracy: Standard vs Kahan', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{n:.0e}' for n in ns], rotation=45, fontsize=9)
    ax1.set_yscale('log')
    ax1.legend()

    # 改善倍数
    bars = ax2.bar(range(len(ns)), improvement, color=COLORS['purple'], edgecolor='white')
    _add_value_labels(ax2, bars, fmt='{:.0f}×')
    ax2.set_xlabel('Number of Elements')
    ax2.set_ylabel('Improvement Factor (×)')
    ax2.set_title('Kahan Sum Improvement over Standard', fontweight='bold')
    ax2.set_xticks(range(len(ns)))
    ax2.set_xticklabels([f'{n:.0e}' for n in ns], rotation=45, fontsize=9)
    ax2.set_yscale('log')

    fig.suptitle('Experiment 5b: Kahan Compensated Summation', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig6_kahan_accuracy.png")


# ================================================================
#  图7: 稀疏感知 (多规模)
# ================================================================
def fig7_sparse() -> None:
    files = load_json("exp5_innovation*.json")
    if not files:
        return
    data = list(files.values())[0]
    sparse = data["data"].get("sparse_aware", [])
    if not sparse:
        return

    Ns = sorted(set(x["N"] for x in sparse))

    fig, axes = plt.subplots(1, max(len(Ns), 1), figsize=(7 * max(len(Ns), 1), 6))
    if len(Ns) == 1:
        axes = [axes]

    for idx, N in enumerate(Ns):
        ax = axes[idx]
        items = [x for x in sparse if x["N"] == N]
        sparsity = [x["sparsity"] for x in items]
        std_t = [x["standard_mean_ms"] for x in items]
        std_err = [x["standard_std_ms"] for x in items]
        sp_t = [x["sparse_aware_mean_ms"] for x in items]
        sp_err = [x["sparse_aware_std_ms"] for x in items]

        ax.errorbar(sparsity, std_t, yerr=std_err, fmt='o-', color=C_SINGLE_GPU,
                    linewidth=2, markersize=7, capsize=3, label='Standard Dense')
        ax.errorbar(sparsity, sp_t, yerr=sp_err, fmt='s-', color=C_DISTRIBUTED,
                    linewidth=2, markersize=7, capsize=3, label='Sparse-Aware')
        ax.fill_between(sparsity, std_t, sp_t, alpha=0.1, color=C_SPEEDUP)
        ax.set_xlabel('Sparsity Ratio')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Sparse-Aware MatMul (N={N})', fontweight='bold')
        ax.legend()

    fig.suptitle('Experiment 5c: Sparse-Aware Computation', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig7_sparse_aware.png")


# ================================================================
#  图8: Pencil FFT
# ================================================================
def fig8_pencil_fft() -> None:
    files = load_json("exp5_innovation*.json")
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
    w = 0.35
    ax1.bar(x - w / 2, batch_t, w, yerr=batch_err, capsize=3,
            label='Batch FFT', color=C_SINGLE_GPU, edgecolor='white')
    ax1.bar(x + w / 2, pencil_t, w, yerr=pencil_err, capsize=3,
            label='Pencil FFT', color=C_DISTRIBUTED, edgecolor='white')
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Batch FFT vs Pencil FFT', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(g) for g in grids])
    ax1.legend()

    # 折线对比
    ax2.plot(grids, batch_t, 'o-', color=C_SINGLE_GPU, linewidth=2.5, markersize=8, label='Batch FFT')
    ax2.plot(grids, pencil_t, 's-', color=C_DISTRIBUTED, linewidth=2.5, markersize=8, label='Pencil FFT')
    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('FFT Performance Trend', fontweight='bold')
    ax2.legend()
    ax2.set_xscale('log', base=2)

    fig.suptitle('Experiment 5d: Pencil FFT Decomposition', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig8_pencil_fft.png")


# ================================================================
#  图9: 流水线优化 (含误差棒, 热力图)
# ================================================================
def fig9_pipeline() -> None:
    files = load_json("exp6_pipeline*.json")
    if not files:
        print("  ⏭ 跳过: exp6 数据不存在"); return

    data = list(files.values())[0]
    d = data["data"]

    sizes = sorted(set(x["matrix_size"] for x in d))
    chunks_list = sorted(set(x["num_chunks"] for x in d))

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # (0) 柱状图对比
    ax = axes[0]
    baselines: Dict[int, float] = {}
    for item in d:
        baselines[item["matrix_size"]] = item["baseline_mean_ms"]

    x = np.arange(len(sizes))
    w = 0.18
    ax.bar(x - 2 * w, [baselines[s] for s in sizes], w,
           label='Baseline', color=C_BASELINE, edgecolor='white')
    for i, nc in enumerate(chunks_list):
        vals, errs = [], []
        for s in sizes:
            match = [item for item in d if item["matrix_size"] == s and item["num_chunks"] == nc]
            vals.append(match[0]["pipeline_mean_ms"] if match else 0)
            errs.append(match[0]["pipeline_std_ms"] if match else 0)
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2,
               label=f'Pipeline (chunks={nc})', color=PALETTE[i % len(PALETTE)], edgecolor='white')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Pipeline Optimization', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.legend(fontsize=8)

    # (1) 加速比折线
    ax = axes[1]
    for i, nc in enumerate(chunks_list):
        ss, sps = [], []
        for s in sizes:
            match = [item for item in d if item["matrix_size"] == s and item["num_chunks"] == nc]
            if match:
                ss.append(s)
                sps.append(match[0]["speedup"])
        ax.plot(ss, sps, 'o-', color=PALETTE[i % len(PALETTE)], linewidth=2,
                markersize=7, label=f'chunks={nc}')
    ax.axhline(y=1.0, color=COLORS['red'], linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Speedup')
    ax.set_title('Pipeline Speedup', fontweight='bold')
    ax.legend(fontsize=9)

    # (2) 加速比热力图
    ax = axes[2]
    speedup_matrix = np.zeros((len(chunks_list), len(sizes)))
    for i, nc in enumerate(chunks_list):
        for j, s in enumerate(sizes):
            match = [item for item in d if item["matrix_size"] == s and item["num_chunks"] == nc]
            if match:
                speedup_matrix[i, j] = match[0]["speedup"]
    im = ax.imshow(speedup_matrix, cmap='YlGn', aspect='auto', vmin=0.5, vmax=2.0)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
    ax.set_yticks(range(len(chunks_list)))
    ax.set_yticklabels([str(c) for c in chunks_list])
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Num Chunks')
    ax.set_title('Pipeline Speedup Heatmap', fontweight='bold')
    for i in range(len(chunks_list)):
        for j in range(len(sizes)):
            ax.text(j, i, f'{speedup_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Experiment 6: Pipeline Optimization', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig9_pipeline.png")


# ================================================================
#  图10: 代价模型策略选择 (含最优标记)
# ================================================================
def fig10_cost_model() -> None:
    files = load_json("exp7_cost_model*.json")
    if not files:
        print("  ⏭ 跳过: exp7 数据不存在"); return

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
    w = 0.25
    ax1.bar(x - w, row_t, w, yerr=row_err, capsize=3,
            label='Row Split', color=COLORS['blue'], edgecolor='white')
    ax1.bar(x, col_t, w, yerr=col_err, capsize=3,
            label='Column Split', color=COLORS['orange'], edgecolor='white')
    ax1.bar(x + w, auto_t, w, yerr=auto_err, capsize=3,
            label='Auto (Cost Model)', color=COLORS['green'], edgecolor='white')
    for i, opt in enumerate(optimal):
        marker = '✓' if opt else '✗'
        color = COLORS['green'] if opt else COLORS['red']
        ax1.annotate(marker, xy=(i + w, auto_t[i]), fontsize=14,
                     ha='center', va='bottom', color=color, fontweight='bold')
    ax1.set_xlabel('Matrix Shape Type')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Cost Model Strategy Selection', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.legend()

    # 策略选择准确率 (饼图)
    accuracy = sum(optimal) / max(len(optimal), 1) * 100
    strategies = [x.get("auto_strategy", "?") for x in d]
    strategy_counts: Dict[str, int] = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    wedges, texts, autotexts = ax2.pie(
        strategy_counts.values(), labels=strategy_counts.keys(),
        autopct='%1.0f%%', colors=PALETTE[:len(strategy_counts)],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    for t in autotexts:
        t.set_fontweight('bold')
    ax2.set_title(f'Strategy Distribution (Accuracy: {accuracy:.0f}%)', fontweight='bold')

    fig.suptitle('Experiment 7: Cost Model Strategy', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig10_cost_model.png")


# ================================================================
#  图11: Stencil 应用
# ================================================================
def fig11_stencil() -> None:
    files = load_json("exp8_applications*.json")
    if not files:
        return
    data = list(files.values())[0]
    stencil = data["data"].get("stencil", [])
    if not stencil:
        return

    grid_sizes = sorted(set(x["grid_size"] for x in stencil))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, gs in enumerate(grid_sizes):
        items = sorted([x for x in stencil if x["grid_size"] == gs],
                       key=lambda x: x["iterations"])
        iters = [x["iterations"] for x in items]
        times = [x["mean_ms"] for x in items]
        stds = [x["std_ms"] for x in items]
        throughput = [x.get("throughput_cells_per_sec", 0) for x in items]

        c = PALETTE[i % len(PALETTE)]
        ax1.errorbar(iters, times, yerr=stds, fmt='o-', color=c,
                     linewidth=2, markersize=7, capsize=3, label=f'Grid {gs}×{gs}')
        ax2.plot(iters, [t / 1e9 for t in throughput], 's-', color=c,
                 linewidth=2, markersize=7, label=f'Grid {gs}×{gs}')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Stencil 2D: Execution Time', fontweight='bold')
    ax1.legend(fontsize=9)

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Throughput (GCells/s)')
    ax2.set_title('Stencil 2D: Throughput', fontweight='bold')
    ax2.legend(fontsize=9)

    fig.suptitle('Experiment 8a: Stencil 2D', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig11_stencil.png")


# ================================================================
#  图12: Jacobi 应用
# ================================================================
def fig12_jacobi() -> None:
    files = load_json("exp8_applications*.json")
    if not files:
        return
    data = list(files.values())[0]
    jacobi = data["data"].get("jacobi", [])
    if not jacobi:
        return

    grid_sizes = sorted(set(x["grid_size"] for x in jacobi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, gs in enumerate(grid_sizes):
        items = sorted([x for x in jacobi if x["grid_size"] == gs],
                       key=lambda x: x["max_iterations"])
        iters = [x["max_iterations"] for x in items]
        times = [x["mean_ms"] for x in items]
        stds = [x["std_ms"] for x in items]
        ips = [x["iter_per_sec"] for x in items]

        c = PALETTE[i % len(PALETTE)]
        ax1.errorbar(iters, times, yerr=stds, fmt='o-', color=c,
                     linewidth=2, markersize=7, capsize=3, label=f'Grid {gs}×{gs}')
        ax2.plot(iters, ips, 's-', color=c, linewidth=2, markersize=7, label=f'Grid {gs}×{gs}')

    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Jacobi 2D: Execution Time', fontweight='bold')
    ax1.legend(fontsize=9)

    ax2.set_xlabel('Max Iterations')
    ax2.set_ylabel('Iterations/sec')
    ax2.set_title('Jacobi 2D: Iteration Throughput', fontweight='bold')
    ax2.legend(fontsize=9)

    fig.suptitle('Experiment 8b: Jacobi 2D Solver', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig12_jacobi.png")


# ================================================================
#  图13: Conv2d 应用
# ================================================================
def fig13_conv() -> None:
    files = load_json("exp8_applications*.json")
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
    w = 0.35
    ax1.bar(x - w / 2, single, w, yerr=single_err, capsize=3,
            label='Single GPU', color=C_SINGLE_GPU, edgecolor='white')
    ax1.bar(x + w / 2, dist, w, yerr=dist_err, capsize=3,
            label='Distributed', color=C_DISTRIBUTED, edgecolor='white')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Distributed Conv2d Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
    ax1.legend()

    bars = ax2.bar(range(len(configs)), speedups, color=C_SPEEDUP, edgecolor='white')
    ax2.axhline(y=1.0, color=COLORS['red'], linestyle='--', alpha=0.5)
    _add_value_labels(ax2, bars, fmt='{:.2f}×')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Conv2d Speedup', fontweight='bold')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)

    fig.suptitle('Experiment 8c: Distributed Conv2d', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig13_conv2d.png")


# ================================================================
#  图14: Einsum 应用
# ================================================================
def fig14_einsum() -> None:
    files = load_json("exp8_applications*.json")
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
    w = 0.35
    ax1.bar(x - w / 2, single, w, label='Single GPU', color=C_SINGLE_GPU, edgecolor='white')
    ax1.bar(x + w / 2, dist, w, label='Distributed', color=C_DISTRIBUTED, edgecolor='white')
    ax1.set_xlabel('Operation')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Distributed Einsum Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(descs, rotation=30, ha='right', fontsize=9)
    ax1.legend()

    bars = ax2.barh(range(len(descs)), speedups, color=C_SPEEDUP, edgecolor='white')
    ax2.axvline(x=1.0, color=COLORS['red'], linestyle='--', alpha=0.5)
    ax2.set_xlabel('Speedup')
    ax2.set_yticks(range(len(descs)))
    ax2.set_yticklabels(descs, fontsize=9)
    ax2.set_title('Einsum Speedup', fontweight='bold')

    fig.suptitle('Experiment 8d: Distributed Einsum', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig14_einsum.png")


# ================================================================
#  图15: 内存效率分析 (新增 — 实验9)
# ================================================================
def fig15_memory_efficiency() -> None:
    files = load_json("exp9_memory_efficiency*.json")
    if not files:
        print("  ⏭ 跳过: exp9 数据不存在"); return

    data = list(files.values())[0]
    d = data.get("data", [])
    if not d:
        return

    # exp9 数据结构: 每项包含 matrix_size, single_gpu_peak_gb, distributed_peak_gb,
    # memory_saving_pct, single_utilization_pct, dist_utilization_pct, total_gpu_mem_gb 等
    sizes = [x["matrix_size"] for x in d]
    single_peak = [x["single_gpu_peak_gb"] for x in d]
    dist_peak = [x["distributed_peak_gb"] for x in d]
    saving_pct = [x["memory_saving_pct"] for x in d]
    theory_full = [x.get("theory_full_gb", 0) for x in d]
    theory_dist = [x.get("theory_distributed_gb", 0) for x in d]
    n_gpu = data.get("gpu_count", "?")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) 峰值显存对比: 单GPU vs 分布式
    ax = axes[0, 0]
    x = np.arange(len(sizes))
    w = 0.35
    b1 = ax.bar(x - w / 2, single_peak, w, label='Single GPU Peak',
                color=C_SINGLE_GPU, alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w / 2, dist_peak, w, label=f'Distributed Peak ({n_gpu} GPUs)',
                color=C_DISTRIBUTED, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Peak Memory (GB)')
    ax.set_title('Peak GPU Memory: Single vs Distributed', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
    ax.legend()

    # (0,1) 显存节省百分比
    ax = axes[0, 1]
    colors_save = [COLORS['green'] if s > 30 else COLORS['gold'] if s > 0 else COLORS['red']
                   for s in saving_pct]
    bars = ax.bar(range(len(sizes)), saving_pct, color=colors_save, edgecolor='white')
    _add_value_labels(ax, bars, fmt='{:.1f}%')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Memory Saving (%)')
    ax.set_title('Memory Saving with Distribution', fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # (1,0) 理论 vs 实际显存 (分布式)
    ax = axes[1, 0]
    ax.plot(sizes, theory_full, 's--', color=COLORS['red'], linewidth=1.5,
            label='Theory (Full, Single GPU)', alpha=0.7)
    ax.plot(sizes, theory_dist, 'o--', color=COLORS['cyan'], linewidth=1.5,
            label='Theory (Distributed)', alpha=0.7)
    ax.plot(sizes, single_peak, 's-', color=C_SINGLE_GPU, linewidth=2.5,
            label='Actual (Single GPU)')
    ax.plot(sizes, dist_peak, 'o-', color=C_DISTRIBUTED, linewidth=2.5,
            label='Actual (Distributed)')
    ax.fill_between(sizes, dist_peak, single_peak, alpha=0.1, color=COLORS['green'])
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Theory vs Actual Memory Usage', fontweight='bold')
    ax.legend(fontsize=9)

    # (1,1) 显存利用率 (占总显存百分比)
    ax = axes[1, 1]
    single_util = [x.get("single_utilization_pct", 0) for x in d]
    dist_util = [x.get("dist_utilization_pct", 0) for x in d]
    ax.plot(sizes, single_util, 's-', color=C_SINGLE_GPU, linewidth=2.5,
            label='Single GPU')
    ax.plot(sizes, dist_util, 'o-', color=C_DISTRIBUTED, linewidth=2.5,
            label=f'Distributed ({n_gpu} GPUs)')
    ax.fill_between(sizes, 0, dist_util, alpha=0.1, color=C_DISTRIBUTED)
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('GPU Memory Utilization (%)')
    ax.set_title('GPU Memory Utilization', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 105)

    fig.suptitle(f'Experiment 9: GPU Memory Efficiency ({n_gpu} GPUs)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig15_memory_efficiency.png")


# ================================================================
#  图16: 多算子综合对比 — 雷达图 + 柱状图 (新增 — 实验10)
# ================================================================
def fig16_multi_operator_radar() -> None:
    files = load_json("exp10_operator_comparison*.json")
    if not files:
        print("  ⏭ 跳过: exp10 数据不存在"); return

    data = list(files.values())[0]
    d = data.get("data", {})
    # exp10 数据结构: data = {"operators": [{"name": ..., "speedup": ...}, ...],
    #                         "matrix_size": ..., "gpu_count": ...}
    operators = d.get("operators", [])
    if not operators or len(operators) < 2:
        print("  ⏭ 跳过: exp10 算子数据不足"); return

    op_names = [item["name"] for item in operators]
    speedups = [item["speedup"] for item in operators]
    n_gpu = d.get("gpu_count", data.get("gpu_count", "?"))
    matrix_size = d.get("matrix_size", "?")

    # 归一化加速比到 [0, 1] (用于雷达图)
    max_speedup = max(speedups) if speedups else 1.0
    norm_speedups = [s / max_speedup for s in speedups] if max_speedup > 0 else speedups

    fig = plt.figure(figsize=(14, 7))

    # ── 左: 雷达图 (归一化加速比) ──
    ax_radar = fig.add_subplot(121, polar=True)
    N_ops = len(op_names)
    angles = [n / float(N_ops) * 2 * math.pi for n in range(N_ops)]
    angles += angles[:1]  # 闭合

    values = norm_speedups + norm_speedups[:1]
    c = COLORS['blue']
    ax_radar.plot(angles, values, 'o-', linewidth=2.5, color=c, label='Normalized Speedup')
    ax_radar.fill(angles, values, alpha=0.15, color=c)

    # 标注每个算子的实际加速比
    for i, (angle, val, sp) in enumerate(zip(angles[:-1], norm_speedups, speedups)):
        ax_radar.text(angle, val + 0.08, f'{sp:.2f}×',
                      ha='center', va='bottom', fontsize=8, fontweight='bold',
                      color=PALETTE[i % len(PALETTE)])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(op_names, fontsize=10, fontweight='bold')
    ax_radar.set_ylim(0, 1.25)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='gray')
    ax_radar.set_title(f'Normalized Speedup Radar\n(N={matrix_size}, {n_gpu} GPUs)',
                       fontweight='bold', pad=20)

    # ── 右: 水平柱状图 (原始加速比) ──
    ax_bar = fig.add_subplot(122)
    y = np.arange(len(op_names))
    bars = ax_bar.barh(y, speedups, 0.6,
                       color=PALETTE[:len(op_names)], edgecolor='white')
    ax_bar.axvline(x=1.0, color=COLORS['red'], linestyle='--', alpha=0.5, label='Baseline (1×)')
    ax_bar.set_xlabel('Speedup')
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(op_names, fontsize=10)
    ax_bar.set_title(f'Speedup per Operator ({n_gpu} GPUs)', fontweight='bold')
    ax_bar.legend(fontsize=9)
    for bar, val in zip(bars, speedups):
        ax_bar.text(bar.get_width() + max(speedups) * 0.02,
                    bar.get_y() + bar.get_height() / 2.,
                    f'{val:.3f}×', va='center', fontsize=9, fontweight='bold')

    fig.suptitle(f'Experiment 10: Multi-Operator Comparison (N={matrix_size})',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig16_multi_operator_radar.png")


# ================================================================
#  图17: 全实验性能概览仪表盘 (新增)
# ================================================================
def fig17_summary_dashboard() -> None:
    """汇总所有实验的关键指标, 生成一张概览仪表盘"""
    summary_data: List[Dict[str, Any]] = []

    # 从各实验中提取关键指标
    # Exp1: 最大加速比
    exp1 = load_json("exp1_compute_performance*.json")
    if exp1:
        data = list(exp1.values())[0]
        d1 = data.get("data")
        if isinstance(d1, list):
            max_speedup = max((x.get("speedup", 0) for x in d1), default=0)
            max_gflops = max((x.get("distributed_gflops", 0) for x in d1), default=0)
        elif isinstance(d1, dict):
            mm = d1.get("matmul", [])
            max_speedup = max((x.get("speedup", 0) for x in mm), default=0)
            # 估算峰值GFLOPS: 2*N^3 / time
            max_gflops = 0.0
            for x in mm:
                N = x.get("size")
                t_ms = x.get("dist_ms", 0)
                if N and t_ms and t_ms > 0:
                    flops = 2.0 * (float(N) ** 3)
                    g = flops / (t_ms / 1000.0) / 1e9
                    if g > max_gflops:
                        max_gflops = g
        else:
            max_speedup = 0
            max_gflops = 0

        summary_data.append({
            "metric": "Max Speedup\n(MatMul)", "value": max_speedup,
            "unit": "×", "color": COLORS['blue']
        })
        summary_data.append({
            "metric": "Peak GFLOPS\n(MatMul)", "value": max_gflops,
            "unit": "GFLOPS", "color": COLORS['purple']
        })

    # Exp5: 混合精度加速
    exp5 = load_json("exp5_innovation*.json")
    if exp5:
        data = list(exp5.values())[0]
        mp = data["data"].get("mixed_precision", [])
        if mp:
            max_mp_speedup = max((x.get("speedup", 0) for x in mp), default=0)
            summary_data.append({
                "metric": "Mixed Precision\nSpeedup", "value": max_mp_speedup,
                "unit": "×", "color": COLORS['orange']
            })
        kahan = data["data"].get("kahan_sum", [])
        if kahan:
            max_improvement = max((x.get("improvement_factor", 0) for x in kahan), default=0)
            summary_data.append({
                "metric": "Kahan Accuracy\nImprovement", "value": max_improvement,
                "unit": "×", "color": COLORS['green']
            })

    # Exp6: 流水线加速
    exp6 = load_json("exp6_pipeline*.json")
    if exp6:
        data = list(exp6.values())[0]
        max_pipe_speedup = max((x.get("speedup", 0) for x in data["data"]), default=0)
        summary_data.append({
            "metric": "Pipeline\nSpeedup", "value": max_pipe_speedup,
            "unit": "×", "color": COLORS['cyan']
        })

    # Exp7: 代价模型准确率
    exp7 = load_json("exp7_cost_model*.json")
    if exp7:
        data = list(exp7.values())[0]
        optimal_list = [x.get("auto_is_optimal", False) for x in data["data"]]
        accuracy = sum(optimal_list) / max(len(optimal_list), 1) * 100
        summary_data.append({
            "metric": "Cost Model\nAccuracy", "value": accuracy,
            "unit": "%", "color": COLORS['gold']
        })

    # Exp9: 最大显存节省
    exp9 = load_json("exp9_memory_efficiency*.json")
    if exp9:
        data = list(exp9.values())[0]
        d9 = data.get("data", [])
        if d9:
            max_saving = max((x.get("memory_saving_pct", 0) for x in d9), default=0)
            summary_data.append({
                "metric": "Max Memory\nSaving", "value": max_saving,
                "unit": "%", "color": COLORS['red']
            })

    if not summary_data:
        print("  ⏭ 跳过: 仪表盘 (无数据)")
        return

    # 布局: 一行 N 个指标卡片
    n = len(summary_data)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, summary_data):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 背景卡片
        bg = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                            boxstyle="round,pad=0.05",
                            facecolor=item["color"], alpha=0.12,
                            edgecolor=item["color"], linewidth=2)
        ax.add_patch(bg)

        # 数值
        val = item["value"]
        if val >= 1000:
            val_str = f'{val:.0f}'
        elif val >= 10:
            val_str = f'{val:.1f}'
        else:
            val_str = f'{val:.2f}'
        ax.text(0.5, 0.55, val_str + item["unit"],
                ha='center', va='center', fontsize=26, fontweight='bold',
                color=item["color"])
        # 标签
        ax.text(0.5, 0.2, item["metric"],
                ha='center', va='center', fontsize=10, color='#374151',
                fontweight='bold')

    fig.suptitle('Performance Overview Dashboard',
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    _save_fig(fig, "fig17_summary_dashboard.png")


# ================================================================
#  Main
# ================================================================
# ================================================================
#  图18~21: 跨框架对比图表
#  数据来源: experiments/benchmark_comparison.py 生成的 comparison_*.json
# ================================================================

def _load_comparison_json(name: str):
    """加载跨框架对比 JSON (搜索 results/comparison_*/ 目录)"""
    # 先在当前 RESULTS_DIR 中找
    direct = os.path.join(RESULTS_DIR, name)
    if os.path.exists(direct):
        with open(direct) as f:
            return json.load(f)
    # 再搜索所有 comparison_ 目录 (取最新)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(project_root, "results")
    candidates = sorted(glob.glob(os.path.join(results_root, "comparison_*", name)),
                        key=os.path.getmtime, reverse=True)
    if candidates:
        with open(candidates[0]) as f:
            return json.load(f)
    return None


def fig18_comparison_matmul() -> None:
    """图18: 矩阵乘法跨框架对比 (本框架 vs NCCL vs 单GPU)"""
    data = _load_comparison_json("comparison_matmul.json")
    if not data or not data.get("comparison"):
        print("  ⏭ 跳过: comparison_matmul.json 不存在"); return

    comp = data["comparison"]
    n_gpu = data.get("gpu_count", 3)
    sizes = [c["size"] for c in comp]
    single = [c["single_ms"] for c in comp]
    nccl = [c["nccl_ms"] for c in comp]
    ours = [c["ours_ms"] for c in comp]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (0) 耗时对比 (对数坐标)
    ax = axes[0]
    x = np.arange(len(sizes))
    w = 0.25
    ax.bar(x - w, single, w, label='Single GPU', color=C_SINGLE_GPU, alpha=0.85)
    ax.bar(x, nccl, w, label=f'NCCL ({n_gpu} GPUs)', color=COLORS['green'], alpha=0.85)
    ax.bar(x + w, ours, w, label=f'Ours ({n_gpu} GPUs)', color=C_DISTRIBUTED, alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Time (ms, log scale)')
    ax.set_title('Computation Time Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
    ax.legend()

    # (1) 加速比
    ax = axes[1]
    sp_nccl = [c.get("nccl_speedup", 0) for c in comp]
    sp_ours = [c.get("ours_speedup", 0) for c in comp]
    ax.plot(sizes, sp_nccl, 'o-', color=COLORS['green'], linewidth=2.5, markersize=8, label='NCCL')
    ax.plot(sizes, sp_ours, 's-', color=C_DISTRIBUTED, linewidth=2.5, markersize=8, label='Ours (mpi4py)')
    ax.axhline(y=1.0, color=C_BASELINE, linestyle='--', alpha=0.6, label='Baseline (1×)')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Speedup vs Single GPU')
    ax.set_title('Speedup Comparison', fontweight='bold')
    ax.legend()
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9) if len(sizes) <= 10 else None

    # (2) 本框架/NCCL 比值
    ax = axes[2]
    ratio = [c.get("ours_div_nccl", 0) for c in comp]
    colors_bar = [COLORS['green'] if r < 1 else COLORS['red'] for r in ratio]
    bars = ax.bar(range(len(sizes)), ratio, color=colors_bar, edgecolor='white', alpha=0.85)
    ax.axhline(y=1.0, color=C_BASELINE, linestyle='--', alpha=0.8, label='Equal (1×)')
    _add_value_labels(ax, bars, fmt='{:.2f}×')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Ours / NCCL (lower is better)')
    ax.set_title('Ours vs NCCL Ratio', fontweight='bold')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=9)
    ax.legend()

    fig.suptitle(f'Cross-Framework: MatMul ({n_gpu} GPUs)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig18_comparison_matmul.png")


def fig19_comparison_allreduce() -> None:
    """图19: AllReduce 跨框架对比 (本框架 vs NCCL)"""
    data = _load_comparison_json("comparison_allreduce.json")
    if not data or not data.get("comparison"):
        print("  ⏭ 跳过: comparison_allreduce.json 不存在"); return

    comp = data["comparison"]
    n_gpu = data.get("gpu_count", 3)
    sizes_mb = [c["size_mb"] for c in comp]
    nccl_ms = [c["nccl_ms"] for c in comp]
    ours_ms = [c["ours_ms"] for c in comp]
    nccl_bw = [c.get("nccl_bw_gbps", 0) for c in comp]
    ours_bw = [c.get("ours_bw_gbps", 0) for c in comp]
    ratio = [c.get("ours_div_nccl", 0) for c in comp]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (0) 时间对比 (对数)
    ax = axes[0]
    ax.plot(sizes_mb, nccl_ms, 'o-', color=COLORS['green'], linewidth=2.5, markersize=8, label='NCCL')
    ax.plot(sizes_mb, ours_ms, 's-', color=C_DISTRIBUTED, linewidth=2.5, markersize=8, label='Ours (mpi4py)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Data Size (MB)')
    ax.set_ylabel('Time (ms, log scale)')
    ax.set_title('AllReduce Latency', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1) 有效带宽
    ax = axes[1]
    ax.plot(sizes_mb, nccl_bw, 'o-', color=COLORS['green'], linewidth=2.5, markersize=8, label='NCCL')
    ax.plot(sizes_mb, ours_bw, 's-', color=C_DISTRIBUTED, linewidth=2.5, markersize=8, label='Ours (mpi4py)')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Data Size (MB)')
    ax.set_ylabel('Effective Bandwidth (GB/s)')
    ax.set_title('Effective Bandwidth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) 倍率差异
    ax = axes[2]
    ax.bar(range(len(sizes_mb)), ratio, color=COLORS['red'], edgecolor='white', alpha=0.85)
    ax.set_xlabel('Data Size (MB)')
    ax.set_ylabel('Ours / NCCL (lower is better)')
    ax.set_title('Slowdown Factor', fontweight='bold')
    ax.set_xticks(range(len(sizes_mb)))
    ax.set_xticklabels([f'{s}MB' for s in sizes_mb], rotation=45, fontsize=9)
    for i, v in enumerate(ratio):
        ax.text(i, v + 0.5, f'{v:.1f}×', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle(f'Cross-Framework: AllReduce ({n_gpu} GPUs)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig19_comparison_allreduce.png")


def fig20_comparison_jacobi() -> None:
    """图20: Jacobi 迭代跨框架对比 (本框架 GPU vs PETSc/NumPy CPU)"""
    data = _load_comparison_json("comparison_jacobi.json")
    if not data or not data.get("comparison"):
        print("  ⏭ 跳过: comparison_jacobi.json 不存在"); return

    comp = data["comparison"]
    n_gpu = data.get("gpu_count", 3)
    cpu_label = data.get("cpu_backend", "numpy").upper()

    # 按迭代数分组
    iter_groups = {}
    for c in comp:
        it = c["iterations"]
        iter_groups.setdefault(it, []).append(c)

    fig, axes = plt.subplots(1, min(len(iter_groups), 3), figsize=(7 * min(len(iter_groups), 3), 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, (iters, items) in enumerate(sorted(iter_groups.items())):
        if idx >= len(axes):
            break
        ax = axes[idx]
        grids = [c["grid"] for c in items]
        cpu_key = [k for k in items[0].keys() if "cpu_ms" in k][0]
        cpu_ms = [c[cpu_key] for c in items]
        gpu_ms = [c["gpu_dist_ms"] for c in items]
        speedup = [c["gpu_speedup_vs_cpu"] for c in items]

        x = np.arange(len(grids))
        w = 0.35
        bars1 = ax.bar(x - w/2, cpu_ms, w, label=f'{cpu_label} CPU', color=COLORS['blue'], alpha=0.85)
        bars2 = ax.bar(x + w/2, gpu_ms, w, label=f'Ours GPU ({n_gpu}卡)', color=COLORS['green'], alpha=0.85)
        ax.set_yscale('log')
        ax.set_xlabel('Grid Size (N×N)')
        ax.set_ylabel('Time (ms, log scale)')
        ax.set_title(f'{iters} Iterations', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in grids], rotation=45)
        ax.legend()

        # 在GPU bar上标注加速比
        for i, (bar, sp) in enumerate(zip(bars2, speedup)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                    f'{sp:.0f}×', ha='center', fontsize=9, fontweight='bold', color=COLORS['green'])

    fig.suptitle(f'Cross-Framework: Jacobi Iteration ({cpu_label} CPU vs Ours {n_gpu} GPUs)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig20_comparison_jacobi.png")


def fig21_comparison_fft() -> None:
    """图21: FFT2D 跨框架对比 (本框架 vs Dask-CUDA vs 单GPU)"""
    data = _load_comparison_json("comparison_fft.json")
    if not data or not data.get("comparison"):
        print("  ⏭ 跳过: comparison_fft.json 不存在"); return

    comp = data["comparison"]
    n_gpu = data.get("gpu_count", 3)
    labels = [f"b{c['batch']}×{c['grid']}²" for c in comp]
    single = [c["single_ms"] for c in comp]
    dask = [c.get("dask_ms", 0) for c in comp]
    ours = [c["ours_ms"] for c in comp]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (0) 时间对比 (不含Dask，因为Dask太慢会压缩其他bar)
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, single, w, label='Single GPU', color=C_SINGLE_GPU, alpha=0.85)
    ax.bar(x + w/2, ours, w, label=f'Ours ({n_gpu} GPUs)', color=C_DISTRIBUTED, alpha=0.85)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('FFT2D: Single GPU vs Ours', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.legend()

    # (1) 加速比对比
    ax = axes[1]
    sp_ours = [c.get("ours_vs_single", 0) for c in comp]
    colors_bar = [COLORS['green'] if sp >= 0.8 else COLORS['gold'] if sp >= 0.5 else COLORS['red']
                  for sp in sp_ours]
    bars = ax.bar(range(len(labels)), sp_ours, color=colors_bar, edgecolor='white', alpha=0.85)
    ax.axhline(y=1.0, color=C_BASELINE, linestyle='--', alpha=0.6, label='Baseline (1×)')
    _add_value_labels(ax, bars, fmt='{:.2f}×')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Speedup vs Single GPU')
    ax.set_title('FFT2D: Distributed Speedup', fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.legend()

    fig.suptitle(f'Cross-Framework: FFT2D ({n_gpu} GPUs)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig21_comparison_fft.png")


def main() -> None:
    global RESULTS_DIR, FIGURES_DIR

    parser = argparse.ArgumentParser(
        description="根据实验数据生成论文图表 (增强版)")
    parser.add_argument("--data-dir", "-d", type=str, default=None,
                        help="实验数据目录路径 (默认: 自动使用最新一次运行)")
    parser.add_argument("--list-runs", action="store_true",
                        help="列出所有可用的运行记录")
    args = parser.parse_args()

    # 列出历史运行
    if args.list_runs:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_root = os.path.join(project_root, "results")
        if not os.path.isdir(results_root):
            print("没有找到任何运行记录。")
            sys.exit(0)
        print("\n可用的运行记录:")
        print("-" * 70)
        print(f"  {'目录名':<30} {'数据文件数':<10} {'图表数':<10}")
        print("-" * 70)
        for d in sorted(os.listdir(results_root)):
            full = os.path.join(results_root, d)
            if os.path.isdir(full) and d.startswith("n") and "_" in d:
                jsons = glob.glob(os.path.join(full, "*.json"))
                figs_dir = os.path.join(full, "figures")
                pngs = glob.glob(os.path.join(figs_dir, "*.png")) if os.path.isdir(figs_dir) else []
                print(f"  {d:<30} {len(jsons):<10} {len(pngs):<10}")
        print("-" * 70)
        sys.exit(0)

    # 确定数据目录
    if args.data_dir:
        RESULTS_DIR = args.data_dir
    else:
        RESULTS_DIR = find_latest_run_dir()
        if RESULTS_DIR is None:
            print("错误: 没有找到任何实验数据。请先运行实验。")
            print("  python run_experiments.py --gpus 4 --exp all")
            sys.exit(1)

    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    ensure_dir(FIGURES_DIR)

    # 应用全局样式
    setup_style()

    # 提取运行信息
    run_name = os.path.basename(RESULTS_DIR)
    json_count = len(glob.glob(os.path.join(RESULTS_DIR, "*.json")))

    print(f"\n{'=' * 60}")
    print(f"  📊 生成论文图表 (增强版)")
    print(f"  数据目录: {RESULTS_DIR}")
    print(f"  运行标识: {run_name}")
    print(f"  数据文件: {json_count} 个")
    print(f"  图表输出: {FIGURES_DIR}")
    print(f"  Seaborn:  {'✓ 已启用' if HAS_SEABORN else '✗ 未安装 (使用默认样式)'}")
    print(f"  输出格式: PNG (300 DPI) + PDF")
    print(f"{'=' * 60}\n")

    # ── 生成所有图表 ──
    figure_funcs = [
        ("图1: 计算性能对比",     fig1_compute_performance),
        ("图2: 通信开销分析",     fig2_comm_overhead),
        ("图3: 强可扩展性",       fig3_strong_scaling),
        ("图4: 弱可扩展性",       fig4_weak_scaling),
        ("图5: 混合精度",         fig5_mixed_precision),
        ("图6: Kahan求和精度",    fig6_kahan),
        ("图7: 稀疏感知",         fig7_sparse),
        ("图8: Pencil FFT",       fig8_pencil_fft),
        ("图9: 流水线优化",       fig9_pipeline),
        ("图10: 代价模型策略",    fig10_cost_model),
        ("图11: Stencil应用",     fig11_stencil),
        ("图12: Jacobi应用",      fig12_jacobi),
        ("图13: Conv2d应用",      fig13_conv),
        ("图14: Einsum应用",      fig14_einsum),
        ("图15: 内存效率分析",    fig15_memory_efficiency),
        ("图16: 多算子雷达图",    fig16_multi_operator_radar),
        ("图17: 性能概览仪表盘",  fig17_summary_dashboard),
        ("图18: 跨框架MatMul对比", fig18_comparison_matmul),
        ("图19: 跨框架AllReduce对比", fig19_comparison_allreduce),
        ("图20: 跨框架Jacobi对比", fig20_comparison_jacobi),
        ("图21: 跨框架FFT对比",   fig21_comparison_fft),
    ]

    for name, func in figure_funcs:
        print(f"[{name}]")
        try:
            func()
        except Exception as e:
            print(f"  ⚠ 错误: {e}")

    # 统计生成数量
    png_count = len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
    pdf_count = len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.pdf')])

    print(f"\n{'=' * 60}")
    print(f"  ✅ 图表生成完成!")
    print(f"  PNG: {png_count} 张 (300 DPI)")
    print(f"  PDF: {pdf_count} 张 (矢量)")
    print(f"  保存在: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
