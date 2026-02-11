#!/usr/bin/env python3
"""
实验运行入口 - 自动检测 GPU 并通过 MPI 启动实验，实验结束后自动生成图表

用法:
  python run_experiments.py                    # 自动检测GPU数量，运行全部实验并生成图表
  python run_experiments.py --gpus 2           # 使用2个GPU运行全部实验
  python run_experiments.py --gpus 4 --exp 1   # 使用4个GPU只运行实验1
  python run_experiments.py --gpus 8 --exp all # 使用8个GPU运行全部实验
  python run_experiments.py --list             # 查看可用实验列表
  python run_experiments.py --figures           # 只生成图表(使用最新实验数据)
  python run_experiments.py --figures --data-dir results/n4_20260211_143025  # 指定数据目录生成图表
  python run_experiments.py --list-runs         # 查看所有历史运行记录
  python run_experiments.py --no-figures         # 只运行实验，不生成图表

每次运行的结果保存在 results/n{GPU数}_{时间戳}/ 目录下，多次运行互不覆盖。
"""

import argparse
import subprocess
import sys
import os
import shutil
from datetime import datetime


def detect_gpu_count():
    """自动检测可用 GPU 数量"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    # fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 1


def print_experiment_list():
    """打印可用实验列表"""
    experiments = [
        ("1", "计算性能对比", "矩阵乘法不同规模的GFLOPS和耗时"),
        ("2", "通信开销分析", "不同数据量下MPI通信时间占比"),
        ("3", "强可扩展性",   "固定问题规模，增加GPU数量的加速比"),
        ("4", "弱可扩展性",   "每GPU固定数据量，增加GPU数量的效率"),
        ("5", "创新算子对比", "混合精度/稀疏感知/Kahan求和/Pencil FFT"),
        ("6", "流水线优化",   "计算-通信重叠的加速效果"),
        ("7", "代价模型策略", "行分割/列分割/2D块分割的自动选择"),
        ("8", "科学计算应用", "Stencil/Jacobi/Conv2D/Einsum"),
    ]
    print("\n可用实验列表:")
    print("-" * 65)
    print(f"  {'ID':<4} {'名称':<16} {'说明'}")
    print("-" * 65)
    for eid, name, desc in experiments:
        print(f"  {eid:<4} {name:<16} {desc}")
    print("-" * 65)
    print(f"  {'all':<4} {'运行全部':<16} {'依次运行实验1~8'}")
    print()


def list_run_history():
    """列出所有历史运行记录"""
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "results")

    if not os.path.isdir(results_root):
        print("没有找到任何运行记录。")
        return

    runs = []
    for d in sorted(os.listdir(results_root)):
        full = os.path.join(results_root, d)
        if os.path.isdir(full) and d.startswith("n") and "_" in d:
            jsons = glob.glob(os.path.join(full, "*.json"))
            figs_dir = os.path.join(full, "figures")
            pngs = glob.glob(os.path.join(figs_dir, "*.png")) if os.path.isdir(figs_dir) else []
            # 解析目录名: n{gpu}_{timestamp}
            parts = d.split("_", 1)
            gpu_str = parts[0][1:] if parts[0].startswith("n") else "?"
            ts_str = parts[1] if len(parts) > 1 else "?"
            runs.append((d, gpu_str, ts_str, len(jsons), len(pngs)))

    if not runs:
        # 检查旧版目录
        legacy = os.path.join(results_root, "thesis_enhanced")
        if os.path.isdir(legacy):
            jsons = glob.glob(os.path.join(legacy, "*.json"))
            figs_dir = os.path.join(legacy, "figures")
            pngs = glob.glob(os.path.join(figs_dir, "*.png")) if os.path.isdir(figs_dir) else []
            print(f"\n  (旧版) thesis_enhanced/: {len(jsons)} 个数据文件, {len(pngs)} 张图")
        else:
            print("没有找到任何运行记录。")
        return

    print(f"\n历史运行记录 ({len(runs)} 次):")
    print("-" * 72)
    print(f"  {'目录名':<30} {'GPU数':<6} {'时间戳':<16} {'数据':<6} {'图表':<6}")
    print("-" * 72)
    for name, gpus, ts, nj, np_ in runs:
        # 格式化时间戳
        ts_display = ts
        if len(ts) == 15:  # 20260211_143025
            try:
                ts_display = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
            except:
                pass
        print(f"  {name:<30} {gpus:<6} {ts_display:<16} {nj:<6} {np_:<6}")
    print("-" * 72)
    print(f"\n使用 --data-dir results/<目录名> 指定特定运行来生成图表")


def generate_figures(data_dir=None):
    """运行图表生成脚本"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_script = os.path.join(script_dir, "experiments", "generate_thesis_figures_enhanced.py")

    if not os.path.exists(fig_script):
        print("⚠ 图表生成脚本不存在，跳过图表生成")
        return 1

    cmd = [sys.executable, fig_script]
    if data_dir:
        cmd += ["--data-dir", data_dir]

    print(f"\n📊 开始生成图表...")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="distributed_gpu 实验运行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_experiments.py                             # 自动检测GPU，运行全部实验+生成图表
  python run_experiments.py --gpus 2                    # 2个GPU，运行全部
  python run_experiments.py --gpus 4 --exp 1            # 4个GPU，只运行实验1
  python run_experiments.py --gpus 8 --exp all          # 8个GPU，运行全部
  python run_experiments.py --list                      # 查看实验列表
  python run_experiments.py --figures                   # 只生成图表(最新数据)
  python run_experiments.py --figures --data-dir results/n4_20260211_143025
  python run_experiments.py --list-runs                 # 查看历史运行
  python run_experiments.py --no-figures                # 只运行实验，不生成图表

结果目录结构:
  results/
  ├── n4_20260211_143025/          ← 用4个GPU的第1次运行
  │   ├── exp1_compute_performance.json
  │   ├── exp2_comm_overhead.json
  │   ├── ...
  │   └── figures/
  │       ├── fig1_compute_perf_n4.png
  │       └── ...
  ├── n8_20260211_150000/          ← 用8个GPU的运行
  │   └── ...
  └── n4_20260212_091000/          ← 用4个GPU的第2次运行(不覆盖)
        """
    )
    parser.add_argument("--gpus", "-g", type=int, default=None,
                        help="使用的GPU数量 (默认: 自动检测全部可用GPU)")
    parser.add_argument("--exp", "-e", type=str, default="all",
                        help="实验ID: 1~8 或 all (默认: all)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="查看可用实验列表")
    parser.add_argument("--list-runs", action="store_true",
                        help="查看所有历史运行记录")
    parser.add_argument("--figures", action="store_true",
                        help="只生成图表(不运行实验)")
    parser.add_argument("--no-figures", action="store_true",
                        help="只运行实验，不自动生成图表")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="指定数据目录来生成图表 (与 --figures 配合使用)")

    args = parser.parse_args()

    # 查看实验列表
    if args.list:
        print_experiment_list()
        return 0

    # 查看历史运行
    if args.list_runs:
        list_run_history()
        return 0

    # 只生成图表
    if args.figures:
        return generate_figures(args.data_dir)

    # === 运行实验 ===

    # 检查 mpirun
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if not mpirun:
        print("❌ 错误: 未找到 mpirun/mpiexec，请先安装 MPI")
        print("   修复: conda install -c conda-forge openmpi -y")
        return 1

    # 确定 GPU 数量
    if args.gpus is not None:
        gpu_count = args.gpus
    else:
        gpu_count = detect_gpu_count()

    if gpu_count < 1:
        print("❌ 错误: 未检测到可用 GPU")
        return 1

    # 生成输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"n{gpu_count}_{timestamp}"
    output_dir = os.path.join(script_dir, "results", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # 实验脚本路径
    exp_script = os.path.join(script_dir, "experiments", "thesis_experiments_enhanced.py")

    if not os.path.exists(exp_script):
        print(f"❌ 错误: 实验脚本不存在: {exp_script}")
        return 1

    # 构建 mpirun 命令
    cmd = [
        mpirun,
        "-n", str(gpu_count),
        "--allow-run-as-root",       # 兼容 Docker/root 用户
        "--oversubscribe",           # GPU数 < 进程数时允许共享
        sys.executable,              # 当前 Python 解释器
        exp_script,
        args.exp,
        "--output-dir", output_dir,
    ]

    print(f"🚀 启动实验")
    print(f"   GPU 数量:  {gpu_count}")
    print(f"   实验:      {'全部 (1~8)' if args.exp == 'all' else f'实验{args.exp}'}")
    print(f"   输出目录:  {output_dir}")
    print(f"   命令:      {' '.join(cmd)}")
    print()

    # 执行实验
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ 实验运行失败 (退出码: {result.returncode})")
        return result.returncode

    # 自动生成图表
    if not args.no_figures:
        fig_ret = generate_figures(output_dir)
        if fig_ret != 0:
            print("⚠ 图表生成失败，但实验数据已保存")
    else:
        print(f"\n📁 实验数据已保存至: {output_dir}")
        print(f"   稍后可运行: python run_experiments.py --figures --data-dir {output_dir}")

    print(f"\n✅ 完成! 结果目录: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
