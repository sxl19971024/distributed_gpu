#!/usr/bin/env python3
"""
实验运行入口 - 自动检测 GPU 并通过 MPI 启动实验

用法:
  python run_experiments.py                    # 自动检测GPU数量，运行全部实验
  python run_experiments.py --gpus 2           # 使用2个GPU运行全部实验
  python run_experiments.py --gpus 4 --exp 1   # 使用4个GPU只运行实验1
  python run_experiments.py --exp 3            # 自动检测GPU，只运行实验3
  python run_experiments.py --list             # 查看可用实验列表
"""

import argparse
import subprocess
import sys
import os
import shutil


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


def main():
    parser = argparse.ArgumentParser(
        description="distributed_gpu 实验运行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_experiments.py                    # 自动检测GPU，运行全部
  python run_experiments.py --gpus 2           # 2个GPU，运行全部
  python run_experiments.py --gpus 4 --exp 1   # 4个GPU，只运行实验1
  python run_experiments.py --list             # 查看实验列表
        """
    )
    parser.add_argument("--gpus", "-g", type=int, default=None,
                        help="使用的GPU数量 (默认: 自动检测全部可用GPU)")
    parser.add_argument("--exp", "-e", type=str, default="all",
                        help="实验ID: 1~8 或 all (默认: all)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="查看可用实验列表")

    args = parser.parse_args()

    if args.list:
        print_experiment_list()
        return 0

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

    # 实验脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
    ]

    print(f"🚀 启动实验")
    print(f"   GPU 数量: {gpu_count}")
    print(f"   实验: {'全部 (1~8)' if args.exp == 'all' else f'实验{args.exp}'}")
    print(f"   命令: {' '.join(cmd)}")
    print()

    # 执行
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
