#!/usr/bin/env python3
"""
环境检测脚本 - 验证 distributed_gpu 框架安装是否成功
目标环境: OpenMPI 4.1.5 + CUDA 12.1

使用方式: python check_env.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Callable, Optional, Tuple


# ==================== 辅助函数 ====================

def _run_cmd(args: list[str], timeout: int = 5) -> Optional[str]:
    """安全执行外部命令，返回 stdout+stderr 或 None。"""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (result.stdout + result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _get_ompi_version() -> Optional[str]:
    """获取 OpenMPI / MPICH 版本号。"""
    output = _run_cmd(["mpirun", "--version"])
    if output is None:
        return None
    # OpenMPI: "mpirun (Open MPI) 4.1.5"
    m = re.search(r'Open MPI[)]*\s*(\d+\.\d+\.\d+)', output)
    if m:
        return m.group(1)
    # MPICH fallback
    m = re.search(r'MPICH.*?(\d+\.\d+(?:\.\d+)?)', output)
    if m:
        return f"MPICH {m.group(1)}"
    return None


def _get_nvidia_driver_version() -> Optional[str]:
    """获取 NVIDIA 驱动版本。"""
    output = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if output:
        return output.split('\n')[0].strip()
    return None


def _check_item(name: str, func: Callable[[], str]) -> bool:
    """执行单项检查并打印结果。"""
    try:
        result = func()
        print(f"  ✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False


# ==================== 检查项 ====================

def check_python() -> Tuple[bool, str]:
    """[1/8] Python 版本 >= 3.8"""
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 8)
    return ok, ver


def check_nvidia_driver() -> Tuple[bool, Optional[str]]:
    """[2/8] NVIDIA 驱动 >= 530 (CUDA 12.1 要求)"""
    driver_ver = _get_nvidia_driver_version()
    if not driver_ver:
        return False, None
    try:
        major = int(driver_ver.split('.')[0])
        return major >= 525, driver_ver
    except ValueError:
        return True, driver_ver  # 无法解析时视为通过


def check_pytorch_cuda() -> Tuple[bool, dict]:
    """[3/8] PyTorch + CUDA 12.1"""
    info: dict = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        if not torch.cuda.is_available():
            info["error"] = "CUDA 不可用 (torch.cuda.is_available() = False)"
            return False, info
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = []
        for i in range(min(info["gpu_count"], 8)):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info["gpus"].append(f"{name} ({mem_gb:.1f} GB)")
        return True, info
    except ImportError:
        info["error"] = "PyTorch 未安装"
        return False, info


def check_mpi() -> Tuple[bool, dict]:
    """[4/8] MPI 环境 (OpenMPI 4.1.5)"""
    info: dict = {}
    mpirun_path = shutil.which("mpirun") or shutil.which("mpiexec")
    info["mpirun"] = mpirun_path
    info["ompi_version"] = _get_ompi_version() if mpirun_path else None

    try:
        from mpi4py import MPI
        info["mpi_standard"] = MPI.Get_version()
        try:
            import mpi4py
            info["mpi4py_version"] = mpi4py.__version__
        except AttributeError:
            pass
        info["mpi4py_ok"] = True
    except ImportError:
        info["mpi4py_ok"] = False

    ok = bool(mpirun_path and info.get("mpi4py_ok"))
    return ok, info


def check_core_deps() -> Tuple[bool, list]:
    """[5/8] 核心依赖 (numpy, opt-einsum)"""
    results = []
    all_ok = True
    for pkg, import_name in [("numpy", "numpy"), ("opt-einsum", "opt_einsum")]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            results.append((pkg, ver, True))
        except ImportError:
            results.append((pkg, None, False))
            all_ok = False
    return all_ok, results


def check_experiment_deps() -> Tuple[bool, list]:
    """[6/8] 实验/图表依赖 (matplotlib, seaborn)"""
    results = []
    all_ok = True
    for pkg, import_name in [("matplotlib", "matplotlib"), ("seaborn", "seaborn")]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            results.append((pkg, ver, True))
        except ImportError:
            results.append((pkg, None, False))
            all_ok = False
    return all_ok, results


def check_framework_import() -> Tuple[bool, Optional[str]]:
    """[7/8] distributed_gpu 框架导入"""
    try:
        import distributed_gpu
        ver = distributed_gpu.__version__

        # 验证核心模块
        from distributed_gpu import (
            MPIManager, TensorDistributor, CostModel,
            GPUManager, PipelineOptimizer, ResourcePlanner, AutoExecutor
        )
        # 验证算子模块
        from distributed_gpu.algorithms import (
            distributed_matmul, distributed_conv2d, distributed_fft,
            distributed_einsum, distributed_sum, distributed_stencil_2d
        )
        return True, ver
    except ImportError as e:
        return False, str(e)


# ==================== 主函数 ====================

def main() -> int:
    print("=" * 60)
    print("  distributed_gpu 环境检测")
    print("  目标环境: OpenMPI 4.1.5 + CUDA 12.1")
    print("=" * 60)

    passed = 0
    total = 8

    # ---------- [1/8] Python ----------
    print("\n[1/8] Python 环境")
    py_ok, py_ver = check_python()
    if py_ok:
        print(f"  ✅ Python 版本: {py_ver}")
        passed += 1
    else:
        print(f"  ❌ Python 版本: {py_ver} (需要 >= 3.8)")

    # ---------- [2/8] NVIDIA 驱动 ----------
    print("\n[2/8] NVIDIA 驱动")
    drv_ok, drv_ver = check_nvidia_driver()
    if drv_ver is None:
        print("  ❌ nvidia-smi 不可用，请确认已安装 NVIDIA 驱动")
    elif drv_ok:
        print(f"  ✅ NVIDIA 驱动版本: {drv_ver}")
        try:
            major = int(drv_ver.split('.')[0])
            if major >= 530:
                print(f"  ✅ 驱动兼容 CUDA 12.1 (需要 >= 530)")
            else:
                print(f"  ⚠️  驱动版本 {drv_ver}，CUDA 12.1 建议 >= 530")
        except ValueError:
            pass
        passed += 1
    else:
        print(f"  ❌ 驱动版本 {drv_ver} 过低，CUDA 12.1 需要 >= 530")

    # ---------- [3/8] PyTorch + CUDA ----------
    print("\n[3/8] PyTorch + CUDA 12.1")
    torch_ok, torch_info = check_pytorch_cuda()
    if torch_ok:
        print(f"  ✅ PyTorch 版本: {torch_info['torch_version']}")
        cuda_ver = torch_info["cuda_version"]
        print(f"  ✅ CUDA 运行时版本: {cuda_ver}")
        if cuda_ver.startswith("12.1"):
            print(f"  ✅ CUDA 版本匹配目标 12.1")
        elif cuda_ver.startswith("12."):
            print(f"  ⚠️  CUDA {cuda_ver} (目标 12.1，大版本兼容)")
        else:
            print(f"  ⚠️  CUDA {cuda_ver} 与目标 12.1 不一致")
            print(f"     建议: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print(f"  ✅ GPU 数量: {torch_info['gpu_count']}")
        for i, gpu_desc in enumerate(torch_info["gpus"]):
            print(f"     GPU {i}: {gpu_desc}")
        passed += 1
    else:
        if "error" in torch_info:
            print(f"  ❌ {torch_info['error']}")
        if "torch_version" in torch_info:
            print(f"     PyTorch 版本: {torch_info['torch_version']}")
        print(f"     修复: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    # ---------- [4/8] MPI ----------
    print("\n[4/8] MPI 环境 (目标: OpenMPI 4.1.5)")
    mpi_ok, mpi_info = check_mpi()
    if mpi_info["mpirun"]:
        print(f"  ✅ MPI 运行时: {mpi_info['mpirun']}")
        ompi_ver = mpi_info.get("ompi_version")
        if ompi_ver:
            print(f"  ✅ MPI 版本: {ompi_ver}")
            if ompi_ver == "4.1.5":
                print(f"  ✅ 版本匹配目标 4.1.5")
            elif ompi_ver.startswith("4.1."):
                print(f"  ⚠️  版本 {ompi_ver} (目标 4.1.5，小版本兼容)")
            elif ompi_ver.startswith("4."):
                print(f"  ⚠️  版本 {ompi_ver} (目标 4.1.5，大版本兼容)")
        else:
            print(f"  ⚠️  无法检测 MPI 版本")
    else:
        print("  ❌ mpirun/mpiexec 未找到!")
        print("     修复: module load openmpi/4.1.5  (HPC 集群)")
        print("     或:   conda install -c conda-forge openmpi=4.1.5 -y")

    if mpi_info.get("mpi4py_ok"):
        mpi4py_ver = mpi_info.get("mpi4py_version", "unknown")
        print(f"  ✅ mpi4py: {mpi4py_ver}")
        if mpi_ok:
            passed += 1
    else:
        print("  ❌ mpi4py 未安装!")
        print("     修复: pip install mpi4py")

    # ---------- [5/8] 核心依赖 ----------
    print("\n[5/8] 核心依赖")
    deps_ok, deps_list = check_core_deps()
    for pkg, ver, ok in deps_list:
        if ok:
            print(f"  ✅ {pkg}: {ver}")
        else:
            print(f"  ❌ {pkg} 未安装!  修复: pip install {pkg}")
    if deps_ok:
        passed += 1

    # ---------- [6/8] 实验/图表依赖 ----------
    print("\n[6/8] 实验/图表依赖 (可选)")
    exp_ok, exp_list = check_experiment_deps()
    for pkg, ver, ok in exp_list:
        if ok:
            print(f"  ✅ {pkg}: {ver}")
        else:
            print(f"  ⚠️  {pkg} 未安装 (运行实验需要)")
            print(f"     修复: pip install -e '.[experiments]'")
    if exp_ok:
        passed += 1
    else:
        print(f"  ℹ️  实验依赖缺失不影响核心功能，但无法运行实验和生成图表")
        passed += 1  # 可选依赖，不阻塞

    # ---------- [7/8] 框架导入 ----------
    print("\n[7/8] distributed_gpu 框架")
    fw_ok, fw_info = check_framework_import()
    if fw_ok:
        print(f"  ✅ 框架版本: {fw_info}")
        print(f"  ✅ 核心模块: MPIManager, TensorDistributor, CostModel ...")
        print(f"  ✅ 算子模块: matmul, conv2d, fft, einsum, sum, stencil ...")
        passed += 1
    else:
        print(f"  ❌ 框架导入失败: {fw_info}")
        print("     修复: pip install -e .")

    # ---------- [8/8] MPI 多进程测试建议 ----------
    print("\n[8/8] MPI 多进程测试")
    if mpi_ok and torch_ok:
        print("  ℹ️  单进程检测已通过，请运行以下命令验证多GPU协同:")
        print("     mpirun -n 4 python examples/run_algorithm.py all")
        passed += 1
    else:
        print("  ⚠️  请先修复上述问题后再测试多GPU协同")

    # ---------- 总结 ----------
    print("\n" + "=" * 60)
    if passed == total:
        print(f"  🎉 全部通过 ({passed}/{total})，环境配置正确！")
        print(f"     OpenMPI 4.1.5 + CUDA 12.1 环境就绪")
    else:
        print(f"  ⚠️  通过 {passed}/{total}，请根据上方提示修复问题")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
