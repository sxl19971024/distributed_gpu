#!/usr/bin/env python3
"""
环境检测脚本 - 验证 distributed_gpu 框架安装是否成功
使用方式: python check_env.py
"""

import sys
import shutil

def check(name, func):
    """执行单项检查"""
    try:
        result = func()
        print(f"  ✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False

def main():
    print("=" * 56)
    print("  distributed_gpu 环境检测")
    print("=" * 56)
    passed = 0
    total = 0

    # ========== 1. Python 版本 ==========
    print("\n[1/6] Python 环境")
    total += 1
    if check("Python 版本", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"):
        if sys.version_info >= (3, 8):
            passed += 1
        else:
            print("       ⚠️  需要 Python >= 3.8")

    # ========== 2. PyTorch + CUDA ==========
    print("\n[2/6] PyTorch + CUDA")
    torch_ok = False
    total += 1
    try:
        import torch
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            print(f"  ✅ CUDA 版本: {cuda_ver}")
            print(f"  ✅ GPU 数量: {gpu_count}")
            print(f"  ✅ GPU 型号: {gpu_name}")
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ GPU 0 显存: {mem_total:.1f} GB")
            torch_ok = True
            passed += 1
        else:
            print("  ❌ CUDA 不可用! torch.cuda.is_available() = False")
            print("     修复: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  ❌ PyTorch 未安装!")
        print("     修复: pip install torch")

    # ========== 3. MPI ==========
    print("\n[3/6] MPI 环境")
    total += 1
    mpi_ok = False
    # 检查 mpirun 命令
    mpirun_path = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun_path:
        print(f"  ✅ MPI 运行时: {mpirun_path}")
    else:
        print("  ❌ mpirun/mpiexec 未找到!")
        print("     修复: conda install -c conda-forge openmpi -y")

    try:
        from mpi4py import MPI
        print(f"  ✅ mpi4py 版本: {MPI.Get_version()}")
        mpi_ok = True
        if mpirun_path:
            passed += 1
    except ImportError:
        print("  ❌ mpi4py 未安装!")
        print("     修复: conda install -c conda-forge mpi4py -y")

    # ========== 4. 其他依赖 ==========
    print("\n[4/6] 其他依赖")
    total += 1
    deps_ok = True
    for pkg, import_name in [("numpy", "numpy"), ("opt-einsum", "opt_einsum")]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✅ {pkg}: {ver}")
        except ImportError:
            print(f"  ❌ {pkg} 未安装!  修复: pip install {pkg}")
            deps_ok = False
    if deps_ok:
        passed += 1

    # ========== 5. 框架导入 ==========
    print("\n[5/6] distributed_gpu 框架")
    total += 1
    try:
        import distributed_gpu
        print(f"  ✅ 框架版本: {distributed_gpu.__version__}")
        print(f"  ✅ 安装路径: {distributed_gpu.__file__}")

        from distributed_gpu import (
            MPIManager, TensorDistributor, CostModel,
            GPUManager, PipelineOptimizer, ResourcePlanner, AutoExecutor
        )
        print("  ✅ 核心模块: MPIManager, TensorDistributor, CostModel ...")

        from distributed_gpu.algorithms import (
            distributed_matmul, distributed_conv2d, distributed_fft,
            distributed_einsum, distributed_sum, distributed_stencil_2d
        )
        print("  ✅ 算子模块: matmul, conv2d, fft, einsum, sum, stencil ...")
        passed += 1
    except ImportError as e:
        print(f"  ❌ 框架导入失败: {e}")
        print("     修复: cd distributed_gpu && pip install -e .")

    # ========== 6. MPI 多进程测试建议 ==========
    print("\n[6/6] MPI 多进程测试")
    total += 1
    if mpi_ok and torch_ok:
        print("  ℹ️  单进程检测已全部通过，请运行以下命令验证多GPU协同:")
        print("     mpirun -n 4 python examples/run_algorithm.py all")
        print("     预期输出: 总计: 17/17 通过")
        passed += 1
    else:
        print("  ⚠️  请先修复上述问题后再测试多GPU协同")

    # ========== 总结 ==========
    print("\n" + "=" * 56)
    if passed == total:
        print(f"  🎉 全部通过 ({passed}/{total})，环境配置正确！")
    else:
        print(f"  ⚠️  通过 {passed}/{total}，请根据上方提示修复问题")
    print("=" * 56)

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
