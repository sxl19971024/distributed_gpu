#!/usr/bin/env python
"""
综合测试脚本

运行所有分布式算法测试，验证框架正确性。

运行方式：
    mpirun -n 4 python examples/test_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.mpi_manager import MPIManager
from src.tensor_distributor import TensorDistributor
from src.gpu_manager import GPUManager
from src.utils.profiler import Profiler

from src.algorithms.matrix_ops import distributed_matmul, distributed_batch_matmul
from src.algorithms.convolution import distributed_conv2d
from src.algorithms.fft import distributed_fft, distributed_fft2d
from src.algorithms.einsum import distributed_einsum
from src.algorithms.reduction import distributed_sum, distributed_mean


def test_matmul(mpi, distributor, profiler):
    """测试分布式矩阵乘法"""
    M, K, N = 1000, 1000, 1000
    
    if mpi.is_master_process():
        print(f"\n[测试] 分布式矩阵乘法 [{M}, {K}] @ [{K}, {N}]")
        A = torch.randn(M, K, dtype=torch.float32).cuda()
        B = torch.randn(K, N, dtype=torch.float32).cuda()
    else:
        A, B = None, None
    
    profiler.start("matmul")
    C = distributed_matmul(A, B, mpi, distributor)
    profiler.end("matmul")
    
    # 验证结果
    if mpi.is_master_process():
        C_expected = torch.matmul(A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        status = "✓ 通过" if error < 1e-4 else "✗ 失败"
        print(f"  最大误差: {error:.6e} {status}")
        return error < 1e-4
    return True


def test_batch_matmul(mpi, distributor, profiler):
    """测试分布式批量矩阵乘法"""
    batch, M, K, N = 32, 256, 256, 256
    
    if mpi.is_master_process():
        print(f"\n[测试] 分布式批量矩阵乘法 [{batch}, {M}, {K}] @ [{batch}, {K}, {N}]")
        A = torch.randn(batch, M, K, dtype=torch.float32).cuda()
        B = torch.randn(batch, K, N, dtype=torch.float32).cuda()
    else:
        A, B = None, None
    
    profiler.start("batch_matmul")
    C = distributed_batch_matmul(A, B, mpi, distributor)
    profiler.end("batch_matmul")
    
    if mpi.is_master_process():
        C_expected = torch.bmm(A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        status = "✓ 通过" if error < 1e-4 else "✗ 失败"
        print(f"  最大误差: {error:.6e} {status}")
        return error < 1e-4
    return True


def test_conv2d(mpi, distributor, profiler):
    """测试分布式2D卷积"""
    N, C_in, H, W = 32, 64, 128, 128
    C_out, kH, kW = 128, 3, 3
    
    if mpi.is_master_process():
        print(f"\n[测试] 分布式2D卷积 [{N}, {C_in}, {H}, {W}] * [{C_out}, {C_in}, {kH}, {kW}]")
        input_tensor = torch.randn(N, C_in, H, W, dtype=torch.float32).cuda()
        weight = torch.randn(C_out, C_in, kH, kW, dtype=torch.float32).cuda()
    else:
        input_tensor, weight = None, None
    
    profiler.start("conv2d")
    output = distributed_conv2d(input_tensor, weight, mpi, distributor, padding=(1, 1))
    profiler.end("conv2d")
    
    if mpi.is_master_process():
        import torch.nn.functional as F
        output_expected = F.conv2d(input_tensor, weight, padding=(1, 1))
        error = torch.max(torch.abs(output - output_expected)).item()
        status = "✓ 通过" if error < 1e-4 else "✗ 失败"
        print(f"  最大误差: {error:.6e} {status}")
        return error < 1e-4
    return True


def test_fft(mpi, distributor, profiler):
    """测试分布式FFT"""
    batch, length = 64, 1024
    
    if mpi.is_master_process():
        print(f"\n[测试] 分布式FFT [{batch}, {length}]")
        input_tensor = torch.randn(batch, length, dtype=torch.float32).cuda()
    else:
        input_tensor = None
    
    profiler.start("fft")
    output = distributed_fft(input_tensor, mpi, distributor)
    profiler.end("fft")
    
    if mpi.is_master_process():
        output_expected = torch.fft.fft(input_tensor)
        error = torch.max(torch.abs(output - output_expected)).item()
        status = "✓ 通过" if error < 1e-4 else "✗ 失败"
        print(f"  最大误差: {error:.6e} {status}")
        return error < 1e-4
    return True


def test_fft2d(mpi, distributor, profiler):
    """测试分布式2D FFT"""
    batch, H, W = 32, 256, 256
    
    if mpi.is_master_process():
        print(f"\n[测试] 分布式2D FFT [{batch}, {H}, {W}]")
        input_tensor = torch.randn(batch, H, W, dtype=torch.float32).cuda()
    else:
        input_tensor = None
    
    profiler.start("fft2d")
    output = distributed_fft2d(input_tensor, mpi, distributor)
    profiler.end("fft2d")
    
    if mpi.is_master_process():
        output_expected = torch.fft.fft2(input_tensor)
        error = torch.max(torch.abs(output - output_expected)).item()
        status = "✓ 通过" if error < 1e-4 else "✗ 失败"
        print(f"  最大误差: {error:.6e} {status}")
        return error < 1e-4
    return True


def test_einsum(mpi, distributor, profiler):
    """测试分布式Einsum"""
    batch, M, K, N = 32, 128, 128, 128
    
    if mpi.is_master_process():
        print(f"\n[测试] 分布式Einsum 'bij,bjk->bik' [{batch}, {M}, {K}]")
        A = torch.randn(batch, M, K, dtype=torch.float32).cuda()
        B = torch.randn(batch, K, N, dtype=torch.float32).cuda()
    else:
        A, B = None, None
    
    profiler.start("einsum")
    C = distributed_einsum("bij,bjk->bik", A, B, mpi=mpi, distributor=distributor)
    profiler.end("einsum")
    
    if mpi.is_master_process():
        C_expected = torch.einsum("bij,bjk->bik", A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        status = "✓ 通过" if error < 1e-4 else "✗ 失败"
        print(f"  最大误差: {error:.6e} {status}")
        return error < 1e-4
    return True


def test_reduction(mpi, distributor, profiler):
    """测试分布式归约操作"""
    shape = (64, 128, 128)
    
    if mpi.is_master_process():
        print(f"\n[测试] 分布式归约 {shape}")
        tensor = torch.randn(shape, dtype=torch.float32).cuda()
    else:
        tensor = None
    
    # 测试求和
    profiler.start("sum")
    result_sum = distributed_sum(tensor, mpi, distributor, dim=None)
    profiler.end("sum")
    
    # 测试均值
    profiler.start("mean")
    result_mean = distributed_mean(tensor, mpi, distributor, dim=None)
    profiler.end("mean")
    
    if mpi.is_master_process():
        expected_sum = tensor.sum()
        expected_mean = tensor.mean()
        
        error_sum = abs(result_sum.item() - expected_sum.item())
        error_mean = abs(result_mean.item() - expected_mean.item())
        
        status_sum = "✓" if error_sum < 1e-3 else "✗"
        status_mean = "✓" if error_mean < 1e-5 else "✗"
        
        print(f"  Sum误差: {error_sum:.6e} {status_sum}")
        print(f"  Mean误差: {error_mean:.6e} {status_mean}")
        
        return error_sum < 1e-3 and error_mean < 1e-5
    return True


def main():
    # 初始化
    mpi = MPIManager()
    distributor = TensorDistributor(mpi)
    gpu_manager = GPUManager(mpi.get_gpu_id())
    profiler = Profiler(enabled=mpi.is_master_process())
    
    if mpi.is_master_process():
        print("=" * 60)
        print("分布式GPU计算框架 - 综合测试")
        print("=" * 60)
        gpu_manager.print_info()
    
    mpi.synchronize()
    
    # 运行所有测试
    tests = [
        ("矩阵乘法", test_matmul),
        ("批量矩阵乘法", test_batch_matmul),
        ("2D卷积", test_conv2d),
        ("1D FFT", test_fft),
        ("2D FFT", test_fft2d),
        ("Einsum", test_einsum),
        ("归约操作", test_reduction),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func(mpi, distributor, profiler)
            results.append((name, passed))
        except Exception as e:
            if mpi.is_master_process():
                print(f"\n[测试] {name}: ✗ 异常 - {str(e)}")
            results.append((name, False))
        
        mpi.synchronize()
    
    # 打印摘要
    if mpi.is_master_process():
        print("\n" + "=" * 60)
        print("测试结果摘要")
        print("=" * 60)
        
        passed_count = sum(1 for _, p in results if p)
        total_count = len(results)
        
        for name, passed in results:
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"  {name}: {status}")
        
        print(f"\n总计: {passed_count}/{total_count} 通过")
        
        profiler.print_summary()
        gpu_manager.print_memory_info()
    
    mpi.synchronize()


if __name__ == "__main__":
    main()
