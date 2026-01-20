#!/usr/bin/env python
"""
交互式算法测试脚本

用户可以选择想要运行的算法进行测试。

运行方式：
    mpirun -n 4 --allow-run-as-root python examples/run_algorithm.py [算法名]

示例：
    mpirun -n 4 python examples/run_algorithm.py matmul
    mpirun -n 4 python examples/run_algorithm.py conv2d
    mpirun -n 4 python examples/run_algorithm.py fft
    mpirun -n 4 python examples/run_algorithm.py all

可用算法：
    matmul       - 矩阵乘法
    batch_matmul - 批量矩阵乘法
    transpose    - 矩阵转置
    add          - 张量加法
    conv2d       - 2D卷积
    conv3d       - 3D卷积
    fft          - 1D FFT
    fft2d        - 2D FFT
    ifft         - 逆FFT
    rfft         - 实数FFT
    einsum       - Einstein求和
    tensordot    - 张量点积
    sum          - 求和
    mean         - 均值
    max          - 最大值
    min          - 最小值
    all          - 运行所有算法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

from src.mpi_manager import MPIManager
from src.tensor_distributor import TensorDistributor
from src.gpu_manager import GPUManager


# ==================== 算法实现 ====================

def test_matmul(mpi, distributor):
    """矩阵乘法测试"""
    from src.algorithms.matrix_ops import distributed_matmul
    
    M, K, N = 2000, 1500, 1000
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"矩阵乘法: [{M}, {K}] @ [{K}, {N}]")
        print(f"{'='*50}")
        A = torch.randn(M, K).cuda()
        B = torch.randn(K, N).cuda()
    else:
        A, B = None, None
    
    start = time.time()
    C = distributed_matmul(A, B, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {C.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        # 验证
        C_expected = torch.matmul(A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_batch_matmul(mpi, distributor):
    """批量矩阵乘法测试"""
    from src.algorithms.matrix_ops import distributed_batch_matmul
    
    batch, M, K, N = 32, 256, 128, 256
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"批量矩阵乘法: [{batch}, {M}, {K}] @ [{batch}, {K}, {N}]")
        print(f"{'='*50}")
        A = torch.randn(batch, M, K).cuda()
        B = torch.randn(batch, K, N).cuda()
    else:
        A, B = None, None
    
    start = time.time()
    C = distributed_batch_matmul(A, B, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {C.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        C_expected = torch.bmm(A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_transpose(mpi, distributor):
    """矩阵转置测试"""
    from src.algorithms.matrix_ops import distributed_transpose
    
    M, N = 1000, 800
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"矩阵转置: [{M}, {N}] -> [{N}, {M}]")
        print(f"{'='*50}")
        A = torch.randn(M, N).cuda()
    else:
        A = None
    
    start = time.time()
    A_T = distributed_transpose(A, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {A_T.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        A_T_expected = A.T
        error = torch.max(torch.abs(A_T - A_T_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-6 else '✗ 失败'}")


def test_add(mpi, distributor):
    """张量加法测试"""
    from src.algorithms.matrix_ops import distributed_add
    
    shape = (1000, 1000)
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"张量加法: {shape} + {shape}")
        print(f"{'='*50}")
        A = torch.randn(shape).cuda()
        B = torch.randn(shape).cuda()
    else:
        A, B = None, None
    
    start = time.time()
    C = distributed_add(A, B, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {C.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        C_expected = A + B
        error = torch.max(torch.abs(C - C_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-6 else '✗ 失败'}")


def test_conv2d(mpi, distributor):
    """2D卷积测试"""
    from src.algorithms.convolution import distributed_conv2d
    
    N, C_in, H, W = 16, 32, 64, 64
    C_out, kH, kW = 64, 3, 3
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"2D卷积: [{N}, {C_in}, {H}, {W}] * [{C_out}, {C_in}, {kH}, {kW}]")
        print(f"{'='*50}")
        x = torch.randn(N, C_in, H, W).cuda()
        w = torch.randn(C_out, C_in, kH, kW).cuda()
    else:
        x, w = None, None
    
    start = time.time()
    y = distributed_conv2d(x, w, mpi, distributor, padding=(1, 1))
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {y.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        import torch.nn.functional as F
        y_expected = F.conv2d(x, w, padding=(1, 1))
        error = torch.max(torch.abs(y - y_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_conv3d(mpi, distributor):
    """3D卷积测试"""
    from src.algorithms.convolution import distributed_conv3d
    
    N, C_in, D, H, W = 8, 16, 16, 32, 32
    C_out, kD, kH, kW = 32, 3, 3, 3
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"3D卷积: [{N}, {C_in}, {D}, {H}, {W}] * [{C_out}, {C_in}, {kD}, {kH}, {kW}]")
        print(f"{'='*50}")
        x = torch.randn(N, C_in, D, H, W).cuda()
        w = torch.randn(C_out, C_in, kD, kH, kW).cuda()
    else:
        x, w = None, None
    
    start = time.time()
    y = distributed_conv3d(x, w, mpi, distributor, padding=(1, 1, 1))
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {y.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        import torch.nn.functional as F
        y_expected = F.conv3d(x, w, padding=(1, 1, 1))
        error = torch.max(torch.abs(y - y_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_fft(mpi, distributor):
    """1D FFT测试"""
    from src.algorithms.fft import distributed_fft
    
    batch, length = 64, 1024
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"1D FFT: [{batch}, {length}]")
        print(f"{'='*50}")
        x = torch.randn(batch, length).cuda()
    else:
        x = None
    
    start = time.time()
    y = distributed_fft(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {y.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        y_expected = torch.fft.fft(x)
        error = torch.max(torch.abs(y - y_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_ifft(mpi, distributor):
    """逆FFT测试"""
    from src.algorithms.fft import distributed_ifft
    
    batch, length = 64, 1024
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"逆FFT: [{batch}, {length}]")
        print(f"{'='*50}")
        x = torch.randn(batch, length, dtype=torch.complex64).cuda()
    else:
        x = None
    
    start = time.time()
    y = distributed_ifft(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {y.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        y_expected = torch.fft.ifft(x)
        error = torch.max(torch.abs(y - y_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_fft2d(mpi, distributor):
    """2D FFT测试"""
    from src.algorithms.fft import distributed_fft2d
    
    batch, H, W = 32, 128, 128
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"2D FFT: [{batch}, {H}, {W}]")
        print(f"{'='*50}")
        x = torch.randn(batch, H, W).cuda()
    else:
        x = None
    
    start = time.time()
    y = distributed_fft2d(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {y.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        y_expected = torch.fft.fft2(x)
        error = torch.max(torch.abs(y - y_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_rfft(mpi, distributor):
    """实数FFT测试"""
    from src.algorithms.fft import distributed_rfft
    
    batch, length = 64, 1024
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"实数FFT: [{batch}, {length}]")
        print(f"{'='*50}")
        x = torch.randn(batch, length).cuda()
    else:
        x = None
    
    start = time.time()
    y = distributed_rfft(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {y.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        y_expected = torch.fft.rfft(x)
        error = torch.max(torch.abs(y - y_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_einsum(mpi, distributor):
    """Einstein求和测试"""
    from src.algorithms.einsum import distributed_einsum
    
    batch, M, K, N = 32, 128, 64, 128
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"Einstein求和 'bij,bjk->bik': [{batch}, {M}, {K}] @ [{batch}, {K}, {N}]")
        print(f"{'='*50}")
        A = torch.randn(batch, M, K).cuda()
        B = torch.randn(batch, K, N).cuda()
    else:
        A, B = None, None
    
    start = time.time()
    C = distributed_einsum('bij,bjk->bik', A, B, mpi=mpi, distributor=distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {C.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        C_expected = torch.einsum('bij,bjk->bik', A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_tensordot(mpi, distributor):
    """张量点积测试"""
    from src.algorithms.einsum import distributed_tensordot
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"张量点积: [64, 32, 16] · [16, 32, 64], dims=2")
        print(f"{'='*50}")
        a = torch.randn(64, 32, 16).cuda()
        b = torch.randn(16, 32, 64).cuda()
    else:
        a, b = None, None
    
    start = time.time()
    c = distributed_tensordot(a, b, dims=2, mpi=mpi, distributor=distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {c.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        c_expected = torch.tensordot(a, b, dims=2)
        error = torch.max(torch.abs(c - c_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_sum(mpi, distributor):
    """求和测试"""
    from src.algorithms.reduction import distributed_sum
    
    shape = (64, 128, 128)
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"求和: {shape}")
        print(f"{'='*50}")
        x = torch.randn(shape).cuda()
    else:
        x = None
    
    start = time.time()
    s = distributed_sum(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果: {s.item():.4f}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        s_expected = x.sum()
        error = abs(s.item() - s_expected.item())
        print(f"误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-3 else '✗ 失败'}")


def test_mean(mpi, distributor):
    """均值测试"""
    from src.algorithms.reduction import distributed_mean
    
    shape = (64, 128, 128)
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"均值: {shape}")
        print(f"{'='*50}")
        x = torch.randn(shape).cuda()
    else:
        x = None
    
    start = time.time()
    m = distributed_mean(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果: {m.item():.6f}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        m_expected = x.mean()
        error = abs(m.item() - m_expected.item())
        print(f"误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-5 else '✗ 失败'}")


def test_max(mpi, distributor):
    """最大值测试"""
    from src.algorithms.reduction import distributed_max
    
    shape = (64, 128, 128)
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"最大值: {shape}")
        print(f"{'='*50}")
        x = torch.randn(shape).cuda()
    else:
        x = None
    
    start = time.time()
    mx = distributed_max(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果: {mx.item():.4f}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        mx_expected = x.max()
        error = abs(mx.item() - mx_expected.item())
        print(f"误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-5 else '✗ 失败'}")


def test_min(mpi, distributor):
    """最小值测试"""
    from src.algorithms.reduction import distributed_min
    
    shape = (64, 128, 128)
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"最小值: {shape}")
        print(f"{'='*50}")
        x = torch.randn(shape).cuda()
    else:
        x = None
    
    start = time.time()
    mn = distributed_min(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果: {mn.item():.4f}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        mn_expected = x.min()
        error = abs(mn.item() - mn_expected.item())
        print(f"误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-5 else '✗ 失败'}")


# ==================== 算法映射 ====================

ALGORITHMS = {
    'matmul': ('矩阵乘法', test_matmul),
    'batch_matmul': ('批量矩阵乘法', test_batch_matmul),
    'transpose': ('矩阵转置', test_transpose),
    'add': ('张量加法', test_add),
    'conv2d': ('2D卷积', test_conv2d),
    'conv3d': ('3D卷积', test_conv3d),
    'fft': ('1D FFT', test_fft),
    'ifft': ('逆FFT', test_ifft),
    'fft2d': ('2D FFT', test_fft2d),
    'rfft': ('实数FFT', test_rfft),
    'einsum': ('Einstein求和', test_einsum),
    'tensordot': ('张量点积', test_tensordot),
    'sum': ('求和', test_sum),
    'mean': ('均值', test_mean),
    'max': ('最大值', test_max),
    'min': ('最小值', test_min),
}


def print_help(mpi):
    """打印帮助信息"""
    if mpi.is_master_process():
        print("\n" + "=" * 60)
        print("分布式GPU计算框架 - 算法测试工具")
        print("=" * 60)
        print("\n用法:")
        print("  mpirun -n 4 python examples/run_algorithm.py <算法名>")
        print("\n可用算法:")
        for key, (name, _) in ALGORITHMS.items():
            print(f"  {key:15s} - {name}")
        print(f"  {'all':15s} - 运行所有算法")
        print(f"  {'list':15s} - 显示此帮助")
        print("\n示例:")
        print("  mpirun -n 4 python examples/run_algorithm.py matmul")
        print("  mpirun -n 4 python examples/run_algorithm.py conv2d")
        print("  mpirun -n 4 python examples/run_algorithm.py all")
        print("=" * 60)


def main():
    # 初始化
    mpi = MPIManager()
    distributor = TensorDistributor(mpi)
    gpu = GPUManager(mpi.get_gpu_id())
    
    # 解析命令行参数
    if len(sys.argv) < 2:
        print_help(mpi)
        return
    
    algo_name = sys.argv[1].lower()
    
    if algo_name == 'list' or algo_name == 'help' or algo_name == '-h':
        print_help(mpi)
        return
    
    # 打印 GPU 信息
    gpu.print_info()
    
    if algo_name == 'all':
        # 运行所有算法
        if mpi.is_master_process():
            print("\n运行所有算法测试...")
        
        for key, (name, func) in ALGORITHMS.items():
            try:
                func(mpi, distributor)
            except Exception as e:
                if mpi.is_master_process():
                    print(f"\n{name}: ✗ 错误 - {e}")
            mpi.synchronize()
        
        if mpi.is_master_process():
            print("\n" + "=" * 50)
            print("所有测试完成！")
            print("=" * 50)
    
    elif algo_name in ALGORITHMS:
        # 运行指定算法
        name, func = ALGORITHMS[algo_name]
        try:
            func(mpi, distributor)
        except Exception as e:
            if mpi.is_master_process():
                print(f"\n错误: {e}")
    
    else:
        if mpi.is_master_process():
            print(f"\n错误: 未知算法 '{algo_name}'")
        print_help(mpi)
    
    mpi.synchronize()


if __name__ == "__main__":
    main()
