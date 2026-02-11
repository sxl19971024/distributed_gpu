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
    matmul          - 矩阵乘法
    batch_matmul    - 批量矩阵乘法
    transpose       - 矩阵转置
    add             - 张量加法
    mixed_precision - 混合精度矩阵乘法 (FP16通信)
    sparse_aware    - 稀疏感知矩阵乘法
    pipeline_matmul - 流水线矩阵乘法 (CUDA双流)
    conv2d          - 2D卷积
    conv3d          - 3D卷积
    fft             - 1D FFT
    ifft            - 逆FFT
    fft2d           - 2D FFT
    rfft            - 实数FFT
    pencil_fft      - Pencil 2D FFT
    einsum          - Einstein求和
    tensordot       - 张量点积
    sum             - 求和
    mean            - 均值
    max             - 最大值
    min             - 最小值
    kahan_sum       - Kahan补偿求和
    kahan_mean      - Kahan补偿均值
    stencil         - Stencil 2D (Halo Exchange)
    jacobi          - Jacobi 2D (Poisson方程)
    all             - 运行所有算法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.gpu_manager import GPUManager


# ==================== 算法实现 ====================

def test_matmul(mpi, distributor):
    """矩阵乘法测试"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul
    
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
    from distributed_gpu.algorithms.matrix_ops import distributed_batch_matmul
    
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
    from distributed_gpu.algorithms.matrix_ops import distributed_transpose
    
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
    from distributed_gpu.algorithms.matrix_ops import distributed_add
    
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
    from distributed_gpu.algorithms.convolution import distributed_conv2d
    
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
    from distributed_gpu.algorithms.convolution import distributed_conv3d
    
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
    from distributed_gpu.algorithms.fft import distributed_fft
    
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
    from distributed_gpu.algorithms.fft import distributed_ifft
    
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
    from distributed_gpu.algorithms.fft import distributed_fft2d
    
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
    from distributed_gpu.algorithms.fft import distributed_rfft
    
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
    from distributed_gpu.algorithms.einsum import distributed_einsum
    
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
    from distributed_gpu.algorithms.einsum import distributed_tensordot
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"张量点积: [64, 32, 16] · [32, 16, 48], dims=2")
        print(f"{'='*50}")
        a = torch.randn(64, 32, 16).cuda()
        b = torch.randn(32, 16, 48).cuda()
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
    from distributed_gpu.algorithms.reduction import distributed_sum
    
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
    from distributed_gpu.algorithms.reduction import distributed_mean
    
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
    from distributed_gpu.algorithms.reduction import distributed_max
    
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
    from distributed_gpu.algorithms.reduction import distributed_min
    
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


# ==================== 创新算子测试 ====================

def test_mixed_precision(mpi, distributor):
    """混合精度通信矩阵乘法 (FP16通信 + FP32计算)"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul_mixed_precision
    
    M, K, N = 500, 300, 400
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"混合精度矩阵乘法: [{M}, {K}] @ [{K}, {N}] (FP16通信)")
        print(f"{'='*50}")
        A = torch.randn(M, K).cuda()
        B = torch.randn(K, N).cuda()
    else:
        A, B = None, None
    
    start = time.time()
    C = distributed_matmul_mixed_precision(A, B, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {C.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        C_expected = torch.matmul(A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        C_norm = torch.max(torch.abs(C_expected)).item()
        rel_error = error / max(C_norm, 1e-10)
        print(f"最大绝对误差: {error:.2e}")
        print(f"最大相对误差: {rel_error:.2e}")
        print(f"状态: {'✓ 通过' if rel_error < 0.05 else '✗ 失败'}")


def test_sparse_aware(mpi, distributor):
    """稀疏感知自适应矩阵乘法"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul_sparse_aware
    
    M, K, N = 500, 300, 400
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"稀疏感知矩阵乘法: [{M}, {K}] @ [{K}, {N}] (70%稀疏)")
        print(f"{'='*50}")
        A = torch.randn(M, K).cuda()
        mask = (torch.rand(M, K, device=A.device) < 0.3).float()
        A = A * mask
        B = torch.randn(K, N).cuda()
    else:
        A, B = None, None
    
    start = time.time()
    C = distributed_matmul_sparse_aware(A, B, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {C.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        C_expected = torch.matmul(A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_pencil_fft(mpi, distributor):
    """Pencil 分解 2D FFT"""
    from distributed_gpu.algorithms.fft import distributed_fft2d_pencil
    
    H, W = 256, 256
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"Pencil 2D FFT: [{H}, {W}]")
        print(f"{'='*50}")
        x = torch.randn(H, W).cuda()
    else:
        x = None
    
    start = time.time()
    y = distributed_fft2d_pencil(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {y.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        y_expected = torch.fft.fft2(x)
        error = torch.max(torch.abs(y - y_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-3 else '✗ 失败'}")


def test_kahan_sum(mpi, distributor):
    """Kahan 补偿求和 (高精度)"""
    from distributed_gpu.algorithms.reduction import distributed_sum, distributed_sum_kahan
    
    shape = (64, 128, 128)
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"Kahan 补偿求和: {shape}")
        print(f"{'='*50}")
        x = torch.randn(shape).cuda()
    else:
        x = None
    
    start = time.time()
    result_kahan = distributed_sum_kahan(x, mpi, distributor)
    elapsed = time.time() - start
    result_naive = distributed_sum(x, mpi, distributor)
    
    if mpi.is_master_process():
        expected = x.to(torch.float64).sum().item()
        error_kahan = abs(result_kahan.item() - expected)
        error_naive = abs(result_naive.item() - expected)
        improvement = error_naive / max(error_kahan, 1e-15)
        print(f"耗时: {elapsed*1000:.2f} ms")
        print(f"Kahan误差: {error_kahan:.6e}")
        print(f"朴素误差:  {error_naive:.6e}")
        print(f"精度提升:  {improvement:.1f}x")
        print(f"状态: {'✓ 通过' if error_kahan <= error_naive * 1.1 else '✗ 失败'}")


def test_stencil(mpi, distributor):
    """Stencil 2D (5-point Laplacian + Halo Exchange)"""
    from distributed_gpu.algorithms.stencil import distributed_stencil_2d
    import torch.nn.functional as F
    
    H, W = 256, 256
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"Stencil 2D (Laplacian): [{H}, {W}]")
        print(f"{'='*50}")
        grid = torch.randn(H, W).cuda()
    else:
        grid = None
    
    start = time.time()
    result = distributed_stencil_2d(grid, mpi, distributor, iterations=1)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {result.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                              device=grid.device)
        expected = F.conv2d(grid.unsqueeze(0).unsqueeze(0),
                            kernel.unsqueeze(0).unsqueeze(0),
                            padding=(1, 1)).squeeze()
        error = torch.max(torch.abs(result - expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_jacobi(mpi, distributor):
    """Jacobi 2D 迭代 (Poisson 方程求解)"""
    from distributed_gpu.algorithms.stencil import distributed_jacobi_2d
    
    N = 128
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"Jacobi 2D: [{N}, {N}], 50 iterations")
        print(f"{'='*50}")
        grid = torch.zeros(N, N).cuda()
        rhs = torch.randn(N, N).cuda() * 0.01
    else:
        grid, rhs = None, None
    
    start = time.time()
    result = distributed_jacobi_2d(grid, rhs, mpi, distributor, iterations=50)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {result.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        print(f"结果范围: [{result.min().item():.4f}, {result.max().item():.4f}]")
        print(f"状态: ✓ 完成")


def test_pipeline_matmul(mpi, distributor):
    """流水线矩阵乘法 (CUDA 双流计算-通信重叠)"""
    from distributed_gpu.pipeline_optimizer import PipelineOptimizer, PipelineConfig
    
    M, K, N = 1000, 500, 800
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"流水线矩阵乘法: [{M}, {K}] @ [{K}, {N}], 4 chunks")
        print(f"{'='*50}")
        A = torch.randn(M, K).cuda()
        B = torch.randn(K, N).cuda()
    else:
        A, B = None, None
    
    pipeline = PipelineOptimizer(mpi, PipelineConfig(num_chunks=4))
    
    start = time.time()
    C = pipeline.pipelined_matmul(A, B, num_chunks=4)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        print(f"结果形状: {C.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        C_expected = torch.matmul(A, B)
        error = torch.max(torch.abs(C - C_expected)).item()
        print(f"最大误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-4 else '✗ 失败'}")


def test_kahan_mean(mpi, distributor):
    """Kahan 补偿均值 (高精度)"""
    from distributed_gpu.algorithms.reduction import distributed_mean_kahan
    
    shape = (64, 128, 128)
    
    if mpi.is_master_process():
        print(f"\n{'='*50}")
        print(f"Kahan 补偿均值: {shape}")
        print(f"{'='*50}")
        x = torch.randn(shape).cuda()
    else:
        x = None
    
    start = time.time()
    result = distributed_mean_kahan(x, mpi, distributor)
    elapsed = time.time() - start
    
    if mpi.is_master_process():
        expected = x.to(torch.float64).mean().item()
        error = abs(result.item() - expected)
        print(f"结果: {result.item():.8f}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        print(f"误差: {error:.2e}")
        print(f"状态: {'✓ 通过' if error < 1e-5 else '✗ 失败'}")


# ==================== 算法映射 ====================

ALGORITHMS = {
    # 基础矩阵运算
    'matmul': ('矩阵乘法', test_matmul),
    'batch_matmul': ('批量矩阵乘法', test_batch_matmul),
    'transpose': ('矩阵转置', test_transpose),
    'add': ('张量加法', test_add),
    # 创新矩阵运算
    'mixed_precision': ('混合精度矩阵乘法', test_mixed_precision),
    'sparse_aware': ('稀疏感知矩阵乘法', test_sparse_aware),
    'pipeline_matmul': ('流水线矩阵乘法', test_pipeline_matmul),
    # 卷积
    'conv2d': ('2D卷积', test_conv2d),
    'conv3d': ('3D卷积', test_conv3d),
    # FFT
    'fft': ('1D FFT', test_fft),
    'ifft': ('逆FFT', test_ifft),
    'fft2d': ('2D FFT', test_fft2d),
    'rfft': ('实数FFT', test_rfft),
    'pencil_fft': ('Pencil 2D FFT', test_pencil_fft),
    # Einstein 求和
    'einsum': ('Einstein求和', test_einsum),
    'tensordot': ('张量点积', test_tensordot),
    # 归约
    'sum': ('求和', test_sum),
    'mean': ('均值', test_mean),
    'max': ('最大值', test_max),
    'min': ('最小值', test_min),
    'kahan_sum': ('Kahan补偿求和', test_kahan_sum),
    'kahan_mean': ('Kahan补偿均值', test_kahan_mean),
    # Stencil / PDE
    'stencil': ('Stencil 2D', test_stencil),
    'jacobi': ('Jacobi 2D', test_jacobi),
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
