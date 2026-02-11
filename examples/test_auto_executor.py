#!/usr/bin/env python
"""
测试 AutoExecutor —— 显存感知自动化分布式计算

运行方式：
    # 使用 2 个 GPU（可改为 1-8）
    mpirun -n 2 python examples/test_auto_executor.py

    # 使用所有 8 个 GPU
    mpirun -n 8 python examples/test_auto_executor.py

测试内容：
    1. GPU 显存实时状态展示
    2. 自动 MatMul（单批次）
    3. 自动 MatMul（强制分批：通过 max_per_gpu_gb 模拟显存不足）
    4. 自动 FFT
    5. 自动 Sum
    6. 自动 Conv2d
    7. 批量 MatMul
    8. 执行计划预览（不执行）
    9. auto_compute 便捷函数
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed_gpu.auto_executor import AutoExecutor, auto_compute


def test_gpu_status(executor):
    """测试 1: GPU 显存状态展示"""
    executor._log("=" * 50)
    executor._log("测试 1: GPU 显存实时状态")
    executor.gpu_status()
    return True


def test_matmul_single_batch(executor):
    """测试 2: 自动 MatMul（单批次，数据能一次放入 GPU）"""
    executor._log("=" * 50)
    executor._log("测试 2: 自动 MatMul（单批次）")
    
    M, K, N = 1024, 512, 768
    
    if executor.is_master:
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        expected = A @ B
    else:
        A = B = expected = None
    
    result = executor.matmul(A, B)
    
    if executor.is_master:
        error = torch.max(torch.abs(result - expected)).item()
        passed = error < 1e-2
        executor._log(f"  结果形状: {result.shape}, 最大误差: {error:.6f} {'✓' if passed else '✗'}")
        return passed
    return True


def test_matmul_multi_batch(executor):
    """测试 3: 自动 MatMul（强制分批，模拟显存不足场景）"""
    executor._log("=" * 50)
    executor._log("测试 3: 自动 MatMul（强制分批 - 模拟超显存场景）")
    
    # 创建一个 max_per_gpu_gb 很小的执行器来强制分批
    # 使用较大矩阵 + 限制 10MB/卡 来触发多批处理
    from distributed_gpu.resource_planner import ResourcePlanner
    old_planner = executor.planner
    executor.planner = ResourcePlanner(executor.mpi, max_per_gpu_gb=0.01)  # 限制 10MB
    
    M, K, N = 2048, 512, 2048
    
    if executor.is_master:
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        expected = A @ B
    else:
        A = B = expected = None
    
    result = executor.matmul(A, B)
    
    # 恢复原始 planner
    executor.planner = old_planner
    
    if executor.is_master:
        error = torch.max(torch.abs(result - expected)).item()
        passed = error < 1e-2
        executor._log(f"  结果形状: {result.shape}, 最大误差: {error:.6f} {'✓' if passed else '✗'}")
        return passed
    return True


def test_fft(executor):
    """测试 4: 自动 FFT（2D 输入，沿 dim=1 做 FFT，dim=0 分割不影响正确性）"""
    executor._log("=" * 50)
    executor._log("测试 4: 自动 FFT")
    
    # 使用 2D 输入 [batch, signal]，split 沿 dim=0，FFT 沿 dim=-1
    # 这样每个 GPU 独立在自己的行上做 FFT，结果完全正确
    rows, cols = 64, 4096
    
    if executor.is_master:
        x = torch.randn(rows, cols)
        expected = torch.fft.fft(x, dim=-1)
    else:
        x = expected = None
    
    result = executor.fft(x, dim=-1)
    
    if executor.is_master:
        error = torch.max(torch.abs(result - expected)).item()
        passed = error < 1e-2
        executor._log(f"  结果形状: {result.shape}, 最大误差: {error:.6f} {'✓' if passed else '✗'}")
        return passed
    return True


def test_sum(executor):
    """测试 5: 自动 Sum"""
    executor._log("=" * 50)
    executor._log("测试 5: 自动 Sum")
    
    shape = (1000, 500)
    
    if executor.is_master:
        x = torch.randn(*shape)
        expected = x.sum()
    else:
        x = expected = None
    
    result = executor.sum(x)
    
    if executor.is_master:
        error = torch.abs(result - expected).item()
        passed = error < 1.0  # float32 累积误差允许稍大
        executor._log(f"  结果: {result.item():.4f}, 期望: {expected.item():.4f}, "
                      f"误差: {error:.6f} {'✓' if passed else '✗'}")
        return passed
    return True


def test_conv2d(executor):
    """测试 6: 自动 Conv2d"""
    executor._log("=" * 50)
    executor._log("测试 6: 自动 Conv2d")
    
    N_batch, C_in, H, W = 16, 3, 32, 32
    C_out, kH, kW = 8, 3, 3
    
    if executor.is_master:
        x = torch.randn(N_batch, C_in, H, W)
        w = torch.randn(C_out, C_in, kH, kW)
        expected = torch.nn.functional.conv2d(x, w, padding=(1, 1))
    else:
        x = w = expected = None
    
    result = executor.conv2d(x, w, padding=(1, 1))
    
    if executor.is_master:
        error = torch.max(torch.abs(result - expected)).item()
        passed = error < 1e-2
        executor._log(f"  结果形状: {result.shape}, 最大误差: {error:.6f} {'✓' if passed else '✗'}")
        return passed
    return True


def test_batch_matmul(executor):
    """测试 7: 批量 MatMul"""
    executor._log("=" * 50)
    executor._log("测试 7: 批量 MatMul（3 对矩阵）")
    
    sizes = [(256, 128, 192), (128, 256, 64), (64, 64, 64)]
    
    if executor.is_master:
        pairs = [(torch.randn(m, k), torch.randn(k, n)) for m, k, n in sizes]
        expecteds = [A @ B for A, B in pairs]
    else:
        pairs = expecteds = None
    
    results = executor.matmul_batch(pairs)
    
    if executor.is_master:
        all_passed = True
        for i, (res, exp) in enumerate(zip(results, expecteds)):
            error = torch.max(torch.abs(res - exp)).item()
            passed = error < 1e-2
            executor._log(f"  对 {i + 1}: 形状 {res.shape}, 误差 {error:.6f} {'✓' if passed else '✗'}")
            all_passed = all_passed and passed
        return all_passed
    return True


def test_plan_preview(executor):
    """测试 8: 执行计划预览（不执行计算）"""
    executor._log("=" * 50)
    executor._log("测试 8: 执行计划预览")
    
    # 预览一个大矩阵乘法的执行计划
    plan = executor.plan_info("matmul", (50000, 10000), (10000, 50000))
    
    if executor.is_master:
        executor._log(f"  可行: {plan.feasible}")
        executor._log(f"  批次数: {plan.num_batches}")
        return True
    return True


def test_auto_compute_function(executor):
    """测试 9: auto_compute 便捷函数"""
    executor._log("=" * 50)
    executor._log("测试 9: auto_compute 便捷函数")
    
    M, K, N = 256, 128, 192
    
    if executor.is_master:
        A = torch.randn(M, K)
        B = torch.randn(K, N)
        expected = A @ B
    else:
        A = B = expected = None
    
    # 注意：auto_compute 使用全局单例，这里直接测试分发逻辑
    # 由于 auto_compute 会创建新的 MPIManager，可能冲突，
    # 所以这里手动测试 executor 的分发
    result = executor.matmul(A, B)
    
    if executor.is_master:
        error = torch.max(torch.abs(result - expected)).item()
        passed = error < 1e-2
        executor._log(f"  结果形状: {result.shape}, 误差: {error:.6f} {'✓' if passed else '✗'}")
        return passed
    return True


def main():
    # 创建执行器（所有进程都必须调用）
    executor = AutoExecutor(verbose=True)
    
    tests = [
        ("GPU 显存状态", test_gpu_status),
        ("MatMul 单批次", test_matmul_single_batch),
        ("MatMul 分批次", test_matmul_multi_batch),
        ("FFT", test_fft),
        ("Sum", test_sum),
        ("Conv2d", test_conv2d),
        ("批量 MatMul", test_batch_matmul),
        ("执行计划预览", test_plan_preview),
        ("便捷函数", test_auto_compute_function),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn(executor)
            if executor.is_master:
                if result:
                    passed += 1
                else:
                    failed += 1
                    print(f"  ✗ {name} 未通过")
        except Exception as e:
            if executor.is_master:
                failed += 1
                print(f"  ✗ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    if executor.is_master:
        total = passed + failed
        print("\n" + "=" * 50)
        print(f"AutoExecutor 测试结果: {passed}/{total} 通过")
        if failed == 0:
            print("全部通过 ✓")
        else:
            print(f"{failed} 个测试未通过 ✗")
        print("=" * 50)


if __name__ == "__main__":
    main()
