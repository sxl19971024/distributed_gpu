#!/usr/bin/env python
"""
分布式矩阵乘法示例

演示如何使用框架进行大规模矩阵乘法计算。

运行方式：
    mpirun -n 4 python examples/matrix_multiplication.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.mpi_manager import MPIManager
from src.tensor_distributor import TensorDistributor
from src.gpu_manager import GPUManager
from src.cost_model import CostModel, ClusterConfig
from src.utils.profiler import Profiler
from src.algorithms.matrix_ops import distributed_matmul


def main():
    # 初始化MPI环境
    mpi = MPIManager()
    distributor = TensorDistributor(mpi)
    gpu_manager = GPUManager(mpi.get_gpu_id())
    profiler = Profiler(enabled=mpi.is_master_process())
    
    # 打印GPU信息
    gpu_manager.print_info()
    
    # 设置矩阵大小
    M, K, N = 5000, 5000, 5000
    
    if mpi.is_master_process():
        print(f"\n开始分布式矩阵乘法: [{M}, {K}] @ [{K}, {N}] = [{M}, {N}]")
        print(f"使用 {mpi.get_size()} 个GPU节点\n")
        
        # 代价模型分析
        config = ClusterConfig.from_auto_detect(mpi.get_size())
        cost_model = CostModel(config)
        cost_model.print_analysis(M, K, N)
        
        # 创建矩阵
        A = torch.randn(M, K, dtype=torch.float32).cuda()
        B = torch.randn(K, N, dtype=torch.float32).cuda()
    else:
        A, B = None, None
    
    mpi.synchronize()
    
    # 执行分布式矩阵乘法
    profiler.start("distributed_matmul")
    C = distributed_matmul(A, B, mpi, distributor)
    profiler.end("distributed_matmul")
    
    # 打印结果
    if mpi.is_master_process():
        print(f"\n计算完成！结果形状: {C.shape}")
        gpu_manager.print_memory_info()
        
        # 验证结果（可选）
        if M <= 2000:
            print("\n验证结果正确性...")
            C_expected = torch.matmul(A, B)
            max_error = torch.max(torch.abs(C - C_expected)).item()
            print(f"最大误差: {max_error:.6e}")
            
            if max_error < 1e-4:
                print("✓ 结果正确！")
            else:
                print("✗ 结果可能有误差")
    
    profiler.print_summary()
    mpi.synchronize()


if __name__ == "__main__":
    main()
