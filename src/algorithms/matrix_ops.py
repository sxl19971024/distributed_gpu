"""
分布式矩阵运算

实现大规模矩阵的分布式计算，包括矩阵乘法、批量矩阵乘法等。
"""

import torch
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor


def distributed_matmul(A: Optional[torch.Tensor],
                       B: Optional[torch.Tensor],
                       mpi: MPIManager,
                       distributor: TensorDistributor) -> Optional[torch.Tensor]:
    """
    分布式矩阵乘法 C = A @ B
    
    策略：按行分割A，广播B，收集结果C
    
    重要：所有进程都必须调用此函数！
    
    Args:
        A: 矩阵A [M, K]（仅主进程需要提供有效数据）
        B: 矩阵B [K, N]（仅主进程需要提供有效数据）
        mpi: MPI管理器
        distributor: 张量分配器
    
    Returns:
        结果矩阵C [M, N]（仅主进程返回有效结果，其他进程返回None）
    """
    mpi.synchronize()
    
    # 1. 分发A（按行分割）和广播B
    A_local, B_local = distributor.distribute_with_broadcast(A, B, split_dim=0)
    
    # 2. 本地计算
    C_local = torch.matmul(A_local, B_local)
    
    # 3. 收集结果
    C = distributor.gather(C_local, dim=0)
    
    mpi.synchronize()
    return C


def distributed_batch_matmul(A: Optional[torch.Tensor],
                             B: Optional[torch.Tensor],
                             mpi: MPIManager,
                             distributor: TensorDistributor) -> Optional[torch.Tensor]:
    """
    分布式批量矩阵乘法 C = A @ B
    
    输入形状：A [batch, M, K], B [batch, K, N] 或 [K, N]
    
    策略：按batch维度同时分割A和B
    
    重要：所有进程都必须调用此函数！
    
    Args:
        A: 批量矩阵A [batch, M, K]（仅主进程需要提供）
        B: 批量矩阵B [batch, K, N] 或 [K, N]（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
    
    Returns:
        结果矩阵C [batch, M, N]（仅主进程返回有效结果）
    """
    mpi.synchronize()
    
    # 检查B的维度并相应处理
    # 先广播B的维度信息
    if mpi.is_master_process():
        b_dim = B.dim()
    else:
        b_dim = None
    b_dim = mpi.broadcast(b_dim)
    
    # 分发A
    A_local = distributor.distribute(A, dim=0)
    
    if b_dim == 3:
        # B也有batch维度，需要同时分割
        B_local = distributor.distribute(B, dim=0)
        C_local = torch.bmm(A_local, B_local)
    else:
        # B是2D，广播到所有进程
        B_local = distributor.broadcast(B)
        C_local = torch.matmul(A_local, B_local)
    
    # 收集结果
    C = distributor.gather(C_local, dim=0)
    
    mpi.synchronize()
    return C


def distributed_transpose(A: Optional[torch.Tensor],
                          mpi: MPIManager,
                          distributor: TensorDistributor,
                          dim0: int = 0,
                          dim1: int = 1) -> Optional[torch.Tensor]:
    """
    分布式矩阵转置
    
    重要：所有进程都必须调用此函数！
    
    Args:
        A: 输入矩阵（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim0, dim1: 要交换的维度
    
    Returns:
        转置后的矩阵（仅主进程返回）
    """
    mpi.synchronize()
    
    # 分发
    A_local = distributor.distribute(A, dim=0)
    
    # 本地转置
    A_T_local = torch.transpose(A_local, dim0, dim1)
    
    # 收集
    A_T = distributor.gather(A_T_local, dim=0)
    
    mpi.synchronize()
    return A_T


def distributed_add(A: Optional[torch.Tensor],
                    B: Optional[torch.Tensor],
                    mpi: MPIManager,
                    distributor: TensorDistributor) -> Optional[torch.Tensor]:
    """
    分布式张量加法 C = A + B
    
    重要：所有进程都必须调用此函数！
    """
    mpi.synchronize()
    
    A_local = distributor.distribute(A, dim=0)
    B_local = distributor.distribute(B, dim=0)
    
    C_local = A_local + B_local
    
    C = distributor.gather(C_local, dim=0)
    
    mpi.synchronize()
    return C
