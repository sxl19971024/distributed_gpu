"""
分布式卷积运算

实现大规模卷积操作的分布式计算。
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor


def distributed_conv2d(input: Optional[torch.Tensor],
                       weight: Optional[torch.Tensor],
                       mpi: MPIManager,
                       distributor: TensorDistributor,
                       bias: Optional[torch.Tensor] = None,
                       stride: Tuple[int, int] = (1, 1),
                       padding: Tuple[int, int] = (0, 0),
                       dilation: Tuple[int, int] = (1, 1),
                       groups: int = 1) -> Optional[torch.Tensor]:
    """
    分布式2D卷积
    
    策略：按batch维度分割输入，广播权重
    
    重要：所有进程都必须调用此函数！
    
    Args:
        input: 输入张量 [N, C_in, H, W]（仅主进程需要提供）
        weight: 卷积核 [C_out, C_in, kH, kW]（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        bias: 偏置 [C_out]（仅主进程需要提供）
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
    
    Returns:
        输出张量 [N, C_out, H_out, W_out]（仅主进程返回）
    """
    mpi.synchronize()
    
    # 1. 分发输入（按batch维度）和广播权重
    input_local = distributor.distribute(input, dim=0)
    weight_local = distributor.broadcast(weight)
    bias_local = distributor.broadcast(bias) if bias is not None else None
    
    # 2. 本地卷积计算
    output_local = F.conv2d(
        input_local,
        weight_local,
        bias_local,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    
    # 3. 收集结果
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output


def distributed_conv3d(input: Optional[torch.Tensor],
                       weight: Optional[torch.Tensor],
                       mpi: MPIManager,
                       distributor: TensorDistributor,
                       bias: Optional[torch.Tensor] = None,
                       stride: Tuple[int, int, int] = (1, 1, 1),
                       padding: Tuple[int, int, int] = (0, 0, 0),
                       dilation: Tuple[int, int, int] = (1, 1, 1),
                       groups: int = 1) -> Optional[torch.Tensor]:
    """
    分布式3D卷积
    
    重要：所有进程都必须调用此函数！
    
    Args:
        input: 输入张量 [N, C_in, D, H, W]（仅主进程需要提供）
        weight: 卷积核 [C_out, C_in, kD, kH, kW]（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        bias: 偏置 [C_out]
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
    
    Returns:
        输出张量（仅主进程返回）
    """
    mpi.synchronize()
    
    # 分发和广播
    input_local = distributor.distribute(input, dim=0)
    weight_local = distributor.broadcast(weight)
    bias_local = distributor.broadcast(bias) if bias is not None else None
    
    # 本地计算
    output_local = F.conv3d(
        input_local,
        weight_local,
        bias_local,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    
    # 收集
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output
