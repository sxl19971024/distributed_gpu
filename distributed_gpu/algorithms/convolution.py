"""
分布式卷积运算

实现大规模卷积操作的分布式计算（2D / 3D）。
策略：按 batch 维度（dim=0）分割输入，广播卷积核到所有进程。

该策略的优势：
- 卷积核通常远小于输入，广播开销低
- batch 维度分割无需 halo 交换，通信简单
- 各进程完全独立计算，无依赖
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu

# 卷积参数的联合类型
_Stride2D = Tuple[int, int]
_Stride3D = Tuple[int, int, int]


def _distributed_conv_generic(
    conv_fn: Callable[..., torch.Tensor],
    input: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    bias: Optional[torch.Tensor],
    stride: Union[_Stride2D, _Stride3D],
    padding: Union[_Stride2D, _Stride3D],
    dilation: Union[_Stride2D, _Stride3D],
    groups: int,
) -> Optional[torch.Tensor]:
    """
    通用分布式卷积模板（2D/3D 共用）。

    流程：
    1. 按 batch 维度 scatter 输入
    2. broadcast 卷积核（+ bias）
    3. 各进程独立计算
    4. gather 结果

    Args:
        conv_fn: F.conv2d 或 F.conv3d
        input: 输入张量 [N, C_in, ...]（仅主进程）
        weight: 卷积核 [C_out, C_in, ...]（仅主进程）
        bias: 偏置 [C_out]（仅主进程，可为 None）
        其余参数同 torch.nn.functional.conv2d / conv3d
    """
    # 单卡快速路径
    if should_use_single_gpu(mpi, input, weight):
        if mpi.is_master_process():
            return conv_fn(
                input, weight, bias,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
            )
        return None

    # 1. 分发输入（按 batch 维度）+ 广播权重
    input_local: torch.Tensor = distributor.distribute(input, dim=0)
    weight_local: torch.Tensor = distributor.broadcast(weight)

    # 所有进程统一判断是否有 bias（避免分支不一致导致 MPI 死锁）
    has_bias: bool = mpi.broadcast(bias is not None if mpi.is_master_process() else None)
    bias_local: Optional[torch.Tensor] = distributor.broadcast(bias) if has_bias else None

    # 2. 本地卷积计算
    output_local: torch.Tensor = conv_fn(
        input_local, weight_local, bias_local,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )

    # 性能优化：显式释放不再需要的中间张量，减少峰值显存
    del input_local, weight_local, bias_local

    # 3. 收集结果
    return distributor.gather(output_local, dim=0)


# ==================== 公开 API ====================


def distributed_conv2d(
    input: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    bias: Optional[torch.Tensor] = None,
    stride: _Stride2D = (1, 1),
    padding: _Stride2D = (0, 0),
    dilation: _Stride2D = (1, 1),
    groups: int = 1,
) -> Optional[torch.Tensor]:
    """
    分布式 2D 卷积

    策略：按 batch 维度分割输入，广播权重。

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
    return _distributed_conv_generic(
        F.conv2d, input, weight, mpi, distributor,
        bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups,
    )


def distributed_conv3d(
    input: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    bias: Optional[torch.Tensor] = None,
    stride: _Stride3D = (1, 1, 1),
    padding: _Stride3D = (0, 0, 0),
    dilation: _Stride3D = (1, 1, 1),
    groups: int = 1,
) -> Optional[torch.Tensor]:
    """
    分布式 3D 卷积

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
    return _distributed_conv_generic(
        F.conv3d, input, weight, mpi, distributor,
        bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
