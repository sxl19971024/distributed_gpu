"""
分布式快速傅里叶变换（FFT）

实现大规模FFT的分布式计算。
"""

import torch
import torch.fft
from typing import Tuple, Optional

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor


def distributed_fft(input: Optional[torch.Tensor],
                    mpi: MPIManager,
                    distributor: TensorDistributor,
                    n: Optional[int] = None,
                    dim: int = -1,
                    norm: str = "backward") -> Optional[torch.Tensor]:
    """
    分布式1D FFT
    
    策略：按batch维度分割，每个进程独立计算FFT
    
    重要：所有进程都必须调用此函数！
    
    Args:
        input: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        n: FFT长度
        dim: FFT维度
        norm: 归一化模式
    
    Returns:
        FFT结果（仅主进程返回）
    """
    mpi.synchronize()
    
    # 分发
    input_local = distributor.distribute(input, dim=0)
    
    # 本地FFT
    output_local = torch.fft.fft(input_local, n=n, dim=dim, norm=norm)
    
    # 收集
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output


def distributed_ifft(input: Optional[torch.Tensor],
                     mpi: MPIManager,
                     distributor: TensorDistributor,
                     n: Optional[int] = None,
                     dim: int = -1,
                     norm: str = "backward") -> Optional[torch.Tensor]:
    """
    分布式1D 逆FFT
    
    重要：所有进程都必须调用此函数！
    
    Args:
        input: 输入张量（频域）（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        n: FFT长度
        dim: FFT维度
        norm: 归一化模式
    
    Returns:
        逆FFT结果（时域）（仅主进程返回）
    """
    mpi.synchronize()
    
    input_local = distributor.distribute(input, dim=0)
    output_local = torch.fft.ifft(input_local, n=n, dim=dim, norm=norm)
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output


def distributed_fft2d(input: Optional[torch.Tensor],
                      mpi: MPIManager,
                      distributor: TensorDistributor,
                      s: Optional[Tuple[int, int]] = None,
                      dim: Tuple[int, int] = (-2, -1),
                      norm: str = "backward") -> Optional[torch.Tensor]:
    """
    分布式2D FFT
    
    重要：所有进程都必须调用此函数！
    
    Args:
        input: 输入张量 [..., H, W]（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        s: FFT大小
        dim: FFT维度
        norm: 归一化模式
    
    Returns:
        2D FFT结果（仅主进程返回）
    """
    mpi.synchronize()
    
    input_local = distributor.distribute(input, dim=0)
    output_local = torch.fft.fft2(input_local, s=s, dim=dim, norm=norm)
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output


def distributed_rfft(input: Optional[torch.Tensor],
                     mpi: MPIManager,
                     distributor: TensorDistributor,
                     n: Optional[int] = None,
                     dim: int = -1,
                     norm: str = "backward") -> Optional[torch.Tensor]:
    """
    分布式实数FFT（更高效，只计算正频率）
    
    重要：所有进程都必须调用此函数！
    
    Args:
        input: 输入实数张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        n: FFT长度
        dim: FFT维度
        norm: 归一化模式
    
    Returns:
        实数FFT结果（仅主进程返回）
    """
    mpi.synchronize()
    
    input_local = distributor.distribute(input, dim=0)
    output_local = torch.fft.rfft(input_local, n=n, dim=dim, norm=norm)
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output
