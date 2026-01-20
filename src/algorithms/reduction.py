"""
分布式归约操作

实现分布式求和、均值、最大值、最小值等归约操作。
"""

import torch
from typing import Optional, List, Union
from mpi4py import MPI

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor


def distributed_sum(tensor: Optional[torch.Tensor],
                    mpi: MPIManager,
                    distributor: TensorDistributor,
                    dim: Optional[int] = None,
                    keepdim: bool = False) -> Optional[torch.Tensor]:
    """
    分布式求和
    
    重要：所有进程都必须调用此函数！
    
    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求和维度（None表示全部求和）
        keepdim: 是否保持维度
    
    Returns:
        求和结果（仅主进程返回）
    """
    mpi.synchronize()
    
    # 分发
    tensor_local = distributor.distribute(tensor, dim=0)
    
    # 本地求和
    if dim is None:
        local_sum = tensor_local.sum()
        # 全局归约
        total_sum = mpi.allreduce_tensor(local_sum.unsqueeze(0))
        if mpi.is_master_process():
            return total_sum.squeeze()
        return None
    elif dim == 0:
        # 沿分割维度求和需要全局归约
        local_sum = tensor_local.sum(dim=0, keepdim=keepdim)
        total_sum = mpi.allreduce_tensor(local_sum)
        if mpi.is_master_process():
            return total_sum
        return None
    else:
        # 其他维度可以本地计算后收集
        local_sum = tensor_local.sum(dim=dim, keepdim=keepdim)
        result = distributor.gather(local_sum, dim=0)
        return result


def distributed_mean(tensor: Optional[torch.Tensor],
                     mpi: MPIManager,
                     distributor: TensorDistributor,
                     dim: Optional[int] = None,
                     keepdim: bool = False) -> Optional[torch.Tensor]:
    """
    分布式均值
    
    重要：所有进程都必须调用此函数！
    
    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求均值维度（None表示全部求均值）
        keepdim: 是否保持维度
    
    Returns:
        均值结果（仅主进程返回）
    """
    mpi.synchronize()
    
    tensor_local = distributor.distribute(tensor, dim=0)
    
    if dim is None:
        # 全局均值
        local_sum = tensor_local.sum()
        local_count = tensor_local.numel()
        
        total_sum = mpi.allreduce_tensor(local_sum.unsqueeze(0))
        total_count = mpi.allreduce(local_count)
        
        if mpi.is_master_process():
            return total_sum.squeeze() / total_count
        return None
    elif dim == 0:
        # 沿分割维度求均值
        local_sum = tensor_local.sum(dim=0, keepdim=keepdim)
        local_count = tensor_local.shape[0]
        
        total_sum = mpi.allreduce_tensor(local_sum)
        total_count = mpi.allreduce(local_count)
        
        if mpi.is_master_process():
            return total_sum / total_count
        return None
    else:
        # 其他维度本地计算
        local_mean = tensor_local.mean(dim=dim, keepdim=keepdim)
        result = distributor.gather(local_mean, dim=0)
        return result


def distributed_max(tensor: Optional[torch.Tensor],
                    mpi: MPIManager,
                    distributor: TensorDistributor,
                    dim: Optional[int] = None,
                    keepdim: bool = False) -> Optional[torch.Tensor]:
    """
    分布式最大值
    
    重要：所有进程都必须调用此函数！
    
    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求最大值维度（None表示全局最大值）
        keepdim: 是否保持维度
    
    Returns:
        最大值结果（仅主进程返回）
    """
    mpi.synchronize()
    
    tensor_local = distributor.distribute(tensor, dim=0)
    
    if dim is None:
        # 全局最大值
        local_max = tensor_local.max()
        # 使用MPI_MAX操作
        global_max_np = mpi.allreduce(local_max.cpu().numpy(), op=MPI.MAX)
        
        if mpi.is_master_process():
            return torch.tensor(global_max_np, device=tensor_local.device)
        return None
    elif dim == 0:
        # 沿分割维度求最大值
        local_max = tensor_local.max(dim=0, keepdim=keepdim).values
        # 全局最大值
        result_np = mpi.allreduce(local_max.cpu().numpy(), op=MPI.MAX)
        
        if mpi.is_master_process():
            result = torch.from_numpy(result_np)
            if torch.cuda.is_available():
                result = result.cuda(mpi.get_gpu_id())
            return result
        return None
    else:
        # 其他维度本地计算
        local_max = tensor_local.max(dim=dim, keepdim=keepdim).values
        result = distributor.gather(local_max, dim=0)
        return result


def distributed_min(tensor: Optional[torch.Tensor],
                    mpi: MPIManager,
                    distributor: TensorDistributor,
                    dim: Optional[int] = None,
                    keepdim: bool = False) -> Optional[torch.Tensor]:
    """
    分布式最小值
    
    重要：所有进程都必须调用此函数！
    
    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求最小值维度（None表示全局最小值）
        keepdim: 是否保持维度
    
    Returns:
        最小值结果（仅主进程返回）
    """
    mpi.synchronize()
    
    tensor_local = distributor.distribute(tensor, dim=0)
    
    if dim is None:
        local_min = tensor_local.min()
        global_min_np = mpi.allreduce(local_min.cpu().numpy(), op=MPI.MIN)
        
        if mpi.is_master_process():
            return torch.tensor(global_min_np, device=tensor_local.device)
        return None
    elif dim == 0:
        local_min = tensor_local.min(dim=0, keepdim=keepdim).values
        result_np = mpi.allreduce(local_min.cpu().numpy(), op=MPI.MIN)
        
        if mpi.is_master_process():
            result = torch.from_numpy(result_np)
            if torch.cuda.is_available():
                result = result.cuda(mpi.get_gpu_id())
            return result
        return None
    else:
        local_min = tensor_local.min(dim=dim, keepdim=keepdim).values
        result = distributor.gather(local_min, dim=0)
        return result
