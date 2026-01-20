"""
张量分配器

负责将大规模张量智能切割并分配到各个GPU节点。
使用MPI集合操作确保通信正确性。
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from .mpi_manager import MPIManager


class TensorDistributor:
    """张量分配器"""
    
    def __init__(self, mpi_manager: MPIManager):
        """
        初始化张量分配器
        
        Args:
            mpi_manager: MPI管理器实例
        """
        self.mpi = mpi_manager
        self.num_nodes = mpi_manager.get_size()
        self.rank = mpi_manager.get_rank()
        self.gpu_id = mpi_manager.get_gpu_id()
    
    def distribute(self, tensor: Optional[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """
        将张量分配到所有进程（使用scatter）
        
        重要：所有进程都必须调用此函数！
        
        Args:
            tensor: 输入张量（仅主进程需要提供有效张量）
            dim: 分割维度
        
        Returns:
            当前进程分配到的张量块
        """
        return self.mpi.scatter_tensor(tensor, dim=dim, root=0)
    
    def gather(self, local_tensor: torch.Tensor, dim: int = 0) -> Optional[torch.Tensor]:
        """
        从各个进程收集张量并拼接（仅主进程得到结果）
        
        重要：所有进程都必须调用此函数！
        
        Args:
            local_tensor: 当前进程的本地张量
            dim: 拼接维度
        
        Returns:
            拼接后的完整张量（仅主进程返回有效数据）
        """
        return self.mpi.gather_tensor(local_tensor, dim=dim, root=0)
    
    def allgather(self, local_tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        从所有进程收集张量并广播到所有进程
        
        重要：所有进程都必须调用此函数！
        
        Args:
            local_tensor: 当前进程的本地张量
            dim: 拼接维度
        
        Returns:
            拼接后的完整张量（所有进程都得到）
        """
        return self.mpi.allgather_tensor(local_tensor, dim=dim)
    
    def broadcast(self, tensor: Optional[torch.Tensor], root: int = 0) -> torch.Tensor:
        """
        广播张量到所有进程
        
        重要：所有进程都必须调用此函数！
        
        Args:
            tensor: 要广播的张量（仅root进程需要提供有效张量）
            root: 源进程rank
        
        Returns:
            广播后的张量
        """
        return self.mpi.broadcast_tensor(tensor, root=root)
    
    def distribute_with_broadcast(self, tensor_to_split: Optional[torch.Tensor],
                                   tensor_to_broadcast: Optional[torch.Tensor],
                                   split_dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时执行分割和广播操作（常用于矩阵乘法等场景）
        
        重要：所有进程都必须调用此函数！
        
        Args:
            tensor_to_split: 要分割的张量（仅主进程需要提供）
            tensor_to_broadcast: 要广播的张量（仅主进程需要提供）
            split_dim: 分割维度
        
        Returns:
            (分割后的本地张量, 广播后的张量)
        """
        local_split = self.distribute(tensor_to_split, dim=split_dim)
        broadcasted = self.broadcast(tensor_to_broadcast)
        return local_split, broadcasted
    
    def reduce_sum(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """
        对所有进程的张量求和，所有进程都得到结果
        
        重要：所有进程都必须调用此函数！
        """
        return self.mpi.allreduce_tensor(local_tensor)
    
    def calculate_split_sizes(self, total_size: int) -> List[int]:
        """
        计算每个进程的分割大小
        
        Args:
            total_size: 总大小
        
        Returns:
            每个进程的大小列表
        """
        chunk_size = total_size // self.num_nodes
        remainder = total_size % self.num_nodes
        
        sizes = []
        for i in range(self.num_nodes):
            size = chunk_size + (1 if i < remainder else 0)
            sizes.append(size)
        
        return sizes
    
    def get_local_range(self, total_size: int) -> Tuple[int, int]:
        """
        获取当前进程负责的数据范围
        
        Args:
            total_size: 总大小
        
        Returns:
            (start_idx, end_idx)
        """
        sizes = self.calculate_split_sizes(total_size)
        start = sum(sizes[:self.rank])
        end = start + sizes[self.rank]
        return start, end
