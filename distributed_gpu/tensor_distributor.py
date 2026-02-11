"""
张量分配器

负责将大规模张量智能切割并分配到各个GPU节点。
支持 1D（行/列）和 2D 块分割。
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

    # ==================== 2D 块分割支持 ====================
    
    def distribute_2d(self, A: Optional[torch.Tensor], B: Optional[torch.Tensor],
                      grid_rows: int, grid_cols: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        2D 块分割分发：将 A 按行分成 grid_rows 块，B 按列分成 grid_cols 块。
        进程 rank 映射到网格 (i, j)，接收 A 的第 i 行块和 B 的第 j 列块。
        
        重要：所有进程都必须调用此函数！
        
        Args:
            A: 矩阵 A [M, K]（仅主进程提供）
            B: 矩阵 B [K, N]（仅主进程提供）
            grid_rows: 网格行数
            grid_cols: 网格列数（grid_rows * grid_cols == num_nodes）
        
        Returns:
            (A_local, B_local)：当前进程的 A 行块和 B 列块
        """
        assert grid_rows * grid_cols == self.num_nodes, \
            f"网格大小 {grid_rows}x{grid_cols} != 进程数 {self.num_nodes}"
        
        # 当前进程在网格中的坐标
        row_idx = self.rank // grid_cols
        col_idx = self.rank % grid_cols
        
        # 主进程准备数据：为每个 rank 生成 (A_block, B_block)
        if self.mpi.is_master_process():
            M, K = A.shape
            N = B.shape[1]
            
            # 按行切 A
            A_row_chunks = self._split_even(A, grid_rows, dim=0)
            # 按列切 B
            B_col_chunks = self._split_even(B, grid_cols, dim=1)
            
            # 为每个 rank 生成数据对
            a_list = []
            b_list = []
            for r in range(self.num_nodes):
                ri = r // grid_cols
                ci = r % grid_cols
                a_list.append(A_row_chunks[ri].cpu().numpy())
                b_list.append(B_col_chunks[ci].cpu().numpy())
        else:
            a_list = None
            b_list = None
        
        # 广播元信息（dtype）
        if self.mpi.is_master_process():
            meta = {'a_dtype': A.dtype, 'b_dtype': B.dtype}
        else:
            meta = None
        meta = self.mpi.comm.bcast(meta, root=0)
        
        # scatter A 块和 B 块
        a_local_np = self.mpi.comm.scatter(a_list, root=0)
        b_local_np = self.mpi.comm.scatter(b_list, root=0)
        
        A_local = torch.from_numpy(a_local_np.copy())
        B_local = torch.from_numpy(b_local_np.copy())
        if torch.cuda.is_available():
            A_local = A_local.cuda(self.gpu_id)
            B_local = B_local.cuda(self.gpu_id)
        
        return A_local, B_local
    
    def gather_2d(self, C_local: torch.Tensor,
                  grid_rows: int, grid_cols: int,
                  M: int, N: int) -> Optional[torch.Tensor]:
        """
        2D 块收集：将各进程的 C_ij 子块重组为完整矩阵 C [M, N]。
        
        重要：所有进程都必须调用此函数！
        
        Args:
            C_local: 当前进程的结果块 [local_M, local_N]
            grid_rows: 网格行数
            grid_cols: 网格列数
            M: 完整矩阵行数
            N: 完整矩阵列数
        
        Returns:
            完整矩阵 C [M, N]（仅主进程返回）
        """
        local_data = C_local.cpu().numpy()
        gathered = self.mpi.comm.gather(local_data, root=0)
        
        if self.mpi.is_master_process():
            # 重组 2D 网格：gathered[rank] 对应网格 (rank//grid_cols, rank%grid_cols)
            row_blocks = []
            for ri in range(grid_rows):
                col_blocks = []
                for ci in range(grid_cols):
                    rank_idx = ri * grid_cols + ci
                    col_blocks.append(gathered[rank_idx])
                # 沿列方向拼接同一行的块
                row_block = np.concatenate(col_blocks, axis=1)
                row_blocks.append(row_block)
            # 沿行方向拼接所有行
            C_full = np.concatenate(row_blocks, axis=0)
            
            result = torch.from_numpy(C_full)
            if torch.cuda.is_available():
                result = result.cuda(self.gpu_id)
            return result
        return None
    
    def _split_even(self, tensor: torch.Tensor, num_splits: int, dim: int) -> List[torch.Tensor]:
        """沿指定维度均匀分割张量"""
        total = tensor.shape[dim]
        chunk_size = total // num_splits
        remainder = total % num_splits
        
        chunks = []
        start = 0
        for i in range(num_splits):
            size = chunk_size + (1 if i < remainder else 0)
            chunks.append(tensor.narrow(dim, start, size))
            start += size
        return chunks
    
    # ==================== 混合精度通信压缩 ====================
    
    def distribute_compressed(self, tensor: Optional[torch.Tensor], dim: int = 0,
                              comm_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        混合精度分发：传输时压缩为低精度（默认 FP16），接收后恢复原精度。
        通信量减少约 50%（FP32→FP16）。
        
        使用 pickle-based MPI scatter（兼容 FP16 等非标准 MPI 数据类型）。
        
        重要：所有进程都必须调用此函数！
        
        Args:
            tensor: 输入张量（仅主进程提供）
            dim: 分割维度
            comm_dtype: 通信时使用的数据类型（默认 torch.float16）
        
        Returns:
            当前进程的本地张量（已恢复为原始精度）
        """
        if self.mpi.is_master_process():
            orig_dtype = tensor.dtype
            tensor_c = tensor.to(comm_dtype)
            # 手动切分为 numpy 数组列表
            total = tensor_c.shape[dim]
            chunk_size = total // self.num_nodes
            remainder = total % self.num_nodes
            chunks = []
            start = 0
            for i in range(self.num_nodes):
                size = chunk_size + (1 if i < remainder else 0)
                chunks.append(tensor_c.narrow(dim, start, size).cpu().numpy())
                start += size
        else:
            orig_dtype = None
            chunks = None
        
        orig_dtype = self.mpi.broadcast(orig_dtype)
        local_np = self.mpi.comm.scatter(chunks, root=0)
        
        local_tensor = torch.from_numpy(local_np.copy())
        if torch.cuda.is_available():
            local_tensor = local_tensor.cuda(self.gpu_id)
        return local_tensor.to(orig_dtype)
    
    def gather_compressed(self, local_tensor: torch.Tensor, dim: int = 0,
                          comm_dtype: torch.dtype = torch.float16) -> Optional[torch.Tensor]:
        """
        混合精度收集：传输时压缩为低精度，收集后恢复原精度。
        
        重要：所有进程都必须调用此函数！
        """
        orig_dtype = local_tensor.dtype
        local_np = local_tensor.to(comm_dtype).cpu().numpy()
        gathered = self.mpi.comm.gather(local_np, root=0)
        
        if self.mpi.is_master_process():
            import numpy as np_mod
            result_np = np_mod.concatenate(gathered, axis=dim)
            result = torch.from_numpy(result_np.copy())
            if torch.cuda.is_available():
                result = result.cuda(self.gpu_id)
            return result.to(orig_dtype)
        return None
    
    def broadcast_compressed(self, tensor: Optional[torch.Tensor],
                             comm_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        混合精度广播：传输时压缩为低精度，接收后恢复原精度。
        
        重要：所有进程都必须调用此函数！
        """
        if self.mpi.is_master_process():
            orig_dtype = tensor.dtype
            data = tensor.to(comm_dtype).cpu().numpy()
        else:
            orig_dtype = None
            data = None
        
        orig_dtype = self.mpi.broadcast(orig_dtype)
        data = self.mpi.comm.bcast(data, root=0)
        
        local_tensor = torch.from_numpy(data.copy())
        if torch.cuda.is_available():
            local_tensor = local_tensor.cuda(self.gpu_id)
        return local_tensor.to(orig_dtype)