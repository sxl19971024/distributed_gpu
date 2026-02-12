"""
张量分配器 (Tensor Distributor)

负责将大规模张量智能切割并分配到各个 GPU 节点。
支持 1D（行/列）和 2D 块分割，以及混合精度压缩通信。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .mpi_manager import MPIManager, _split_to_numpy_chunks


class TensorDistributor:
    """
    张量分配器

    基于 :class:`MPIManager` 提供高层张量分发 / 收集 / 广播操作，
    支持 1D 分割、2D 块分割和混合精度压缩通信。
    """

    def __init__(self, mpi_manager: MPIManager) -> None:
        self.mpi: MPIManager = mpi_manager
        self.num_nodes: int = mpi_manager.get_size()
        self.rank: int = mpi_manager.get_rank()
        self.gpu_id: int = mpi_manager.get_gpu_id()

    # ══════════════════════════════════════════════════════════════
    #  1D 分发 / 收集 / 广播
    # ══════════════════════════════════════════════════════════════

    def distribute(self, tensor: Optional[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """
        将张量沿 *dim* 分配到所有进程（scatter）。

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
        从各个进程收集张量并拼接（仅主进程得到结果）。

        重要：所有进程都必须调用此函数！
        """
        return self.mpi.gather_tensor(local_tensor, dim=dim, root=0)

    def allgather(self, local_tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        从所有进程收集张量并广播到所有进程。

        重要：所有进程都必须调用此函数！
        """
        return self.mpi.allgather_tensor(local_tensor, dim=dim)

    def broadcast(self, tensor: Optional[torch.Tensor], root: int = 0) -> torch.Tensor:
        """
        广播张量到所有进程。

        重要：所有进程都必须调用此函数！
        """
        return self.mpi.broadcast_tensor(tensor, root=root)

    def distribute_with_broadcast(
        self,
        tensor_to_split: Optional[torch.Tensor],
        tensor_to_broadcast: Optional[torch.Tensor],
        split_dim: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时执行分割和广播操作（常用于矩阵乘法等场景）。

        重要：所有进程都必须调用此函数！

        Returns:
            (分割后的本地张量, 广播后的张量)
        """
        local_split = self.distribute(tensor_to_split, dim=split_dim)
        broadcasted = self.broadcast(tensor_to_broadcast)
        return local_split, broadcasted

    def reduce_sum(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """
        对所有进程的张量求和（allreduce），所有进程都得到结果。

        重要：所有进程都必须调用此函数！
        """
        return self.mpi.allreduce_tensor(local_tensor)

    # ── 分割计算辅助 ────────────────────────────────────────────

    def calculate_split_sizes(self, total_size: int) -> List[int]:
        """计算每个进程的分割大小（余数均匀分配到前几个进程）。"""
        chunk_size = total_size // self.num_nodes
        remainder = total_size % self.num_nodes
        return [chunk_size + (1 if i < remainder else 0) for i in range(self.num_nodes)]

    def get_local_range(self, total_size: int) -> Tuple[int, int]:
        """获取当前进程负责的数据范围 ``[start, end)``。"""
        sizes = self.calculate_split_sizes(total_size)
        start = sum(sizes[: self.rank])
        end = start + sizes[self.rank]
        return start, end

    # ══════════════════════════════════════════════════════════════
    #  2D 块分割支持
    # ══════════════════════════════════════════════════════════════

    def distribute_2d(
        self,
        A: Optional[torch.Tensor],
        B: Optional[torch.Tensor],
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        2D 块分割分发。

        将 A 按行分成 *grid_rows* 块，B 按列分成 *grid_cols* 块。
        进程 rank 映射到网格 ``(i, j)``，接收 A 的第 i 行块和 B 的第 j 列块。

        重要：所有进程都必须调用此函数！

        Args:
            A: 矩阵 A ``[M, K]``（仅主进程提供）
            B: 矩阵 B ``[K, N]``（仅主进程提供）
            grid_rows: 网格行数
            grid_cols: 网格列数（``grid_rows * grid_cols == num_nodes``）

        Returns:
            ``(A_local, B_local)``：当前进程的 A 行块和 B 列块
        """
        assert grid_rows * grid_cols == self.num_nodes, (
            f"网格大小 {grid_rows}×{grid_cols} != 进程数 {self.num_nodes}"
        )

        # 主进程准备数据：为每个 rank 生成 (A_block, B_block)
        if self.mpi.is_master_process():
            assert A is not None and B is not None
            A_row_chunks = _split_even(A, grid_rows, dim=0)
            B_col_chunks = _split_even(B, grid_cols, dim=1)

            a_list: List[np.ndarray] = []
            b_list: List[np.ndarray] = []
            for r in range(self.num_nodes):
                ri, ci = divmod(r, grid_cols)
                a_list.append(self.mpi._to_pinned_numpy(A_row_chunks[ri]))
                b_list.append(self.mpi._to_pinned_numpy(B_col_chunks[ci]))
            meta = {"a_dtype": A.dtype, "b_dtype": B.dtype}
        else:
            a_list = None  # type: ignore[assignment]
            b_list = None  # type: ignore[assignment]
            meta = None

        meta = self.mpi._safe_call("bcast(2d_meta)", self.mpi.comm.bcast, meta, root=0)

        a_local_np: np.ndarray = self.mpi._safe_call("scatter(2d_A)", self.mpi.comm.scatter, a_list, root=0)
        b_local_np: np.ndarray = self.mpi._safe_call("scatter(2d_B)", self.mpi.comm.scatter, b_list, root=0)

        return (
            self.mpi._from_numpy_to_gpu(a_local_np),
            self.mpi._from_numpy_to_gpu(b_local_np),
        )

    def gather_2d(
        self,
        C_local: torch.Tensor,
        grid_rows: int,
        grid_cols: int,
        M: int,
        N: int,
    ) -> Optional[torch.Tensor]:
        """
        2D 块收集：将各进程的 ``C_ij`` 子块重组为完整矩阵 ``C [M, N]``。

        性能优化：使用预分配的输出缓冲区，避免多次 np.concatenate 的临时内存分配。

        重要：所有进程都必须调用此函数！
        """
        local_data = self.mpi._to_pinned_numpy(C_local)
        gathered: List[np.ndarray] = self.mpi._safe_call(
            "gather(2d_C)", self.mpi.comm.gather, local_data, root=0,
        )

        if self.mpi.is_master_process():
            # 预分配完整输出矩阵，直接写入子块（避免多次 concatenate）
            C_full = np.empty((M, N), dtype=local_data.dtype)
            row_offset = 0
            for ri in range(grid_rows):
                block_rows = gathered[ri * grid_cols].shape[0]
                col_offset = 0
                for ci in range(grid_cols):
                    block = gathered[ri * grid_cols + ci]
                    block_cols = block.shape[1]
                    C_full[row_offset:row_offset + block_rows,
                           col_offset:col_offset + block_cols] = block
                    col_offset += block_cols
                row_offset += block_rows
            return self.mpi._from_numpy_to_gpu(C_full)
        return None

    # ══════════════════════════════════════════════════════════════
    #  混合精度压缩通信
    # ══════════════════════════════════════════════════════════════

    def distribute_compressed(
        self,
        tensor: Optional[torch.Tensor],
        dim: int = 0,
        comm_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        混合精度分发：传输时压缩为低精度（默认 FP16），接收后恢复原精度。

        通信量减少约 50%（FP32 → FP16）。

        重要：所有进程都必须调用此函数！
        """
        if self.mpi.is_master_process():
            assert tensor is not None
            orig_dtype = tensor.dtype
            chunks = _split_to_numpy_chunks(
                tensor.to(comm_dtype), self.num_nodes, dim, self.mpi._to_pinned_numpy,
            )
        else:
            orig_dtype = None
            chunks = None

        orig_dtype = self.mpi.broadcast(orig_dtype)
        local_np: np.ndarray = self.mpi._safe_call(
            "scatter(compressed)", self.mpi.comm.scatter, chunks, root=0,
        )
        return self.mpi._from_numpy_to_gpu(local_np).to(orig_dtype)

    def gather_compressed(
        self,
        local_tensor: torch.Tensor,
        dim: int = 0,
        comm_dtype: torch.dtype = torch.float16,
    ) -> Optional[torch.Tensor]:
        """
        混合精度收集：传输时压缩为低精度，收集后恢复原精度。

        重要：所有进程都必须调用此函数！
        """
        orig_dtype = local_tensor.dtype
        local_np = self.mpi._to_pinned_numpy(local_tensor.to(comm_dtype))
        gathered: List[np.ndarray] = self.mpi._safe_call(
            "gather(compressed)", self.mpi.comm.gather, local_np, root=0,
        )

        if self.mpi.is_master_process():
            result_np = np.concatenate(gathered, axis=dim)
            return self.mpi._from_numpy_to_gpu(result_np).to(orig_dtype)
        return None

    def broadcast_compressed(
        self,
        tensor: Optional[torch.Tensor],
        comm_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        混合精度广播：传输时压缩为低精度，接收后恢复原精度。

        重要：所有进程都必须调用此函数！
        """
        if self.mpi.is_master_process():
            assert tensor is not None
            orig_dtype = tensor.dtype
            data: Optional[np.ndarray] = self.mpi._to_pinned_numpy(tensor.to(comm_dtype))
        else:
            orig_dtype = None
            data = None

        orig_dtype = self.mpi.broadcast(orig_dtype)
        data = self.mpi._safe_call("bcast(compressed)", self.mpi.comm.bcast, data, root=0)
        return self.mpi._from_numpy_to_gpu(data).to(orig_dtype)


# ══════════════════════════════════════════════════════════════════
#  模块级工具函数
# ══════════════════════════════════════════════════════════════════

def _split_even(tensor: torch.Tensor, num_splits: int, dim: int) -> List[torch.Tensor]:
    """
    沿指定维度均匀分割张量（确保连续内存布局）。

    性能优化：仅在 narrow 产生非连续视图时才调用 contiguous()。
    当 dim=0 且张量本身连续时，narrow 结果通常也是连续的。
    """
    total = tensor.shape[dim]
    chunk_size = total // num_splits
    remainder = total % num_splits

    chunks: List[torch.Tensor] = []
    start = 0
    for i in range(num_splits):
        size = chunk_size + (1 if i < remainder else 0)
        chunk = tensor.narrow(dim, start, size)
        # 仅在必要时调用 contiguous()（dim=0 时 narrow 通常已连续）
        if not chunk.is_contiguous():
            chunk = chunk.contiguous()
        chunks.append(chunk)
        start += size
    return chunks
