"""
分布式矩阵运算

实现大规模矩阵的分布式计算，包括矩阵乘法、批量矩阵乘法等。
支持三种分割策略：行分割、列分割、2D 块分割（SUMMA），
通过代价模型自适应选择最优策略。

创新算子：
- 混合精度通信矩阵乘法：FP16 通信 + FP32 计算，通信量减半
- 稀疏感知自适应矩阵乘法：自动检测稀疏度切换 COO/Dense 路径
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ..cost_model import (
    CostModel,
    SplitStrategy,
    SplitPlan,
    ClusterConfig,
    compute_2d_grid,
)
from ._utils import should_use_single_gpu


# ==================== 内部实现：三种分割策略 ====================


def _matmul_row_split(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
) -> Optional[torch.Tensor]:
    """行分割矩阵乘法：scatter A (dim=0), broadcast B, gather C (dim=0)"""
    A_local, B_local = distributor.distribute_with_broadcast(A, B, split_dim=0)
    C_local: torch.Tensor = torch.matmul(A_local, B_local)
    return distributor.gather(C_local, dim=0)


def _matmul_col_split(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
) -> Optional[torch.Tensor]:
    """列分割矩阵乘法：broadcast A, scatter B (dim=1), gather C (dim=1)"""
    A_local: torch.Tensor = distributor.broadcast(A)
    B_local: torch.Tensor = distributor.distribute(B, dim=1)
    C_local: torch.Tensor = torch.matmul(A_local, B_local)
    return distributor.gather(C_local, dim=1)


def _matmul_block_2d(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    grid_rows: int,
    grid_cols: int,
) -> Optional[torch.Tensor]:
    """
    2D 块分割矩阵乘法（SUMMA 风格）

    将 p 个进程排列为 grid_rows × grid_cols 的 2D 网格。
    进程 (i, j) 持有 A 的第 i 行块 和 B 的第 j 列块，
    计算 C_ij = A_i @ B_j → 结果的 (i,j) 子块。
    最后重组为完整矩阵 C。
    """
    # 广播 M, N 以便 gather_2d 使用
    if mpi.is_master_process():
        dims: Tuple[int, int] = (A.shape[0], B.shape[1])  # type: ignore[union-attr]
    else:
        dims = (0, 0)
    M, N = mpi.broadcast(dims)

    # 2D 分发 → 本地计算 → 2D 收集
    A_local, B_local = distributor.distribute_2d(A, B, grid_rows, grid_cols)
    C_local: torch.Tensor = torch.matmul(A_local, B_local)
    return distributor.gather_2d(C_local, grid_rows, grid_cols, M, N)


# ==================== 策略选择辅助 ====================


def _resolve_strategy(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    cost_model: Optional[CostModel],
    strategy: Optional[SplitStrategy],
) -> Tuple[SplitStrategy, int, int]:
    """
    决定分割策略并广播到所有进程。

    返回 (strategy, grid_rows, grid_cols)

    性能优化：将 3 次 broadcast 合并为 1 次，减少通信延迟。
    """
    if strategy is not None:
        chosen = strategy
        grid_r, grid_c = compute_2d_grid(mpi.get_size())
    elif cost_model is not None:
        # 广播矩阵维度
        if mpi.is_master_process():
            dims = (A.shape[0], A.shape[1], B.shape[1])  # type: ignore[union-attr]
        else:
            dims = None  # type: ignore[assignment]
        M, K, N = mpi.broadcast(dims)

        plan: SplitPlan = cost_model.find_optimal_strategy(M, K, N)
        chosen = plan.strategy
        grid_r, grid_c = plan.grid_rows, plan.grid_cols

        if mpi.is_master_process():
            suffix = (
                f" ({grid_r}x{grid_c})"
                if plan.strategy == SplitStrategy.BLOCK_2D
                else ""
            )
            mpi.print_master(
                f"[CostModel] 选择策略: {plan.strategy.value}{suffix}"
                f" | 预估效率: {plan.cost.efficiency * 100:.1f}%"
            )
    else:
        chosen = SplitStrategy.ROW_SPLIT
        grid_r, grid_c = mpi.get_size(), 1

    # 将策略+网格参数打包为一次广播，减少 3→1 次通信延迟
    packed = mpi.broadcast((chosen, grid_r, grid_c))
    return packed[0], packed[1], packed[2]


# ==================== 公开 API ====================


def distributed_matmul(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    cost_model: Optional[CostModel] = None,
    strategy: Optional[SplitStrategy] = None,
) -> Optional[torch.Tensor]:
    """
    分布式矩阵乘法 C = A @ B

    支持三种分割策略，通过代价模型自适应选择：
    - ROW_SPLIT:    按 A 的行分割（适合 M >> N）
    - COLUMN_SPLIT: 按 B 的列分割（适合 N >> M）
    - BLOCK_2D:     2D 块分割（适合 M ≈ N，大幅降低每卡显存）

    重要：所有进程都必须调用此函数！

    Args:
        A: 矩阵A [M, K]（仅主进程需要提供有效数据）
        B: 矩阵B [K, N]（仅主进程需要提供有效数据）
        mpi: MPI管理器
        distributor: 张量分配器
        cost_model: 代价模型（None 则使用行分割；提供则自动选择最优策略）
        strategy: 强制使用指定策略（优先级高于 cost_model）

    Returns:
        结果矩阵C [M, N]（仅主进程返回有效结果，其他进程返回None）
    """
    # 单卡快速路径
    if should_use_single_gpu(mpi, A, B):
        if mpi.is_master_process():
            return torch.matmul(A, B)  # type: ignore[arg-type]
        return None

    chosen, grid_r, grid_c = _resolve_strategy(A, B, mpi, cost_model, strategy)

    _dispatch = {
        SplitStrategy.ROW_SPLIT: lambda: _matmul_row_split(A, B, mpi, distributor),
        SplitStrategy.COLUMN_SPLIT: lambda: _matmul_col_split(A, B, mpi, distributor),
        SplitStrategy.BLOCK_2D: lambda: _matmul_block_2d(
            A, B, mpi, distributor, grid_r, grid_c
        ),
    }
    return _dispatch.get(chosen, _dispatch[SplitStrategy.ROW_SPLIT])()


def distributed_batch_matmul(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
) -> Optional[torch.Tensor]:
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
    if should_use_single_gpu(mpi, A, B):
        if mpi.is_master_process():
            return torch.bmm(A, B) if B.dim() == 3 else torch.matmul(A, B)  # type: ignore[union-attr]
        return None

    # 检查B的维度并相应处理
    b_dim: int = mpi.broadcast(B.dim() if mpi.is_master_process() else None)  # type: ignore[union-attr]

    A_local: torch.Tensor = distributor.distribute(A, dim=0)

    if b_dim == 3:
        B_local: torch.Tensor = distributor.distribute(B, dim=0)
        C_local: torch.Tensor = torch.bmm(A_local, B_local)
    else:
        B_local = distributor.broadcast(B)
        C_local = torch.matmul(A_local, B_local)

    return distributor.gather(C_local, dim=0)


def distributed_transpose(
    A: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim0: int = 0,
    dim1: int = 1,
) -> Optional[torch.Tensor]:
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
    if should_use_single_gpu(mpi, A):
        if mpi.is_master_process():
            return torch.transpose(A, dim0, dim1)  # type: ignore[arg-type]
        return None

    split_dim: int = 0
    A_local: torch.Tensor = distributor.distribute(A, dim=split_dim)
    A_T_local: torch.Tensor = torch.transpose(A_local, dim0, dim1)

    # 转置后，原来的 split_dim 位置发生了变化
    if split_dim == dim0:
        gather_dim = dim1
    elif split_dim == dim1:
        gather_dim = dim0
    else:
        gather_dim = split_dim

    return distributor.gather(A_T_local, dim=gather_dim)


def distributed_add(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
) -> Optional[torch.Tensor]:
    """
    分布式张量加法 C = A + B

    重要：所有进程都必须调用此函数！
    """
    if should_use_single_gpu(mpi, A, B):
        if mpi.is_master_process():
            return A + B  # type: ignore[operator]
        return None

    A_local: torch.Tensor = distributor.distribute(A, dim=0)
    B_local: torch.Tensor = distributor.distribute(B, dim=0)
    # 原地加法，避免分配额外的临时张量
    A_local.add_(B_local)
    return distributor.gather(A_local, dim=0)


# ==================== 混合精度通信矩阵乘法 ====================


def distributed_matmul_mixed_precision(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    comm_dtype: torch.dtype = torch.float16,
) -> Optional[torch.Tensor]:
    """
    混合精度分布式矩阵乘法

    通信时使用低精度（默认 FP16），计算时使用原始精度（FP32）。
    通信量减少约 50%，适用于对精度要求不极端的科学计算场景。

    误差上界：‖C_mixed - C_exact‖ ≤ O(ε_comm × √K × ‖A‖ × ‖B‖)
    其中 ε_comm 为通信精度的机器精度（FP16: ~5e-4）。

    重要：所有进程都必须调用此函数！

    Args:
        A: 矩阵A [M, K]（仅主进程）
        B: 矩阵B [K, N]（仅主进程）
        mpi: MPI管理器
        distributor: 张量分配器
        comm_dtype: 通信数据类型（默认 FP16）

    Returns:
        结果矩阵C [M, N]（仅主进程）
    """
    if should_use_single_gpu(mpi, A, B):
        if mpi.is_master_process():
            return torch.matmul(A, B)  # type: ignore[arg-type]
        return None

    # 分发时压缩为 comm_dtype，接收后恢复
    A_local: torch.Tensor = distributor.distribute_compressed(
        A, dim=0, comm_dtype=comm_dtype
    )
    B_local: torch.Tensor = distributor.broadcast_compressed(B, comm_dtype=comm_dtype)

    # 计算时使用完整精度
    C_local: torch.Tensor = torch.matmul(A_local, B_local)

    # 收集结果（全精度，不压缩输出以保留计算精度）
    return distributor.gather(C_local, dim=0)


# ==================== 稀疏感知自适应矩阵乘法 ====================


def _matmul_sparse_broadcast(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
) -> Optional[torch.Tensor]:
    """
    稀疏路径：将 B 以 COO 格式广播（仅传非零元素），节省通信量。

    通信量对比：
    - 稠密广播: K × N × 4 字节 (FP32)
    - COO 广播: nnz × 20 字节 (values + indices)
    当稀疏度 > 80% 时 COO 更高效。

    性能优化：
    - 直接使用 torch.sparse.mm 进行稀疏矩阵乘法（避免 to_dense()）
    - 仅在 CUDA 不支持稀疏乘法时回退到 to_dense()
    """
    # 分发 A（稠密行分割）
    A_local: torch.Tensor = distributor.distribute(A, dim=0)

    # 将 B 以 COO 格式广播
    if mpi.is_master_process():
        B_coo = B.to_sparse_coo().coalesce()  # type: ignore[union-attr]
        sparse_meta = {
            "indices": B_coo.indices().cpu().numpy(),
            "values": B_coo.values().cpu().numpy(),
            "shape": list(B.shape),  # type: ignore[union-attr]
        }
    else:
        sparse_meta = None  # type: ignore[assignment]

    sparse_meta = mpi.broadcast(sparse_meta)

    B_indices = torch.from_numpy(sparse_meta["indices"].copy()).long()
    B_values = torch.from_numpy(sparse_meta["values"].copy())
    B_shape = torch.Size(sparse_meta["shape"])
    B_sparse: torch.Tensor = torch.sparse_coo_tensor(
        B_indices, B_values, B_shape
    ).coalesce()

    gpu_id: int = mpi.get_gpu_id()
    if torch.cuda.is_available():
        B_sparse = B_sparse.cuda(gpu_id)

    # 优先使用稀疏矩阵乘法（避免 to_dense() 的内存开销）
    try:
        C_local: torch.Tensor = torch.sparse.mm(
            A_local.to_sparse_coo(), B_sparse
        ).to_dense()
    except (RuntimeError, NotImplementedError):
        # 回退到稠密乘法（某些 CUDA 版本/dtype 不支持稀疏 mm）
        B_local: torch.Tensor = B_sparse.to_dense()
        C_local = torch.matmul(A_local, B_local)

    return distributor.gather(C_local, dim=0)


def distributed_matmul_sparse_aware(
    A: Optional[torch.Tensor],
    B: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    sparsity_threshold: float = 0.5,
) -> Optional[torch.Tensor]:
    """
    稀疏感知自适应分布式矩阵乘法

    自动检测矩阵稀疏度，根据阈值切换稠密/稀疏路径：
    - 稀疏度 > threshold：以 COO 格式广播 B，节省通信量
    - 稀疏度 ≤ threshold：使用标准稠密行分割

    适用场景：有限元刚度矩阵、图邻接矩阵等科学计算中的稀疏矩阵。

    重要：所有进程都必须调用此函数！

    Args:
        A: 矩阵A [M, K]（仅主进程）
        B: 矩阵B [K, N]（仅主进程）
        mpi: MPI管理器
        distributor: 张量分配器
        sparsity_threshold: 稀疏度阈值（0-1）

    Returns:
        结果矩阵C [M, N]（仅主进程）
    """
    if should_use_single_gpu(mpi, A, B):
        if mpi.is_master_process():
            return torch.matmul(A, B)  # type: ignore[arg-type]
        return None

    # 广播稀疏度检测结果
    if mpi.is_master_process():
        sparsity_A: float = 1.0 - (torch.count_nonzero(A).item() / A.numel())  # type: ignore[union-attr]
        sparsity_B: float = 1.0 - (torch.count_nonzero(B).item() / B.numel())  # type: ignore[union-attr]
        use_sparse: bool = max(sparsity_A, sparsity_B) > sparsity_threshold
        info = {
            "use_sparse": use_sparse,
            "sparsity_A": sparsity_A,
            "sparsity_B": sparsity_B,
        }
    else:
        info = None  # type: ignore[assignment]

    info = mpi.broadcast(info)

    if mpi.is_master_process():
        sp_str = f"A={info['sparsity_A']:.1%}, B={info['sparsity_B']:.1%}"
        path_str = "稀疏COO" if info["use_sparse"] else "稠密"
        mpi.print_master(f"  [稀疏检测] {sp_str} → {path_str}路径")

    if info["use_sparse"]:
        return _matmul_sparse_broadcast(A, B, mpi, distributor)
    return _matmul_row_split(A, B, mpi, distributor)
