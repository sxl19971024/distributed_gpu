"""
分布式快速傅里叶变换（FFT）

实现大规模 FFT 的分布式计算。
- 标准 FFT（1D/2D/IFFT/RFFT）：按 batch 维度分割，各进程独立计算
- Pencil 分解 2D FFT：沿变换维度分割，处理单张超大 2D 网格

Pencil 分解特别适用于物理模拟中的大规模场数据（如湍流、等离子体），
其中数据本身就是一张巨大的 2D/3D 网格，无 batch 维度可供分割。
"""

from __future__ import annotations

import torch
import torch.fft
import numpy as np
from typing import Callable, Optional, Tuple

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu


# ==================== 通用 batch 分割 FFT ====================


def _distributed_fft_generic(
    fft_fn: Callable[..., torch.Tensor],
    input: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    **fft_kwargs,
) -> Optional[torch.Tensor]:
    """
    通用的 batch 分割 FFT 模板。

    所有标准 FFT 变体（fft/ifft/fft2/rfft）共享相同的
    distribute → local_compute → gather 流程，仅 fft_fn 不同。

    Args:
        fft_fn: 具体的 torch.fft.* 函数
        input: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        **fft_kwargs: 传递给 fft_fn 的额外参数（n, dim, s, norm 等）
    """
    if should_use_single_gpu(mpi, input):
        if mpi.is_master_process():
            return fft_fn(input, **fft_kwargs)
        return None

    input_local: torch.Tensor = distributor.distribute(input, dim=0)
    output_local: torch.Tensor = fft_fn(input_local, **fft_kwargs)
    return distributor.gather(output_local, dim=0)


# ==================== 公开 API ====================


def distributed_fft(
    input: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: str = "backward",
) -> Optional[torch.Tensor]:
    """
    分布式 1D FFT

    策略：按 batch 维度（dim=0）分割，每个进程独立计算 FFT。

    重要：所有进程都必须调用此函数！

    Args:
        input: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        n: FFT 长度
        dim: FFT 维度
        norm: 归一化模式

    Returns:
        FFT 结果（仅主进程返回）
    """
    return _distributed_fft_generic(
        torch.fft.fft, input, mpi, distributor, n=n, dim=dim, norm=norm
    )


def distributed_ifft(
    input: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: str = "backward",
) -> Optional[torch.Tensor]:
    """
    分布式 1D 逆 FFT

    重要：所有进程都必须调用此函数！

    Args:
        input: 输入张量（频域）（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        n: FFT 长度
        dim: FFT 维度
        norm: 归一化模式

    Returns:
        逆 FFT 结果（时域）（仅主进程返回）
    """
    return _distributed_fft_generic(
        torch.fft.ifft, input, mpi, distributor, n=n, dim=dim, norm=norm
    )


def distributed_fft2d(
    input: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    s: Optional[Tuple[int, int]] = None,
    dim: Tuple[int, int] = (-2, -1),
    norm: str = "backward",
) -> Optional[torch.Tensor]:
    """
    分布式 2D FFT

    重要：所有进程都必须调用此函数！

    Args:
        input: 输入张量 [..., H, W]（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        s: FFT 大小
        dim: FFT 维度
        norm: 归一化模式

    Returns:
        2D FFT 结果（仅主进程返回）
    """
    return _distributed_fft_generic(
        torch.fft.fft2, input, mpi, distributor, s=s, dim=dim, norm=norm
    )


def distributed_rfft(
    input: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: str = "backward",
) -> Optional[torch.Tensor]:
    """
    分布式实数 FFT（仅计算正频率，效率更高）

    重要：所有进程都必须调用此函数！

    Args:
        input: 输入实数张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        n: FFT 长度
        dim: FFT 维度
        norm: 归一化模式

    Returns:
        实数 FFT 结果（仅主进程返回）
    """
    return _distributed_fft_generic(
        torch.fft.rfft, input, mpi, distributor, n=n, dim=dim, norm=norm
    )


# ==================== Pencil 分解 2D FFT ====================


def _alltoall_rows_to_cols(
    local_data: torch.Tensor, mpi: MPIManager
) -> torch.Tensor:
    """
    All-to-All 转置：行分布 → 列分布。

    输入：各进程持有 [local_H, W]
    输出：各进程持有 [H, local_W]

    性能优化：
    - 确保发送切片是连续的（避免 _to_pinned_numpy 中的隐式拷贝）
    - 使用预分配的输出缓冲区代替 np.concatenate
    """
    P: int = mpi.get_size()
    local_H: int = local_data.shape[-2]
    W: int = local_data.shape[-1]
    local_W: int = W // P

    # 将数据转为 numpy 一次，然后切片（避免 P 次 GPU→CPU 传输）
    local_np = mpi._to_pinned_numpy(local_data)
    send_list = [
        np.ascontiguousarray(local_np[:, j * local_W : (j + 1) * local_W])
        for j in range(P)
    ]
    recv_list = mpi._safe_call("alltoall(r2c)", mpi.comm.alltoall, send_list)

    # 预分配输出缓冲区，避免 np.concatenate 的临时分配
    H: int = local_H * P
    result_np = np.empty((H, local_W), dtype=local_np.dtype)
    for i, chunk in enumerate(recv_list):
        result_np[i * local_H : (i + 1) * local_H, :] = chunk

    return mpi._from_numpy_to_gpu(result_np)


def _alltoall_cols_to_rows(
    local_data: torch.Tensor, mpi: MPIManager
) -> torch.Tensor:
    """
    All-to-All 转置：列分布 → 行分布。

    输入：各进程持有 [H, local_W]
    输出：各进程持有 [local_H, W]

    性能优化：
    - 一次 GPU→CPU 传输后在 CPU 侧切片
    - 使用预分配的输出缓冲区代替 np.concatenate
    """
    P: int = mpi.get_size()
    H: int = local_data.shape[-2]
    local_H: int = H // P
    local_W: int = local_data.shape[-1]

    # 将数据转为 numpy 一次，然后切片
    local_np = mpi._to_pinned_numpy(local_data)
    send_list = [
        np.ascontiguousarray(local_np[i * local_H : (i + 1) * local_H, :])
        for i in range(P)
    ]
    recv_list = mpi._safe_call("alltoall(c2r)", mpi.comm.alltoall, send_list)

    # 预分配输出缓冲区，避免 np.concatenate 的临时分配
    W: int = local_W * P
    result_np = np.empty((local_H, W), dtype=local_np.dtype)
    for j, chunk in enumerate(recv_list):
        result_np[:, j * local_W : (j + 1) * local_W] = chunk

    return mpi._from_numpy_to_gpu(result_np)


def distributed_fft2d_pencil(
    input: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    norm: str = "backward",
) -> Optional[torch.Tensor]:
    """
    基于 Pencil 分解的分布式 2D FFT

    与 distributed_fft2d 按 batch 分割不同，本函数沿变换维度分割，
    能够处理单张超大 2D 网格（如物理模拟中的大规模场数据）。

    要求 H 和 W 都能被进程数 P 整除。

    算法步骤：
    1. 按行分发 → 每个进程持有 [local_H, W]
    2. 沿 W 方向做 1D FFT
    3. All-to-All 转置 → [H, local_W]
    4. 沿 H 方向做 1D FFT
    5. All-to-All 转置回 → [local_H, W]
    6. 收集结果

    重要：所有进程都必须调用此函数！仅支持 2D 输入 [H, W]。

    Args:
        input: 2D 张量 [H, W]（仅主进程提供）
        mpi: MPI管理器
        distributor: 张量分配器
        norm: FFT 归一化模式

    Returns:
        2D FFT 结果 [H, W]（仅主进程返回）
    """
    if should_use_single_gpu(mpi, input):
        if mpi.is_master_process():
            return torch.fft.fft2(input, norm=norm)
        return None

    # Step 1: 按行分发 → [local_H, W]
    local_data: torch.Tensor = distributor.distribute(input, dim=0)

    # Step 2: 沿 W 方向 1D FFT
    local_fft_w: torch.Tensor = torch.fft.fft(local_data, dim=-1, norm=norm)

    # Step 3: All-to-All 转置 → [H, local_W]
    transposed: torch.Tensor = _alltoall_rows_to_cols(local_fft_w, mpi)

    # Step 4: 沿 H 方向 1D FFT
    local_fft_h: torch.Tensor = torch.fft.fft(transposed, dim=-2, norm=norm)

    # Step 5: All-to-All 转置回 → [local_H, W]
    local_result: torch.Tensor = _alltoall_cols_to_rows(local_fft_h, mpi)

    # Step 6: 收集
    return distributor.gather(local_result, dim=0)
