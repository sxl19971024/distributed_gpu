"""
分布式快速傅里叶变换（FFT）

实现大规模FFT的分布式计算。
包括按 batch 分割的标准 FFT 和基于 Pencil 分解的 2D FFT。
"""

import torch
import torch.fft
import numpy as np
from typing import Tuple, Optional

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu


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
    if should_use_single_gpu(mpi, input):
        if mpi.is_master_process():
            return torch.fft.fft(input, n=n, dim=dim, norm=norm)
        return None
    
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
    if should_use_single_gpu(mpi, input):
        if mpi.is_master_process():
            return torch.fft.ifft(input, n=n, dim=dim, norm=norm)
        return None
    
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
    if should_use_single_gpu(mpi, input):
        if mpi.is_master_process():
            return torch.fft.fft2(input, s=s, dim=dim, norm=norm)
        return None
    
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
    if should_use_single_gpu(mpi, input):
        if mpi.is_master_process():
            return torch.fft.rfft(input, n=n, dim=dim, norm=norm)
        return None
    
    input_local = distributor.distribute(input, dim=0)
    output_local = torch.fft.rfft(input_local, n=n, dim=dim, norm=norm)
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output


# ==================== Pencil 分解 2D FFT ====================

def _alltoall_rows_to_cols(local_data: torch.Tensor, mpi: MPIManager) -> torch.Tensor:
    """
    All-to-All 转置：行分布 → 列分布。
    输入：各进程持有 [local_H, W]
    输出：各进程持有 [H, local_W]
    """
    P = mpi.get_size()
    W = local_data.shape[-1]
    local_W = W // P

    send_list = []
    for j in range(P):
        chunk = local_data[:, j * local_W:(j + 1) * local_W].contiguous().cpu().numpy()
        send_list.append(chunk)

    recv_list = mpi.comm.alltoall(send_list)

    # recv_list[i] 来自进程 i，shape [local_H, local_W]
    # 沿行拼接 → [H, local_W]
    result_np = np.concatenate(recv_list, axis=0)

    result = torch.from_numpy(result_np.copy())
    if torch.cuda.is_available():
        result = result.cuda(mpi.get_gpu_id())
    return result


def _alltoall_cols_to_rows(local_data: torch.Tensor, mpi: MPIManager) -> torch.Tensor:
    """
    All-to-All 转置：列分布 → 行分布。
    输入：各进程持有 [H, local_W]
    输出：各进程持有 [local_H, W]
    """
    P = mpi.get_size()
    H = local_data.shape[-2]
    local_H = H // P

    send_list = []
    for i in range(P):
        chunk = local_data[i * local_H:(i + 1) * local_H, :].contiguous().cpu().numpy()
        send_list.append(chunk)

    recv_list = mpi.comm.alltoall(send_list)

    # recv_list[j] 来自进程 j，shape [local_H, local_W]
    # 沿列拼接 → [local_H, W]
    result_np = np.concatenate(recv_list, axis=1)

    result = torch.from_numpy(result_np.copy())
    if torch.cuda.is_available():
        result = result.cuda(mpi.get_gpu_id())
    return result


def distributed_fft2d_pencil(input: Optional[torch.Tensor],
                              mpi: MPIManager,
                              distributor: TensorDistributor,
                              norm: str = "backward") -> Optional[torch.Tensor]:
    """
    基于 Pencil 分解的分布式 2D FFT
    
    与按 batch 分割的 distributed_fft2d 不同，本函数沿变换维度分割，
    能够处理单张超大 2D 网格（如物理模拟中的大规模场数据）。
    
    要求 H 和 W 都能被进程数 P 整除。
    
    算法步骤：
    1. 按行分发 → 每个进程持有 [local_H, W]
    2. 沿 W 方向做 1D FFT
    3. All-to-All 转置 → [H, local_W]
    4. 沿 H 方向做 1D FFT
    5. All-to-All 转置回 → [local_H, W]
    6. 收集结果
    
    重要：所有进程都必须调用此函数！
    仅支持 2D 输入 [H, W]。
    
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

    # Step 1: 按行分发
    local_data = distributor.distribute(input, dim=0)  # [local_H, W]

    # Step 2: 沿 W 方向 1D FFT
    local_fft_w = torch.fft.fft(local_data, dim=-1, norm=norm)

    # Step 3: All-to-All 转置 → [H, local_W]
    transposed = _alltoall_rows_to_cols(local_fft_w, mpi)

    # Step 4: 沿 H 方向 1D FFT
    local_fft_h = torch.fft.fft(transposed, dim=-2, norm=norm)

    # Step 5: All-to-All 转置回 → [local_H, W]
    local_result = _alltoall_cols_to_rows(local_fft_h, mpi)

    # Step 6: 收集
    result = distributor.gather(local_result, dim=0)

    mpi.synchronize()
    return result
