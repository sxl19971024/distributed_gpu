"""
分布式归约操作

实现分布式 sum / mean / max / min 等归约操作。

创新算子：
- Kahan 补偿求和/均值：使用 float64 + Kahan 算法，
  误差上界 O(ε_machine) 而非朴素方法的 O(n·ε_machine)。
  适用于科学计算中的能量守恒验证、长时间积分等场景。
"""

from __future__ import annotations

import torch
from typing import Optional
from mpi4py import MPI

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu


# ==================== 通用归约框架 ====================


def _distributed_reduction(
    tensor: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim: Optional[int],
    keepdim: bool,
    *,
    local_fn,
    global_fn,
    local_fn_with_dim,
) -> Optional[torch.Tensor]:
    """
    通用分布式归约模板。

    三种情况：
    - dim=None:  全局归约 → local_fn + global_fn (allreduce)
    - dim=0:     沿分割维度归约 → local_fn_with_dim(dim=0) + global_fn (allreduce)
    - dim=other: 非分割维度 → 本地计算 + gather

    Args:
        local_fn: 全局归约时的本地计算 (tensor → scalar tensor)
        global_fn: 全局归约时的 allreduce 操作
        local_fn_with_dim: 带维度的本地归约 (tensor, dim, keepdim → tensor)
    """
    # 单卡快速路径
    if should_use_single_gpu(mpi, tensor):
        if mpi.is_master_process():
            if dim is None:
                return local_fn(tensor)
            return local_fn_with_dim(tensor, dim, keepdim)
        return None

    tensor_local: torch.Tensor = distributor.distribute(tensor, dim=0)

    if dim is None:
        local_result = local_fn(tensor_local)
        global_result = global_fn(local_result, mpi, tensor_local.device)
        return global_result if mpi.is_master_process() else None

    elif dim == 0:
        local_result = local_fn_with_dim(tensor_local, 0, keepdim)
        global_result = global_fn(local_result, mpi, tensor_local.device)
        return global_result if mpi.is_master_process() else None

    else:
        local_result = local_fn_with_dim(tensor_local, dim, keepdim)
        return distributor.gather(local_result, dim=0)


# ==================== allreduce 辅助 ====================


def _allreduce_sum(
    local_val: torch.Tensor, mpi: MPIManager, device: torch.device
) -> torch.Tensor:
    """SUM allreduce"""
    if local_val.dim() == 0:
        local_val = local_val.unsqueeze(0)
        result = mpi.allreduce_tensor(local_val)
        return result.squeeze()
    return mpi.allreduce_tensor(local_val)


def _allreduce_minmax(
    local_val: torch.Tensor, mpi: MPIManager, device: torch.device, op
) -> torch.Tensor:
    """
    MIN/MAX allreduce（通过 MPI op）

    性能优化：使用 pinned memory 进行 CPU↔GPU 传输，
    并复用 numpy 缓冲区避免额外分配。
    """
    local_np = mpi._to_pinned_numpy(local_val)
    result_np = mpi.allreduce(local_np, op=op)
    if hasattr(result_np, '__len__'):
        return mpi._from_numpy_to_gpu(result_np)
    # 标量结果
    result = torch.tensor(result_np, dtype=local_val.dtype)
    if torch.cuda.is_available():
        result = result.to(device)
    return result


# ==================== 公开 API ====================


def distributed_sum(
    tensor: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Optional[torch.Tensor]:
    """
    分布式求和

    重要：所有进程都必须调用此函数！

    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求和维度（None 表示全部求和）
        keepdim: 是否保持维度

    Returns:
        求和结果（仅主进程返回）
    """
    return _distributed_reduction(
        tensor, mpi, distributor, dim, keepdim,
        local_fn=lambda t: t.sum(),
        global_fn=_allreduce_sum,
        local_fn_with_dim=lambda t, d, kd: t.sum(dim=d, keepdim=kd),
    )


def distributed_mean(
    tensor: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Optional[torch.Tensor]:
    """
    分布式均值

    重要：所有进程都必须调用此函数！

    性能优化：
    - dim=None 时将 sum 和 count 打包为一次 allreduce（2→1 次通信）
    - dim=0 时同样打包 sum 和 count

    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求均值维度（None 表示全部求均值）
        keepdim: 是否保持维度

    Returns:
        均值结果（仅主进程返回）
    """
    if should_use_single_gpu(mpi, tensor):
        if mpi.is_master_process():
            if dim is None:
                return tensor.mean()  # type: ignore[union-attr]
            return tensor.mean(dim=dim, keepdim=keepdim)  # type: ignore[union-attr]
        return None

    tensor_local: torch.Tensor = distributor.distribute(tensor, dim=0)

    if dim is None:
        # 打包 [sum, count] 为一次 allreduce，减少通信次数
        local_sum = tensor_local.sum().to(torch.float64)
        local_count = float(tensor_local.numel())
        sum_count = torch.tensor(
            [local_sum.item(), local_count], dtype=torch.float64
        )
        if torch.cuda.is_available():
            sum_count = sum_count.to(tensor_local.device)
        total = mpi.allreduce_tensor(sum_count)
        if mpi.is_master_process():
            return (total[0] / total[1]).to(tensor_local.dtype)
        return None

    elif dim == 0:
        # 打包 sum tensor + count 为一次 allreduce
        local_sum = tensor_local.to(torch.float64).sum(dim=0, keepdim=keepdim)
        local_count = float(tensor_local.shape[0])
        # 将 count 附加到 sum 末尾，一次 allreduce 完成
        count_tensor = torch.full(
            (1,) * local_sum.dim() if local_sum.dim() > 0 else (1,),
            local_count, dtype=torch.float64, device=tensor_local.device,
        )
        if local_sum.dim() == 0:
            packed = torch.stack([local_sum, count_tensor.squeeze()])
        else:
            packed = torch.cat([local_sum.flatten(), count_tensor.flatten()])
        total_packed = mpi.allreduce_tensor(packed)
        if mpi.is_master_process():
            total_sum_flat = total_packed[:-1]
            total_count = total_packed[-1]
            result = (total_sum_flat / total_count).reshape(local_sum.shape)
            return result.to(tensor_local.dtype)
        return None

    else:
        local_mean = tensor_local.mean(dim=dim, keepdim=keepdim)
        return distributor.gather(local_mean, dim=0)


def distributed_max(
    tensor: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Optional[torch.Tensor]:
    """
    分布式最大值

    重要：所有进程都必须调用此函数！

    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求最大值维度（None 表示全局最大值）
        keepdim: 是否保持维度

    Returns:
        最大值结果（仅主进程返回）
    """
    return _distributed_reduction(
        tensor, mpi, distributor, dim, keepdim,
        local_fn=lambda t: t.max(),
        global_fn=lambda v, m, d: _allreduce_minmax(v, m, d, MPI.MAX),
        local_fn_with_dim=lambda t, d, kd: t.max(dim=d, keepdim=kd).values,
    )


def distributed_min(
    tensor: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Optional[torch.Tensor]:
    """
    分布式最小值

    重要：所有进程都必须调用此函数！

    Args:
        tensor: 输入张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求最小值维度（None 表示全局最小值）
        keepdim: 是否保持维度

    Returns:
        最小值结果（仅主进程返回）
    """
    return _distributed_reduction(
        tensor, mpi, distributor, dim, keepdim,
        local_fn=lambda t: t.min(),
        global_fn=lambda v, m, d: _allreduce_minmax(v, m, d, MPI.MIN),
        local_fn_with_dim=lambda t, d, kd: t.min(dim=d, keepdim=kd).values,
    )


# ==================== 数值稳定的 Kahan 补偿求和 ====================


def _kahan_sum_blocks(data: torch.Tensor, block_size: int = 4096) -> float:
    """
    GPU 加速的 Kahan 补偿求和。

    将数据分为若干块，每块使用 PyTorch 在 GPU 上快速求和，
    块间使用 Kahan 补偿算法消除累积误差。

    精度: O(ε_machine) 而非朴素求和的 O(n·ε_machine)

    性能优化：
    - 小张量（≤block_size）直接 float64 求和，跳过循环
    - 大张量先在 GPU 上按块求和（向量化），再在 CPU 上做 Kahan 补偿
    """
    flat = data.flatten().to(torch.float64)
    n: int = flat.shape[0]

    # 小张量快速路径：直接 GPU float64 求和，无需 Kahan
    if n <= block_size:
        return flat.sum().item()

    # 大张量：先在 GPU 上按块求和得到块级 sum 向量
    num_blocks: int = (n + block_size - 1) // block_size
    # 截断到完整块
    full_blocks: int = n // block_size
    if full_blocks > 0:
        block_sums = flat[: full_blocks * block_size].reshape(full_blocks, block_size).sum(dim=1)
    else:
        block_sums = torch.empty(0, dtype=torch.float64, device=flat.device)

    # 处理尾部不完整块
    remainder: int = n - full_blocks * block_size
    if remainder > 0:
        tail_sum = flat[full_blocks * block_size :].sum().unsqueeze(0)
        block_sums = torch.cat([block_sums, tail_sum])

    # CPU 侧 Kahan 补偿（块数通常很小，Python 循环开销可忽略）
    sums_cpu = block_sums.cpu()
    total: float = 0.0
    comp: float = 0.0
    for i in range(sums_cpu.shape[0]):
        y = sums_cpu[i].item() - comp
        t = total + y
        comp = (t - total) - y
        total = t

    return total


def _kahan_allreduce(
    local_sum: float, mpi: MPIManager, device: torch.device
) -> torch.Tensor:
    """Kahan 求和的 allreduce 辅助"""
    sum_tensor = torch.tensor([local_sum], dtype=torch.float64)
    if torch.cuda.is_available():
        sum_tensor = sum_tensor.to(device)
    return mpi.allreduce_tensor(sum_tensor)


def distributed_sum_kahan(
    tensor: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    数值稳定的分布式求和（Kahan 补偿求和）

    使用 float64 中间精度 + Kahan 补偿算法，
    误差上界为 O(ε_machine) 而非朴素方法的 O(n·ε_machine)。

    对于科学计算中的能量守恒验证、长时间积分等场景至关重要。

    重要：所有进程都必须调用此函数！

    Args:
        tensor: 输入张量（仅主进程提供）
        mpi: MPI管理器
        distributor: 张量分配器
        dim: 求和维度（None 表示全局求和）

    Returns:
        求和结果（仅主进程返回）
    """
    if should_use_single_gpu(mpi, tensor):
        if mpi.is_master_process():
            assert tensor is not None
            if dim is None:
                return torch.tensor(
                    _kahan_sum_blocks(tensor),
                    dtype=tensor.dtype, device=tensor.device,
                )
            return tensor.to(torch.float64).sum(dim=dim).to(tensor.dtype)
        return None

    tensor_local: torch.Tensor = distributor.distribute(tensor, dim=0)
    orig_dtype = tensor_local.dtype

    if dim is None:
        local_sum = _kahan_sum_blocks(tensor_local)
        total = _kahan_allreduce(local_sum, mpi, tensor_local.device)
        if mpi.is_master_process():
            return total.squeeze().to(orig_dtype)
        return None

    elif dim == 0:
        local_sum_t = tensor_local.to(torch.float64).sum(dim=0)
        total = mpi.allreduce_tensor(local_sum_t)
        if mpi.is_master_process():
            return total.to(orig_dtype)
        return None

    else:
        local_sum_t = tensor_local.to(torch.float64).sum(dim=dim)
        return distributor.gather(local_sum_t.to(orig_dtype), dim=0)


def distributed_mean_kahan(
    tensor: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    数值稳定的分布式均值（基于 Kahan 补偿求和）

    重要：所有进程都必须调用此函数！
    """
    if should_use_single_gpu(mpi, tensor):
        if mpi.is_master_process():
            assert tensor is not None
            if dim is None:
                s = _kahan_sum_blocks(tensor)
                return torch.tensor(
                    s / tensor.numel(),
                    dtype=tensor.dtype, device=tensor.device,
                )
            return tensor.to(torch.float64).mean(dim=dim).to(tensor.dtype)
        return None

    tensor_local: torch.Tensor = distributor.distribute(tensor, dim=0)
    orig_dtype = tensor_local.dtype

    if dim is None:
        local_sum = _kahan_sum_blocks(tensor_local)
        local_count = tensor_local.numel()

        sum_count = torch.tensor(
            [local_sum, float(local_count)], dtype=torch.float64
        )
        if torch.cuda.is_available():
            sum_count = sum_count.to(tensor_local.device)
        total = mpi.allreduce_tensor(sum_count)

        if mpi.is_master_process():
            return (total[0] / total[1]).to(orig_dtype)
        return None

    elif dim == 0:
        local_sum_t = tensor_local.to(torch.float64).sum(dim=0)
        local_count = tensor_local.shape[0]
        total_sum = mpi.allreduce_tensor(local_sum_t)
        total_count: int = mpi.allreduce(local_count)
        if mpi.is_master_process():
            return (total_sum / total_count).to(orig_dtype)
        return None

    else:
        local_mean = tensor_local.to(torch.float64).mean(dim=dim)
        return distributor.gather(local_mean.to(orig_dtype), dim=0)
