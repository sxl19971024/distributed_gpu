"""
算子层共享工具 — 单卡快速路径判断

当输入张量足以放进单张 GPU 的可用显存时，直接在 rank 0 上本地计算，
跳过 scatter/broadcast/gather 等 MPI 通信开销。
小数据场景下性能与 CuPy 单卡一致。
"""

import torch
from ..mpi_manager import MPIManager


def should_use_single_gpu(mpi: MPIManager, *tensors,
                          mem_multiplier: float = 3.0) -> bool:
    """
    判断是否应使用单卡快速路径。

    ** 所有 rank 都必须调用此函数（内部含 broadcast） **

    逻辑：
      1. 单进程 (size==1) → True（无需广播）
      2. 多进程 → master 估算：输入张量总字节 × mem_multiplier
         如果 < 单卡可用显存的 60%，则 True（广播决策到所有 rank）

    Args:
        mpi: MPI 管理器
        *tensors: 输入张量（仅 master 上为有效数据，其他 rank 为 None）
        mem_multiplier: 输入字节数 × 此倍数 ≈ 总所需显存
                        （含输出 + 中间变量，默认 3.0）

    Returns:
        True → 应走单卡路径；False → 应走分布式路径
    """
    # 单进程：必定单卡，无需广播
    if mpi.get_size() == 1:
        return True

    # 多进程：master 估算显存，广播决策
    if mpi.is_master_process():
        total_bytes = 0
        for t in tensors:
            if t is not None and isinstance(t, torch.Tensor):
                total_bytes += t.nelement() * t.element_size()
        estimated = int(total_bytes * mem_multiplier)
        try:
            free_mem, _ = torch.cuda.mem_get_info(mpi.get_gpu_id())
            use_single = estimated < free_mem * 0.6
        except Exception:
            use_single = False
    else:
        use_single = None

    return mpi.broadcast(use_single)
