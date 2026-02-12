"""
分布式 Stencil 计算（含 Halo Exchange）

实现物理模拟中的核心计算模式：有限差分法的 Stencil 操作。
支持 2D 网格的分布式计算，通过 Halo Exchange（光晕交换）
实现进程间边界数据的高效交换。

典型应用：
- 热传导方程求解
- 泊松方程求解（Jacobi 迭代）
- 波动方程模拟
- 流体力学中的格点计算

通信优化：Halo 交换通信量 O(W)，远小于全局通信 O(H×W)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from mpi4py import MPI
from typing import Optional, Tuple

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu


# ==================== 预定义 Stencil 核 ====================

DEFAULT_LAPLACIAN_5PT: torch.Tensor = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)

DEFAULT_LAPLACIAN_9PT: torch.Tensor = torch.tensor(
    [[1.0, 4.0, 1.0],
     [4.0, -20.0, 4.0],
     [1.0, 4.0, 1.0]],
    dtype=torch.float32,
) / 6.0


# ==================== Halo Exchange ====================


class _HaloBuffers:
    """
    预分配的 Halo Exchange 缓冲区。

    在迭代 Stencil 计算中，每次迭代都需要交换 halo 数据。
    预分配缓冲区避免了每次迭代的内存分配开销。
    """

    __slots__ = (
        "W", "up_rank", "down_rank",
        "top_row_np", "bottom_row_np",
        "top_halo_np", "bottom_halo_np",
        "top_halo_gpu", "bottom_halo_gpu",
    )

    def __init__(
        self,
        W: int,
        dtype_np: np.dtype,
        up_rank: int,
        down_rank: int,
        device: torch.device,
    ) -> None:
        self.W = W
        self.up_rank = up_rank
        self.down_rank = down_rank
        # 预分配 numpy 发送/接收缓冲区
        self.top_row_np = np.zeros(W, dtype=dtype_np)
        self.bottom_row_np = np.zeros(W, dtype=dtype_np)
        self.top_halo_np = np.zeros(W, dtype=dtype_np)
        self.bottom_halo_np = np.zeros(W, dtype=dtype_np)
        # 预分配 GPU 接收张量
        torch_dtype = torch.float32 if dtype_np == np.float32 else torch.float64
        self.top_halo_gpu = torch.zeros(W, dtype=torch_dtype, device=device)
        self.bottom_halo_gpu = torch.zeros(W, dtype=torch_dtype, device=device)


def _create_halo_buffers(
    local_grid: torch.Tensor,
    mpi: MPIManager,
    boundary: str = "zero",
) -> _HaloBuffers:
    """创建预分配的 halo 交换缓冲区（仅在首次迭代前调用一次）。"""
    rank: int = mpi.get_rank()
    size: int = mpi.get_size()
    W: int = local_grid.shape[-1]
    dtype_np = local_grid.cpu().numpy().dtype

    if boundary == "periodic":
        up_rank: int = (rank - 1) % size
        down_rank: int = (rank + 1) % size
    else:
        up_rank = rank - 1 if rank > 0 else MPI.PROC_NULL
        down_rank = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    return _HaloBuffers(W, dtype_np, up_rank, down_rank, local_grid.device)


def _halo_exchange(
    local_grid: torch.Tensor,
    mpi: MPIManager,
    boundary: str = "zero",
    buffers: Optional[_HaloBuffers] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Halo Exchange（光晕交换）— 无死锁实现

    使用 MPI Sendrecv（大写/缓冲区版本）配合 MPI.PROC_NULL
    处理边界，保证所有进程同步参与通信，无死锁。

    性能优化：当提供 ``buffers`` 时，复用预分配的 numpy 和 GPU 缓冲区，
    避免每次迭代的内存分配开销。

    Args:
        local_grid: 本地网格 [local_H, W]
        mpi: MPI管理器
        boundary: 'zero' 或 'periodic'
        buffers: 预分配缓冲区（可选，用于迭代场景）

    Returns:
        (top_halo, bottom_halo) 各为 [W] 的 1D 张量
    """
    if buffers is None:
        # 兼容旧接口：无预分配缓冲区，动态分配
        rank: int = mpi.get_rank()
        size: int = mpi.get_size()
        W: int = local_grid.shape[-1]
        dtype_np = local_grid.cpu().numpy().dtype

        if boundary == "periodic":
            up_rank: int = (rank - 1) % size
            down_rank: int = (rank + 1) % size
        else:
            up_rank = rank - 1 if rank > 0 else MPI.PROC_NULL
            down_rank = rank + 1 if rank < size - 1 else MPI.PROC_NULL

        top_row_np = np.ascontiguousarray(local_grid[0].cpu().numpy())
        bottom_row_np = np.ascontiguousarray(local_grid[-1].cpu().numpy())
        top_halo_np = np.zeros(W, dtype=dtype_np)
        bottom_halo_np = np.zeros(W, dtype=dtype_np)
    else:
        # 性能路径：复用预分配缓冲区，仅拷贝数据
        up_rank = buffers.up_rank
        down_rank = buffers.down_rank
        top_row_np = buffers.top_row_np
        bottom_row_np = buffers.bottom_row_np
        top_halo_np = buffers.top_halo_np
        bottom_halo_np = buffers.bottom_halo_np
        # 将 GPU 数据拷贝到预分配的 numpy 缓冲区（避免新建 numpy 数组）
        top_row_np[:] = local_grid[0].cpu().numpy()
        bottom_row_np[:] = local_grid[-1].cpu().numpy()
        # 清零接收缓冲区
        top_halo_np[:] = 0
        bottom_halo_np[:] = 0

    # Exchange A: 向下发送 bottom_row，从上接收 top_halo
    mpi._safe_call(
        "Sendrecv(halo_top)", mpi.comm.Sendrecv,
        sendbuf=bottom_row_np, dest=down_rank, sendtag=0,
        recvbuf=top_halo_np, source=up_rank, recvtag=0,
    )

    # Exchange B: 向上发送 top_row，从下接收 bottom_halo
    mpi._safe_call(
        "Sendrecv(halo_bot)", mpi.comm.Sendrecv,
        sendbuf=top_row_np, dest=up_rank, sendtag=1,
        recvbuf=bottom_halo_np, source=down_rank, recvtag=1,
    )

    if buffers is not None:
        # 性能路径：直接拷贝到预分配的 GPU 张量，避免新建张量
        buffers.top_halo_gpu.copy_(torch.from_numpy(top_halo_np))
        buffers.bottom_halo_gpu.copy_(torch.from_numpy(bottom_halo_np))
        return buffers.top_halo_gpu, buffers.bottom_halo_gpu

    top_halo: torch.Tensor = mpi._from_numpy_to_gpu(top_halo_np)
    bottom_halo: torch.Tensor = mpi._from_numpy_to_gpu(bottom_halo_np)

    return top_halo, bottom_halo


# ==================== 辅助：单卡 Stencil ====================


def _single_gpu_stencil(
    grid: torch.Tensor,
    stencil_kernel: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """单 GPU 上的 Stencil 迭代"""
    if grid.is_cuda:
        stencil_kernel = stencil_kernel.to(grid.device)
    kH, kW = stencil_kernel.shape
    kernel_4d = stencil_kernel.unsqueeze(0).unsqueeze(0)  # [1,1,kH,kW]

    result = grid
    for _ in range(iterations):
        padded = F.pad(
            result.unsqueeze(0).unsqueeze(0),
            (kW // 2, kW // 2, kH // 2, kH // 2),
            mode="constant",
        )
        result = F.conv2d(padded, kernel_4d).squeeze(0).squeeze(0)
    return result


# ==================== 公开 API ====================


def distributed_stencil_2d(
    grid: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    stencil_kernel: Optional[torch.Tensor] = None,
    boundary: str = "zero",
    iterations: int = 1,
) -> Optional[torch.Tensor]:
    """
    分布式 2D Stencil 计算

    将 2D 网格按行分割到各进程，每次迭代通过 Halo Exchange 交换
    边界数据，然后在本地应用 Stencil 核（利用 conv2d 实现高效 GPU 计算）。

    通信量分析：
    - Halo Exchange: O(W) per iteration per process
    - 计算量: O(local_H × W × kernel_size²) per process
    - 通信计算比: O(1/local_H)，网格越大越高效

    要求 H 能被进程数 P 整除。

    重要：所有进程都必须调用此函数！

    Args:
        grid: 2D 网格 [H, W]（仅主进程提供）
        mpi: MPI管理器
        distributor: 张量分配器
        stencil_kernel: Stencil 核（2D 张量，如 3×3）
                        默认: 5 点 Laplacian [[0,1,0],[1,-4,1],[0,1,0]]
        boundary: 边界条件 'zero' 或 'periodic'
        iterations: 迭代次数

    Returns:
        结果网格 [H, W]（仅主进程返回）
    """
    # 准备 Stencil 核
    if stencil_kernel is None:
        stencil_kernel = DEFAULT_LAPLACIAN_5PT.clone()

    # 单卡快速路径
    if should_use_single_gpu(mpi, grid):
        if mpi.is_master_process():
            return _single_gpu_stencil(grid, stencil_kernel, iterations)  # type: ignore[arg-type]
        return None

    # 广播 Stencil 核
    if mpi.is_master_process() and torch.cuda.is_available():
        stencil_kernel = stencil_kernel.cuda()
    kernel: torch.Tensor = distributor.broadcast(
        stencil_kernel if mpi.is_master_process() else None
    )
    kH, kW = kernel.shape
    pad_w: int = kW // 2
    kernel_4d: torch.Tensor = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,kH,kW]

    # 分发网格行
    local_grid: torch.Tensor = distributor.distribute(grid, dim=0)

    # 性能优化：预分配 halo 缓冲区，避免每次迭代重新分配
    halo_bufs: _HaloBuffers = _create_halo_buffers(local_grid, mpi, boundary)

    # 性能优化：预分配 padded 张量 [local_H+2, W]，每次迭代复用
    local_H: int = local_grid.shape[0]
    W: int = local_grid.shape[1]
    padded = torch.zeros(
        local_H + 2, W, dtype=local_grid.dtype, device=local_grid.device
    )

    for _ in range(iterations):
        # 1. Halo Exchange（使用预分配缓冲区）
        top_halo, bottom_halo = _halo_exchange(
            local_grid, mpi, boundary, buffers=halo_bufs,
        )

        # 2. 拼接 halo 行 → [local_H+2, W]（原地填充，避免 torch.cat 分配）
        padded[0] = top_halo
        padded[1:-1] = local_grid
        padded[-1] = bottom_halo

        # 3. conv2d 应用 Stencil（top/bottom 已 halo 填充，只 pad left/right）
        padded_4d = padded.unsqueeze(0).unsqueeze(0)
        result_4d = F.conv2d(padded_4d, kernel_4d, padding=(0, pad_w))
        local_grid = result_4d.squeeze(0).squeeze(0)

    return distributor.gather(local_grid, dim=0)


def distributed_jacobi_2d(
    grid: Optional[torch.Tensor],
    rhs: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    dx: float = 1.0,
    boundary: str = "zero",
    iterations: int = 100,
    tol: float = 1e-6,
) -> Optional[torch.Tensor]:
    """
    分布式 Jacobi 迭代求解 2D 泊松方程 ∇²u = f

    Jacobi 迭代公式：
    u_new[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - dx²·f[i,j]) / 4

    这是科学计算中最经典的 Stencil 应用之一。

    重要：所有进程都必须调用此函数！

    Args:
        grid: 初始猜测 [H, W]（仅主进程）
        rhs: 右端项 f [H, W]（仅主进程）
        mpi: MPI管理器
        distributor: 张量分配器
        dx: 网格间距
        boundary: 边界条件
        iterations: 最大迭代次数
        tol: 收敛容差

    Returns:
        求解结果 [H, W]（仅主进程）
    """
    dx2: float = dx * dx

    # 单卡快速路径
    if should_use_single_gpu(mpi, grid, rhs):
        if mpi.is_master_process():
            assert grid is not None and rhs is not None
            u = grid.clone()
            for it in range(iterations):
                padded = F.pad(
                    u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="constant"
                ).squeeze(0).squeeze(0)
                u_new = (
                    padded[:-2, 1:-1] + padded[2:, 1:-1]
                    + padded[1:-1, :-2] + padded[1:-1, 2:]
                    - dx2 * rhs
                ) / 4.0
                diff: float = torch.norm(u_new - u).item()
                u = u_new
                if diff < tol:
                    print(f"  Jacobi 收敛于第 {it + 1} 次迭代 (残差={diff:.2e})")
                    break
            return u
        return None

    # 分发网格和右端项
    local_u: torch.Tensor = distributor.distribute(grid, dim=0)
    local_f: torch.Tensor = distributor.distribute(rhs, dim=0)

    # 性能优化：预分配 halo 缓冲区和 padded 张量
    halo_bufs: _HaloBuffers = _create_halo_buffers(local_u, mpi, boundary)
    local_H: int = local_u.shape[0]
    W: int = local_u.shape[1]
    padded = torch.zeros(
        local_H + 2, W, dtype=local_u.dtype, device=local_u.device
    )
    # 预计算 dx² * f（在循环外计算，避免每次迭代重复乘法）
    dx2_f: torch.Tensor = dx2 * local_f

    for it in range(iterations):
        # Halo Exchange（使用预分配缓冲区）
        top_halo, bottom_halo = _halo_exchange(
            local_u, mpi, boundary, buffers=halo_bufs,
        )

        # 原地填充 padded 张量，避免 torch.cat 分配
        padded[0] = top_halo
        padded[1:-1] = local_u
        padded[-1] = bottom_halo

        u_up = padded[0:local_H]
        u_down = padded[2 : local_H + 2]
        u_left = F.pad(local_u[:, :-1], (1, 0))
        u_right = F.pad(local_u[:, 1:], (0, 1))

        local_u_new = (u_up + u_down + u_left + u_right - dx2_f) / 4.0

        # 全局收敛检测
        diff: float = torch.norm(local_u_new - local_u).item()
        diff_global: float = mpi.allreduce(diff)

        local_u = local_u_new

        if diff_global < tol:
            if mpi.is_master_process():
                print(f"  Jacobi 收敛于第 {it + 1} 次迭代 (残差={diff_global:.2e})")
            break

    return distributor.gather(local_u, dim=0)
