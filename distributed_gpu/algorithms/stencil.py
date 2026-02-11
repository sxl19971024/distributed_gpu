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

import torch
import torch.nn.functional as F
import numpy as np
from mpi4py import MPI
from typing import Optional

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu


# 预定义 Stencil 核
DEFAULT_LAPLACIAN_5PT = torch.tensor([
    [0., 1., 0.],
    [1., -4., 1.],
    [0., 1., 0.]
], dtype=torch.float32)

DEFAULT_LAPLACIAN_9PT = torch.tensor([
    [1., 4., 1.],
    [4., -20., 4.],
    [1., 4., 1.]
], dtype=torch.float32) / 6.0


def _halo_exchange(local_grid: torch.Tensor, mpi: MPIManager,
                   boundary: str = 'zero') -> tuple:
    """
    Halo Exchange（光晕交换）— 无死锁实现
    
    使用 MPI Sendrecv（大写/缓冲区版本）配合 MPI.PROC_NULL
    处理边界，保证所有进程同步参与通信，无死锁。
    
    Args:
        local_grid: 本地网格 [local_H, W]
        mpi: MPI管理器
        boundary: 'zero' 或 'periodic'
    
    Returns:
        (top_halo, bottom_halo)
    """
    rank = mpi.get_rank()
    size = mpi.get_size()
    W = local_grid.shape[-1]
    dtype_np = local_grid.cpu().numpy().dtype

    if boundary == 'periodic':
        up = (rank - 1) % size
        down = (rank + 1) % size
    else:
        up = rank - 1 if rank > 0 else MPI.PROC_NULL
        down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    top_row_np = np.ascontiguousarray(local_grid[0].cpu().numpy())
    bottom_row_np = np.ascontiguousarray(local_grid[-1].cpu().numpy())

    top_halo_np = np.zeros(W, dtype=dtype_np)
    bottom_halo_np = np.zeros(W, dtype=dtype_np)

    # Exchange A: bottom → down, recv top ← up
    mpi.comm.Sendrecv(
        sendbuf=bottom_row_np, dest=down, sendtag=0,
        recvbuf=top_halo_np, source=up, recvtag=0
    )

    # Exchange B: top → up, recv bottom ← down
    mpi.comm.Sendrecv(
        sendbuf=top_row_np, dest=up, sendtag=1,
        recvbuf=bottom_halo_np, source=down, recvtag=1
    )

    device = local_grid.device
    top_halo = torch.from_numpy(top_halo_np.copy()).to(device)
    bottom_halo = torch.from_numpy(bottom_halo_np.copy()).to(device)

    return top_halo, bottom_halo


def distributed_stencil_2d(grid: Optional[torch.Tensor],
                            mpi: MPIManager,
                            distributor: TensorDistributor,
                            stencil_kernel: Optional[torch.Tensor] = None,
                            boundary: str = 'zero',
                            iterations: int = 1) -> Optional[torch.Tensor]:
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
    if should_use_single_gpu(mpi, grid):
        if mpi.is_master_process():
            if stencil_kernel is None:
                stencil_kernel = DEFAULT_LAPLACIAN_5PT.clone()
            if grid.is_cuda:
                stencil_kernel = stencil_kernel.to(grid.device)
            kernel_4d = stencil_kernel.unsqueeze(0).unsqueeze(0)
            kH, kW = stencil_kernel.shape
            result = grid
            for _ in range(iterations):
                padded = F.pad(result.unsqueeze(0).unsqueeze(0),
                               (kW // 2, kW // 2, kH // 2, kH // 2), mode='constant')
                result = F.conv2d(padded, kernel_4d).squeeze(0).squeeze(0)
            return result
        return None

    # 广播 Stencil 核
    if mpi.is_master_process():
        if stencil_kernel is None:
            stencil_kernel = DEFAULT_LAPLACIAN_5PT.clone()
        if torch.cuda.is_available():
            stencil_kernel = stencil_kernel.cuda()

    kernel = distributor.broadcast(stencil_kernel if mpi.is_master_process() else None)
    kH, kW = kernel.shape
    pad_w = kW // 2

    # 分发网格行
    local_grid = distributor.distribute(grid, dim=0)

    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]

    for it in range(iterations):
        # 1. Halo Exchange
        top_halo, bottom_halo = _halo_exchange(local_grid, mpi, boundary)

        # 2. 拼接 halo 行
        padded = torch.cat([
            top_halo.unsqueeze(0),
            local_grid,
            bottom_halo.unsqueeze(0)
        ], dim=0)

        # 3. conv2d 应用 Stencil（top/bottom 已 halo 填充，只 pad left/right）
        padded_4d = padded.unsqueeze(0).unsqueeze(0)
        result_4d = F.conv2d(padded_4d, kernel_4d, padding=(0, pad_w))
        local_grid = result_4d.squeeze(0).squeeze(0)

    result = distributor.gather(local_grid, dim=0)
    mpi.synchronize()
    return result


def distributed_jacobi_2d(grid: Optional[torch.Tensor],
                           rhs: Optional[torch.Tensor],
                           mpi: MPIManager,
                           distributor: TensorDistributor,
                           dx: float = 1.0,
                           boundary: str = 'zero',
                           iterations: int = 100,
                           tol: float = 1e-6) -> Optional[torch.Tensor]:
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
    if should_use_single_gpu(mpi, grid, rhs):
        if mpi.is_master_process():
            u = grid.clone()
            dx2 = dx * dx
            for it in range(iterations):
                padded = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant')
                padded = padded.squeeze(0).squeeze(0)
                u_new = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                         padded[1:-1, :-2] + padded[1:-1, 2:] - dx2 * rhs) / 4.0
                diff = torch.norm(u_new - u).item()
                u = u_new
                if diff < tol:
                    print(f"  Jacobi 收敛于第 {it+1} 次迭代 (残差={diff:.2e})")
                    break
            return u
        return None

    local_u = distributor.distribute(grid, dim=0)
    local_f = distributor.distribute(rhs, dim=0)

    dx2 = dx * dx

    for it in range(iterations):
        top_halo, bottom_halo = _halo_exchange(local_u, mpi, boundary)

        padded = torch.cat([top_halo.unsqueeze(0), local_u, bottom_halo.unsqueeze(0)], dim=0)

        local_H = local_u.shape[0]

        u_up = padded[0:local_H]
        u_down = padded[2:local_H + 2]
        u_left = F.pad(local_u[:, :-1], (1, 0))
        u_right = F.pad(local_u[:, 1:], (0, 1))

        local_u_new = (u_up + u_down + u_left + u_right - dx2 * local_f) / 4.0

        diff = torch.norm(local_u_new - local_u).item()
        diff_global = mpi.allreduce(diff)

        local_u = local_u_new

        if diff_global < tol:
            if mpi.is_master_process():
                print(f"  Jacobi 收敛于第 {it+1} 次迭代 (残差={diff_global:.2e})")
            break

    result = distributor.gather(local_u, dim=0)
    mpi.synchronize()
    return result
