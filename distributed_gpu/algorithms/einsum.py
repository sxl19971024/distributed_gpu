"""
分布式 Einstein 求和（Einsum）

实现通用张量收缩的分布式计算。
集成 opt_einsum 提供最优收缩路径优化。

支持的操作示例：
- 矩阵乘法:       "ij,jk->ik"
- 批量矩阵乘法:   "bij,bjk->bik"
- 外积:           "i,j->ij"
- 迹:             "ii->"
- 张量收缩:       "ijkl,klmn->ijmn"
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Any, List, Optional, Tuple

import opt_einsum as oe

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu


# ==================== 路径优化工具 ====================


def get_optimal_path(
    equation: str,
    *shapes: Tuple[int, ...],
    optimize: str = "optimal",
) -> Tuple[list, Any]:
    """
    使用 opt_einsum 获取最优收缩路径

    Args:
        equation: Einstein 求和表达式
        *shapes: 操作数形状列表
        optimize: 优化策略
            - 'optimal':        穷举搜索（适合小规模，≤6 个操作数）
            - 'dp':             动态规划（适合中等规模）
            - 'greedy':         贪心算法（适合大规模）
            - 'random-greedy':  随机贪心
            - 'branch-all':     分支定界
            - 'auto':           自动选择

    Returns:
        (path, path_info) 元组
    """
    dummy_arrays = [np.empty(shape) for shape in shapes]
    return oe.contract_path(equation, *dummy_arrays, optimize=optimize)


def print_path_info(
    equation: str,
    *shapes: Tuple[int, ...],
    optimize: str = "optimal",
) -> None:
    """打印最优路径的详细信息"""
    path, path_info = get_optimal_path(equation, *shapes, optimize=optimize)

    print(f"\nEinsum 表达式: {equation}")
    print(f"操作数形状: {shapes}")
    print(f"优化策略: {optimize}")
    print(f"\n最优收缩路径: {path}")
    print(f"\n详细信息:")
    print(path_info)


def compare_optimization_strategies(
    equation: str,
    *shapes: Tuple[int, ...],
) -> None:
    """比较不同优化策略的性能"""
    strategies = ["optimal", "dp", "greedy", "random-greedy", "auto"]

    print(f"\n{'=' * 60}")
    print(f"Einsum 优化策略比较: {equation}")
    print(f"操作数形状: {shapes}")
    print(f"{'=' * 60}")

    for strategy in strategies:
        try:
            path, info = get_optimal_path(equation, *shapes, optimize=strategy)
            print(f"\n策略: {strategy}")
            print(f"  收缩路径: {path}")
            print(f"  计算复杂度: {info.opt_cost:,.0f} FLOPs")
        except Exception as e:
            print(f"\n策略: {strategy} - 错误: {e}")

    print(f"\n{'=' * 60}")


def create_contraction_expression(
    equation: str,
    *shapes: Tuple[int, ...],
    optimize: str = "auto",
) -> Any:
    """
    创建可复用的收缩表达式（预编译）

    当需要多次执行相同结构的 einsum 时，预编译可以提升性能。

    使用示例::

        expr = create_contraction_expression('ij,jk->ik', (100,50), (50,80))
        result = expr(A, B)  # 多次调用
    """
    return oe.contract_expression(equation, *shapes, optimize=optimize)


# ==================== 内部辅助 ====================


def _parse_equation(equation: str) -> Tuple[List[str], str, Optional[str]]:
    """
    解析 einsum 表达式。

    Returns:
        (subscript_list, output_subscript, first_dim_subscript)
    """
    input_subscripts, output_subscript = equation.replace(" ", "").split("->")
    subscript_list = input_subscripts.split(",")
    first_dim_subscript = subscript_list[0][0] if subscript_list[0] else None
    return subscript_list, output_subscript, first_dim_subscript


def _ensure_torch_tensor(
    result: Any, gpu_id: int
) -> torch.Tensor:
    """
    将 opt_einsum 的输出（可能是 np.ndarray）转为 GPU tensor。

    性能优化：使用 non_blocking 传输和避免不必要的拷贝。
    """
    if isinstance(result, np.ndarray):
        # 使用 from_numpy（零拷贝）然后异步传输到 GPU
        result = torch.from_numpy(result)
        if torch.cuda.is_available():
            result = result.cuda(gpu_id, non_blocking=True)
    elif isinstance(result, torch.Tensor) and torch.cuda.is_available():
        if not result.is_cuda:
            result = result.cuda(gpu_id, non_blocking=True)
    return result


def _distribute_operands(
    operands: Tuple[Optional[torch.Tensor], ...],
    subscript_list: List[str],
    first_dim_subscript: Optional[str],
    distributor: TensorDistributor,
) -> List[torch.Tensor]:
    """
    根据下标分析分发操作数：
    - 与第一个操作数共享首维下标的 → scatter (dim=0)
    - 其他 → broadcast
    """
    local_operands: List[torch.Tensor] = []
    for i, op in enumerate(operands):
        subscripts = subscript_list[i] if i < len(subscript_list) else ""
        if subscripts and subscripts[0] == first_dim_subscript:
            local_operands.append(distributor.distribute(op, dim=0))
        else:
            local_operands.append(distributor.broadcast(op))
    return local_operands


def _collect_result(
    output_local: torch.Tensor,
    first_dim_subscript: Optional[str],
    output_subscript: str,
    mpi: MPIManager,
    distributor: TensorDistributor,
) -> Optional[torch.Tensor]:
    """
    根据输出下标决定收集策略：
    - 首维下标出现在输出中 → gather (dim=0)
    - 否则 → reduce_sum
    """
    if first_dim_subscript and first_dim_subscript in output_subscript:
        return distributor.gather(output_local, dim=0)
    else:
        result = distributor.reduce_sum(output_local)
        return result if mpi.is_master_process() else None


# ==================== 公开 API ====================


def distributed_einsum(
    equation: str,
    *operands: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    optimize: str = "auto",
    use_opt_einsum: bool = True,
) -> Optional[torch.Tensor]:
    """
    分布式 Einstein 求和（集成 opt_einsum 最优路径）

    策略：按第一个维度分割所有共享该维度的操作数。

    重要：所有进程都必须调用此函数！

    Args:
        equation: Einstein 求和表达式
        *operands: 操作数张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        optimize: opt_einsum 优化策略
        use_opt_einsum: 是否使用 opt_einsum（默认 True）

    Returns:
        计算结果（仅主进程返回）
    """
    if len(operands) < 1:
        raise ValueError("至少需要一个操作数")

    # 单卡快速路径
    if should_use_single_gpu(mpi, *operands):
        if mpi.is_master_process():
            if use_opt_einsum:
                result = oe.contract(equation, *operands, optimize=optimize)
                return _ensure_torch_tensor(result, mpi.get_gpu_id())
            return torch.einsum(equation, *operands)
        return None

    subscript_list, output_subscript, first_dim_subscript = _parse_equation(equation)
    local_operands = _distribute_operands(
        operands, subscript_list, first_dim_subscript, distributor
    )

    # 本地计算
    # 性能优化：当操作数在 GPU 上时，优先使用 torch.einsum 避免
    # opt_einsum 的 GPU→CPU→GPU 往返（opt_einsum 内部使用 numpy）
    if use_opt_einsum:
        # 对于 2 个操作数的简单情况，torch.einsum 已经足够高效
        # 且避免了 numpy 转换开销；仅对 3+ 操作数使用 opt_einsum
        if len(local_operands) <= 2 and all(
            op.is_cuda for op in local_operands
        ):
            output_local = torch.einsum(equation, *local_operands)
        else:
            output_local = oe.contract(equation, *local_operands, optimize=optimize)
            output_local = _ensure_torch_tensor(output_local, mpi.get_gpu_id())
    else:
        output_local = torch.einsum(equation, *local_operands)

    return _collect_result(
        output_local, first_dim_subscript, output_subscript, mpi, distributor
    )


def distributed_einsum_with_path(
    equation: str,
    *operands: Optional[torch.Tensor],
    mpi: MPIManager,
    distributor: TensorDistributor,
    path: Optional[list] = None,
    optimize: str = "auto",
) -> Optional[torch.Tensor]:
    """
    使用指定收缩路径的分布式 Einstein 求和

    当需要多次执行相同结构的 einsum 时，可以预先计算路径然后复用。

    重要：所有进程都必须调用此函数！

    Args:
        equation: Einstein 求和表达式
        *operands: 操作数张量
        mpi: MPI管理器
        distributor: 张量分配器
        path: 预计算的收缩路径（None 则自动计算）
        optimize: 当 path 为 None 时的优化策略

    Returns:
        计算结果
    """
    if len(operands) < 1:
        raise ValueError("至少需要一个操作数")

    # 单卡快速路径
    if should_use_single_gpu(mpi, *operands):
        if mpi.is_master_process():
            if path is None:
                shapes = [op.shape for op in operands]  # type: ignore[union-attr]
                path, _ = get_optimal_path(equation, *shapes, optimize=optimize)
            result = oe.contract(equation, *operands, optimize=path)
            return _ensure_torch_tensor(result, mpi.get_gpu_id())
        return None

    subscript_list, output_subscript, first_dim_subscript = _parse_equation(equation)
    local_operands = _distribute_operands(
        operands, subscript_list, first_dim_subscript, distributor
    )

    # 如果没有提供路径，计算最优路径
    if path is None:
        shapes = [op.shape for op in local_operands]
        path, _ = get_optimal_path(equation, *shapes, optimize=optimize)

    output_local = oe.contract(equation, *local_operands, optimize=path)
    output_local = _ensure_torch_tensor(output_local, mpi.get_gpu_id())

    return _collect_result(
        output_local, first_dim_subscript, output_subscript, mpi, distributor
    )


def distributed_tensordot(
    a: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    dims: int,
    mpi: MPIManager,
    distributor: TensorDistributor,
) -> Optional[torch.Tensor]:
    """
    分布式张量点积

    重要：所有进程都必须调用此函数！

    Args:
        a: 第一个张量（仅主进程需要提供）
        b: 第二个张量（仅主进程需要提供）
        dims: 收缩的维度数
        mpi: MPI管理器
        distributor: 张量分配器

    Returns:
        张量点积结果（仅主进程返回）
    """
    if should_use_single_gpu(mpi, a, b):
        if mpi.is_master_process():
            return torch.tensordot(a, b, dims=dims)  # type: ignore[arg-type]
        return None

    a_local: torch.Tensor = distributor.distribute(a, dim=0)
    b_local: torch.Tensor = distributor.broadcast(b)
    output_local: torch.Tensor = torch.tensordot(a_local, b_local, dims=dims)
    return distributor.gather(output_local, dim=0)
