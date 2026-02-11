"""
分布式Einstein求和（Einsum）

实现通用张量收缩的分布式计算。
集成 opt_einsum 提供最优收缩路径优化。
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Union
import opt_einsum as oe

from ..mpi_manager import MPIManager
from ..tensor_distributor import TensorDistributor
from ._utils import should_use_single_gpu


def get_optimal_path(equation: str, *shapes, optimize: str = 'optimal') -> Tuple:
    """
    使用 opt_einsum 获取最优收缩路径
    
    Args:
        equation: Einstein求和表达式
        *shapes: 操作数形状列表
        optimize: 优化策略
            - 'optimal': 最优路径（穷举搜索，适合小规模）
            - 'dp': 动态规划（适合中等规模）
            - 'greedy': 贪心算法（适合大规模）
            - 'random-greedy': 随机贪心
            - 'branch-all': 分支定界
            - 'auto': 自动选择
    
    Returns:
        (path, path_info) 元组
    """
    # 创建虚拟数组用于路径优化
    dummy_arrays = [np.empty(shape) for shape in shapes]
    
    # 获取最优路径
    path, path_info = oe.contract_path(equation, *dummy_arrays, optimize=optimize)
    
    return path, path_info


def print_path_info(equation: str, *shapes, optimize: str = 'optimal'):
    """
    打印最优路径信息
    
    Args:
        equation: Einstein求和表达式
        *shapes: 操作数形状列表
        optimize: 优化策略
    """
    path, path_info = get_optimal_path(equation, *shapes, optimize=optimize)
    
    print(f"\nEinsum 表达式: {equation}")
    print(f"操作数形状: {shapes}")
    print(f"优化策略: {optimize}")
    print(f"\n最优收缩路径: {path}")
    print(f"\n详细信息:")
    print(path_info)


def distributed_einsum(equation: str,
                       *operands: Optional[torch.Tensor],
                       mpi: MPIManager,
                       distributor: TensorDistributor,
                       optimize: str = 'auto',
                       use_opt_einsum: bool = True) -> Optional[torch.Tensor]:
    """
    分布式Einstein求和（集成opt_einsum最优路径）
    
    支持的操作示例：
    - 矩阵乘法: "ij,jk->ik"
    - 批量矩阵乘法: "bij,bjk->bik"
    - 外积: "i,j->ij"
    - 迹: "ii->"
    - 张量收缩: "ijkl,klmn->ijmn"
    
    策略：按第一个维度分割所有共享该维度的操作数
    
    重要：所有进程都必须调用此函数！
    
    Args:
        equation: Einstein求和表达式
        *operands: 操作数张量（仅主进程需要提供）
        mpi: MPI管理器
        distributor: 张量分配器
        optimize: opt_einsum优化策略
            - 'optimal': 最优路径（穷举搜索）
            - 'dp': 动态规划
            - 'greedy': 贪心算法
            - 'auto': 自动选择（默认）
        use_opt_einsum: 是否使用opt_einsum（默认True）
    
    Returns:
        计算结果（仅主进程返回）
    """
    if len(operands) < 1:
        raise ValueError("至少需要一个操作数")

    if should_use_single_gpu(mpi, *operands):
        if mpi.is_master_process():
            if use_opt_einsum:
                result = oe.contract(equation, *operands, optimize=optimize)
                if isinstance(result, np.ndarray):
                    result = torch.from_numpy(result)
                    if torch.cuda.is_available():
                        result = result.cuda(mpi.get_gpu_id())
                return result
            else:
                return torch.einsum(equation, *operands)
        return None
    
    # 解析equation，找出每个操作数的下标
    input_subscripts, output_subscript = equation.replace(' ', '').split('->')
    subscript_list = input_subscripts.split(',')
    
    # 第一个操作数的第一个下标（用于确定分布式分割维度）
    first_dim_subscript = subscript_list[0][0] if subscript_list[0] else None
    
    # 分发操作数
    local_operands = []
    
    for i, op in enumerate(operands):
        subscripts = subscript_list[i] if i < len(subscript_list) else ""
        
        # 检查该操作数是否以相同的下标开头（需要同时分割）
        if subscripts and subscripts[0] == first_dim_subscript:
            # 按第一维分割
            local_op = distributor.distribute(op, dim=0)
        else:
            # 广播
            local_op = distributor.broadcast(op)
        local_operands.append(local_op)
    
    # 本地计算（使用opt_einsum最优路径或原生torch.einsum）
    if use_opt_einsum:
        # 使用 opt_einsum 的 contract 函数（带最优路径）
        output_local = oe.contract(equation, *local_operands, optimize=optimize)
        # opt_einsum 返回 numpy array 或 torch tensor，确保是 torch tensor
        if isinstance(output_local, np.ndarray):
            output_local = torch.from_numpy(output_local)
            if torch.cuda.is_available():
                output_local = output_local.cuda(mpi.get_gpu_id())
    else:
        # 使用原生 torch.einsum
        output_local = torch.einsum(equation, *local_operands)
    
    # 根据输出维度决定如何收集
    if first_dim_subscript and first_dim_subscript in output_subscript:
        # 按维度收集
        output = distributor.gather(output_local, dim=0)
    else:
        # 需要归约（求和）
        output = distributor.reduce_sum(output_local)
        if mpi.is_master_process():
            output = output
        else:
            output = None
    
    mpi.synchronize()
    return output


def distributed_einsum_with_path(equation: str,
                                  *operands: Optional[torch.Tensor],
                                  mpi: MPIManager,
                                  distributor: TensorDistributor,
                                  path: Optional[List] = None,
                                  optimize: str = 'auto') -> Optional[torch.Tensor]:
    """
    使用指定收缩路径的分布式Einstein求和
    
    当需要多次执行相同结构的 einsum 时，可以预先计算路径然后复用。
    
    重要：所有进程都必须调用此函数！
    
    Args:
        equation: Einstein求和表达式
        *operands: 操作数张量
        mpi: MPI管理器
        distributor: 张量分配器
        path: 预计算的收缩路径（None则自动计算）
        optimize: 当path为None时的优化策略
    
    Returns:
        计算结果
    """
    if len(operands) < 1:
        raise ValueError("至少需要一个操作数")

    if should_use_single_gpu(mpi, *operands):
        if mpi.is_master_process():
            if path is None:
                shapes = [op.shape for op in operands]
                path, _ = get_optimal_path(equation, *shapes, optimize=optimize)
            result = oe.contract(equation, *operands, optimize=path)
            if isinstance(result, np.ndarray):
                result = torch.from_numpy(result)
                if torch.cuda.is_available():
                    result = result.cuda(mpi.get_gpu_id())
            return result
        return None
    
    # 解析equation
    input_subscripts, output_subscript = equation.replace(' ', '').split('->')
    subscript_list = input_subscripts.split(',')
    first_dim_subscript = subscript_list[0][0] if subscript_list[0] else None
    
    # 分发操作数
    local_operands = []
    for i, op in enumerate(operands):
        subscripts = subscript_list[i] if i < len(subscript_list) else ""
        if subscripts and subscripts[0] == first_dim_subscript:
            local_op = distributor.distribute(op, dim=0)
        else:
            local_op = distributor.broadcast(op)
        local_operands.append(local_op)
    
    # 如果没有提供路径，计算最优路径
    if path is None:
        shapes = [op.shape for op in local_operands]
        path, _ = get_optimal_path(equation, *shapes, optimize=optimize)
    
    # 使用指定路径计算
    output_local = oe.contract(equation, *local_operands, optimize=path)
    
    if isinstance(output_local, np.ndarray):
        output_local = torch.from_numpy(output_local)
        if torch.cuda.is_available():
            output_local = output_local.cuda(mpi.get_gpu_id())
    
    # 收集结果
    if first_dim_subscript and first_dim_subscript in output_subscript:
        output = distributor.gather(output_local, dim=0)
    else:
        output = distributor.reduce_sum(output_local)
        if not mpi.is_master_process():
            output = None
    
    mpi.synchronize()
    return output


def compare_optimization_strategies(equation: str, *shapes):
    """
    比较不同优化策略的性能
    
    Args:
        equation: Einstein求和表达式
        *shapes: 操作数形状列表
    """
    strategies = ['optimal', 'dp', 'greedy', 'random-greedy', 'auto']
    
    print(f"\n{'='*60}")
    print(f"Einsum优化策略比较: {equation}")
    print(f"操作数形状: {shapes}")
    print(f"{'='*60}")
    
    for strategy in strategies:
        try:
            path, info = get_optimal_path(equation, *shapes, optimize=strategy)
            
            # 提取关键信息
            flops = info.opt_cost
            
            print(f"\n策略: {strategy}")
            print(f"  收缩路径: {path}")
            print(f"  计算复杂度: {flops:,.0f} FLOPs")
            
        except Exception as e:
            print(f"\n策略: {strategy} - 错误: {e}")
    
    print(f"\n{'='*60}")


def distributed_tensordot(a: Optional[torch.Tensor],
                          b: Optional[torch.Tensor],
                          dims: int,
                          mpi: MPIManager,
                          distributor: TensorDistributor) -> Optional[torch.Tensor]:
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
            return torch.tensordot(a, b, dims=dims)
        return None
    
    # 分发a，广播b
    a_local = distributor.distribute(a, dim=0)
    b_local = distributor.broadcast(b)
    
    # 本地计算
    output_local = torch.tensordot(a_local, b_local, dims=dims)
    
    # 收集
    output = distributor.gather(output_local, dim=0)
    
    mpi.synchronize()
    return output


def create_contraction_expression(equation: str, *shapes, optimize: str = 'auto'):
    """
    创建可复用的收缩表达式（预编译）
    
    当需要多次执行相同结构的 einsum 时，预编译可以提升性能。
    
    Args:
        equation: Einstein求和表达式
        *shapes: 操作数形状列表
        optimize: 优化策略
    
    Returns:
        可复用的收缩表达式对象
    
    使用示例:
        expr = create_contraction_expression('ij,jk->ik', (100,50), (50,80))
        result = expr(A, B)  # 多次调用
    """
    return oe.contract_expression(equation, *shapes, optimize=optimize)
