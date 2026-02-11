"""
基于代价模型的自适应张量分割算法

创新点1：提出一种基于计算代价、通信代价和显存代价联合优化的自适应分割策略。

理论贡献：
1. 建立了分布式张量计算的代价模型
2. 提出了多目标优化的分割策略选择算法
3. 设计了自适应的负载均衡机制
"""

import math
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class SplitStrategy(Enum):
    """分割策略类型"""
    ROW_SPLIT = "row"           # 按行分割 A
    COLUMN_SPLIT = "column"     # 按列分割 B
    BLOCK_2D = "block_2d"       # 2D块分割（SUMMA）
    BATCH_SPLIT = "batch"       # 按批次分割


@dataclass
class ClusterConfig:
    """集群配置信息"""
    num_nodes: int                          # 节点数量
    gpu_memory_gb: List[float]              # 每个GPU的显存大小（GB）
    gpu_tflops: List[float]                 # 每个GPU的计算能力（TFLOPS）
    intra_node_bandwidth: float             # 节点内带宽（GB/s）
    inter_node_bandwidth: float             # 节点间带宽（GB/s）
    network_latency_ms: float               # 网络延迟（ms）
    
    @classmethod
    def from_auto_detect(cls, num_nodes: int) -> 'ClusterConfig':
        """自动检测集群配置"""
        gpu_memory_gb = []
        gpu_tflops = []
        
        for i in range(min(num_nodes, torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(i)
            gpu_memory_gb.append(props.total_memory / (1024**3))
            # 估算计算能力（基于CUDA核心数，假设典型频率）
            # SM数 × 每SM核心数 × 2 (FMA) × 典型频率(~1.5GHz)
            tflops = props.multi_processor_count * 128 * 2 * 1.5  # 粗略估算
            gpu_tflops.append(tflops)
        
        # 如果GPU数量不足，使用平均值填充
        while len(gpu_memory_gb) < num_nodes:
            gpu_memory_gb.append(gpu_memory_gb[-1] if gpu_memory_gb else 8.0)
            gpu_tflops.append(gpu_tflops[-1] if gpu_tflops else 10.0)
        
        return cls(
            num_nodes=num_nodes,
            gpu_memory_gb=gpu_memory_gb,
            gpu_tflops=gpu_tflops,
            intra_node_bandwidth=25.0,      # PCIe 4.0: ~25GB/s
            inter_node_bandwidth=10.0,       # 保守估计
            network_latency_ms=0.5
        )
    
    @property
    def total_memory_gb(self) -> float:
        """总显存"""
        return sum(self.gpu_memory_gb)
    
    @property
    def total_tflops(self) -> float:
        """总算力"""
        return sum(self.gpu_tflops)
    
    @property
    def min_memory_gb(self) -> float:
        """最小单卡显存"""
        return min(self.gpu_memory_gb)


@dataclass
class CostEstimate:
    """代价估算结果"""
    compute_time_ms: float      # 计算时间（ms）
    comm_time_ms: float         # 通信时间（ms）
    memory_gb: float            # 显存需求（GB）
    total_time_ms: float        # 总时间（ms）
    efficiency: float           # 并行效率（0-1）
    speedup: float              # 相对单GPU的加速比
    feasible: bool              # 是否可行（显存是否足够）


@dataclass
class SplitPlan:
    """分割方案"""
    strategy: SplitStrategy
    split_dim: int                          # 主分割维度 (0=row, 1=col)
    chunk_sizes: List[int]                  # 每个节点的块大小
    cost: CostEstimate
    grid_rows: int = 1                      # 2D 分割的网格行数
    grid_cols: int = 1                      # 2D 分割的网格列数


def compute_2d_grid(num_nodes: int) -> Tuple[int, int]:
    """
    计算最接近正方形的 2D 进程网格
    
    Args:
        num_nodes: 总进程数
    
    Returns:
        (grid_rows, grid_cols) 使得 grid_rows * grid_cols == num_nodes
        且尽量接近 sqrt(num_nodes) × sqrt(num_nodes)
    """
    sqrt_n = int(math.sqrt(num_nodes))
    for r in range(sqrt_n, 0, -1):
        if num_nodes % r == 0:
            return r, num_nodes // r
    return 1, num_nodes


class CostModel:
    """
    分布式计算代价模型
    
    核心公式：
    T_total = max(T_compute_i) + T_communication
    
    其中：
    - T_compute_i = FLOP_i / GPU_power_i （第i个节点的计算时间）
    - T_communication = Volume / Bandwidth + Latency
    
    优化目标：
    min T_total
    s.t. Memory_i <= GPU_memory_i for all i
    """
    
    def __init__(self, config: ClusterConfig):
        """
        初始化代价模型
        
        Args:
            config: 集群配置
        """
        self.config = config
        self.num_nodes = config.num_nodes
    
    def estimate_matmul_cost(self, M: int, K: int, N: int,
                             strategy: SplitStrategy,
                             dtype_bytes: int = 4) -> CostEstimate:
        """
        估算矩阵乘法 C[M,N] = A[M,K] @ B[K,N] 的代价
        
        Args:
            M, K, N: 矩阵维度
            strategy: 分割策略
            dtype_bytes: 数据类型字节数
        
        Returns:
            代价估算结果
        """
        total_flops = 2 * M * K * N
        
        # 单GPU计算时间（理论）
        single_gpu_time_ms = total_flops / (self.config.gpu_tflops[0] * 1e12) * 1000
        
        bandwidth = self.config.inter_node_bandwidth * 1e9  # bytes/s
        
        if strategy == SplitStrategy.ROW_SPLIT:
            # 按A的行分割：scatter A, broadcast B, gather C
            local_M = M // self.num_nodes
            
            # 每个节点的计算量
            local_flops = 2 * local_M * K * N
            compute_times = [local_flops / (tflops * 1e12) * 1000 
                           for tflops in self.config.gpu_tflops]
            compute_time_ms = max(compute_times)
            
            # 通信量：scatter A + broadcast B + gather C
            scatter_volume = M * K * dtype_bytes
            broadcast_volume = K * N * dtype_bytes
            gather_volume = M * N * dtype_bytes
            total_comm_volume = scatter_volume + broadcast_volume + gather_volume
            
            # 通信时间
            comm_time_ms = (total_comm_volume / bandwidth) * 1000 + self.config.network_latency_ms * 3
            
            # 显存需求（每个节点）
            memory_gb = (local_M * K + K * N + local_M * N) * dtype_bytes / (1024**3)
            
        elif strategy == SplitStrategy.COLUMN_SPLIT:
            # 按B的列分割：broadcast A, scatter B, gather C
            local_N = N // self.num_nodes
            
            local_flops = 2 * M * K * local_N
            compute_times = [local_flops / (tflops * 1e12) * 1000 
                           for tflops in self.config.gpu_tflops]
            compute_time_ms = max(compute_times)
            
            broadcast_volume = M * K * dtype_bytes
            scatter_volume = K * N * dtype_bytes
            gather_volume = M * N * dtype_bytes
            total_comm_volume = broadcast_volume + scatter_volume + gather_volume
            
            comm_time_ms = (total_comm_volume / bandwidth) * 1000 + self.config.network_latency_ms * 3
            
            memory_gb = (M * K + K * local_N + M * local_N) * dtype_bytes / (1024**3)
            
        elif strategy == SplitStrategy.BLOCK_2D:
            # 2D 块分割：将 A 按行分成 p_r 块，B 按列分成 p_c 块
            # 进程 (i,j) 持有 A_i [local_M, K] 和 B_j [K, local_N]
            # 计算 C_ij = A_i @ B_j → [local_M, local_N]
            grid_rows, grid_cols = compute_2d_grid(self.num_nodes)
            local_M = M // grid_rows
            local_N = N // grid_cols
            
            local_flops = 2 * local_M * K * local_N
            compute_times = [local_flops / (tflops * 1e12) * 1000 
                           for tflops in self.config.gpu_tflops]
            compute_time_ms = max(compute_times)
            
            # 通信量：
            # - 分发 A 行块到各行组（每个行块 local_M*K 发给 grid_cols 个进程）
            #   等效于 scatter A + 行内 broadcast = M*K + (grid_cols-1)*local_M*K
            # - 分发 B 列块到各列组（每个列块 K*local_N 发给 grid_rows 个进程）
            #   等效于 scatter B + 列内 broadcast = K*N + (grid_rows-1)*K*local_N
            # - 收集 C 的所有子块 = M * N
            dist_A_volume = M * K * dtype_bytes  # scatter A 行块
            dist_B_volume = K * N * dtype_bytes   # scatter B 列块
            gather_volume = M * N * dtype_bytes    # gather C 块
            total_comm_volume = dist_A_volume + dist_B_volume + gather_volume
            
            comm_time_ms = (total_comm_volume / bandwidth) * 1000 + self.config.network_latency_ms * 4
            
            # 显存需求（每个节点）：大幅降低！
            memory_gb = (local_M * K + K * local_N + local_M * local_N) * dtype_bytes / (1024**3)
            
        else:
            # 默认使用行分割
            return self.estimate_matmul_cost(M, K, N, SplitStrategy.ROW_SPLIT, dtype_bytes)
        
        total_time_ms = compute_time_ms + comm_time_ms
        speedup = single_gpu_time_ms / total_time_ms if total_time_ms > 0 else 0
        efficiency = speedup / self.num_nodes
        feasible = memory_gb <= self.config.min_memory_gb
        
        return CostEstimate(
            compute_time_ms=compute_time_ms,
            comm_time_ms=comm_time_ms,
            memory_gb=memory_gb,
            total_time_ms=total_time_ms,
            efficiency=efficiency,
            speedup=speedup,
            feasible=feasible
        )
    
    def find_optimal_strategy(self, M: int, K: int, N: int,
                              dtype_bytes: int = 4) -> SplitPlan:
        """
        寻找最优分割策略
        
        遍历所有可用策略（行分割、列分割、2D块分割），
        选择总时间最短且显存可行的方案。
        
        Args:
            M, K, N: 矩阵维度
            dtype_bytes: 数据类型字节数
        
        Returns:
            最优分割方案
        """
        candidates = [SplitStrategy.ROW_SPLIT, SplitStrategy.COLUMN_SPLIT]
        
        # 2D 块分割要求进程数可分解为 ≥2 的网格
        grid_r, grid_c = compute_2d_grid(self.num_nodes)
        if grid_r >= 2 and grid_c >= 2 and M >= grid_r and N >= grid_c:
            candidates.append(SplitStrategy.BLOCK_2D)
        
        best_plan = None
        best_time = float('inf')
        
        for strategy in candidates:
            cost = self.estimate_matmul_cost(M, K, N, strategy, dtype_bytes)
            
            if cost.feasible and cost.total_time_ms < best_time:
                best_time = cost.total_time_ms
                
                if strategy == SplitStrategy.ROW_SPLIT:
                    chunk_sizes = self._calculate_chunk_sizes(M)
                    split_dim = 0
                    gr, gc = self.num_nodes, 1
                elif strategy == SplitStrategy.COLUMN_SPLIT:
                    chunk_sizes = self._calculate_chunk_sizes(N)
                    split_dim = 1
                    gr, gc = 1, self.num_nodes
                else:  # BLOCK_2D
                    chunk_sizes = self._calculate_chunk_sizes(M)  # 行方向
                    split_dim = -1  # 特殊标记：两个维度都分割
                    gr, gc = grid_r, grid_c
                
                best_plan = SplitPlan(
                    strategy=strategy,
                    split_dim=split_dim,
                    chunk_sizes=chunk_sizes,
                    cost=cost,
                    grid_rows=gr,
                    grid_cols=gc,
                )
        
        if best_plan is None:
            # 如果没有可行方案，返回行分割（即使显存不足）
            cost = self.estimate_matmul_cost(M, K, N, SplitStrategy.ROW_SPLIT, dtype_bytes)
            best_plan = SplitPlan(
                strategy=SplitStrategy.ROW_SPLIT,
                split_dim=0,
                chunk_sizes=self._calculate_chunk_sizes(M),
                cost=cost,
                grid_rows=self.num_nodes,
                grid_cols=1,
            )
        
        return best_plan
    
    def _calculate_chunk_sizes(self, total_size: int) -> List[int]:
        """计算每个节点的块大小"""
        chunk_size = total_size // self.num_nodes
        remainder = total_size % self.num_nodes
        
        sizes = []
        for i in range(self.num_nodes):
            size = chunk_size + (1 if i < remainder else 0)
            sizes.append(size)
        
        return sizes
    
    def estimate_memory_requirement(self, *shapes, dtype_bytes: int = 4) -> float:
        """
        估算显存需求
        
        Args:
            *shapes: 张量形状列表
            dtype_bytes: 数据类型字节数
        
        Returns:
            显存需求（GB）
        """
        total_bytes = 0
        for shape in shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total_bytes += elements * dtype_bytes
        
        return total_bytes / (1024**3)
    
    def should_use_distributed(self, *shapes, dtype_bytes: int = 4) -> bool:
        """
        判断是否应该使用分布式计算
        
        Args:
            *shapes: 张量形状列表
            dtype_bytes: 数据类型字节数
        
        Returns:
            是否应该使用分布式
        """
        memory_needed = self.estimate_memory_requirement(*shapes, dtype_bytes=dtype_bytes)
        return memory_needed > self.config.min_memory_gb * 0.8  # 80%显存阈值
    
    def print_analysis(self, M: int, K: int, N: int):
        """打印分析结果"""
        print(f"\n矩阵乘法分析: [{M}, {K}] @ [{K}, {N}]")
        print(f"集群配置: {self.num_nodes} 节点")
        print("-" * 70)
        
        strategies = [SplitStrategy.ROW_SPLIT, SplitStrategy.COLUMN_SPLIT]
        grid_r, grid_c = compute_2d_grid(self.num_nodes)
        if grid_r >= 2 and grid_c >= 2:
            strategies.append(SplitStrategy.BLOCK_2D)
        
        for strategy in strategies:
            cost = self.estimate_matmul_cost(M, K, N, strategy)
            status = "✓" if cost.feasible else "✗"
            label = strategy.value
            if strategy == SplitStrategy.BLOCK_2D:
                label += f"({grid_r}x{grid_c})"
            print(f"{label:16s} | 计算: {cost.compute_time_ms:8.2f}ms | "
                  f"通信: {cost.comm_time_ms:8.2f}ms | "
                  f"总计: {cost.total_time_ms:8.2f}ms | "
                  f"显存: {cost.memory_gb:6.3f}GB | "
                  f"效率: {cost.efficiency*100:5.1f}% | {status}")
        
        best = self.find_optimal_strategy(M, K, N)
        print(f"\n推荐策略: {best.strategy.value}", end="")
        if best.strategy == SplitStrategy.BLOCK_2D:
            print(f" (网格 {best.grid_rows}×{best.grid_cols})", end="")
        print(f"\n预估加速比: {best.cost.speedup:.2f}x")
        print(f"每卡显存需求: {best.cost.memory_gb:.3f} GB")
