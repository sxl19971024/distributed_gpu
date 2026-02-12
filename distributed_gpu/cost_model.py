"""
基于代价模型的自适应张量分割算法

创新点1：提出一种基于计算代价、通信代价和显存代价联合优化的自适应分割策略。

理论贡献：
1. 建立了分布式张量计算的代价模型
2. 提出了多目标优化的分割策略选择算法
3. 设计了自适应的负载均衡机制
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ── LRU 缓存大小限制 ──
_COST_CACHE_MAX_SIZE: int = 256


# ---------------------------------------------------------------------------
# 枚举 & 数据类
# ---------------------------------------------------------------------------

class SplitStrategy(Enum):
    """分割策略类型"""
    ROW_SPLIT = "row"           # 按行分割 A
    COLUMN_SPLIT = "column"     # 按列分割 B
    BLOCK_2D = "block_2d"       # 2D块分割（SUMMA）
    BATCH_SPLIT = "batch"       # 按批次分割


@dataclass
class ClusterConfig:
    """集群配置信息

    Attributes:
        num_nodes: 节点（进程）数量
        gpu_memory_gb: 每个 GPU 的显存大小（GB）
        gpu_tflops: 每个 GPU 的计算能力（TFLOPS）
        intra_node_bandwidth: 节点内带宽（GB/s）
        inter_node_bandwidth: 节点间带宽（GB/s）
        network_latency_ms: 网络延迟（ms）
    """
    num_nodes: int
    gpu_memory_gb: List[float]
    gpu_tflops: List[float]
    intra_node_bandwidth: float
    inter_node_bandwidth: float
    network_latency_ms: float

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def from_auto_detect(cls, num_nodes: int) -> ClusterConfig:
        """自动检测集群配置

        Args:
            num_nodes: 期望的节点数量

        Returns:
            自动填充的 ClusterConfig 实例
        """
        gpu_memory_gb: List[float] = []
        gpu_tflops: List[float] = []

        for i in range(min(num_nodes, torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(i)
            gpu_memory_gb.append(props.total_memory / (1024 ** 3))
            # 粗略估算：SM数 × 每SM核心数 × 2 (FMA) × 典型频率(~1.5GHz)
            tflops = props.multi_processor_count * 128 * 2 * 1.5
            gpu_tflops.append(tflops)

        # 如果 GPU 数量不足，使用最后一个值填充
        default_mem = gpu_memory_gb[-1] if gpu_memory_gb else 8.0
        default_tflops = gpu_tflops[-1] if gpu_tflops else 10.0
        while len(gpu_memory_gb) < num_nodes:
            gpu_memory_gb.append(default_mem)
            gpu_tflops.append(default_tflops)

        return cls(
            num_nodes=num_nodes,
            gpu_memory_gb=gpu_memory_gb,
            gpu_tflops=gpu_tflops,
            intra_node_bandwidth=25.0,   # PCIe 4.0: ~25 GB/s
            inter_node_bandwidth=10.0,   # 保守估计
            network_latency_ms=0.5,
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def total_memory_gb(self) -> float:
        """总显存（GB）"""
        return sum(self.gpu_memory_gb)

    @property
    def total_tflops(self) -> float:
        """总算力（TFLOPS）"""
        return sum(self.gpu_tflops)

    @property
    def min_memory_gb(self) -> float:
        """最小单卡显存（GB）"""
        return min(self.gpu_memory_gb)


@dataclass
class CostEstimate:
    """代价估算结果

    Attributes:
        compute_time_ms: 计算时间（ms）
        comm_time_ms: 通信时间（ms）
        memory_gb: 每节点显存需求（GB）
        total_time_ms: 总时间（ms）
        efficiency: 并行效率（0-1）
        speedup: 相对单 GPU 的加速比
        feasible: 是否可行（显存是否足够）
    """
    compute_time_ms: float
    comm_time_ms: float
    memory_gb: float
    total_time_ms: float
    efficiency: float
    speedup: float
    feasible: bool


@dataclass
class SplitPlan:
    """分割方案

    Attributes:
        strategy: 分割策略
        split_dim: 主分割维度（0=row, 1=col, -1=2D）
        chunk_sizes: 每个节点的块大小
        cost: 代价估算
        grid_rows: 2D 分割的网格行数
        grid_cols: 2D 分割的网格列数
    """
    strategy: SplitStrategy
    split_dim: int
    chunk_sizes: List[int]
    cost: CostEstimate
    grid_rows: int = 1
    grid_cols: int = 1


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def compute_2d_grid(num_nodes: int) -> Tuple[int, int]:
    """计算最接近正方形的 2D 进程网格

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


# ---------------------------------------------------------------------------
# 核心代价模型
# ---------------------------------------------------------------------------

class CostModel:
    """分布式计算代价模型

    核心公式::

        T_total = max(T_compute_i) + T_communication

    其中:
        - T_compute_i = FLOP_i / GPU_power_i （第 i 个节点的计算时间）
        - T_communication = Volume / Bandwidth + Latency

    优化目标::

        min  T_total
        s.t. Memory_i <= GPU_memory_i  ∀ i
    """

    def __init__(self, config: ClusterConfig) -> None:
        """
        Args:
            config: 集群配置
        """
        self.config: ClusterConfig = config
        self.num_nodes: int = config.num_nodes

        # LRU 缓存：避免重复计算相同维度的代价估算
        # key = (M, K, N, strategy, dtype_bytes) → CostEstimate
        self._cost_cache: OrderedDict[Tuple, CostEstimate] = OrderedDict()
        # key = (M, K, N, dtype_bytes) → SplitPlan
        self._plan_cache: OrderedDict[Tuple, SplitPlan] = OrderedDict()

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """清空代价估算和策略搜索的 LRU 缓存。"""
        self._cost_cache.clear()
        self._plan_cache.clear()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _max_compute_time(self, local_flops: float) -> float:
        """计算各节点中最慢的计算时间（ms）

        假设各节点分配到相同的 FLOP，但计算能力不同。
        """
        return max(
            local_flops / (tflops * 1e12) * 1000
            for tflops in self.config.gpu_tflops
        )

    def _comm_time(self, volume_bytes: float, num_ops: int = 1) -> float:
        """估算通信时间（ms）

        Args:
            volume_bytes: 总通信数据量（字节）
            num_ops: 通信操作次数（用于累加延迟）
        """
        bandwidth_bps = self.config.inter_node_bandwidth * 1e9  # bytes/s
        return (volume_bytes / bandwidth_bps) * 1000 + self.config.network_latency_ms * num_ops

    @staticmethod
    def _memory_gb(num_elements: int, dtype_bytes: int) -> float:
        """将元素数量转换为 GB"""
        return num_elements * dtype_bytes / (1024 ** 3)

    def _build_estimate(
        self,
        compute_time_ms: float,
        comm_time_ms: float,
        memory_gb: float,
        single_gpu_time_ms: float,
    ) -> CostEstimate:
        """根据计算/通信/显存构建 CostEstimate"""
        total_time_ms = compute_time_ms + comm_time_ms
        speedup = single_gpu_time_ms / total_time_ms if total_time_ms > 0 else 0.0
        efficiency = speedup / self.num_nodes
        feasible = memory_gb <= self.config.min_memory_gb
        return CostEstimate(
            compute_time_ms=compute_time_ms,
            comm_time_ms=comm_time_ms,
            memory_gb=memory_gb,
            total_time_ms=total_time_ms,
            efficiency=efficiency,
            speedup=speedup,
            feasible=feasible,
        )

    # ------------------------------------------------------------------
    # 矩阵乘法代价估算
    # ------------------------------------------------------------------

    def estimate_matmul_cost(
        self,
        M: int,
        K: int,
        N: int,
        strategy: SplitStrategy,
        dtype_bytes: int = 4,
    ) -> CostEstimate:
        """估算矩阵乘法 C[M,N] = A[M,K] @ B[K,N] 的代价

        使用 LRU 缓存避免对相同维度和策略的重复计算。

        Args:
            M, K, N: 矩阵维度
            strategy: 分割策略
            dtype_bytes: 数据类型字节数（默认 float32 = 4）

        Returns:
            CostEstimate 实例
        """
        cache_key = (M, K, N, strategy, dtype_bytes)

        # 查询缓存（命中时移到末尾，实现 LRU）
        if cache_key in self._cost_cache:
            self._cost_cache.move_to_end(cache_key)
            return self._cost_cache[cache_key]

        total_flops = 2.0 * M * K * N
        single_gpu_time_ms = total_flops / (self.config.gpu_tflops[0] * 1e12) * 1000

        if strategy == SplitStrategy.ROW_SPLIT:
            result = self._estimate_row_split(M, K, N, dtype_bytes, total_flops, single_gpu_time_ms)
        elif strategy == SplitStrategy.COLUMN_SPLIT:
            result = self._estimate_col_split(M, K, N, dtype_bytes, total_flops, single_gpu_time_ms)
        elif strategy == SplitStrategy.BLOCK_2D:
            result = self._estimate_block_2d(M, K, N, dtype_bytes, total_flops, single_gpu_time_ms)
        else:
            result = self._estimate_row_split(M, K, N, dtype_bytes, total_flops, single_gpu_time_ms)

        # 写入缓存（超限时淘汰最旧条目）
        self._cost_cache[cache_key] = result
        if len(self._cost_cache) > _COST_CACHE_MAX_SIZE:
            self._cost_cache.popitem(last=False)

        return result

    def _estimate_row_split(
        self, M: int, K: int, N: int,
        dtype_bytes: int, total_flops: float, single_gpu_time_ms: float,
    ) -> CostEstimate:
        """按行分割 A：scatter A, broadcast B, gather C"""
        local_M = M // self.num_nodes
        local_flops = 2.0 * local_M * K * N
        compute_time_ms = self._max_compute_time(local_flops)

        # 通信量：scatter A + broadcast B + gather C
        comm_volume = (M * K + K * N + M * N) * dtype_bytes
        comm_time_ms = self._comm_time(comm_volume, num_ops=3)

        memory_gb = self._memory_gb(local_M * K + K * N + local_M * N, dtype_bytes)
        return self._build_estimate(compute_time_ms, comm_time_ms, memory_gb, single_gpu_time_ms)

    def _estimate_col_split(
        self, M: int, K: int, N: int,
        dtype_bytes: int, total_flops: float, single_gpu_time_ms: float,
    ) -> CostEstimate:
        """按列分割 B：broadcast A, scatter B, gather C"""
        local_N = N // self.num_nodes
        local_flops = 2.0 * M * K * local_N
        compute_time_ms = self._max_compute_time(local_flops)

        comm_volume = (M * K + K * N + M * N) * dtype_bytes
        comm_time_ms = self._comm_time(comm_volume, num_ops=3)

        memory_gb = self._memory_gb(M * K + K * local_N + M * local_N, dtype_bytes)
        return self._build_estimate(compute_time_ms, comm_time_ms, memory_gb, single_gpu_time_ms)

    def _estimate_block_2d(
        self, M: int, K: int, N: int,
        dtype_bytes: int, total_flops: float, single_gpu_time_ms: float,
    ) -> CostEstimate:
        """2D 块分割（SUMMA 风格）"""
        grid_rows, grid_cols = compute_2d_grid(self.num_nodes)
        local_M = M // grid_rows
        local_N = N // grid_cols
        local_flops = 2.0 * local_M * K * local_N
        compute_time_ms = self._max_compute_time(local_flops)

        # 通信量：distribute A 行块 + distribute B 列块 + gather C
        comm_volume = (M * K + K * N + M * N) * dtype_bytes
        comm_time_ms = self._comm_time(comm_volume, num_ops=4)

        memory_gb = self._memory_gb(local_M * K + K * local_N + local_M * local_N, dtype_bytes)
        return self._build_estimate(compute_time_ms, comm_time_ms, memory_gb, single_gpu_time_ms)

    # ------------------------------------------------------------------
    # 最优策略搜索
    # ------------------------------------------------------------------

    def find_optimal_strategy(
        self,
        M: int,
        K: int,
        N: int,
        dtype_bytes: int = 4,
    ) -> SplitPlan:
        """寻找最优分割策略

        遍历所有可用策略（行分割、列分割、2D 块分割），
        选择总时间最短且显存可行的方案。
        使用 LRU 缓存避免重复搜索。

        Args:
            M, K, N: 矩阵维度
            dtype_bytes: 数据类型字节数

        Returns:
            最优 SplitPlan
        """
        plan_key = (M, K, N, dtype_bytes)
        if plan_key in self._plan_cache:
            self._plan_cache.move_to_end(plan_key)
            return self._plan_cache[plan_key]

        candidates: List[SplitStrategy] = [
            SplitStrategy.ROW_SPLIT,
            SplitStrategy.COLUMN_SPLIT,
        ]

        grid_r, grid_c = compute_2d_grid(self.num_nodes)
        if grid_r >= 2 and grid_c >= 2 and M >= grid_r and N >= grid_c:
            candidates.append(SplitStrategy.BLOCK_2D)

        best_plan: Optional[SplitPlan] = None
        best_time: float = float("inf")

        for strategy in candidates:
            cost = self.estimate_matmul_cost(M, K, N, strategy, dtype_bytes)
            if not cost.feasible or cost.total_time_ms >= best_time:
                continue

            best_time = cost.total_time_ms
            split_dim, gr, gc = self._strategy_metadata(strategy, grid_r, grid_c)
            dim_size = M if strategy != SplitStrategy.COLUMN_SPLIT else N

            best_plan = SplitPlan(
                strategy=strategy,
                split_dim=split_dim,
                chunk_sizes=self._calculate_chunk_sizes(dim_size),
                cost=cost,
                grid_rows=gr,
                grid_cols=gc,
            )

        # 如果没有可行方案，回退到行分割
        if best_plan is None:
            cost = self.estimate_matmul_cost(M, K, N, SplitStrategy.ROW_SPLIT, dtype_bytes)
            best_plan = SplitPlan(
                strategy=SplitStrategy.ROW_SPLIT,
                split_dim=0,
                chunk_sizes=self._calculate_chunk_sizes(M),
                cost=cost,
                grid_rows=self.num_nodes,
                grid_cols=1,
            )

        # 写入 LRU 缓存
        self._plan_cache[plan_key] = best_plan
        if len(self._plan_cache) > _COST_CACHE_MAX_SIZE:
            self._plan_cache.popitem(last=False)

        return best_plan

    @staticmethod
    def _strategy_metadata(
        strategy: SplitStrategy,
        grid_r: int,
        grid_c: int,
    ) -> Tuple[int, int, int]:
        """返回 (split_dim, grid_rows, grid_cols)"""
        if strategy == SplitStrategy.ROW_SPLIT:
            return 0, grid_r * grid_c, 1  # num_nodes, 1
        elif strategy == SplitStrategy.COLUMN_SPLIT:
            return 1, 1, grid_r * grid_c
        else:  # BLOCK_2D
            return -1, grid_r, grid_c

    def _calculate_chunk_sizes(self, total_size: int) -> List[int]:
        """计算每个节点的块大小（处理不整除情况）

        Args:
            total_size: 要分割的总大小

        Returns:
            长度为 num_nodes 的列表
        """
        chunk, remainder = divmod(total_size, self.num_nodes)
        return [chunk + (1 if i < remainder else 0) for i in range(self.num_nodes)]

    # ------------------------------------------------------------------
    # 通用工具方法
    # ------------------------------------------------------------------

    def estimate_memory_requirement(
        self,
        *shapes: Sequence[int],
        dtype_bytes: int = 4,
    ) -> float:
        """估算多个张量的总显存需求

        Args:
            *shapes: 张量形状列表
            dtype_bytes: 数据类型字节数

        Returns:
            显存需求（GB）
        """
        total_elements = 0
        for shape in shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total_elements += elements
        return self._memory_gb(total_elements, dtype_bytes)

    def should_use_distributed(
        self,
        *shapes: Sequence[int],
        dtype_bytes: int = 4,
    ) -> bool:
        """判断是否应该使用分布式计算

        当显存需求超过最小单卡显存的 80% 时建议使用分布式。

        Args:
            *shapes: 张量形状列表
            dtype_bytes: 数据类型字节数

        Returns:
            是否应该使用分布式
        """
        memory_needed = self.estimate_memory_requirement(*shapes, dtype_bytes=dtype_bytes)
        return memory_needed > self.config.min_memory_gb * 0.8

    # ------------------------------------------------------------------
    # 诊断输出
    # ------------------------------------------------------------------

    def print_analysis(self, M: int, K: int, N: int) -> None:
        """打印矩阵乘法代价分析结果

        Args:
            M, K, N: 矩阵维度
        """
        print(f"\n矩阵乘法分析: [{M}, {K}] @ [{K}, {N}]")
        print(f"集群配置: {self.num_nodes} 节点")
        print("-" * 70)

        strategies: List[SplitStrategy] = [
            SplitStrategy.ROW_SPLIT,
            SplitStrategy.COLUMN_SPLIT,
        ]
        grid_r, grid_c = compute_2d_grid(self.num_nodes)
        if grid_r >= 2 and grid_c >= 2:
            strategies.append(SplitStrategy.BLOCK_2D)

        for strategy in strategies:
            cost = self.estimate_matmul_cost(M, K, N, strategy)
            status = "✓" if cost.feasible else "✗"
            label = strategy.value
            if strategy == SplitStrategy.BLOCK_2D:
                label += f"({grid_r}x{grid_c})"
            print(
                f"{label:16s} | 计算: {cost.compute_time_ms:8.2f}ms | "
                f"通信: {cost.comm_time_ms:8.2f}ms | "
                f"总计: {cost.total_time_ms:8.2f}ms | "
                f"显存: {cost.memory_gb:6.3f}GB | "
                f"效率: {cost.efficiency * 100:5.1f}% | {status}"
            )

        best = self.find_optimal_strategy(M, K, N)
        print(f"\n推荐策略: {best.strategy.value}", end="")
        if best.strategy == SplitStrategy.BLOCK_2D:
            print(f" (网格 {best.grid_rows}×{best.grid_cols})", end="")
        print(f"\n预估加速比: {best.cost.speedup:.2f}x")
        print(f"每卡显存需求: {best.cost.memory_gb:.3f} GB")
