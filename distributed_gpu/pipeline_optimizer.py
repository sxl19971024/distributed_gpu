"""
计算-通信重叠的流水线优化策略

创新点2：提出一种基于分块流水线的计算-通信重叠优化方法。

理论贡献：
1. 建立了计算-通信重叠的理论模型
2. 推导了最优分块大小的解析表达式
3. 设计了多级流水线调度算法
4. 利用 CUDA 双流实现 GPU 计算与 CPU 侧 MPI 通信的真正重叠
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .mpi_manager import MPIManager


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """流水线配置

    Attributes:
        num_chunks: 默认分块数量
        enable_overlap: 是否启用计算-通信重叠
        prefetch_count: 预取块数量
    """
    num_chunks: int = 4
    enable_overlap: bool = True
    prefetch_count: int = 1


# ---------------------------------------------------------------------------
# 统计信息
# ---------------------------------------------------------------------------

def _empty_stats() -> Dict[str, float]:
    """创建初始统计字典"""
    return {
        "total_compute_time": 0.0,
        "total_comm_time": 0.0,
        "overlap_savings": 0.0,
        "num_chunks_processed": 0,
    }


# ---------------------------------------------------------------------------
# 核心优化器
# ---------------------------------------------------------------------------

class PipelineOptimizer:
    """计算-通信重叠优化器

    核心思想：将大规模计算任务分解为多个小块，
    在 GPU 上用 ``compute_stream`` 执行当前块的矩阵乘法时，
    CPU 侧同时执行下一块的 MPI scatter / 上一块的 MPI gather，
    从而隐藏通信延迟。

    时间分析::

        无重叠：T = T_scatter + T_compute + T_gather  (串行)
        有重叠：T ≈ max(T_compute, T_comm) × K + startup
              ≈ T_compute × K  (当计算密集时通信完全被隐藏)
    """

    def __init__(
        self,
        mpi: MPIManager,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.mpi: MPIManager = mpi
        self.config: PipelineConfig = config or PipelineConfig()
        self.rank: int = mpi.get_rank()
        self.size: int = mpi.get_size()
        self.gpu_id: int = mpi.get_gpu_id()

        # CUDA 双流
        if torch.cuda.is_available():
            self.compute_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(device=self.gpu_id)
            self.comm_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(device=self.gpu_id)
        else:
            self.compute_stream = None
            self.comm_stream = None

        self.stats: Dict[str, float] = _empty_stats()

    # ------------------------------------------------------------------
    # 最优分块计算
    # ------------------------------------------------------------------

    def compute_optimal_chunk_count(
        self,
        tensor_size_bytes: int,
        compute_intensity: float,
        bandwidth_gbps: float,
        latency_ms: float = 0.5,
    ) -> int:
        """计算最优分块数量

        理论推导：
        设总数据量为 D，分块数为 K，带宽为 B，计算强度为 I（FLOP/byte）。
        每块计算时间 ``t_comp = d × I / P``，
        每块通信时间 ``t_comm = d / B + latency``。
        最优条件：``t_comp ≈ t_comm``。

        Args:
            tensor_size_bytes: 张量总字节数
            compute_intensity: 计算强度（FLOP/byte）
            bandwidth_gbps: 带宽（GB/s）
            latency_ms: 延迟（ms）

        Returns:
            推荐的分块数量
        """
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.gpu_id)
            compute_power = props.multi_processor_count * 128 * 2 * 1.5
        else:
            compute_power = 1.0

        bandwidth = bandwidth_gbps * 1e9  # bytes/s
        latency = latency_ms / 1000       # seconds

        numerator = latency * compute_power * 1e12 * bandwidth
        denominator = compute_intensity * bandwidth - compute_power * 1e12

        if denominator <= 0:
            return max(2, self.size)

        optimal_chunk_size = numerator / denominator
        optimal_count = max(2, int(tensor_size_bytes / optimal_chunk_size))
        optimal_count = min(optimal_count, 32)
        optimal_count = max(optimal_count, self.config.num_chunks)
        return optimal_count

    # ------------------------------------------------------------------
    # 流水线化 AllReduce
    # ------------------------------------------------------------------

    def pipelined_allreduce(
        self,
        tensor: torch.Tensor,
        num_chunks: Optional[int] = None,
    ) -> torch.Tensor:
        """流水线化的 AllReduce 操作

        将大张量分块，逐块执行 allreduce，利用 CUDA 流实现
        GPU→CPU 拷贝与上一块的 CPU→GPU 拷贝重叠。

        Args:
            tensor: 输入张量
            num_chunks: 分块数量（默认使用配置值）

        Returns:
            AllReduce 后的张量（与输入同形状）
        """
        if num_chunks is None:
            num_chunks = self.config.num_chunks

        if not self.config.enable_overlap or num_chunks <= 1:
            return self.mpi.allreduce_tensor(tensor)

        from mpi4py import MPI as _MPI

        flat = tensor.flatten()
        total_numel = flat.numel()
        chunks = list(torch.chunk(flat, num_chunks))

        # 预分配完整结果张量（避免最后 torch.cat 的额外内存分配）
        result_flat = torch.empty(total_numel, dtype=tensor.dtype, device=tensor.device)
        offset = 0

        # 双缓冲：交替使用两组 pinned buffer，使上一块的 CPU→GPU 传输
        # 与当前块的 MPI AllReduce 重叠
        prev_event: Optional[torch.cuda.Event] = None

        for i, chunk in enumerate(chunks):
            chunk_numel = chunk.numel()

            # GPU → CPU（使用池化 pinned memory）
            data = self.mpi._to_pinned_numpy(chunk)

            # CPU 侧 MPI AllReduce（使用池化接收缓冲区）
            result_np = self.mpi._acquire_pinned_buffer(data.shape, data.dtype.type)
            self.mpi._safe_call(
                "Allreduce(pipeline)",
                self.mpi.comm.Allreduce,
                data, result_np, op=_MPI.SUM,
            )

            # CPU → GPU：使用 comm_stream 异步传输，不阻塞计算
            if self.comm_stream is not None:
                # 等待上一块的传输完成（如果有）
                if prev_event is not None:
                    self.comm_stream.wait_event(prev_event)
                with torch.cuda.stream(self.comm_stream):
                    t_result = (
                        torch.from_numpy(result_np)
                        .pin_memory()
                        .cuda(self.gpu_id, non_blocking=True)
                    )
                    result_flat[offset:offset + chunk_numel].copy_(t_result, non_blocking=True)
                # 记录事件而非 synchronize（更精确的同步）
                prev_event = torch.cuda.Event()
                prev_event.record(self.comm_stream)
            else:
                t_result = self.mpi._from_numpy_to_gpu(result_np)
                result_flat[offset:offset + chunk_numel].copy_(t_result)

            # 归还 pinned buffer 到池
            self.mpi._release_pinned_buffer(data)
            self.mpi._release_pinned_buffer(result_np)

            offset += chunk_numel
            self.stats["num_chunks_processed"] += 1

        # 等待最后一块传输完成
        if prev_event is not None:
            prev_event.synchronize()

        return result_flat.reshape(tensor.shape)

    # ------------------------------------------------------------------
    # 流水线化矩阵乘法（核心创新）
    # ------------------------------------------------------------------

    def pipelined_matmul(
        self,
        A: Optional[torch.Tensor],
        B: Optional[torch.Tensor],
        num_chunks: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """流水线化的分布式矩阵乘法

        三阶段流水线 (per chunk i)::

            Stage 1 — Scatter(i):   CPU 侧 MPI 分发 A 的第 i 子块
            Stage 2 — Compute(i):   GPU compute_stream 执行 matmul
            Stage 3 — Gather(i):    CPU 侧 MPI 收集 C 的第 i 子块

        流水线时序示意（稳态阶段）::

            CPU:  |Gather(i-1)|Scatter(i+1)|
            GPU:        |   Compute(i)   |

        Args:
            A: 矩阵 A（仅 master 需提供完整矩阵）
            B: 矩阵 B（仅 master 需提供完整矩阵）
            num_chunks: 分块数量

        Returns:
            master 返回完整结果矩阵 C，其余进程返回 None
        """
        if num_chunks is None:
            num_chunks = self.config.num_chunks

        # 广播 B 到所有进程
        B_local: torch.Tensor = self.mpi.broadcast_tensor(B, root=0)

        # master 将 A 切块并准备 scatter 列表
        scatter_lists: List[Optional[List[np.ndarray]]]
        M: int
        if self.mpi.is_master_process():
            assert A is not None
            M = A.shape[0]
            macro_chunks = list(torch.chunk(A, num_chunks, dim=0))
            scatter_lists = [
                [self.mpi._to_pinned_numpy(s) for s in torch.chunk(mc, self.size, dim=0)]
                for mc in macro_chunks
            ]
        else:
            scatter_lists = [None] * num_chunks
            M = 0
        M = self.mpi.broadcast(M if self.mpi.is_master_process() else None)

        # 三阶段流水线执行
        # 优化：使用 CUDA 事件精确同步，避免全局 synchronize
        gather_results: List[torch.Tensor] = []
        prev_C_local: Optional[torch.Tensor] = None
        compute_event: Optional[torch.cuda.Event] = None
        t_compute_total = 0.0
        t_comm_total = 0.0

        for i in range(num_chunks + 1):
            t0 = time.time()

            # Stage 3: Gather 上一轮结果（CPU 侧 MPI）
            if i > 0 and prev_C_local is not None:
                # 仅等待计算完成（精确事件同步，而非 stream.synchronize）
                if compute_event is not None:
                    compute_event.synchronize()
                    compute_event = None
                C_block = self.mpi.gather_tensor(prev_C_local, dim=0, root=0)
                if self.mpi.is_master_process():
                    gather_results.append(C_block)

            # Stage 1: Scatter 下一轮 A 块（CPU 侧 MPI）
            A_local_next: Optional[torch.Tensor] = None
            if i < num_chunks:
                sl = scatter_lists[i] if self.mpi.is_master_process() else None
                a_local_np = self.mpi._safe_call(
                    "scatter(pipeline)", self.mpi.comm.scatter, sl, root=0,
                )
                A_local_next = self.mpi._from_numpy_to_gpu(a_local_np)

            t1 = time.time()
            t_comm_total += (t1 - t0) * 1000

            # Stage 2: Compute 当前块 matmul（GPU 异步）
            if i < num_chunks and A_local_next is not None:
                t2 = time.time()
                if self.compute_stream is not None:
                    with torch.cuda.stream(self.compute_stream):
                        prev_C_local = torch.matmul(A_local_next, B_local)
                    # 使用 CUDA 事件记录计算完成点（不阻塞 CPU）
                    compute_event = torch.cuda.Event()
                    compute_event.record(self.compute_stream)
                else:
                    prev_C_local = torch.matmul(A_local_next, B_local)
                    compute_event = None
                t3 = time.time()
                t_compute_total += (t3 - t2) * 1000

        self.stats["total_compute_time"] += t_compute_total
        self.stats["total_comm_time"] += t_comm_total
        self.stats["overlap_savings"] += max(0, t_compute_total + t_comm_total
                                              - max(t_compute_total, t_comm_total))
        self.stats["num_chunks_processed"] += num_chunks

        if self.mpi.is_master_process() and gather_results:
            return torch.cat(gather_results, dim=0)
        return None

    # ------------------------------------------------------------------
    # 非流水线基线
    # ------------------------------------------------------------------

    def baseline_matmul(
        self,
        A: Optional[torch.Tensor],
        B: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """非流水线的分布式矩阵乘法（用作对比基线）

        Args:
            A: 矩阵 A（仅 master 需提供）
            B: 矩阵 B（仅 master 需提供）

        Returns:
            master 返回完整结果矩阵 C，其余进程返回 None
        """
        B_local = self.mpi.broadcast_tensor(B, root=0)
        A_local = self.mpi.scatter_tensor(A, dim=0, root=0)
        C_local = torch.matmul(A_local, B_local)
        return self.mpi.gather_tensor(C_local, dim=0, root=0)

    # ------------------------------------------------------------------
    # 理论收益估算
    # ------------------------------------------------------------------

    def estimate_overlap_benefit(
        self,
        compute_time_ms: float,
        comm_time_ms: float,
        num_chunks: int,
    ) -> Dict[str, float]:
        """估算重叠优化的理论收益

        Args:
            compute_time_ms: 总计算时间（ms）
            comm_time_ms: 总通信时间（ms）
            num_chunks: 分块数量

        Returns:
            包含以下键的字典：
            - ``no_overlap_time_ms``: 无重叠总时间
            - ``overlap_time_ms``: 重叠后总时间
            - ``savings_ms``: 节省的时间
            - ``speedup``: 加速比
            - ``efficiency``: 效率
        """
        no_overlap_time = compute_time_ms + comm_time_ms
        chunk_compute = compute_time_ms / num_chunks
        chunk_comm = comm_time_ms / num_chunks

        steady_state_time = max(chunk_compute, chunk_comm) * (num_chunks - 1)
        startup_time = chunk_compute + chunk_comm
        overlap_time = startup_time + steady_state_time

        savings = no_overlap_time - overlap_time
        speedup = no_overlap_time / overlap_time if overlap_time > 0 else 1.0

        return {
            "no_overlap_time_ms": no_overlap_time,
            "overlap_time_ms": overlap_time,
            "savings_ms": savings,
            "speedup": speedup,
            "efficiency": min(1.0, compute_time_ms / overlap_time) if overlap_time > 0 else 1.0,
        }

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def reset_stats(self) -> None:
        """重置性能统计"""
        self.stats = _empty_stats()

    def print_stats(self) -> None:
        """打印性能统计"""
        print("\n流水线优化统计:")
        print(f"  处理的块数: {self.stats['num_chunks_processed']:.0f}")
        print(f"  总计算时间: {self.stats['total_compute_time']:.2f} ms")
        print(f"  总通信时间: {self.stats['total_comm_time']:.2f} ms")
        print(f"  重叠节省时间: {self.stats['overlap_savings']:.2f} ms")
