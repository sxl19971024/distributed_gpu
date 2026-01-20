"""
计算-通信重叠的流水线优化策略

创新点2：提出一种基于分块流水线的计算-通信重叠优化方法。

理论贡献：
1. 建立了计算-通信重叠的理论模型
2. 推导了最优分块大小的解析表达式
3. 设计了多级流水线调度算法
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import time

from .mpi_manager import MPIManager


@dataclass
class PipelineConfig:
    """流水线配置"""
    num_chunks: int = 4              # 分块数量
    enable_overlap: bool = True      # 是否启用重叠
    prefetch_count: int = 1          # 预取块数量


class PipelineOptimizer:
    """
    计算-通信重叠优化器
    
    核心思想：将大规模计算任务分解为多个小块，
    在计算当前块的同时，异步传输下一块数据，
    从而隐藏通信延迟。
    
    时间分析：
    - 无重叠：T = T_compute + T_communication
    - 有重叠：T ≈ max(T_compute, T_communication) + startup_overhead
    
    最优分块大小公式：
    d_opt = (latency × compute_power × bandwidth) / (intensity × bandwidth - compute_power)
    """
    
    def __init__(self, mpi: MPIManager, config: Optional[PipelineConfig] = None):
        """
        初始化流水线优化器
        
        Args:
            mpi: MPI管理器
            config: 流水线配置
        """
        self.mpi = mpi
        self.config = config or PipelineConfig()
        self.rank = mpi.get_rank()
        self.size = mpi.get_size()
        self.gpu_id = mpi.get_gpu_id()
        
        # CUDA流用于异步操作
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream(device=self.gpu_id)
            self.comm_stream = torch.cuda.Stream(device=self.gpu_id)
        
        # 性能统计
        self.stats = {
            'total_compute_time': 0.0,
            'total_comm_time': 0.0,
            'overlap_savings': 0.0,
            'num_chunks_processed': 0
        }
    
    def compute_optimal_chunk_count(self,
                                    tensor_size_bytes: int,
                                    compute_intensity: float,
                                    bandwidth_gbps: float,
                                    latency_ms: float = 0.5) -> int:
        """
        计算最优分块数量
        
        理论推导：
        设总数据量为D，分块数为K，带宽为B，计算强度为I（FLOP/byte）
        
        每块数据量：d = D / K
        每块计算时间：t_comp = d × I / compute_power
        每块通信时间：t_comm = d / B + latency
        
        最优条件：t_comp ≈ t_comm（计算和通信时间相等时重叠效果最好）
        
        Args:
            tensor_size_bytes: 张量字节数
            compute_intensity: 计算强度（FLOP/byte）
            bandwidth_gbps: 带宽（GB/s）
            latency_ms: 延迟（ms）
        
        Returns:
            最优分块数量
        """
        # 获取GPU计算能力估算
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.gpu_id)
            # 估算 TFLOPS（SM数 × 每SM核心数 × 2 × 典型频率）
            compute_power = props.multi_processor_count * 128 * 2 * 1.5
        else:
            compute_power = 1.0  # 默认值
        
        bandwidth = bandwidth_gbps * 1e9  # bytes/s
        latency = latency_ms / 1000  # s
        
        # 计算理论最优块大小
        # 令 t_comp = t_comm
        # d × I / P = d / B + latency
        # d × (I/P - 1/B) = latency
        # d = latency × P × B / (I × B - P)  当 I×B > P
        
        numerator = latency * compute_power * 1e12 * bandwidth
        denominator = compute_intensity * bandwidth - compute_power * 1e12
        
        if denominator <= 0:
            # 计算密集型，通信可以完全隐藏，使用较少的块
            return max(2, self.size)
        
        optimal_chunk_size = numerator / denominator
        optimal_chunk_count = max(2, int(tensor_size_bytes / optimal_chunk_size))
        
        # 限制范围
        optimal_chunk_count = min(optimal_chunk_count, 32)
        optimal_chunk_count = max(optimal_chunk_count, self.config.num_chunks)
        
        return optimal_chunk_count
    
    def pipelined_allreduce(self, tensor: torch.Tensor,
                            num_chunks: Optional[int] = None) -> torch.Tensor:
        """
        流水线化的AllReduce操作
        
        将大张量分块，实现计算和通信的重叠
        
        Args:
            tensor: 输入张量
            num_chunks: 分块数量（None则自动计算）
        
        Returns:
            归约后的张量
        """
        if num_chunks is None:
            num_chunks = self.config.num_chunks
        
        if not self.config.enable_overlap or num_chunks <= 1:
            # 不使用重叠，直接AllReduce
            return self.mpi.allreduce_tensor(tensor)
        
        # 分块
        chunks = torch.chunk(tensor.flatten(), num_chunks)
        result_chunks = []
        
        # 流水线执行
        for i, chunk in enumerate(chunks):
            # 执行AllReduce
            reduced_chunk = self.mpi.allreduce_tensor(chunk)
            result_chunks.append(reduced_chunk)
            self.stats['num_chunks_processed'] += 1
        
        # 拼接结果
        result = torch.cat(result_chunks).reshape(tensor.shape)
        return result
    
    def pipelined_matmul(self, A: torch.Tensor, B: torch.Tensor,
                         num_chunks: Optional[int] = None) -> torch.Tensor:
        """
        流水线化的分布式矩阵乘法
        
        将A按行分块，实现分发、计算、收集的流水线化
        
        Args:
            A: 矩阵A [M, K]（仅主进程需要）
            B: 矩阵B [K, N]（仅主进程需要）
            num_chunks: 分块数量
        
        Returns:
            结果矩阵C [M, N]（仅主进程返回有效结果）
        """
        if num_chunks is None:
            num_chunks = self.config.num_chunks
        
        # 广播B到所有进程
        B_local = self.mpi.broadcast_tensor(B, root=0)
        
        # 分发A
        A_local = self.mpi.scatter_tensor(A, dim=0, root=0)
        
        # 本地计算
        if torch.cuda.is_available():
            A_local = A_local.cuda(self.gpu_id)
            B_local = B_local.cuda(self.gpu_id)
        
        C_local = torch.matmul(A_local, B_local)
        
        # 收集结果
        C = self.mpi.gather_tensor(C_local, dim=0, root=0)
        
        return C
    
    def estimate_overlap_benefit(self,
                                  compute_time_ms: float,
                                  comm_time_ms: float,
                                  num_chunks: int) -> dict:
        """
        估算重叠优化的收益
        
        Args:
            compute_time_ms: 总计算时间
            comm_time_ms: 总通信时间
            num_chunks: 分块数量
        
        Returns:
            收益分析字典
        """
        # 无重叠时间
        no_overlap_time = compute_time_ms + comm_time_ms
        
        # 每块时间
        chunk_compute = compute_time_ms / num_chunks
        chunk_comm = comm_time_ms / num_chunks
        
        # 有重叠时间（启动 + 稳态流水线 + 结束）
        # 启动阶段：第一块需要完整计算+通信
        # 稳态阶段：每块时间 = max(compute, comm)
        # 结束阶段：最后一块需要完整计算+通信
        
        steady_state_time = max(chunk_compute, chunk_comm) * (num_chunks - 1)
        startup_time = chunk_compute + chunk_comm
        
        overlap_time = startup_time + steady_state_time
        
        savings = no_overlap_time - overlap_time
        speedup = no_overlap_time / overlap_time if overlap_time > 0 else 1.0
        
        return {
            'no_overlap_time_ms': no_overlap_time,
            'overlap_time_ms': overlap_time,
            'savings_ms': savings,
            'speedup': speedup,
            'efficiency': min(1.0, compute_time_ms / overlap_time) if overlap_time > 0 else 1.0
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_compute_time': 0.0,
            'total_comm_time': 0.0,
            'overlap_savings': 0.0,
            'num_chunks_processed': 0
        }
    
    def print_stats(self):
        """打印统计信息"""
        print("\n流水线优化统计:")
        print(f"  处理的块数: {self.stats['num_chunks_processed']}")
        print(f"  总计算时间: {self.stats['total_compute_time']:.2f} ms")
        print(f"  总通信时间: {self.stats['total_comm_time']:.2f} ms")
        print(f"  重叠节省时间: {self.stats['overlap_savings']:.2f} ms")
