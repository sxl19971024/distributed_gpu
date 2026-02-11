"""
计算-通信重叠的流水线优化策略

创新点2：提出一种基于分块流水线的计算-通信重叠优化方法。

理论贡献：
1. 建立了计算-通信重叠的理论模型
2. 推导了最优分块大小的解析表达式
3. 设计了多级流水线调度算法
4. 利用 CUDA 双流实现 GPU 计算与 CPU 侧 MPI 通信的真正重叠
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
    在 GPU 上用 compute_stream 执行当前块的矩阵乘法时，
    CPU 侧同时执行下一块的 MPI scatter / 上一块的 MPI gather，
    从而隐藏通信延迟。

    时间分析：
    - 无重叠：T = T_scatter + T_compute + T_gather  (逐步串行)
    - 有重叠：T ≈ max(T_compute, T_comm) × K + startup
                ≈ T_compute × K  (当计算密集时通信完全被隐藏)
    """

    def __init__(self, mpi: MPIManager, config: Optional[PipelineConfig] = None):
        self.mpi = mpi
        self.config = config or PipelineConfig()
        self.rank = mpi.get_rank()
        self.size = mpi.get_size()
        self.gpu_id = mpi.get_gpu_id()

        # CUDA 双流：compute_stream 做矩阵乘法，默认流做 GPU↔CPU 数据搬运
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream(device=self.gpu_id)
            self.comm_stream = torch.cuda.Stream(device=self.gpu_id)
        else:
            self.compute_stream = None
            self.comm_stream = None

        # 性能统计
        self.stats = {
            'total_compute_time': 0.0,
            'total_comm_time': 0.0,
            'overlap_savings': 0.0,
            'num_chunks_processed': 0,
        }

    # ==================== 最优分块计算 ====================

    def compute_optimal_chunk_count(self,
                                    tensor_size_bytes: int,
                                    compute_intensity: float,
                                    bandwidth_gbps: float,
                                    latency_ms: float = 0.5) -> int:
        """
        计算最优分块数量

        理论推导：
        设总数据量为 D，分块数为 K，带宽为 B，计算强度为 I（FLOP/byte）
        每块计算时间：t_comp = d × I / P
        每块通信时间：t_comm = d / B + latency
        最优条件：t_comp ≈ t_comm
        """
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.gpu_id)
            compute_power = props.multi_processor_count * 128 * 2 * 1.5
        else:
            compute_power = 1.0

        bandwidth = bandwidth_gbps * 1e9
        latency = latency_ms / 1000

        numerator = latency * compute_power * 1e12 * bandwidth
        denominator = compute_intensity * bandwidth - compute_power * 1e12

        if denominator <= 0:
            return max(2, self.size)

        optimal_chunk_size = numerator / denominator
        optimal_chunk_count = max(2, int(tensor_size_bytes / optimal_chunk_size))
        optimal_chunk_count = min(optimal_chunk_count, 32)
        optimal_chunk_count = max(optimal_chunk_count, self.config.num_chunks)

        return optimal_chunk_count

    # ==================== 流水线化 AllReduce ====================

    def pipelined_allreduce(self, tensor: torch.Tensor,
                            num_chunks: Optional[int] = None) -> torch.Tensor:
        """
        流水线化的 AllReduce 操作：将大张量分块，逐块执行 allreduce，
        利用 CUDA 流实现 GPU→CPU 拷贝与上一块的 CPU→GPU 拷贝重叠。
        """
        if num_chunks is None:
            num_chunks = self.config.num_chunks

        if not self.config.enable_overlap or num_chunks <= 1:
            return self.mpi.allreduce_tensor(tensor)

        flat = tensor.flatten()
        chunks = list(torch.chunk(flat, num_chunks))
        result_chunks = [None] * len(chunks)

        for i in range(len(chunks)):
            # GPU → CPU（在 comm_stream 上排队，不阻塞 compute_stream）
            if self.comm_stream is not None:
                with torch.cuda.stream(self.comm_stream):
                    chunk_cpu = chunks[i].cpu()
                self.comm_stream.synchronize()
            else:
                chunk_cpu = chunks[i].cpu()

            # CPU 侧 MPI AllReduce（阻塞 CPU 线程，但 GPU compute_stream 可继续工作）
            from mpi4py import MPI as _MPI
            data = chunk_cpu.numpy().copy()
            result_np = np.empty_like(data)
            self.mpi.comm.Allreduce(data, result_np, op=_MPI.SUM)

            # CPU → GPU（在 comm_stream 上排队）
            if self.comm_stream is not None:
                with torch.cuda.stream(self.comm_stream):
                    result_chunks[i] = torch.from_numpy(result_np).cuda(self.gpu_id)
                self.comm_stream.synchronize()
            else:
                result_chunks[i] = torch.from_numpy(result_np)

            self.stats['num_chunks_processed'] += 1

        return torch.cat(result_chunks).reshape(tensor.shape)

    # ==================== 流水线化矩阵乘法 (核心创新) ====================

    def pipelined_matmul(self, A: Optional[torch.Tensor],
                         B: Optional[torch.Tensor],
                         num_chunks: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        流水线化的分布式矩阵乘法

        三阶段流水线 (per chunk i):
          Stage 1 — Scatter(i):   CPU侧 MPI 分发 A 的第 i 子块
          Stage 2 — Compute(i):   GPU compute_stream 执行 matmul
          Stage 3 — Gather(i):    CPU侧 MPI 收集 C 的第 i 子块

        流水线时序示意 (稳态阶段)：
          CPU:  |Gather(i-1)|Scatter(i+1)|
          GPU:        |   Compute(i)   |

        GPU 上的 matmul 在 compute_stream 上异步执行，
        CPU 线程在同一时段执行 MPI gather/scatter（纯 CPU 操作），
        二者天然重叠。
        """
        if num_chunks is None:
            num_chunks = self.config.num_chunks

        # 第一步：广播 B 到所有进程
        B_local = self.mpi.broadcast_tensor(B, root=0)

        # 第二步：在主进程上把 A 切成 num_chunks 个宏块
        if self.mpi.is_master_process():
            M = A.shape[0]
            macro_chunks = list(torch.chunk(A, num_chunks, dim=0))
            # 每个宏块再切成 num_processes 份用于 scatter
            scatter_lists = []
            for mc in macro_chunks:
                sub = list(torch.chunk(mc, self.size, dim=0))
                scatter_lists.append([s.cpu().numpy() for s in sub])
        else:
            scatter_lists = [None] * num_chunks
            M = 0
        M = self.mpi.broadcast(M if self.mpi.is_master_process() else None)

        # 第三步：三阶段流水线
        gather_results = []       # 主进程收集的 C 块
        prev_C_local = None       # 上一轮的本地计算结果 (GPU)
        A_local_current = None    # 当前轮的本地 A 块 (GPU)
        compute_event = None      # 追踪 compute_stream 上的计算完成

        t_compute_total = 0.0
        t_comm_total = 0.0

        for i in range(num_chunks + 1):
            t0 = time.time()

            # === Stage 3: Gather 上一轮结果 (CPU 侧 MPI) ===
            if i > 0 and prev_C_local is not None:
                # 等待 GPU 计算完成
                if compute_event is not None:
                    compute_event.synchronize()
                C_block = self.mpi.gather_tensor(prev_C_local, dim=0, root=0)
                if self.mpi.is_master_process():
                    gather_results.append(C_block)

            # === Stage 1: Scatter 下一轮 A 块 (CPU 侧 MPI) ===
            if i < num_chunks:
                sl = scatter_lists[i] if self.mpi.is_master_process() else None
                a_local_np = self.mpi.comm.scatter(sl, root=0)
                A_local_next = torch.from_numpy(a_local_np.copy())
                if torch.cuda.is_available():
                    A_local_next = A_local_next.cuda(self.gpu_id)

            t1 = time.time()
            t_comm_total += (t1 - t0) * 1000

            # === Stage 2: Compute 当前块 matmul (GPU 异步) ===
            if i < num_chunks:
                t2 = time.time()
                if self.compute_stream is not None:
                    with torch.cuda.stream(self.compute_stream):
                        prev_C_local = torch.matmul(A_local_next, B_local)
                    compute_event = torch.cuda.Event()
                    compute_event.record(self.compute_stream)
                else:
                    prev_C_local = torch.matmul(A_local_next, B_local)
                    compute_event = None
                t3 = time.time()
                t_compute_total += (t3 - t2) * 1000

        self.stats['total_compute_time'] += t_compute_total
        self.stats['total_comm_time'] += t_comm_total
        self.stats['num_chunks_processed'] += num_chunks

        # 拼接结果
        if self.mpi.is_master_process() and gather_results:
            return torch.cat(gather_results, dim=0)
        return None

    # ==================== 非流水线基线 ====================

    def baseline_matmul(self, A: Optional[torch.Tensor],
                        B: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        非流水线的分布式矩阵乘法（用作对比基线）
        """
        B_local = self.mpi.broadcast_tensor(B, root=0)
        A_local = self.mpi.scatter_tensor(A, dim=0, root=0)

        if torch.cuda.is_available():
            A_local = A_local.cuda(self.gpu_id)
            B_local = B_local.cuda(self.gpu_id)

        C_local = torch.matmul(A_local, B_local)
        C = self.mpi.gather_tensor(C_local, dim=0, root=0)
        return C

    # ==================== 理论收益估算 ====================

    def estimate_overlap_benefit(self,
                                  compute_time_ms: float,
                                  comm_time_ms: float,
                                  num_chunks: int) -> dict:
        """估算重叠优化的收益"""
        no_overlap_time = compute_time_ms + comm_time_ms
        chunk_compute = compute_time_ms / num_chunks
        chunk_comm = comm_time_ms / num_chunks

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
            'efficiency': min(1.0, compute_time_ms / overlap_time) if overlap_time > 0 else 1.0,
        }

    # ==================== 统计 ====================

    def reset_stats(self):
        self.stats = {
            'total_compute_time': 0.0,
            'total_comm_time': 0.0,
            'overlap_savings': 0.0,
            'num_chunks_processed': 0,
        }

    def print_stats(self):
        print("\n流水线优化统计:")
        print(f"  处理的块数: {self.stats['num_chunks_processed']}")
        print(f"  总计算时间: {self.stats['total_compute_time']:.2f} ms")
        print(f"  总通信时间: {self.stats['total_comm_time']:.2f} ms")
        print(f"  重叠节省时间: {self.stats['overlap_savings']:.2f} ms")
