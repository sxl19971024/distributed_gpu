"""
显存感知的自适应资源调度器 (Memory-Aware Adaptive Resource Scheduler)

创新点：将代价模型从"选什么分割策略"升级到"用多少资源、怎么调度"，
形成完整的自动化资源调度链。

核心能力：
1. 实时扫描所有 GPU 的 **真实可用显存**（OS 级别，包含其他进程占用）
2. 基于可用显存智能决定使用多少 GPU 和批次策略
3. 超显存自动分批处理（Auto-Batching）
4. 安全边际机制（默认 20%）防止 OOM

关键技术：
- 使用 torch.cuda.mem_get_info() 获取 CUDA Driver 级别的显存信息，
  而非 PyTorch 的 memory_allocated()，从而正确反映其他进程占用的显存。
- 分批策略保证每批数据 + 中间结果不超过最小单卡可用显存。
- B 矩阵只传输一次，后续批次复用，减少通信开销。
"""

import math
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .mpi_manager import MPIManager


# ==================== 数据结构 ====================

@dataclass
class GPUStatus:
    """单张 GPU 的实时状态"""
    gpu_id: int                 # GPU 设备 ID
    rank: int                   # 对应的 MPI 进程 rank
    total_memory_gb: float      # 总显存 (GB)
    free_memory_gb: float       # OS 级别真实可用显存 (GB)
    usable_memory_gb: float     # 扣除安全边际后的可用显存 (GB)


@dataclass
class ExecutionPlan:
    """显存感知执行计划"""
    operation: str              # 操作名称
    num_gpus_active: int        # 实际参与计算的 GPU 数量
    num_batches: int            # 分批次数（1 = 一次性完成, 0 = 不可行）
    batch_sizes: List[int]      # 每批的大小（沿 batch_dim 的尺寸）
    batch_dim: int              # 分批维度
    per_gpu_memory_gb: float    # 每 GPU 单批次估算显存需求 (GB)
    total_data_memory_gb: float # 原始数据总显存需求 (GB)
    total_available_gb: float   # 所有 GPU 安全可用显存总和 (GB)
    min_gpu_available_gb: float # 最小单卡安全可用显存 (GB)
    feasible: bool              # 是否可行
    description: str            # 人类可读描述


# ==================== 资源规划器 ====================

class ResourcePlanner:
    """
    显存感知资源规划器
    
    核心流程：
        scan_all_gpus()  →  estimate_*_per_gpu()  →  plan_*()
    
    设计要点：
    - 使用 torch.cuda.mem_get_info() 获取 OS 级真实可用显存
    - 20% 安全边际防止其他进程临时分配导致 OOM
    - 以最小单卡可用显存为瓶颈约束，保证所有 GPU 都不溢出
    - 可通过 max_per_gpu_gb 人为限制每卡显存，模拟资源受限场景
    """
    
    SAFETY_MARGIN = 0.20  # 20% 安全边际
    
    def __init__(self, mpi: MPIManager, max_per_gpu_gb: Optional[float] = None):
        """
        初始化资源规划器
        
        Args:
            mpi: MPI 管理器实例
            max_per_gpu_gb: 人为限制每 GPU 可用显存上限 (GB)，
                            None 表示使用实际可用显存。
                            用于测试或资源共享场景。
        """
        self.mpi = mpi
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()
        self.max_per_gpu_gb = max_per_gpu_gb
    
    # ==================== GPU 显存扫描 ====================
    
    def scan_all_gpus(self) -> List[GPUStatus]:
        """
        扫描所有进程绑定的 GPU **真实可用显存**。
        
        关键点：使用 torch.cuda.mem_get_info(device) 而非
        torch.cuda.memory_allocated()。前者从 CUDA Driver 获取信息，
        包含所有进程（包括其他用户的训练任务）占用的显存；
        后者仅统计当前 PyTorch 进程的分配量。
        
        Returns:
            所有 GPU 的实时状态列表（所有进程通过 AllGather 获得完整列表）
        """
        gpu_id = self.mpi.get_gpu_id()
        
        if torch.cuda.is_available() and gpu_id >= 0:
            # torch.cuda.mem_get_info 返回 (free_bytes, total_bytes)
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            usable = (free_bytes / (1024 ** 3)) * (1.0 - self.SAFETY_MARGIN)
            local_status = GPUStatus(
                gpu_id=gpu_id,
                rank=self.rank,
                total_memory_gb=total_bytes / (1024 ** 3),
                free_memory_gb=free_bytes / (1024 ** 3),
                usable_memory_gb=max(usable, 0.0),
            )
        else:
            local_status = GPUStatus(
                gpu_id=gpu_id, rank=self.rank,
                total_memory_gb=0.0, free_memory_gb=0.0, usable_memory_gb=0.0,
            )
        
        # AllGather: 所有进程获得全局视图
        all_statuses = self.mpi.allgather(local_status)
        return all_statuses
    
    def get_usable_memory(self) -> List[float]:
        """
        获取每个进程绑定 GPU 的安全可用显存 (GB)。
        
        如果设置了 max_per_gpu_gb，则取 min(实际可用, 上限)。
        
        Returns:
            长度为 world_size 的列表
        """
        statuses = self.scan_all_gpus()
        mem = [s.usable_memory_gb for s in statuses]
        if self.max_per_gpu_gb is not None:
            mem = [min(m, self.max_per_gpu_gb) for m in mem]
        return mem
    
    # ==================== 显存估算 ====================
    
    @staticmethod
    def dtype_bytes(dtype: torch.dtype) -> int:
        """获取数据类型的字节数"""
        return {
            torch.float16: 2, torch.bfloat16: 2,
            torch.float32: 4, torch.float64: 8,
            torch.int32: 4, torch.int64: 8,
            torch.complex64: 8, torch.complex128: 16,
        }.get(dtype, 4)
    
    @staticmethod
    def _bytes_to_gb(num_bytes: float) -> float:
        return num_bytes / (1024 ** 3)
    
    def estimate_matmul_per_gpu(self, M: int, K: int, N: int,
                                 num_gpus: int, db: int = 4) -> float:
        """
        Matmul A[M,K] @ B[K,N] 行分割时每 GPU 显存需求 (GB)。
        
        每 GPU 存储: A_chunk[ceil(M/P), K] + B_full[K, N] + C_chunk[ceil(M/P), N]
        """
        local_M = math.ceil(M / num_gpus)
        mem = (local_M * K + K * N + local_M * N) * db
        return self._bytes_to_gb(mem)
    
    def estimate_fft_per_gpu(self, total_elements: int,
                              num_gpus: int, db: int = 4) -> float:
        """FFT 每 GPU: input_chunk + complex_output_chunk ≈ 3x"""
        local_n = math.ceil(total_elements / num_gpus)
        mem = local_n * db * 3
        return self._bytes_to_gb(mem)
    
    def estimate_conv2d_per_gpu(self, N: int, C: int, H: int, W: int,
                                 out_C: int, kH: int, kW: int,
                                 num_gpus: int, db: int = 4) -> float:
        """Conv2d 每 GPU: input_chunk + weight(full) + output_chunk"""
        local_N = math.ceil(N / num_gpus)
        mem = (local_N * C * H * W + out_C * C * kH * kW + local_N * out_C * H * W) * db
        return self._bytes_to_gb(mem)
    
    def estimate_reduction_per_gpu(self, total_elements: int,
                                    num_gpus: int, db: int = 4) -> float:
        """Reduction 每 GPU: input_chunk + result ≈ 2x"""
        local_n = math.ceil(total_elements / num_gpus)
        return self._bytes_to_gb(local_n * db * 2)
    
    # ==================== 计划生成 ====================
    
    def plan_matmul(self, M: int, K: int, N: int,
                     dtype: torch.dtype = torch.float32) -> ExecutionPlan:
        """
        为 Matmul A[M,K] @ B[K,N] 生成显存感知执行计划。
        
        决策流程：
        1. 检查 B 矩阵能否放入单卡（必要条件）
        2. 尝试全部 GPU 一次性完成
        3. 如果不行，计算分批数量（沿 A 的行维度分批，B 只传输一次）
        4. 验证最小批次是否可行
        
        Args:
            M, K, N: 矩阵维度（A[M,K] @ B[K,N] → C[M,N]）
            dtype: 数据类型
        
        Returns:
            执行计划
        """
        db = self.dtype_bytes(dtype)
        usable_mem = self.get_usable_memory()
        min_usable = min(usable_mem)
        total_usable = sum(usable_mem)
        P = self.world_size
        
        # 原始数据总大小
        total_data_gb = self._bytes_to_gb((M * K + K * N + M * N) * db)
        
        # ---------- 优先检查：单卡是否足够 ----------
        # 如果全部数据 (A + B + C) 能放入一张卡，直接单卡计算，
        # 跳过所有分布式通信开销（scatter/gather/allreduce）。
        single_gpu_mem = self._bytes_to_gb((M * K + K * N + M * N) * db)
        rank0_usable = usable_mem[0]
        if P > 1 and single_gpu_mem <= rank0_usable:
            return ExecutionPlan(
                operation="matmul", num_gpus_active=1,
                num_batches=1, batch_sizes=[M], batch_dim=0,
                per_gpu_memory_gb=single_gpu_mem,
                total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable,
                min_gpu_available_gb=min_usable,
                feasible=True,
                description=(f"⚡ 单卡直连 | 数据 {total_data_gb:.3f} GB "
                             f"≤ 单卡可用 {rank0_usable:.2f} GB"),
            )
        
        # ---------- 前置检查：B 必须能放入单卡 ----------
        B_mem_gb = self._bytes_to_gb(K * N * db)
        if B_mem_gb > min_usable * 0.8:  # B 最多占 80% 可用
            return ExecutionPlan(
                operation="matmul", num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=0.0, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description=(f"不可行: B 矩阵 [{K}×{N}] 需要 {B_mem_gb:.2f} GB，"
                             f"超过单卡可用 {min_usable:.2f} GB 的 80%"),
            )
        
        # ---------- 尝试一次性完成 ----------
        per_gpu = self.estimate_matmul_per_gpu(M, K, N, P, db)
        if per_gpu <= min_usable:
            return ExecutionPlan(
                operation="matmul", num_gpus_active=P,
                num_batches=1, batch_sizes=[M], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=True,
                description=f"单批次 × {P} GPU 并行 | 每GPU {per_gpu:.3f} GB",
            )
        
        # ---------- 分批处理 ----------
        # B 在每 GPU 上需要完整存储，剩余空间用于 A_chunk 和 C_chunk
        remaining_for_AC = min_usable - B_mem_gb
        # A_chunk[batch_M/P, K] + C_chunk[batch_M/P, N] = (batch_M/P)*(K+N)*db
        # → batch_M = remaining * 1024^3 * P / ((K+N) * db)
        if remaining_for_AC <= 0 or (K + N) == 0:
            return ExecutionPlan(
                operation="matmul", num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description="不可行: 显存不足以容纳 B 矩阵和最小 A 块",
            )
        
        batch_M = int(remaining_for_AC * (1024 ** 3) * P / ((K + N) * db))
        batch_M = max((batch_M // P) * P, P)  # 对齐到 P 倍数，至少 P 行
        
        # 验证最小批次可行性
        actual_per_gpu = self.estimate_matmul_per_gpu(batch_M, K, N, P, db)
        if actual_per_gpu > min_usable:
            return ExecutionPlan(
                operation="matmul", num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=actual_per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description=f"不可行: 即使最小批 ({batch_M} 行) 仍需 {actual_per_gpu:.3f} GB/卡",
            )
        
        # 生成批次列表
        num_batches = math.ceil(M / batch_M)
        batch_sizes = []
        remaining = M
        for _ in range(num_batches):
            bs = min(batch_M, remaining)
            batch_sizes.append(bs)
            remaining -= bs
        
        return ExecutionPlan(
            operation="matmul", num_gpus_active=P,
            num_batches=num_batches, batch_sizes=batch_sizes, batch_dim=0,
            per_gpu_memory_gb=actual_per_gpu, total_data_memory_gb=total_data_gb,
            total_available_gb=total_usable, min_gpu_available_gb=min_usable,
            feasible=True,
            description=(f"{num_batches} 批次 × {P} GPU 并行 | "
                         f"每批 {batch_M} 行 | 每GPU {actual_per_gpu:.3f} GB"),
        )
    
    def plan_operation(self, op: str, shapes: List[Tuple],
                        dtype: torch.dtype = torch.float32) -> ExecutionPlan:
        """
        通用操作的显存感知执行计划（FFT, Reduction, Conv2d 等）。
        
        沿第 0 维（batch 维度）分批处理。
        
        Args:
            op: 操作名称
            shapes: 输入张量形状列表
            dtype: 数据类型
        
        Returns:
            执行计划
        """
        db = self.dtype_bytes(dtype)
        usable_mem = self.get_usable_memory()
        min_usable = min(usable_mem)
        total_usable = sum(usable_mem)
        P = self.world_size
        
        # 计算总数据量
        total_elements = 0
        for shape in shapes:
            e = 1
            for s in shape:
                e *= s
            total_elements += e
        total_data_gb = self._bytes_to_gb(total_elements * db)
        
        # 估算每 GPU 内存（约 3x: input + output + temp）
        per_gpu = self._bytes_to_gb(math.ceil(total_elements / P) * db * 3)
        
        dim0 = shapes[0][0] if shapes else 0
        
        # ---------- 优先检查：单卡是否足够 ----------
        single_gpu_mem = self._bytes_to_gb(total_elements * db * 3)
        rank0_usable = usable_mem[0]
        if P > 1 and single_gpu_mem <= rank0_usable:
            return ExecutionPlan(
                operation=op, num_gpus_active=1,
                num_batches=1, batch_sizes=[dim0], batch_dim=0,
                per_gpu_memory_gb=single_gpu_mem,
                total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable,
                min_gpu_available_gb=min_usable,
                feasible=True,
                description=(f"⚡ 单卡直连 | 数据 {total_data_gb:.3f} GB "
                             f"≤ 单卡可用 {rank0_usable:.2f} GB"),
            )
        
        if per_gpu <= min_usable:
            return ExecutionPlan(
                operation=op, num_gpus_active=P,
                num_batches=1, batch_sizes=[dim0], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=True,
                description=f"单批次 × {P} GPU 并行 | 每GPU {per_gpu:.3f} GB",
            )
        
        # 分批处理
        if dim0 == 0:
            return ExecutionPlan(
                operation=op, num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description="不可行: 第 0 维大小为 0",
            )
        
        elements_per_row = max(total_elements // dim0, 1)
        max_local_elements = int(min_usable * (1024 ** 3) / (db * 3))
        batch_dim0_local = max(max_local_elements // elements_per_row, 1)
        batch_dim0 = max(batch_dim0_local * P, P)
        
        num_batches = math.ceil(dim0 / batch_dim0)
        batch_sizes = []
        remaining = dim0
        for _ in range(num_batches):
            bs = min(batch_dim0, remaining)
            batch_sizes.append(bs)
            remaining -= bs
        
        actual_per_gpu = self._bytes_to_gb(
            math.ceil(batch_dim0 * elements_per_row / P) * db * 3
        )
        
        return ExecutionPlan(
            operation=op, num_gpus_active=P,
            num_batches=num_batches, batch_sizes=batch_sizes, batch_dim=0,
            per_gpu_memory_gb=actual_per_gpu, total_data_memory_gb=total_data_gb,
            total_available_gb=total_usable, min_gpu_available_gb=min_usable,
            feasible=True,
            description=(f"{num_batches} 批次 × {P} GPU 并行 | "
                         f"每批 dim0={batch_dim0} | 每GPU {actual_per_gpu:.3f} GB"),
        )
    
    # ==================== 信息展示 ====================
    
    def print_gpu_status(self):
        """
        打印所有 GPU 的显存实时状态（仅 master 打印）。
        
        包含可视化进度条，直观展示每卡的显存占用情况。
        """
        statuses = self.scan_all_gpus()
        if self.mpi.is_master_process():
            print("=" * 70)
            print("  GPU 显存状态（实时检测，含其他进程占用）")
            print("-" * 70)
            for s in statuses:
                bar_len = 30
                if s.total_memory_gb > 0:
                    used_ratio = 1.0 - (s.free_memory_gb / s.total_memory_gb)
                else:
                    used_ratio = 0
                filled = int(bar_len * used_ratio)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"  GPU {s.gpu_id} (Rank {s.rank:2d}): "
                      f"[{bar}] "
                      f"{s.free_memory_gb:.1f} / {s.total_memory_gb:.1f} GB 空闲 | "
                      f"安全可用 {s.usable_memory_gb:.1f} GB")
            print(f"  总安全可用: {sum(s.usable_memory_gb for s in statuses):.1f} GB")
            print("=" * 70)
