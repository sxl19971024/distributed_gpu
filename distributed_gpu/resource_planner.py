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
- 使用 ``torch.cuda.mem_get_info()`` 获取 CUDA Driver 级别的显存信息，
  而非 PyTorch 的 ``memory_allocated()``，从而正确反映其他进程占用的显存。
- 分批策略保证每批数据 + 中间结果不超过最小单卡可用显存。
- B 矩阵只传输一次，后续批次复用，减少通信开销。
- GPU 显存扫描结果带短期缓存，避免连续操作时重复 AllGather。
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .mpi_manager import MPIManager


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class GPUStatus:
    """单张 GPU 的实时状态

    Attributes:
        gpu_id: GPU 设备 ID
        rank: 对应的 MPI 进程 rank
        total_memory_gb: 总显存（GB）
        free_memory_gb: OS 级别真实可用显存（GB）
        usable_memory_gb: 扣除安全边际后的可用显存（GB）
    """
    gpu_id: int
    rank: int
    total_memory_gb: float
    free_memory_gb: float
    usable_memory_gb: float


@dataclass
class ExecutionPlan:
    """显存感知执行计划

    Attributes:
        operation: 操作名称
        num_gpus_active: 实际参与计算的 GPU 数量
        num_batches: 分批次数（1 = 一次性完成，0 = 不可行）
        batch_sizes: 每批的大小（沿 batch_dim 的尺寸）
        batch_dim: 分批维度
        per_gpu_memory_gb: 每 GPU 单批次估算显存需求（GB）
        total_data_memory_gb: 原始数据总显存需求（GB）
        total_available_gb: 所有 GPU 安全可用显存总和（GB）
        min_gpu_available_gb: 最小单卡安全可用显存（GB）
        feasible: 是否可行
        description: 人类可读描述
    """
    operation: str
    num_gpus_active: int
    num_batches: int
    batch_sizes: List[int]
    batch_dim: int
    per_gpu_memory_gb: float
    total_data_memory_gb: float
    total_available_gb: float
    min_gpu_available_gb: float
    feasible: bool
    description: str


# ---------------------------------------------------------------------------
# 资源规划器
# ---------------------------------------------------------------------------

# 字节 → GB 转换常量
_GB = 1024 ** 3

# dtype → 字节数映射
_DTYPE_BYTES: Dict[torch.dtype, int] = {
    torch.float16: 2, torch.bfloat16: 2,
    torch.float32: 4, torch.float64: 8,
    torch.int32: 4, torch.int64: 8,
    torch.complex64: 8, torch.complex128: 16,
}


class ResourcePlanner:
    """显存感知资源规划器

    核心流程::

        scan_all_gpus()  →  estimate_*_per_gpu()  →  plan_*()

    设计要点：
    - 使用 ``torch.cuda.mem_get_info()`` 获取 OS 级真实可用显存
    - 20% 安全边际防止其他进程临时分配导致 OOM
    - 以最小单卡可用显存为瓶颈约束，保证所有 GPU 都不溢出
    - 可通过 ``max_per_gpu_gb`` 人为限制每卡显存，模拟资源受限场景
    """

    SAFETY_MARGIN: float = 0.20   # 20% 安全边际
    MEM_CACHE_TTL: float = 2.0    # 显存扫描缓存有效期（秒）

    def __init__(
        self,
        mpi: MPIManager,
        max_per_gpu_gb: Optional[float] = None,
    ) -> None:
        """
        Args:
            mpi: MPI 管理器实例
            max_per_gpu_gb: 人为限制每 GPU 可用显存上限（GB），
                            ``None`` 表示使用实际可用显存。
        """
        self.mpi: MPIManager = mpi
        self.rank: int = mpi.get_rank()
        self.world_size: int = mpi.get_size()
        self.max_per_gpu_gb: Optional[float] = max_per_gpu_gb

        # 显存扫描缓存
        self._mem_cache: Optional[List[float]] = None
        self._mem_cache_time: float = 0.0

    # ------------------------------------------------------------------
    # GPU 显存扫描
    # ------------------------------------------------------------------

    def scan_all_gpus(self) -> List[GPUStatus]:
        """扫描所有进程绑定的 GPU **真实可用显存**

        使用 ``torch.cuda.mem_get_info(device)`` 从 CUDA Driver 获取信息，
        包含所有进程（包括其他用户的训练任务）占用的显存。

        Returns:
            所有 GPU 的实时状态列表（通过 AllGather 获得完整列表）
        """
        gpu_id: int = self.mpi.get_gpu_id()

        if torch.cuda.is_available() and gpu_id >= 0:
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            usable = (free_bytes / _GB) * (1.0 - self.SAFETY_MARGIN)
            local_status = GPUStatus(
                gpu_id=gpu_id,
                rank=self.rank,
                total_memory_gb=total_bytes / _GB,
                free_memory_gb=free_bytes / _GB,
                usable_memory_gb=max(usable, 0.0),
            )
        else:
            local_status = GPUStatus(
                gpu_id=gpu_id, rank=self.rank,
                total_memory_gb=0.0, free_memory_gb=0.0, usable_memory_gb=0.0,
            )

        return self.mpi.allgather(local_status)

    def get_usable_memory(self) -> List[float]:
        """获取每个进程绑定 GPU 的安全可用显存（GB）

        带短期缓存（默认 2 秒），避免连续操作时重复 AllGather。
        若设置了 ``max_per_gpu_gb``，则取 ``min(实际可用, 上限)``。

        Returns:
            长度为 ``world_size`` 的列表
        """
        now = _time.monotonic()
        if self._mem_cache is not None and (now - self._mem_cache_time) < self.MEM_CACHE_TTL:
            return self._mem_cache

        statuses = self.scan_all_gpus()
        mem = [s.usable_memory_gb for s in statuses]
        if self.max_per_gpu_gb is not None:
            mem = [min(m, self.max_per_gpu_gb) for m in mem]

        self._mem_cache = mem
        self._mem_cache_time = now
        return mem

    def invalidate_mem_cache(self) -> None:
        """手动使显存缓存失效（大量分配/释放后调用）"""
        self._mem_cache = None

    # ------------------------------------------------------------------
    # 显存估算工具
    # ------------------------------------------------------------------

    @staticmethod
    def dtype_bytes(dtype: torch.dtype) -> int:
        """获取数据类型的字节数"""
        return _DTYPE_BYTES.get(dtype, 4)

    @staticmethod
    def _bytes_to_gb(num_bytes: float) -> float:
        """字节转 GB"""
        return num_bytes / _GB

    def estimate_matmul_per_gpu(
        self, M: int, K: int, N: int, num_gpus: int, db: int = 4,
    ) -> float:
        """Matmul ``A[M,K] @ B[K,N]`` 行分割时每 GPU 显存需求（GB）

        每 GPU 存储: ``A_chunk[⌈M/P⌉, K] + B_full[K, N] + C_chunk[⌈M/P⌉, N]``
        """
        local_M = math.ceil(M / num_gpus)
        return self._bytes_to_gb((local_M * K + K * N + local_M * N) * db)

    def estimate_fft_per_gpu(
        self, total_elements: int, num_gpus: int, db: int = 4,
    ) -> float:
        """FFT 每 GPU: ``input_chunk + complex_output_chunk ≈ 3×``"""
        local_n = math.ceil(total_elements / num_gpus)
        return self._bytes_to_gb(local_n * db * 3)

    def estimate_conv2d_per_gpu(
        self,
        N: int, C: int, H: int, W: int,
        out_C: int, kH: int, kW: int,
        num_gpus: int, db: int = 4,
    ) -> float:
        """Conv2d 每 GPU: ``input_chunk + weight(full) + output_chunk``"""
        local_N = math.ceil(N / num_gpus)
        mem = (local_N * C * H * W + out_C * C * kH * kW + local_N * out_C * H * W) * db
        return self._bytes_to_gb(mem)

    def estimate_reduction_per_gpu(
        self, total_elements: int, num_gpus: int, db: int = 4,
    ) -> float:
        """Reduction 每 GPU: ``input_chunk + result ≈ 2×``"""
        local_n = math.ceil(total_elements / num_gpus)
        return self._bytes_to_gb(local_n * db * 2)

    # ------------------------------------------------------------------
    # 执行计划生成
    # ------------------------------------------------------------------

    def _make_plan(
        self,
        operation: str,
        num_gpus_active: int,
        num_batches: int,
        batch_sizes: List[int],
        batch_dim: int,
        per_gpu_memory_gb: float,
        total_data_memory_gb: float,
        total_available_gb: float,
        min_gpu_available_gb: float,
        feasible: bool,
        description: str,
    ) -> ExecutionPlan:
        """统一的 ExecutionPlan 构造器"""
        return ExecutionPlan(
            operation=operation,
            num_gpus_active=num_gpus_active,
            num_batches=num_batches,
            batch_sizes=batch_sizes,
            batch_dim=batch_dim,
            per_gpu_memory_gb=per_gpu_memory_gb,
            total_data_memory_gb=total_data_memory_gb,
            total_available_gb=total_available_gb,
            min_gpu_available_gb=min_gpu_available_gb,
            feasible=feasible,
            description=description,
        )

    def _common_context(self, dtype: torch.dtype) -> Tuple[int, List[float], float, float, int]:
        """提取常用上下文: (db, usable_mem, min_usable, total_usable, P)"""
        db = self.dtype_bytes(dtype)
        usable_mem = self.get_usable_memory()
        min_usable = min(usable_mem)
        total_usable = sum(usable_mem)
        return db, usable_mem, min_usable, total_usable, self.world_size

    def _try_single_gpu(
        self,
        operation: str,
        single_gpu_mem: float,
        usable_mem: List[float],
        total_data_gb: float,
        total_usable: float,
        min_usable: float,
        dim0: int,
        P: int,
    ) -> Optional[ExecutionPlan]:
        """尝试单卡直连计划，可行时返回 ExecutionPlan，否则返回 None"""
        if P > 1 and single_gpu_mem <= usable_mem[0]:
            return self._make_plan(
                operation=operation, num_gpus_active=1,
                num_batches=1, batch_sizes=[dim0], batch_dim=0,
                per_gpu_memory_gb=single_gpu_mem,
                total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable,
                min_gpu_available_gb=min_usable,
                feasible=True,
                description=(
                    f"⚡ 单卡直连 | 数据 {total_data_gb:.3f} GB "
                    f"≤ 单卡可用 {usable_mem[0]:.2f} GB"
                ),
            )
        return None

    @staticmethod
    def _compute_batch_sizes(total_dim: int, batch_dim_size: int) -> List[int]:
        """沿某一维度计算分批大小列表"""
        num_batches = math.ceil(total_dim / batch_dim_size)
        sizes: List[int] = []
        remaining = total_dim
        for _ in range(num_batches):
            bs = min(batch_dim_size, remaining)
            sizes.append(bs)
            remaining -= bs
        return sizes

    # ------------------------------------------------------------------

    def plan_matmul(
        self,
        M: int,
        K: int,
        N: int,
        dtype: torch.dtype = torch.float32,
    ) -> ExecutionPlan:
        """为 Matmul ``A[M,K] @ B[K,N]`` 生成显存感知执行计划

        决策流程：
        1. 检查 B 矩阵能否放入单卡（必要条件）
        2. 尝试全部 GPU 一次性完成
        3. 如果不行，计算分批数量（沿 A 的行维度分批，B 只传输一次）
        4. 验证最小批次是否可行

        Args:
            M, K, N: 矩阵维度
            dtype: 数据类型

        Returns:
            ExecutionPlan
        """
        db, usable_mem, min_usable, total_usable, P = self._common_context(dtype)
        total_data_gb = self._bytes_to_gb((M * K + K * N + M * N) * db)

        # 尝试单卡直连
        single_gpu_mem = total_data_gb  # A + B + C
        plan = self._try_single_gpu(
            "matmul", single_gpu_mem, usable_mem,
            total_data_gb, total_usable, min_usable, M, P,
        )
        if plan is not None:
            return plan

        # B 矩阵必须能放入单卡
        B_mem_gb = self._bytes_to_gb(K * N * db)
        if B_mem_gb > min_usable * 0.8:
            return self._make_plan(
                operation="matmul", num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=0.0, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description=(
                    f"不可行: B 矩阵 [{K}×{N}] 需要 {B_mem_gb:.2f} GB，"
                    f"超过单卡可用 {min_usable:.2f} GB 的 80%"
                ),
            )

        # 尝试一次性完成
        per_gpu = self.estimate_matmul_per_gpu(M, K, N, P, db)
        if per_gpu <= min_usable:
            return self._make_plan(
                operation="matmul", num_gpus_active=P,
                num_batches=1, batch_sizes=[M], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=True,
                description=f"单批次 × {P} GPU 并行 | 每GPU {per_gpu:.3f} GB",
            )

        # 分批处理
        remaining_for_AC = min_usable - B_mem_gb
        if remaining_for_AC <= 0 or (K + N) == 0:
            return self._make_plan(
                operation="matmul", num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description="不可行: 显存不足以容纳 B 矩阵和最小 A 块",
            )

        batch_M = int(remaining_for_AC * _GB * P / ((K + N) * db))
        batch_M = max((batch_M // P) * P, P)

        actual_per_gpu = self.estimate_matmul_per_gpu(batch_M, K, N, P, db)
        if actual_per_gpu > min_usable:
            return self._make_plan(
                operation="matmul", num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=actual_per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description=f"不可行: 即使最小批 ({batch_M} 行) 仍需 {actual_per_gpu:.3f} GB/卡",
            )

        batch_sizes = self._compute_batch_sizes(M, batch_M)
        return self._make_plan(
            operation="matmul", num_gpus_active=P,
            num_batches=len(batch_sizes), batch_sizes=batch_sizes, batch_dim=0,
            per_gpu_memory_gb=actual_per_gpu, total_data_memory_gb=total_data_gb,
            total_available_gb=total_usable, min_gpu_available_gb=min_usable,
            feasible=True,
            description=(
                f"{len(batch_sizes)} 批次 × {P} GPU 并行 | "
                f"每批 {batch_M} 行 | 每GPU {actual_per_gpu:.3f} GB"
            ),
        )

    def plan_operation(
        self,
        op: str,
        shapes: List[Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
    ) -> ExecutionPlan:
        """通用操作的显存感知执行计划（FFT, Reduction, Conv2d 等）

        沿第 0 维（batch 维度）分批处理。

        Args:
            op: 操作名称
            shapes: 输入张量形状列表
            dtype: 数据类型

        Returns:
            ExecutionPlan
        """
        db, usable_mem, min_usable, total_usable, P = self._common_context(dtype)

        # 计算总元素数
        total_elements = sum(math.prod(s) for s in shapes)
        total_data_gb = self._bytes_to_gb(total_elements * db)
        dim0 = shapes[0][0] if shapes else 0

        # 尝试单卡直连
        single_gpu_mem = self._bytes_to_gb(total_elements * db * 3)
        plan = self._try_single_gpu(
            op, single_gpu_mem, usable_mem,
            total_data_gb, total_usable, min_usable, dim0, P,
        )
        if plan is not None:
            return plan

        # 每 GPU 估算（约 3×: input + output + temp）
        per_gpu = self._bytes_to_gb(math.ceil(total_elements / P) * db * 3)
        if per_gpu <= min_usable:
            return self._make_plan(
                operation=op, num_gpus_active=P,
                num_batches=1, batch_sizes=[dim0], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=True,
                description=f"单批次 × {P} GPU 并行 | 每GPU {per_gpu:.3f} GB",
            )

        # 分批处理
        if dim0 == 0:
            return self._make_plan(
                operation=op, num_gpus_active=P,
                num_batches=0, batch_sizes=[], batch_dim=0,
                per_gpu_memory_gb=per_gpu, total_data_memory_gb=total_data_gb,
                total_available_gb=total_usable, min_gpu_available_gb=min_usable,
                feasible=False,
                description="不可行: 第 0 维大小为 0",
            )

        elements_per_row = max(total_elements // dim0, 1)
        max_local_elements = int(min_usable * _GB / (db * 3))
        batch_dim0_local = max(max_local_elements // elements_per_row, 1)
        batch_dim0 = max(batch_dim0_local * P, P)

        batch_sizes = self._compute_batch_sizes(dim0, batch_dim0)
        actual_per_gpu = self._bytes_to_gb(
            math.ceil(batch_dim0 * elements_per_row / P) * db * 3
        )

        return self._make_plan(
            operation=op, num_gpus_active=P,
            num_batches=len(batch_sizes), batch_sizes=batch_sizes, batch_dim=0,
            per_gpu_memory_gb=actual_per_gpu, total_data_memory_gb=total_data_gb,
            total_available_gb=total_usable, min_gpu_available_gb=min_usable,
            feasible=True,
            description=(
                f"{len(batch_sizes)} 批次 × {P} GPU 并行 | "
                f"每批 dim0={batch_dim0} | 每GPU {actual_per_gpu:.3f} GB"
            ),
        )

    # ------------------------------------------------------------------
    # 信息展示
    # ------------------------------------------------------------------

    def print_gpu_status(self) -> None:
        """打印所有 GPU 的显存实时状态（仅 master 打印）

        包含可视化进度条，直观展示每卡的显存占用情况。
        """
        statuses = self.scan_all_gpus()
        if not self.mpi.is_master_process():
            return

        print("=" * 70)
        print("  GPU 显存状态（实时检测，含其他进程占用）")
        print("-" * 70)
        bar_len = 30
        for s in statuses:
            used_ratio = 1.0 - (s.free_memory_gb / s.total_memory_gb) if s.total_memory_gb > 0 else 0.0
            filled = int(bar_len * used_ratio)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"  GPU {s.gpu_id} (Rank {s.rank:2d}): "
                f"[{bar}] "
                f"{s.free_memory_gb:.1f} / {s.total_memory_gb:.1f} GB 空闲 | "
                f"安全可用 {s.usable_memory_gb:.1f} GB"
            )
        print(f"  总安全可用: {sum(s.usable_memory_gb for s in statuses):.1f} GB")
        print("=" * 70)
