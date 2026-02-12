"""
性能分析工具

提供计时、显存监控等功能。
支持 CUDA 同步以获得精确的 GPU 计时。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TimingRecord:
    """单次计时记录"""

    name: str
    start_time: float
    end_time: float
    duration: float


class Profiler:
    """
    性能分析器

    用法::

        prof = Profiler()
        prof.start("matmul")
        # ... 计算 ...
        elapsed = prof.end("matmul")
        prof.print_summary()
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Args:
            enabled: 是否启用分析（False 时所有操作为空操作）
        """
        self.enabled: bool = enabled
        self.timings: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
        self.memory_snapshots: List[Tuple[str, float]] = []

    # ==================== 计时 ====================

    def start(self, name: str) -> None:
        """开始计时（自动 CUDA 同步）"""
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.active_timers[name] = time.time()

    def end(self, name: str) -> float:
        """
        结束计时（自动 CUDA 同步）

        Returns:
            持续时间（秒），未启用或无对应 start 时返回 0.0
        """
        if not self.enabled:
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if name not in self.active_timers:
            return 0.0

        duration: float = time.time() - self.active_timers.pop(name)
        self.timings.setdefault(name, []).append(duration)
        return duration

    # ==================== 显存监控 ====================

    def record_memory(self, label: str) -> None:
        """记录当前 CUDA 显存使用（MB）"""
        if not self.enabled or not torch.cuda.is_available():
            return
        memory_mb: float = torch.cuda.memory_allocated() / (1024 ** 2)
        self.memory_snapshots.append((label, memory_mb))

    # ==================== 统计查询 ====================

    def get_average(self, name: str) -> float:
        """获取指定操作的平均耗时（秒）"""
        times = self.timings.get(name, [])
        return sum(times) / len(times) if times else 0.0

    def get_total(self, name: str) -> float:
        """获取指定操作的总耗时（秒）"""
        return sum(self.timings.get(name, []))

    # ==================== 生命周期 ====================

    def reset(self) -> None:
        """重置所有记录"""
        self.timings.clear()
        self.active_timers.clear()
        self.memory_snapshots.clear()

    # ==================== 输出 ====================

    def print_summary(self) -> None:
        """打印性能摘要"""
        if not self.enabled or not self.timings:
            return

        print("\n" + "=" * 60)
        print("性能分析摘要")
        print("=" * 60)

        for name, times in self.timings.items():
            total: float = sum(times)
            avg: float = total / len(times)
            min_t: float = min(times)
            max_t: float = max(times)

            print(f"\n{name}:")
            print(f"  调用次数: {len(times)}")
            print(f"  总耗时: {total:.4f} 秒")
            print(f"  平均耗时: {avg:.4f} 秒")
            print(f"  最小耗时: {min_t:.4f} 秒")
            print(f"  最大耗时: {max_t:.4f} 秒")
            if len(times) > 1:
                std: float = statistics.stdev(times)
                print(f"  标准差: {std:.4f} 秒")

        if self.memory_snapshots:
            print(f"\n{'─' * 40}")
            print("显存快照:")
            for label, mem_mb in self.memory_snapshots:
                print(f"  {label}: {mem_mb:.1f} MB")

        print("\n" + "=" * 60)
