"""
性能分析工具

提供计时、显存监控等功能。
"""

import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TimingRecord:
    """计时记录"""
    name: str
    start_time: float
    end_time: float
    duration: float


class Profiler:
    """性能分析器"""
    
    def __init__(self, enabled: bool = True):
        """
        初始化分析器
        
        Args:
            enabled: 是否启用分析
        """
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
        self.memory_snapshots: List[tuple] = []
    
    def start(self, name: str):
        """开始计时"""
        if not self.enabled:
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.active_timers[name] = time.time()
    
    def end(self, name: str) -> float:
        """
        结束计时
        
        Returns:
            持续时间（秒）
        """
        if not self.enabled:
            return 0.0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if name not in self.active_timers:
            return 0.0
        
        duration = time.time() - self.active_timers[name]
        del self.active_timers[name]
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
        
        return duration
    
    def record_memory(self, label: str):
        """记录显存使用"""
        if not self.enabled or not torch.cuda.is_available():
            return
        
        memory_mb = torch.cuda.memory_allocated() / (1024**2)
        self.memory_snapshots.append((label, memory_mb))
    
    def get_average(self, name: str) -> float:
        """获取平均时间"""
        if name not in self.timings or not self.timings[name]:
            return 0.0
        return sum(self.timings[name]) / len(self.timings[name])
    
    def get_total(self, name: str) -> float:
        """获取总时间"""
        if name not in self.timings:
            return 0.0
        return sum(self.timings[name])
    
    def reset(self):
        """重置所有记录"""
        self.timings.clear()
        self.active_timers.clear()
        self.memory_snapshots.clear()
    
    def print_summary(self):
        """打印性能摘要"""
        if not self.enabled or not self.timings:
            return
        
        print("\n" + "=" * 60)
        print("性能分析摘要")
        print("=" * 60)
        
        for name, times in self.timings.items():
            total = sum(times)
            avg = total / len(times)
            min_t = min(times)
            max_t = max(times)
            
            print(f"\n{name}:")
            print(f"  调用次数: {len(times)}")
            print(f"  总耗时: {total:.4f} 秒")
            print(f"  平均耗时: {avg:.4f} 秒")
            print(f"  最小耗时: {min_t:.4f} 秒")
            print(f"  最大耗时: {max_t:.4f} 秒")
            if len(times) > 1:
                import statistics
                std = statistics.stdev(times)
                print(f"  标准差: {std:.4f} 秒")
        
        print("\n" + "=" * 60)
