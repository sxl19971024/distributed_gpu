"""
GPU设备管理器

负责GPU设备的管理、显存监控和设备信息获取。
"""

import torch
from typing import Dict, Optional


class GPUManager:
    """GPU设备管理器"""
    
    def __init__(self, gpu_id: int):
        """
        初始化GPU管理器
        
        Args:
            gpu_id: GPU设备ID
        """
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            self.props = torch.cuda.get_device_properties(gpu_id)
            self.name = self.props.name
            self.total_memory = self.props.total_memory
            self.compute_capability = (self.props.major, self.props.minor)
        else:
            self.props = None
            self.name = "CPU"
            self.total_memory = 0
            self.compute_capability = (0, 0)
    
    def get_device(self) -> torch.device:
        """获取torch设备"""
        return self.device
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        获取显存使用信息
        
        Returns:
            包含显存信息的字典（单位：GB）
        """
        if not torch.cuda.is_available():
            return {'total': 0, 'used': 0, 'free': 0, 'usage_percent': 0}
        
        total = self.total_memory / (1024**3)
        used = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.gpu_id) / (1024**3)
        free = total - reserved
        
        return {
            'total': total,
            'used': used,
            'reserved': reserved,
            'free': free,
            'usage_percent': (used / total) * 100 if total > 0 else 0
        }
    
    def print_info(self):
        """打印GPU信息"""
        print(f"[GPU {self.gpu_id}] {self.name}")
        print(f"[GPU {self.gpu_id}] 总显存: {self.total_memory / (1024**3):.2f} GB")
    
    def print_memory_info(self):
        """打印显存使用情况"""
        info = self.get_memory_info()
        print(f"[GPU {self.gpu_id}] 显存使用: {info['used']:.2f} GB / {info['total']:.2f} GB ({info['usage_percent']:.1f}%)")
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量移动到当前GPU"""
        return tensor.to(self.device)
    
    def empty_cache(self):
        """清空GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def synchronize(self):
        """同步GPU操作"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu_id)
    
    def estimate_memory_requirement(self, *tensor_shapes, dtype=torch.float32) -> float:
        """
        估算张量所需显存
        
        Args:
            *tensor_shapes: 张量形状列表
            dtype: 数据类型
        
        Returns:
            所需显存（GB）
        """
        dtype_sizes = {
            torch.float32: 4,
            torch.float64: 8,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int32: 4,
            torch.int64: 8,
        }
        
        bytes_per_element = dtype_sizes.get(dtype, 4)
        total_bytes = 0
        
        for shape in tensor_shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            total_bytes += elements * bytes_per_element
        
        return total_bytes / (1024**3)
    
    def can_fit(self, *tensor_shapes, dtype=torch.float32, safety_margin: float = 0.1) -> bool:
        """
        检查张量是否能放入显存
        
        Args:
            *tensor_shapes: 张量形状列表
            dtype: 数据类型
            safety_margin: 安全边际（0-1）
        
        Returns:
            是否能放入显存
        """
        required = self.estimate_memory_requirement(*tensor_shapes, dtype=dtype)
        info = self.get_memory_info()
        available = info['free'] * (1 - safety_margin)
        return required <= available
