"""
GPU 设备管理器 (GPU Device Manager)

负责 GPU 设备的管理、显存监控和设备信息获取。
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple, Union

import torch

# ── 常量 ──────────────────────────────────────────────────────
_GB: float = 1024.0 ** 3

_DTYPE_BYTES: Dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
    torch.complex64: 8,
    torch.complex128: 16,
}


class GPUManager:
    """
    GPU 设备管理器

    职责：
    - 管理单个 GPU 设备的生命周期
    - 提供实时显存监控
    - 估算张量显存需求
    """

    def __init__(self, gpu_id: int) -> None:
        """
        Args:
            gpu_id: GPU 设备 ID
        """
        self.gpu_id: int = gpu_id
        self.device: torch.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        if torch.cuda.is_available():
            self.props = torch.cuda.get_device_properties(gpu_id)
            self.name: str = self.props.name
            self.total_memory: int = self.props.total_memory
            self.compute_capability: Tuple[int, int] = (self.props.major, self.props.minor)
        else:
            self.props = None
            self.name = "CPU"
            self.total_memory = 0
            self.compute_capability = (0, 0)

    # ── 设备查询 ────────────────────────────────────────────────

    def get_device(self) -> torch.device:
        """获取 ``torch.device``。"""
        return self.device

    def get_memory_info(self) -> Dict[str, float]:
        """
        获取显存使用信息。

        Returns:
            包含显存信息的字典（单位：GB），键包括
            ``total``, ``used``, ``reserved``, ``free``, ``usage_percent``。
        """
        if not torch.cuda.is_available():
            return {"total": 0.0, "used": 0.0, "reserved": 0.0,
                    "free": 0.0, "usage_percent": 0.0}

        total = self.total_memory / _GB
        used = torch.cuda.memory_allocated(self.gpu_id) / _GB
        reserved = torch.cuda.memory_reserved(self.gpu_id) / _GB
        free = total - reserved

        return {
            "total": total,
            "used": used,
            "reserved": reserved,
            "free": free,
            "usage_percent": (used / total * 100.0) if total > 0 else 0.0,
        }

    # ── 信息打印 ────────────────────────────────────────────────

    def print_info(self) -> None:
        """打印 GPU 基本信息。"""
        print(f"[GPU {self.gpu_id}] {self.name}")
        print(f"[GPU {self.gpu_id}] 总显存: {self.total_memory / _GB:.2f} GB")

    def print_memory_info(self) -> None:
        """打印当前显存使用情况。"""
        info = self.get_memory_info()
        print(
            f"[GPU {self.gpu_id}] 显存使用: {info['used']:.2f} GB "
            f"/ {info['total']:.2f} GB ({info['usage_percent']:.1f}%)"
        )

    # ── 设备操作 ────────────────────────────────────────────────

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量移动到当前 GPU。"""
        return tensor.to(self.device)

    def empty_cache(self) -> None:
        """清空 GPU 缓存。"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def synchronize(self) -> None:
        """同步 GPU 操作。"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.gpu_id)

    # ── 显存估算 ────────────────────────────────────────────────

    @staticmethod
    def estimate_memory_requirement(
        *tensor_shapes: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> float:
        """
        估算给定张量形状所需显存。

        Args:
            *tensor_shapes: 张量形状（如 ``(1024, 1024)``）
            dtype: 数据类型

        Returns:
            所需显存（GB）
        """
        bytes_per_element = _DTYPE_BYTES.get(dtype, 4)
        total_bytes = sum(math.prod(shape) * bytes_per_element for shape in tensor_shapes)
        return total_bytes / _GB

    def can_fit(
        self,
        *tensor_shapes: Sequence[int],
        dtype: torch.dtype = torch.float32,
        safety_margin: float = 0.1,
    ) -> bool:
        """
        检查张量是否能放入当前 GPU 显存。

        Args:
            *tensor_shapes: 张量形状
            dtype: 数据类型
            safety_margin: 安全边际比例 (0–1)

        Returns:
            ``True`` 表示可以放入
        """
        required = self.estimate_memory_requirement(*tensor_shapes, dtype=dtype)
        info = self.get_memory_info()
        available = info["free"] * (1.0 - safety_margin)
        return required <= available
