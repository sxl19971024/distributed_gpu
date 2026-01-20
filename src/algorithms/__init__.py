"""
分布式科学计算算法库

包含：
- 矩阵运算（matmul, batch_matmul, transpose）
- 卷积操作（conv2d, conv3d）
- 快速傅里叶变换（fft, ifft, fft2d）
- Einstein求和（einsum）
- 归约操作（sum, mean, max, min）
"""

from .matrix_ops import distributed_matmul, distributed_batch_matmul
from .convolution import distributed_conv2d
from .fft import distributed_fft, distributed_fft2d
from .einsum import distributed_einsum
from .reduction import distributed_sum, distributed_mean, distributed_max, distributed_min

__all__ = [
    "distributed_matmul",
    "distributed_batch_matmul",
    "distributed_conv2d",
    "distributed_fft",
    "distributed_fft2d",
    "distributed_einsum",
    "distributed_sum",
    "distributed_mean",
    "distributed_max",
    "distributed_min",
]
