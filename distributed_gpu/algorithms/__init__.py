"""
分布式科学计算算法库（24 个算子）

包含：
- 矩阵运算 (6)：matmul, batch_matmul, transpose, add, 混合精度, 稀疏感知
- 卷积操作 (2)：conv2d, conv3d
- 快速傅里叶变换 (5)：fft, ifft, fft2d, rfft, pencil fft2d
- Einstein求和 (3)：einsum, einsum_with_path, tensordot
- 归约操作 (6)：sum, mean, max, min, Kahan补偿求和, Kahan补偿均值
- Stencil / PDE (2)：stencil_2d (halo exchange), jacobi_2d
"""

from .matrix_ops import (distributed_matmul, distributed_batch_matmul,
                         distributed_transpose, distributed_add,
                         distributed_matmul_mixed_precision,
                         distributed_matmul_sparse_aware)
from .convolution import distributed_conv2d, distributed_conv3d
from .fft import (distributed_fft, distributed_ifft, distributed_fft2d,
                  distributed_rfft, distributed_fft2d_pencil)
from .einsum import (distributed_einsum, distributed_einsum_with_path,
                     distributed_tensordot)
from .reduction import (distributed_sum, distributed_mean,
                        distributed_max, distributed_min,
                        distributed_sum_kahan, distributed_mean_kahan)
from .stencil import distributed_stencil_2d, distributed_jacobi_2d

__all__ = [
    # 矩阵运算 (6)
    "distributed_matmul",
    "distributed_batch_matmul",
    "distributed_transpose",
    "distributed_add",
    "distributed_matmul_mixed_precision",
    "distributed_matmul_sparse_aware",
    # 卷积 (2)
    "distributed_conv2d",
    "distributed_conv3d",
    # FFT (5)
    "distributed_fft",
    "distributed_ifft",
    "distributed_fft2d",
    "distributed_rfft",
    "distributed_fft2d_pencil",
    # Einstein 求和 (3)
    "distributed_einsum",
    "distributed_einsum_with_path",
    "distributed_tensordot",
    # 归约 (6)
    "distributed_sum",
    "distributed_mean",
    "distributed_max",
    "distributed_min",
    "distributed_sum_kahan",
    "distributed_mean_kahan",
    # Stencil / PDE (2)
    "distributed_stencil_2d",
    "distributed_jacobi_2d",
]
