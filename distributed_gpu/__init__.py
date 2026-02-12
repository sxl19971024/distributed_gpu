"""
分布式GPU计算框架

基于MPI的分布式GPU计算加速方案，面向大规模科学计算。
提供 24 个分布式算子 + 2 个流水线操作（共 26 个操作），
覆盖矩阵运算、卷积、FFT、Einstein 求和、归约操作和 Stencil/PDE 求解。

核心创新点：
1. 基于代价模型的自适应张量分割算法（行/列/2D块，自动选择）
2. 计算-通信重叠的 CUDA 双流流水线优化策略
3. 面向科学计算的创新算子族（混合精度通信、Pencil FFT、Kahan 求和、
   稀疏感知、Stencil + Halo Exchange）
4. 集成 opt_einsum 的分布式张量收缩
5. 显存感知的自适应资源调度（实时扫描可用显存、自动分批、一行式 API）
"""

from .mpi_manager import MPIManager, MPIError
from .tensor_distributor import TensorDistributor
from .gpu_manager import GPUManager
from .cost_model import CostModel, ClusterConfig, SplitStrategy
from .pipeline_optimizer import PipelineOptimizer, PipelineConfig
from .resource_planner import ResourcePlanner, ExecutionPlan, GPUStatus
from .auto_executor import AutoExecutor, auto_compute

__version__ = "1.4.0"
__all__ = [
    "MPIManager",
    "MPIError",
    "TensorDistributor", 
    "GPUManager",
    "CostModel",
    "ClusterConfig",
    "SplitStrategy",
    "PipelineOptimizer",
    "PipelineConfig",
    "ResourcePlanner",
    "ExecutionPlan",
    "GPUStatus",
    "AutoExecutor",
    "auto_compute",
]
