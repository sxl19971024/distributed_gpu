"""
分布式GPU计算框架

基于MPI的分布式GPU计算加速方案，用于大规模科学计算。

核心创新点：
1. 基于代价模型的自适应张量分割算法
2. 计算-通信重叠的流水线优化策略
"""

from .mpi_manager import MPIManager
from .tensor_distributor import TensorDistributor
from .gpu_manager import GPUManager
from .cost_model import CostModel, ClusterConfig, SplitStrategy
from .pipeline_optimizer import PipelineOptimizer

__version__ = "1.0.0"
__all__ = [
    "MPIManager",
    "TensorDistributor", 
    "GPUManager",
    "CostModel",
    "ClusterConfig",
    "SplitStrategy",
    "PipelineOptimizer",
]
