# 基于MPI的分布式GPU计算框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目概述

本项目实现了一个高性能的分布式GPU计算框架，用于处理大规模张量的科学计算任务。通过MPI进行节点间通信，自动将张量切割并分配到各个GPU节点上并行计算，突破单卡显存限制并显著提升计算效率。

## ✨ 核心创新点

### 🎯 创新点1：基于代价模型的自适应张量分割算法
- 建立计算代价、通信代价和显存代价的联合优化模型
- 提出多目标优化的分割策略选择算法
- 支持异构集群的自适应负载均衡

### 🎯 创新点2：计算-通信重叠的流水线优化策略
- 推导最优分块大小的解析表达式
- 多级流水线实现计算-通信重叠
- 理论加速比最高可达2x

### 🎯 创新点3：集成opt_einsum的分布式张量收缩
- 最优收缩路径自动计算
- 支持多种优化策略（optimal/dp/greedy）
- 路径优化与分布式并行双重加速

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/sxl19971024/distributed_gpu.git
cd distributed_gpu

# 安装依赖
pip install -r requirements.txt

# 或使用 pip 安装
pip install -e .
```

### 运行测试

```bash
# 使用4个GPU运行综合测试
mpirun -n 4 --allow-run-as-root python examples/test_all.py
```

### 基本使用

```python
import torch
from src.mpi_manager import MPIManager
from src.tensor_distributor import TensorDistributor
from src.algorithms.matrix_ops import distributed_matmul

# 初始化
mpi = MPIManager()
distributor = TensorDistributor(mpi)

# 创建数据（仅主进程）
if mpi.is_master_process():
    A = torch.randn(10000, 10000).cuda()
    B = torch.randn(10000, 10000).cuda()
else:
    A, B = None, None

# 分布式计算（所有进程都调用）
C = distributed_matmul(A, B, mpi, distributor)

# 结果仅在主进程
if mpi.is_master_process():
    print(f"结果形状: {C.shape}")
```

## 📚 支持的算法

| 类别 | 算法 | 函数 |
|------|------|------|
| **矩阵运算** | 矩阵乘法 | `distributed_matmul` |
| | 批量矩阵乘法 | `distributed_batch_matmul` |
| | 矩阵转置 | `distributed_transpose` |
| | 张量加法 | `distributed_add` |
| **卷积操作** | 2D卷积 | `distributed_conv2d` |
| | 3D卷积 | `distributed_conv3d` |
| **傅里叶变换** | 1D FFT | `distributed_fft` |
| | 1D IFFT | `distributed_ifft` |
| | 2D FFT | `distributed_fft2d` |
| | 实数FFT | `distributed_rfft` |
| **张量收缩** | Einstein求和 | `distributed_einsum` |
| | 张量点积 | `distributed_tensordot` |
| **归约操作** | 求和 | `distributed_sum` |
| | 均值 | `distributed_mean` |
| | 最大值 | `distributed_max` |
| | 最小值 | `distributed_min` |

## 📂 项目结构

```
distributed_gpu_framework/
├── src/
│   ├── __init__.py
│   ├── mpi_manager.py          # MPI通信管理器
│   ├── tensor_distributor.py   # 张量分配器
│   ├── gpu_manager.py          # GPU设备管理
│   ├── cost_model.py           # 代价模型（创新点1）
│   ├── pipeline_optimizer.py   # 流水线优化（创新点2）
│   ├── algorithms/             # 分布式算法库
│   │   ├── matrix_ops.py       # 矩阵运算
│   │   ├── convolution.py      # 卷积操作
│   │   ├── fft.py              # 傅里叶变换
│   │   ├── einsum.py           # Einstein求和（集成opt_einsum）
│   │   └── reduction.py        # 归约操作
│   └── utils/
│       └── profiler.py         # 性能分析工具
├── examples/
│   ├── test_all.py             # 综合测试
│   └── matrix_multiplication.py
├── requirements.txt
├── setup.py
├── INSTALL.md                  # 安装指南
└── README.md
```

## 💻 系统要求

- Python 3.8+
- CUDA 11.0+
- PyTorch 2.0+
- OpenMPI 4.0+ 或 MPICH
- 多个NVIDIA GPU

## 📊 性能测试

在 4×RTX 5090 上的测试结果：

| 算法 | 规模 | 单GPU时间 | 4GPU时间 | 加速比 |
|------|------|-----------|----------|--------|
| 矩阵乘法 | 10000×10000 | 30ms | 8ms | 3.75x |
| 2D卷积 | [32,64,512,512] | 150ms | 42ms | 3.57x |
| 2D FFT | [64,1024,1024] | 25ms | 7ms | 3.57x |

## 📖 文档

- [安装指南](INSTALL.md)
- [API使用指南](docs/API_GUIDE.md) - 每个算法的详细参数和示例
- [使用教程](docs/tutorial.md)（待完善）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📧 联系方式

- 作者: 孙小林
- Email: 1271364457@qq.com

## 📚 引用

如果本框架对您的研究有帮助，请引用：

```bibtex
@software{distributed_gpu_framework,
  author = {孙小林},
  title = {MPI-based Distributed GPU Computing Framework},
  year = {2026},
  url = {https://github.com/sxl19971024/distributed_gpu}
}
```
