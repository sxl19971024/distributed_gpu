# 基于MPI的分布式GPU计算框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目概述

本项目实现了一个面向**科学计算**的高性能分布式GPU计算框架。通过 MPI 进行节点间通信，自动将大规模张量切割并分配到各个GPU节点上并行计算，突破单卡显存限制并显著提升计算效率。

框架提供 **24 个分布式算子 + 2 个流水线操作（共 26 个操作）**，覆盖矩阵运算、卷积、FFT、Einstein 求和、归约操作和 Stencil/PDE 求解，适用于物理模拟、数值求解、大规模线性代数等科学计算场景。

## ✨ 核心创新点

### 🎯 创新点1：基于代价模型的自适应张量分割算法
- 建立了 **计算代价 + 通信代价 + 显存代价** 的联合优化模型
- 支持三种分割策略并自动选择最优方案：
  - **行分割（Row Split）**：按 A 的行分割，适合 M ≫ N
  - **列分割（Column Split）**：按 B 的列分割，适合 N ≫ M
  - **2D 块分割（Block 2D / SUMMA）**：将进程排列为 2D 网格，A 按行、B 按列同时分割，大幅降低每卡显存
- 支持异构集群的自适应负载均衡

### 🎯 创新点2：计算-通信重叠的流水线优化策略
- 基于 CUDA 双流（compute\_stream / comm\_stream）实现 GPU 计算与 CPU 侧 MPI 通信的真正重叠
- 将矩阵乘法分解为 scatter → compute → gather 三阶段流水线
- 推导最优分块大小的解析表达式
- 理论上当计算密集时可完全隐藏通信延迟

### 🎯 创新点3：面向科学计算的创新算子族

| 创新算子 | 核心技术 | 应用场景 |
|---|---|---|
| **混合精度通信** | FP16 传输 + FP32 计算，通信量减半 | 大规模矩阵运算的带宽优化 |
| **Pencil 分解 2D FFT** | 沿变换维度分割 + All-to-All 转置 | 单张超大网格（物理模拟场数据） |
| **Kahan 补偿求和** | float64 中间精度 + 补偿算法，误差 O(ε) | 能量守恒验证、长时间积分 |
| **稀疏感知自适应** | 自动检测稀疏度 → COO 格式广播 | 有限元刚度矩阵、图邻接矩阵 |
| **Stencil + Halo Exchange** | MPI Sendrecv 无死锁边界交换 + conv2d | 热传导、泊松方程、波动方程 |

### 🎯 创新点4：集成 opt\_einsum 的分布式张量收缩
- 自动计算最优收缩路径（支持 optimal / dp / greedy 等策略）
- 路径优化与分布式并行双重加速
- 支持任意 Einstein 求和表达式

### 🎯 创新点5：显存感知的自适应资源调度
- **实时扫描 GPU 可用显存**（使用 `torch.cuda.mem_get_info()` 获取 OS 级别真实可用显存，正确反映其他进程占用）
- **智能执行计划生成**：根据数据总量和各卡可用显存，自动决定分批数量和分割策略
- **超显存自动分批**（Auto-Batching）：数据量超过 GPU 总显存时，自动沿行维度分批处理
  - MatMul 优化：B 矩阵只广播一次，后续批次复用
  - Conv2d 优化：weight/bias 只广播一次
- **20% 安全边际**防止 OOM（其他进程临时分配导致的显存波动）
- **一行式 API**：用户只需提供 CPU 张量，框架全自动完成 GPU 显存扫描 → 资源规划 → 分批执行 → 结果拼接

### 🛡️ 错误处理与容错
- 自定义 `MPIError` 异常类，附带 rank 信息便于调试
- `_safe_call` 包装器捕获 MPI 通信错误，避免无信息死锁
- `check_health` 轻量级心跳检测所有进程存活状态
- scatter 操作前广播错误信息，确保所有 rank 同步退出

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
# 使用4个GPU运行综合测试（17项）
mpirun -n 4 python examples/run_algorithm.py all
```

预期输出：
```
总计: 17/17 通过
```

### 基本使用

```python
import torch
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.algorithms.matrix_ops import distributed_matmul

# 初始化（所有进程都必须执行）
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

### 使用代价模型自动选择最优策略

```python
from distributed_gpu.cost_model import CostModel, ClusterConfig

config = ClusterConfig.from_auto_detect(num_nodes=4)
cost_model = CostModel(config)

# 自动选择行分割 / 列分割 / 2D 块分割
C = distributed_matmul(A, B, mpi, distributor, cost_model=cost_model)
```

### 使用创新算子

```python
from distributed_gpu.algorithms.matrix_ops import distributed_matmul_mixed_precision
from distributed_gpu.algorithms.fft import distributed_fft2d_pencil
from distributed_gpu.algorithms.reduction import distributed_sum_kahan
from distributed_gpu.algorithms.stencil import distributed_stencil_2d

# 混合精度矩阵乘法（通信量减半）
C = distributed_matmul_mixed_precision(A, B, mpi, distributor)

# Pencil 2D FFT（处理单张超大网格）
spectrum = distributed_fft2d_pencil(field, mpi, distributor)

# Kahan 补偿求和（数值稳定）
total = distributed_sum_kahan(tensor, mpi, distributor)

# Stencil 计算（物理模拟）
result = distributed_stencil_2d(grid, mpi, distributor, iterations=100)
```

### 一行式自动化 API（显存感知 + 自动分批）

```python
from distributed_gpu.auto_executor import AutoExecutor

executor = AutoExecutor()  # 初始化（所有进程）

# 查看 GPU 实时显存状态
executor.gpu_status()

# 用户只需提供 CPU 张量（仅 master 进程）
if executor.is_master:
    A = torch.randn(100000, 10000)  # 3.7 GB，可能超单卡显存
    B = torch.randn(10000, 5000)
else:
    A = B = None

# 框架自动：扫描显存 → 生成分批计划 → 分布式执行 → 返回 CPU 结果
C = executor.matmul(A, B)  # 自动分批 + 自动选择最优分割策略

# 同样支持 FFT、Sum、Conv2d、Einsum 等
Y = executor.fft(signal)
S = executor.sum(data)
```

## 📚 支持的算子（25 个）

| 类别 | 算子 | 函数 | 说明 |
|------|------|------|------|
| **矩阵运算** | 矩阵乘法 | `distributed_matmul` | 行/列/2D块分割，代价模型自动选择 |
| | 批量矩阵乘法 | `distributed_batch_matmul` | 按 batch 维度并行 |
| | 矩阵转置 | `distributed_transpose` | 支持任意维度交换 |
| | 张量加法 | `distributed_add` | 逐元素并行 |
| | ⭐ 混合精度矩阵乘法 | `distributed_matmul_mixed_precision` | FP16 通信 + FP32 计算 |
| | ⭐ 稀疏感知矩阵乘法 | `distributed_matmul_sparse_aware` | 自动检测稀疏度，COO 格式广播 |
| **流水线** | 流水线矩阵乘法 | `PipelineOptimizer.pipelined_matmul` | CUDA 双流计算-通信重叠 |
| | 流水线 AllReduce | `PipelineOptimizer.pipelined_allreduce` | 分块异步归约 |
| **卷积** | 2D 卷积 | `distributed_conv2d` | 按 batch 分割，支持 bias |
| **傅里叶变换** | 1D FFT | `distributed_fft` | 按 batch 分割 |
| | 1D IFFT | `distributed_ifft` | 逆变换 |
| | 2D FFT | `distributed_fft2d` | 按 batch 分割 |
| | 实数 FFT | `distributed_rfft` | 正频率优化，计算/通信量减半 |
| | ⭐ Pencil 2D FFT | `distributed_fft2d_pencil` | All-to-All 转置，支持超大单网格 |
| **张量收缩** | Einstein 求和 | `distributed_einsum` | 集成 opt\_einsum 最优路径 |
| | 带路径 Einstein 求和 | `distributed_einsum_with_path` | 复用预计算路径 |
| | 张量点积 | `distributed_tensordot` | 任意维度收缩 |
| **归约操作** | 求和 / 均值 | `distributed_sum` / `distributed_mean` | 全局或按维度归约 |
| | 最大 / 最小值 | `distributed_max` / `distributed_min` | 全局或按维度归约 |
| | ⭐ Kahan 补偿求和 | `distributed_sum_kahan` | 误差 O(ε) 而非 O(n·ε) |
| | ⭐ Kahan 补偿均值 | `distributed_mean_kahan` | 基于 Kahan 的高精度均值 |
| **Stencil / PDE** | ⭐ 2D Stencil | `distributed_stencil_2d` | Halo Exchange + conv2d |
| | ⭐ Jacobi 迭代 | `distributed_jacobi_2d` | 求解 ∇²u = f，自动收敛检测 |

> ⭐ 标记为本框架的**创新算子**

## 📂 项目结构

```
distributed_gpu_framework/
├── distributed_gpu/
│   ├── __init__.py              # 包入口，导出核心类
│   ├── mpi_manager.py           # MPI通信管理器（含错误处理/容错）
│   ├── tensor_distributor.py    # 张量分配器（1D/2D分割 + 混合精度压缩）
│   ├── gpu_manager.py           # GPU设备管理 / 显存监控
│   ├── cost_model.py            # 代价模型与自适应策略选择
│   ├── pipeline_optimizer.py    # CUDA双流流水线优化
│   ├── resource_planner.py     # 显存感知资源规划器（实时扫描+分批策略）
│   ├── auto_executor.py        # 自动化执行器（一行式API+超显存分批）
│   ├── algorithms/              # 分布式算法库
│   │   ├── matrix_ops.py        # 矩阵运算（行/列/2D + 混合精度 + 稀疏感知）
│   │   ├── convolution.py       # 卷积操作
│   │   ├── fft.py               # FFT（1D/2D/实数/Pencil分解）
│   │   ├── einsum.py            # Einstein求和（集成opt_einsum）
│   │   ├── reduction.py         # 归约操作（含Kahan补偿求和）
│   │   └── stencil.py           # Stencil计算 + Halo Exchange + Jacobi迭代
│   └── utils/
│       └── profiler.py          # 性能分析工具
├── examples/
│   ├── run_algorithm.py         # 算法测试工具（24个算子 + 交互式选择）
│   └── test_auto_executor.py   # AutoExecutor 测试（自动分批/单卡快速路径）
├── docs/
│   └── API_GUIDE.md             # API 详细使用指南
├── requirements.txt
├── setup.py
├── INSTALL.md                   # 安装指南
└── README.md
```

## 💻 系统要求

- Python 3.8+
- CUDA 11.0+
- PyTorch 2.0+
- OpenMPI 4.0+ 或 MPICH
- 多个 NVIDIA GPU

## 📖 文档

- [安装指南](INSTALL.md) — 环境配置与依赖安装
- [API使用指南](docs/API_GUIDE.md) — 每个算法的详细参数、示例和高级功能

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
  title = {MPI-based Distributed GPU Computing Framework for Scientific Computing},
  year = {2026},
  url = {https://github.com/sxl19971024/distributed_gpu}
}
```
