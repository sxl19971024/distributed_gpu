# API 使用指南

本文档详细说明框架中每个模块和分布式算法的使用方法、参数和示例。

## 📋 目录

1. [初始化框架](#初始化框架)
2. [矩阵运算](#矩阵运算)
3. [创新算子：混合精度与稀疏感知](#创新算子混合精度与稀疏感知)
4. [卷积操作](#卷积操作)
5. [傅里叶变换](#傅里叶变换)
6. [创新算子：Pencil 分解 2D FFT](#创新算子pencil-分解-2d-fft)
7. [Einstein求和](#einstein求和)
8. [归约操作](#归约操作)
9. [创新算子：Kahan 补偿求和](#创新算子kahan-补偿求和)
10. [创新算子：Stencil 计算与 Jacobi 迭代](#创新算子stencil-计算与-jacobi-迭代)
11. [代价模型与自适应策略](#代价模型与自适应策略)
12. [流水线优化器](#流水线优化器)
13. [错误处理与容错](#错误处理与容错)
14. [GPU 管理与性能分析](#gpu-管理与性能分析)

---

## 初始化框架

在使用任何分布式算法之前，必须先初始化 MPI 环境和张量分配器。

```python
import torch
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor

# 初始化（所有进程都必须执行）
mpi = MPIManager()
distributor = TensorDistributor(mpi)

# 获取当前进程信息
print(f"进程 {mpi.get_rank()}/{mpi.get_size()}, GPU {mpi.get_gpu_id()}")
```

**重要原则**：
- ⚠️ 所有进程都必须调用相同的分布式函数（否则 MPI 会死锁）
- ⚠️ 输入数据仅在主进程（rank=0）提供，其他进程传 `None`
- ⚠️ 输出结果仅主进程返回有效值，其他进程返回 `None`

---

## 矩阵运算

### 1. distributed_matmul — 矩阵乘法

**功能**：计算 C = A @ B，支持三种分割策略，可通过代价模型自动选择。

```python
from distributed_gpu.algorithms.matrix_ops import distributed_matmul
from distributed_gpu.cost_model import CostModel, ClusterConfig, SplitStrategy

C = distributed_matmul(
    A,                    # torch.Tensor [M, K]（仅主进程）
    B,                    # torch.Tensor [K, N]（仅主进程）
    mpi,                  # MPIManager
    distributor,          # TensorDistributor
    cost_model=None,      # CostModel: 传入则自动选择最优策略
    strategy=None         # SplitStrategy: 强制指定策略（优先级高于 cost_model）
)
# 返回: torch.Tensor [M, N]（仅主进程）
```

**strategy 参数（可选）**：

| 值 | 说明 | 适用场景 |
|---|---|---|
| `SplitStrategy.ROW_SPLIT` | 按 A 的行分割 | M ≫ N |
| `SplitStrategy.COLUMN_SPLIT` | 按 B 的列分割 | N ≫ M |
| `SplitStrategy.BLOCK_2D` | 2D 块分割（SUMMA 风格） | M ≈ N，降低每卡显存 |

**示例 — 默认行分割**：

```python
if mpi.is_master_process():
    A = torch.randn(5000, 3000).cuda()
    B = torch.randn(3000, 4000).cuda()
else:
    A, B = None, None

C = distributed_matmul(A, B, mpi, distributor)
```

**示例 — 代价模型自动选择**：

```python
config = ClusterConfig.from_auto_detect(num_nodes=mpi.get_size())
cost_model = CostModel(config)

C = distributed_matmul(A, B, mpi, distributor, cost_model=cost_model)
# 框架会自动在行分割/列分割/2D块分割中选择预估耗时最短的策略
```

---

### 2. distributed_batch_matmul — 批量矩阵乘法

```python
from distributed_gpu.algorithms.matrix_ops import distributed_batch_matmul

C = distributed_batch_matmul(
    A,              # torch.Tensor [batch, M, K]
    B,              # torch.Tensor [batch, K, N] 或 [K, N]
    mpi,
    distributor
)
# 返回: torch.Tensor [batch, M, N]
```

---

### 3. distributed_transpose — 矩阵转置

```python
from distributed_gpu.algorithms.matrix_ops import distributed_transpose

A_T = distributed_transpose(A, mpi, distributor, dim0=0, dim1=1)
```

---

### 4. distributed_add — 张量加法

```python
from distributed_gpu.algorithms.matrix_ops import distributed_add

C = distributed_add(A, B, mpi, distributor)  # C = A + B
```

---

## 创新算子：混合精度与稀疏感知

### 5. distributed_matmul_mixed_precision — 混合精度矩阵乘法 ⭐

**功能**：通信时使用 FP16，计算时使用 FP32。通信量减少约 50%。

**误差上界**：‖C\_mixed - C\_exact‖ ≤ O(ε\_FP16 × √K × ‖A‖ × ‖B‖)

```python
from distributed_gpu.algorithms.matrix_ops import distributed_matmul_mixed_precision

C = distributed_matmul_mixed_precision(
    A,                    # torch.Tensor [M, K]（仅主进程）
    B,                    # torch.Tensor [K, N]（仅主进程）
    mpi,
    distributor,
    comm_dtype=torch.float16  # 通信精度（可选 torch.bfloat16）
)
```

**适用场景**：
- 大矩阵乘法中通信成为瓶颈时
- 结果精度要求在 1e-3 量级即可的场景
- 迭代算法中间步骤（最终结果在全精度下收敛）

---

### 6. distributed_matmul_sparse_aware — 稀疏感知矩阵乘法 ⭐

**功能**：自动检测矩阵稀疏度，决定使用稠密或稀疏（COO 格式广播）路径。

```python
from distributed_gpu.algorithms.matrix_ops import distributed_matmul_sparse_aware

C = distributed_matmul_sparse_aware(
    A,                        # torch.Tensor [M, K]（仅主进程）
    B,                        # torch.Tensor [K, N]（仅主进程）
    mpi,
    distributor,
    sparsity_threshold=0.5    # 稀疏度阈值（超过则使用COO广播）
)
```

**通信量对比**：
- 稠密广播 B：K × N × 4 字节
- COO 广播 B：nnz × 20 字节（稀疏度 80% 时通信量减少 60%）

**适用场景**：有限元刚度矩阵、图邻接矩阵、稀疏物理相互作用矩阵。

---

## 卷积操作

### distributed_conv2d — 2D 卷积

```python
from distributed_gpu.algorithms.convolution import distributed_conv2d

output = distributed_conv2d(
    input,              # torch.Tensor [N, C_in, H, W]
    weight,             # torch.Tensor [C_out, C_in, kH, kW]
    mpi,
    distributor,
    bias=None,          # torch.Tensor [C_out]（可选）
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1
)
```

---

## 傅里叶变换

### 1. distributed_fft — 1D FFT

```python
from distributed_gpu.algorithms.fft import distributed_fft

output = distributed_fft(input, mpi, distributor, n=None, dim=-1, norm="backward")
```

### 2. distributed_ifft — 逆 FFT

```python
from distributed_gpu.algorithms.fft import distributed_ifft

output = distributed_ifft(input, mpi, distributor, n=None, dim=-1, norm="backward")
```

### 3. distributed_fft2d — 2D FFT（按 batch 分割）

```python
from distributed_gpu.algorithms.fft import distributed_fft2d

output = distributed_fft2d(input, mpi, distributor, s=None, dim=(-2, -1), norm="backward")
```

**注意**：此函数按 batch 维度（dim=0）分割，要求输入至少为 3D `[batch, H, W]`。适用于多个小/中网格的并行 FFT。对于单张超大 2D 网格，请使用 `distributed_fft2d_pencil`。

### 4. distributed_rfft — 实数 FFT

```python
from distributed_gpu.algorithms.fft import distributed_rfft

output = distributed_rfft(input, mpi, distributor, n=None, dim=-1, norm="backward")
# 输出长度: N/2 + 1（利用共轭对称性，计算量和通信量约减半）
```

**适用场景**：科学计算中绝大多数物理量（温度、压力、速度、电磁场）都是实数。

---

## 创新算子：Pencil 分解 2D FFT

### distributed_fft2d_pencil ⭐

**功能**：基于 Pencil 分解的分布式 2D FFT。沿变换维度分割（而非 batch 维度），能处理**单张超大 2D 网格**。

```python
from distributed_gpu.algorithms.fft import distributed_fft2d_pencil

output = distributed_fft2d_pencil(
    input,              # torch.Tensor [H, W]（仅主进程，2D）
    mpi,
    distributor,
    norm="backward"
)
```

**算法步骤**：
1. 按行分发 → 各进程持有 `[local_H, W]`
2. 沿 W 方向 1D FFT
3. **All-to-All 转置** → 各进程持有 `[H, local_W]`
4. 沿 H 方向 1D FFT
5. **All-to-All 转置回** → `[local_H, W]`
6. 收集结果

**要求**：H 和 W 都能被进程数 P 整除。

**与 distributed_fft2d 的区别**：

| | `distributed_fft2d` | `distributed_fft2d_pencil` |
|---|---|---|
| 输入 | `[batch, H, W]` | `[H, W]` |
| 分割方式 | batch 维 | 空间维（H → W） |
| 通信模式 | scatter + gather | **All-to-All** |
| 适用场景 | 多个小网格 | 单个超大网格 |

---

## Einstein 求和

### 1. distributed_einsum

```python
from distributed_gpu.algorithms.einsum import distributed_einsum

result = distributed_einsum(
    equation,               # str: 'bij,bjk->bik'
    *operands,              # torch.Tensor
    mpi=mpi,
    distributor=distributor,
    optimize='auto',
    use_opt_einsum=True
)
```

### 2. distributed_einsum_with_path — 带预计算路径

```python
from distributed_gpu.algorithms.einsum import distributed_einsum_with_path, get_optimal_path

path, info = get_optimal_path('ij,jk,kl->il', (100,50), (50,80), (80,40))
result = distributed_einsum_with_path('ij,jk,kl->il', A, B, C,
                                       mpi=mpi, distributor=distributor, path=path)
```

### 3. distributed_tensordot

```python
from distributed_gpu.algorithms.einsum import distributed_tensordot

result = distributed_tensordot(a, b, dims=2, mpi=mpi, distributor=distributor)
```

---

## 归约操作

### distributed_sum / distributed_mean / distributed_max / distributed_min

```python
from distributed_gpu.algorithms.reduction import distributed_sum, distributed_mean
from distributed_gpu.algorithms.reduction import distributed_max, distributed_min

total = distributed_sum(tensor, mpi, distributor, dim=None)
avg = distributed_mean(tensor, mpi, distributor, dim=0, keepdim=True)
maximum = distributed_max(tensor, mpi, distributor)
minimum = distributed_min(tensor, mpi, distributor)
```

**参数**：
- `dim=None`：全局归约（返回标量）
- `dim=0`：沿分割维度归约（需要 allreduce）
- `dim=其他`：本地归约后 gather

---

## 创新算子：Kahan 补偿求和

### distributed_sum_kahan / distributed_mean_kahan ⭐

**功能**：使用 float64 中间精度 + Kahan 补偿算法，误差从 O(n·ε) 降低到 O(ε)。

```python
from distributed_gpu.algorithms.reduction import distributed_sum_kahan, distributed_mean_kahan

# 数值稳定的全局求和
total = distributed_sum_kahan(tensor, mpi, distributor, dim=None)

# 数值稳定的全局均值
avg = distributed_mean_kahan(tensor, mpi, distributor, dim=None)
```

**精度对比**（实测）：

| 方法 | 误差量级 | 性能 |
|---|---|---|
| `distributed_sum` | ~1e-5 | 快 |
| `distributed_sum_kahan` | ~1e-7 | 较慢（float64 + 块补偿） |

**适用场景**：
- 能量守恒验证（误差累积 → 物理结果失真）
- 长时间步积分
- 多次迭代中间值累加

---

## 创新算子：Stencil 计算与 Jacobi 迭代

### distributed_stencil_2d ⭐

**功能**：分布式 2D Stencil 计算，通过 Halo Exchange 交换边界数据。

```python
from distributed_gpu.algorithms.stencil import distributed_stencil_2d

result = distributed_stencil_2d(
    grid,                   # torch.Tensor [H, W]（仅主进程）
    mpi,
    distributor,
    stencil_kernel=None,    # 默认: 5点Laplacian [[0,1,0],[1,-4,1],[0,1,0]]
    boundary='zero',        # 'zero' 或 'periodic'
    iterations=1            # 迭代次数
)
```

**Halo Exchange 实现**：
- 使用 MPI `Sendrecv`（大写/缓冲区版本）+ `MPI.PROC_NULL` 边界处理
- **无死锁保证**：所有进程同步参与通信
- 通信量：每次迭代每进程仅 O(W) 字节（边界行）

**预定义 Stencil 核**：

```python
from distributed_gpu.algorithms.stencil import DEFAULT_LAPLACIAN_5PT, DEFAULT_LAPLACIAN_9PT

# 5点: [[0,1,0],[1,-4,1],[0,1,0]]    — 二阶精度
# 9点: [[1,4,1],[4,-20,4],[1,4,1]]/6  — 更高精度
```

---

### distributed_jacobi_2d ⭐

**功能**：分布式 Jacobi 迭代求解 2D 泊松方程 ∇²u = f。

```python
from distributed_gpu.algorithms.stencil import distributed_jacobi_2d

solution = distributed_jacobi_2d(
    grid,                   # 初始猜测 [H, W]（仅主进程）
    rhs,                    # 右端项 f [H, W]（仅主进程）
    mpi,
    distributor,
    dx=1.0,                 # 网格间距
    boundary='zero',
    iterations=1000,        # 最大迭代次数
    tol=1e-6                # 收敛容差
)
```

**特性**：自动收敛检测（全局残差 < tol 时提前终止）。

---

## 代价模型与自适应策略

### ClusterConfig

```python
from distributed_gpu.cost_model import ClusterConfig

config = ClusterConfig.from_auto_detect(num_nodes=4)  # 自动检测
```

### CostModel

```python
from distributed_gpu.cost_model import CostModel, SplitStrategy

cost_model = CostModel(config)

# 估算特定策略的代价
cost = cost_model.estimate_matmul_cost(M=5000, K=5000, N=5000,
                                        strategy=SplitStrategy.ROW_SPLIT)

# 自动选择最优策略
plan = cost_model.find_optimal_strategy(M=5000, K=5000, N=5000)
print(f"推荐: {plan.strategy.value}")

# 打印完整分析
cost_model.print_analysis(M=5000, K=5000, N=5000)
```

---

## 流水线优化器

```python
from distributed_gpu.pipeline_optimizer import PipelineOptimizer, PipelineConfig

pipeline = PipelineOptimizer(mpi, PipelineConfig(num_chunks=4))

# 流水线矩阵乘法
C = pipeline.pipelined_matmul(A, B, num_chunks=4)

# 流水线 AllReduce
result = pipeline.pipelined_allreduce(tensor, num_chunks=4)

# 理论收益估算
benefit = pipeline.estimate_overlap_benefit(
    compute_time_ms=10.0, comm_time_ms=5.0, num_chunks=4
)
```

---

## 错误处理与容错

### MPIError

```python
from distributed_gpu.mpi_manager import MPIError

try:
    C = distributed_matmul(A, B, mpi, distributor)
except MPIError as e:
    print(f"MPI 错误 (rank {e.rank}): {e}")
```

### check_health

```python
alive = mpi.check_health()  # 轻量级心跳检测
```

---

## GPU 管理与性能分析

### GPUManager

```python
from distributed_gpu.gpu_manager import GPUManager

gpu = GPUManager(mpi.get_gpu_id())
gpu.print_info()
gpu.print_memory_info()
```

### Profiler

```python
from distributed_gpu.utils.profiler import Profiler

profiler = Profiler(enabled=mpi.is_master_process())
profiler.start("matmul")
C = distributed_matmul(A, B, mpi, distributor)
profiler.end("matmul")
profiler.print_summary()
```

### 预热（Warmup）

```python
# GPU / MPI 预热：消除首次调用的 CUDA 内核编译和 MPI 建链延迟
from distributed_gpu.algorithms.matrix_ops import distributed_matmul
for _ in range(3):
    if mpi.is_master_process():
        t = torch.randn(64, 64).cuda()
    else:
        t = None
    distributed_matmul(t, t, mpi, distributor)
mpi.synchronize()
```

---

## 完整示例

```python
#!/usr/bin/env python
"""运行: mpirun -n 4 python example.py"""
import torch
from distributed_gpu.mpi_manager import MPIManager
from distributed_gpu.tensor_distributor import TensorDistributor
from distributed_gpu.cost_model import CostModel, ClusterConfig
from distributed_gpu.algorithms.matrix_ops import (distributed_matmul,
                                        distributed_matmul_mixed_precision,
                                        distributed_matmul_sparse_aware)
from distributed_gpu.algorithms.fft import distributed_fft2d_pencil
from distributed_gpu.algorithms.reduction import distributed_sum_kahan
from distributed_gpu.algorithms.stencil import distributed_stencil_2d, distributed_jacobi_2d

def main():
    mpi = MPIManager()
    distributor = TensorDistributor(mpi)

    # === 矩阵乘法（代价模型自动选择策略）===
    config = ClusterConfig.from_auto_detect(mpi.get_size())
    cost_model = CostModel(config)
    
    if mpi.is_master_process():
        A = torch.randn(2000, 1500).cuda()
        B = torch.randn(1500, 1000).cuda()
    else:
        A, B = None, None
    
    C = distributed_matmul(A, B, mpi, distributor, cost_model=cost_model)
    
    # === 混合精度矩阵乘法（通信量减半）===
    C_mp = distributed_matmul_mixed_precision(A, B, mpi, distributor)
    
    # === Pencil 2D FFT（超大单网格）===
    if mpi.is_master_process():
        field = torch.randn(1024, 1024).cuda()
    else:
        field = None
    spectrum = distributed_fft2d_pencil(field, mpi, distributor)
    
    # === Kahan 补偿求和（高精度）===
    if mpi.is_master_process():
        data = torch.randn(1000, 1000, 100).cuda()
    else:
        data = None
    total = distributed_sum_kahan(data, mpi, distributor)
    
    # === Stencil 计算（物理模拟）===
    if mpi.is_master_process():
        grid = torch.zeros(256, 256).cuda()
        grid[128, 128] = 1000.0  # 热源
        rhs = torch.zeros(256, 256).cuda()
    else:
        grid, rhs = None, None
    
    result = distributed_stencil_2d(grid, mpi, distributor, iterations=50)
    
    # === Jacobi 迭代（泊松方程求解）===
    solution = distributed_jacobi_2d(grid, rhs, mpi, distributor,
                                      dx=0.01, iterations=500, tol=1e-5)
    
    if mpi.is_master_process():
        print("所有操作完成！")

if __name__ == "__main__":
    main()
```

---

## 常见问题

### Q1: 为什么其他进程要传 None？
数据只在主进程存在，分布式函数内部自动将数据分发到其他进程。

### Q2: 为什么所有进程都要调用函数？
MPI 集合通信要求所有进程同步参与，否则会死锁。

### Q3: distributed_sum 和 distributed_sum_kahan 有什么区别？
前者速度快（~0.01s），后者精度高（误差 O(ε) vs O(n·ε)，约慢 10-15 倍）。普通场景用 sum，能量守恒验证等高精度需求用 sum_kahan。

### Q4: distributed_fft2d 和 distributed_fft2d_pencil 有什么区别？
前者按 batch 分割（适合多个小网格），后者按空间维度分割（适合单张超大网格）。单个 2D 网格只能用 pencil 版本。

### Q5: 如何调整 GPU 数量？
修改 `mpirun -n <数量>` 参数即可。2D 块分割要求进程数可分解为 ≥2×2 的网格。Pencil FFT 要求 H 和 W 都能被进程数整除。

---

## AutoExecutor — 显存感知自动化分布式计算

### 概述

`AutoExecutor` 是框架的高层用户接口，实现了**显存感知的自适应资源调度**。
用户只需提供 CPU 张量，框架自动完成：

1. **实时扫描** 所有 GPU 的真实可用显存（OS 级别，含其他进程占用）
2. **智能规划** 根据数据量和可用显存决定分批策略
3. **自动执行** 分布式计算 + 分批处理 + 结果拼接
4. **返回结果** CPU 张量（仅 master 进程）

### 初始化

```python
from distributed_gpu.auto_executor import AutoExecutor

# 所有进程都必须调用
executor = AutoExecutor(
    verbose=True,           # 是否打印执行信息
    max_per_gpu_gb=None,    # 人为限制每GPU可用显存 (测试用)
)
```

### GPU 显存状态查看

```python
executor.gpu_status()
# 输出:
# ======================================================================
#   GPU 显存状态（实时检测，含其他进程占用）
# ----------------------------------------------------------------------
#   GPU 0 (Rank  0): [██████████████████████████░░░░] 4.1 / 31.4 GB 空闲 | 安全可用 3.3 GB
#   GPU 1 (Rank  1): [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 30.4 / 31.4 GB 空闲 | 安全可用 24.3 GB
# ======================================================================
```

### MatMul（支持超显存自动分批）

```python
if executor.is_master:
    A = torch.randn(100000, 10000)  # CPU 张量
    B = torch.randn(10000, 5000)
else:
    A = B = None

C = executor.matmul(A, B)  # 自动: 显存扫描→分批规划→分布式计算→拼接返回
# C 是 CPU tensor (仅 master)
```

**分批优化**：当数据超过 GPU 总显存时，A 沿行分批，B 只广播一次到所有 GPU 并复用。

### FFT / RFFT / FFT2D

```python
result = executor.fft(signal_cpu)
result = executor.rfft(signal_cpu)
result = executor.fft2d(field_cpu)
```

### Sum / Mean

```python
total = executor.sum(data_cpu)
avg = executor.mean(data_cpu, dim=0)
```

### Conv2d

```python
output = executor.conv2d(input_cpu, weight_cpu, bias=bias_cpu, padding=(1,1))
```

**分批优化**：weight 和 bias 只广播一次，input 沿 batch 维度分批。

### Einsum

```python
C = executor.einsum("ij,jk->ik", A_cpu, B_cpu)
```

### 批量 MatMul

```python
if executor.is_master:
    pairs = [(A1, B1), (A2, B2), (A3, B3)]
else:
    pairs = None

results = executor.matmul_batch(pairs)  # 返回 [C1, C2, C3]
```

### 执行计划预览（不执行计算）

```python
plan = executor.plan_info("matmul", (50000, 10000), (10000, 50000))
# 输出:
# [AutoExecutor] ━━━ 执行计划 ━━━
#   操作: matmul
#   策略: 5 批次 × 4 GPU 并行 | 每批 23480 行 | 每GPU 3.175 GB
#   数据总量: 13.039 GB
#   GPU可用显存: 75.72 GB (单卡最小 3.17 GB)
```

### 便捷函数 auto_compute

```python
from distributed_gpu.auto_executor import auto_compute

C = auto_compute("matmul", A_cpu, B_cpu)
Y = auto_compute("fft", X_cpu)
S = auto_compute("sum", data_cpu)
```

### ResourcePlanner API

```python
from distributed_gpu.resource_planner import ResourcePlanner

planner = ResourcePlanner(mpi, max_per_gpu_gb=None)

# 扫描所有 GPU 可用显存
statuses = planner.scan_all_gpus()
for s in statuses:
    print(f"GPU {s.gpu_id}: {s.free_memory_gb:.1f}/{s.total_memory_gb:.1f} GB 空闲")

# 生成执行计划
plan = planner.plan_matmul(M=50000, K=10000, N=50000)
print(f"可行: {plan.feasible}, 批次: {plan.num_batches}")
```

---

## 常见问题（更新）

### Q6: AutoExecutor 和直接调用 distributed_matmul 有什么区别？

| 特性 | `distributed_matmul` | `AutoExecutor.matmul` |
|------|---------------------|----------------------|
| 输入 | GPU 张量 | CPU 张量 |
| 显存检查 | 无 | 自动扫描实时可用显存 |
| 超显存处理 | OOM 崩溃 | 自动分批 |
| 策略选择 | 需手动传 cost_model | 内置自动选择 |
| 适用场景 | 已知数据能放入 GPU | 数据量不确定 / 超大规模 |

### Q7: max_per_gpu_gb 有什么用？
用于测试分批逻辑或共享 GPU 资源的场景。设置后，每 GPU 可用显存不会超过该值。
例如 `AutoExecutor(max_per_gpu_gb=2.0)` 会让每张卡最多使用 2 GB。

### Q8: 安全边际 20% 可以调整吗？
可以通过修改 `ResourcePlanner.SAFETY_MARGIN` 属性。较低的边际意味着更充分利用显存但 OOM 风险更高。
