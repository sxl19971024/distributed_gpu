# API 使用指南

本文档详细说明框架中每个分布式算法的使用方法、参数和示例。

## 📋 目录

1. [初始化框架](#初始化框架)
2. [矩阵运算](#矩阵运算)
3. [卷积操作](#卷积操作)
4. [傅里叶变换](#傅里叶变换)
5. [Einstein求和](#einstein求和)
6. [归约操作](#归约操作)
7. [高级功能](#高级功能)

---

## 初始化框架

在使用任何分布式算法之前，必须先初始化 MPI 环境和张量分配器。

```python
import torch
from src.mpi_manager import MPIManager
from src.tensor_distributor import TensorDistributor

# 初始化（所有进程都必须执行）
mpi = MPIManager()
distributor = TensorDistributor(mpi)

# 获取当前进程信息
print(f"进程 {mpi.get_rank()}/{mpi.get_size()}, GPU {mpi.get_gpu_id()}")
```

**重要原则**：
- ⚠️ 所有进程都必须调用相同的分布式函数
- ⚠️ 输入数据仅在主进程（rank=0）提供，其他进程传 `None`
- ⚠️ 输出结果仅主进程返回有效值，其他进程返回 `None`

---

## 矩阵运算

### 1. distributed_matmul - 矩阵乘法

**功能**：计算 C = A @ B

```python
from src.algorithms.matrix_ops import distributed_matmul

# 参数说明
C = distributed_matmul(
    A,              # torch.Tensor: 矩阵A [M, K]，仅主进程提供
    B,              # torch.Tensor: 矩阵B [K, N]，仅主进程提供
    mpi,            # MPIManager: MPI管理器
    distributor     # TensorDistributor: 张量分配器
)
# 返回: torch.Tensor [M, N]，仅主进程返回有效结果
```

**完整示例**：
```python
import torch
from src.mpi_manager import MPIManager
from src.tensor_distributor import TensorDistributor
from src.algorithms.matrix_ops import distributed_matmul

mpi = MPIManager()
distributor = TensorDistributor(mpi)

# 主进程创建数据
if mpi.is_master_process():
    A = torch.randn(5000, 3000).cuda()  # [M, K]
    B = torch.randn(3000, 4000).cuda()  # [K, N]
else:
    A, B = None, None

# 所有进程调用（重要！）
C = distributed_matmul(A, B, mpi, distributor)

# 主进程处理结果
if mpi.is_master_process():
    print(f"结果形状: {C.shape}")  # [5000, 4000]
```

**运行命令**：
```bash
mpirun -n 4 python your_script.py
```

---

### 2. distributed_batch_matmul - 批量矩阵乘法

**功能**：计算批量矩阵乘法 C = A @ B

```python
from src.algorithms.matrix_ops import distributed_batch_matmul

C = distributed_batch_matmul(
    A,              # torch.Tensor: [batch, M, K]
    B,              # torch.Tensor: [batch, K, N] 或 [K, N]
    mpi,            # MPIManager
    distributor     # TensorDistributor
)
# 返回: torch.Tensor [batch, M, N]
```

**示例**：
```python
if mpi.is_master_process():
    A = torch.randn(64, 256, 128).cuda()  # [batch, M, K]
    B = torch.randn(64, 128, 256).cuda()  # [batch, K, N]
else:
    A, B = None, None

C = distributed_batch_matmul(A, B, mpi, distributor)
# C.shape = [64, 256, 256]
```

---

### 3. distributed_transpose - 矩阵转置

```python
from src.algorithms.matrix_ops import distributed_transpose

A_T = distributed_transpose(
    A,              # torch.Tensor: 输入矩阵
    mpi,            # MPIManager
    distributor,    # TensorDistributor
    dim0=0,         # int: 第一个交换维度（默认0）
    dim1=1          # int: 第二个交换维度（默认1）
)
```

---

### 4. distributed_add - 张量加法

```python
from src.algorithms.matrix_ops import distributed_add

C = distributed_add(
    A,              # torch.Tensor: 张量A
    B,              # torch.Tensor: 张量B（形状与A相同）
    mpi,            # MPIManager
    distributor     # TensorDistributor
)
# 返回: C = A + B
```

---

## 卷积操作

### 1. distributed_conv2d - 2D卷积

**功能**：分布式2D卷积，按batch维度分割

```python
from src.algorithms.convolution import distributed_conv2d

output = distributed_conv2d(
    input,          # torch.Tensor: [N, C_in, H, W] 输入
    weight,         # torch.Tensor: [C_out, C_in, kH, kW] 卷积核
    mpi,            # MPIManager
    distributor,    # TensorDistributor
    bias=None,      # torch.Tensor: [C_out] 偏置（可选）
    stride=(1, 1),  # tuple: 步长
    padding=(0, 0), # tuple: 填充
    dilation=(1, 1),# tuple: 膨胀率
    groups=1        # int: 分组数
)
# 返回: torch.Tensor [N, C_out, H_out, W_out]
```

**完整示例**：
```python
from src.algorithms.convolution import distributed_conv2d

if mpi.is_master_process():
    # 输入: batch=32, channels=64, height=128, width=128
    input = torch.randn(32, 64, 128, 128).cuda()
    # 卷积核: out_channels=128, in_channels=64, kernel=3x3
    weight = torch.randn(128, 64, 3, 3).cuda()
    bias = torch.randn(128).cuda()
else:
    input, weight, bias = None, None, None

output = distributed_conv2d(
    input, weight, mpi, distributor,
    bias=bias,
    stride=(1, 1),
    padding=(1, 1)  # same padding
)

if mpi.is_master_process():
    print(f"输出形状: {output.shape}")  # [32, 128, 128, 128]
```

---

### 2. distributed_conv3d - 3D卷积

```python
from src.algorithms.convolution import distributed_conv3d

output = distributed_conv3d(
    input,          # torch.Tensor: [N, C_in, D, H, W]
    weight,         # torch.Tensor: [C_out, C_in, kD, kH, kW]
    mpi,
    distributor,
    bias=None,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1
)
```

---

## 傅里叶变换

### 1. distributed_fft - 1D FFT

```python
from src.algorithms.fft import distributed_fft

output = distributed_fft(
    input,          # torch.Tensor: 输入张量
    mpi,            # MPIManager
    distributor,    # TensorDistributor
    n=None,         # int: FFT长度（None使用输入长度）
    dim=-1,         # int: FFT维度（默认最后一维）
    norm="backward" # str: 归一化模式 ("backward", "ortho", "forward")
)
```

**示例**：
```python
from src.algorithms.fft import distributed_fft

if mpi.is_master_process():
    # 64个信号，每个1024个采样点
    signal = torch.randn(64, 1024).cuda()
else:
    signal = None

spectrum = distributed_fft(signal, mpi, distributor)

if mpi.is_master_process():
    print(f"频谱形状: {spectrum.shape}")  # [64, 1024]
```

---

### 2. distributed_ifft - 逆FFT

```python
from src.algorithms.fft import distributed_ifft

output = distributed_ifft(
    input,          # torch.Tensor: 频域信号
    mpi,
    distributor,
    n=None,
    dim=-1,
    norm="backward"
)
```

---

### 3. distributed_fft2d - 2D FFT

```python
from src.algorithms.fft import distributed_fft2d

output = distributed_fft2d(
    input,          # torch.Tensor: [..., H, W]
    mpi,
    distributor,
    s=None,         # tuple: FFT大小 (H, W)
    dim=(-2, -1),   # tuple: FFT维度
    norm="backward"
)
```

**示例**（图像频谱分析）：
```python
from src.algorithms.fft import distributed_fft2d

if mpi.is_master_process():
    # 32张 256x256 的图像
    images = torch.randn(32, 256, 256).cuda()
else:
    images = None

spectrum = distributed_fft2d(images, mpi, distributor)

if mpi.is_master_process():
    print(f"频谱形状: {spectrum.shape}")  # [32, 256, 256]
```

---

### 4. distributed_rfft - 实数FFT

```python
from src.algorithms.fft import distributed_rfft

output = distributed_rfft(
    input,          # torch.Tensor: 实数输入
    mpi,
    distributor,
    n=None,
    dim=-1,
    norm="backward"
)
# 返回: 只有正频率部分，更高效
```

---

## Einstein求和

### 1. distributed_einsum - Einstein求和（集成opt_einsum）

**功能**：通用张量收缩，支持最优路径优化

```python
from src.algorithms.einsum import distributed_einsum

result = distributed_einsum(
    equation,       # str: Einstein求和表达式
    *operands,      # torch.Tensor: 操作数（可变参数）
    mpi=mpi,        # MPIManager
    distributor=distributor,  # TensorDistributor
    optimize='auto',# str: 优化策略 ('optimal', 'dp', 'greedy', 'auto')
    use_opt_einsum=True  # bool: 是否使用opt_einsum优化
)
```

**常用表达式**：

| 表达式 | 操作 | 示例形状 |
|--------|------|----------|
| `'ij,jk->ik'` | 矩阵乘法 | [M,K] @ [K,N] |
| `'bij,bjk->bik'` | 批量矩阵乘法 | [B,M,K] @ [B,K,N] |
| `'ii->'` | 矩阵的迹 | [N,N] → 标量 |
| `'ij->ji'` | 转置 | [M,N] → [N,M] |
| `'i,j->ij'` | 外积 | [M], [N] → [M,N] |
| `'ijk,ikl->ijl'` | 张量收缩 | 自定义 |

**示例**：
```python
from src.algorithms.einsum import distributed_einsum

if mpi.is_master_process():
    A = torch.randn(32, 128, 64).cuda()
    B = torch.randn(32, 64, 256).cuda()
else:
    A, B = None, None

# 批量矩阵乘法
C = distributed_einsum('bij,bjk->bik', A, B, mpi=mpi, distributor=distributor)

if mpi.is_master_process():
    print(f"结果形状: {C.shape}")  # [32, 128, 256]
```

---

### 2. 查看最优收缩路径

```python
from src.algorithms.einsum import print_path_info, compare_optimization_strategies

# 打印最优路径信息
print_path_info('ij,jk,kl,lm->im', (100,50), (50,80), (80,60), (60,40))

# 比较不同优化策略
compare_optimization_strategies('ij,jk,kl->il', (100,50), (50,80), (80,40))
```

---

### 3. distributed_tensordot - 张量点积

```python
from src.algorithms.einsum import distributed_tensordot

result = distributed_tensordot(
    a,              # torch.Tensor: 第一个张量
    b,              # torch.Tensor: 第二个张量
    dims,           # int: 收缩的维度数
    mpi,
    distributor
)
```

---

## 归约操作

### 1. distributed_sum - 求和

```python
from src.algorithms.reduction import distributed_sum

result = distributed_sum(
    tensor,         # torch.Tensor: 输入张量
    mpi,            # MPIManager
    distributor,    # TensorDistributor
    dim=None,       # int: 求和维度（None表示全部求和）
    keepdim=False   # bool: 是否保持维度
)
```

**示例**：
```python
from src.algorithms.reduction import distributed_sum

if mpi.is_master_process():
    x = torch.randn(1000, 1000).cuda()
else:
    x = None

# 全局求和
total = distributed_sum(x, mpi, distributor, dim=None)

# 按维度求和
row_sum = distributed_sum(x, mpi, distributor, dim=1)  # [1000]
col_sum = distributed_sum(x, mpi, distributor, dim=0)  # [1000]
```

---

### 2. distributed_mean - 均值

```python
from src.algorithms.reduction import distributed_mean

result = distributed_mean(
    tensor,
    mpi,
    distributor,
    dim=None,       # int: 求均值维度
    keepdim=False
)
```

---

### 3. distributed_max - 最大值

```python
from src.algorithms.reduction import distributed_max

result = distributed_max(
    tensor,
    mpi,
    distributor,
    dim=None,       # int: 求最大值维度
    keepdim=False
)
```

---

### 4. distributed_min - 最小值

```python
from src.algorithms.reduction import distributed_min

result = distributed_min(
    tensor,
    mpi,
    distributor,
    dim=None,
    keepdim=False
)
```

---

## 高级功能

### 1. 代价模型分析

```python
from src.cost_model import CostModel, ClusterConfig

# 自动检测集群配置
config = ClusterConfig.from_auto_detect(num_nodes=4)

# 创建代价模型
cost_model = CostModel(config)

# 分析矩阵乘法代价
cost_model.print_analysis(M=5000, K=5000, N=5000)

# 获取最优分割策略
plan = cost_model.find_optimal_strategy(M=5000, K=5000, N=5000)
print(f"推荐策略: {plan.strategy}")
print(f"预估加速比: {plan.cost.speedup:.2f}x")
```

---

### 2. 性能分析

```python
from src.utils.profiler import Profiler

profiler = Profiler(enabled=mpi.is_master_process())

profiler.start("matmul")
C = distributed_matmul(A, B, mpi, distributor)
profiler.end("matmul")

profiler.print_summary()
```

---

### 3. GPU 管理

```python
from src.gpu_manager import GPUManager

gpu = GPUManager(mpi.get_gpu_id())

# 打印 GPU 信息
gpu.print_info()

# 查看显存使用
gpu.print_memory_info()

# 检查是否能放下张量
can_fit = gpu.can_fit((10000, 10000), (10000, 10000), dtype=torch.float32)
```

---

## 完整示例脚本

```python
#!/usr/bin/env python
"""
完整使用示例
运行: mpirun -n 4 python example.py
"""
import torch
from src.mpi_manager import MPIManager
from src.tensor_distributor import TensorDistributor
from src.gpu_manager import GPUManager
from src.algorithms.matrix_ops import distributed_matmul
from src.algorithms.convolution import distributed_conv2d
from src.algorithms.fft import distributed_fft2d
from src.algorithms.einsum import distributed_einsum
from src.algorithms.reduction import distributed_mean

def main():
    # 初始化
    mpi = MPIManager()
    distributor = TensorDistributor(mpi)
    gpu = GPUManager(mpi.get_gpu_id())
    
    if mpi.is_master_process():
        print("=" * 50)
        print("分布式GPU计算框架使用示例")
        print("=" * 50)
        gpu.print_info()
    
    # ========== 1. 矩阵乘法 ==========
    if mpi.is_master_process():
        print("\n[1] 矩阵乘法")
        A = torch.randn(2000, 1500).cuda()
        B = torch.randn(1500, 1000).cuda()
    else:
        A, B = None, None
    
    C = distributed_matmul(A, B, mpi, distributor)
    
    if mpi.is_master_process():
        print(f"    {A.shape} @ {B.shape} = {C.shape}")
    
    # ========== 2. 卷积 ==========
    if mpi.is_master_process():
        print("\n[2] 2D卷积")
        x = torch.randn(16, 32, 64, 64).cuda()
        w = torch.randn(64, 32, 3, 3).cuda()
    else:
        x, w = None, None
    
    y = distributed_conv2d(x, w, mpi, distributor, padding=(1, 1))
    
    if mpi.is_master_process():
        print(f"    输入: {x.shape}, 卷积核: {w.shape}, 输出: {y.shape}")
    
    # ========== 3. FFT ==========
    if mpi.is_master_process():
        print("\n[3] 2D FFT")
        img = torch.randn(8, 128, 128).cuda()
    else:
        img = None
    
    spec = distributed_fft2d(img, mpi, distributor)
    
    if mpi.is_master_process():
        print(f"    输入: {img.shape}, 频谱: {spec.shape}")
    
    # ========== 4. Einsum ==========
    if mpi.is_master_process():
        print("\n[4] Einstein求和")
        P = torch.randn(16, 64, 32).cuda()
        Q = torch.randn(16, 32, 64).cuda()
    else:
        P, Q = None, None
    
    R = distributed_einsum('bij,bjk->bik', P, Q, mpi=mpi, distributor=distributor)
    
    if mpi.is_master_process():
        print(f"    'bij,bjk->bik': {P.shape}, {Q.shape} -> {R.shape}")
    
    # ========== 5. 归约 ==========
    if mpi.is_master_process():
        print("\n[5] 归约操作")
        data = torch.randn(100, 100, 100).cuda()
    else:
        data = None
    
    avg = distributed_mean(data, mpi, distributor)
    
    if mpi.is_master_process():
        print(f"    均值: {avg.item():.6f}")
    
    # 完成
    if mpi.is_master_process():
        print("\n" + "=" * 50)
        print("所有操作完成！")
        gpu.print_memory_info()

if __name__ == "__main__":
    main()
```

---

## 常见问题

### Q1: 为什么其他进程要传 None？
因为数据只在主进程存在，分布式函数内部会自动将数据分发到其他进程。

### Q2: 为什么所有进程都要调用函数？
MPI 的集合通信要求所有进程都参与，否则会死锁。

### Q3: 如何处理返回的 None？
```python
result = distributed_matmul(A, B, mpi, distributor)
if mpi.is_master_process():
    # 只有主进程处理结果
    save(result)
```

### Q4: 如何调整 GPU 数量？
修改 `mpirun -n <数量>` 参数即可。

---

## 联系方式

- 作者: 孙小林
- Email: 1271364457@qq.com
- GitHub: https://github.com/sxl19971024/distributed_gpu
