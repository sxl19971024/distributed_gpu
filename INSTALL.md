# 安装指南

## 系统要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| **操作系统** | Linux (Ubuntu 18.04+) | 支持 CentOS 7+、Debian 10+ |
| **Python** | 3.8+ | 推荐 3.10 |
| **CUDA** | 11.0+ | 需要 NVIDIA GPU 驱动 |
| **GPU** | NVIDIA (支持 CUDA) | 推荐显存 ≥ 16GB |
| **MPI** | OpenMPI 4.0+ 或 MPICH | 系统级安装 |

## 一键安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/sxl19971024/distributed_gpu.git
cd distributed_gpu

# 2. 创建并激活虚拟环境
conda create -n distributed_gpu python=3.10 -y
conda activate distributed_gpu

# 3. 安装 PyTorch（根据 CUDA 版本选择一条执行）
# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4+:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 MPI 运行时 + mpi4py
conda install -c conda-forge openmpi mpi4py -y

# 5. 安装框架（开发模式，修改代码实时生效）
pip install -e .
```

完成后即可在任何位置使用：
```python
from distributed_gpu import MPIManager, TensorDistributor, AutoExecutor
```

## 手动安装

如果不使用 conda，需要先安装系统级 MPI：

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev

# CentOS / RHEL
sudo yum install -y openmpi openmpi-devel
export PATH=$PATH:/usr/lib64/openmpi/bin
```

然后安装 Python 依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

## 安装可选依赖

```bash
# 方式一：通过 extras
pip install -e ".[full]"

# 方式二：手动安装
pip install -r requirements-optional.txt

# 方式三：单独安装 CuPy（需匹配 CUDA 版本）
# 查看 CUDA 版本: nvcc --version 或 python -c "import torch; print(torch.version.cuda)"
pip install cupy-cuda11x   # CUDA 11.x
pip install cupy-cuda12x   # CUDA 12.x
```

## 验证安装

```bash
# 1. 验证 Python 包导入
python -c "from distributed_gpu import MPIManager; print('导入成功!')"

# 2. 运行综合测试（使用 4 个 GPU）
mpirun -n 4 python examples/run_algorithm.py all

# 3. 预期输出：
#    总计: 17/17 通过
```

## 常见问题

### 1. `mpirun: command not found`

MPI 运行时未安装或未加入 PATH：

```bash
# Ubuntu
sudo apt install openmpi-bin libopenmpi-dev
# 或通过 conda
conda install -c conda-forge openmpi
```

### 2. `mpi4py` 编译失败

需要先安装 MPI 开发头文件：

```bash
sudo apt install libopenmpi-dev   # Ubuntu
sudo yum install openmpi-devel    # CentOS
pip install mpi4py
```

### 3. CUDA 显存不足 (OOM)

- 减小矩阵规模
- 使用更多 GPU 分摊负载：`mpirun -n 8 python your_script.py`
- 框架内置显存检测，会自动跳过超出显存的实验

### 4. `torch.cuda.is_available()` 返回 `False`

- 检查 NVIDIA 驱动：`nvidia-smi`
- 确认 PyTorch 的 CUDA 版本匹配：`python -c "import torch; print(torch.version.cuda)"`
- 重新安装带 CUDA 的 PyTorch 版本

### 5. 多节点运行

```bash
# 创建 hostfile
echo "node1 slots=4" > hostfile
echo "node2 slots=4" >> hostfile

# 运行
mpirun -n 8 --hostfile hostfile python your_script.py
```

### 6. OpenMPI 警告 "No InfiniBand"

```bash
export OMPI_MCA_btl=^openib
# 或在 mpirun 中指定
mpirun --mca btl ^openib -n 4 python your_script.py
```

## 环境变量（可选）

```bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export OMPI_MCA_btl=^openib  # 无 InfiniBand 时
```
