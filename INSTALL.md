# 安装指南

## 系统要求

- **操作系统**: Linux (Ubuntu 18.04+, CentOS 7+)
- **Python**: 3.8+
- **CUDA**: 11.0+
- **GPU**: NVIDIA GPU (支持 CUDA)
- **MPI**: OpenMPI 4.0+ 或 MPICH

## 快速安装

### 方式一：使用 pip 安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/dstar/distributed-gpu-framework.git
cd distributed-gpu-framework

# 2. 创建虚拟环境（推荐）
conda create -n distributed_gpu python=3.10
conda activate distributed_gpu

# 3. 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 MPI
conda install -c conda-forge openmpi mpi4py

# 5. 安装框架
pip install -e .

# 6. 安装完整依赖（可选）
pip install -e ".[full]"
```

### 方式二：手动安装依赖

```bash
# 安装核心依赖
pip install torch>=2.0.0
pip install mpi4py>=3.1.0
pip install numpy>=1.21.0
pip install opt-einsum>=3.3.0

# 安装可选依赖
pip install cupy-cuda11x>=12.0.0
pip install scipy>=1.9.0
pip install psutil>=5.9.0
pip install pyyaml>=6.0
```

## 验证安装

```bash
# 运行综合测试（使用4个GPU）
mpirun -n 4 --allow-run-as-root python examples/test_all.py
```

预期输出：
```
总计: 7/7 通过
```

## 常见问题

### 1. MPI 相关错误

**问题**: `mpirun: command not found`

**解决**:
```bash
# Ubuntu/Debian
sudo apt install openmpi-bin libopenmpi-dev

# 或使用 conda
conda install -c conda-forge openmpi
```

### 2. CUDA 相关错误

**问题**: `CUDA out of memory`

**解决**: 减少数据规模或使用更多 GPU 节点

### 3. mpi4py 安装失败

**问题**: `mpi4py` 编译错误

**解决**:
```bash
# 先安装 MPI 库
sudo apt install libopenmpi-dev

# 然后安装 mpi4py
pip install mpi4py
```

### 4. 多节点运行

```bash
# 创建 hostfile
echo "node1 slots=4" > hostfile
echo "node2 slots=4" >> hostfile

# 运行
mpirun -n 8 --hostfile hostfile python your_script.py
```

## 环境变量配置

```bash
# 添加到 ~/.bashrc
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# MPI 配置（可选）
export OMPI_MCA_btl=^openib  # 禁用 InfiniBand（如果没有）
```

## Docker 安装（可选）

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y openmpi-bin libopenmpi-dev
RUN pip install mpi4py opt-einsum

COPY . /app
WORKDIR /app
RUN pip install -e .
```

## 联系支持

如遇问题，请提交 Issue 或联系作者。
