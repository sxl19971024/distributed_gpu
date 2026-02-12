# 安装指南

## 目标运行环境

| 组件 | 版本 | 说明 |
|------|------|------|
| **操作系统** | Linux (Ubuntu 18.04+ / CentOS 7+) | 需 NVIDIA GPU 驱动 |
| **Python** | 3.8+ | 推荐 3.10 |
| **CUDA** | **12.1** | NVIDIA Driver >= 530.30.02 |
| **MPI** | **OpenMPI 4.1.5** | 其他 4.1.x 版本也兼容 |
| **GPU** | NVIDIA (Compute Capability ≥ 7.0) | 推荐显存 ≥ 16GB |

## 一键安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/sxl19971024/distributed_gpu.git
cd distributed_gpu

# 2. 创建并激活虚拟环境
conda create -n distributed_gpu python=3.10 -y
conda activate distributed_gpu

# 3. 安装 OpenMPI 4.1.5 + mpi4py
conda install -c conda-forge openmpi=4.1.5 mpi4py -y

# 4. 安装 PyTorch (CUDA 12.1 版本)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. 安装框架（开发模式，修改代码实时生效）
pip install -e .

# 6. 验证安装
python check_env.py
```

## 环境验证

```bash
# 单进程检测（检查所有依赖和 GPU）
python check_env.py

# 多 GPU 协同测试
mpirun -n 4 python examples/run_algorithm.py all

# 预期输出: 总计: 17/17 通过
```

## 手动安装

如果不使用 conda，需要先安装系统级 OpenMPI 4.1.5：

```bash
# Ubuntu / Debian (从源码编译 OpenMPI 4.1.5)
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar xzf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5
./configure --prefix=/usr/local --with-cuda=/usr/local/cuda-12.1
make -j$(nproc)
sudo make install
sudo ldconfig

# 或者如果系统已有 OpenMPI，直接安装 mpi4py
pip install mpi4py
```

然后安装 Python 依赖：

```bash
# 安装 PyTorch CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖 + 框架
pip install -r requirements.txt
pip install -e .
```

## HPC 集群安装（module 环境）

很多高性能集群使用 `module` 管理软件，典型流程：

```bash
# 加载已有的 CUDA 和 MPI 模块
module load cuda/12.1.0
module load openmpi/4.1.5
# 或: module load mpi/openmpi-4.1.5

# 确认版本
nvcc --version    # 应显示 CUDA 12.1
mpirun --version  # 应显示 Open MPI 4.1.5

# 创建 Python 环境
conda create -n distributed_gpu python=3.10 -y
conda activate distributed_gpu

# 安装 PyTorch CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 在 login 节点安装（通常有网络）
pip install mpi4py numpy opt-einsum
pip install -e .

# 验证
python check_env.py
```

> **注意**: GPU 计算节点可能没有外网访问，请在 login 节点安装好依赖后再提交作业。

## 安装可选依赖

```bash
# 方式一：通过 extras
pip install -e ".[full]"

# 方式二：手动安装
pip install -r requirements-optional.txt

# matplotlib（生成实验图表必需）
pip install matplotlib seaborn
```

## 运行实验

```bash
# 自动检测 GPU 数量，运行全部实验
python run_experiments.py

# 指定 GPU 数量
python run_experiments.py --gpus 4

# 运行单个实验
python run_experiments.py --gpus 4 --exp 1

# 查看可用实验列表
python run_experiments.py --list
```

## 常见问题

### 1. `MPI_ERR_ARG: invalid argument` (scatter 失败)

v1.4.0 已修复此问题。如果仍出现，请确认：
```bash
# 确认 OpenMPI 版本
mpirun --version
# 确认 mpi4py 与系统 MPI 匹配
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

### 2. `mpirun: command not found`

```bash
# conda 安装
conda install -c conda-forge openmpi=4.1.5 -y
# 或加载 module
module load openmpi/4.1.5
```

### 3. `mpi4py` 编译失败

```bash
# 先安装 MPI 开发头文件
sudo apt install libopenmpi-dev    # Ubuntu
sudo yum install openmpi-devel     # CentOS
pip install mpi4py
```

### 4. CUDA 显存不足 (OOM)

- 框架内置显存感知调度，会自动分批处理
- 也可减小矩阵规模或增加 GPU 数量
- `python run_experiments.py --gpus 8` 使用更多 GPU 分摊负载

### 5. `torch.cuda.is_available()` 返回 `False`

```bash
# 检查 NVIDIA 驱动版本（CUDA 12.1 需要 >= 530）
nvidia-smi

# 确认 PyTorch 的 CUDA 版本
python -c "import torch; print(f'CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"

# 重新安装 CUDA 12.1 版本 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 6. 多节点运行

```bash
# 创建 hostfile
echo "node1 slots=4" > hostfile
echo "node2 slots=4" >> hostfile

# 运行（8 GPU，跨 2 节点）
mpirun -n 8 --hostfile hostfile python run_experiments.py --gpus 8 --exp all
```

### 7. OpenMPI 警告 "No InfiniBand"

```bash
# 在无 InfiniBand 网络的环境中，加此环境变量抑制警告
export OMPI_MCA_btl=^openib
# 或在 mpirun 中指定
mpirun --mca btl ^openib -n 4 python run_experiments.py
```

## 环境变量（可选）

```bash
# CUDA 路径
export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

# 抑制 InfiniBand 警告
export OMPI_MCA_btl=^openib

# 指定可见 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
```
