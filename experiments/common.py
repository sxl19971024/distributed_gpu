"""
实验公共工具模块

提供：
  - 路径与环境设置
  - Matplotlib 论文级图表风格
  - MPI / 框架初始化辅助
  - 计时工具（分布式 & 单机）
  - 结果保存（JSON + PNG）
  - GPU 信息查询
  - 第三方库可用性检测

所有实验脚本通过 `from common import *` 引入。
"""

import os, sys, json, time, datetime, subprocess
import numpy as np
import torch

# ===== 路径设置 =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== NVIDIA CUDA 库路径 (CuPy 需要) =====
_NV = os.path.join(
    sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages", "nvidia",
)
if os.path.isdir(_NV):
    _libs = [os.path.join(_NV, d, "lib") for d in os.listdir(_NV)
             if os.path.isdir(os.path.join(_NV, d, "lib"))]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# ===== 第三方库可用性检测 =====
try:
    import cupy as cp
    _t = cp.zeros(1); del _t
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

try:
    import scipy
    import scipy.signal
    import scipy.fft
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import dask
    import dask.array as da
    HAS_DASK = True
except Exception:
    HAS_DASK = False

# ===== Matplotlib =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107",
          "#9C27B0", "#00BCD4", "#E91E63", "#795548",
          "#607D8B", "#FF9800"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "figure.autolayout": True,
})

# ===== 框架初始化 =====

def setup_framework():
    """初始化本框架 (MPI + TensorDistributor + GPU)"""
    from distributed_gpu.mpi_manager import MPIManager
    from distributed_gpu.tensor_distributor import TensorDistributor
    from distributed_gpu.gpu_manager import GPUManager
    mpi = MPIManager()
    dist = TensorDistributor(mpi)
    gpu = GPUManager(mpi.get_gpu_id())
    return mpi, dist, gpu


def setup_cost_model(mpi):
    """初始化代价模型（用于策略自动选择）"""
    from distributed_gpu.cost_model import CostModel, ClusterConfig
    config = ClusterConfig.from_auto_detect(mpi.get_size())
    return CostModel(config)


def warmup(mpi, distributor, rounds=3):
    """预热 CUDA + MPI"""
    from distributed_gpu.algorithms.matrix_ops import distributed_matmul
    for _ in range(rounds):
        if mpi.is_master_process():
            t = torch.randn(64, 64, dtype=torch.float32).cuda()
        else:
            t = None
        distributed_matmul(t, t, mpi, distributor)
    mpi.synchronize()
    torch.cuda.empty_cache()


def init_torch_distributed(rank, world_size, gpu_id):
    """在 MPI 环境中初始化 PyTorch Distributed (NCCL)"""
    import torch.distributed as tdist
    if tdist.is_initialized():
        return True
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    try:
        tdist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(gpu_id)
        return True
    except Exception as e:
        if rank == 0:
            print(f"  [WARN] PyTorch Distributed init failed: {e}")
        return False


def cleanup_torch_distributed():
    import torch.distributed as tdist
    if tdist.is_initialized():
        tdist.destroy_process_group()


# ===== 计时工具 =====

def timed(func, repeats=5, warmup_runs=2, sync_cuda=True):
    """单机计时，返回 (stats_dict, last_result)"""
    for _ in range(warmup_runs):
        func()
    times = []
    result = None
    for _ in range(repeats):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = func()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {"mean": np.mean(times), "std": np.std(times),
            "min": np.min(times), "max": np.max(times)}, result


def timed_mpi(func, mpi, repeats=5, warmup_runs=2):
    """分布式计时，所有 rank 同步"""
    for _ in range(warmup_runs):
        func()
        mpi.synchronize()
    times = []
    result = None
    for _ in range(repeats):
        mpi.synchronize()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = func()
        mpi.synchronize()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {"mean": np.mean(times), "std": np.std(times),
            "min": np.min(times), "max": np.max(times)}, result


# ===== 结果 I/O =====

def _exp_dir(name):
    d = os.path.join(RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _update_latest(exp_dir, src_filename, ext):
    link = os.path.join(exp_dir, f"latest{ext}")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(src_filename, link)
    except OSError:
        pass

def save_json(name, data):
    ts = _timestamp()
    data["_meta"] = {"experiment": name,
                     "timestamp": datetime.datetime.now().isoformat(),
                     "run_id": ts}
    exp_dir = _exp_dir(name)
    filename = f"{ts}.json"
    p = os.path.join(exp_dir, filename)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    _update_latest(exp_dir, filename, ".json")
    print(f"  [SAVED] {p}")

def load_json(name, run_id=None):
    exp_dir = _exp_dir(name)
    if run_id:
        p = os.path.join(exp_dir, f"{run_id}.json")
    else:
        p = os.path.join(exp_dir, "latest.json")
        if not os.path.exists(p):
            jsons = sorted([f for f in os.listdir(exp_dir)
                            if f.endswith(".json") and f != "latest.json"])
            if not jsons:
                raise FileNotFoundError(f"No results found in {exp_dir}")
            p = os.path.join(exp_dir, jsons[-1])
    with open(p) as f:
        return json.load(f)

def save_fig(fig, name):
    ts = _timestamp()
    exp_dir = _exp_dir(name)
    filename = f"{ts}.png"
    p = os.path.join(exp_dir, filename)
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    _update_latest(exp_dir, filename, ".png")
    print(f"  [SAVED] {p}")


# ===== GPU 信息 =====

def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def gpu_mem_total_mb(device=None):
    if device is None:
        return torch.cuda.get_device_properties(0).total_mem / 1e6
    return torch.cuda.get_device_properties(device).total_mem / 1e6


# ===== 打印辅助 =====

def banner(title, rank=0):
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        sys.stdout.flush()

def log(msg, rank=0):
    if rank == 0:
        print(f"  {msg}")
        sys.stdout.flush()
