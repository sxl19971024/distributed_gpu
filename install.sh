#!/usr/bin/env bash
# ============================================================
#  distributed_gpu 一键安装脚本
#
#  用法:
#    bash install.sh                # 创建 conda 环境 + 安装全部依赖
#    bash install.sh --full         # 同上 + 跨框架对比依赖 (PETSc/Dask-CUDA)
#    bash install.sh --check        # 仅检测环境，不安装
#    bash install.sh --no-conda     # 跳过 conda 环境创建，直接安装到当前环境
#    bash install.sh --env NAME     # 指定 conda 环境名 (默认: distributed_gpu)
#
#  一键安装示例:
#    git clone https://github.com/sxl19971024/distributed_gpu.git
#    cd distributed_gpu
#    bash install.sh --full
#    conda activate distributed_gpu
#    mpirun -n 4 python examples/run_algorithm.py all   # 验证
#
# ============================================================

set -euo pipefail

# ── 颜色 ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ── 参数解析 ──
ENV_NAME="distributed_gpu"
INSTALL_MODE="standard"    # standard | full | check
USE_CONDA=true
PYTHON_VER="3.10"

for arg in "$@"; do
    case "$arg" in
        --full)     INSTALL_MODE="full" ;;
        --check)    INSTALL_MODE="check" ;;
        --no-conda) USE_CONDA=false ;;
        --env)      shift; ENV_NAME="${1:-distributed_gpu}" ;;
        --help|-h)
            cat <<'EOF'
用法: bash install.sh [选项]

选项:
  (无参数)     创建 conda 环境并安装核心 + 实验依赖
  --full       同上，额外安装跨框架对比依赖 (PETSc, Dask-CUDA, CuPy)
  --check      仅检测环境，不安装
  --no-conda   不创建 conda 环境，安装到当前 Python 环境
  --env NAME   指定 conda 环境名称 (默认: distributed_gpu)

安装内容:
  核心:    PyTorch(CUDA), mpi4py, OpenMPI, numpy, opt-einsum
  实验:    matplotlib, seaborn, scipy, cupy
  对比:    petsc4py, dask-cuda (仅 --full 模式)
EOF
            exit 0 ;;
        *) fail "未知参数: $arg (使用 --help 查看帮助)"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     distributed_gpu 一键安装脚本                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  环境名:   $ENV_NAME"
echo "║  安装模式: $INSTALL_MODE"
echo "║  项目目录: $SCRIPT_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════
#  Step 1: 前置检查
# ═══════════════════════════════════════════════

ERRORS=0

# NVIDIA 驱动
info "[1/6] 检测 NVIDIA 驱动..."
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    ok "NVIDIA 驱动 $DRIVER_VER | $GPU_COUNT 块 GPU"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while read -r line; do
        echo "       GPU $line"
    done
else
    fail "nvidia-smi 不可用，请先安装 NVIDIA 驱动"
    ERRORS=$((ERRORS + 1))
fi

# Conda
if [ "$USE_CONDA" = true ]; then
    info "[2/6] 检测 Conda..."
    if command -v conda &>/dev/null; then
        CONDA_VER=$(conda --version 2>/dev/null | awk '{print $2}')
        ok "Conda $CONDA_VER ($(which conda))"
    else
        fail "conda 未找到! 请先安装 Miniconda:"
        echo "       wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "       bash Miniconda3-latest-Linux-x86_64.sh"
        ERRORS=$((ERRORS + 1))
    fi
fi

if [ "$INSTALL_MODE" = "check" ]; then
    echo ""
    if [ "$ERRORS" -eq 0 ]; then
        ok "环境检测通过 ✅"
    else
        fail "发现 $ERRORS 个问题，请先修复"
    fi
    exit "$ERRORS"
fi

[ "$ERRORS" -gt 0 ] && { fail "环境检测失败，请先修复上述问题"; exit 1; }

# ═══════════════════════════════════════════════
#  Step 2: 创建 Conda 环境
# ═══════════════════════════════════════════════

if [ "$USE_CONDA" = true ]; then
    info "[3/6] 创建 Conda 环境: $ENV_NAME (Python $PYTHON_VER)..."

    if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
        warn "环境 '$ENV_NAME' 已存在，将直接使用"
    else
        conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
        ok "Conda 环境 '$ENV_NAME' 创建完成"
    fi

    # 激活环境 (在脚本中需要 source)
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    ok "已激活环境: $ENV_NAME (Python $(python --version 2>&1 | awk '{print $2}'))"
fi

# ═══════════════════════════════════════════════
#  Step 3: 安装 OpenMPI + mpi4py (通过 conda)
# ═══════════════════════════════════════════════

info "[4/6] 安装 OpenMPI + mpi4py..."

# OpenMPI 通过 conda 安装，确保 mpi4py 与之兼容
if python -c "from mpi4py import MPI" 2>/dev/null; then
    MPI4PY_VER=$(python -c "import mpi4py; print(mpi4py.__version__)" 2>/dev/null)
    ok "mpi4py $MPI4PY_VER 已安装"
else
    conda install -c conda-forge openmpi mpi4py -y
    ok "OpenMPI + mpi4py 安装完成"
fi

# ═══════════════════════════════════════════════
#  Step 4: 安装 PyTorch + 核心依赖
# ═══════════════════════════════════════════════

info "[5/6] 安装 PyTorch + 核心依赖..."

# PyTorch (CUDA)
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(f'{torch.__version__} (CUDA {torch.version.cuda})')")
    ok "PyTorch $TORCH_VER 已安装"
else
    info "安装 PyTorch (CUDA 12.x)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# 核心依赖 + 框架本身
pip install -e ".[full]"
ok "核心依赖 + 框架安装完成"

# ═══════════════════════════════════════════════
#  Step 5: 安装跨框架对比依赖 (--full 模式)
# ═══════════════════════════════════════════════

if [ "$INSTALL_MODE" = "full" ]; then
    info "[6/6] 安装跨框架对比依赖..."

    # PETSc (petsc4py) — 必须用 conda 安装
    if python -c "import petsc4py" 2>/dev/null; then
        P4PY_VER=$(python -c "import petsc4py; print(petsc4py.__version__)")
        ok "petsc4py $P4PY_VER 已安装"
    else
        info "安装 PETSc (petsc4py) — 科学计算对比框架..."
        conda install -c conda-forge petsc4py -y
        ok "petsc4py 安装完成"
    fi

    # Dask-CUDA + CuPy
    if python -c "import dask_cuda" 2>/dev/null; then
        DCUDA_VER=$(python -c "import dask_cuda; print(dask_cuda.__version__)")
        ok "dask-cuda $DCUDA_VER 已安装"
    else
        info "安装 Dask-CUDA — 分布式GPU对比框架..."
        pip install dask-cuda
        ok "dask-cuda 安装完成"
    fi

    if python -c "import cupy" 2>/dev/null; then
        CUPY_VER=$(python -c "import cupy; print(cupy.__version__)")
        ok "cupy $CUPY_VER 已安装"
    else
        info "安装 CuPy (GPU NumPy)..."
        pip install cupy-cuda12x
        ok "cupy 安装完成"
    fi
else
    info "[6/6] 跳过跨框架对比依赖 (使用 --full 安装)"
fi

# ═══════════════════════════════════════════════
#  Step 6: 验证安装
# ═══════════════════════════════════════════════

echo ""
info "验证安装..."
echo ""

python -c "
import sys

checks = []

# 核心
try:
    import distributed_gpu
    checks.append(('distributed_gpu', distributed_gpu.__version__, True))
except Exception as e:
    checks.append(('distributed_gpu', str(e), False))

try:
    import torch
    cuda = torch.cuda.is_available()
    ver = f'{torch.__version__} (CUDA {torch.version.cuda}, GPU: {cuda})'
    checks.append(('PyTorch', ver, cuda))
except Exception as e:
    checks.append(('PyTorch', str(e), False))

try:
    from mpi4py import MPI
    checks.append(('mpi4py', f'{MPI.Get_version()}', True))
except Exception as e:
    checks.append(('mpi4py', str(e), False))

try:
    import numpy; checks.append(('numpy', numpy.__version__, True))
except Exception as e:
    checks.append(('numpy', str(e), False))

try:
    import opt_einsum; checks.append(('opt-einsum', opt_einsum.__version__, True))
except Exception as e:
    checks.append(('opt-einsum', str(e), False))

# 可选
for name, imp in [
    ('matplotlib', 'matplotlib'), ('cupy', 'cupy'),
    ('petsc4py', 'petsc4py'), ('dask-cuda', 'dask_cuda'),
]:
    try:
        mod = __import__(imp)
        checks.append((name, getattr(mod, '__version__', 'ok'), True))
    except:
        checks.append((name, '未安装 (可选)', None))

# 输出
print('  ┌─────────────────┬──────────────────────────────────────┬────────┐')
print('  │ 包名            │ 版本                                 │ 状态   │')
print('  ├─────────────────┼──────────────────────────────────────┼────────┤')
for name, ver, ok in checks:
    status = '✅' if ok else ('⚠️  可选' if ok is None else '❌')
    print(f'  │ {name:<15} │ {ver:<36} │ {status:<6} │')
print('  └─────────────────┴──────────────────────────────────────┴────────┘')

core_ok = all(ok for _, _, ok in checks if ok is not None and ok is not False)
if not core_ok:
    print('\n  ❌ 核心依赖安装不完整!')
    sys.exit(1)
"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ 安装完成!                                           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                        ║"
echo "║  激活环境:                                              ║"
echo "║    conda activate $ENV_NAME"
echo "║                                                        ║"
echo "║  快速验证 (24个算子测试):                               ║"
echo "║    mpirun -n 4 python examples/run_algorithm.py all    ║"
echo "║                                                        ║"
echo "║  运行实验:                                              ║"
echo "║    python run_experiments.py --gpus 4 --exp all         ║"
echo "║                                                        ║"
if [ "$INSTALL_MODE" = "full" ]; then
echo "║  跨框架对比:                                            ║"
echo "║    python experiments/benchmark_comparison.py --exp all ║"
echo "║                                                        ║"
fi
echo "║  更多用法见 README.md                                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
