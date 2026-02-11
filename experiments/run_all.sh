#!/bin/bash
#
# 一键运行全部 7 项实验
#
# 使用方式:
#   cd /nfs_vision_share/dstar/test/cursor_install/distributed_gpu_framework
#   bash experiments/run_all.sh [NUM_GPUS]
#
# 参数:
#   NUM_GPUS  默认 GPU 数 (默认 4)
#
# 实验内容:
#   1. 核心计算性能综合对比 (MatMul / FFT / Reduction × 7 框架)
#   2. 通信原语开销对比 (Broadcast / AllReduce × 4 方案)
#   3. 强扩展性 (固定问题规模, 增加 GPU 数)
#   4. 弱扩展性 (固定每 GPU 负载, 增加 GPU 数)
#   5. 领域科学计算应用 (Conv2D / Stencil / Jacobi)
#   6. 创新点消融实验 (代价模型 / Kahan / 稀疏 / Pencil / Pipeline)
#   7. 功能覆盖与开发效率对比 (定性)
#
# 结果输出: results/<实验名>/<时间戳>.{json,png}
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

DEFAULT_N=${1:-4}
SCALING_NS="1 2 4"

# ----- NVIDIA CUDA 库路径 (CuPy) -----
NV_BASE="$(python -c "import sys; print(sys.prefix)")/lib/python$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/site-packages/nvidia"
if [ -d "$NV_BASE" ]; then
    NV_LIBS=""
    for d in "$NV_BASE"/*/lib; do
        [ -d "$d" ] && NV_LIBS="$d:$NV_LIBS"
    done
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
fi

MPI="mpirun --oversubscribe --allow-run-as-root"

# ----- 环境准备 -----
find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
mkdir -p results

echo "============================================================"
echo "  分布式 GPU 科学计算框架 — 综合性能对比实验"
echo "  项目: $PROJECT_DIR"
echo "  GPU:  $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo N/A)"
echo "  默认 GPU 数: $DEFAULT_N"
echo "  扩展性测试: n = $SCALING_NS"
echo "  结果: $PROJECT_DIR/results/"
echo "============================================================"

echo ""
echo ">>> [1/7] 核心计算性能综合对比 (${DEFAULT_N} GPUs)"
$MPI -n $DEFAULT_N -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
    python experiments/exp01_compute_performance.py
echo "    ✓ 完成"

echo ""
echo ">>> [2/7] 通信原语开销对比 (${DEFAULT_N} GPUs)"
$MPI -n $DEFAULT_N -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
    python experiments/exp02_communication_overhead.py
echo "    ✓ 完成"

echo ""
echo ">>> [3/7] 强扩展性 (Strong Scaling)"
for n in $SCALING_NS; do
    echo "    n=$n ..."
    $MPI -n $n -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
        python experiments/exp03_strong_scaling.py
done
echo "    生成图表 ..."
python experiments/exp03_strong_scaling.py --plot
echo "    ✓ 完成"

echo ""
echo ">>> [4/7] 弱扩展性 (Weak Scaling)"
for n in $SCALING_NS; do
    echo "    n=$n ..."
    $MPI -n $n -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
        python experiments/exp04_weak_scaling.py
done
echo "    生成图表 ..."
python experiments/exp04_weak_scaling.py --plot
echo "    ✓ 完成"

echo ""
echo ">>> [5/7] 领域科学计算应用对比 (${DEFAULT_N} GPUs)"
$MPI -n $DEFAULT_N -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
    python experiments/exp05_domain_applications.py
echo "    ✓ 完成"

echo ""
echo ">>> [6/7] 创新点消融实验 (${DEFAULT_N} GPUs)"
$MPI -n $DEFAULT_N -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
    python experiments/exp06_innovation_ablation.py
echo "    ✓ 完成"

echo ""
echo ">>> [7/7] 功能覆盖与开发效率对比"
python experiments/exp07_feature_comparison.py
echo "    ✓ 完成"

echo ""
echo "============================================================"
echo "  全部 7 项实验完成！"
echo "============================================================"
echo "  JSON 结果:"
find results -name "*.json" -not -name "latest.json" 2>/dev/null | sort | sed 's/^/    /'
echo ""
echo "  PNG 图表:"
find results -name "*.png" -not -name "latest.png" 2>/dev/null | sort | sed 's/^/    /'
echo "============================================================"
