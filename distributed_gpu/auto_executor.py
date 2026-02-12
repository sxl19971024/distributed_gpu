"""
自动化分布式计算执行器 (Automatic Distributed Compute Executor)

用户接口层：CPU 张量 → 显存扫描 → 资源规划 → 自动分批 → 分布式计算 → CPU 结果

核心优化：
- 智能 GPU 数量选择：数据能放进一张卡时直接单卡计算，跳过全部 MPI 通信
- 超显存自动分批：数据超过单卡显存时自动分批处理
- 全部 26 个算子统一接口

使用方式（所有 MPI 进程都必须执行相同代码）：

    executor = AutoExecutor()

    if executor.is_master:
        A = torch.randn(100000, 10000)  # CPU 张量
        B = torch.randn(10000, 5000)
    else:
        A = B = None

    C = executor.matmul(A, B)

或使用便捷函数：

    from distributed_gpu.auto_executor import auto_compute
    C = auto_compute("matmul", A, B)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .mpi_manager import MPIManager
from .tensor_distributor import TensorDistributor
from .gpu_manager import GPUManager
from .cost_model import CostModel, ClusterConfig
from .resource_planner import ResourcePlanner, ExecutionPlan


# ---------------------------------------------------------------------------
#  类型别名
# ---------------------------------------------------------------------------
_TensorOrNone = Optional[torch.Tensor]
_ComputeFn = Callable[..., torch.Tensor]


class AutoExecutor:
    """
    显存感知的自动化分布式计算执行器

    ⚡ 智能路由：
      - 单卡直连：数据能放进一张卡时，跳过全部 MPI 通信，性能接近原生 PyTorch
      - 多卡并行：数据超过单卡时，自动分布到多 GPU 并行计算
      - 超显存分批：数据超过所有 GPU 总显存时，自动分批处理

    支持全部 26 个分布式算子，其中 22 个支持超显存自动分批。
    """

    def __init__(
        self,
        mpi: Optional[MPIManager] = None,
        verbose: bool = True,
        max_per_gpu_gb: Optional[float] = None,
    ) -> None:
        self.mpi: MPIManager = mpi or MPIManager()
        self.distributor: TensorDistributor = TensorDistributor(self.mpi)
        self.gpu: GPUManager = GPUManager(self.mpi.get_gpu_id())
        self.planner: ResourcePlanner = ResourcePlanner(
            self.mpi, max_per_gpu_gb=max_per_gpu_gb
        )
        config: ClusterConfig = ClusterConfig.from_auto_detect(self.mpi.get_size())
        self.cost_model: CostModel = CostModel(config)
        self.verbose: bool = verbose
        self.is_master: bool = self.mpi.is_master_process()
        self._pipeline: Optional[Any] = None  # lazy PipelineOptimizer

    # ==================== 日志 ====================

    def _log(self, msg: str) -> None:
        if self.verbose and self.is_master:
            print(f"[AutoExecutor] {msg}")

    def _log_plan(self, plan: ExecutionPlan) -> None:
        if not (self.verbose and self.is_master):
            return
        print(f"[AutoExecutor] ━━━ 执行计划 ━━━")
        print(f"  操作: {plan.operation}")
        print(f"  策略: {plan.description}")
        print(f"  数据总量: {plan.total_data_memory_gb:.3f} GB")
        print(
            f"  GPU可用显存: {plan.total_available_gb:.2f} GB "
            f"(单卡最小 {plan.min_gpu_available_gb:.2f} GB)"
        )
        print(f"  每GPU/批: {plan.per_gpu_memory_gb:.3f} GB")
        if plan.num_batches > 1:
            preview = plan.batch_sizes[:3]
            suffix = (
                f"...共{plan.num_batches}批" if len(plan.batch_sizes) > 3 else ""
            )
            print(f"  批次: {preview}{suffix}")

    # ==================== GPU 辅助 ====================

    def _to_gpu(self, tensor: _TensorOrNone) -> _TensorOrNone:
        """将张量移至本进程对应的 GPU，None 直接返回。"""
        if tensor is None:
            return None
        return tensor.cuda(self.mpi.get_gpu_id())

    def _to_cpu(self, tensor: _TensorOrNone, return_on_cpu: bool) -> _TensorOrNone:
        """根据 ``return_on_cpu`` 标志决定是否拷回 CPU。"""
        if tensor is None:
            return None
        return tensor.cpu() if return_on_cpu else tensor

    def _get_pipeline(self) -> Any:
        """延迟初始化流水线优化器。"""
        if self._pipeline is None:
            from .pipeline_optimizer import PipelineOptimizer, PipelineConfig
            self._pipeline = PipelineOptimizer(self.mpi, PipelineConfig())
        return self._pipeline

    # ==================== 元信息同步 ====================

    def _sync_meta(self, tensor: _TensorOrNone) -> Dict[str, Any]:
        """同步单个张量的 shape / dtype 到所有进程。"""
        if self.is_master:
            meta: Dict[str, Any] = {
                "shape": list(tensor.shape),  # type: ignore[union-attr]
                "dtype": tensor.dtype,  # type: ignore[union-attr]
            }
        else:
            meta = None  # type: ignore[assignment]
        return self.mpi.broadcast(meta)

    def _sync_meta_multi(self, *tensors: _TensorOrNone) -> Dict[str, Any]:
        """同步多个张量的 shapes / dtype 到所有进程。"""
        if self.is_master:
            shapes = [list(t.shape) for t in tensors if t is not None]  # type: ignore[union-attr]
            dtype = next(t.dtype for t in tensors if t is not None)  # type: ignore[union-attr]
            meta: Dict[str, Any] = {"shapes": shapes, "dtype": dtype}
        else:
            meta = None  # type: ignore[assignment]
        return self.mpi.broadcast(meta)

    def _sync_matmul_meta(
        self, A: _TensorOrNone, B: _TensorOrNone
    ) -> Tuple[int, int, int, torch.dtype, ExecutionPlan]:
        """矩阵乘法专用：同步 M/K/N/dtype 并生成执行计划。"""
        if self.is_master:
            meta = {
                "M": A.shape[0], "K": A.shape[1], "N": B.shape[1],  # type: ignore[union-attr]
                "dtype": A.dtype,  # type: ignore[union-attr]
            }
        else:
            meta = None  # type: ignore[assignment]
        meta = self.mpi.broadcast(meta)
        M, K, N, dtype = meta["M"], meta["K"], meta["N"], meta["dtype"]
        plan = self.planner.plan_matmul(M, K, N, dtype)
        self._log_plan(plan)
        if not plan.feasible:
            raise RuntimeError(f"[AutoExecutor] 不可行: {plan.description}")
        return M, K, N, dtype, plan

    def _make_plan(
        self,
        op_name: str,
        meta: Dict[str, Any],
        shapes_override: Optional[List[tuple]] = None,
    ) -> ExecutionPlan:
        """生成执行计划并打印。"""
        shapes = shapes_override or [tuple(meta["shape"])]
        plan = self.planner.plan_operation(op_name, shapes, meta["dtype"])
        self._log_plan(plan)
        return plan

    # ================================================================
    #              ⚡ 单卡直连快速路径（核心优化）
    # ================================================================

    def _single_gpu_compute(
        self,
        compute_fn: _ComputeFn,
        *cpu_tensors: _TensorOrNone,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """
        单卡快速路径：仅 master 在 GPU 上计算，跳过所有分布式通信。

        当 ResourcePlanner 判定数据可放入一张卡时调用此路径，
        避免 scatter/gather/allreduce 等 MPI 集合通信开销。
        非 master 进程直接返回 None。

        Args:
            compute_fn: 接受 GPU 张量的计算函数
            *cpu_tensors: CPU 上的输入张量（仅 master 有值）
            return_on_cpu: 是否将结果拷回 CPU

        Returns:
            master 返回计算结果，其他进程返回 None
        """
        if self.is_master:
            gpu_id = self.mpi.get_gpu_id()
            gpu_tensors = [
                t.cuda(gpu_id) if t is not None else None for t in cpu_tensors
            ]
            result = compute_fn(*gpu_tensors)
            # 释放输入显存
            del gpu_tensors
            self.gpu.empty_cache()
            return result.cpu() if return_on_cpu else result
        return None

    # ================================================================
    #            通用分批执行器（多卡路径）
    # ================================================================

    def _run_unary_op(
        self,
        tensor: _TensorOrNone,
        op_fn: _ComputeFn,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """单批次单输入：CPU→GPU→op→CPU"""
        t_gpu = self._to_gpu(tensor) if self.is_master else None
        result = op_fn(t_gpu)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def _batched_unary_op(
        self,
        tensor: _TensorOrNone,
        plan: ExecutionPlan,
        op_fn: _ComputeFn,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """
        分批单输入操作（FFT/IFFT/RFFT/Transpose 等）

        沿 dim-0 分批 → 每批 op_fn → cat(dim=0)
        """
        self._log(f"启动分批处理: {plan.num_batches} 批次")
        results: List[torch.Tensor] = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}: [{offset}:{offset + bs}]")
            if self.is_master:
                batch = tensor[offset : offset + bs].cuda(self.mpi.get_gpu_id())  # type: ignore[index]
            else:
                batch = None
            r = op_fn(batch)
            if self.is_master:
                results.append(r.cpu() if return_on_cpu else r)
            self.gpu.empty_cache()
            offset += bs
        return torch.cat(results, dim=0) if self.is_master else None

    def _run_binary_op(
        self,
        A: _TensorOrNone,
        B: _TensorOrNone,
        op_fn: _ComputeFn,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """单批次双输入：CPU→GPU→op→CPU"""
        if self.is_master:
            a_gpu, b_gpu = self._to_gpu(A), self._to_gpu(B)
        else:
            a_gpu = b_gpu = None
        result = op_fn(a_gpu, b_gpu)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def _batched_binary_op(
        self,
        A: _TensorOrNone,
        B: _TensorOrNone,
        plan: ExecutionPlan,
        op_fn: _ComputeFn,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """
        分批双输入操作（Add / BatchMatMul 等）

        A 和 B 均沿 dim-0 分批 → 每批 op_fn → cat(dim=0)
        """
        self._log(f"启动分批处理: {plan.num_batches} 批次")
        results: List[torch.Tensor] = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}: [{offset}:{offset + bs}]")
            if self.is_master:
                a_b = A[offset : offset + bs].cuda(self.mpi.get_gpu_id())  # type: ignore[index]
                b_b = B[offset : offset + bs].cuda(self.mpi.get_gpu_id())  # type: ignore[index]
            else:
                a_b = b_b = None
            r = op_fn(a_b, b_b)
            if self.is_master:
                results.append(r.cpu() if return_on_cpu else r)
            self.gpu.empty_cache()
            offset += bs
        return torch.cat(results, dim=0) if self.is_master else None

    def _batched_matmul_variant(
        self,
        A: _TensorOrNone,
        B: _TensorOrNone,
        plan: ExecutionPlan,
        op_fn: _ComputeFn,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """
        分批 Matmul 变体（B 只传输一次，A 沿行分批）

        适用于: matmul, matmul_mixed_precision, matmul_sparse_aware
        """
        self._log(f"启动分批处理: {plan.num_batches} 批次 (B 只传输一次)")
        if self.is_master:
            B_gpu = self._to_gpu(B)
        else:
            B_gpu = None
        B_all = self.distributor.broadcast(B_gpu)
        if self.is_master:
            del B_gpu
            self.gpu.empty_cache()

        results: List[torch.Tensor] = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}: 行[{offset}:{offset + bs}]")
            if self.is_master:
                A_batch = A[offset : offset + bs].cuda(self.mpi.get_gpu_id())  # type: ignore[index]
            else:
                A_batch = None
            A_local = self.distributor.distribute(A_batch, dim=0)
            C_local = torch.matmul(A_local, B_all)
            C_batch = self.distributor.gather(C_local, dim=0)
            if self.is_master:
                results.append(C_batch.cpu() if return_on_cpu else C_batch)
            del A_local, C_local
            if A_batch is not None:
                del A_batch
            self.gpu.empty_cache()
            offset += bs
        return torch.cat(results, dim=0) if self.is_master else None

    def _batched_conv(
        self,
        input_t: _TensorOrNone,
        weight: _TensorOrNone,
        bias: _TensorOrNone,
        plan: ExecutionPlan,
        conv_fn: Callable,
        stride: Union[Tuple[int, int], Tuple[int, int, int]],
        padding: Union[Tuple[int, int], Tuple[int, int, int]],
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """
        分批卷积（weight/bias 只传输一次，input 沿 batch-dim 分批）

        适用于: conv2d, conv3d
        """
        self._log(f"启动分批处理: {plan.num_batches} 批次 (weight 只传输一次)")
        w_all = self.distributor.broadcast(
            self._to_gpu(weight) if self.is_master else None
        )
        has_bias: bool = self.mpi.broadcast(
            bias is not None if self.is_master else None
        )
        b_all: _TensorOrNone = None
        if has_bias:
            b_all = self.distributor.broadcast(
                self._to_gpu(bias) if self.is_master else None
            )

        results: List[torch.Tensor] = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}")
            if self.is_master:
                batch = input_t[offset : offset + bs].cuda(self.mpi.get_gpu_id())  # type: ignore[index]
            else:
                batch = None
            inp_local = self.distributor.distribute(batch, dim=0)
            out_local = conv_fn(inp_local, w_all, b_all, stride=stride, padding=padding)
            out_batch = self.distributor.gather(out_local, dim=0)
            if self.is_master:
                results.append(out_batch.cpu() if return_on_cpu else out_batch)
            del inp_local, out_local
            self.gpu.empty_cache()
            offset += bs
        return torch.cat(results, dim=0) if self.is_master else None

    def _batched_reduction(
        self,
        tensor: _TensorOrNone,
        plan: ExecutionPlan,
        op_name: str,
        dim: Optional[int] = None,
        keepdim: bool = False,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """
        分批归约操作

        策略：
        - dim != 0 且 dim is not None → 各批独立归约 → cat(dim=0)
        - dim == 0 或 dim is None    → 各批 sum/max/min → 跨批合并
        """
        from .algorithms.reduction import (
            distributed_sum,
            distributed_mean,
            distributed_max,
            distributed_min,
            distributed_sum_kahan,
            distributed_mean_kahan,
        )

        self._log(f"启动分批归约: {plan.num_batches} 批次")
        total_shape: List[int] = self.mpi.broadcast(
            list(tensor.shape) if self.is_master else None  # type: ignore[union-attr]
        )
        ndim = len(total_shape)

        # 如果 dim 不是 0，各批可以独立处理
        independent = dim is not None and dim != 0 and (dim + ndim) % ndim != 0

        # 底层函数映射
        fn_map: Dict[str, Callable] = {
            "sum": distributed_sum,
            "mean": distributed_mean,
            "max": distributed_max,
            "min": distributed_min,
            "sum_kahan": distributed_sum_kahan,
            "mean_kahan": distributed_mean_kahan,
        }

        if independent:
            # 各批独立归约 → cat(dim=0)
            results: List[torch.Tensor] = []
            offset = 0
            for i, bs in enumerate(plan.batch_sizes):
                self._log(f"  批次 {i + 1}/{plan.num_batches}")
                batch = (
                    tensor[offset : offset + bs].cuda(self.mpi.get_gpu_id())  # type: ignore[index]
                    if self.is_master
                    else None
                )
                p = fn_map[op_name](
                    batch, self.mpi, self.distributor, dim=dim, keepdim=keepdim
                )
                if self.is_master:
                    results.append(p.cpu())
                self.gpu.empty_cache()
                offset += bs
            return torch.cat(results, dim=0) if self.is_master else None

        # dim==0 或 dim is None → 需要跨批合并
        if op_name in ("sum", "mean", "sum_kahan", "mean_kahan"):
            batch_fn = fn_map.get(
                "sum_kahan" if "kahan" in op_name else "sum", distributed_sum
            )
        elif op_name == "max":
            batch_fn = distributed_max
        elif op_name == "min":
            batch_fn = distributed_min
        else:
            batch_fn = fn_map.get(op_name, distributed_sum)

        partials: List[torch.Tensor] = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}")
            batch = (
                tensor[offset : offset + bs].cuda(self.mpi.get_gpu_id())  # type: ignore[index]
                if self.is_master
                else None
            )
            p = batch_fn(batch, self.mpi, self.distributor, dim=dim, keepdim=keepdim)
            if self.is_master:
                partials.append(p.cpu())
            self.gpu.empty_cache()
            offset += bs

        if not self.is_master:
            return None

        return self._merge_reduction_partials(partials, op_name, dim, total_shape)

    def _merge_reduction_partials(
        self,
        partials: List[torch.Tensor],
        op_name: str,
        dim: Optional[int],
        total_shape: List[int],
    ) -> torch.Tensor:
        """合并分批归约的中间结果。仅 master 调用。"""
        if op_name in ("sum", "sum_kahan"):
            result = partials[0]
            for p in partials[1:]:
                result = result + p
            return result
        elif op_name in ("mean", "mean_kahan"):
            total = partials[0]
            for p in partials[1:]:
                total = total + p
            if dim is None:
                numel = 1
                for s in total_shape:
                    numel *= s
                return total / numel
            else:
                return total / total_shape[dim]
        elif op_name == "max":
            result = partials[0]
            for p in partials[1:]:
                result = torch.maximum(result, p)
            return result
        elif op_name == "min":
            result = partials[0]
            for p in partials[1:]:
                result = torch.minimum(result, p)
            return result
        # fallback
        return partials[0]

    # ================================================================
    #  卷积 / FFT / Einsum / Stencil 的通用多卡单批 + 分批路由
    # ================================================================

    def _route_unary(
        self,
        tensor: _TensorOrNone,
        plan: ExecutionPlan,
        single_gpu_fn: _ComputeFn,
        dist_fn: _ComputeFn,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """统一路由：单卡 / 多卡单批 / 多卡分批（单输入操作）。"""
        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                single_gpu_fn, tensor, return_on_cpu=return_on_cpu
            )
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, dist_fn, return_on_cpu)
        return self._batched_unary_op(tensor, plan, dist_fn, return_on_cpu)

    def _route_binary(
        self,
        A: _TensorOrNone,
        B: _TensorOrNone,
        plan: ExecutionPlan,
        single_gpu_fn: _ComputeFn,
        dist_fn: _ComputeFn,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """统一路由：单卡 / 多卡单批 / 多卡分批（双输入操作）。"""
        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                single_gpu_fn, A, B, return_on_cpu=return_on_cpu
            )
        if plan.num_batches == 1:
            return self._run_binary_op(A, B, dist_fn, return_on_cpu)
        return self._batched_binary_op(A, B, plan, dist_fn, return_on_cpu)

    def _route_conv(
        self,
        input_tensor: _TensorOrNone,
        weight: _TensorOrNone,
        bias: _TensorOrNone,
        plan: ExecutionPlan,
        single_gpu_fn: _ComputeFn,
        dist_conv_fn: Callable,
        torch_conv_fn: Callable,
        stride: Union[Tuple[int, int], Tuple[int, int, int]],
        padding: Union[Tuple[int, int], Tuple[int, int, int]],
        dilation: Union[Tuple[int, int], Tuple[int, int, int]],
        groups: int,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """统一路由卷积操作：单卡 / 多卡单批 / 多卡分批。"""
        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                single_gpu_fn, input_tensor, weight, bias,
                return_on_cpu=return_on_cpu,
            )
        if plan.num_batches == 1:
            if self.is_master:
                gpu = self.mpi.get_gpu_id()
                i_g = input_tensor.cuda(gpu)  # type: ignore[union-attr]
                w_g = weight.cuda(gpu)  # type: ignore[union-attr]
                b_g = bias.cuda(gpu) if bias is not None else None
            else:
                i_g = w_g = b_g = None
            r = dist_conv_fn(
                i_g, w_g, self.mpi, self.distributor,
                bias=b_g, stride=stride, padding=padding,
                dilation=dilation, groups=groups,
            )
            return self._to_cpu(r, return_on_cpu) if self.is_master else None
        return self._batched_conv(
            input_tensor, weight, bias, plan,
            torch_conv_fn, stride, padding, return_on_cpu,
        )

    # ================================================================
    #                        矩阵运算 (6 个)
    # ================================================================

    def matmul(
        self,
        A: _TensorOrNone = None,
        B: _TensorOrNone = None,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """
        显存感知的自动化矩阵乘法 C = A @ B

        ⚡ 单卡直连 → 多卡并行 → 超显存分批，三级自适应
        """
        M, K, N, dtype, plan = self._sync_matmul_meta(A, B)
        self._log(f"MatMul [{M},{K}] × [{K},{N}]")

        t0 = time.time()
        if plan.num_gpus_active == 1:
            result = self._single_gpu_compute(
                lambda a, b: torch.matmul(a, b), A, B, return_on_cpu=return_on_cpu
            )
        elif plan.num_batches == 1:
            result = self._matmul_single(A, B, return_on_cpu)
        else:
            result = self._batched_matmul_variant(
                A, B, plan, lambda a, b: None, return_on_cpu
            )
        self._log(f"完成: {time.time() - t0:.3f}s")
        return result

    def _matmul_single(
        self, A: _TensorOrNone, B: _TensorOrNone, return_on_cpu: bool
    ) -> _TensorOrNone:
        from .algorithms.matrix_ops import distributed_matmul

        if self.is_master:
            A_g, B_g = self._to_gpu(A), self._to_gpu(B)
        else:
            A_g = B_g = None
        r = distributed_matmul(A_g, B_g, self.mpi, self.distributor, self.cost_model)
        return self._to_cpu(r, return_on_cpu) if self.is_master else None

    def batch_matmul(
        self,
        A: _TensorOrNone = None,
        B: _TensorOrNone = None,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """
        分布式批量矩阵乘法 C[b] = A[b] @ B[b]

        ✅ 支持超显存自动分批（A 和 B 均沿 batch-dim 分批）
        """
        from .algorithms.matrix_ops import distributed_batch_matmul

        meta = self._sync_meta_multi(A, B)
        self._log(f"BatchMatMul A={meta['shapes'][0]} B={meta['shapes'][1]}")
        plan = self.planner.plan_operation(
            "batch_matmul", [tuple(s) for s in meta["shapes"]], meta["dtype"]
        )
        self._log_plan(plan)

        return self._route_binary(
            A, B, plan,
            single_gpu_fn=lambda a, b: torch.bmm(a, b),
            dist_fn=lambda a, b: distributed_batch_matmul(
                a, b, self.mpi, self.distributor
            ),
            return_on_cpu=return_on_cpu,
        )

    def transpose(
        self,
        tensor: _TensorOrNone = None,
        dim0: int = 0,
        dim1: int = 1,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式矩阵转置  ✅ 支持超显存自动分批"""
        from .algorithms.matrix_ops import distributed_transpose

        meta = self._sync_meta(tensor)
        self._log(f"Transpose shape={meta['shape']} dims=({dim0},{dim1})")
        plan = self._make_plan("transpose", meta)

        return self._route_unary(
            tensor, plan,
            single_gpu_fn=lambda t: t.transpose(dim0, dim1).contiguous(),
            dist_fn=lambda t: distributed_transpose(
                t, self.mpi, self.distributor, dim0=dim0, dim1=dim1
            ),
            return_on_cpu=return_on_cpu,
        )

    def add(
        self,
        A: _TensorOrNone = None,
        B: _TensorOrNone = None,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式张量加法  ✅ 支持超显存自动分批"""
        from .algorithms.matrix_ops import distributed_add

        meta = self._sync_meta_multi(A, B)
        self._log(f"Add shapes={meta['shapes']}")
        plan = self.planner.plan_operation(
            "add", [tuple(s) for s in meta["shapes"]], meta["dtype"]
        )
        self._log_plan(plan)

        return self._route_binary(
            A, B, plan,
            single_gpu_fn=lambda a, b: a + b,
            dist_fn=lambda a, b: distributed_add(a, b, self.mpi, self.distributor),
            return_on_cpu=return_on_cpu,
        )

    def matmul_mixed_precision(
        self,
        A: _TensorOrNone = None,
        B: _TensorOrNone = None,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """混合精度分布式矩阵乘法（FP16 通信 + FP32 计算） ✅ 超显存分批"""
        from .algorithms.matrix_ops import distributed_matmul_mixed_precision

        M, K, N, dtype, plan = self._sync_matmul_meta(A, B)
        self._log(f"MatMul(MixedPrecision) [{M},{K}] × [{K},{N}]")

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.matmul(a.float(), b.float()),
                A, B, return_on_cpu=return_on_cpu,
            )

        op_fn = lambda a, b: distributed_matmul_mixed_precision(
            a, b, self.mpi, self.distributor
        )
        if plan.num_batches == 1:
            return self._run_binary_op(A, B, op_fn, return_on_cpu)
        return self._batched_matmul_variant(A, B, plan, op_fn, return_on_cpu)

    def matmul_sparse_aware(
        self,
        A: _TensorOrNone = None,
        B: _TensorOrNone = None,
        sparsity_threshold: float = 0.5,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """稀疏感知自适应分布式矩阵乘法  ✅ 超显存分批"""
        from .algorithms.matrix_ops import distributed_matmul_sparse_aware

        M, K, N, dtype, plan = self._sync_matmul_meta(A, B)
        self._log(f"MatMul(SparseAware) [{M},{K}] × [{K},{N}]")

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.matmul(a, b),
                A, B, return_on_cpu=return_on_cpu,
            )

        op_fn = lambda a, b: distributed_matmul_sparse_aware(
            a, b, self.mpi, self.distributor, sparsity_threshold=sparsity_threshold
        )
        if plan.num_batches == 1:
            return self._run_binary_op(A, B, op_fn, return_on_cpu)
        return self._batched_matmul_variant(A, B, plan, op_fn, return_on_cpu)

    # ================================================================
    #                        流水线 (2 个)
    # ================================================================

    def pipelined_matmul(
        self,
        A: _TensorOrNone = None,
        B: _TensorOrNone = None,
        num_chunks: int = 4,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """CUDA 双流流水线矩阵乘法（计算-通信重叠）  ⚡ 小数据自动降级单卡"""
        M, K, N, dtype, plan = self._sync_matmul_meta(A, B)
        self._log(f"PipelinedMatMul [{M},{K}] × [{K},{N}]")

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.matmul(a, b), A, B, return_on_cpu=return_on_cpu
            )

        pipe = self._get_pipeline()
        if self.is_master:
            A_gpu, B_gpu = self._to_gpu(A), self._to_gpu(B)
        else:
            A_gpu = B_gpu = None
        result = pipe.pipelined_matmul(A_gpu, B_gpu, num_chunks=num_chunks)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def pipelined_allreduce(
        self,
        tensor: _TensorOrNone = None,
        num_chunks: int = 4,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """CUDA 双流流水线 AllReduce  ⚡ 单卡时为恒等操作"""
        meta = self._sync_meta(tensor)
        plan = self._make_plan("allreduce", meta)

        if plan.num_gpus_active == 1:
            if self.is_master:
                return tensor.clone() if return_on_cpu else self._to_gpu(tensor)  # type: ignore[union-attr]
            return None

        pipe = self._get_pipeline()
        self._log(f"PipelinedAllReduce chunks={num_chunks}")
        if self.is_master:
            t_gpu = self._to_gpu(tensor)
        else:
            t_gpu = torch.zeros(
                meta["shape"], dtype=meta["dtype"]
            ).cuda(self.mpi.get_gpu_id())
        result = pipe.pipelined_allreduce(t_gpu, num_chunks=num_chunks)
        return self._to_cpu(result, return_on_cpu)

    # ================================================================
    #                        卷积 (2 个)
    # ================================================================

    def conv2d(
        self,
        input_tensor: _TensorOrNone = None,
        weight: _TensorOrNone = None,
        bias: _TensorOrNone = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """显存感知的自动化分布式 2D 卷积  ✅ 超显存分批"""
        from .algorithms.convolution import distributed_conv2d

        if self.is_master:
            meta: Dict[str, Any] = {
                "input_shape": list(input_tensor.shape),  # type: ignore[union-attr]
                "weight_shape": list(weight.shape),  # type: ignore[union-attr]
                "has_bias": bias is not None,
                "dtype": input_tensor.dtype,  # type: ignore[union-attr]
            }
        else:
            meta = None  # type: ignore[assignment]
        meta = self.mpi.broadcast(meta)
        self._log(f"Conv2d input={meta['input_shape']} weight={meta['weight_shape']}")
        plan = self.planner.plan_operation(
            "conv2d",
            [tuple(meta["input_shape"]), tuple(meta["weight_shape"])],
            meta["dtype"],
        )
        self._log_plan(plan)

        return self._route_conv(
            input_tensor, weight, bias, plan,
            single_gpu_fn=lambda i, w, b: F.conv2d(
                i, w, b, stride=stride, padding=padding,
                dilation=dilation, groups=groups,
            ),
            dist_conv_fn=distributed_conv2d,
            torch_conv_fn=F.conv2d,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            return_on_cpu=return_on_cpu,
        )

    def conv3d(
        self,
        input_tensor: _TensorOrNone = None,
        weight: _TensorOrNone = None,
        bias: _TensorOrNone = None,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """显存感知的自动化分布式 3D 卷积  ✅ 超显存分批"""
        from .algorithms.convolution import distributed_conv3d

        if self.is_master:
            meta: Dict[str, Any] = {
                "input_shape": list(input_tensor.shape),  # type: ignore[union-attr]
                "weight_shape": list(weight.shape),  # type: ignore[union-attr]
                "has_bias": bias is not None,
                "dtype": input_tensor.dtype,  # type: ignore[union-attr]
            }
        else:
            meta = None  # type: ignore[assignment]
        meta = self.mpi.broadcast(meta)
        self._log(f"Conv3d input={meta['input_shape']} weight={meta['weight_shape']}")
        plan = self.planner.plan_operation(
            "conv3d",
            [tuple(meta["input_shape"]), tuple(meta["weight_shape"])],
            meta["dtype"],
        )
        self._log_plan(plan)

        return self._route_conv(
            input_tensor, weight, bias, plan,
            single_gpu_fn=lambda i, w, b: F.conv3d(
                i, w, b, stride=stride, padding=padding,
                dilation=dilation, groups=groups,
            ),
            dist_conv_fn=distributed_conv3d,
            torch_conv_fn=F.conv3d,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            return_on_cpu=return_on_cpu,
        )

    # ================================================================
    #                     傅里叶变换 (5 个)
    # ================================================================

    def fft(
        self,
        input_tensor: _TensorOrNone = None,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "backward",
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式 1D FFT  ✅ 超显存分批"""
        from .algorithms.fft import distributed_fft

        meta = self._sync_meta(input_tensor)
        self._log(f"FFT shape={meta['shape']}")
        plan = self._make_plan("fft", meta)

        return self._route_unary(
            input_tensor, plan,
            single_gpu_fn=lambda t: torch.fft.fft(t, n=n, dim=dim, norm=norm),
            dist_fn=lambda t: distributed_fft(
                t, self.mpi, self.distributor, n=n, dim=dim, norm=norm
            ),
            return_on_cpu=return_on_cpu,
        )

    def ifft(
        self,
        input_tensor: _TensorOrNone = None,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "backward",
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式 1D IFFT（逆变换）  ✅ 超显存分批"""
        from .algorithms.fft import distributed_ifft

        meta = self._sync_meta(input_tensor)
        self._log(f"IFFT shape={meta['shape']}")
        plan = self._make_plan("ifft", meta)

        return self._route_unary(
            input_tensor, plan,
            single_gpu_fn=lambda t: torch.fft.ifft(t, n=n, dim=dim, norm=norm),
            dist_fn=lambda t: distributed_ifft(
                t, self.mpi, self.distributor, n=n, dim=dim, norm=norm
            ),
            return_on_cpu=return_on_cpu,
        )

    def fft2d(
        self,
        input_tensor: _TensorOrNone = None,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式 2D FFT  ✅ 超显存分批"""
        from .algorithms.fft import distributed_fft2d

        meta = self._sync_meta(input_tensor)
        self._log(f"FFT2D shape={meta['shape']}")
        plan = self._make_plan("fft2d", meta)

        return self._route_unary(
            input_tensor, plan,
            single_gpu_fn=lambda t: torch.fft.fft2(t),
            dist_fn=lambda t: distributed_fft2d(t, self.mpi, self.distributor),
            return_on_cpu=return_on_cpu,
        )

    def rfft(
        self,
        input_tensor: _TensorOrNone = None,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "backward",
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式实数 FFT  ✅ 超显存分批"""
        from .algorithms.fft import distributed_rfft

        meta = self._sync_meta(input_tensor)
        self._log(f"RFFT shape={meta['shape']}")
        plan = self._make_plan("rfft", meta)

        return self._route_unary(
            input_tensor, plan,
            single_gpu_fn=lambda t: torch.fft.rfft(t, n=n, dim=dim, norm=norm),
            dist_fn=lambda t: distributed_rfft(
                t, self.mpi, self.distributor, n=n, dim=dim, norm=norm
            ),
            return_on_cpu=return_on_cpu,
        )

    def fft2d_pencil(
        self,
        input_tensor: _TensorOrNone = None,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """Pencil 分解 2D FFT（适合超大网格）  ✅ 超显存分批"""
        from .algorithms.fft import distributed_fft2d_pencil

        meta = self._sync_meta(input_tensor)
        self._log(f"FFT2D_Pencil shape={meta['shape']}")
        plan = self._make_plan("fft2d_pencil", meta)

        return self._route_unary(
            input_tensor, plan,
            single_gpu_fn=lambda t: torch.fft.fft2(t),
            dist_fn=lambda t: distributed_fft2d_pencil(
                t, self.mpi, self.distributor
            ),
            return_on_cpu=return_on_cpu,
        )

    # ================================================================
    #                     Einstein 求和 (3 个)
    # ================================================================

    def einsum(
        self,
        equation: str,
        *operands: _TensorOrNone,
        return_on_cpu: bool = True,
        optimize: str = "auto",
        use_opt_einsum: bool = True,
    ) -> _TensorOrNone:
        """分布式 Einstein 求和（集成 opt_einsum 最优路径）"""
        from .algorithms.einsum import distributed_einsum

        if self.is_master:
            shapes = [list(op.shape) for op in operands]  # type: ignore[union-attr]
            meta: Dict[str, Any] = {
                "equation": equation,
                "shapes": shapes,
                "dtype": operands[0].dtype,  # type: ignore[union-attr]
            }
        else:
            meta = None  # type: ignore[assignment]
        meta = self.mpi.broadcast(meta)
        self._log(f"Einsum '{meta['equation']}' shapes={meta['shapes']}")

        plan = self.planner.plan_operation(
            "einsum", [tuple(s) for s in meta["shapes"]], meta["dtype"]
        )
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda *ops: torch.einsum(equation, *ops),
                *operands, return_on_cpu=return_on_cpu,
            )

        if self.is_master:
            gpu_ops = tuple(op.cuda(self.mpi.get_gpu_id()) for op in operands)  # type: ignore[union-attr]
        else:
            gpu_ops = tuple(None for _ in meta["shapes"])
        result = distributed_einsum(
            equation, *gpu_ops,
            mpi=self.mpi, distributor=self.distributor,
            optimize=optimize, use_opt_einsum=use_opt_einsum,
        )
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def einsum_with_path(
        self,
        equation: str,
        *operands: _TensorOrNone,
        path: Optional[List] = None,
        optimize: str = "auto",
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式 Einstein 求和（自定义收缩路径）"""
        from .algorithms.einsum import distributed_einsum_with_path

        if self.is_master:
            shapes = [list(op.shape) for op in operands]  # type: ignore[union-attr]
            meta: Dict[str, Any] = {
                "equation": equation,
                "shapes": shapes,
                "dtype": operands[0].dtype,  # type: ignore[union-attr]
            }
        else:
            meta = None  # type: ignore[assignment]
        meta = self.mpi.broadcast(meta)
        self._log(f"EinsumWithPath '{meta['equation']}' shapes={meta['shapes']}")

        plan = self.planner.plan_operation(
            "einsum_with_path", [tuple(s) for s in meta["shapes"]], meta["dtype"]
        )
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda *ops: torch.einsum(equation, *ops),
                *operands, return_on_cpu=return_on_cpu,
            )

        if self.is_master:
            gpu_ops = tuple(op.cuda(self.mpi.get_gpu_id()) for op in operands)  # type: ignore[union-attr]
        else:
            gpu_ops = tuple(None for _ in meta["shapes"])
        result = distributed_einsum_with_path(
            equation, *gpu_ops,
            mpi=self.mpi, distributor=self.distributor,
            path=path, optimize=optimize,
        )
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def tensordot(
        self,
        A: _TensorOrNone = None,
        B: _TensorOrNone = None,
        dims: int = 2,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式张量点积"""
        from .algorithms.einsum import distributed_tensordot

        meta = self._sync_meta_multi(A, B)
        self._log(
            f"Tensordot A={meta['shapes'][0]} B={meta['shapes'][1]} dims={dims}"
        )
        plan = self.planner.plan_operation(
            "tensordot", [tuple(s) for s in meta["shapes"]], meta["dtype"]
        )
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.tensordot(a, b, dims=dims),
                A, B, return_on_cpu=return_on_cpu,
            )

        return self._run_binary_op(
            A, B,
            lambda a, b: distributed_tensordot(
                a, b, self.mpi, self.distributor, dims=dims
            ),
            return_on_cpu,
        )

    # ================================================================
    #                     归约操作 (6 个)
    # ================================================================

    def _single_gpu_reduction(
        self,
        tensor: _TensorOrNone,
        op: str,
        dim: Optional[int],
        keepdim: bool,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """单卡归约的统一实现。"""
        _reduce_fns: Dict[str, _ComputeFn] = {
            "sum": lambda t: (
                torch.sum(t) if dim is None
                else torch.sum(t, dim=dim, keepdim=keepdim)
            ),
            "sum_kahan": lambda t: (
                torch.sum(t) if dim is None
                else torch.sum(t, dim=dim, keepdim=keepdim)
            ),
            "mean": lambda t: (
                torch.mean(t.float()) if dim is None
                else torch.mean(t.float(), dim=dim, keepdim=keepdim)
            ),
            "mean_kahan": lambda t: (
                torch.mean(t.float()) if dim is None
                else torch.mean(t.float(), dim=dim, keepdim=keepdim)
            ),
            "max": lambda t: (
                t.max() if dim is None
                else torch.max(t, dim=dim, keepdim=keepdim).values
            ),
            "min": lambda t: (
                t.min() if dim is None
                else torch.min(t, dim=dim, keepdim=keepdim).values
            ),
        }
        compute_fn = _reduce_fns.get(op, _reduce_fns["sum"])
        return self._single_gpu_compute(
            compute_fn, tensor, return_on_cpu=return_on_cpu
        )

    def _reduction_op(
        self,
        tensor: _TensorOrNone,
        op_name: str,
        dist_fn: Callable,
        dim: Optional[int],
        keepdim: bool,
        return_on_cpu: bool,
    ) -> _TensorOrNone:
        """归约操作的统一模板，消除 6 个方法间的重复代码。"""
        meta = self._sync_meta(tensor)
        self._log(f"{op_name.capitalize()} shape={meta['shape']} dim={dim}")
        plan = self._make_plan(op_name, meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_reduction(
                tensor, op_name, dim, keepdim, return_on_cpu
            )

        op_fn = lambda t: dist_fn(
            t, self.mpi, self.distributor, dim=dim, keepdim=keepdim
        )
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        return self._batched_reduction(
            tensor, plan, op_name,
            dim=dim, keepdim=keepdim, return_on_cpu=return_on_cpu,
        )

    def sum(
        self,
        tensor: _TensorOrNone = None,
        dim: Optional[int] = None,
        keepdim: bool = False,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式求和  ✅ 超显存分批"""
        from .algorithms.reduction import distributed_sum
        return self._reduction_op(
            tensor, "sum", distributed_sum, dim, keepdim, return_on_cpu
        )

    def mean(
        self,
        tensor: _TensorOrNone = None,
        dim: Optional[int] = None,
        keepdim: bool = False,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式均值  ✅ 超显存分批"""
        from .algorithms.reduction import distributed_mean
        return self._reduction_op(
            tensor, "mean", distributed_mean, dim, keepdim, return_on_cpu
        )

    def max(
        self,
        tensor: _TensorOrNone = None,
        dim: Optional[int] = None,
        keepdim: bool = False,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式最大值  ✅ 超显存分批"""
        from .algorithms.reduction import distributed_max
        return self._reduction_op(
            tensor, "max", distributed_max, dim, keepdim, return_on_cpu
        )

    def min(
        self,
        tensor: _TensorOrNone = None,
        dim: Optional[int] = None,
        keepdim: bool = False,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式最小值  ✅ 超显存分批"""
        from .algorithms.reduction import distributed_min
        return self._reduction_op(
            tensor, "min", distributed_min, dim, keepdim, return_on_cpu
        )

    def sum_kahan(
        self,
        tensor: _TensorOrNone = None,
        dim: Optional[int] = None,
        keepdim: bool = False,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """Kahan 补偿求和（高精度）  ✅ 超显存分批"""
        from .algorithms.reduction import distributed_sum_kahan
        return self._reduction_op(
            tensor, "sum_kahan", distributed_sum_kahan, dim, keepdim, return_on_cpu
        )

    def mean_kahan(
        self,
        tensor: _TensorOrNone = None,
        dim: Optional[int] = None,
        keepdim: bool = False,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """Kahan 补偿均值（高精度）  ✅ 超显存分批"""
        from .algorithms.reduction import distributed_mean_kahan
        return self._reduction_op(
            tensor, "mean_kahan", distributed_mean_kahan, dim, keepdim, return_on_cpu
        )

    # ================================================================
    #                   Stencil / PDE (2 个)
    # ================================================================

    def stencil_2d(
        self,
        grid: _TensorOrNone = None,
        stencil_kernel: _TensorOrNone = None,
        boundary: str = "zero",
        iterations: int = 1,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式 2D Stencil 计算（Halo Exchange）  ⚡ 小数据自动降级单卡"""
        from .algorithms.stencil import distributed_stencil_2d

        meta = self._sync_meta(grid)
        self._log(f"Stencil2D shape={meta['shape']} iters={iterations}")
        plan = self._make_plan("stencil_2d", meta)

        if plan.num_gpus_active == 1:
            # 单卡直接在 GPU 上迭代，无需 Halo Exchange
            if self.is_master:
                gpu_id = self.mpi.get_gpu_id()
                g = grid.float().cuda(gpu_id)  # type: ignore[union-attr]
                if stencil_kernel is not None:
                    k = stencil_kernel.float().cuda(gpu_id)
                else:
                    k = torch.tensor(
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        dtype=torch.float32, device=f"cuda:{gpu_id}",
                    )
                k_4d = k.unsqueeze(0).unsqueeze(0)
                pad = k.shape[-1] // 2
                for _ in range(iterations):
                    g_4d = g.unsqueeze(0).unsqueeze(0)
                    if boundary == "periodic":
                        g_4d = F.pad(g_4d, [pad] * 4, mode="circular")
                        g = F.conv2d(g_4d, k_4d).squeeze(0).squeeze(0)
                    else:
                        g = F.conv2d(g_4d, k_4d, padding=pad).squeeze(0).squeeze(0)
                result = g.to(grid.dtype)  # type: ignore[union-attr]
                self.gpu.empty_cache()
                return result.cpu() if return_on_cpu else result
            return None

        g_gpu = self._to_gpu(grid) if self.is_master else None
        k_gpu = (
            self._to_gpu(stencil_kernel)
            if (self.is_master and stencil_kernel is not None)
            else None
        )
        result = distributed_stencil_2d(
            g_gpu, self.mpi, self.distributor,
            stencil_kernel=k_gpu, boundary=boundary, iterations=iterations,
        )
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def jacobi_2d(
        self,
        grid: _TensorOrNone = None,
        rhs: _TensorOrNone = None,
        dx: float = 1.0,
        boundary: str = "zero",
        iterations: int = 100,
        tol: float = 1e-6,
        return_on_cpu: bool = True,
    ) -> _TensorOrNone:
        """分布式 2D Jacobi 迭代（求解 Poisson 方程）  ⚡ 小数据自动降级单卡"""
        from .algorithms.stencil import distributed_jacobi_2d

        meta = self._sync_meta(grid)
        self._log(f"Jacobi2D shape={meta['shape']} iters={iterations} tol={tol}")
        plan = self._make_plan("jacobi_2d", meta)

        if plan.num_gpus_active == 1:
            if self.is_master:
                gpu_id = self.mpi.get_gpu_id()
                g = grid.float().cuda(gpu_id)  # type: ignore[union-attr]
                f = rhs.float().cuda(gpu_id) if rhs is not None else torch.zeros_like(g)
                dx2 = dx * dx
                for it in range(iterations):
                    g_old = g.clone()
                    g_pad = F.pad(
                        g.unsqueeze(0).unsqueeze(0),
                        [1, 1, 1, 1], mode="constant", value=0,
                    )
                    neighbors = (
                        g_pad[:, :, :-2, 1:-1]
                        + g_pad[:, :, 2:, 1:-1]
                        + g_pad[:, :, 1:-1, :-2]
                        + g_pad[:, :, 1:-1, 2:]
                    )
                    g = (neighbors.squeeze(0).squeeze(0) - dx2 * f) / 4.0
                    diff = torch.max(torch.abs(g - g_old)).item()
                    if diff < tol:
                        break
                result = g.to(grid.dtype)  # type: ignore[union-attr]
                self.gpu.empty_cache()
                return result.cpu() if return_on_cpu else result
            return None

        g_gpu = self._to_gpu(grid) if self.is_master else None
        r_gpu = (
            self._to_gpu(rhs)
            if (self.is_master and rhs is not None)
            else None
        )
        result = distributed_jacobi_2d(
            g_gpu, r_gpu, self.mpi, self.distributor,
            dx=dx, boundary=boundary, iterations=iterations, tol=tol,
        )
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    # ================================================================
    #                       批量操作
    # ================================================================

    def matmul_batch(
        self,
        pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_on_cpu: bool = True,
    ) -> Optional[List[torch.Tensor]]:
        """批量矩阵乘法：处理多对 (A, B)"""
        if self.is_master:
            num_pairs = len(pairs)  # type: ignore[arg-type]
        else:
            num_pairs = None  # type: ignore[assignment]
        num_pairs = self.mpi.broadcast(num_pairs)
        self._log(f"批量 MatMul: {num_pairs} 对矩阵")
        results: List[torch.Tensor] = []
        for i in range(num_pairs):
            self._log(f"  === 第 {i + 1}/{num_pairs} 对 ===")
            if self.is_master:
                A_i, B_i = pairs[i]  # type: ignore[index]
            else:
                A_i = B_i = None
            r = self.matmul(A_i, B_i, return_on_cpu=return_on_cpu)
            if self.is_master:
                results.append(r)  # type: ignore[arg-type]
        return results if self.is_master else None

    # ================================================================
    #                       信息查询
    # ================================================================

    def gpu_status(self) -> None:
        """打印当前 GPU 显存状态。"""
        self.planner.print_gpu_status()

    def plan_info(
        self, op: str, *shapes: tuple, dtype: torch.dtype = torch.float32
    ) -> ExecutionPlan:
        """查看执行计划（不执行）。"""
        if op == "matmul" and len(shapes) == 2:
            M, K = shapes[0]
            N = shapes[1][1]
            plan = self.planner.plan_matmul(M, K, N, dtype)
        else:
            plan = self.planner.plan_operation(op, list(shapes), dtype)
        self._log_plan(plan)
        return plan


# ==================== 便捷函数 ====================

_global_executor: Optional[AutoExecutor] = None


def auto_compute(op: str, *args: Any, **kwargs: Any) -> _TensorOrNone:
    """
    一行式自动化分布式计算 — 智能路由：单卡直连 / 多卡并行 / 超显存分批

    所有 MPI 进程都必须调用此函数。

    支持全部 26 个操作:
        矩阵运算(6): matmul, batch_matmul, transpose, add,
                      matmul_mixed_precision, matmul_sparse_aware
        流水线(2):    pipelined_matmul, pipelined_allreduce
        卷积(2):      conv2d, conv3d
        FFT(5):       fft, ifft, fft2d, rfft, fft2d_pencil
        Einsum(3):    einsum, einsum_with_path, tensordot
        归约(6):      sum, mean, max, min, sum_kahan, mean_kahan
        Stencil(2):   stencil_2d, jacobi_2d

    用法::

        C = auto_compute("matmul", A_cpu, B_cpu)
        Y = auto_compute("fft", X_cpu)
        S = auto_compute("sum", X_cpu, dim=0)
    """
    global _global_executor

    verbose: bool = kwargs.pop("verbose", True)
    max_per_gpu_gb: Optional[float] = kwargs.pop("max_per_gpu_gb", None)
    return_on_cpu: bool = kwargs.pop("return_on_cpu", True)

    if _global_executor is None:
        _global_executor = AutoExecutor(
            verbose=verbose, max_per_gpu_gb=max_per_gpu_gb
        )
    ex = _global_executor

    dispatch: Dict[str, Callable[[], _TensorOrNone]] = {
        # 矩阵运算 (6)
        "matmul": lambda: ex.matmul(*args, return_on_cpu=return_on_cpu, **kwargs),
        "batch_matmul": lambda: ex.batch_matmul(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        "transpose": lambda: ex.transpose(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        "add": lambda: ex.add(*args, return_on_cpu=return_on_cpu, **kwargs),
        "matmul_mixed_precision": lambda: ex.matmul_mixed_precision(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        "matmul_sparse_aware": lambda: ex.matmul_sparse_aware(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        # 流水线 (2)
        "pipelined_matmul": lambda: ex.pipelined_matmul(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        "pipelined_allreduce": lambda: ex.pipelined_allreduce(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        # 卷积 (2)
        "conv2d": lambda: ex.conv2d(*args, return_on_cpu=return_on_cpu, **kwargs),
        "conv3d": lambda: ex.conv3d(*args, return_on_cpu=return_on_cpu, **kwargs),
        # FFT (5)
        "fft": lambda: ex.fft(*args, return_on_cpu=return_on_cpu, **kwargs),
        "ifft": lambda: ex.ifft(*args, return_on_cpu=return_on_cpu, **kwargs),
        "fft2d": lambda: ex.fft2d(*args, return_on_cpu=return_on_cpu, **kwargs),
        "rfft": lambda: ex.rfft(*args, return_on_cpu=return_on_cpu, **kwargs),
        "fft2d_pencil": lambda: ex.fft2d_pencil(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        # Einsum (3)
        "einsum": lambda: ex.einsum(
            args[0], *args[1:], return_on_cpu=return_on_cpu, **kwargs
        ),
        "einsum_with_path": lambda: ex.einsum_with_path(
            args[0], *args[1:], return_on_cpu=return_on_cpu, **kwargs
        ),
        "tensordot": lambda: ex.tensordot(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        # 归约 (6)
        "sum": lambda: ex.sum(*args, return_on_cpu=return_on_cpu, **kwargs),
        "mean": lambda: ex.mean(*args, return_on_cpu=return_on_cpu, **kwargs),
        "max": lambda: ex.max(*args, return_on_cpu=return_on_cpu, **kwargs),
        "min": lambda: ex.min(*args, return_on_cpu=return_on_cpu, **kwargs),
        "sum_kahan": lambda: ex.sum_kahan(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        "mean_kahan": lambda: ex.mean_kahan(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        # Stencil / PDE (2)
        "stencil_2d": lambda: ex.stencil_2d(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
        "jacobi_2d": lambda: ex.jacobi_2d(
            *args, return_on_cpu=return_on_cpu, **kwargs
        ),
    }

    if op not in dispatch:
        supported = ", ".join(sorted(dispatch.keys()))
        raise ValueError(f"不支持的操作: '{op}'. 可选:\n  {supported}")

    return dispatch[op]()
