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

import time
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from .mpi_manager import MPIManager
from .tensor_distributor import TensorDistributor
from .gpu_manager import GPUManager
from .cost_model import CostModel, ClusterConfig
from .resource_planner import ResourcePlanner, ExecutionPlan


class AutoExecutor:
    """
    显存感知的自动化分布式计算执行器

    ⚡ 智能路由：
      - 单卡直连：数据能放进一张卡时，跳过全部 MPI 通信，性能接近原生 PyTorch/CuPy
      - 多卡并行：数据超过单卡时，自动分布到多 GPU 并行计算
      - 超显存分批：数据超过所有 GPU 总显存时，自动分批处理

    支持全部 26 个分布式算子，其中 22 个支持超显存自动分批。
    """

    def __init__(self, mpi: Optional[MPIManager] = None,
                 verbose: bool = True,
                 max_per_gpu_gb: Optional[float] = None):
        self.mpi = mpi or MPIManager()
        self.distributor = TensorDistributor(self.mpi)
        self.gpu = GPUManager(self.mpi.get_gpu_id())
        self.planner = ResourcePlanner(self.mpi, max_per_gpu_gb=max_per_gpu_gb)
        config = ClusterConfig.from_auto_detect(self.mpi.get_size())
        self.cost_model = CostModel(config)
        self.verbose = verbose
        self.is_master = self.mpi.is_master_process()
        self._pipeline = None

    # ==================== 日志 ====================

    def _log(self, msg: str):
        if self.verbose and self.is_master:
            print(f"[AutoExecutor] {msg}")

    def _log_plan(self, plan: ExecutionPlan):
        if self.verbose and self.is_master:
            print(f"[AutoExecutor] ━━━ 执行计划 ━━━")
            print(f"  操作: {plan.operation}")
            print(f"  策略: {plan.description}")
            print(f"  数据总量: {plan.total_data_memory_gb:.3f} GB")
            print(f"  GPU可用显存: {plan.total_available_gb:.2f} GB "
                  f"(单卡最小 {plan.min_gpu_available_gb:.2f} GB)")
            print(f"  每GPU/批: {plan.per_gpu_memory_gb:.3f} GB")
            if plan.num_batches > 1:
                preview = plan.batch_sizes[:3]
                suffix = f"...共{plan.num_batches}批" if len(plan.batch_sizes) > 3 else ""
                print(f"  批次: {preview}{suffix}")

    def _to_gpu(self, tensor):
        if tensor is None:
            return None
        return tensor.cuda(self.mpi.get_gpu_id())

    def _to_cpu(self, tensor, return_on_cpu):
        if tensor is None:
            return None
        return tensor.cpu() if return_on_cpu else tensor

    def _get_pipeline(self):
        if self._pipeline is None:
            from .pipeline_optimizer import PipelineOptimizer, PipelineConfig
            self._pipeline = PipelineOptimizer(self.mpi, self.distributor,
                                                PipelineConfig())
        return self._pipeline

    # ==================== 元信息同步 ====================

    def _sync_meta(self, tensor):
        if self.is_master:
            meta = {'shape': list(tensor.shape), 'dtype': tensor.dtype}
        else:
            meta = None
        return self.mpi.broadcast(meta)

    def _sync_meta_multi(self, *tensors):
        if self.is_master:
            shapes = [list(t.shape) for t in tensors if t is not None]
            dtype = tensors[0].dtype
            meta = {'shapes': shapes, 'dtype': dtype}
        else:
            meta = None
        return self.mpi.broadcast(meta)

    def _make_plan(self, op_name, meta, shapes_override=None):
        """生成执行计划并打印"""
        if shapes_override:
            shapes = shapes_override
        else:
            shapes = [tuple(meta['shape'])]
        plan = self.planner.plan_operation(op_name, shapes, meta['dtype'])
        self._log_plan(plan)
        return plan

    # ================================================================
    #              ⚡ 单卡直连快速路径（核心优化）
    # ================================================================

    def _single_gpu_compute(self, compute_fn, *cpu_tensors, return_on_cpu=True):
        """
        单卡快速路径：仅 master 在 GPU 上计算，跳过所有分布式通信。

        当 ResourcePlanner 判定数据可放入一张卡时调用此路径，
        避免 scatter/gather/allreduce 等 MPI 集合通信开销。
        非 master 进程直接返回 None，不做任何 GPU 操作。

        Args:
            compute_fn: 接受 GPU 张量的计算函数
            *cpu_tensors: CPU 上的输入张量（仅 master 有值，其他进程为 None）
            return_on_cpu: 是否将结果拷回 CPU

        Returns:
            master 返回计算结果，其他进程返回 None
        """
        if self.is_master:
            gpu_id = self.mpi.get_gpu_id()
            gpu_tensors = []
            for t in cpu_tensors:
                if t is not None:
                    gpu_tensors.append(t.cuda(gpu_id))
                else:
                    gpu_tensors.append(None)
            result = compute_fn(*gpu_tensors)
            # 释放输入显存
            for t in gpu_tensors:
                if t is not None:
                    del t
            self.gpu.empty_cache()
            return result.cpu() if return_on_cpu else result
        return None

    # ================================================================
    #            通用分批执行器（多卡路径）
    # ================================================================

    def _run_unary_op(self, tensor, op_fn, return_on_cpu):
        """单批次单输入：CPU→GPU→op→CPU"""
        t_gpu = self._to_gpu(tensor) if self.is_master else None
        result = op_fn(t_gpu)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def _batched_unary_op(self, tensor, plan, op_fn, return_on_cpu):
        """
        分批单输入操作（FFT/IFFT/RFFT/Transpose 等）

        沿 dim-0 分批 → 每批 op_fn → cat(dim=0)
        """
        self._log(f"启动分批处理: {plan.num_batches} 批次")
        results = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}: [{offset}:{offset + bs}]")
            if self.is_master:
                batch = tensor[offset:offset + bs].cuda(self.mpi.get_gpu_id())
            else:
                batch = None
            r = op_fn(batch)
            if self.is_master:
                results.append(r.cpu() if return_on_cpu else r)
            self.gpu.empty_cache()
            offset += bs
        return torch.cat(results, dim=0) if self.is_master else None

    def _run_binary_op(self, A, B, op_fn, return_on_cpu):
        """单批次双输入：CPU→GPU→op→CPU"""
        if self.is_master:
            a_gpu, b_gpu = self._to_gpu(A), self._to_gpu(B)
        else:
            a_gpu = b_gpu = None
        result = op_fn(a_gpu, b_gpu)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def _batched_binary_op(self, A, B, plan, op_fn, return_on_cpu):
        """
        分批双输入操作（Add / BatchMatMul 等）

        A 和 B 均沿 dim-0 分批 → 每批 op_fn → cat(dim=0)
        """
        self._log(f"启动分批处理: {plan.num_batches} 批次")
        results = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}: [{offset}:{offset + bs}]")
            if self.is_master:
                a_b = A[offset:offset + bs].cuda(self.mpi.get_gpu_id())
                b_b = B[offset:offset + bs].cuda(self.mpi.get_gpu_id())
            else:
                a_b = b_b = None
            r = op_fn(a_b, b_b)
            if self.is_master:
                results.append(r.cpu() if return_on_cpu else r)
            self.gpu.empty_cache()
            offset += bs
        return torch.cat(results, dim=0) if self.is_master else None

    def _batched_matmul_variant(self, A, B, plan, op_fn, return_on_cpu):
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

        results = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}: 行[{offset}:{offset + bs}]")
            if self.is_master:
                A_batch = A[offset:offset + bs].cuda(self.mpi.get_gpu_id())
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

    def _batched_conv(self, input_t, weight, bias, plan,
                       conv_fn, stride, padding, return_on_cpu,
                       dilation=None, groups=1):
        """
        分批卷积（weight/bias 只传输一次，input 沿 batch-dim 分批）

        适用于: conv2d, conv3d
        """
        self._log(f"启动分批处理: {plan.num_batches} 批次 (weight 只传输一次)")
        w_all = self.distributor.broadcast(
            self._to_gpu(weight) if self.is_master else None)
        has_bias = self.mpi.broadcast(bias is not None if self.is_master else None)
        b_all = None
        if has_bias:
            b_all = self.distributor.broadcast(
                self._to_gpu(bias) if self.is_master else None)

        results = []
        offset = 0
        for i, bs in enumerate(plan.batch_sizes):
            self._log(f"  批次 {i + 1}/{plan.num_batches}")
            if self.is_master:
                batch = input_t[offset:offset + bs].cuda(self.mpi.get_gpu_id())
            else:
                batch = None
            inp_local = self.distributor.distribute(batch, dim=0)
            out_local = conv_fn(inp_local, w_all, b_all,
                                 stride=stride, padding=padding)
            out_batch = self.distributor.gather(out_local, dim=0)
            if self.is_master:
                results.append(out_batch.cpu() if return_on_cpu else out_batch)
            del inp_local, out_local
            self.gpu.empty_cache()
            offset += bs
        return torch.cat(results, dim=0) if self.is_master else None

    def _batched_reduction(self, tensor, plan, op_name,
                            dim=None, keepdim=False, return_on_cpu=True):
        """
        分批归约操作

        策略：
        - dim != 0 且 dim is not None → 各批独立归约 → cat(dim=0)
        - dim == 0 或 dim is None    → 各批 sum/max/min → 跨批合并
        """
        from .algorithms.reduction import (
            distributed_sum, distributed_mean,
            distributed_max, distributed_min,
            distributed_sum_kahan, distributed_mean_kahan,
        )

        self._log(f"启动分批归约: {plan.num_batches} 批次")
        total_shape = self.mpi.broadcast(
            list(tensor.shape) if self.is_master else None)
        ndim = len(total_shape)

        # 如果 dim 不是 0，各批可以独立处理
        independent = (dim is not None and dim != 0 and (dim + ndim) % ndim != 0)

        # 选择底层函数
        fn_map = {
            "sum": distributed_sum,
            "mean": distributed_mean,
            "max": distributed_max,
            "min": distributed_min,
            "sum_kahan": distributed_sum_kahan,
            "mean_kahan": distributed_mean_kahan,
        }

        if independent:
            # 各批独立归约 → cat(dim=0)
            results = []
            offset = 0
            for i, bs in enumerate(plan.batch_sizes):
                self._log(f"  批次 {i + 1}/{plan.num_batches}")
                batch = tensor[offset:offset + bs].cuda(
                    self.mpi.get_gpu_id()) if self.is_master else None
                p = fn_map[op_name](batch, self.mpi, self.distributor,
                                     dim=dim, keepdim=keepdim)
                if self.is_master:
                    results.append(p.cpu())
                self.gpu.empty_cache()
                offset += bs
            return torch.cat(results, dim=0) if self.is_master else None
        else:
            # dim==0 或 dim is None → 需要跨批合并
            if op_name in ("sum", "mean", "sum_kahan", "mean_kahan"):
                batch_fn = fn_map.get("sum_kahan" if "kahan" in op_name else "sum")
            elif op_name == "max":
                batch_fn = distributed_max
            elif op_name == "min":
                batch_fn = distributed_min
            else:
                batch_fn = fn_map.get(op_name, distributed_sum)

            partials = []
            offset = 0
            for i, bs in enumerate(plan.batch_sizes):
                self._log(f"  批次 {i + 1}/{plan.num_batches}")
                batch = tensor[offset:offset + bs].cuda(
                    self.mpi.get_gpu_id()) if self.is_master else None
                p = batch_fn(batch, self.mpi, self.distributor,
                              dim=dim, keepdim=keepdim)
                if self.is_master:
                    partials.append(p.cpu())
                self.gpu.empty_cache()
                offset += bs

            if self.is_master:
                if op_name in ("sum", "sum_kahan"):
                    total = partials[0]
                    for p in partials[1:]:
                        total = total + p
                    return total
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
            return None

    # ================================================================
    #                        矩阵运算 (6 个)
    # ================================================================

    def matmul(self, A: Optional[torch.Tensor] = None,
               B: Optional[torch.Tensor] = None,
               return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        显存感知的自动化矩阵乘法 C = A @ B

        ⚡ 单卡直连 → 多卡并行 → 超显存分批，三级自适应
        """
        if self.is_master:
            meta = {'M': A.shape[0], 'K': A.shape[1], 'N': B.shape[1],
                    'dtype': A.dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        M, K, N, dtype = meta['M'], meta['K'], meta['N'], meta['dtype']

        self._log(f"MatMul [{M},{K}] × [{K},{N}]")
        plan = self.planner.plan_matmul(M, K, N, dtype)
        self._log_plan(plan)
        if not plan.feasible:
            raise RuntimeError(f"[AutoExecutor] 不可行: {plan.description}")

        t0 = time.time()
        if plan.num_gpus_active == 1:
            result = self._single_gpu_compute(
                lambda a, b: torch.matmul(a, b),
                A, B, return_on_cpu=return_on_cpu)
        elif plan.num_batches == 1:
            result = self._matmul_single(A, B, return_on_cpu)
        else:
            result = self._batched_matmul_variant(A, B, plan,
                lambda a, b: None, return_on_cpu)
        self._log(f"完成: {time.time() - t0:.3f}s")
        return result

    def _matmul_single(self, A, B, return_on_cpu):
        from .algorithms.matrix_ops import distributed_matmul
        if self.is_master:
            A_g, B_g = self._to_gpu(A), self._to_gpu(B)
        else:
            A_g = B_g = None
        r = distributed_matmul(A_g, B_g, self.mpi, self.distributor, self.cost_model)
        return self._to_cpu(r, return_on_cpu) if self.is_master else None

    def batch_matmul(self, A: Optional[torch.Tensor] = None,
                     B: Optional[torch.Tensor] = None,
                     return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式批量矩阵乘法 C[b] = A[b] @ B[b]

        ✅ 支持超显存自动分批（A 和 B 均沿 batch-dim 分批）
        """
        from .algorithms.matrix_ops import distributed_batch_matmul
        meta = self._sync_meta_multi(A, B)
        self._log(f"BatchMatMul A={meta['shapes'][0]} B={meta['shapes'][1]}")
        plan = self.planner.plan_operation("batch_matmul",
            [tuple(s) for s in meta['shapes']], meta['dtype'])
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.bmm(a, b),
                A, B, return_on_cpu=return_on_cpu)

        op_fn = lambda a, b: distributed_batch_matmul(a, b, self.mpi, self.distributor)
        if plan.num_batches == 1:
            return self._run_binary_op(A, B, op_fn, return_on_cpu)
        else:
            return self._batched_binary_op(A, B, plan, op_fn, return_on_cpu)

    def transpose(self, tensor: Optional[torch.Tensor] = None,
                  dim0: int = 0, dim1: int = 1,
                  return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式矩阵转置

        ✅ 支持超显存自动分批
        """
        from .algorithms.matrix_ops import distributed_transpose
        meta = self._sync_meta(tensor)
        self._log(f"Transpose shape={meta['shape']} dims=({dim0},{dim1})")
        plan = self._make_plan("transpose", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda t: t.transpose(dim0, dim1).contiguous(),
                tensor, return_on_cpu=return_on_cpu)

        op_fn = lambda t: distributed_transpose(t, self.mpi, self.distributor,
                                                  dim0=dim0, dim1=dim1)
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        else:
            return self._batched_unary_op(tensor, plan, op_fn, return_on_cpu)

    def add(self, A: Optional[torch.Tensor] = None,
            B: Optional[torch.Tensor] = None,
            return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式张量加法

        ✅ 支持超显存自动分批（A 和 B 均沿 dim-0 分批）
        """
        from .algorithms.matrix_ops import distributed_add
        meta = self._sync_meta_multi(A, B)
        self._log(f"Add shapes={meta['shapes']}")
        plan = self.planner.plan_operation("add",
            [tuple(s) for s in meta['shapes']], meta['dtype'])
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: a + b,
                A, B, return_on_cpu=return_on_cpu)

        op_fn = lambda a, b: distributed_add(a, b, self.mpi, self.distributor)
        if plan.num_batches == 1:
            return self._run_binary_op(A, B, op_fn, return_on_cpu)
        else:
            return self._batched_binary_op(A, B, plan, op_fn, return_on_cpu)

    def matmul_mixed_precision(self, A: Optional[torch.Tensor] = None,
                                B: Optional[torch.Tensor] = None,
                                return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        混合精度分布式矩阵乘法（FP16 通信 + FP32 计算）

        ✅ 支持超显存自动分批（B 只传输一次）
        """
        from .algorithms.matrix_ops import distributed_matmul_mixed_precision
        if self.is_master:
            meta = {'M': A.shape[0], 'K': A.shape[1], 'N': B.shape[1],
                    'dtype': A.dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        M, K, N, dtype = meta['M'], meta['K'], meta['N'], meta['dtype']
        self._log(f"MatMul(MixedPrecision) [{M},{K}] × [{K},{N}]")
        plan = self.planner.plan_matmul(M, K, N, dtype)
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.matmul(a.float(), b.float()),
                A, B, return_on_cpu=return_on_cpu)

        op_fn = lambda a, b: distributed_matmul_mixed_precision(
            a, b, self.mpi, self.distributor)
        if plan.num_batches == 1:
            return self._run_binary_op(A, B, op_fn, return_on_cpu)
        else:
            return self._batched_matmul_variant(A, B, plan, op_fn, return_on_cpu)

    def matmul_sparse_aware(self, A: Optional[torch.Tensor] = None,
                             B: Optional[torch.Tensor] = None,
                             sparsity_threshold: float = 0.5,
                             return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        稀疏感知自适应分布式矩阵乘法

        ✅ 支持超显存自动分批（B 只传输一次）
        """
        from .algorithms.matrix_ops import distributed_matmul_sparse_aware
        if self.is_master:
            meta = {'M': A.shape[0], 'K': A.shape[1], 'N': B.shape[1],
                    'dtype': A.dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        M, K, N, dtype = meta['M'], meta['K'], meta['N'], meta['dtype']
        self._log(f"MatMul(SparseAware) [{M},{K}] × [{K},{N}]")
        plan = self.planner.plan_matmul(M, K, N, dtype)
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.matmul(a, b),
                A, B, return_on_cpu=return_on_cpu)

        op_fn = lambda a, b: distributed_matmul_sparse_aware(
            a, b, self.mpi, self.distributor,
            sparsity_threshold=sparsity_threshold)
        if plan.num_batches == 1:
            return self._run_binary_op(A, B, op_fn, return_on_cpu)
        else:
            return self._batched_matmul_variant(A, B, plan, op_fn, return_on_cpu)

    # ================================================================
    #                        流水线 (2 个)
    # ================================================================

    def pipelined_matmul(self, A: Optional[torch.Tensor] = None,
                          B: Optional[torch.Tensor] = None,
                          num_chunks: int = 4,
                          return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        CUDA 双流流水线矩阵乘法（计算-通信重叠）

        ⚡ 数据量小时自动降级为单卡直连
        """
        # 元信息同步 + 计划
        if self.is_master:
            meta = {'M': A.shape[0], 'K': A.shape[1], 'N': B.shape[1],
                    'dtype': A.dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        M, K, N, dtype = meta['M'], meta['K'], meta['N'], meta['dtype']
        plan = self.planner.plan_matmul(M, K, N, dtype)
        self._log(f"PipelinedMatMul [{M},{K}] × [{K},{N}]")
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.matmul(a, b),
                A, B, return_on_cpu=return_on_cpu)

        pipe = self._get_pipeline()
        if self.is_master:
            A_gpu, B_gpu = self._to_gpu(A), self._to_gpu(B)
        else:
            A_gpu = B_gpu = None
        result = pipe.pipelined_matmul(A_gpu, B_gpu, num_chunks=num_chunks)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def pipelined_allreduce(self, tensor: Optional[torch.Tensor] = None,
                             num_chunks: int = 4,
                             return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        CUDA 双流流水线 AllReduce

        ⚡ 单卡时 AllReduce 为恒等操作，直接返回
        """
        meta = self._sync_meta(tensor)
        plan = self._make_plan("allreduce", meta)

        if plan.num_gpus_active == 1:
            # AllReduce on 1 GPU = identity
            if self.is_master:
                return tensor.clone() if return_on_cpu else self._to_gpu(tensor)
            return None

        pipe = self._get_pipeline()
        self._log(f"PipelinedAllReduce chunks={num_chunks}")
        if self.is_master:
            t_gpu = self._to_gpu(tensor)
        else:
            t_gpu = torch.zeros(meta['shape'], dtype=meta['dtype']).cuda(
                self.mpi.get_gpu_id())
        result = pipe.pipelined_allreduce(t_gpu, num_chunks=num_chunks)
        return self._to_cpu(result, return_on_cpu)

    # ================================================================
    #                        卷积 (2 个)
    # ================================================================

    def conv2d(self, input_tensor: Optional[torch.Tensor] = None,
               weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None,
               stride: Tuple[int, int] = (1, 1),
               padding: Tuple[int, int] = (0, 0),
               dilation: Tuple[int, int] = (1, 1),
               groups: int = 1,
               return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        显存感知的自动化分布式 2D 卷积

        ✅ 支持超显存自动分批（weight/bias 只传输一次，input 沿 batch-dim 分批）
        """
        from .algorithms.convolution import distributed_conv2d
        if self.is_master:
            meta = {'input_shape': list(input_tensor.shape),
                    'weight_shape': list(weight.shape),
                    'has_bias': bias is not None, 'dtype': input_tensor.dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        self._log(f"Conv2d input={meta['input_shape']} weight={meta['weight_shape']}")
        plan = self.planner.plan_operation(
            "conv2d", [tuple(meta['input_shape']), tuple(meta['weight_shape'])],
            meta['dtype'])
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda i, w, b: F.conv2d(i, w, b, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups),
                input_tensor, weight, bias, return_on_cpu=return_on_cpu)

        if plan.num_batches == 1:
            if self.is_master:
                gpu = self.mpi.get_gpu_id()
                i_g, w_g = input_tensor.cuda(gpu), weight.cuda(gpu)
                b_g = bias.cuda(gpu) if bias is not None else None
            else:
                i_g = w_g = b_g = None
            r = distributed_conv2d(i_g, w_g, self.mpi, self.distributor,
                                    bias=b_g, stride=stride, padding=padding,
                                    dilation=dilation, groups=groups)
            return self._to_cpu(r, return_on_cpu) if self.is_master else None
        else:
            return self._batched_conv(input_tensor, weight, bias, plan,
                                       F.conv2d, stride, padding, return_on_cpu)

    def conv3d(self, input_tensor: Optional[torch.Tensor] = None,
               weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None,
               stride: Tuple[int, int, int] = (1, 1, 1),
               padding: Tuple[int, int, int] = (0, 0, 0),
               dilation: Tuple[int, int, int] = (1, 1, 1),
               groups: int = 1,
               return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        显存感知的自动化分布式 3D 卷积

        ✅ 支持超显存自动分批（weight/bias 只传输一次，input 沿 batch-dim 分批）
        """
        from .algorithms.convolution import distributed_conv3d
        if self.is_master:
            meta = {'input_shape': list(input_tensor.shape),
                    'weight_shape': list(weight.shape),
                    'has_bias': bias is not None, 'dtype': input_tensor.dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        self._log(f"Conv3d input={meta['input_shape']} weight={meta['weight_shape']}")
        plan = self.planner.plan_operation(
            "conv3d", [tuple(meta['input_shape']), tuple(meta['weight_shape'])],
            meta['dtype'])
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda i, w, b: F.conv3d(i, w, b, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups),
                input_tensor, weight, bias, return_on_cpu=return_on_cpu)

        if plan.num_batches == 1:
            if self.is_master:
                gpu = self.mpi.get_gpu_id()
                i_g, w_g = input_tensor.cuda(gpu), weight.cuda(gpu)
                b_g = bias.cuda(gpu) if bias is not None else None
            else:
                i_g = w_g = b_g = None
            r = distributed_conv3d(i_g, w_g, self.mpi, self.distributor,
                                    bias=b_g, stride=stride, padding=padding,
                                    dilation=dilation, groups=groups)
            return self._to_cpu(r, return_on_cpu) if self.is_master else None
        else:
            return self._batched_conv(input_tensor, weight, bias, plan,
                                       F.conv3d, stride, padding, return_on_cpu)

    # ================================================================
    #                     傅里叶变换 (5 个)
    # ================================================================

    def fft(self, input_tensor: Optional[torch.Tensor] = None,
            n: Optional[int] = None, dim: int = -1,
            norm: str = "backward",
            return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式 1D FFT

        ✅ 支持超显存自动分批
        """
        from .algorithms.fft import distributed_fft
        meta = self._sync_meta(input_tensor)
        self._log(f"FFT shape={meta['shape']}")
        plan = self._make_plan("fft", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda t: torch.fft.fft(t, n=n, dim=dim, norm=norm),
                input_tensor, return_on_cpu=return_on_cpu)

        op_fn = lambda t: distributed_fft(t, self.mpi, self.distributor,
                                           n=n, dim=dim, norm=norm)
        if plan.num_batches == 1:
            return self._run_unary_op(input_tensor, op_fn, return_on_cpu)
        else:
            return self._batched_unary_op(input_tensor, plan, op_fn, return_on_cpu)

    def ifft(self, input_tensor: Optional[torch.Tensor] = None,
             n: Optional[int] = None, dim: int = -1,
             norm: str = "backward",
             return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式 1D IFFT（逆变换）

        ✅ 支持超显存自动分批
        """
        from .algorithms.fft import distributed_ifft
        meta = self._sync_meta(input_tensor)
        self._log(f"IFFT shape={meta['shape']}")
        plan = self._make_plan("ifft", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda t: torch.fft.ifft(t, n=n, dim=dim, norm=norm),
                input_tensor, return_on_cpu=return_on_cpu)

        op_fn = lambda t: distributed_ifft(t, self.mpi, self.distributor,
                                            n=n, dim=dim, norm=norm)
        if plan.num_batches == 1:
            return self._run_unary_op(input_tensor, op_fn, return_on_cpu)
        else:
            return self._batched_unary_op(input_tensor, plan, op_fn, return_on_cpu)

    def fft2d(self, input_tensor: Optional[torch.Tensor] = None,
              return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式 2D FFT

        ✅ 支持超显存自动分批
        """
        from .algorithms.fft import distributed_fft2d
        meta = self._sync_meta(input_tensor)
        self._log(f"FFT2D shape={meta['shape']}")
        plan = self._make_plan("fft2d", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda t: torch.fft.fft2(t),
                input_tensor, return_on_cpu=return_on_cpu)

        op_fn = lambda t: distributed_fft2d(t, self.mpi, self.distributor)
        if plan.num_batches == 1:
            return self._run_unary_op(input_tensor, op_fn, return_on_cpu)
        else:
            return self._batched_unary_op(input_tensor, plan, op_fn, return_on_cpu)

    def rfft(self, input_tensor: Optional[torch.Tensor] = None,
             n: Optional[int] = None, dim: int = -1,
             norm: str = "backward",
             return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式实数 FFT

        ✅ 支持超显存自动分批
        """
        from .algorithms.fft import distributed_rfft
        meta = self._sync_meta(input_tensor)
        self._log(f"RFFT shape={meta['shape']}")
        plan = self._make_plan("rfft", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda t: torch.fft.rfft(t, n=n, dim=dim, norm=norm),
                input_tensor, return_on_cpu=return_on_cpu)

        op_fn = lambda t: distributed_rfft(t, self.mpi, self.distributor,
                                            n=n, dim=dim, norm=norm)
        if plan.num_batches == 1:
            return self._run_unary_op(input_tensor, op_fn, return_on_cpu)
        else:
            return self._batched_unary_op(input_tensor, plan, op_fn, return_on_cpu)

    def fft2d_pencil(self, input_tensor: Optional[torch.Tensor] = None,
                      return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        Pencil 分解 2D FFT（适合超大网格）

        ✅ 支持超显存自动分批
        """
        from .algorithms.fft import distributed_fft2d_pencil
        meta = self._sync_meta(input_tensor)
        self._log(f"FFT2D_Pencil shape={meta['shape']}")
        plan = self._make_plan("fft2d_pencil", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda t: torch.fft.fft2(t),
                input_tensor, return_on_cpu=return_on_cpu)

        op_fn = lambda t: distributed_fft2d_pencil(t, self.mpi, self.distributor)
        if plan.num_batches == 1:
            return self._run_unary_op(input_tensor, op_fn, return_on_cpu)
        else:
            return self._batched_unary_op(input_tensor, plan, op_fn, return_on_cpu)

    # ================================================================
    #                     Einstein 求和 (3 个)
    # ================================================================

    def einsum(self, equation: str,
               *operands: Optional[torch.Tensor],
               return_on_cpu: bool = True,
               optimize: str = 'auto',
               use_opt_einsum: bool = True) -> Optional[torch.Tensor]:
        """分布式 Einstein 求和（集成 opt_einsum 最优路径）"""
        from .algorithms.einsum import distributed_einsum
        if self.is_master:
            shapes = [list(op.shape) for op in operands]
            meta = {'equation': equation, 'shapes': shapes,
                    'dtype': operands[0].dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        self._log(f"Einsum '{meta['equation']}' shapes={meta['shapes']}")

        # 单卡检查
        plan = self.planner.plan_operation(
            "einsum", [tuple(s) for s in meta['shapes']], meta['dtype'])
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda *ops: torch.einsum(equation, *ops),
                *operands, return_on_cpu=return_on_cpu)

        if self.is_master:
            gpu_ops = tuple(op.cuda(self.mpi.get_gpu_id()) for op in operands)
        else:
            gpu_ops = tuple(None for _ in meta['shapes'])
        result = distributed_einsum(equation, *gpu_ops, mpi=self.mpi,
                                     distributor=self.distributor,
                                     optimize=optimize,
                                     use_opt_einsum=use_opt_einsum)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def einsum_with_path(self, equation: str,
                          *operands: Optional[torch.Tensor],
                          path: Optional[List] = None,
                          optimize: str = 'auto',
                          return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """分布式 Einstein 求和（自定义收缩路径）"""
        from .algorithms.einsum import distributed_einsum_with_path
        if self.is_master:
            shapes = [list(op.shape) for op in operands]
            meta = {'equation': equation, 'shapes': shapes,
                    'dtype': operands[0].dtype}
        else:
            meta = None
        meta = self.mpi.broadcast(meta)
        self._log(f"EinsumWithPath '{meta['equation']}' shapes={meta['shapes']}")

        # 单卡检查
        plan = self.planner.plan_operation(
            "einsum_with_path", [tuple(s) for s in meta['shapes']], meta['dtype'])
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda *ops: torch.einsum(equation, *ops),
                *operands, return_on_cpu=return_on_cpu)

        if self.is_master:
            gpu_ops = tuple(op.cuda(self.mpi.get_gpu_id()) for op in operands)
        else:
            gpu_ops = tuple(None for _ in meta['shapes'])
        result = distributed_einsum_with_path(equation, *gpu_ops, mpi=self.mpi,
                                               distributor=self.distributor,
                                               path=path, optimize=optimize)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def tensordot(self, A: Optional[torch.Tensor] = None,
                   B: Optional[torch.Tensor] = None,
                   dims: int = 2,
                   return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """分布式张量点积"""
        from .algorithms.einsum import distributed_tensordot
        meta = self._sync_meta_multi(A, B)
        self._log(f"Tensordot A={meta['shapes'][0]} B={meta['shapes'][1]} dims={dims}")

        # 单卡检查
        plan = self.planner.plan_operation(
            "tensordot", [tuple(s) for s in meta['shapes']], meta['dtype'])
        self._log_plan(plan)

        if plan.num_gpus_active == 1:
            return self._single_gpu_compute(
                lambda a, b: torch.tensordot(a, b, dims=dims),
                A, B, return_on_cpu=return_on_cpu)

        return self._run_binary_op(
            A, B, lambda a, b: distributed_tensordot(a, b, self.mpi,
                                                      self.distributor, dims=dims),
            return_on_cpu)

    # ================================================================
    #                     归约操作 (6 个)
    # ================================================================

    def _single_gpu_reduction(self, tensor, op, dim, keepdim, return_on_cpu):
        """单卡归约的统一实现"""
        def compute_fn(t):
            if op == "sum" or op == "sum_kahan":
                return torch.sum(t) if dim is None else torch.sum(t, dim=dim, keepdim=keepdim)
            elif op == "mean" or op == "mean_kahan":
                return torch.mean(t.float()) if dim is None else torch.mean(t.float(), dim=dim, keepdim=keepdim)
            elif op == "max":
                if dim is None:
                    return t.max()
                else:
                    return torch.max(t, dim=dim, keepdim=keepdim).values
            elif op == "min":
                if dim is None:
                    return t.min()
                else:
                    return torch.min(t, dim=dim, keepdim=keepdim).values
            return torch.sum(t)
        return self._single_gpu_compute(compute_fn, tensor, return_on_cpu=return_on_cpu)

    def sum(self, tensor: Optional[torch.Tensor] = None,
            dim: Optional[int] = None, keepdim: bool = False,
            return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式求和

        ✅ 支持超显存自动分批
        """
        from .algorithms.reduction import distributed_sum
        meta = self._sync_meta(tensor)
        self._log(f"Sum shape={meta['shape']} dim={dim}")
        plan = self._make_plan("sum", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_reduction(tensor, "sum", dim, keepdim, return_on_cpu)

        op_fn = lambda t: distributed_sum(t, self.mpi, self.distributor,
                                           dim=dim, keepdim=keepdim)
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        else:
            return self._batched_reduction(tensor, plan, "sum",
                                            dim=dim, keepdim=keepdim,
                                            return_on_cpu=return_on_cpu)

    def mean(self, tensor: Optional[torch.Tensor] = None,
             dim: Optional[int] = None, keepdim: bool = False,
             return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式均值

        ✅ 支持超显存自动分批
        """
        from .algorithms.reduction import distributed_mean
        meta = self._sync_meta(tensor)
        self._log(f"Mean shape={meta['shape']} dim={dim}")
        plan = self._make_plan("mean", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_reduction(tensor, "mean", dim, keepdim, return_on_cpu)

        op_fn = lambda t: distributed_mean(t, self.mpi, self.distributor,
                                            dim=dim, keepdim=keepdim)
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        else:
            return self._batched_reduction(tensor, plan, "mean",
                                            dim=dim, keepdim=keepdim,
                                            return_on_cpu=return_on_cpu)

    def max(self, tensor: Optional[torch.Tensor] = None,
            dim: Optional[int] = None, keepdim: bool = False,
            return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式最大值

        ✅ 支持超显存自动分批（各批 max → element-wise max 合并）
        """
        from .algorithms.reduction import distributed_max
        meta = self._sync_meta(tensor)
        self._log(f"Max shape={meta['shape']} dim={dim}")
        plan = self._make_plan("max", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_reduction(tensor, "max", dim, keepdim, return_on_cpu)

        op_fn = lambda t: distributed_max(t, self.mpi, self.distributor,
                                           dim=dim, keepdim=keepdim)
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        else:
            return self._batched_reduction(tensor, plan, "max",
                                            dim=dim, keepdim=keepdim,
                                            return_on_cpu=return_on_cpu)

    def min(self, tensor: Optional[torch.Tensor] = None,
            dim: Optional[int] = None, keepdim: bool = False,
            return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式最小值

        ✅ 支持超显存自动分批（各批 min → element-wise min 合并）
        """
        from .algorithms.reduction import distributed_min
        meta = self._sync_meta(tensor)
        self._log(f"Min shape={meta['shape']} dim={dim}")
        plan = self._make_plan("min", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_reduction(tensor, "min", dim, keepdim, return_on_cpu)

        op_fn = lambda t: distributed_min(t, self.mpi, self.distributor,
                                           dim=dim, keepdim=keepdim)
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        else:
            return self._batched_reduction(tensor, plan, "min",
                                            dim=dim, keepdim=keepdim,
                                            return_on_cpu=return_on_cpu)

    def sum_kahan(self, tensor: Optional[torch.Tensor] = None,
                   dim: Optional[int] = None, keepdim: bool = False,
                   return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        Kahan 补偿求和（高精度）

        ✅ 支持超显存自动分批
        """
        from .algorithms.reduction import distributed_sum_kahan
        meta = self._sync_meta(tensor)
        self._log(f"SumKahan shape={meta['shape']} dim={dim}")
        plan = self._make_plan("sum_kahan", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_reduction(tensor, "sum_kahan", dim, keepdim, return_on_cpu)

        op_fn = lambda t: distributed_sum_kahan(t, self.mpi, self.distributor,
                                                 dim=dim, keepdim=keepdim)
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        else:
            return self._batched_reduction(tensor, plan, "sum_kahan",
                                            dim=dim, keepdim=keepdim,
                                            return_on_cpu=return_on_cpu)

    def mean_kahan(self, tensor: Optional[torch.Tensor] = None,
                    dim: Optional[int] = None, keepdim: bool = False,
                    return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        Kahan 补偿均值（高精度）

        ✅ 支持超显存自动分批
        """
        from .algorithms.reduction import distributed_mean_kahan
        meta = self._sync_meta(tensor)
        self._log(f"MeanKahan shape={meta['shape']} dim={dim}")
        plan = self._make_plan("mean_kahan", meta)

        if plan.num_gpus_active == 1:
            return self._single_gpu_reduction(tensor, "mean_kahan", dim, keepdim, return_on_cpu)

        op_fn = lambda t: distributed_mean_kahan(t, self.mpi, self.distributor,
                                                  dim=dim, keepdim=keepdim)
        if plan.num_batches == 1:
            return self._run_unary_op(tensor, op_fn, return_on_cpu)
        else:
            return self._batched_reduction(tensor, plan, "mean_kahan",
                                            dim=dim, keepdim=keepdim,
                                            return_on_cpu=return_on_cpu)

    # ================================================================
    #                   Stencil / PDE (2 个)
    # ================================================================

    def stencil_2d(self, grid: Optional[torch.Tensor] = None,
                    stencil_kernel: Optional[torch.Tensor] = None,
                    boundary: str = 'zero',
                    iterations: int = 1,
                    return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式 2D Stencil 计算（Halo Exchange）

        ⚡ 数据量小时自动降级为单卡直接迭代
        """
        from .algorithms.stencil import distributed_stencil_2d
        meta = self._sync_meta(grid)
        self._log(f"Stencil2D shape={meta['shape']} iters={iterations}")
        plan = self._make_plan("stencil_2d", meta)

        if plan.num_gpus_active == 1:
            # 单卡直接在 GPU 上迭代，无需 Halo Exchange
            if self.is_master:
                gpu_id = self.mpi.get_gpu_id()
                g = grid.float().cuda(gpu_id)
                if stencil_kernel is not None:
                    k = stencil_kernel.float().cuda(gpu_id)
                else:
                    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                     dtype=torch.float32, device=f'cuda:{gpu_id}')
                k_4d = k.unsqueeze(0).unsqueeze(0)
                pad = k.shape[-1] // 2
                for _ in range(iterations):
                    g_4d = g.unsqueeze(0).unsqueeze(0)
                    if boundary == 'periodic':
                        g_4d = F.pad(g_4d, [pad]*4, mode='circular')
                        g = F.conv2d(g_4d, k_4d).squeeze(0).squeeze(0)
                    else:
                        g = F.conv2d(g_4d, k_4d, padding=pad).squeeze(0).squeeze(0)
                result = g.to(grid.dtype)
                self.gpu.empty_cache()
                return result.cpu() if return_on_cpu else result
            return None

        g_gpu = self._to_gpu(grid) if self.is_master else None
        k_gpu = self._to_gpu(stencil_kernel) if (
            self.is_master and stencil_kernel is not None) else None
        result = distributed_stencil_2d(g_gpu, self.mpi, self.distributor,
                                         stencil_kernel=k_gpu,
                                         boundary=boundary,
                                         iterations=iterations)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    def jacobi_2d(self, grid: Optional[torch.Tensor] = None,
                   rhs: Optional[torch.Tensor] = None,
                   dx: float = 1.0,
                   boundary: str = 'zero',
                   iterations: int = 100,
                   tol: float = 1e-6,
                   return_on_cpu: bool = True) -> Optional[torch.Tensor]:
        """
        分布式 2D Jacobi 迭代（求解 Poisson 方程）

        ⚡ 数据量小时自动降级为单卡直接迭代
        """
        from .algorithms.stencil import distributed_jacobi_2d
        meta = self._sync_meta(grid)
        self._log(f"Jacobi2D shape={meta['shape']} iters={iterations} tol={tol}")
        plan = self._make_plan("jacobi_2d", meta)

        if plan.num_gpus_active == 1:
            # 单卡直接 Jacobi 迭代
            if self.is_master:
                gpu_id = self.mpi.get_gpu_id()
                g = grid.float().cuda(gpu_id)
                if rhs is not None:
                    f = rhs.float().cuda(gpu_id)
                else:
                    f = torch.zeros_like(g)
                dx2 = dx * dx
                for it in range(iterations):
                    g_old = g.clone()
                    # 5-point Jacobi: u_new[i,j] = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1] - dx^2*f) / 4
                    g_pad = F.pad(g.unsqueeze(0).unsqueeze(0), [1,1,1,1], mode='constant', value=0)
                    neighbors = (g_pad[:,:,:-2,1:-1] + g_pad[:,:,2:,1:-1] +
                                 g_pad[:,:,1:-1,:-2] + g_pad[:,:,1:-1,2:])
                    g = ((neighbors.squeeze(0).squeeze(0) - dx2 * f) / 4.0)
                    diff = torch.max(torch.abs(g - g_old)).item()
                    if diff < tol:
                        break
                result = g.to(grid.dtype)
                self.gpu.empty_cache()
                return result.cpu() if return_on_cpu else result
            return None

        g_gpu = self._to_gpu(grid) if self.is_master else None
        r_gpu = self._to_gpu(rhs) if (
            self.is_master and rhs is not None) else None
        result = distributed_jacobi_2d(g_gpu, r_gpu, self.mpi, self.distributor,
                                        dx=dx, boundary=boundary,
                                        iterations=iterations, tol=tol)
        return self._to_cpu(result, return_on_cpu) if self.is_master else None

    # ================================================================
    #                       批量操作
    # ================================================================

    def matmul_batch(self, pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                     return_on_cpu: bool = True) -> Optional[List[torch.Tensor]]:
        """批量矩阵乘法：处理多对 (A, B)"""
        if self.is_master:
            num_pairs = len(pairs)
        else:
            num_pairs = None
        num_pairs = self.mpi.broadcast(num_pairs)
        self._log(f"批量 MatMul: {num_pairs} 对矩阵")
        results = []
        for i in range(num_pairs):
            self._log(f"  === 第 {i + 1}/{num_pairs} 对 ===")
            if self.is_master:
                A_i, B_i = pairs[i]
            else:
                A_i = B_i = None
            r = self.matmul(A_i, B_i, return_on_cpu=return_on_cpu)
            if self.is_master:
                results.append(r)
        return results if self.is_master else None

    # ================================================================
    #                       信息查询
    # ================================================================

    def gpu_status(self):
        """打印当前 GPU 显存状态"""
        self.planner.print_gpu_status()

    def plan_info(self, op: str, *shapes, dtype=torch.float32) -> ExecutionPlan:
        """查看执行计划（不执行）"""
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


def auto_compute(op: str, *args, **kwargs) -> Optional[torch.Tensor]:
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

    用法:
        C = auto_compute("matmul", A_cpu, B_cpu)
        Y = auto_compute("fft", X_cpu)
        S = auto_compute("sum", X_cpu, dim=0)
    """
    global _global_executor

    verbose = kwargs.pop("verbose", True)
    max_per_gpu_gb = kwargs.pop("max_per_gpu_gb", None)
    return_on_cpu = kwargs.pop("return_on_cpu", True)

    if _global_executor is None:
        _global_executor = AutoExecutor(verbose=verbose,
                                         max_per_gpu_gb=max_per_gpu_gb)
    ex = _global_executor

    dispatch = {
        # 矩阵运算 (6)
        "matmul":       lambda: ex.matmul(*args, return_on_cpu=return_on_cpu, **kwargs),
        "batch_matmul": lambda: ex.batch_matmul(*args, return_on_cpu=return_on_cpu, **kwargs),
        "transpose":    lambda: ex.transpose(*args, return_on_cpu=return_on_cpu, **kwargs),
        "add":          lambda: ex.add(*args, return_on_cpu=return_on_cpu, **kwargs),
        "matmul_mixed_precision": lambda: ex.matmul_mixed_precision(
            *args, return_on_cpu=return_on_cpu, **kwargs),
        "matmul_sparse_aware": lambda: ex.matmul_sparse_aware(
            *args, return_on_cpu=return_on_cpu, **kwargs),
        # 流水线 (2)
        "pipelined_matmul":    lambda: ex.pipelined_matmul(
            *args, return_on_cpu=return_on_cpu, **kwargs),
        "pipelined_allreduce": lambda: ex.pipelined_allreduce(
            *args, return_on_cpu=return_on_cpu, **kwargs),
        # 卷积 (2)
        "conv2d": lambda: ex.conv2d(*args, return_on_cpu=return_on_cpu, **kwargs),
        "conv3d": lambda: ex.conv3d(*args, return_on_cpu=return_on_cpu, **kwargs),
        # FFT (5)
        "fft":          lambda: ex.fft(*args, return_on_cpu=return_on_cpu, **kwargs),
        "ifft":         lambda: ex.ifft(*args, return_on_cpu=return_on_cpu, **kwargs),
        "fft2d":        lambda: ex.fft2d(*args, return_on_cpu=return_on_cpu, **kwargs),
        "rfft":         lambda: ex.rfft(*args, return_on_cpu=return_on_cpu, **kwargs),
        "fft2d_pencil": lambda: ex.fft2d_pencil(*args, return_on_cpu=return_on_cpu, **kwargs),
        # Einsum (3)
        "einsum":    lambda: ex.einsum(args[0], *args[1:],
                                        return_on_cpu=return_on_cpu, **kwargs),
        "einsum_with_path": lambda: ex.einsum_with_path(args[0], *args[1:],
                                        return_on_cpu=return_on_cpu, **kwargs),
        "tensordot": lambda: ex.tensordot(*args, return_on_cpu=return_on_cpu, **kwargs),
        # 归约 (6)
        "sum":        lambda: ex.sum(*args, return_on_cpu=return_on_cpu, **kwargs),
        "mean":       lambda: ex.mean(*args, return_on_cpu=return_on_cpu, **kwargs),
        "max":        lambda: ex.max(*args, return_on_cpu=return_on_cpu, **kwargs),
        "min":        lambda: ex.min(*args, return_on_cpu=return_on_cpu, **kwargs),
        "sum_kahan":  lambda: ex.sum_kahan(*args, return_on_cpu=return_on_cpu, **kwargs),
        "mean_kahan": lambda: ex.mean_kahan(*args, return_on_cpu=return_on_cpu, **kwargs),
        # Stencil / PDE (2)
        "stencil_2d": lambda: ex.stencil_2d(*args, return_on_cpu=return_on_cpu, **kwargs),
        "jacobi_2d":  lambda: ex.jacobi_2d(*args, return_on_cpu=return_on_cpu, **kwargs),
    }

    if op not in dispatch:
        supported = ", ".join(sorted(dispatch.keys()))
        raise ValueError(f"不支持的操作: '{op}'. 可选:\n  {supported}")

    return dispatch[op]()
