"""
MPI 通信管理器 (MPI Communication Manager)

负责 MPI 环境的初始化、进程间通信和协调。
确保所有 MPI 集合操作被所有进程正确调用。
包含错误处理和容错机制。

性能优化：
- Pinned memory 缓存池：避免每次通信都重新分配 pinned memory
- CUDA 事件同步：精确同步而非全局 synchronize
- 预分配接收缓冲区：allreduce_tensor 复用 numpy 缓冲区
- 避免冗余 contiguous() 调用：先检查再拷贝
"""

from __future__ import annotations

import logging
import traceback
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from mpi4py import MPI

# ── 模块级 logger ──────────────────────────────────────────────
_logger = logging.getLogger("distributed_gpu.mpi")

# ── dtype 映射表（模块级常量，避免每次调用重建） ────────────────
_TORCH_TO_NUMPY_DTYPE: Dict[torch.dtype, Type[np.generic]] = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bfloat16: np.float32,  # bfloat16 无 numpy 对应，fallback fp32
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}

# ── Pinned Memory 缓冲区池大小限制 ──────────────────────────────
_PINNED_POOL_MAX_BYTES: int = 512 * 1024 * 1024  # 512 MB 总池上限
_PINNED_POOL_MAX_ENTRIES: int = 32                # 最多缓存 32 个缓冲区


# ══════════════════════════════════════════════════════════════════
#  异常类
# ══════════════════════════════════════════════════════════════════

class MPIError(RuntimeError):
    """MPI 操作异常（附带 rank 信息）。"""

    def __init__(
        self,
        msg: str,
        rank: int = -1,
        original: Optional[Exception] = None,
    ) -> None:
        self.rank: int = rank
        self.original: Optional[Exception] = original
        super().__init__(f"[Rank {rank}] {msg}")


# ══════════════════════════════════════════════════════════════════
#  MPI 管理器
# ══════════════════════════════════════════════════════════════════

class MPIManager:
    """
    MPI 通信管理器

    职责：
    - 初始化 MPI 环境并绑定 GPU 设备
    - 提供带错误处理的集合通信操作 (broadcast / scatter / gather / allreduce …)
    - 提供高性能 NumPy ↔ GPU 张量传输辅助方法

    性能特性：
    - Pinned memory 缓冲区池：复用 pinned memory 避免重复分配开销
    - 智能 GPU→CPU 传输：避免冗余的 contiguous() 和 numpy 转换
    - 预分配接收缓冲区：allreduce 等操作复用 numpy 缓冲区
    """

    # ── 初始化 ──────────────────────────────────────────────────

    def __init__(self) -> None:
        """初始化 MPI 环境并绑定 GPU 设备。"""
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()
        self.is_master: bool = (self.rank == 0)

        # 注意: 不设置 ERRORS_RETURN，因为 mpi4py 的 pickle 级通信
        # (comm.scatter/bcast 小写) 在 ERRORS_RETURN 模式下可能出现
        # MPI_ERR_ARG，与部分 OpenMPI 版本不兼容。
        # 错误处理已由 _safe_call 的 try/except 覆盖。

        # 设置 GPU 设备（每个进程绑定一个 GPU）
        if torch.cuda.is_available():
            self.gpu_count: int = torch.cuda.device_count()
            self.gpu_id: int = self.rank % self.gpu_count
            torch.cuda.set_device(self.gpu_id)
        else:
            self.gpu_count = 0
            self.gpu_id = -1

        # ── Pinned memory 缓冲区池 ──
        # 按 (nbytes, dtype) 索引，存储可复用的 pinned numpy 数组
        # 避免每次通信都执行昂贵的 cudaHostAlloc / pin_memory
        self._pinned_pool: Dict[Tuple[int, Type[np.generic]], List[np.ndarray]] = {}
        self._pinned_pool_total_bytes: int = 0
        self._pinned_pool_lock: threading.Lock = threading.Lock()

        # ── 预分配的 CUDA 传输流 ──
        # 用于异步 CPU→GPU 传输，避免阻塞默认计算流
        if torch.cuda.is_available() and self.gpu_id >= 0:
            self._transfer_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(
                device=self.gpu_id
            )
        else:
            self._transfer_stream = None

        if self.is_master:
            _logger.info("MPI 环境初始化: %d 个进程, %d 个 GPU", self.size, self.gpu_count)
            print(f"MPI环境初始化: {self.size} 个进程")
            print(f"可用GPU数量: {self.gpu_count}")

    # ── 内部：安全调用包装器 ────────────────────────────────────

    def _safe_call(self, func_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        带错误处理的 MPI 操作包装器。

        如果 MPI 操作失败，会记录日志并抛出 :class:`MPIError`，
        避免无信息的段错误或死锁。
        """
        try:
            return fn(*args, **kwargs)
        except MPI.Exception as exc:
            msg = f"MPI 操作 '{func_name}' 失败: {exc}"
            _logger.error(msg)
            raise MPIError(msg, rank=self.rank, original=exc) from exc
        except Exception as exc:
            msg = f"操作 '{func_name}' 异常: {exc}\n{traceback.format_exc()}"
            _logger.error(msg)
            raise MPIError(msg, rank=self.rank, original=exc) from exc

    # ── 状态查询 ────────────────────────────────────────────────

    def check_health(self) -> bool:
        """检查所有进程是否存活（轻量级 barrier 心跳）。"""
        try:
            self.comm.Barrier()
            return True
        except Exception:
            return False

    def get_rank(self) -> int:
        """获取当前进程 rank。"""
        return self.rank

    def get_size(self) -> int:
        """获取总进程数。"""
        return self.size

    def is_master_process(self) -> bool:
        """判断是否是主进程 (rank == 0)。"""
        return self.is_master

    def get_gpu_id(self) -> int:
        """获取当前进程绑定的 GPU ID。"""
        return self.gpu_id

    # ── 同步 ────────────────────────────────────────────────────

    def barrier(self) -> None:
        """同步所有进程 (MPI_Barrier)。"""
        self.comm.Barrier()

    def synchronize(self) -> None:
        """同步所有 GPU 操作 + MPI barrier。"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.barrier()

    # ══════════════════════════════════════════════════════════════
    #  集合通信操作
    #  注意：所有进程都必须调用这些函数
    # ══════════════════════════════════════════════════════════════

    def broadcast(self, data: Any, root: int = 0) -> Any:
        """
        广播 Python 对象（pickle 序列化）。

        重要：所有进程都必须调用此函数！

        Args:
            data: 要广播的数据（仅 *root* 进程的数据有效）
            root: 源进程 rank

        Returns:
            广播后的数据（所有进程都得到相同数据）
        """
        return self._safe_call("bcast", self.comm.bcast, data, root=root)

    def broadcast_tensor(self, tensor: Optional[torch.Tensor], root: int = 0) -> torch.Tensor:
        """
        广播 PyTorch 张量（使用高性能 MPI_Bcast）。

        Args:
            tensor: 要广播的张量（仅 *root* 进程需要提供有效张量）
            root: 源进程 rank

        Returns:
            广播后的 GPU 张量

        Raises:
            MPIError: 如果 root 进程的 tensor 为 None 或通信失败
        """
        # 先广播张量元信息
        if self.rank == root:
            if tensor is None:
                raise MPIError(f"root={root} 进程的 tensor 不能为 None", rank=self.rank)
            meta = {"shape": tensor.shape, "dtype": tensor.dtype}
            data = self._to_pinned_numpy(tensor)
        else:
            meta = None
            data = None

        meta = self._safe_call("bcast(meta)", self.comm.bcast, meta, root=root)

        if self.rank != root:
            data = np.empty(meta["shape"], dtype=torch_to_numpy_dtype(meta["dtype"]))

        self._safe_call("Bcast", self.comm.Bcast, data, root=root)
        return self._from_numpy_to_gpu(data)

    def scatter(self, data_list: Optional[List[Any]], root: int = 0) -> Any:
        """
        将数据列表从 root 进程分散到各个进程。

        重要：所有进程都必须调用此函数！

        Args:
            data_list: 数据列表（仅 root 进程需要提供，长度应等于进程数）
            root: 源进程 rank

        Returns:
            当前进程分配到的数据
        """
        return self._safe_call("scatter", self.comm.scatter, data_list, root=root)

    def scatter_tensor(
        self,
        tensor: Optional[torch.Tensor],
        dim: int = 0,
        root: int = 0,
    ) -> torch.Tensor:
        """
        将张量沿指定维度分散到各个进程。

        Args:
            tensor: 要分散的张量（仅 root 进程需要提供）
            dim: 分割维度
            root: 源进程 rank

        Returns:
            当前进程分配到的张量块

        Raises:
            MPIError: 如果通信失败或维度大小不足
        """
        # root 进程检查并准备数据；检查结果广播给所有进程避免死锁
        if self.rank == root:
            error_msg: Optional[str] = None
            if tensor is None:
                error_msg = f"root={root} 进程的 tensor 不能为 None"
            elif tensor.shape[dim] < self.size:
                error_msg = (
                    f"张量维度 {dim} 大小 ({tensor.shape[dim]}) "
                    f"小于进程数 ({self.size})，无法分割"
                )
        else:
            error_msg = None

        # 所有进程同步检查结果
        error_msg = self.comm.bcast(error_msg, root=root)
        if error_msg is not None:
            raise MPIError(error_msg, rank=self.rank)

        if self.rank == root:
            assert tensor is not None  # 已由上面的检查保证
            chunks = _split_to_numpy_chunks(tensor, self.size, dim, self._to_pinned_numpy)
            meta = {"dtype": tensor.dtype, "dim": dim}
        else:
            chunks = None
            meta = None

        meta = self._safe_call("bcast(meta)", self.comm.bcast, meta, root=root)
        local_data: np.ndarray = self._safe_call("scatter", self.comm.scatter, chunks, root=root)
        return self._from_numpy_to_gpu(local_data)

    def gather(self, data: Any, root: int = 0) -> Optional[List[Any]]:
        """
        从各个进程收集数据到 root 进程。

        重要：所有进程都必须调用此函数！

        Args:
            data: 当前进程的数据
            root: 目标进程 rank

        Returns:
            收集的数据列表（仅 root 进程返回有效数据）
        """
        return self._safe_call("gather", self.comm.gather, data, root=root)

    def gather_tensor(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
        root: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        从各个进程收集张量并在 root 进程拼接。

        Args:
            tensor: 当前进程的张量
            dim: 拼接维度
            root: 目标进程 rank

        Returns:
            拼接后的张量（仅 root 进程返回有效数据）

        Raises:
            MPIError: 如果通信失败
        """
        if tensor is None:
            raise MPIError("gather_tensor 的 tensor 参数不能为 None", rank=self.rank)

        local_data = self._to_pinned_numpy(tensor)
        gathered: List[np.ndarray] = self._safe_call(
            "gather(tensor)", self.comm.gather, local_data, root=root,
        )

        if self.rank == root:
            return self._from_numpy_to_gpu(np.concatenate(gathered, axis=dim))
        return None

    def allgather(self, data: Any) -> List[Any]:
        """
        从所有进程收集数据并广播到所有进程。

        重要：所有进程都必须调用此函数！
        """
        return self._safe_call("allgather", self.comm.allgather, data)

    def allgather_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        从所有进程收集张量并拼接，所有进程都得到完整结果。

        Args:
            tensor: 当前进程的张量
            dim: 拼接维度

        Returns:
            拼接后的完整张量
        """
        local_data = self._to_pinned_numpy(tensor)
        gathered: List[np.ndarray] = self._safe_call(
            "allgather(tensor)", self.comm.allgather, local_data,
        )
        return self._from_numpy_to_gpu(np.concatenate(gathered, axis=dim))

    def reduce(self, data: Any, op: Any = MPI.SUM, root: int = 0) -> Any:
        """
        归约操作。

        重要：所有进程都必须调用此函数！
        """
        return self._safe_call("reduce", self.comm.reduce, data, op=op, root=root)

    def allreduce(self, data: Any, op: Any = MPI.SUM) -> Any:
        """
        全归约操作（所有进程都得到结果）。

        重要：所有进程都必须调用此函数！
        """
        return self._safe_call("allreduce", self.comm.allreduce, data, op=op)

    def allreduce_tensor(self, tensor: torch.Tensor, op: Any = MPI.SUM) -> torch.Tensor:
        """
        张量全归约操作。

        Args:
            tensor: 当前进程的张量
            op: 归约操作（默认 MPI.SUM）

        Returns:
            归约后的 GPU 张量

        Raises:
            MPIError: 如果 tensor 为 None 或通信失败
        """
        if tensor is None:
            raise MPIError("allreduce_tensor 的 tensor 参数不能为 None", rank=self.rank)

        data = self._to_pinned_numpy(tensor)
        # 使用池化缓冲区作为接收缓冲区，避免每次 allreduce 都新建 numpy 数组
        result = self._acquire_pinned_buffer(data.shape, data.dtype.type)
        self._safe_call("Allreduce", self.comm.Allreduce, data, result, op=op)
        gpu_result = self._from_numpy_to_gpu(result)
        # 归还两个 pinned 缓冲区到池中
        self._release_pinned_buffer(data)
        self._release_pinned_buffer(result)
        return gpu_result

    # ══════════════════════════════════════════════════════════════
    #  点对点通信
    # ══════════════════════════════════════════════════════════════

    def send(self, data: Any, dest: int, tag: int = 0) -> None:
        """发送数据到指定进程。"""
        self._safe_call("send", self.comm.send, data, dest=dest, tag=tag)

    def recv(self, source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG) -> Any:
        """从指定进程接收数据。"""
        return self._safe_call("recv", self.comm.recv, source=source, tag=tag)

    def sendrecv(
        self,
        sendobj: Any,
        dest: int,
        source: int,
        sendtag: int = 0,
        recvtag: int = 0,
    ) -> Any:
        """同时发送和接收。"""
        return self._safe_call(
            "sendrecv", self.comm.sendrecv, sendobj,
            dest=dest, source=source,
            sendtag=sendtag, recvtag=recvtag,
        )

    # ══════════════════════════════════════════════════════════════
    #  高性能传输辅助
    # ══════════════════════════════════════════════════════════════

    # ── Pinned Memory 缓冲区池管理 ──

    def _acquire_pinned_buffer(
        self, shape: Tuple[int, ...], dtype: Type[np.generic],
    ) -> np.ndarray:
        """
        从缓冲区池获取一个 pinned numpy 数组。

        如果池中有匹配 (nbytes, dtype) 的缓冲区，直接复用；
        否则新建一个 pinned 缓冲区。池总大小受 _PINNED_POOL_MAX_BYTES 限制。
        """
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        key = (nbytes, dtype)

        with self._pinned_pool_lock:
            pool_list = self._pinned_pool.get(key)
            if pool_list:
                buf = pool_list.pop()
                self._pinned_pool_total_bytes -= nbytes
                # reshape 到目标形状（底层内存不变）
                return buf.reshape(shape)

        # 池中没有可用缓冲区 → 新建
        # 通过 PyTorch 的 pin_memory() 获得 CUDA host-pinned 内存
        t = torch.empty(shape, dtype=torch.from_numpy(np.empty(0, dtype=dtype)).dtype)
        t_pinned = t.pin_memory()
        return t_pinned.numpy()

    def _release_pinned_buffer(self, buf: np.ndarray) -> None:
        """
        将 pinned numpy 数组归还缓冲区池。

        如果池已满（总字节数超限或条目数超限），直接丢弃让 GC 回收。
        """
        nbytes = buf.nbytes
        dtype = buf.dtype.type
        key = (nbytes, dtype)

        with self._pinned_pool_lock:
            # 检查池大小限制
            if (self._pinned_pool_total_bytes + nbytes > _PINNED_POOL_MAX_BYTES
                    or sum(len(v) for v in self._pinned_pool.values()) >= _PINNED_POOL_MAX_ENTRIES):
                return  # 丢弃，让 GC 回收

            pool_list = self._pinned_pool.setdefault(key, [])
            pool_list.append(buf.ravel())  # 存为 1D 以便后续 reshape
            self._pinned_pool_total_bytes += nbytes

    def clear_pinned_pool(self) -> None:
        """清空 pinned memory 缓冲区池，释放所有缓存的 host 内存。"""
        with self._pinned_pool_lock:
            self._pinned_pool.clear()
            self._pinned_pool_total_bytes = 0
        _logger.debug("Pinned memory pool cleared")

    def _to_pinned_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        将 GPU/CPU 张量高效转为连续 NumPy 数组（使用 pinned memory 池）。

        优化点：
        1. 避免冗余 contiguous() — 先检查 is_contiguous()
        2. 使用 pinned buffer 池 — 避免每次重新分配 host 内存
        3. GPU 直接拷贝到 pinned buffer — 利用 DMA 快速传输
        """
        t = tensor.detach()

        # 仅在必要时调用 contiguous()（避免冗余拷贝）
        if not t.is_contiguous():
            t = t.contiguous()

        if t.is_cuda:
            # GPU → pinned numpy：使用池化缓冲区 + 异步拷贝
            np_dtype = _TORCH_TO_NUMPY_DTYPE.get(t.dtype, np.float32)
            buf = self._acquire_pinned_buffer(t.shape, np_dtype)
            # 创建一个指向 pinned buffer 的 CPU tensor（零拷贝）
            t_pinned = torch.from_numpy(buf)
            # GPU → pinned CPU（DMA 直传）
            t_pinned.copy_(t, non_blocking=False)
            return buf
        else:
            # CPU tensor → numpy（零拷贝如果已经 contiguous）
            return np.ascontiguousarray(t.numpy())

    def _from_numpy_to_gpu(self, arr: np.ndarray) -> torch.Tensor:
        """
        将 NumPy 数组高效传回 GPU（使用 pinned memory + 异步传输）。

        优化点：
        1. 检查数组是否已在 pinned memory 中（从池获取的缓冲区已 pinned）
        2. 使用专用传输流 — 不阻塞默认计算流
        3. 非阻塞传输 — 利用 DMA 异步拷贝
        """
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        t = torch.from_numpy(arr)

        if torch.cuda.is_available() and self.gpu_id >= 0:
            # 尝试使用专用传输流进行异步传输
            if self._transfer_stream is not None:
                with torch.cuda.stream(self._transfer_stream):
                    # 如果底层内存已经是 pinned，pin_memory() 是空操作
                    result = t.pin_memory().cuda(self.gpu_id, non_blocking=True)
                # 在默认流上等待传输完成
                torch.cuda.current_stream(self.gpu_id).wait_stream(self._transfer_stream)
                return result
            else:
                return t.pin_memory().cuda(self.gpu_id, non_blocking=True)
        return t

    # ══════════════════════════════════════════════════════════════
    #  辅助方法
    # ══════════════════════════════════════════════════════════════

    def get_neighbors(self) -> Tuple[int, int]:
        """获取环形拓扑中的相邻进程 (prev_rank, next_rank)。"""
        prev_rank = (self.rank - 1) % self.size
        next_rank = (self.rank + 1) % self.size
        return prev_rank, next_rank

    def print_master(self, message: str) -> None:
        """仅在主进程打印消息。"""
        if self.is_master:
            print(message)

    def print_all(self, message: str) -> None:
        """所有进程打印消息（带 rank 标识）。"""
        print(f"[Rank {self.rank}] {message}")

    # 向后兼容别名
    def _torch_to_numpy_dtype(self, torch_dtype: torch.dtype) -> Type[np.generic]:
        """PyTorch dtype → NumPy dtype（向后兼容别名）。"""
        return torch_to_numpy_dtype(torch_dtype)


# ══════════════════════════════════════════════════════════════════
#  模块级工具函数
# ══════════════════════════════════════════════════════════════════

def torch_to_numpy_dtype(torch_dtype: torch.dtype) -> Type[np.generic]:
    """将 PyTorch dtype 转换为对应的 NumPy dtype。"""
    return _TORCH_TO_NUMPY_DTYPE.get(torch_dtype, np.float32)


def _split_to_numpy_chunks(
    tensor: torch.Tensor,
    num_chunks: int,
    dim: int,
    to_numpy_fn: Callable[[torch.Tensor], np.ndarray],
) -> List[np.ndarray]:
    """
    将张量沿 *dim* 均匀分割为 *num_chunks* 个 NumPy 数组。

    余数部分会分配给前 ``remainder`` 个 chunk（每个多 1 行/列）。
    """
    total = tensor.shape[dim]
    chunk_size = total // num_chunks
    remainder = total % num_chunks

    chunks: List[np.ndarray] = []
    start = 0
    for i in range(num_chunks):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(to_numpy_fn(tensor.narrow(dim, start, size)))
        start += size
    return chunks
