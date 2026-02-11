"""
MPI通信管理器

负责MPI环境的初始化、进程间通信和协调。
确保所有MPI集合操作被所有进程正确调用。
包含错误处理和容错机制。
"""

import logging
import traceback
import numpy as np
import torch
from mpi4py import MPI
from typing import Any, Optional, List

# 模块级 logger
_logger = logging.getLogger("distributed_gpu.mpi")


class MPIError(RuntimeError):
    """MPI 操作异常（附带 rank 信息）"""
    def __init__(self, msg: str, rank: int = -1, original: Optional[Exception] = None):
        self.rank = rank
        self.original = original
        super().__init__(f"[Rank {rank}] {msg}")


class MPIManager:
    """MPI通信管理器"""
    
    def __init__(self):
        """初始化MPI环境"""
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_master = (self.rank == 0)
        
        # 设置错误处理器
        self.comm.Set_errhandler(MPI.ERRORS_RETURN)
        
        # 设置GPU设备（每个进程绑定一个GPU）
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            self.gpu_id = self.rank % self.gpu_count
            torch.cuda.set_device(self.gpu_id)
        else:
            self.gpu_count = 0
            self.gpu_id = -1
        
        if self.is_master:
            print(f"MPI环境初始化: {self.size} 个进程")
            print(f"可用GPU数量: {self.gpu_count}")
    
    def _safe_call(self, func_name: str, fn, *args, **kwargs):
        """
        带错误处理的 MPI 操作包装器
        
        如果 MPI 操作失败，会记录日志并抛出 MPIError，
        避免无信息的段错误或死锁。
        """
        try:
            return fn(*args, **kwargs)
        except MPI.Exception as e:
            msg = f"MPI 操作 '{func_name}' 失败: {e}"
            _logger.error(msg)
            raise MPIError(msg, rank=self.rank, original=e) from e
        except Exception as e:
            msg = f"操作 '{func_name}' 异常: {e}\n{traceback.format_exc()}"
            _logger.error(msg)
            raise MPIError(msg, rank=self.rank, original=e) from e
    
    def check_health(self) -> bool:
        """
        检查所有进程是否存活（轻量级心跳）
        
        Returns:
            True 如果所有进程都响应了 barrier
        """
        try:
            self.comm.Barrier()
            return True
        except Exception:
            return False
    
    def get_rank(self) -> int:
        """获取当前进程rank"""
        return self.rank
    
    def get_size(self) -> int:
        """获取总进程数"""
        return self.size
    
    def is_master_process(self) -> bool:
        """判断是否是主进程"""
        return self.is_master
    
    def get_gpu_id(self) -> int:
        """获取当前进程绑定的GPU ID"""
        return self.gpu_id
    
    def barrier(self):
        """同步所有进程"""
        self.comm.Barrier()
    
    def synchronize(self):
        """同步所有GPU操作和进程"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.barrier()
    
    # ==================== 集合通信操作 ====================
    # 注意：所有进程都必须调用这些函数
    
    def broadcast(self, data: Any, root: int = 0) -> Any:
        """
        广播数据从root进程到所有进程
        
        重要：所有进程都必须调用此函数！
        
        Args:
            data: 要广播的数据（仅root进程的数据有效）
            root: 源进程rank
        
        Returns:
            广播后的数据（所有进程都得到相同数据）
        """
        return self.comm.bcast(data, root=root)
    
    def broadcast_tensor(self, tensor: Optional[torch.Tensor], root: int = 0) -> torch.Tensor:
        """
        广播PyTorch张量
        
        Args:
            tensor: 要广播的张量（仅root进程需要提供有效张量）
            root: 源进程rank
        
        Returns:
            广播后的张量
        
        Raises:
            MPIError: 如果通信失败
        """
        # 先广播张量元信息
        if self.rank == root:
            if tensor is None:
                raise MPIError(f"root={root} 进程的 tensor 不能为 None", rank=self.rank)
            meta = {
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': 'cpu'
            }
            data = tensor.cpu().numpy()
        else:
            meta = None
            data = None
        
        meta = self._safe_call('bcast(meta)', self.comm.bcast, meta, root=root)
        
        if self.rank != root:
            data = np.empty(meta['shape'], dtype=self._torch_to_numpy_dtype(meta['dtype']))
        
        self._safe_call('Bcast', self.comm.Bcast, data, root=root)
        
        result = torch.from_numpy(data.copy())
        if torch.cuda.is_available():
            result = result.cuda(self.gpu_id)
        
        return result
    
    def scatter(self, data_list: Optional[List], root: int = 0) -> Any:
        """
        将数据列表从root进程分散到各个进程
        
        重要：所有进程都必须调用此函数！
        
        Args:
            data_list: 数据列表（仅root进程需要提供，长度应等于进程数）
            root: 源进程rank
        
        Returns:
            当前进程分配到的数据
        """
        return self.comm.scatter(data_list, root=root)
    
    def scatter_tensor(self, tensor: Optional[torch.Tensor], dim: int = 0, root: int = 0) -> torch.Tensor:
        """
        将张量沿指定维度分散到各个进程
        
        Args:
            tensor: 要分散的张量（仅root进程需要提供）
            dim: 分割维度
            root: 源进程rank
        
        Returns:
            当前进程分配到的张量块
        
        Raises:
            MPIError: 如果通信失败
            ValueError: 如果分割维度大小不足
        """
        # root 进程检查并准备数据；检查结果广播给所有进程避免死锁
        if self.rank == root:
            error_msg = None
            if tensor is None:
                error_msg = f"root={root} 进程的 tensor 不能为 None"
            elif tensor.shape[dim] < self.size:
                error_msg = (f"张量维度 {dim} 大小 ({tensor.shape[dim]}) "
                             f"小于进程数 ({self.size})，无法分割")
        else:
            error_msg = None
        
        # 所有进程同步检查结果
        error_msg = self.comm.bcast(error_msg, root=root)
        if error_msg is not None:
            raise MPIError(error_msg, rank=self.rank)
        
        if self.rank == root:
            shape = list(tensor.shape)
            chunk_size = shape[dim] // self.size
            remainder = shape[dim] % self.size
            dtype = tensor.dtype
            
            chunks = []
            start = 0
            for i in range(self.size):
                size = chunk_size + (1 if i < remainder else 0)
                chunk = tensor.narrow(dim, start, size).cpu().numpy()
                chunks.append(chunk)
                start += size
            
            meta = {'dtype': dtype, 'dim': dim}
        else:
            chunks = None
            meta = None
        
        meta = self._safe_call('bcast(meta)', self.comm.bcast, meta, root=root)
        local_data = self._safe_call('scatter', self.comm.scatter, chunks, root=root)
        
        result = torch.from_numpy(local_data.copy())
        if torch.cuda.is_available():
            result = result.cuda(self.gpu_id)
        
        return result
    
    def gather(self, data: Any, root: int = 0) -> Optional[List]:
        """
        从各个进程收集数据到root进程
        
        重要：所有进程都必须调用此函数！
        
        Args:
            data: 当前进程的数据
            root: 目标进程rank
        
        Returns:
            收集的数据列表（仅root进程返回有效数据）
        """
        return self.comm.gather(data, root=root)
    
    def gather_tensor(self, tensor: torch.Tensor, dim: int = 0, root: int = 0) -> Optional[torch.Tensor]:
        """
        从各个进程收集张量并在root进程拼接
        
        Args:
            tensor: 当前进程的张量
            dim: 拼接维度
            root: 目标进程rank
        
        Returns:
            拼接后的张量（仅root进程返回有效数据）
        
        Raises:
            MPIError: 如果通信失败
        """
        if tensor is None:
            raise MPIError("gather_tensor 的 tensor 参数不能为 None", rank=self.rank)
        
        local_data = tensor.cpu().numpy()
        gathered = self._safe_call('gather', self.comm.gather, local_data, root=root)
        
        if self.rank == root:
            result = torch.from_numpy(np.concatenate(gathered, axis=dim))
            if torch.cuda.is_available():
                result = result.cuda(self.gpu_id)
            return result
        return None
    
    def allgather(self, data: Any) -> List:
        """
        从所有进程收集数据并广播到所有进程
        
        重要：所有进程都必须调用此函数！
        
        Args:
            data: 当前进程的数据
        
        Returns:
            所有进程数据的列表
        """
        return self.comm.allgather(data)
    
    def allgather_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        从所有进程收集张量并拼接，所有进程都得到完整结果
        
        Args:
            tensor: 当前进程的张量
            dim: 拼接维度
        
        Returns:
            拼接后的完整张量
        """
        local_data = tensor.cpu().numpy()
        gathered = self.comm.allgather(local_data)
        
        result = torch.from_numpy(np.concatenate(gathered, axis=dim))
        if torch.cuda.is_available():
            result = result.cuda(self.gpu_id)
        
        return result
    
    def reduce(self, data: Any, op=MPI.SUM, root: int = 0) -> Any:
        """
        归约操作
        
        重要：所有进程都必须调用此函数！
        """
        return self.comm.reduce(data, op=op, root=root)
    
    def allreduce(self, data: Any, op=MPI.SUM) -> Any:
        """
        全归约操作（所有进程都得到结果）
        
        重要：所有进程都必须调用此函数！
        """
        return self.comm.allreduce(data, op=op)
    
    def allreduce_tensor(self, tensor: torch.Tensor, op=MPI.SUM) -> torch.Tensor:
        """
        张量全归约操作
        
        Args:
            tensor: 当前进程的张量
            op: 归约操作（默认求和）
        
        Returns:
            归约后的张量
        
        Raises:
            MPIError: 如果通信失败
        """
        if tensor is None:
            raise MPIError("allreduce_tensor 的 tensor 参数不能为 None", rank=self.rank)
        
        data = tensor.cpu().numpy().copy()
        result = np.empty_like(data)
        self._safe_call('Allreduce', self.comm.Allreduce, data, result, op=op)
        
        result_tensor = torch.from_numpy(result)
        if torch.cuda.is_available():
            result_tensor = result_tensor.cuda(self.gpu_id)
        
        return result_tensor
    
    # ==================== 点对点通信 ====================
    
    def send(self, data: Any, dest: int, tag: int = 0):
        """发送数据到指定进程"""
        self.comm.send(data, dest=dest, tag=tag)
    
    def recv(self, source: int = MPI.ANY_SOURCE, tag: int = MPI.ANY_TAG) -> Any:
        """从指定进程接收数据"""
        return self.comm.recv(source=source, tag=tag)
    
    def sendrecv(self, sendobj: Any, dest: int, source: int, 
                 sendtag: int = 0, recvtag: int = 0) -> Any:
        """同时发送和接收"""
        return self.comm.sendrecv(sendobj, dest=dest, source=source,
                                   sendtag=sendtag, recvtag=recvtag)
    
    # ==================== 辅助函数 ====================
    
    def get_neighbors(self):
        """获取环形拓扑中的相邻进程"""
        prev_rank = (self.rank - 1) % self.size
        next_rank = (self.rank + 1) % self.size
        return prev_rank, next_rank
    
    def print_master(self, message: str):
        """仅在主进程打印消息"""
        if self.is_master:
            print(message)
    
    def print_all(self, message: str):
        """所有进程打印消息（带rank标识）"""
        print(f"[Rank {self.rank}] {message}")
    
    def _torch_to_numpy_dtype(self, torch_dtype):
        """PyTorch dtype 转 NumPy dtype"""
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.complex64: np.complex64,
            torch.complex128: np.complex128,
        }
        return dtype_map.get(torch_dtype, np.float32)
