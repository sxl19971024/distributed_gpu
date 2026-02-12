#!/usr/bin/env python
"""
distributed_gpu 安装脚本 (向后兼容)

所有项目元数据已统一至 pyproject.toml。
此文件仅保留以兼容 `python setup.py install` / `pip install -e .` 等旧式调用。

目标运行环境：
  - CUDA 12.1  (NVIDIA Driver >= 530.30.02)
  - OpenMPI 4.1.5 + mpi4py >= 3.1
  - PyTorch >= 2.1.0 (含 CUDA 12.1 支持)
"""

from setuptools import setup

# 所有配置项由 pyproject.toml 提供，此处仅做 fallback 触发
setup()
