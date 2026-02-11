#!/usr/bin/env python
"""
distributed_gpu 安装脚本
"""

from setuptools import setup, find_packages
import os

# 安全读取 README
here = os.path.abspath(os.path.dirname(__file__))
long_description = ""
readme_path = os.path.join(here, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="distributed-gpu",
    version="1.3.0",
    author="孙小林",
    author_email="1271364457@qq.com",
    description="基于MPI的分布式GPU计算框架，支持大规模张量并行计算",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dstar/distributed_gpu",
    packages=find_packages(exclude=["experiments*", "examples*", "docs*", "results*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.2.0",
        "mpi4py>=3.1.0",
        "numpy>=1.21.0",
        "opt-einsum>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mpi>=0.6",
            "matplotlib>=3.5.0",
        ],
        "full": [
            "cupy>=12.0.0",
            "scipy>=1.9.0",
            "psutil>=5.9.0",
            "pyyaml>=6.0",
        ],
    },
)
