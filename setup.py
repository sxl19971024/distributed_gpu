#!/usr/bin/env python
"""
分布式GPU计算框架安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="distributed-gpu-framework",
    version="1.0.0",
    author="dstar",
    author_email="dstar@example.com",
    description="基于MPI的分布式GPU计算框架，用于大规模科学计算",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dstar/distributed-gpu-framework",
    packages=find_packages(),
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
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "mpi4py>=3.1.0",
        "numpy>=1.21.0",
        "opt-einsum>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mpi>=0.6",
        ],
        "full": [
            "cupy-cuda11x>=12.0.0",
            "scipy>=1.9.0",
            "psutil>=5.9.0",
            "pyyaml>=6.0",
        ],
    },
)
