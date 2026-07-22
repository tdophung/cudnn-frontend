# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Grouped GEMM SwiGLU Kernel Module

This module provides the forward grouped GEMM with SwiGLU activation
for MoE (Mixture of Experts) workloads on SM100+ GPUs.
"""

from importlib import import_module


_LAZY_IMPORTS = {
    "GroupedGemmSwigluSm100": (".api", "GroupedGemmSwigluSm100"),
    "grouped_gemm_swiglu_wrapper_sm100": (".api", "grouped_gemm_swiglu_wrapper_sm100"),
    "GroupedGemmElementType": (".compile", "GroupedGemmElementType"),
    "GroupedGemmOperandDesc": (".compile", "GroupedGemmOperandDesc"),
    "GroupedGemmSwigluConfig": (".compile", "GroupedGemmSwigluConfig"),
    "GroupedGemmAbiEntry": (".compile", "GroupedGemmAbiEntry"),
    "GroupedGemmAbiDescriptor": (".compile", "GroupedGemmAbiDescriptor"),
    "compile_grouped_gemm_swiglu": (".compile", "compile_grouped_gemm_swiglu"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, package=__name__), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "GroupedGemmSwigluSm100",
    "grouped_gemm_swiglu_wrapper_sm100",
    "GroupedGemmElementType",
    "GroupedGemmOperandDesc",
    "GroupedGemmSwigluConfig",
    "GroupedGemmAbiEntry",
    "GroupedGemmAbiDescriptor",
    "compile_grouped_gemm_swiglu",
]
