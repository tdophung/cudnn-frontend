# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy public exports for grouped GEMM CuTeDSL APIs.

Keeping these imports lazy lets framework-neutral compile-only entrypoints be
used in environments that intentionally do not install PyTorch.
"""

from importlib import import_module


_LAZY_IMPORTS = {
    "GroupedGemmSwigluSm100": (".grouped_gemm_swiglu", "GroupedGemmSwigluSm100"),
    "grouped_gemm_swiglu_wrapper_sm100": (".grouped_gemm_swiglu", "grouped_gemm_swiglu_wrapper_sm100"),
    "GroupedGemmElementType": (".grouped_gemm_swiglu", "GroupedGemmElementType"),
    "GroupedGemmOperandDesc": (".grouped_gemm_swiglu", "GroupedGemmOperandDesc"),
    "GroupedGemmSwigluConfig": (".grouped_gemm_swiglu", "GroupedGemmSwigluConfig"),
    "GroupedGemmAbiEntry": (".grouped_gemm_swiglu", "GroupedGemmAbiEntry"),
    "GroupedGemmAbiDescriptor": (".grouped_gemm_swiglu", "GroupedGemmAbiDescriptor"),
    "compile_grouped_gemm_swiglu": (".grouped_gemm_swiglu", "compile_grouped_gemm_swiglu"),
    "GroupedGemmDswigluSm100": (".grouped_gemm_dswiglu.api", "GroupedGemmDswigluSm100"),
    "grouped_gemm_dswiglu_wrapper_sm100": (".grouped_gemm_dswiglu.api", "grouped_gemm_dswiglu_wrapper_sm100"),
    "GroupedGemmDswigluConfig": (".grouped_gemm_dswiglu", "GroupedGemmDswigluConfig"),
    "compile_grouped_gemm_dswiglu": (".grouped_gemm_dswiglu", "compile_grouped_gemm_dswiglu"),
    "GroupedGemmQuantSm100": (".grouped_gemm_quant.api", "GroupedGemmQuantSm100"),
    "grouped_gemm_quant_wrapper_sm100": (".grouped_gemm_quant.api", "grouped_gemm_quant_wrapper_sm100"),
    "GroupedGemmSreluSm100": (".grouped_gemm_srelu.api", "GroupedGemmSreluSm100"),
    "grouped_gemm_srelu_wrapper_sm100": (".grouped_gemm_srelu.api", "grouped_gemm_srelu_wrapper_sm100"),
    "GroupedGemmDsreluSm100": (".grouped_gemm_dsrelu.api", "GroupedGemmDsreluSm100"),
    "grouped_gemm_dsrelu_wrapper_sm100": (".grouped_gemm_dsrelu.api", "grouped_gemm_dsrelu_wrapper_sm100"),
    "GroupedGemmGluSm100": (".grouped_gemm_glu.api", "GroupedGemmGluSm100"),
    "grouped_gemm_glu_wrapper_sm100": (".grouped_gemm_glu.api", "grouped_gemm_glu_wrapper_sm100"),
    "GroupedGemmGluHadamardSm100": (".grouped_gemm_glu_hadamard.api", "GroupedGemmGluHadamardSm100"),
    "grouped_gemm_glu_hadamard_wrapper_sm100": (".grouped_gemm_glu_hadamard.api", "grouped_gemm_glu_hadamard_wrapper_sm100"),
    "GroupedGemmDgluSm100": (".grouped_gemm_dglu.api", "GroupedGemmDgluSm100"),
    "grouped_gemm_dglu_wrapper_sm100": (".grouped_gemm_dglu.api", "grouped_gemm_dglu_wrapper_sm100"),
    "GroupedGemmWgradSm100": (".grouped_gemm_wgrad.api", "GroupedGemmWgradSm100"),
    "grouped_gemm_wgrad_wrapper_sm100": (".grouped_gemm_wgrad.api", "grouped_gemm_wgrad_wrapper_sm100"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, package=__name__), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_IMPORTS)
