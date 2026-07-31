# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from importlib import import_module


_LAZY_IMPORTS = {
    "GroupedGemmDswigluSm100": (".api", "GroupedGemmDswigluSm100"),
    "grouped_gemm_dswiglu_wrapper_sm100": (".api", "grouped_gemm_dswiglu_wrapper_sm100"),
    "GroupedGemmDswigluConfig": (".compile", "GroupedGemmDswigluConfig"),
    "compile_grouped_gemm_dswiglu": (".compile", "compile_grouped_gemm_dswiglu"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, package=__name__), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "GroupedGemmDswigluSm100",
    "grouped_gemm_dswiglu_wrapper_sm100",
    "GroupedGemmDswigluConfig",
    "compile_grouped_gemm_dswiglu",
]
