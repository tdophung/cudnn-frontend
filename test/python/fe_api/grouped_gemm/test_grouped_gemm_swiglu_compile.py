# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Tests for the framework-neutral grouped SwiGLU compile API."""

import subprocess
import sys

import pytest
import torch


def _operands(m=256, k=128, n=256, experts=1):
    from cudnn import GroupedGemmElementType as DType
    from cudnn import GroupedGemmOperandDesc as Operand

    def scale_size(rows, cols, colwise=False):
        if colwise:
            rows, cols = cols, rows
        ceil_div = lambda x, y: (x + y - 1) // y
        return 32 * 4 * ceil_div(rows, 128) * 4 * ceil_div(ceil_div(cols, 32), 4)

    return {
        "a": Operand("a", DType.FLOAT8_E4M3FN, (m, k, 1), (1, 0, 2)),
        "b": Operand("b", DType.FLOAT8_E4M3FN, (n, k, experts), (1, 0, 2)),
        "sfa": Operand("sfa", DType.FLOAT8_E8M0FNU, (scale_size(m, k),), (0,)),
        "sfb": Operand("sfb", DType.FLOAT8_E8M0FNU, (scale_size(n, k) * experts,), (0,)),
        "padded_offsets": Operand("padded_offsets", DType.INT32, (experts,), (0,)),
        "alpha": Operand("alpha", DType.FLOAT32, (experts,), (0,)),
        "prob": Operand("prob", DType.FLOAT32, (m, 1, 1), (1, 0, 2)),
        "norm_const": Operand("norm_const", DType.FLOAT32, (1,), (0,)),
        "c": Operand("c", DType.BFLOAT16, (m, n, 1), (1, 0, 2)),
        "d": Operand("d", DType.FLOAT8_E4M3FN, (m, n // 2, 1), (1, 0, 2)),
        "d_col": Operand("d_col", DType.FLOAT8_E4M3FN, (m, n // 2, 1), (1, 0, 2)),
        "sfd_row": Operand("sfd_row", DType.FLOAT8_E8M0FNU, (scale_size(m, n // 2),), (0,)),
        "sfd_col": Operand("sfd_col", DType.FLOAT8_E8M0FNU, (scale_size(m, n // 2, colwise=True),), (0,)),
    }


@pytest.mark.L0
def test_compile_api_imports_without_torch():
    script = r"""
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch intentionally blocked")
        return None

sys.meta_path.insert(0, BlockTorch())
from cudnn import compile_grouped_gemm_swiglu
assert callable(compile_grouped_gemm_swiglu)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.L0
def test_compile_api_rejects_unaligned_m_before_compile():
    from cudnn import compile_grouped_gemm_swiglu

    with pytest.raises(ValueError, match="M must be 256-aligned"):
        compile_grouped_gemm_swiglu(operands=_operands(m=128))


@pytest.mark.L1
def test_compile_api_returns_native_function_and_self_describing_abi():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Grouped GEMM SwiGLU compile requires SM100+")
    tvm_ffi = pytest.importorskip("tvm_ffi")
    from cudnn import compile_grouped_gemm_swiglu

    function, abi = compile_grouped_gemm_swiglu(operands=_operands())
    cached_function, cached_abi = compile_grouped_gemm_swiglu(operands=_operands())

    assert isinstance(function, tvm_ffi.Function)
    assert cached_function is function
    assert cached_abi == abi
    assert abi.arg_spec == ("args", "rets", "ctx.stream")
    assert [(entry.role, entry.kind) for entry in abi.entries] == [
        ("a", "arg"),
        ("b", "arg"),
        ("sfa", "arg"),
        ("sfb", "arg"),
        ("padded_offsets", "arg"),
        ("alpha", "arg"),
        ("prob", "arg"),
        ("norm_const", "arg"),
        ("c", "ret"),
        ("d", "ret"),
        ("d_col", "ret"),
        ("sfd_row", "ret"),
        ("sfd_col", "ret"),
        ("stream", "stream"),
    ]
