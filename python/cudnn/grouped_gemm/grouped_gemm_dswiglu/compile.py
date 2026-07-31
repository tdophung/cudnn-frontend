# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral compile-only API for grouped GEMM + dSwiGLU.

This module consumes tensor metadata, compiles a native TVM-FFI function,
and returns a self-describing ABI. It has no PyTorch dependency, allocates no
framework buffers, and never launches the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Any, Mapping

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from ..grouped_gemm_swiglu.compile import (
    GroupedGemmAbiDescriptor,
    GroupedGemmAbiEntry,
    GroupedGemmElementType,
    GroupedGemmOperandDesc,
    _cutlass_dtype,
    _make_fake,
    _normalize_dtype,
    _scale_size,
)
from .grouped_gemm_dswiglu_quant import BlockScaledContiguousGroupedGemmKernel


@dataclass(frozen=True)
class GroupedGemmDswigluConfig:
    """Compile-time configuration for the SM100 grouped dSwiGLU kernel."""

    sf_vec_size: int = 32
    acc_dtype: GroupedGemmElementType | str = GroupedGemmElementType.FLOAT32
    mma_tiler_mn: tuple[int, int] = (256, 256)
    cluster_shape_mn: tuple[int, int] = (2, 1)
    vector_f32: bool = False
    discrete_col_sfd: bool = True
    use_mono_increase_expert_idx: bool = True
    num_cluster_overlap_margin: int = 0

    def __post_init__(self):
        object.__setattr__(self, "acc_dtype", _normalize_dtype(self.acc_dtype))
        object.__setattr__(self, "mma_tiler_mn", tuple(self.mma_tiler_mn))
        object.__setattr__(self, "cluster_shape_mn", tuple(self.cluster_shape_mn))
        if self.sf_vec_size not in (16, 32):
            raise ValueError(f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        if self.acc_dtype != GroupedGemmElementType.FLOAT32:
            raise ValueError("Grouped GEMM dSwiGLU requires a float32 accumulator")


_INPUT_ROLES = (
    "a",
    "b",
    "c",
    "sfa",
    "sfb",
    "padded_offsets",
    "alpha",
    "beta",
    "prob",
    "norm_const",
)
_OUTPUT_ROLES = ("d", "d_col", "sfd_row", "sfd_col", "dprob")
_ALL_ROLES = _INPUT_ROLES + _OUTPUT_ROLES
_COMPILE_CACHE: dict[str, tuple[Any, GroupedGemmAbiDescriptor]] = {}
_COMPILE_LOCK = threading.Lock()


def _expect_shape(operand: GroupedGemmOperandDesc, shape: tuple[int, ...]) -> None:
    if operand.shape != shape:
        raise ValueError(f"{operand.role}: expected shape {shape}, got {operand.shape}")


def _validate_operands(
    operands: Mapping[str, GroupedGemmOperandDesc],
    config: GroupedGemmDswigluConfig,
) -> None:
    missing = sorted(set(_ALL_ROLES) - set(operands))
    extra = sorted(set(operands) - set(_ALL_ROLES))
    if missing or extra:
        raise ValueError(f"Invalid operand roles: missing={missing}, extra={extra}")
    for role, operand in operands.items():
        if operand.role != role:
            raise ValueError(
                f"Operand mapping key {role!r} does not match descriptor role {operand.role!r}"
            )

    a, b, c = operands["a"], operands["b"], operands["c"]
    if len(a.shape) != 3 or a.shape[2] != 1:
        raise ValueError(f"a: expected [M,K,1], got {a.shape}")
    if len(b.shape) != 3:
        raise ValueError(f"b: expected [N,K,E], got {b.shape}")
    m, k, _ = a.shape
    n, b_k, experts = b.shape
    if k != b_k:
        raise ValueError(f"A/B K mismatch: {k} != {b_k}")
    if m % BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE:
        raise ValueError(
            f"M must be {BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE}-aligned, got {m}"
        )
    if experts > 1024:
        raise ValueError(f"Expert count must be <= 1024, got {experts}")

    _expect_shape(c, (m, 2 * n, 1))
    _expect_shape(operands["d"], (m, 2 * n, 1))
    _expect_shape(operands["d_col"], (m, 2 * n, 1))
    _expect_shape(operands["padded_offsets"], (experts,))
    _expect_shape(operands["alpha"], (experts,))
    _expect_shape(operands["beta"], (experts,))
    _expect_shape(operands["prob"], (m, 1, 1))
    _expect_shape(operands["norm_const"], (1,))
    _expect_shape(operands["dprob"], (m, 1, 1))

    expected_sfa = _scale_size(m, k, config.sf_vec_size, colwise=False)
    expected_sfb = _scale_size(n, k, config.sf_vec_size, colwise=False) * experts
    if len(operands["sfa"].shape) != 1 or operands["sfa"].shape[0] < expected_sfa:
        raise ValueError(f"sfa: expected a flat buffer with at least {expected_sfa} values")
    if len(operands["sfb"].shape) != 1 or operands["sfb"].shape[0] < expected_sfb:
        raise ValueError(f"sfb: expected a flat buffer with at least {expected_sfb} values")
    _expect_shape(
        operands["sfd_row"],
        (_scale_size(m, 2 * n, config.sf_vec_size, colwise=False),),
    )
    _expect_shape(
        operands["sfd_col"],
        (_scale_size(m, 2 * n, config.sf_vec_size, colwise=True),),
    )

    if a.dtype != b.dtype or a.dtype not in (
        GroupedGemmElementType.FLOAT8_E4M3FN,
        GroupedGemmElementType.FLOAT8_E5M2,
    ):
        raise ValueError(f"A/B must have one matching FP8 dtype, got {a.dtype}/{b.dtype}")
    if c.dtype not in (GroupedGemmElementType.BFLOAT16, GroupedGemmElementType.FLOAT16):
        raise ValueError(f"c: expected bfloat16 or float16, got {c.dtype.value}")
    if operands["d"].dtype != operands["d_col"].dtype:
        raise ValueError("d and d_col must have matching dtypes")
    if operands["d"].dtype not in (
        GroupedGemmElementType.FLOAT8_E4M3FN,
        GroupedGemmElementType.FLOAT8_E5M2,
    ):
        raise ValueError(f"d/d_col must have an FP8 dtype, got {operands['d'].dtype.value}")
    sf_dtype = GroupedGemmElementType.FLOAT8_E8M0FNU
    for role in ("sfa", "sfb", "sfd_row", "sfd_col"):
        if operands[role].dtype != sf_dtype:
            raise ValueError(f"{role}: expected {sf_dtype.value}, got {operands[role].dtype.value}")
    for role in ("alpha", "beta", "prob", "norm_const", "dprob"):
        if operands[role].dtype != GroupedGemmElementType.FLOAT32:
            raise ValueError(f"{role}: expected float32")
    if operands["padded_offsets"].dtype != GroupedGemmElementType.INT32:
        raise ValueError("padded_offsets: expected int32")


def _cache_key(
    operands: Mapping[str, GroupedGemmOperandDesc],
    config: GroupedGemmDswigluConfig,
) -> str:
    payload = {
        "operands": [
            {
                "role": role,
                "dtype": operands[role].dtype.value,
                "shape": operands[role].shape,
                "stride_order": operands[role].stride_order,
                "align": operands[role].align,
            }
            for role in _ALL_ROLES
        ],
        "config": {
            "sf_vec_size": config.sf_vec_size,
            "acc_dtype": config.acc_dtype.value,
            "mma_tiler_mn": config.mma_tiler_mn,
            "cluster_shape_mn": config.cluster_shape_mn,
            "vector_f32": config.vector_f32,
            "discrete_col_sfd": config.discrete_col_sfd,
            "use_mono_increase_expert_idx": config.use_mono_increase_expert_idx,
            "num_cluster_overlap_margin": config.num_cluster_overlap_margin,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return f"cudnn.grouped_gemm_dswiglu.{digest}"


def _compile_native(
    operands: Mapping[str, GroupedGemmOperandDesc],
    config: GroupedGemmDswigluConfig,
    key: str,
) -> tuple[Any, GroupedGemmAbiDescriptor]:
    kernel = BlockScaledContiguousGroupedGemmKernel(
        sf_vec_size=config.sf_vec_size,
        acc_dtype=_cutlass_dtype(config.acc_dtype),
        use_2cta_instrs=config.mma_tiler_mn[0] == 256,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
        vectorized_f32=config.vector_f32,
        discrete_col_sfd=config.discrete_col_sfd,
        expert_cnt=operands["padded_offsets"].shape[0],
        use_mono_increase_expert_idx=config.use_mono_increase_expert_idx,
    )
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
        config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
    )
    max_active_clusters -= config.num_cluster_overlap_margin
    if max_active_clusters <= 0:
        raise ValueError("num_cluster_overlap_margin leaves no active clusters")

    @cute.jit
    def launch(
        a,
        b,
        c,
        sfa,
        sfb,
        padded_offsets,
        alpha,
        beta,
        prob,
        norm_const,
        d,
        d_col,
        sfd_row,
        sfd_col,
        dprob,
        stream: cuda.CUstream,
    ):
        kernel(
            a,
            b,
            c,
            d,
            d_col,
            sfa,
            sfb,
            sfd_row,
            sfd_col,
            None,
            norm_const,
            padded_offsets,
            alpha,
            beta,
            prob,
            dprob,
            max_active_clusters,
            stream,
        )

    fake_operands = [_make_fake(operands[role]) for role in _ALL_ROLES]
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    compiled = cute.compile(launch, *fake_operands, fake_stream, options="--enable-tvm-ffi")
    entries = tuple(
        GroupedGemmAbiEntry(
            role,
            "arg",
            operands[role].dtype,
            operands[role].shape,
            operands[role].stride_order,
        )
        for role in _INPUT_ROLES
    ) + tuple(
        GroupedGemmAbiEntry(
            role,
            "ret",
            operands[role].dtype,
            operands[role].shape,
            operands[role].stride_order,
        )
        for role in _OUTPUT_ROLES
    ) + (GroupedGemmAbiEntry("stream", "stream", None),)
    descriptor = GroupedGemmAbiDescriptor(entries=entries, key=key)
    descriptor.arg_spec
    return compiled, descriptor


def compile_grouped_gemm_dswiglu(
    *,
    operands: Mapping[str, GroupedGemmOperandDesc],
    config: GroupedGemmDswigluConfig | None = None,
) -> tuple[Any, GroupedGemmAbiDescriptor]:
    """Compile and cache a native grouped GEMM + dSwiGLU TVM-FFI function."""
    config = GroupedGemmDswigluConfig() if config is None else config
    _validate_operands(operands, config)
    key = _cache_key(operands, config)
    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(key)
        if cached is None:
            cached = _compile_native(operands, config, key)
            _COMPILE_CACHE[key] = cached
        return cached


__all__ = ["GroupedGemmDswigluConfig", "compile_grouped_gemm_dswiglu"]
