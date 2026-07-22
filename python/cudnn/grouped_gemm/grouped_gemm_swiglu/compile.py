# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral compile-only API for grouped GEMM + SwiGLU.

This module deliberately has no PyTorch dependency. It consumes tensor
metadata, compiles a native TVM-FFI function, and returns a self-describing
ABI. It allocates no framework buffers and never launches the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import threading
from typing import Any, Literal, Mapping

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from .grouped_gemm_swiglu_quant import BlockScaledContiguousGroupedGemmKernel


class GroupedGemmElementType(str, Enum):
    """Framework-neutral element types accepted by the compile API."""

    FLOAT4_E2M1FN_X2 = "float4_e2m1fn_x2"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT8_E8M0FNU = "float8_e8m0fnu"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    INT32 = "int32"
    UINT8 = "uint8"


_CUTLASS_DTYPES = {
    GroupedGemmElementType.FLOAT4_E2M1FN_X2: "Float4E2M1FN",
    GroupedGemmElementType.FLOAT8_E4M3FN: "Float8E4M3FN",
    GroupedGemmElementType.FLOAT8_E5M2: "Float8E5M2",
    GroupedGemmElementType.FLOAT8_E8M0FNU: "Float8E8M0FNU",
    GroupedGemmElementType.FLOAT16: "Float16",
    GroupedGemmElementType.BFLOAT16: "BFloat16",
    GroupedGemmElementType.FLOAT32: "Float32",
    GroupedGemmElementType.INT32: "Int32",
    GroupedGemmElementType.UINT8: "Uint8",
}


def _normalize_dtype(dtype: GroupedGemmElementType | str) -> GroupedGemmElementType:
    if isinstance(dtype, GroupedGemmElementType):
        return dtype
    return GroupedGemmElementType(dtype)


def _cutlass_dtype(dtype: GroupedGemmElementType | str) -> type[cutlass.Numeric]:
    dtype = _normalize_dtype(dtype)
    try:
        return getattr(cutlass, _CUTLASS_DTYPES[dtype])
    except AttributeError as exc:
        raise RuntimeError(f"Installed nvidia-cutlass-dsl does not provide {dtype.value}") from exc


@dataclass(frozen=True)
class GroupedGemmOperandDesc:
    """Storage-free metadata for one logical kernel operand."""

    role: str
    dtype: GroupedGemmElementType | str
    shape: tuple[int, ...]
    stride_order: tuple[int, ...]
    align: int = 16

    def __post_init__(self):
        dtype = _normalize_dtype(self.dtype)
        shape = tuple(int(dim) for dim in self.shape)
        stride_order = tuple(int(dim) for dim in self.stride_order)
        if not self.role:
            raise ValueError("Operand role must not be empty")
        if not shape or any(dim < 0 for dim in shape):
            raise ValueError(f"{self.role}: shape must contain non-negative dimensions, got {shape}")
        if tuple(sorted(stride_order)) != tuple(range(len(shape))):
            raise ValueError(f"{self.role}: invalid stride_order {stride_order} for rank {len(shape)}")
        if self.align <= 0 or self.align & (self.align - 1):
            raise ValueError(f"{self.role}: align must be a positive power of two, got {self.align}")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride_order", stride_order)


@dataclass(frozen=True)
class GroupedGemmSwigluConfig:
    """Compile-time configuration for the SM100 grouped SwiGLU kernel."""

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
            raise ValueError("Grouped GEMM SwiGLU requires a float32 accumulator")


@dataclass(frozen=True)
class GroupedGemmAbiEntry:
    """One positional slot in the returned native TVM-FFI function."""

    role: str
    kind: Literal["arg", "ret", "stream"]
    dtype: GroupedGemmElementType | None
    shape: tuple[int, ...] = ()
    stride_order: tuple[int, ...] = ()
    optional: bool = False


@dataclass(frozen=True)
class GroupedGemmAbiDescriptor:
    """The exact positional ABI and stable registration key for a compile."""

    entries: tuple[GroupedGemmAbiEntry, ...]
    key: str

    @property
    def arg_spec(self) -> tuple[str, ...]:
        """jax-tvm-ffi routing for this native, inputs-then-results ABI."""
        kinds = tuple(entry.kind for entry in self.entries)
        first_ret = next((idx for idx, kind in enumerate(kinds) if kind == "ret"), len(kinds))
        first_stream = next((idx for idx, kind in enumerate(kinds) if kind == "stream"), len(kinds))
        if any(kind != "arg" for kind in kinds[:first_ret]):
            raise ValueError("ABI args must be contiguous")
        if any(kind != "ret" for kind in kinds[first_ret:first_stream]):
            raise ValueError("ABI returns must be contiguous")
        if kinds[first_stream:] != ("stream",):
            raise ValueError("ABI must contain exactly one trailing stream")
        return ("args", "rets", "ctx.stream")


_INPUT_ROLES = ("a", "b", "sfa", "sfb", "padded_offsets", "alpha", "prob", "norm_const")
_OUTPUT_ROLES = ("c", "d", "d_col", "sfd_row", "sfd_col")
_ALL_ROLES = _INPUT_ROLES + _OUTPUT_ROLES
_COMPILE_CACHE: dict[str, tuple[Any, GroupedGemmAbiDescriptor]] = {}
_COMPILE_LOCK = threading.Lock()


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _scale_size(rows: int, cols: int, sf_vec_size: int, *, colwise: bool) -> int:
    if colwise:
        rows, cols = cols, rows
    return 32 * 4 * _ceil_div(rows, 128) * 4 * _ceil_div(_ceil_div(cols, sf_vec_size), 4)


def _expect_shape(operand: GroupedGemmOperandDesc, shape: tuple[int, ...]) -> None:
    if operand.shape != shape:
        raise ValueError(f"{operand.role}: expected shape {shape}, got {operand.shape}")


def _validate_operands(operands: Mapping[str, GroupedGemmOperandDesc], config: GroupedGemmSwigluConfig) -> None:
    missing = sorted(set(_ALL_ROLES) - set(operands))
    extra = sorted(set(operands) - set(_ALL_ROLES))
    if missing or extra:
        raise ValueError(f"Invalid operand roles: missing={missing}, extra={extra}")
    for role, operand in operands.items():
        if operand.role != role:
            raise ValueError(f"Operand mapping key {role!r} does not match descriptor role {operand.role!r}")

    a, b = operands["a"], operands["b"]
    if len(a.shape) != 3 or a.shape[2] != 1:
        raise ValueError(f"a: expected [M,K,1], got {a.shape}")
    if len(b.shape) != 3:
        raise ValueError(f"b: expected [N,K,E], got {b.shape}")
    m, k, _ = a.shape
    n, b_k, experts = b.shape
    if k != b_k:
        raise ValueError(f"A/B K mismatch: {k} != {b_k}")
    if m % BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE:
        raise ValueError(f"M must be {BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE}-aligned, got {m}")
    if n % 2:
        raise ValueError(f"Combined SwiGLU N must be even, got {n}")
    if experts > 1024:
        raise ValueError(f"Expert count must be <= 1024, got {experts}")

    _expect_shape(operands["c"], (m, n, 1))
    _expect_shape(operands["d"], (m, n // 2, 1))
    _expect_shape(operands["d_col"], (m, n // 2, 1))
    _expect_shape(operands["padded_offsets"], (experts,))
    _expect_shape(operands["alpha"], (experts,))
    _expect_shape(operands["prob"], (m, 1, 1))
    _expect_shape(operands["norm_const"], (1,))
    expected_sfa = _scale_size(m, k, config.sf_vec_size, colwise=False)
    expected_sfb = _scale_size(n, k, config.sf_vec_size, colwise=False) * experts
    if len(operands["sfa"].shape) != 1 or operands["sfa"].shape[0] < expected_sfa:
        raise ValueError(f"sfa: expected a flat buffer with at least {expected_sfa} values")
    if len(operands["sfb"].shape) != 1 or operands["sfb"].shape[0] < expected_sfb:
        raise ValueError(f"sfb: expected a flat buffer with at least {expected_sfb} values")
    _expect_shape(operands["sfd_row"], (_scale_size(m, n // 2, config.sf_vec_size, colwise=False),))
    _expect_shape(operands["sfd_col"], (_scale_size(m, n // 2, config.sf_vec_size, colwise=True),))

    if a.dtype != b.dtype or a.dtype not in (GroupedGemmElementType.FLOAT8_E4M3FN, GroupedGemmElementType.FLOAT8_E5M2):
        raise ValueError(f"A/B must have one matching FP8 dtype, got {a.dtype}/{b.dtype}")
    sf_dtype = GroupedGemmElementType.FLOAT8_E8M0FNU
    for role in ("sfa", "sfb", "sfd_row", "sfd_col"):
        if operands[role].dtype != sf_dtype:
            raise ValueError(f"{role}: expected {sf_dtype.value}, got {operands[role].dtype.value}")
    if operands["c"].dtype not in (GroupedGemmElementType.BFLOAT16, GroupedGemmElementType.FLOAT16):
        raise ValueError(f"c: expected bfloat16 or float16, got {operands['c'].dtype.value}")
    if operands["d"].dtype != operands["d_col"].dtype:
        raise ValueError("d and d_col must have matching dtypes")
    if operands["d"].dtype not in (
        GroupedGemmElementType.FLOAT8_E4M3FN,
        GroupedGemmElementType.FLOAT8_E5M2,
    ):
        raise ValueError(f"d/d_col must have an FP8 dtype, got {operands['d'].dtype.value}")
    for role in ("alpha", "prob", "norm_const"):
        if operands[role].dtype != GroupedGemmElementType.FLOAT32:
            raise ValueError(f"{role}: expected float32")
    if operands["padded_offsets"].dtype != GroupedGemmElementType.INT32:
        raise ValueError("padded_offsets: expected int32")


def _cache_key(operands: Mapping[str, GroupedGemmOperandDesc], config: GroupedGemmSwigluConfig) -> str:
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
    return f"cudnn.grouped_gemm_swiglu.{digest}"


def _make_fake(desc: GroupedGemmOperandDesc):
    return cute.runtime.make_fake_compact_tensor(
        dtype=_cutlass_dtype(desc.dtype),
        shape=desc.shape,
        stride_order=desc.stride_order,
        assumed_align=desc.align,
    )


def _compile_native(
    operands: Mapping[str, GroupedGemmOperandDesc],
    config: GroupedGemmSwigluConfig,
    key: str,
) -> tuple[Any, GroupedGemmAbiDescriptor]:
    kernel = BlockScaledContiguousGroupedGemmKernel(
        sf_vec_size=config.sf_vec_size,
        acc_dtype=_cutlass_dtype(config.acc_dtype),
        use_2cta_instrs=config.mma_tiler_mn[0] == 256,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
        vector_f32=config.vector_f32,
        generate_sfd=True,
        discrete_col_sfd=config.discrete_col_sfd,
        expert_cnt=operands["padded_offsets"].shape[0],
        use_mono_increase_expert_idx=config.use_mono_increase_expert_idx,
    )
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(config.cluster_shape_mn[0] * config.cluster_shape_mn[1])
    max_active_clusters -= config.num_cluster_overlap_margin
    if max_active_clusters <= 0:
        raise ValueError("num_cluster_overlap_margin leaves no active clusters")

    # jax-tvm-ffi routes complete input and result groups. Compile a native
    # reorder launcher with that stable grouping while keeping the volatile
    # raw-kernel ABI private to cuDNN-FE. This remains native code: no Python
    # callback is present on the execution path.
    @cute.jit
    def launch(a, b, sfa, sfb, padded_offsets, alpha, prob, norm_const, c, d, d_col, sfd_row, sfd_col, stream: cuda.CUstream):
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
            prob,
            max_active_clusters,
            stream,
        )

    fake_operands = [_make_fake(operands[role]) for role in _ALL_ROLES]
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    compiled = cute.compile(launch, *fake_operands, fake_stream, options="--enable-tvm-ffi")
    entries = tuple(
        GroupedGemmAbiEntry(role, "arg", operands[role].dtype, operands[role].shape, operands[role].stride_order)
        for role in _INPUT_ROLES
    ) + tuple(
        GroupedGemmAbiEntry(role, "ret", operands[role].dtype, operands[role].shape, operands[role].stride_order)
        for role in _OUTPUT_ROLES
    ) + (GroupedGemmAbiEntry("stream", "stream", None),)
    descriptor = GroupedGemmAbiDescriptor(entries=entries, key=key)
    descriptor.arg_spec  # Validate the grouped ABI before publishing it.
    return compiled, descriptor


def compile_grouped_gemm_swiglu(
    *,
    operands: Mapping[str, GroupedGemmOperandDesc],
    config: GroupedGemmSwigluConfig | None = None,
) -> tuple[Any, GroupedGemmAbiDescriptor]:
    """Compile and cache a native grouped GEMM + SwiGLU TVM-FFI function.

    The call consumes metadata only. It does not allocate output buffers and
    does not launch. The returned descriptor is the positional ABI authority
    for the exact returned function.
    """
    config = GroupedGemmSwigluConfig() if config is None else config
    _validate_operands(operands, config)
    key = _cache_key(operands, config)
    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(key)
        if cached is None:
            cached = _compile_native(operands, config, key)
            _COMPILE_CACHE[key] = cached
        return cached


__all__ = [
    "GroupedGemmElementType",
    "GroupedGemmOperandDesc",
    "GroupedGemmSwigluConfig",
    "GroupedGemmAbiEntry",
    "GroupedGemmAbiDescriptor",
    "compile_grouped_gemm_swiglu",
]
