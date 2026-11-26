# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared SwiGLU MLPs using b12x projections and caller-owned buffers."""

from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl
from b12x.gemm.blockscaled._quantize import _scale_offset
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall

from hyperflare_recipes.components import sm103
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.weights import MLPWeights
from ignition.preparation.tuning import declare


@triton.jit
def _swiglu_mxfp8_kernel(
    GateUp, Values, ScaleRows, ScaleMma,
    TOKENS,
    WIDTH: tl.constexpr, PADDED_WIDTH: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Fuse rounded BF16 SwiGLU, zero-padding to 128-aligned MXFP8 K width, and
    # block-32 E4M3/UE8M0 quantization into one pass without a BF16 round-trip.
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < TOKENS
    cols = col_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    in_width = cols < WIDTH
    in_padded = cols < PADDED_WIDTH
    load_mask = rmask[:, None] & in_width[None, :]
    store_mask = rmask[:, None] & in_padded[None, :]

    row_off = rows.to(tl.int64)[:, None] * (2 * WIDTH)
    gate = tl.load(GateUp + row_off + cols[None, :], mask=load_mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp + row_off + WIDTH + cols[None, :], mask=load_mask, other=0.0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    val = tl.where(load_mask, (activated * up).to(tl.bfloat16), 0.0).to(tl.bfloat16)

    groups: tl.constexpr = PADDED_WIDTH // 32
    local_groups: tl.constexpr = BLOCK_W // 32
    x = tl.reshape(val.to(tl.float32), (BLOCK_R, local_groups, 32))
    amax = tl.max(tl.abs(x), axis=2)
    ratio = tl.where(amax > 0, amax / 448.0, 1.0)
    exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(ratio)), -127.0), 127.0)
    quantized = (x / tl.exp2(exponent)[:, :, None]).to(tl.float8e4nv)
    tl.store(
        Values + rows.to(tl.int64)[:, None] * PADDED_WIDTH + cols[None, :],
        tl.reshape(quantized, (BLOCK_R, BLOCK_W)),
        mask=store_mask,
    )
    group = col_tile * local_groups + tl.arange(0, local_groups)
    gmask = rmask[:, None] & (group[None, :] < groups)
    byte = (exponent + 127).to(tl.uint8)
    tl.store(ScaleRows + rows.to(tl.int64)[:, None] * groups + group[None, :], byte, mask=gmask)
    tl.store(ScaleMma + _scale_offset(rows[:, None], group[None, :], groups), byte, mask=gmask)


def _launch_swiglu_mxfp8(gate_up: torch.Tensor, encoded_bases: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         *, width: int, padded_width: int) -> None:
    tokens = gate_up.shape[0]
    block_r = 16 if tokens >= 8192 else 8
    block_w = 128
    grid = (triton.cdiv(tokens, block_r), padded_width // block_w)
    torch.library.wrap_triton(_swiglu_mxfp8_kernel)[grid](
        gate_up,
        encoded_bases[0],
        encoded_bases[1],
        encoded_bases[2],
        tokens,
        WIDTH=width,
        PADDED_WIDTH=padded_width,
        BLOCK_R=block_r,
        BLOCK_W=block_w,
        num_warps=4,
    )


@dataclass(frozen=True)
class MLPScratch:
    gate_up: torch.Tensor
    intermediate: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, hidden, intermediate, device, *, buffers=None):
        if any(type(value) is not int or value <= 0 for value in (tokens, hidden, intermediate)):
            raise ValueError("MLP scratch dimensions must be positive integers")
        return cls(*((torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
                      if buffers is None else buffers.zeros((tokens, width), torch.bfloat16))
                     for width in (2 * intermediate, intermediate, hidden)))

    def rows(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.output.shape[0]:
            raise ValueError("MLP rows exceed scratch capacity")
        return MLPScratch(self.gate_up[:tokens], self.intermediate[:tokens], self.output[:tokens])


class PreparedMLP:
    """Dense or shared-expert MLP, including rank-local intermediate partitions.

    Gate/up projection rows are gate first, then up. The output is a partial
    hidden vector for sharded weights; the caller owns its collective reduction.
    All output and intermediate buffers must remain live through graph replay.
    """

    def __init__(self, source, weights: MLPWeights, gate_up, intermediate, output, *,
                 session, name, workspace, swiglu_limit=float("inf"), round_silu=True,
                 activation_mode="auto", gate_up_bias=None, down_bias=None, clamp_after_silu=False,
                 swiglu_alpha=1., swiglu_beta=0., encoded=None):
        self.gate_up, self.intermediate, self.output = gate_up, intermediate, output
        self.swiglu_limit, self.round_silu = swiglu_limit, round_silu
        if source.ndim != 2 or intermediate.ndim != 2:
            raise ValueError("MLP input and intermediate must be matrices")
        tokens, hidden = source.shape
        width = intermediate.shape[1]
        for tensor, shape in ((source, (tokens, hidden)), (gate_up, (tokens, 2 * width)),
                              (intermediate, (tokens, width)), (output, (tokens, hidden))):
            if (tensor.dtype != torch.bfloat16 or tensor.shape != shape
                    or tensor.device != source.device or not tensor.is_contiguous()):
                raise ValueError("MLP buffers require contiguous BF16 storage with matching geometry and device")

        def projection(input, weight, out, suffix, bias, *, preencoded=None, mxfp8_min_amax=None):
            from hyperflare_recipes.components.block_fp8 import BlockFP8Projection, BlockFP8Weight

            if isinstance(weight, BlockFP8Weight):
                if bias is not None or preencoded is not None or mxfp8_min_amax is not None:
                    raise ValueError("block-FP8 MLP projections do not support bias or preencoded MXFP8 inputs")
                return BlockFP8Projection(input, weight, out, session=session, name=f"{name}.{suffix}")
            if isinstance(weight, torch.Tensor):
                if preencoded is not None or mxfp8_min_amax is not None:
                    raise ValueError("unquantized MLP projections do not support preencoded MXFP8 inputs")
                return UnquantizedProjection(input, weight, out, session=session, name=f"{name}.{suffix}", bias=bias)
            return QuantizedProjection(input, weight, out, session=session,
                                       name=f"{name}.{suffix}", workspace=workspace,
                                       activation_mode=activation_mode, bias=bias, encoded=preencoded,
                                       mxfp8_min_amax=mxfp8_min_amax)

        from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

        use_cute_up_mxfp8 = (
            encoded is None
            and tokens >= 4096
            and source.is_cuda
            and not sm103.is_sm103(source.device)
            and isinstance(weights.gate_up, MXFP8LinearWeight)
            and workspace is not None
            and gate_up_bias is None
            and activation_mode in ("auto", "quantized")
            and weights.gate_up.padded_in_features == hidden
            and weights.gate_up.out_features >= 1280
        )
        self.up_projection = projection(
            source, weights.gate_up, gate_up, "gate_up", gate_up_bias,
            preencoded=encoded, mxfp8_min_amax=0.0 if use_cute_up_mxfp8 else None,
        )
        fuse_down_mxfp8 = (
            tokens > 256
            and source.is_cuda
            and not sm103.is_sm103(source.device)
            and isinstance(weights.down, MXFP8LinearWeight)
            and workspace is not None
            and down_bias is None
            and activation_mode in ("auto", "quantized")
            and math.isinf(swiglu_limit) and swiglu_limit > 0
            and round_silu
            and not clamp_after_silu
            and swiglu_alpha == 1.0
            and swiglu_beta == 0.0
            and width % 32 == 0
            and weights.down.padded_in_features >= width
            and weights.down.padded_in_features % 128 == 0
        )
        self._down_encoded_bases = None
        self._down_encoding_layout = None
        self._down_source = None
        self._down_width = width
        self._down_padded_width = width
        self._down_tokens = tokens
        if fuse_down_mxfp8:
            from b12x.quantization import mxfp8

            padded_width = weights.down.padded_in_features
            self._down_padded_width = padded_width
            shapes = mxfp8.dense_rows_storage_shapes(tokens, padded_width)
            bases = tuple(
                torch.full(shape, 127, dtype=dtype, device=source.device)
                if index > 0 else torch.zeros(shape, dtype=dtype, device=source.device)
                for index, (shape, dtype) in enumerate(zip(shapes, (torch.float8_e4m3fn, torch.uint8, torch.uint8), strict=True))
            )
            down_encoded = mxfp8.dense_rows_from_bases(*bases)
            down_source = (
                intermediate if padded_width == width
                else torch.zeros((tokens, padded_width), dtype=torch.bfloat16, device=source.device)
            )
            self._down_encoded_bases = bases
            self._down_source = down_source
            self.down_projection = projection(down_source, weights.down, output, "down", down_bias,
                                              preencoded=down_encoded)
            self.activation_plan = None
        else:
            self.down_projection = projection(intermediate, weights.down, output, "down", down_bias)
            self.activation_plan = hyperconnection.plan(
                hyperconnection.Caps(device=source.device, max_tokens=tokens, hidden_size=width),
                invocation={"operation": "swiglu", "limit": swiglu_limit, "round_silu": round_silu,
                            "clamp_after_silu": clamp_after_silu,
                            "swiglu_alpha": swiglu_alpha, "swiglu_beta": swiglu_beta},
            )
            declare(session, (self.activation_plan.request(
                name=f"{name}.swiglu", prepare_call=lambda state: PreparedCall(
                    run=lambda: state.launch(gate_up, gate_up, intermediate),
                ),
            ),))
        self._functional_layout = None

    def functional(self, *, session):
        """Run the prepared MLP with compiler-local projection and activation outputs."""
        if self._functional_layout is None:
            self._functional_layout = (
                tuple(self.gate_up.shape), self.gate_up.stride(), self.gate_up.dtype,
                tuple(self.intermediate.shape), self.intermediate.stride(), self.intermediate.dtype,
                tuple(self.output.shape), self.output.stride(), self.output.dtype,
            )
            if self._down_encoded_bases is not None:
                self._down_encoding_layout = tuple(
                    (tuple(base.shape), base.dtype) for base in self._down_encoded_bases
                )
                self._down_encoded_bases = None
                self._down_source = None
            self.up_projection.functional(session=session)
            self.down_projection.functional(session=session)
            self.gate_up = self.intermediate = self.output = None
        return self

    def __call__(self, source, *, encoded_bases=None):
        up_kwargs = {} if encoded_bases is None else {"encoded_bases": encoded_bases}
        if self._functional_layout is not None:
            _, _, _, intermediate_shape, intermediate_stride, intermediate_dtype, _, _, _ = self._functional_layout
            gate_up = self.up_projection(source, **up_kwargs)
            if self._down_encoding_layout is not None:
                mma_unpadded = self._down_tokens % 128 == 0
                down_bases = tuple(
                    torch.empty(shape, dtype=dtype, device=source.device)
                    if index < 2 or mma_unpadded
                    else torch.full(shape, 127, dtype=dtype, device=source.device)
                    for index, (shape, dtype) in enumerate(self._down_encoding_layout)
                )
                _launch_swiglu_mxfp8(gate_up, down_bases, width=self._down_width,
                                     padded_width=self._down_padded_width)
                return self.down_projection(down_bases[0], encoded_bases=down_bases)
            intermediate = torch.empty_strided(intermediate_shape, intermediate_stride, dtype=intermediate_dtype,
                                               device=source.device)
            hyperconnection.run_swiglu(gate_up, limit=self.swiglu_limit,
                                      round_silu=self.round_silu, out=intermediate,
                                      plan=self.activation_plan)
            return self.down_projection(intermediate)
        self.up_projection(source, **up_kwargs)
        if self._down_encoded_bases is not None:
            _launch_swiglu_mxfp8(self.gate_up, self._down_encoded_bases, width=self._down_width,
                                 padded_width=self._down_padded_width)
            return self.down_projection(self._down_source)
        hyperconnection.run_swiglu(self.gate_up, limit=self.swiglu_limit,
                                  round_silu=self.round_silu, out=self.intermediate,
                                  plan=self.activation_plan)
        return self.down_projection(self.intermediate)


class PreparedUngatedMLP:
    """Two projections with a BF16 ReLU-squared or SiLU intermediate."""

    def __init__(self, source, up_weight, down_weight, projected, intermediate, output, *,
                 activation, session, name, workspace, up_bias=None, down_bias=None):
        if activation not in ("relu2", "silu"):
            raise ValueError("ungated MLP activation must be relu2 or silu")
        if (source.ndim != 2 or projected.ndim != 2 or projected.shape != intermediate.shape
                or projected.shape[0] != source.shape[0] or output.shape[0] != source.shape[0]
                or any(t.dtype != torch.bfloat16 or t.device != source.device or not t.is_contiguous()
                       for t in (source, projected, intermediate, output))):
            raise ValueError("ungated MLP buffers require matching contiguous BF16 rows")
        self.activation, self.intermediate = activation, intermediate

        def projection(x, weight, y, suffix, bias):
            from hyperflare_recipes.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight

            if isinstance(weight, TensorFP8Weight):
                if bias is not None:
                    raise ValueError("tensor-FP8 MLP projections do not support bias")
                return TensorFP8Projection(x, weight, y, session=session, name=name + suffix)
            kwargs = dict(session=session, name=name + suffix, bias=bias)
            if isinstance(weight, torch.Tensor):
                return UnquantizedProjection(x, weight, y, **kwargs)
            return QuantizedProjection(x, weight, y, workspace=workspace, activation_mode="auto", **kwargs)

        self.up_projection = projection(source, up_weight, projected, ".up", up_bias)
        self.down_projection = projection(intermediate, down_weight, output, ".down", down_bias)
        if activation == "relu2":
            self.activation_plan = hyperconnection.plan(hyperconnection.Caps(device=source.device,
                max_tokens=source.shape[0], hidden_size=projected.shape[1]), invocation={"operation": "relu2"})
            declare(session, (self.activation_plan.request(name=name + ".relu2", prepare_call=lambda state:
                PreparedCall(run=lambda: state.launch(projected, projected, intermediate))),))

    def functional(self, *, session):
        """Keep the prepared arithmetic while allocating intermediate storage locally."""
        if self.intermediate is not None:
            self.intermediate_layout = (tuple(self.intermediate.shape), self.intermediate.stride(),
                                        self.intermediate.dtype)
            self.up_projection.functional(session=session)
            self.down_projection.functional(session=session)
            self.intermediate = None
        return self

    def __call__(self, source):
        value = self.up_projection(source)
        intermediate = self.intermediate
        if intermediate is None:
            shape, stride, dtype = self.intermediate_layout
            intermediate = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        intermediate.copy_(torch.relu(value).square() if self.activation == "relu2"
                           else torch.nn.functional.silu(value))
        return self.down_projection(intermediate)
