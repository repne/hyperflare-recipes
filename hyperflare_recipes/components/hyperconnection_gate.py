# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared gate projection and stream mean with caller-owned outputs."""

from functools import partial

import torch
import triton
import triton.language as tl
from b12x.gemm import hyperconnection_gate
from b12x.norm.hyperconnection import api as hc
from b12x.preparation import FrozenMapping, PreparedCall

from ignition.components.collectives import PreparedChunkedAllReduce
from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.fixtures import hyperconnection as hc_fixtures
from hyperflare_recipes.fixtures.hyperconnection_gate import HyperconnectionGateInputs
from ignition.preparation.triton import warmup_triton
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

# Launch configurations ride on the kernels via one-Config autotune decorators so
# whole-forward compilation, which drops launch keywords, retains them.
_M1_GEMV_CONFIG = triton.Config({}, num_warps=1, num_stages=2)
_BPTR_4ACC_CONFIG = triton.Config({}, num_warps=4, num_stages=3)


@triton.autotune(configs=[_M1_GEMV_CONFIG], key=[])
@triton.jit
def _hc_gate_m1_gemv(
    PROJ,
    WEIGHT,
    NORM,
    OUT,
    H: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    BH: tl.constexpr,
):
    """Single-token fused scaled-SiLU, rank-320 GEMV, and BF16-rounded stream mean."""
    pid = tl.program_id(0)
    h = pid * BH + tl.arange(0, BH)
    hm = h < H
    k256 = tl.arange(0, 256)
    k64 = tl.arange(0, 64)
    r0 = tl.load(PROJ + k256).to(tl.float32)
    s0 = (r0 * (1.0 / S)).to(tl.bfloat16).to(tl.float32)
    x0 = (s0 * tl.sigmoid(s0)).to(tl.bfloat16).to(tl.float32)
    r1 = tl.load(PROJ + 256 + k64).to(tl.float32)
    s1 = (r1 * (1.0 / S)).to(tl.bfloat16).to(tl.float32)
    x1 = (s1 * tl.sigmoid(s1)).to(tl.bfloat16).to(tl.float32)

    total = tl.zeros((BH,), dtype=tl.float32)
    for stream in tl.static_range(S):
        base = (stream * H + h)[:, None].to(tl.int64) * R
        w0 = tl.load(WEIGHT + base + k256[None, :], hm[:, None], other=0.0, cache_modifier=".cs").to(tl.float32)
        w1 = tl.load(WEIGHT + base + (256 + k64)[None, :], hm[:, None], other=0.0, cache_modifier=".cs").to(tl.float32)
        acc = tl.sum(w0 * x0[None, :], axis=1) + tl.sum(w1 * x1[None, :], axis=1)
        gate = tl.sigmoid(acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        val = tl.load(NORM + stream * H + h, hm, other=0.0).to(tl.float32)
        total += (gate * val).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + h, (total * (1.0 / S)).to(tl.bfloat16), hm)


@triton.autotune(configs=[_BPTR_4ACC_CONFIG], key=[])
@triton.jit(do_not_specialize=["ROWS"])
def _hc_gate_bptr_4acc(
    X,
    WEIGHT,
    NORM,
    OUT,
    ROWS,
    H: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVICT_NORM: tl.constexpr,
):
    """Pipelined 4-accumulator affine block-pointer projection gate for 4 streams."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(ROWS, BM)
    num_n = tl.cdiv(H, BH)
    if GROUP_M > 1:
        num_in_group = GROUP_M * num_n
        group_id = pid // num_in_group
        first_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % num_in_group) % group_size_m)
        pid_n = (pid % num_in_group) // group_size_m
    else:
        pid_m = pid // num_n
        pid_n = pid % num_n

    m0 = pid_m * BM
    h0 = pid_n * BH
    x_ptr = tl.make_block_ptr(X, (ROWS, R), (R, 1), (m0, 0), (BM, BK), (1, 0))
    w0_ptr = tl.make_block_ptr(WEIGHT + 0 * H * R, (R, H), (1, R), (0, h0), (BK, BH), (0, 1))
    w1_ptr = tl.make_block_ptr(WEIGHT + 1 * H * R, (R, H), (1, R), (0, h0), (BK, BH), (0, 1))
    w2_ptr = tl.make_block_ptr(WEIGHT + 2 * H * R, (R, H), (1, R), (0, h0), (BK, BH), (0, 1))
    w3_ptr = tl.make_block_ptr(WEIGHT + 3 * H * R, (R, H), (1, R), (0, h0), (BK, BH), (0, 1))
    acc0 = tl.zeros((BM, BH), dtype=tl.float32)
    acc1 = tl.zeros((BM, BH), dtype=tl.float32)
    acc2 = tl.zeros((BM, BH), dtype=tl.float32)
    acc3 = tl.zeros((BM, BH), dtype=tl.float32)
    for _ in range(R // BK):
        x = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
        acc0 = tl.dot(x, tl.load(w0_ptr, boundary_check=(0, 1), padding_option="zero"), acc0)
        acc1 = tl.dot(x, tl.load(w1_ptr, boundary_check=(0, 1), padding_option="zero"), acc1)
        acc2 = tl.dot(x, tl.load(w2_ptr, boundary_check=(0, 1), padding_option="zero"), acc2)
        acc3 = tl.dot(x, tl.load(w3_ptr, boundary_check=(0, 1), padding_option="zero"), acc3)
        x_ptr = tl.advance(x_ptr, (0, BK))
        w0_ptr = tl.advance(w0_ptr, (BK, 0))
        w1_ptr = tl.advance(w1_ptr, (BK, 0))
        w2_ptr = tl.advance(w2_ptr, (BK, 0))
        w3_ptr = tl.advance(w3_ptr, (BK, 0))

    m = (m0 + tl.arange(0, BM)).to(tl.int64)
    h = h0 + tl.arange(0, BH)
    mask = (m[:, None] < ROWS) & (h[None, :] < H)
    m_base = m[:, None] * (S * H)
    if EVICT_NORM:
        g0 = tl.sigmoid(acc0.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total = (g0 * tl.load(NORM + m_base + (0 * H + h)[None, :], mask, other=0.0, eviction_policy="evict_first").to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g1 = tl.sigmoid(acc1.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total += (g1 * tl.load(NORM + m_base + (1 * H + h)[None, :], mask, other=0.0, eviction_policy="evict_first").to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g2 = tl.sigmoid(acc2.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total += (g2 * tl.load(NORM + m_base + (2 * H + h)[None, :], mask, other=0.0, eviction_policy="evict_first").to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g3 = tl.sigmoid(acc3.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total += (g3 * tl.load(NORM + m_base + (3 * H + h)[None, :], mask, other=0.0, eviction_policy="evict_first").to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        g0 = tl.sigmoid(acc0.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total = (g0 * tl.load(NORM + m_base + (0 * H + h)[None, :], mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g1 = tl.sigmoid(acc1.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total += (g1 * tl.load(NORM + m_base + (1 * H + h)[None, :], mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g2 = tl.sigmoid(acc2.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total += (g2 * tl.load(NORM + m_base + (2 * H + h)[None, :], mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g3 = tl.sigmoid(acc3.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        total += (g3 * tl.load(NORM + m_base + (3 * H + h)[None, :], mask, other=0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + m[:, None] * H + h[None, :], (total * (1.0 / S)).to(tl.bfloat16), mask)


def _is_sm120(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (12, 0)


class PreparedHyperconnectionGate:
    def __init__(self, source, weight, normalized, output, workspace, *, session, name,
                 projected_down=None):
        rows, rank = source.shape
        hidden = output.shape[1]
        streams, remainder = divmod(normalized.shape[1], hidden)
        if remainder:
            raise ValueError("normalized width must contain complete residual streams")
        self.weight, self.output, self.workspace = weight, output, workspace
        self.rows, self.rank, self.hidden, self.streams = rows, rank, hidden, streams
        self.plan = None
        self.fuses_scaled_silu = False
        self.implementation = None
        self._grid = None
        self._options = None

        triton_eligible = (
            _is_sm120(source.device)
            and streams == 4
            and rank == 320
            and source.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and normalized.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and weight.is_contiguous()
            and normalized.is_contiguous()
            and output.is_contiguous()
        )
        fused_source_ok = (
            projected_down is not None
            and projected_down.shape == (rows, rank)
            and projected_down.dtype == torch.bfloat16
            and projected_down.device == source.device
            and projected_down.stride(1) == 1
        )
        if triton_eligible and rows == 1 and hidden % 2 == 0 and fused_source_ok:
            self.fuses_scaled_silu = True
            self.implementation = "triton_m1_gemv"
            bh = 4
            self._grid = (triton.cdiv(hidden, bh), 1, 1)
            self._options = {"H": hidden, "S": streams, "R": rank, "BH": bh}
            warmup_triton(
                _hc_gate_m1_gemv.fn,
                (projected_down, weight, normalized, output),
                self._grid,
                num_warps=_M1_GEMV_CONFIG.num_warps,
                num_stages=_M1_GEMV_CONFIG.num_stages,
                **self._options,
            )
            return
        if triton_eligible and rows >= 64 and hidden % 32 == 0 and source.is_contiguous():
            self.implementation = "triton_bptr_4acc"
            bm, bh, bk, gm = 64, 32, 32, 8
            self._grid = (triton.cdiv(rows, bm) * triton.cdiv(hidden, bh), 1, 1)
            self._options = {
                "H": hidden,
                "S": streams,
                "R": rank,
                "BM": bm,
                "BH": bh,
                "BK": bk,
                "GROUP_M": gm,
                "EVICT_NORM": rows >= 4096,
            }
            warmup_triton(
                _hc_gate_bptr_4acc.fn,
                (source, weight, normalized, output, rows),
                self._grid,
                num_warps=_BPTR_4ACC_CONFIG.num_warps,
                num_stages=_BPTR_4ACC_CONFIG.num_stages,
                **self._options,
            )
            return

        self.plan = hyperconnection_gate.plan(hyperconnection_gate.Query(
            max_rows=rows, rank=rank, hidden_size=hidden, streams=streams))

        def benchmark(state):
            inputs = benchmark_resource(session,
                ("hyperconnection_gate", str(source.device), rows, rank, hidden, streams),
                lambda: HyperconnectionGateInputs(source, normalized, output, workspace))
            return inputs.call(state, weight)

        declare(session, (self.plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(source, weight, normalized, output, workspace)),
            benchmark_call=benchmark,
        ),))

    def functional(self, *, session):
        self.output_shape, self.workspace_shape = tuple(self.output.shape), tuple(self.workspace.shape)
        self.output = self.workspace = None
        return self

    def __call__(self, source, normalized):
        output = torch.empty(self.output_shape, dtype=normalized.dtype, device=normalized.device)
        if self.implementation == "triton_m1_gemv":
            torch.library.wrap_triton(_hc_gate_m1_gemv)[self._grid](
                source, self.weight, normalized, output, **self._options,
            )
            return output
        if self.implementation == "triton_bptr_4acc":
            torch.library.wrap_triton(_hc_gate_bptr_4acc)[self._grid](
                source, self.weight, normalized, output, source.shape[0], **self._options,
            )
            return output
        workspace = torch.empty(self.workspace_shape, dtype=normalized.dtype, device=normalized.device)
        return hyperconnection_gate.run(source, self.weight, normalized, out=output,
                                       workspace=workspace, plan=self.plan)


class PreparedShardedHyperconnectionGate:
    """Reduce BF16 partial gate logits before sigmoid and the residual stream mean.

    Each rank owns matching columns of the gate weight and bottleneck activation.
    Normalized residual streams are replicated; partial and reduced logits require
    separate caller-owned BF16 storage. Collectives execute in the same order on
    every rank, including inactive rows.
    """

    def __init__(self, source, weight, normalized, output, partial_logits, logits, *,
                 runtime, coordinator, session, name):
        rows, lowrank = source.shape
        hidden = output.shape[1]
        streams, remainder = divmod(normalized.shape[1], hidden)
        if remainder or runtime is None:
            raise ValueError("sharded gates require complete residual streams and a TP collective")
        self.up = UnquantizedProjection(source, weight, partial_logits,
            session=session, name=f"{name}.up")
        self.reduce = PreparedChunkedAllReduce(partial_logits, logits, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.reduce")
        self.logits, self.output = logits, output
        caps = hc.Caps(device=source.device, max_tokens=rows, hidden_size=hidden,
                       streams=streams, lowrank=lowrank)
        self.plan = hc.plan(caps, invocation=FrozenMapping({"operation": "gate_mean"}))
        arguments = (normalized, logits, output)
        declare(session, (self.plan.request(name=f"{name}.mean",
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(*arguments)),
            benchmark_call=partial(hc_fixtures.benchmark_call, operation="gate_mean",
                                   arguments=arguments, kwargs={})),))

        def finalize():
            self.binding = hc.bind(self.plan, tokens=rows, normalized=normalized,
                                   bottleneck=source, block_input=output)

        register_finalizer(session, finalize)

    def functional(self, *, session):
        self.output_shape = tuple(self.output.shape)
        self.up.functional(session=session)
        self.reduce.functional(session=session)
        self.logits = self.output = None
        register_finalizer(session, lambda: setattr(self, "binding", None))
        return self

    def __call__(self, source, normalized):
        logits = self.reduce(self.up(source))
        output = torch.empty(self.output_shape, dtype=normalized.dtype, device=normalized.device)
        torch.ops.b12x.hyperconnection_gate_mean(normalized, logits, output, self.plan.handle)
        return output
