# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120/SM121 token-major Triton hyperconnection combine-norm kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import warmup_triton

_COMBINE_NORM_CONFIG = triton.Config({}, num_warps=4, num_stages=2)
_TILED_DOWN_CONFIG = triton.Config({}, num_warps=8, num_stages=3)
_TILED_REDUCE_CONFIG = triton.Config({}, num_warps=4, num_stages=2)
_M1_SPLIT4_DOWN_CONFIG = triton.Config({}, num_warps=4, num_stages=2)
_M1_PARTIAL_GATE_CONFIG = triton.Config({}, num_warps=1, num_stages=2)


@triton.autotune(configs=[_COMBINE_NORM_CONFIG], key=[])
@triton.jit(do_not_specialize=["INJ_STRIDE", "SHARED_GATE_STRIDE"])
def _combine_norm_5x512(
    STATE,
    BLOCK,
    PROJ_SHARED,
    SHARED_GATE,
    INJ,
    GAIN,
    COMBINED,
    NORMALIZED,
    INJ_STRIDE,
    SHARED_GATE_STRIDE,
    EPS: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    EVICT_RES: tl.constexpr,
    FUSE_MLP: tl.constexpr,
):
    """Token-major 4-CTA-per-token combine_norm over five unmasked 512-element chunks.

    Scheduling ``pid = token * 4 + stream`` keeps ``block_output[token]`` hot in L2 across
    all four streams instead of sweeping all tokens of stream 0 before touching stream 1.
    When ``FUSE_MLP`` is enabled, computes ``routed + shared_projected * sigmoid(shared_gate)``
    with BF16 rounding in registers, avoiding materializing the intermediate MLP sum.
    """
    pid = tl.program_id(0).to(tl.int64)
    tok = pid // S
    stream = pid % S
    k = tl.arange(0, 512)
    b_base = tok * H
    s_base = (tok * S + stream) * H
    g_base = stream * H

    logit = tl.load(INJ + tok * INJ_STRIDE + stream).to(tl.float32)
    scale = 2.0 * tl.sigmoid(logit * (1.0 / S))

    if FUSE_MLP:
        sg = tl.sigmoid(tl.load(SHARED_GATE + tok * SHARED_GATE_STRIDE).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        b0 = (
            tl.load(BLOCK + b_base + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + b_base + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b1 = (
            tl.load(BLOCK + b_base + 512 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + b_base + 512 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b2 = (
            tl.load(BLOCK + b_base + 1024 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + b_base + 1024 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b3 = (
            tl.load(BLOCK + b_base + 1536 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + b_base + 1536 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b4 = (
            tl.load(BLOCK + b_base + 2048 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + b_base + 2048 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
    else:
        b0 = tl.load(BLOCK + b_base + k).to(tl.float32)
        b1 = tl.load(BLOCK + b_base + 512 + k).to(tl.float32)
        b2 = tl.load(BLOCK + b_base + 1024 + k).to(tl.float32)
        b3 = tl.load(BLOCK + b_base + 1536 + k).to(tl.float32)
        b4 = tl.load(BLOCK + b_base + 2048 + k).to(tl.float32)

    if EVICT_RES:
        c0 = (tl.load(STATE + s_base + k, eviction_policy="evict_first").to(tl.float32) + scale * b0).to(tl.bfloat16)
        c1 = (tl.load(STATE + s_base + 512 + k, eviction_policy="evict_first").to(tl.float32) + scale * b1).to(tl.bfloat16)
        c2 = (tl.load(STATE + s_base + 1024 + k, eviction_policy="evict_first").to(tl.float32) + scale * b2).to(tl.bfloat16)
        c3 = (tl.load(STATE + s_base + 1536 + k, eviction_policy="evict_first").to(tl.float32) + scale * b3).to(tl.bfloat16)
        c4 = (tl.load(STATE + s_base + 2048 + k, eviction_policy="evict_first").to(tl.float32) + scale * b4).to(tl.bfloat16)
        tl.store(COMBINED + s_base + k, c0, eviction_policy="evict_first")
        tl.store(COMBINED + s_base + 512 + k, c1, eviction_policy="evict_first")
        tl.store(COMBINED + s_base + 1024 + k, c2, eviction_policy="evict_first")
        tl.store(COMBINED + s_base + 1536 + k, c3, eviction_policy="evict_first")
        tl.store(COMBINED + s_base + 2048 + k, c4, eviction_policy="evict_first")
    else:
        c0 = (tl.load(STATE + s_base + k).to(tl.float32) + scale * b0).to(tl.bfloat16)
        c1 = (tl.load(STATE + s_base + 512 + k).to(tl.float32) + scale * b1).to(tl.bfloat16)
        c2 = (tl.load(STATE + s_base + 1024 + k).to(tl.float32) + scale * b2).to(tl.bfloat16)
        c3 = (tl.load(STATE + s_base + 1536 + k).to(tl.float32) + scale * b3).to(tl.bfloat16)
        c4 = (tl.load(STATE + s_base + 2048 + k).to(tl.float32) + scale * b4).to(tl.bfloat16)
        tl.store(COMBINED + s_base + k, c0)
        tl.store(COMBINED + s_base + 512 + k, c1)
        tl.store(COMBINED + s_base + 1024 + k, c2)
        tl.store(COMBINED + s_base + 1536 + k, c3)
        tl.store(COMBINED + s_base + 2048 + k, c4)

    f0, f1, f2, f3, f4 = (
        c0.to(tl.float32),
        c1.to(tl.float32),
        c2.to(tl.float32),
        c3.to(tl.float32),
        c4.to(tl.float32),
    )
    sq = tl.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4, axis=0)
    inv_rms = tl.rsqrt(sq * (1.0 / H) + EPS)

    tl.store(NORMALIZED + s_base + k, (f0 * inv_rms * tl.load(GAIN + g_base + k).to(tl.float32)).to(tl.bfloat16))
    tl.store(NORMALIZED + s_base + 512 + k, (f1 * inv_rms * tl.load(GAIN + g_base + 512 + k).to(tl.float32)).to(tl.bfloat16))
    tl.store(NORMALIZED + s_base + 1024 + k, (f2 * inv_rms * tl.load(GAIN + g_base + 1024 + k).to(tl.float32)).to(tl.bfloat16))
    tl.store(NORMALIZED + s_base + 1536 + k, (f3 * inv_rms * tl.load(GAIN + g_base + 1536 + k).to(tl.float32)).to(tl.bfloat16))
    tl.store(NORMALIZED + s_base + 2048 + k, (f4 * inv_rms * tl.load(GAIN + g_base + 2048 + k).to(tl.float32)).to(tl.bfloat16))


def supports_triton_combine_norm(
    tokens: int,
    device: torch.device,
    *,
    hidden: int = 2560,
    streams: int = 4,
    fuse_mlp: bool = False,
) -> bool:
    min_tokens = 2 if fuse_mlp else 1024
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
        and hidden == 2560
        and streams == 4
        and tokens >= min_tokens
    )


def warmup_combine_norm(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_gain: torch.Tensor,
    combined: torch.Tensor,
    normalized: torch.Tensor,
    *,
    eps: float,
    fuse_mlp: bool = False,
) -> tuple[tuple[int, int, int], dict]:
    tokens = state.shape[0]
    evict_res = 2048 < tokens <= 4096
    grid = (tokens * 4, 1, 1)
    options = {
        "EPS": float(eps),
        "H": 2560,
        "S": 4,
        "EVICT_RES": evict_res,
        "FUSE_MLP": bool(fuse_mlp),
    }
    shared_gate = injection_logits[:, :1]
    warmup_triton(
        _combine_norm_5x512.fn,
        (
            state,
            block_output,
            block_output,
            shared_gate,
            injection_logits,
            norm_gain,
            combined,
            normalized,
            injection_logits.stride(0),
            shared_gate.stride(0),
        ),
        grid,
        num_warps=_COMBINE_NORM_CONFIG.num_warps,
        num_stages=_COMBINE_NORM_CONFIG.num_stages,
        **options,
    )
    return grid, options


def launch_combine_norm(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_gain: torch.Tensor,
    *,
    grid: tuple[int, int, int],
    options: dict,
    shared_projected: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.empty_like(state)
    normalized = torch.empty_like(state)
    proj = block_output if shared_projected is None else shared_projected
    gate = injection_logits if shared_gate is None else shared_gate
    torch.library.wrap_triton(_combine_norm_5x512)[grid](
        state,
        block_output,
        proj,
        gate,
        injection_logits,
        norm_gain,
        combined,
        normalized,
        injection_logits.stride(0),
        gate.stride(0),
        **options,
    )
    return combined, normalized


@triton.autotune(configs=[_M1_SPLIT4_DOWN_CONFIG], key=[])
@triton.jit
def _combine_norm_down_m1_split4(
    STATE,
    BLOCK,
    PROJ_SHARED,
    SHARED_GATE,
    INJ,
    GAIN,
    DOWN_W,
    COMBINED,
    NORMALIZED,
    PARTIAL,
    EPS: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    BR: tl.constexpr,
    FUSE_MLP: tl.constexpr,
):
    """Single-token split-4-stream fused combine_norm + down projection.

    Launches ``(cdiv(OUT_ROWS, BR), 4)`` programs where each program processes one stream
    across five unmasked 512-element chunks. Programs ``(0, 0..3)`` write ``COMBINED`` and
    ``NORMALIZED``; all programs write deterministic FP32 stream partials ``[4, OUT_ROWS]``
    that ``_hc_gate_m1_from_partial`` sums before applying scaled SiLU and gate projection.
    """
    pid_r = tl.program_id(0)
    stream = tl.program_id(1)
    r0 = pid_r * BR
    k = tl.arange(0, 512)
    s_base = stream * H

    logit = tl.load(INJ + stream).to(tl.float32)
    scale = 2.0 * tl.sigmoid(logit * (1.0 / S))

    if FUSE_MLP:
        sg = tl.sigmoid(tl.load(SHARED_GATE).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        b0 = (
            tl.load(BLOCK + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b1 = (
            tl.load(BLOCK + 512 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + 512 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b2 = (
            tl.load(BLOCK + 1024 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + 1024 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b3 = (
            tl.load(BLOCK + 1536 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + 1536 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        b4 = (
            tl.load(BLOCK + 2048 + k).to(tl.float32)
            + (tl.load(PROJ_SHARED + 2048 + k).to(tl.float32) * sg).to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
    else:
        b0 = tl.load(BLOCK + k).to(tl.float32)
        b1 = tl.load(BLOCK + 512 + k).to(tl.float32)
        b2 = tl.load(BLOCK + 1024 + k).to(tl.float32)
        b3 = tl.load(BLOCK + 1536 + k).to(tl.float32)
        b4 = tl.load(BLOCK + 2048 + k).to(tl.float32)

    c0 = (tl.load(STATE + s_base + k).to(tl.float32) + scale * b0).to(tl.bfloat16)
    c1 = (tl.load(STATE + s_base + 512 + k).to(tl.float32) + scale * b1).to(tl.bfloat16)
    c2 = (tl.load(STATE + s_base + 1024 + k).to(tl.float32) + scale * b2).to(tl.bfloat16)
    c3 = (tl.load(STATE + s_base + 1536 + k).to(tl.float32) + scale * b3).to(tl.bfloat16)
    c4 = (tl.load(STATE + s_base + 2048 + k).to(tl.float32) + scale * b4).to(tl.bfloat16)

    f0, f1, f2, f3, f4 = (
        c0.to(tl.float32),
        c1.to(tl.float32),
        c2.to(tl.float32),
        c3.to(tl.float32),
        c4.to(tl.float32),
    )
    inv_rms = tl.rsqrt(tl.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4, axis=0) * (1.0 / H) + EPS)

    n0 = (f0 * inv_rms * tl.load(GAIN + s_base + k).to(tl.float32)).to(tl.bfloat16)
    n1 = (f1 * inv_rms * tl.load(GAIN + s_base + 512 + k).to(tl.float32)).to(tl.bfloat16)
    n2 = (f2 * inv_rms * tl.load(GAIN + s_base + 1024 + k).to(tl.float32)).to(tl.bfloat16)
    n3 = (f3 * inv_rms * tl.load(GAIN + s_base + 1536 + k).to(tl.float32)).to(tl.bfloat16)
    n4 = (f4 * inv_rms * tl.load(GAIN + s_base + 2048 + k).to(tl.float32)).to(tl.bfloat16)

    if pid_r == 0:
        tl.store(COMBINED + s_base + k, c0)
        tl.store(COMBINED + s_base + 512 + k, c1)
        tl.store(COMBINED + s_base + 1024 + k, c2)
        tl.store(COMBINED + s_base + 1536 + k, c3)
        tl.store(COMBINED + s_base + 2048 + k, c4)
        tl.store(NORMALIZED + s_base + k, n0)
        tl.store(NORMALIZED + s_base + 512 + k, n1)
        tl.store(NORMALIZED + s_base + 1024 + k, n2)
        tl.store(NORMALIZED + s_base + 1536 + k, n3)
        tl.store(NORMALIZED + s_base + 2048 + k, n4)

    nf0, nf1, nf2, nf3, nf4 = (
        n0.to(tl.float32),
        n1.to(tl.float32),
        n2.to(tl.float32),
        n3.to(tl.float32),
        n4.to(tl.float32),
    )
    rs = r0 + tl.arange(0, BR)
    rm = rs < OUT_ROWS
    wbase = rs.to(tl.int64)[:, None] * (S * H) + s_base
    acc = (
        nf0[None, :]
        * tl.load(DOWN_W + wbase + k[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cs").to(tl.float32)
        + nf1[None, :]
        * tl.load(DOWN_W + wbase + 512 + k[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cs").to(tl.float32)
        + nf2[None, :]
        * tl.load(DOWN_W + wbase + 1024 + k[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cs").to(tl.float32)
        + nf3[None, :]
        * tl.load(DOWN_W + wbase + 1536 + k[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cs").to(tl.float32)
        + nf4[None, :]
        * tl.load(DOWN_W + wbase + 2048 + k[None, :], mask=rm[:, None], other=0.0, cache_modifier=".cs").to(tl.float32)
    )
    tl.store(PARTIAL + stream * OUT_ROWS + rs, tl.sum(acc, axis=1), mask=rm)


@triton.autotune(configs=[_M1_PARTIAL_GATE_CONFIG], key=[])
@triton.jit
def _hc_gate_m1_from_partial(
    PARTIAL,
    WEIGHT,
    NORM,
    OUT,
    INJ_OUT,
    H: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    BH: tl.constexpr,
):
    """Sum deterministic FP32 stream partials, emit injection logits, and run M=1 gate GEMV."""
    pid = tl.program_id(0)
    if pid == 0 and OUT_ROWS > R:
        j = tl.arange(0, 4)
        inj = (
            tl.load(PARTIAL + 0 * OUT_ROWS + R + j)
            + tl.load(PARTIAL + 1 * OUT_ROWS + R + j)
            + tl.load(PARTIAL + 2 * OUT_ROWS + R + j)
            + tl.load(PARTIAL + 3 * OUT_ROWS + R + j)
        ).to(tl.bfloat16)
        tl.store(INJ_OUT + j, inj)

    h = pid * BH + tl.arange(0, BH)
    hm = h < H
    k256 = tl.arange(0, 256)
    k64 = tl.arange(0, 64)
    r0 = (
        tl.load(PARTIAL + 0 * OUT_ROWS + k256)
        + tl.load(PARTIAL + 1 * OUT_ROWS + k256)
        + tl.load(PARTIAL + 2 * OUT_ROWS + k256)
        + tl.load(PARTIAL + 3 * OUT_ROWS + k256)
    ).to(tl.bfloat16).to(tl.float32)
    s0 = (r0 * (1.0 / S)).to(tl.bfloat16).to(tl.float32)
    x0 = (s0 * tl.sigmoid(s0)).to(tl.bfloat16).to(tl.float32)
    r1 = (
        tl.load(PARTIAL + 0 * OUT_ROWS + 256 + k64)
        + tl.load(PARTIAL + 1 * OUT_ROWS + 256 + k64)
        + tl.load(PARTIAL + 2 * OUT_ROWS + 256 + k64)
        + tl.load(PARTIAL + 3 * OUT_ROWS + 256 + k64)
    ).to(tl.bfloat16).to(tl.float32)
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


def supports_m1_combine_mix(
    tokens: int,
    following_tokens: int,
    following_lowrank: int,
    device: torch.device,
    *,
    following_gate_impl: str | None,
) -> bool:
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
        and tokens == 1
        and following_tokens == 1
        and following_lowrank == 320
        and following_gate_impl == "triton_m1_gemv"
    )


def warmup_m1_combine_mix(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_gain: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    combined: torch.Tensor,
    normalized: torch.Tensor,
    mixed_out: torch.Tensor,
    *,
    eps: float,
    has_next_injection: bool,
    fuse_mlp: bool = False,
) -> tuple[tuple[int, int, int], dict, tuple[int, int, int], dict, int]:
    out_rows = 324 if has_next_injection else 320
    br = 4
    cn_grid = (triton.cdiv(out_rows, br), 4, 1)
    cn_options = {
        "EPS": float(eps),
        "H": 2560,
        "S": 4,
        "OUT_ROWS": out_rows,
        "BR": br,
        "FUSE_MLP": bool(fuse_mlp),
    }
    partial = torch.empty((4, out_rows), dtype=torch.float32, device=state.device)
    inj_out = torch.empty((1, 4), dtype=torch.bfloat16, device=state.device)
    shared_gate = injection_logits[:, :1]
    warmup_triton(
        _combine_norm_down_m1_split4.fn,
        (state, block_output, block_output, shared_gate, injection_logits, norm_gain, down_weight, combined, normalized, partial),
        cn_grid,
        num_warps=_M1_SPLIT4_DOWN_CONFIG.num_warps,
        num_stages=_M1_SPLIT4_DOWN_CONFIG.num_stages,
        **cn_options,
    )
    gate_grid = (2560 // 4, 1, 1)
    gate_options = {"H": 2560, "S": 4, "R": 320, "OUT_ROWS": out_rows, "BH": 4}
    warmup_triton(
        _hc_gate_m1_from_partial.fn,
        (partial, up_weight, normalized, mixed_out, inj_out),
        gate_grid,
        num_warps=_M1_PARTIAL_GATE_CONFIG.num_warps,
        num_stages=_M1_PARTIAL_GATE_CONFIG.num_stages,
        **gate_options,
    )
    return cn_grid, cn_options, gate_grid, gate_options, out_rows


def launch_m1_combine_mix(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_gain: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    spec: tuple[tuple[int, int, int], dict, tuple[int, int, int], dict, int],
    shared_projected: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cn_grid, cn_options, gate_grid, gate_options, out_rows = spec
    device = state.device
    combined = torch.empty_like(state)
    normalized = torch.empty_like(state)
    partial = torch.empty((4, out_rows), dtype=torch.float32, device=device)
    mixed = torch.empty((1, 2560), dtype=torch.bfloat16, device=device)
    next_injection = torch.empty((1, 4), dtype=torch.bfloat16, device=device) if out_rows > 320 else None
    proj = block_output if shared_projected is None else shared_projected
    gate = injection_logits if shared_gate is None else shared_gate
    torch.library.wrap_triton(_combine_norm_down_m1_split4)[cn_grid](
        state,
        block_output,
        proj,
        gate,
        injection_logits,
        norm_gain,
        down_weight,
        combined,
        normalized,
        partial,
        **cn_options,
    )
    torch.library.wrap_triton(_hc_gate_m1_from_partial)[gate_grid](
        partial,
        up_weight,
        normalized,
        mixed,
        mixed if next_injection is None else next_injection,
        **gate_options,
    )
    return combined, mixed, next_injection


@triton.autotune(configs=[_TILED_DOWN_CONFIG], key=[])
@triton.jit(do_not_specialize=["ROWS", "INJ_STRIDE", "SHARED_GATE_STRIDE"])
def _combine_norm_down_tiled_split4(
    STATE,
    BLOCK,
    PROJ_SHARED,
    SHARED_GATE,
    INJ,
    GAIN,
    DOWN_W,
    COMBINED,
    NORMALIZED,
    PARTIAL,
    ROWS,
    INJ_STRIDE,
    SHARED_GATE_STRIDE,
    EPS: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    DOWN_ROWS: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    FUSE_MLP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BK_P1: tl.constexpr,
):
    """Fused large-batch combine_norm + hyperconnection down projection over one stream per CTA.

    Each program handles ``BM`` tokens of one residual stream in reverse token-major order
    (``pid_m = (num_m - 1) - pid // 4``, ``stream = pid % 4``) so ``BLOCK`` stays hot in L2
    across streams, ``COMBINED`` and ``NORMALIZED`` stay in L2 between passes, and low-index
    rows of ``NORMALIZED`` remain L2-resident for the immediately following hyperconnection
    gate, avoiding the DRAM re-read of ``NORMALIZED`` when ``tokens * 10240 * 2`` exceeds L2.
    Writes deterministic FP32 stream partials ``[ROWS, 4, OUT_ROWS]`` consumed by
    ``_reduce_down_partial_silu``.
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(ROWS, BM)
    # Process token tiles in reverse so low-index NORMALIZED rows stay warm in L2
    # for the immediately following hyperconnection gate, which scans from row 0.
    pid_m = (num_m - 1) - (pid // S)
    stream = pid % S
    m0 = pid_m * BM
    ms = m0 + tl.arange(0, BM)
    mm = ms < ROWS
    ms64 = ms.to(tl.int64)

    logit = tl.load(INJ + ms64 * INJ_STRIDE + stream, mask=mm, other=0.0).to(tl.float32)
    scale = 2.0 * tl.sigmoid(logit * (1.0 / S))
    if FUSE_MLP:
        sg = (
            tl.sigmoid(tl.load(SHARED_GATE + ms64 * SHARED_GATE_STRIDE, mask=mm, other=0.0).to(tl.float32))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    else:
        sg = tl.zeros((BM,), dtype=tl.float32)
    s_row = (ms64 * S + stream) * H
    g_base = stream * H

    k1 = tl.arange(0, BK_P1)
    sq = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(0, H, BK_P1):
        kk = k0 + k1
        if FUSE_MLP:
            b = (
                tl.load(BLOCK + ms64[:, None] * H + kk[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
                + (
                    tl.load(PROJ_SHARED + ms64[:, None] * H + kk[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
                    * sg[:, None]
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            ).to(tl.bfloat16).to(tl.float32)
        else:
            b = tl.load(BLOCK + ms64[:, None] * H + kk[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        st = tl.load(
            STATE + s_row[:, None] + kk[None, :],
            mask=mm[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        c = (st + scale[:, None] * b).to(tl.bfloat16)
        tl.store(COMBINED + s_row[:, None] + kk[None, :], c, mask=mm[:, None])
        cf = c.to(tl.float32)
        sq += tl.sum(cf * cf, axis=1)
    inv_rms = tl.rsqrt(sq * (1.0 / H) + EPS)

    pbase = (ms64 * S + stream)[:, None] * OUT_ROWS

    # Pass 2: normalize from L2-hot COMBINED, write NORMALIZED, and compute columns 0..256.
    c_ptr = tl.make_block_ptr(COMBINED + stream * H, (ROWS, H), (S * H, 1), (m0, 0), (BM, BK), (1, 0))
    n_store_ptr = tl.make_block_ptr(NORMALIZED + stream * H, (ROWS, H), (S * H, 1), (m0, 0), (BM, BK), (1, 0))
    w0_ptr = tl.make_block_ptr(DOWN_W, (S * H, DOWN_ROWS), (1, S * H), (stream * H, 0), (BK, 256), (0, 1))
    acc0 = tl.zeros((BM, 256), dtype=tl.float32)
    k = tl.arange(0, BK)
    for k0 in range(0, H, BK):
        c = tl.load(c_ptr, boundary_check=(0,), padding_option="zero", eviction_policy="evict_first").to(tl.float32)
        g = tl.load(GAIN + g_base + k0 + k).to(tl.float32)
        n = (c * inv_rms[:, None] * g[None, :]).to(tl.bfloat16)
        tl.store(n_store_ptr, n, boundary_check=(0,), eviction_policy="evict_last")
        acc0 = tl.dot(n, tl.load(w0_ptr), acc0)
        c_ptr = tl.advance(c_ptr, (0, BK))
        n_store_ptr = tl.advance(n_store_ptr, (0, BK))
        w0_ptr = tl.advance(w0_ptr, (BK, 0))
    tl.store(PARTIAL + pbase + tl.arange(0, 256)[None, :], acc0, mask=mm[:, None])

    # Pass 3: read L2-hot NORMALIZED (keeping it resident for the following gate) and compute columns 256..OUT_ROWS.
    n_ptr = tl.make_block_ptr(NORMALIZED + stream * H, (ROWS, H), (S * H, 1), (m0, 0), (BM, BK), (1, 0))
    w1_ptr = tl.make_block_ptr(DOWN_W, (S * H, DOWN_ROWS), (1, S * H), (stream * H, 256), (BK, 128), (0, 1))
    acc1 = tl.zeros((BM, 128), dtype=tl.float32)
    for _ in range(0, H, BK):
        n = tl.load(n_ptr, boundary_check=(0,), padding_option="zero", eviction_policy="evict_last")
        acc1 = tl.dot(n, tl.load(w1_ptr, boundary_check=(1,), padding_option="zero"), acc1)
        n_ptr = tl.advance(n_ptr, (0, BK))
        w1_ptr = tl.advance(w1_ptr, (BK, 0))
    c1 = tl.arange(0, 128)
    tail_cols: tl.constexpr = OUT_ROWS - 256
    tl.store(PARTIAL + pbase + (256 + c1)[None, :], acc1, mask=mm[:, None] & (c1[None, :] < tail_cols))


@triton.autotune(configs=[_TILED_REDUCE_CONFIG], key=[])
@triton.jit(do_not_specialize=["ROWS"])
def _reduce_down_partial_silu(
    PARTIAL,
    BOTTLENECK,
    INJ_OUT,
    ROWS,
    OUT_ROWS: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    HAS_NEXT_INJ: tl.constexpr,
    BM: tl.constexpr,
):
    """Sum token-major FP32 stream partials [ROWS, 4, OUT_ROWS] and apply BF16-rounded ScaledSilu."""
    pid = tl.program_id(0)
    ms = pid * BM + tl.arange(0, BM)
    mm = ms < ROWS
    ms64 = ms.to(tl.int64)
    c0 = tl.arange(0, 256)
    c1 = 256 + tl.arange(0, 64)
    r0 = (ms64 * S + 0)[:, None] * OUT_ROWS
    r1 = (ms64 * S + 1)[:, None] * OUT_ROWS
    r2 = (ms64 * S + 2)[:, None] * OUT_ROWS
    r3 = (ms64 * S + 3)[:, None] * OUT_ROWS
    a0 = (
        tl.load(PARTIAL + r0 + c0[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
        + tl.load(PARTIAL + r1 + c0[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
        + tl.load(PARTIAL + r2 + c0[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
        + tl.load(PARTIAL + r3 + c0[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
    )
    s0 = (a0.to(tl.bfloat16).to(tl.float32) * (1.0 / S)).to(tl.bfloat16).to(tl.float32)
    tl.store(BOTTLENECK + ms64[:, None] * R + c0[None, :], (s0 * tl.sigmoid(s0)).to(tl.bfloat16), mask=mm[:, None])
    a1 = (
        tl.load(PARTIAL + r0 + c1[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
        + tl.load(PARTIAL + r1 + c1[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
        + tl.load(PARTIAL + r2 + c1[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
        + tl.load(PARTIAL + r3 + c1[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
    )
    s1 = (a1.to(tl.bfloat16).to(tl.float32) * (1.0 / S)).to(tl.bfloat16).to(tl.float32)
    tl.store(BOTTLENECK + ms64[:, None] * R + c1[None, :], (s1 * tl.sigmoid(s1)).to(tl.bfloat16), mask=mm[:, None])
    if HAS_NEXT_INJ:
        j = tl.arange(0, 4)
        ai = (
            tl.load(PARTIAL + r0 + (R + j)[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
            + tl.load(PARTIAL + r1 + (R + j)[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
            + tl.load(PARTIAL + r2 + (R + j)[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
            + tl.load(PARTIAL + r3 + (R + j)[None, :], mask=mm[:, None], other=0.0, eviction_policy="evict_first")
        )
        tl.store(INJ_OUT + ms64[:, None] * 4 + j[None, :], ai.to(tl.bfloat16), mask=mm[:, None])


def supports_tiled_combine_down(
    tokens: int,
    following_tokens: int,
    following_lowrank: int,
    down_weight: torch.Tensor,
    device: torch.device,
    *,
    following_fuses_scaled_silu: bool,
) -> bool:
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
        and tokens > 4096
        and following_tokens == tokens
        and following_lowrank == 320
        and not following_fuses_scaled_silu
        and down_weight.dtype == torch.bfloat16
        and down_weight.is_contiguous()
        and down_weight.shape[1] == 10240
        and down_weight.shape[0] >= 320
    )


def warmup_tiled_combine_down(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_gain: torch.Tensor,
    down_weight: torch.Tensor,
    combined: torch.Tensor,
    normalized: torch.Tensor,
    bottleneck: torch.Tensor,
    *,
    eps: float,
    has_next_injection: bool,
    fuse_mlp: bool = False,
) -> tuple[tuple[int, int, int], dict, tuple[int, int, int], dict, int]:
    tokens = state.shape[0]
    out_rows = 324 if has_next_injection else 320
    bm = 64
    red_bm = 32
    cn_grid = (triton.cdiv(tokens, bm) * 4, 1, 1)
    cn_options = {
        "EPS": float(eps),
        "H": 2560,
        "S": 4,
        "DOWN_ROWS": down_weight.shape[0],
        "OUT_ROWS": out_rows,
        "FUSE_MLP": bool(fuse_mlp),
        "BM": bm,
        "BK": 64,
        "BK_P1": 128,
    }
    partial = torch.empty((tokens, 4, out_rows), dtype=torch.float32, device=state.device)
    inj_out = torch.empty((tokens, 4), dtype=torch.bfloat16, device=state.device)
    shared_gate = injection_logits[:, :1]
    warmup_triton(
        _combine_norm_down_tiled_split4.fn,
        (
            state,
            block_output,
            block_output,
            shared_gate,
            injection_logits,
            norm_gain,
            down_weight,
            combined,
            normalized,
            partial,
            tokens,
            injection_logits.stride(0),
            shared_gate.stride(0),
        ),
        cn_grid,
        num_warps=_TILED_DOWN_CONFIG.num_warps,
        num_stages=_TILED_DOWN_CONFIG.num_stages,
        **cn_options,
    )
    red_grid = (triton.cdiv(tokens, red_bm), 1, 1)
    red_options = {
        "OUT_ROWS": out_rows,
        "R": 320,
        "S": 4,
        "HAS_NEXT_INJ": bool(has_next_injection),
        "BM": red_bm,
    }
    warmup_triton(
        _reduce_down_partial_silu.fn,
        (partial, bottleneck, inj_out, tokens),
        red_grid,
        num_warps=_TILED_REDUCE_CONFIG.num_warps,
        num_stages=_TILED_REDUCE_CONFIG.num_stages,
        **red_options,
    )
    return cn_grid, cn_options, red_grid, red_options, out_rows


def launch_tiled_combine_down(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_gain: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    spec: tuple[tuple[int, int, int], dict, tuple[int, int, int], dict, int],
    shared_projected: torch.Tensor | None = None,
    shared_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cn_grid, cn_options, red_grid, red_options, out_rows = spec
    tokens = state.shape[0]
    device = state.device
    combined = torch.empty_like(state)
    normalized = torch.empty_like(state)
    partial = torch.empty((tokens, 4, out_rows), dtype=torch.float32, device=device)
    bottleneck = torch.empty((tokens, 320), dtype=torch.bfloat16, device=device)
    next_injection = torch.empty((tokens, 4), dtype=torch.bfloat16, device=device) if out_rows > 320 else None
    proj = block_output if shared_projected is None else shared_projected
    gate = injection_logits if shared_gate is None else shared_gate
    torch.library.wrap_triton(_combine_norm_down_tiled_split4)[cn_grid](
        state,
        block_output,
        proj,
        gate,
        injection_logits,
        norm_gain,
        down_weight,
        combined,
        normalized,
        partial,
        tokens,
        injection_logits.stride(0),
        gate.stride(0),
        **cn_options,
    )
    torch.library.wrap_triton(_reduce_down_partial_silu)[red_grid](
        partial,
        bottleneck,
        bottleneck if next_injection is None else next_injection,
        tokens,
        **red_options,
    )
    return combined, normalized, bottleneck, next_injection
