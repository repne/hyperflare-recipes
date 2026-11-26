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


@triton.autotune(configs=[_COMBINE_NORM_CONFIG], key=[])
@triton.jit(do_not_specialize=["INJ_STRIDE"])
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
        sg = tl.sigmoid(tl.load(SHARED_GATE + tok).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
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


def supports_triton_combine_norm(tokens: int, device: torch.device, *, hidden: int = 2560, streams: int = 4) -> bool:
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))
        and hidden == 2560
        and streams == 4
        and tokens >= 1024
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
        **options,
    )
    return combined, normalized
