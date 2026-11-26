# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compact PLE row delivery and prepared n-gram residual contribution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import ple_embedding, short_conv
from b12x.sequence.short_conv.api import _pipeline as short_conv_out
from safetensors import safe_open

from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.recipes.qwen38.checkpoint import CheckpointManifest
from ignition.progress import phase
from .checkpoint_schema import PLE_PREFIX, TEXT_PREFIX
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.triton import warmup_triton
from ignition.preparation.tuning import declare, register_finalizer

PLE_LAYER_PREFIX = f"{TEXT_PREFIX}.layers.1.ple"
TABLE_ROWS = 320001536
ROWS_PER_PLANE = 2500012
TABLE_BYTES = TABLE_ROWS * 90
_PLE_KERNEL_CONFIG = triton.Config({}, num_warps=4, num_stages=2)
_PLE_MAX_INLINE_SEQS = 512


@triton.autotune(configs=[_PLE_KERNEL_CONFIG], key=[])
@triton.jit
def _ple_gate_norm_5x512(
    KEY,
    VALUE,
    SOURCE,
    W_KEY,
    W_QUERY,
    W_CONV,
    Q_START,
    CONV_SLOTS,
    NUM_SEQS,
    NUM_TOKENS,
    SIG_GATE,
    NORMALIZED,
    EPS_K: tl.constexpr,
    EPS_Q: tl.constexpr,
    EPS_C: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    EVICT_IO: tl.constexpr,
):
    """Fused PLE group-RMSNorm (key/query), BF16 dot-gate, value gating, conv-norm, and live mask.

    Writes compact ``SIG_GATE`` ``[tokens, 4]`` (using ``-1.0`` as the inactive-row sentinel)
    and ``NORMALIZED`` ``[tokens, 10240]``, avoiding materializing the intermediate ``gated``
    buffer before ``short_conv``.
    """
    pid = tl.program_id(0).to(tl.int64)
    tok = pid // 4
    stream = pid % 4
    k = tl.arange(0, 512)
    s_base = (tok * 4 + stream) * 2560
    g_base = stream * 2560

    s = tl.arange(0, SEQ_BLOCK)
    sm = s < MAX_SEQS
    num_seqs = tl.load(NUM_SEQS)
    num_tokens = tl.load(NUM_TOKENS)
    starts = tl.load(Q_START + s, mask=sm, other=0)
    ends = tl.load(Q_START + s + 1, mask=sm, other=0)
    slots = tl.load(CONV_SLOTS + s, mask=sm, other=-1)
    tok_i32 = tok.to(tl.int32)
    is_live = tl.sum(
        (
            (tok_i32 >= starts)
            & (tok_i32 < ends)
            & (s < num_seqs)
            & (slots >= 0)
            & (tok_i32 < num_tokens)
            & sm
        ).to(tl.int32),
        axis=0,
    ) > 0

    if not is_live:
        tl.store(SIG_GATE + tok * 4 + stream, tl.full((), -1.0, dtype=tl.bfloat16))
        z = tl.zeros((512,), dtype=tl.bfloat16)
        tl.store(NORMALIZED + s_base + k, z)
        tl.store(NORMALIZED + s_base + 512 + k, z)
        tl.store(NORMALIZED + s_base + 1024 + k, z)
        tl.store(NORMALIZED + s_base + 1536 + k, z)
        tl.store(NORMALIZED + s_base + 2048 + k, z)
        return

    if EVICT_IO:
        k0 = tl.load(KEY + s_base + k, eviction_policy="evict_first").to(tl.float32)
        k1 = tl.load(KEY + s_base + 512 + k, eviction_policy="evict_first").to(tl.float32)
        k2 = tl.load(KEY + s_base + 1024 + k, eviction_policy="evict_first").to(tl.float32)
        k3 = tl.load(KEY + s_base + 1536 + k, eviction_policy="evict_first").to(tl.float32)
        k4 = tl.load(KEY + s_base + 2048 + k, eviction_policy="evict_first").to(tl.float32)
    else:
        k0 = tl.load(KEY + s_base + k).to(tl.float32)
        k1 = tl.load(KEY + s_base + 512 + k).to(tl.float32)
        k2 = tl.load(KEY + s_base + 1024 + k).to(tl.float32)
        k3 = tl.load(KEY + s_base + 1536 + k).to(tl.float32)
        k4 = tl.load(KEY + s_base + 2048 + k).to(tl.float32)
    inv_k = tl.rsqrt(tl.sum(k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3 + k4 * k4, axis=0) * (1.0 / 2560.0) + EPS_K)
    kn0 = (k0 * inv_k * tl.load(W_KEY + g_base + k)).to(tl.bfloat16).to(tl.float32)
    kn1 = (k1 * inv_k * tl.load(W_KEY + g_base + 512 + k)).to(tl.bfloat16).to(tl.float32)
    kn2 = (k2 * inv_k * tl.load(W_KEY + g_base + 1024 + k)).to(tl.bfloat16).to(tl.float32)
    kn3 = (k3 * inv_k * tl.load(W_KEY + g_base + 1536 + k)).to(tl.bfloat16).to(tl.float32)
    kn4 = (k4 * inv_k * tl.load(W_KEY + g_base + 2048 + k)).to(tl.bfloat16).to(tl.float32)

    if EVICT_IO:
        q0 = tl.load(SOURCE + s_base + k, eviction_policy="evict_first").to(tl.float32)
        q1 = tl.load(SOURCE + s_base + 512 + k, eviction_policy="evict_first").to(tl.float32)
        q2 = tl.load(SOURCE + s_base + 1024 + k, eviction_policy="evict_first").to(tl.float32)
        q3 = tl.load(SOURCE + s_base + 1536 + k, eviction_policy="evict_first").to(tl.float32)
        q4 = tl.load(SOURCE + s_base + 2048 + k, eviction_policy="evict_first").to(tl.float32)
    else:
        q0 = tl.load(SOURCE + s_base + k).to(tl.float32)
        q1 = tl.load(SOURCE + s_base + 512 + k).to(tl.float32)
        q2 = tl.load(SOURCE + s_base + 1024 + k).to(tl.float32)
        q3 = tl.load(SOURCE + s_base + 1536 + k).to(tl.float32)
        q4 = tl.load(SOURCE + s_base + 2048 + k).to(tl.float32)
    inv_q = tl.rsqrt(tl.sum(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4, axis=0) * (1.0 / 2560.0) + EPS_Q)
    qn0 = (q0 * inv_q * tl.load(W_QUERY + g_base + k)).to(tl.bfloat16).to(tl.float32)
    qn1 = (q1 * inv_q * tl.load(W_QUERY + g_base + 512 + k)).to(tl.bfloat16).to(tl.float32)
    qn2 = (q2 * inv_q * tl.load(W_QUERY + g_base + 1024 + k)).to(tl.bfloat16).to(tl.float32)
    qn3 = (q3 * inv_q * tl.load(W_QUERY + g_base + 1536 + k)).to(tl.bfloat16).to(tl.float32)
    qn4 = (q4 * inv_q * tl.load(W_QUERY + g_base + 2048 + k)).to(tl.bfloat16).to(tl.float32)

    dot = tl.sum(
        (kn0 * qn0).to(tl.bfloat16).to(tl.float32)
        + (kn1 * qn1).to(tl.bfloat16).to(tl.float32)
        + (kn2 * qn2).to(tl.bfloat16).to(tl.float32)
        + (kn3 * qn3).to(tl.bfloat16).to(tl.float32)
        + (kn4 * qn4).to(tl.bfloat16).to(tl.float32),
        axis=0,
    ).to(tl.bfloat16).to(tl.float32)
    gate = (dot / 50.59644256269407).to(tl.bfloat16).to(tl.float32)
    sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
    clamped = tl.maximum(tl.abs(gate), 1e-6).to(tl.bfloat16).to(tl.float32)
    gate_s = (tl.sqrt(clamped).to(tl.bfloat16).to(tl.float32) * sign).to(tl.bfloat16).to(tl.float32)
    sig_bf16 = tl.sigmoid(gate_s).to(tl.bfloat16)
    tl.store(SIG_GATE + tok * 4 + stream, sig_bf16)
    sig = sig_bf16.to(tl.float32)

    val_base = tok * 2560
    g0 = (sig * tl.load(VALUE + val_base + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    g1 = (sig * tl.load(VALUE + val_base + 512 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    g2 = (sig * tl.load(VALUE + val_base + 1024 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    g3 = (sig * tl.load(VALUE + val_base + 1536 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    g4 = (sig * tl.load(VALUE + val_base + 2048 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    inv_c = tl.rsqrt(tl.sum(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4, axis=0) * (1.0 / 2560.0) + EPS_C)
    n0 = (g0 * inv_c * tl.load(W_CONV + g_base + k)).to(tl.bfloat16)
    n1 = (g1 * inv_c * tl.load(W_CONV + g_base + 512 + k)).to(tl.bfloat16)
    n2 = (g2 * inv_c * tl.load(W_CONV + g_base + 1024 + k)).to(tl.bfloat16)
    n3 = (g3 * inv_c * tl.load(W_CONV + g_base + 1536 + k)).to(tl.bfloat16)
    n4 = (g4 * inv_c * tl.load(W_CONV + g_base + 2048 + k)).to(tl.bfloat16)

    if EVICT_IO:
        tl.store(NORMALIZED + s_base + k, n0, eviction_policy="evict_first")
        tl.store(NORMALIZED + s_base + 512 + k, n1, eviction_policy="evict_first")
        tl.store(NORMALIZED + s_base + 1024 + k, n2, eviction_policy="evict_first")
        tl.store(NORMALIZED + s_base + 1536 + k, n3, eviction_policy="evict_first")
        tl.store(NORMALIZED + s_base + 2048 + k, n4, eviction_policy="evict_first")
    else:
        tl.store(NORMALIZED + s_base + k, n0)
        tl.store(NORMALIZED + s_base + 512 + k, n1)
        tl.store(NORMALIZED + s_base + 1024 + k, n2)
        tl.store(NORMALIZED + s_base + 1536 + k, n3)
        tl.store(NORMALIZED + s_base + 2048 + k, n4)


@triton.autotune(configs=[_PLE_KERNEL_CONFIG], key=[])
@triton.jit
def _ple_combine_output_5x512(
    RESIDUAL,
    CONVOLVED,
    VALUE,
    SIG_GATE,
    NORM_GAIN,
    OUT,
    NEXT_NORM,
    ADD_RESIDUAL: tl.constexpr,
    EMIT_NORM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    EVICT_IO: tl.constexpr,
):
    """Combine rank-1 gated PLE values with short-conv output, optional residual add, and next norm."""
    pid = tl.program_id(0).to(tl.int64)
    tok = pid // 4
    stream = pid % 4
    k = tl.arange(0, 512)
    s_base = (tok * 4 + stream) * 2560
    val_base = tok * 2560
    g_base = stream * 2560

    sig = tl.load(SIG_GATE + tok * 4 + stream).to(tl.float32)
    if sig >= 0.0:
        g0 = (sig * tl.load(VALUE + val_base + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g1 = (sig * tl.load(VALUE + val_base + 512 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g2 = (sig * tl.load(VALUE + val_base + 1024 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g3 = (sig * tl.load(VALUE + val_base + 1536 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        g4 = (sig * tl.load(VALUE + val_base + 2048 + k).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    else:
        g0 = tl.zeros((512,), dtype=tl.float32)
        g1 = tl.zeros((512,), dtype=tl.float32)
        g2 = tl.zeros((512,), dtype=tl.float32)
        g3 = tl.zeros((512,), dtype=tl.float32)
        g4 = tl.zeros((512,), dtype=tl.float32)

    if EVICT_IO:
        c0 = tl.load(CONVOLVED + s_base + k, eviction_policy="evict_first").to(tl.float32)
        c1 = tl.load(CONVOLVED + s_base + 512 + k, eviction_policy="evict_first").to(tl.float32)
        c2 = tl.load(CONVOLVED + s_base + 1024 + k, eviction_policy="evict_first").to(tl.float32)
        c3 = tl.load(CONVOLVED + s_base + 1536 + k, eviction_policy="evict_first").to(tl.float32)
        c4 = tl.load(CONVOLVED + s_base + 2048 + k, eviction_policy="evict_first").to(tl.float32)
    else:
        c0 = tl.load(CONVOLVED + s_base + k).to(tl.float32)
        c1 = tl.load(CONVOLVED + s_base + 512 + k).to(tl.float32)
        c2 = tl.load(CONVOLVED + s_base + 1024 + k).to(tl.float32)
        c3 = tl.load(CONVOLVED + s_base + 1536 + k).to(tl.float32)
        c4 = tl.load(CONVOLVED + s_base + 2048 + k).to(tl.float32)

    p0 = (g0 + c0).to(tl.bfloat16)
    p1 = (g1 + c1).to(tl.bfloat16)
    p2 = (g2 + c2).to(tl.bfloat16)
    p3 = (g3 + c3).to(tl.bfloat16)
    p4 = (g4 + c4).to(tl.bfloat16)

    if ADD_RESIDUAL:
        if EVICT_IO:
            r0 = (tl.load(RESIDUAL + s_base + k, eviction_policy="evict_first").to(tl.float32) + p0.to(tl.float32)).to(tl.bfloat16)
            r1 = (tl.load(RESIDUAL + s_base + 512 + k, eviction_policy="evict_first").to(tl.float32) + p1.to(tl.float32)).to(tl.bfloat16)
            r2 = (tl.load(RESIDUAL + s_base + 1024 + k, eviction_policy="evict_first").to(tl.float32) + p2.to(tl.float32)).to(tl.bfloat16)
            r3 = (tl.load(RESIDUAL + s_base + 1536 + k, eviction_policy="evict_first").to(tl.float32) + p3.to(tl.float32)).to(tl.bfloat16)
            r4 = (tl.load(RESIDUAL + s_base + 2048 + k, eviction_policy="evict_first").to(tl.float32) + p4.to(tl.float32)).to(tl.bfloat16)
        else:
            r0 = (tl.load(RESIDUAL + s_base + k).to(tl.float32) + p0.to(tl.float32)).to(tl.bfloat16)
            r1 = (tl.load(RESIDUAL + s_base + 512 + k).to(tl.float32) + p1.to(tl.float32)).to(tl.bfloat16)
            r2 = (tl.load(RESIDUAL + s_base + 1024 + k).to(tl.float32) + p2.to(tl.float32)).to(tl.bfloat16)
            r3 = (tl.load(RESIDUAL + s_base + 1536 + k).to(tl.float32) + p3.to(tl.float32)).to(tl.bfloat16)
            r4 = (tl.load(RESIDUAL + s_base + 2048 + k).to(tl.float32) + p4.to(tl.float32)).to(tl.bfloat16)
    else:
        r0, r1, r2, r3, r4 = p0, p1, p2, p3, p4

    if EVICT_IO:
        tl.store(OUT + s_base + k, r0, eviction_policy="evict_first")
        tl.store(OUT + s_base + 512 + k, r1, eviction_policy="evict_first")
        tl.store(OUT + s_base + 1024 + k, r2, eviction_policy="evict_first")
        tl.store(OUT + s_base + 1536 + k, r3, eviction_policy="evict_first")
        tl.store(OUT + s_base + 2048 + k, r4, eviction_policy="evict_first")
    else:
        tl.store(OUT + s_base + k, r0)
        tl.store(OUT + s_base + 512 + k, r1)
        tl.store(OUT + s_base + 1024 + k, r2)
        tl.store(OUT + s_base + 1536 + k, r3)
        tl.store(OUT + s_base + 2048 + k, r4)

    if EMIT_NORM:
        f0 = r0.to(tl.float32)
        f1 = r1.to(tl.float32)
        f2 = r2.to(tl.float32)
        f3 = r3.to(tl.float32)
        f4 = r4.to(tl.float32)
        inv = tl.rsqrt(tl.sum(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4, axis=0) * (1.0 / 2560.0) + NORM_EPS)
        tl.store(NEXT_NORM + s_base + k, (f0 * inv * tl.load(NORM_GAIN + g_base + k)).to(tl.bfloat16))
        tl.store(NEXT_NORM + s_base + 512 + k, (f1 * inv * tl.load(NORM_GAIN + g_base + 512 + k)).to(tl.bfloat16))
        tl.store(NEXT_NORM + s_base + 1024 + k, (f2 * inv * tl.load(NORM_GAIN + g_base + 1024 + k)).to(tl.bfloat16))
        tl.store(NEXT_NORM + s_base + 1536 + k, (f3 * inv * tl.load(NORM_GAIN + g_base + 1536 + k)).to(tl.bfloat16))
        tl.store(NEXT_NORM + s_base + 2048 + k, (f4 * inv * tl.load(NORM_GAIN + g_base + 2048 + k)).to(tl.bfloat16))


def _read_small(manifest: CheckpointManifest, name: str) -> torch.Tensor:
    metadata = manifest.tensors[name]
    if metadata.nbytes > 1024:
        raise ValueError(f"{name} exceeds the PLE geometry/scalar read limit")
    from ignition.preparation.checkpoint_source import read_small_tensor, source_for

    if source_for(manifest.snapshot) is not None:
        return read_small_tensor(manifest, name, limit=1024)
    with safe_open(
        manifest.snapshot / metadata.shard, framework="pt", device="cpu"
    ) as source:
        view = source.get_slice(name)
        if (
            tuple(view.get_shape()) != metadata.shape
            or view.get_dtype() != metadata.dtype
        ):
            raise ValueError(f"{name} metadata changed after checkpoint inspection")
        return source.get_tensor(name).clone()


def _caps(config: dict, device, tokens: int, sequences: int, *, rank=0, world_size=1):
    eos = config["eos_token_id"]
    return ple_embedding.Caps(
        device=device,
        max_tokens=tokens,
        max_seqs=sequences,
        vocab_size=config["vocab_size"],
        eos_token_id=eos[0] if isinstance(eos, list) else eos,
        max_order=config["ngram_size"],
        heads_per_order=config["heads_per_ngram"],
        dense_layer_ordinal=0,
        base_table_size=config["ngram_vocab_size_base"],
        embedding_dim=config["ple_embed_dim"],
        tp_size=world_size,
        tp_rank=rank,
        table_alignment=config["make_ngram_vocab_size_divisible_by"],
        quant_mode="nvfp4_group16",
        table_memory="io_uring",
    )


def table_layout(manifest: CheckpointManifest, device, *, rank=0, world_size=1):
    """Validate exact checkpoint geometry without allocating its table."""
    config = json.loads((manifest.snapshot / "config.json").read_text())["text_config"]
    caps = _caps(config, torch.device(device), 1, 1, rank=rank, world_size=world_size)
    geometry = ple_embedding.compute_geometry(
        caps,
        prime_sizes=_read_small(manifest, f"{PLE_PREFIX}.ngram_heads_vocab_sizes"),
        table_offsets=_read_small(manifest, f"{PLE_PREFIX}.ngram_heads_offsets"),
        multipliers=_read_small(manifest, f"{PLE_PREFIX}.layer_multipliers"),
    )
    layout = ple_embedding.storage_layout(caps, geometry=geometry)
    if layout.weight_shape != (TABLE_ROWS // world_size, 80) or layout.weight_scale_shape != (
        TABLE_ROWS // world_size,
        10,
    ):
        raise ValueError(
            "PLE checkpoint requires 128 packed planes totaling 320001536 rows"
        )
    return config, layout


@dataclass
class PLETable:
    config: dict
    layout: ple_embedding.TableLayout
    geometry: ple_embedding.GeometryTensors
    scale: torch.Tensor
    weights: torch.Tensor
    scales: torch.Tensor
    hashes: torch.Tensor
    mailbox: object
    ready: object
    sources: tuple
    global_scale: float
    sequence: torch.Tensor | None = None
    timings: object = None

    @classmethod
    def load(cls, manifest, device, *, tokens, progress=None, host_allocator=None):
        from b12x.sequence._shared.disk_table import MappedHostAllocation
        from ignition.preparation.runtime_core import FileSource
        from ignition.preparation.persistent import persistent_empty

        device = torch.device(device)
        config, layout = table_layout(manifest, device)
        scale = _read_small(manifest, f"{PLE_PREFIX}.ngram_embedding.weight_scale_2")
        if scale.dtype != torch.float32 or scale.shape != (1,) or not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("PLE global scale must be a finite positive FP32 scalar")
        sources = []
        with phase(progress, "Prepare embedding tables") as stage:
            for suffix, columns, dtype in (("weight", 80, "U8"), ("weight_scale", 10, "F8_E4M3")):
                plane = []
                for shard in range(TABLE_ROWS // ROWS_PER_PLANE):
                    name = f"{PLE_PREFIX}.ngram_embedding.shard_{shard}.{suffix}"
                    metadata = manifest.tensors[name]
                    if metadata.shape != (ROWS_PER_PLANE, columns) or metadata.dtype != dtype:
                        raise ValueError(f"{name}: invalid PLE table geometry or dtype")
                    plane.append(FileSource(manifest.snapshot / metadata.shard, metadata.offset, metadata.nbytes))
                sources.append(tuple(plane))
            heads = layout.caps.head_count
            shape = (tokens * heads + 1,)
            mailbox = (MappedHostAllocation(shape, torch.int64, device) if host_allocator is None
                       else host_allocator("ple_requests", shape, torch.int64))
            try:
                mailbox.host_view.zero_()
                ready = torch.cuda.Event(external=True)
                ready.record(torch.cuda.current_stream(device))
                table = cls(config, layout, ple_embedding.allocate_geometry(layout.geometry, device=device,
                    allocate=persistent_empty),
                    persistent_empty(scale.shape, scale.dtype, device).copy_(scale),
                    torch.zeros((tokens * heads, 80), dtype=torch.uint8, device=device),
                    torch.zeros((tokens * heads, 10), dtype=torch.uint8, device=device),
                    torch.full((tokens, heads), -1, dtype=torch.int64, device=device), mailbox, ready, tuple(sources),
                    float(scale.item()))
            except BaseException:
                mailbox.close()
                raise
            stage.update(f"{TABLE_BYTES / (1 << 30):.1f} GiB table")
        return table

    def buffers(self):
        return dict(ple_weights=self.weights, ple_scales=self.scales, ple_hashes=self.hashes,
                    ple_requests=self.mailbox.device_view, **self.timings.buffers())

    def events(self):
        return {"ple_ready": self.ready.cuda_event}

    def export_metadata(self):
        """Describe compact hash geometry and checkpoint spans for native publication."""
        hashes = dict(method="EOS_XOR_MODULO", orders=[order for order in range(2, self.config["ngram_size"] + 1)
                      for _ in range(self.config["heads_per_ngram"])])
        for name in ("multipliers", "prime_sizes", "table_offsets"):
            target = {"prime_sizes": "primes", "table_offsets": "offsets"}.get(name, name)
            hashes[target] = list(getattr(self.layout.geometry, name))
        table = dict(layer=1, rows=TABLE_ROWS, embedding_width=160, scale_block=16,
            encoding="NVFP4_E4M3_GLOBAL", weight_buffer="ple_weights", scale_buffer="ple_scales",
            hash_buffer="ple_hashes", ready_event="ple_ready", request_buffer="ple_requests",
            global_scale=self.global_scale, hash=hashes)
        for name, sources in zip(("weights", "scales"), self.sources, strict=True):
            table[name] = [dict(path=str(source.path), offset=source.offset, nbytes=source.nbytes)
                           for source in sources]
        return dict(pad_token_id=self.layout.caps.eos_token_id,
                    compressed_vocabulary=self.config["vocab_size"], tables=[table])

    def close(self):
        self.mailbox.close()


@dataclass(frozen=True)
class PLEMetadata:
    """Fixed device inputs for packed n-gram lookup and convolution history.

    ``committed_history`` contains each request's preceding two accepted token
    IDs, padded with the model's EOS ID. The complete model graph gathers it
    from persistent request history before lookup and updates that history after
    consuming tokens. PLE lookup itself only reads these packed rows.
    """

    token_ids: torch.Tensor
    query_start_loc: torch.Tensor
    committed_history: torch.Tensor
    convolution_slots: torch.Tensor
    fresh: torch.Tensor
    accepted_tokens: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor

    @classmethod
    def allocate(
        cls, tokens: int, sequences: int, device, *, eos_token_id: int = 248044
    ) -> PLEMetadata:
        if any(type(value) is not int or value <= 0 for value in (tokens, sequences)):
            raise ValueError("PLE token and sequence capacities must be positive")
        return cls(
            torch.full((tokens,), eos_token_id, dtype=torch.int64, device=device),
            torch.zeros((sequences + 1,), dtype=torch.int32, device=device),
            torch.full((sequences, 2), eos_token_id, dtype=torch.int64, device=device),
            torch.full((sequences,), -1, dtype=torch.int64, device=device),
            torch.zeros((sequences,), dtype=torch.bool, device=device),
            torch.ones((sequences,), dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
        )


@dataclass(frozen=True)
class PLEScratch:
    embedding: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gated: torch.Tensor
    normalized: torch.Tensor
    convolved: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device) -> PLEScratch:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("PLE token capacity must be positive")
        return cls(
            *(
                torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
                for width in (2560, 10240, 2560, 10240, 10240, 10240, 10240)
            )
        )


def allocate_convolution_state(
    slots: int, device, *, speculative_tokens: int = 0
) -> torch.Tensor:
    if (
        type(slots) is not int
        or slots <= 0
        or type(speculative_tokens) is not int
        or speculative_tokens < 0
    ):
        raise ValueError(
            "PLE needs positive state capacity and nonnegative speculative capacity"
        )
    return torch.zeros(
        (slots, 10240, 9 + speculative_tokens), dtype=torch.bfloat16, device=device
    )


def _validate_geometry(module: torch.nn.Module) -> None:
    if (module.hidden_size, module.hc_count, module.short_conv_state_len) != (
        2560,
        4,
        9,
    ):
        raise ValueError("PLE geometry differs from Qwen3.8 Flash Next")
    if (
        tuple(module.conv1d.weight.shape) != (10240, 1, 4)
        or module.conv1d.bias is not None
    ):
        raise ValueError("PLE requires unbiased width-10240 kernel-4 convolution")
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = getattr(module, name)
        if (
            norm.weight.shape != (10240,)
            or norm.weight.dtype != torch.float32
            or norm.group_size != 2560
        ):
            raise ValueError("PLE requires exact FP32 zero-centered group norm weights")


class PreparedPLE:
    """Recorded PLE lookup, BF16 projections, canonical gates, and convolution.

    Canonical normalization and gate tensor expressions preserve FP32 norm
    weights and the BF16 rounding of each gate operation. Their intermediates
    stay inside the complete forward graph. Exported contribution and history
    storage is caller-owned and remains valid across shared graph-pool replays.
    """

    def __init__(
        self,
        module,
        projections,
        source,
        scratch: PLEScratch,
        convolution_state,
        metadata: PLEMetadata,
        table: PLETable,
        *,
        mode: str,
        session: PreparationSession,
        name: str,
        workspace=None,
        partitions=None,
        runtime=None,
        coordinator=None,
    ):
        _validate_geometry(module)
        if mode not in ("decode", "prefill"):
            raise ValueError("PLE graph mode must be decode or prefill")
        if (
            source.ndim not in (2, 3)
            or math.prod(source.shape[1:]) != 10240
            or source.dtype != torch.bfloat16
            or not source.is_contiguous()
        ):
            raise ValueError(
                "PLE source must be contiguous BF16 [tokens,10240] or [tokens,4,2560]"
            )
        if source.ndim == 3 and tuple(source.shape[1:]) != (4, 2560):
            raise ValueError("PLE residual streams must have shape [tokens,4,2560]")
        self.module, self.table, self.scratch, self.metadata = (
            module,
            table,
            scratch,
            metadata,
        )
        self.mode, self.source, self.state = mode, source, convolution_state
        self.tokens, self.device = source.shape[0], source.device
        if self.device != table.layout.caps.device or self.device.type != "cuda":
            raise ValueError("PLE table and graph operands must share a CUDA device")
        for field, width in zip(
            fields(scratch), (2560, 10240, 2560, 10240, 10240, 10240, 10240)
        ):
            tensor = getattr(scratch, field.name)
            if (
                tuple(tensor.shape) != (self.tokens, width)
                or tensor.dtype != torch.bfloat16
                or tensor.device != self.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"PLE scratch {field.name} has incompatible geometry")
        if (
            convolution_state.ndim != 3
            or convolution_state.shape[1] != 10240
            or convolution_state.shape[2] < 9
        ):
            raise ValueError("PLE convolution history requires [slots,10240,9+drafts]")
        if metadata.num_seqs.item() or metadata.num_tokens.item():
            raise ValueError("PLE preparation requires zero live counts")
        if set(projections) != {"key_proj", "value_proj"}:
            raise ValueError("PLE requires key_proj and value_proj BF16 weights")
        for parameter in (
            module.conv1d.weight,
            module.norm_key.weight,
            module.norm_query.weight,
            module.norm_conv.weight,
        ):
            if parameter.device != self.device:
                raise ValueError("canonical PLE parameters must share the graph device")
        sequences = metadata.query_start_loc.numel() - 1
        caps = _caps(table.config, self.device, self.tokens, sequences,
                     rank=table.layout.caps.tp_rank, world_size=table.layout.caps.tp_size)
        from ignition.components.ngram_delivery import PreparedPLELookup

        self.embedding_lookup = PreparedPLELookup(table, metadata, scratch.embedding, caps,
            session=session, name=f"{name}.embedding", runtime=runtime, coordinator=coordinator)
        self.embedding_output = scratch.embedding
        self.key_projection = UnquantizedProjection(
            self.embedding_output,
            projections["key_proj"],
            scratch.key,
            session=session,
            name=f"{name}.key",
        )
        self.value_projection = UnquantizedProjection(
            self.embedding_output,
            projections["value_proj"],
            scratch.value,
            session=session,
            name=f"{name}.value",
        )
        self.partitions = partitions
        if partitions is None:
            args = {
                "x": scratch.normalized,
                "conv_weight": module.conv1d.weight.squeeze(1),
                "query_start_loc": metadata.query_start_loc,
                "state_slot_ids": metadata.convolution_slots,
                "state_is_fresh": metadata.fresh,
                "num_accepted_tokens": metadata.accepted_tokens,
                "num_seqs": metadata.num_seqs,
                "num_tokens": metadata.num_tokens,
                "conv_state": convolution_state,
                "out": scratch.convolved,
            }
            self.convolution_plan = short_conv.plan(
                short_conv.Caps(
                    device=self.device,
                    mode=mode,
                    channels=10240,
                    kernel_size=4,
                    dilation=3,
                    max_tokens=self.tokens,
                    max_seqs=sequences,
                    max_state_slots=convolution_state.shape[0],
                    max_speculative_tokens=convolution_state.shape[2] - 9,
                ),
                invocation=short_conv.invocation_from_tensors(**args),
            )
            self.convolution_binding = self._prepare_binding(
                self.convolution_plan,
                args,
                session,
                f"{name}.convolution",
                short_conv.bind,
                bind_plan=True,
                attribute="convolution_binding",
                workspace=workspace,
            )
        else:
            self.convolutions = tuple(PreparedPLEConvolution(module, part.scratch, convolution_state,
                part.metadata, mode=part.mode, session=session, name=f"{name}.{part.mode}", workspace=workspace)
                for part in partitions.parts)
        self.token_rows = torch.arange(
            self.tokens, dtype=torch.int32, device=self.device
        ).view(-1, 1)
        self.sequence_rows = torch.arange(
            sequences, dtype=torch.int32, device=self.device
        ).view(1, -1)
        self._max_seqs = sequences
        self._fused_triton = sequences <= _PLE_MAX_INLINE_SEQS
        self._evict_io = self.tokens >= 2048
        self._ple_grid = (self.tokens * 4, 1, 1)
        self._w_key = self._w_query = self._w_conv = None
        self._next_norm_gain = None
        self._next_norm_eps = 0.0
        if self._fused_triton:
            self._w_key = persistent_copy((1.0 + module.norm_key.weight.detach().float()).contiguous())
            self._w_query = persistent_copy((1.0 + module.norm_query.weight.detach().float()).contiguous())
            self._w_conv = persistent_copy((1.0 + module.norm_conv.weight.detach().float()).contiguous())
            self._gate_options = {
                "EPS_K": float(module.norm_key.eps),
                "EPS_Q": float(module.norm_query.eps),
                "EPS_C": float(module.norm_conv.eps),
                "MAX_SEQS": sequences,
                "SEQ_BLOCK": triton.next_power_of_2(sequences),
                "EVICT_IO": self._evict_io,
            }
            sig_gate = torch.empty((self.tokens, 4), dtype=torch.bfloat16, device=self.device)
            warmup_triton(
                _ple_gate_norm_5x512.fn,
                (
                    scratch.key,
                    scratch.value,
                    source.view(self.tokens, 10240),
                    self._w_key,
                    self._w_query,
                    self._w_conv,
                    metadata.query_start_loc,
                    metadata.convolution_slots,
                    metadata.num_seqs,
                    metadata.num_tokens,
                    sig_gate,
                    scratch.normalized,
                ),
                self._ple_grid,
                num_warps=_PLE_KERNEL_CONFIG.num_warps,
                num_stages=_PLE_KERNEL_CONFIG.num_stages,
                **self._gate_options,
            )
            for add_residual in (False, True):
                warmup_triton(
                    _ple_combine_output_5x512.fn,
                    (
                        source.view(self.tokens, 10240),
                        scratch.convolved,
                        scratch.value,
                        sig_gate,
                        self._w_conv,
                        scratch.output,
                        scratch.gated,
                    ),
                    self._ple_grid,
                    num_warps=_PLE_KERNEL_CONFIG.num_warps,
                    num_stages=_PLE_KERNEL_CONFIG.num_stages,
                    ADD_RESIDUAL=add_residual,
                    EMIT_NORM=False,
                    NORM_EPS=0.0,
                    EVICT_IO=self._evict_io,
                )

    def prepare_residual_norm(self, norm_gain: torch.Tensor, *, eps: float) -> None:
        """Warm up fused residual addition + following layer grouped RMSNorm."""
        if not self._fused_triton or self.scratch is None:
            return
        self._next_norm_gain = norm_gain
        self._next_norm_eps = float(eps)
        sig_gate = torch.empty((self.tokens, 4), dtype=torch.bfloat16, device=self.device)
        warmup_triton(
            _ple_combine_output_5x512.fn,
            (
                self.source.view(self.tokens, 10240),
                self.scratch.convolved,
                self.scratch.value,
                sig_gate,
                norm_gain,
                self.scratch.output,
                self.scratch.gated,
            ),
            self._ple_grid,
            num_warps=_PLE_KERNEL_CONFIG.num_warps,
            num_stages=_PLE_KERNEL_CONFIG.num_stages,
            ADD_RESIDUAL=True,
            EMIT_NORM=True,
            NORM_EPS=self._next_norm_eps,
            EVICT_IO=self._evict_io,
        )

    def _prepare_binding(self, plan, args, session, name, public_bind, *,
                         attribute, bind_plan=False, workspace=None):

        def prepare_call(prepared):
            specs = plan.scratch_specs() if bind_plan else prepared.layout.scratch_specs()
            scratch = planned_scratch(specs, workspace)
            binding = prepared.bind(
                scratch=scratch, **args, **({"plan": plan} if bind_plan else {})
            )
            return PreparedCall(run=lambda: prepared.run(binding), owners=prime_owners(workspace, scratch, binding))

        declare(session, (plan.request(name=name, prepare_call=prepare_call),))

        def refresh():
            scratch = planned_scratch(plan.scratch_specs(), workspace)
            setattr(self, attribute, public_bind(plan, scratch=scratch, **args))
        setattr(self, attribute, None)

        register_finalizer(session, refresh, workspace=workspace)
        return getattr(self, attribute)

    def functional(self, *, session):
        self.embedding_lookup.functional(session=session)
        self.key_projection.functional(session=session)
        self.value_projection.functional(session=session)
        if self.partitions is not None:
            for convolution in self.convolutions:
                convolution.functional(session=session)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        if self.partitions is None:
            self.convolution_scratch = tuple((tuple(t.shape), tuple(t.stride()), t.dtype)
                for t in (self.convolution_binding.gathered_state, self.convolution_binding.request_ids))
            self.convolution_binding = None
        self.scratch = self.source = self.metadata = None
        self.embedding_output = self.partitions = None
        self.token_rows = self.sequence_rows = None

    def _run_convolve(self, normalized: torch.Tensor, metadata, partitions) -> torch.Tensor:
        if partitions is None:
            convolved = torch.empty_like(normalized)
            gathered, request_ids = (torch.empty_strided(shape, stride, dtype=dtype, device=normalized.device)
                for shape, stride, dtype in self.convolution_scratch)
            short_conv_out(normalized, self.module.conv1d.weight.squeeze(1), metadata.query_start_loc,
                metadata.convolution_slots, metadata.fresh, metadata.accepted_tokens, metadata.fresh,
                metadata.num_seqs, metadata.num_tokens, self.state, convolved, gathered, request_ids,
                None, self.convolution_plan.handle)
            return convolved
        values = tuple(convolution(part.gather(normalized), part.metadata)
            for part, convolution in zip(partitions.parts, self.convolutions, strict=True))
        return partitions.combine(values)

    def _run_fused(self, source: torch.Tensor, metadata, partitions, *, add_residual: bool, emit_norm: bool):
        embedding = self.embedding_lookup(source, metadata)
        key_raw = self.key_projection(embedding)
        value = self.value_projection(embedding)
        src_flat = source.view(self.tokens, 10240)
        sig_gate = torch.empty((self.tokens, 4), dtype=torch.bfloat16, device=source.device)
        normalized = torch.empty((self.tokens, 10240), dtype=torch.bfloat16, device=source.device)
        torch.library.wrap_triton(_ple_gate_norm_5x512)[self._ple_grid](
            key_raw,
            value,
            src_flat,
            self._w_key,
            self._w_query,
            self._w_conv,
            metadata.query_start_loc,
            metadata.convolution_slots,
            metadata.num_seqs,
            metadata.num_tokens,
            sig_gate,
            normalized,
            **self._gate_options,
        )
        convolved = self._run_convolve(normalized, metadata, partitions)
        out = torch.empty((self.tokens, 10240), dtype=torch.bfloat16, device=source.device)
        if emit_norm:
            next_norm = torch.empty((self.tokens, 10240), dtype=torch.bfloat16, device=source.device)
            gain = self._next_norm_gain
            norm_eps = self._next_norm_eps
        else:
            next_norm = torch.empty((1,), dtype=torch.bfloat16, device=source.device)
            gain = self._w_conv
            norm_eps = 0.0
        torch.library.wrap_triton(_ple_combine_output_5x512)[self._ple_grid](
            src_flat,
            convolved,
            value,
            sig_gate,
            gain,
            out,
            next_norm,
            ADD_RESIDUAL=add_residual,
            EMIT_NORM=emit_norm,
            NORM_EPS=norm_eps,
            EVICT_IO=self._evict_io,
        )
        return out, (next_norm if emit_norm else None)

    def add_to_residual(self, residual: torch.Tensor, metadata, partitions=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute PLE contribution, add it to ``residual``, and optionally emit layer-1 norm."""
        if getattr(self, "_fused_triton", False):
            return self._run_fused(
                residual, metadata, partitions,
                add_residual=True,
                emit_norm=self._next_norm_gain is not None,
            )
        residual.add_(self(residual, metadata, partitions))
        return residual, None

    def __call__(self, source: torch.Tensor, metadata, partitions=None) -> torch.Tensor:
        if getattr(self, "_fused_triton", False):
            out, _ = self._run_fused(source, metadata, partitions, add_residual=False, emit_norm=False)
            return out
        embedding = self.embedding_lookup(source, metadata)
        key = self.module.norm_key(self.key_projection(embedding)).view(self.tokens, 4, 2560)
        value = self.value_projection(embedding)
        query = self.module.norm_query(source.view(self.tokens, 10240)).view(self.tokens, 4, 2560)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(2560)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(1)).flatten(1)
        normalized = self.module.norm_conv(gated)
        token_rows = torch.arange(self.tokens, dtype=torch.int32, device=source.device).view(-1, 1)
        sequence_rows = torch.arange(metadata.query_start_loc.numel() - 1, dtype=torch.int32, device=source.device).view(1, -1)
        live = ((token_rows >= metadata.query_start_loc[:-1].view(1, -1))
            & (token_rows < metadata.query_start_loc[1:].view(1, -1))
            & (sequence_rows < metadata.num_seqs)
            & (metadata.convolution_slots.view(1, -1) >= 0)
            & (token_rows < metadata.num_tokens)).any(dim=1, keepdim=True)
        gated = gated.masked_fill(~live, 0)
        normalized = normalized.masked_fill(~live, 0)
        convolved = self._run_convolve(normalized, metadata, partitions)
        return gated + convolved



class PreparedPLEConvolution:
    """Phase-specific history update after the shared PLE projections and gates."""

    def __init__(self, module, scratch, state, metadata, *, mode, session, name, workspace):
        from hyperflare_recipes.components.recurrent import prepare_sequence_binding

        caps = short_conv.Caps(device=scratch.normalized.device, mode=mode, channels=10240,
            kernel_size=4, dilation=3, max_tokens=scratch.normalized.shape[0],
            max_seqs=metadata.query_start_loc.numel() - 1, max_state_slots=state.shape[0],
            max_speculative_tokens=state.shape[2] - 9)
        args = dict(x=scratch.normalized, conv_weight=module.conv1d.weight.squeeze(1),
            query_start_loc=metadata.query_start_loc, state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh, num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens, conv_state=state,
            out=scratch.convolved)
        plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**args))
        self.plan, self.weight, self.state = plan, args["conv_weight"], state
        self.binding = prepare_sequence_binding(plan, args, session=session, name=name,
            public_bind=short_conv.bind, mode=mode, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "binding", binding))

    def functional(self, *, session):
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(t.shape), tuple(t.stride()), t.dtype)
            for t in (self.binding.gathered_state, self.binding.request_ids))
        self.binding = None

    def __call__(self, source, metadata):
        output = torch.empty_like(source)
        gathered, request_ids = (torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            for shape, stride, dtype in self.scratch_layout)
        short_conv_out(source, self.weight, metadata.query_start_loc, metadata.convolution_slots,
            metadata.fresh, metadata.accepted_tokens, metadata.fresh, metadata.num_seqs,
            metadata.num_tokens, self.state, output, gathered, request_ids, None, self.plan.handle)
        return output


__all__ = [
    "PLETable",
    "PLEMetadata",
    "PLEScratch",
    "PreparedPLE",
    "allocate_convolution_state",
    "table_layout",
]
