# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact-checkpoint decoder weight ownership and canonical module parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl

from hyperflare_recipes.components.parallel import TensorParallelShard
from ignition.planning import SUPPORTED_ARCHITECTURES, SUPPORTED_CAPABILITIES
from hyperflare_recipes.recipes.qwen38.checkpoint import CheckpointManifest
from ignition.preparation.host_memory import device_memory_info
from ignition.preparation.persistent import allocated_tensor_bytes
from .checkpoint_schema import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.triton import warmup_triton

from .hyperconnection import packed_down_projection, hyperconnection_extents
from .model import create_model_tree
from .moe import QwenMoeWeights
from .vision import VisionWeights
from .weights import QwenWeightLoader
from ignition.preparation.rebuild_manifest import weight_container


@weight_container(('attention', 'moe'))
@dataclass(frozen=True)
class DecoderLayerWeights:
    attention: dict[str, object]
    moe: QwenMoeWeights


@dataclass(frozen=True)
class DecoderAuxiliary:
    residual: torch.Tensor
    draft_features: torch.Tensor | None


torch.utils._pytree.register_dataclass(DecoderAuxiliary)


class PreparedDecoderForward:
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self):
        decoder = self.decoder
        errors = None if decoder.embedding_injection is None else decoder.embedding_injection.validate()
        runtime = decoder.state.begin_runtime()
        runtime.injection_errors = errors
        decoder(runtime)


@dataclass(frozen=True)
class ModelWeights:
    """Own the canonical tree and every packed GPU weight used by its recipes.

    Canonical quantized Linear and expert parameters remain on meta storage;
    their prepared replacements own the checkpoint's packed representations.
    PLE table ownership is separate because those tables reside in mapped host
    memory. Expert layers expand their CSF block scales once at load.
    """

    tree: torch.nn.Module
    layers: tuple[DecoderLayerWeights, ...]
    vision: VisionWeights
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)

    @classmethod
    def load(cls, manifest: CheckpointManifest, device: torch.device,
             *, progress: Callable[[str], None] | None = None,
             shard: TensorParallelShard = TensorParallelShard(0, 1),
             shard_hyperconnections: bool = False, force_moe_decode_a16: bool = False) -> ModelWeights:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"the Qwen recipe requires {SUPPORTED_ARCHITECTURES}")
        if force_moe_decode_a16 and torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
            raise ValueError("force_moe_decode_a16 requires SM120/121 packed NVFP4 experts")
        free_bytes, _ = device_memory_info(device)
        stored = sum(tensor.nbytes for tensor in manifest.tensors.values()
                     if tensor.component not in ("mtp", "ple_tables"))
        # Additional layouts, state, graph memory, and cache pages are planned separately.
        if free_bytes < stored // shard.world_size + (8 << 30):
            raise MemoryError(f"model weight loading requires at least {stored // shard.world_size + (8 << 30)} free CUDA bytes; found {free_bytes}")
        tree = create_model_tree(manifest)
        before = allocated_tensor_bytes(device)
        layers = []
        with QwenWeightLoader(manifest, device, progress=progress, shard=shard,
                              force_moe_decode_a16=force_moe_decode_a16) as loader:
            def parameters(paths, extents=None, *, temporary=False):
                for path, value in zip(paths, loader.load_tensors(paths, extents=extents, temporary=temporary), strict=True):
                    parent, _, attribute = path.rpartition(".")
                    setattr(tree.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))

            def hyperconnection(prefix: str):
                names = ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
                if tree.get_submodule(prefix).block_inject_weight is not None:
                    names += ("block_inject_weight",)
                extents = hyperconnection_extents(shard, shard_hyperconnections)
                packed = tree.get_submodule(prefix).block_inject_weight is not None
                for temporary in (False, True):
                    selected = tuple(name for name in names
                        if (packed and name in ("input_mix_weight_down", "block_inject_weight")) == temporary)
                    if selected:
                        parameters(tuple(f"{prefix}.{name}.weight" for name in selected),
                                   tuple(extents.get(f"{name}.weight") for name in selected), temporary=temporary)
                packed_down_projection(tree.get_submodule(prefix))

            parameters((f"{TEXT_PREFIX}.embed_tokens.weight", "lm_head.weight"),
                       (None, slice(*shard.partition(248320))))
            hyperconnection(f"{TEXT_PREFIX}.hyper_connection_mixer")
            for layer in range(48):
                prefix = f"{TEXT_PREFIX}.layers.{layer}"
                hyperconnection(f"{prefix}.attn_hyper_connection")
                hyperconnection(f"{prefix}.mlp_hyper_connection")
                if layer % 4 == 3:
                    attention_prefix = f"{prefix}.self_attn"
                    load_projections = loader.load_qsa_projections
                    names = ("q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight",
                             "indexer.k_layernorm.weight")
                else:
                    attention_prefix = f"{prefix}.linear_attn"
                    load_projections = loader.load_gdn_projections
                    names = ("A_log", "dt_bias", "conv1d.weight", "norm.weight")
                if layer % 4 == 3:
                    parameters(tuple(f"{attention_prefix}.{name}" for name in names))
                else:
                    heads = slice(*shard.partition(48))
                    parameters(tuple(f"{attention_prefix}.{name}" for name in ("A_log", "dt_bias", "norm.weight")),
                               (heads, heads, None))
                    channels = shard.fused_partitions((2048, 2048, 6144))
                    from ignition.preparation.persistent import persistent_cat

                    convolution = persistent_cat(loader.load_tensors((f"{attention_prefix}.conv1d.weight",) * 3,
                        extents=tuple(slice(*part) for part in channels), temporary=True))
                    module = tree.get_submodule(attention_prefix)
                    module.conv1d.weight = torch.nn.Parameter(convolution, requires_grad=False)
                    module.num_k_heads //= shard.world_size
                    module.num_v_heads //= shard.world_size
                    module.conv_dim //= shard.world_size
                attention = load_projections(attention_prefix)
                moe = QwenMoeWeights.load(loader, layer)
                module = tree.get_submodule(f"{prefix}.mlp")
                module.gate.weight = torch.nn.Parameter(moe.router, requires_grad=False)
                module.shared_expert_gate.weight = torch.nn.Parameter(moe.shared.gate, requires_grad=False)
                layers.append(DecoderLayerWeights(attention, moe))

            prefix = f"{TEXT_PREFIX}.layers.1.ple"
            parameters(tuple(f"{prefix}.{name}" for name in
                ("key_proj.weight", "value_proj.weight", "conv1d.weight", "norm_key.weight",
                 "norm_query.weight", "norm_conv.weight")))
            vision = VisionWeights.load(tree.model.visual, loader)
        torch.cuda.synchronize(device)
        return cls(tree, tuple(layers), vision,
                   allocated_tensor_bytes(device) - before, shard)


@dataclass(frozen=True)
class DecoderScratch:
    """Tensor examples used to declare and tune decoder operations."""

    embedding: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    sample_hidden: torch.Tensor
    logits: torch.Tensor
    hyperconnection: object
    gdn: object
    qsa: object
    ple: object
    moe: object
    local_logits: torch.Tensor | None = None

    @classmethod
    def allocate(cls, tokens: int, device, *, sample_rows: int = 4, world_size: int = 1,
                 hyperconnection_lowrank: int = 320):
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("decoder token capacity must be a positive integer")
        if type(sample_rows) is not int or sample_rows <= 0:
            raise ValueError("decoder sampling capacity must be positive")
        from .attention import QSAScratch
        from hyperflare_recipes.components.gated_delta import GDNScratch
        from .hyperconnection import HyperconnectionScratch
        from .moe import QwenMoeScratch
        from .ple import PLEScratch

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        embedding, residual, hidden = zeros(tokens, 2560), zeros(tokens, 10240), zeros(tokens, 2560)
        sample_hidden, logits = zeros(sample_rows, 2560), zeros(sample_rows, 248320)
        local_logits = zeros(sample_rows, 248320 // world_size) if world_size > 1 else None
        block_input, block_output = zeros(tokens, 2560), zeros(tokens, 2560)
        projected = zeros(tokens, (hyperconnection_lowrank + 4 + 15) // 16 * 16)
        ple = PLEScratch.allocate(tokens, device)
        gdn = GDNScratch.allocate(tokens, device, world_size=world_size, output=block_output)
        qsa = QSAScratch.allocate(tokens, device, world_size=world_size, output=block_output)
        moe = QwenMoeScratch.allocate(tokens, device, intermediate=640 // world_size,
                                    output=block_output)
        hc = HyperconnectionScratch.allocate(tokens, device,
                                            projected=projected, block_input=block_input, lowrank=hyperconnection_lowrank)
        return cls(embedding, residual, hidden, sample_hidden, logits, hc, gdn, qsa, ple, moe, local_logits)

    def rows(self, tokens: int, sequences: int):
        if (type(tokens) is not int or type(sequences) is not int
                or not 0 < tokens <= self.embedding.shape[0]
                or not 1 <= sequences <= self.sample_hidden.shape[0]):
            raise ValueError("decoder view exceeds persistent scratch capacity")

        def rows(value):
            return type(value)(*(getattr(value, field)[:tokens] for field in value.__dataclass_fields__))

        return type(self)(self.embedding[:tokens], self.residual[:tokens], self.hidden[:tokens],
            self.sample_hidden[:sequences], self.logits[:sequences], self.hyperconnection.rows(tokens),
            self.gdn.rows(tokens), self.qsa.rows(tokens), rows(self.ple), self.moe.rows(tokens),
            None if self.local_logits is None else self.local_logits[:sequences])


# Minimum prefill chunk capacity for 2-microbatch DMA pipelining. Splitting 512
# routed experts into two halves doubles per-layer expert-weight reads, so
# overlapping copy-engine all-reduce latency requires chunks of at least 4096
# tokens.
_PIPELINED_PREFILL_MIN_TOKENS = 4096


def _is_dma_runtime(runtime) -> bool:
    if runtime is None:
        return False
    from b12x.comm import pcie

    return isinstance(runtime, pcie.DmaAllReduce)


def _non_none(*plans) -> tuple:
    return tuple(plan for plan in plans if plan is not None)


def _hc_plans(hc) -> tuple:
    return _non_none(*hc.plans.values(), hc.down.plan, getattr(hc.gate, "plan", None))


def _attention_plans(attention, *, is_qsa: bool) -> tuple:
    plans = []
    for proj in attention.projections.values():
        plans.append(proj.plan)
        plans.append(getattr(proj, "quantizer_plan", None))
    if is_qsa:
        plans.extend((attention.plan, attention.writer, getattr(attention.sharded, "select_plan", None)))
    else:
        plans.extend((
            getattr(attention, "conv_plan", None),
            getattr(attention, "recurrent_plan", None),
            getattr(getattr(attention, "chunked", None), "plan", None),
        ))
    return _non_none(*plans)


def _moe_shard_plans(moe) -> tuple:
    return _non_none(
        moe.router_projection.plan,
        moe.routing.plan,
        moe.shared.gate_projection.plan,
    )


def _moe_compute_plans(moe) -> tuple:
    return _non_none(
        moe.experts.plan,
        getattr(moe.experts, "scale_plan", None),
        moe.shared.up_projection.plan,
        getattr(moe.shared.up_projection, "quantizer_plan", None),
        moe.shared.activation_plan,
        moe.shared.down_projection.plan,
        getattr(moe.shared.down_projection, "quantizer_plan", None),
    )


def _moe_plans(moe) -> tuple:
    return _moe_shard_plans(moe) + _moe_compute_plans(moe)


_AUX_COPY_CONFIG = triton.Config({}, num_warps=4, num_stages=2)
_AUX_COPY_CTAS = 32


@triton.autotune(configs=[_AUX_COPY_CONFIG], key=[])
@triton.jit(do_not_specialize=["ROWS"])
def _aux_copy_rows_2560_kernel(SRC, DST, ROWS, NUM_CTAS: tl.constexpr):
    """Copy [ROWS, 2560] BF16 prefix rows on the auxiliary stream using a bounded CTA count."""
    pid = tl.program_id(0)
    k = tl.arange(0, 512)
    r = pid
    while r < ROWS:
        base = r.to(tl.int64) * 2560
        tl.store(DST + base + k, tl.load(SRC + base + k))
        tl.store(DST + base + 512 + k, tl.load(SRC + base + 512 + k))
        tl.store(DST + base + 1024 + k, tl.load(SRC + base + 1024 + k))
        tl.store(DST + base + 1536 + k, tl.load(SRC + base + 1536 + k))
        tl.store(DST + base + 2048 + k, tl.load(SRC + base + 2048 + k))
        r += NUM_CTAS


@torch.library.custom_op("flare_recipes::qwen38_aux_copy_prefix", mutates_args=("dst",))
def _aux_copy_prefix(src: torch.Tensor, dst: torch.Tensor, stream: int) -> None:
    rows = src.shape[0]
    auxiliary = torch.cuda.ExternalStream(stream, device=src.device)
    auxiliary.wait_stream(torch.cuda.current_stream(src.device))
    with torch.cuda.stream(auxiliary):
        _aux_copy_rows_2560_kernel[(_AUX_COPY_CTAS,)](src, dst, rows, NUM_CTAS=_AUX_COPY_CTAS)


@_aux_copy_prefix.register_fake
def _fake_aux_copy_prefix(src: torch.Tensor, dst: torch.Tensor, stream: int) -> None:
    pass


def _mxfp8_scale_mma(scale_rows_2d: torch.Tensor, hidden: int) -> torch.Tensor:
    """Swizzle contiguous [rows, hidden // 32] UE8M0 scale rows into b12x MMA tile layout."""
    m_tiles, k_tiles = scale_rows_2d.shape[0] // 128, hidden // 128
    return (
        scale_rows_2d.view(m_tiles, 4, 32, k_tiles, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(1, m_tiles, k_tiles, 32, 4, 4)
    )


def _split_prefill_metadata(gdn, qsa, half: int):
    """Derive half-chunk GDN and QSA metadata for [0, half) and [half, 2 * half) on device."""
    from hyperflare_recipes.components.recurrent import DeltaMetadata
    from .attention import QSAMetadata

    total = gdn.num_tokens
    has_seq = gdn.num_seqs > 0
    one_i32 = torch.ones_like(total)
    len_a = total.clamp(max=half)
    len_b = (total - half).clamp(min=0)
    active_a = (len_a > 0) & has_seq
    active_b = (len_b > 0) & has_seq
    # Build [0, len] with a pointwise select rather than cat so Inductor does not
    # alias len_a/len_b into cu_a/cu_b storage that dies after cu_a/cu_b is snapshotted.
    cu_mask = torch.arange(2, dtype=torch.int32, device=total.device) == 1
    cu_a = torch.where(cu_mask, len_a, 0)
    cu_b = torch.where(cu_mask, len_b, 0)
    conv_a = torch.where(active_a, gdn.convolution_slots, -1)
    conv_b = torch.where(active_b, gdn.convolution_slots, -1)
    fresh_b = torch.zeros_like(gdn.fresh)
    init_a = torch.where(active_a, gdn.initial_state_indices, 0)
    final_a = torch.where(active_a, gdn.final_state_indices, 0)
    init_b = torch.where(active_b, gdn.final_state_indices, 0)
    final_b = torch.where(active_b, gdn.final_state_indices, 0)
    gdn_a = DeltaMetadata(
        cu_a, conv_a, gdn.fresh, gdn.accepted_tokens, gdn.state_indices,
        init_a, final_a, gdn.checkpoint_state_indices, gdn.checkpoint_offsets,
        active_a.to(torch.int32), len_a,
    )
    gdn_b = DeltaMetadata(
        cu_b, conv_b, fresh_b, one_i32, gdn.state_indices,
        init_b, final_b, gdn.checkpoint_state_indices, gdn.checkpoint_offsets,
        active_b.to(torch.int32), len_b,
    )
    seq_len_a = torch.where(active_a, qsa.sequence_lengths - len_b, 0).to(torch.int32)
    seq_len_b = torch.where(active_b, qsa.sequence_lengths, 0).to(torch.int32)
    qsa_a = QSAMetadata(
        qsa.main_block_table, qsa.compressed_block_table,
        torch.where(active_a, qsa.raw_state_slot_ids, -1),
        qsa.request_ids[:half], qsa.query_positions[:half], qsa.rope_positions[:half],
        seq_len_a, cu_a, qsa.num_accepted_tokens, qsa.is_prefilling & active_a,
        qsa.slot_mapping[:half],
    )
    qsa_b = QSAMetadata(
        qsa.main_block_table, qsa.compressed_block_table,
        torch.where(active_b, qsa.raw_state_slot_ids, -1),
        qsa.request_ids[half:], qsa.query_positions[half:], qsa.rope_positions[half:],
        seq_len_b, cu_b, one_i32, qsa.is_prefilling & active_b,
        qsa.slot_mapping[half:],
    )
    return gdn_a, gdn_b, qsa_a, qsa_b


class PreparedDecoder:
    """Capture the canonical 48-layer residual composition with prepared operators.

    Weights, hybrid state, paged attention caches and all escaping tensors have
    explicit owners. Scratch and workspaces are shared across serialized layer
    calls and graph variants. Finalize scratch bindings before capture.
    Python executes this object
    only during preparation and capture.
    """

    def __init__(self, weights: ModelWeights, scratch: DecoderScratch, state,
                 qsa_states, rotary, ple_table, workspaces: ScratchArena,
                 completion, frame_index, submission_sequence,
                 *, session, name: str, progress=None,
                 layer_outputs: torch.Tensor | None = None, embedding_injection=None,
                 max_cache_pages: int | None = None,
                 residual_output: torch.Tensor | None = None, verification: bool = False,
                 gdn_records=None, gdn_in_place: bool = False, runtime=None, barrier_runtime=None, vocabulary_runtime=None,
                 result_runtime=None, qsa_gather_runtime=None, qsa_shard_mode: str | None = None,
                 coordinator=None, sampling_workspace=None, scoring=None, sample: bool = True):
        from hyperflare_recipes.components.embedding import PreparedEmbedding
        from hyperflare_recipes.components.linear import UnquantizedProjection
        from hyperflare_recipes.components.recurrent import DeltaMetadata

        from .attention import PreparedAttention, QSAMetadata
        from hyperflare_recipes.components.gated_delta import PreparedGDN
        from .hyperconnection import PreparedHyperconnection
        from .moe import PreparedQwenMoe
        from .ple import PreparedPLE

        if not sample and (state.mode != "prefill" or verification or scoring is not None):
            raise ValueError("non-sampling decoders require unverified, unscored prefill state")
        self.sample = bool(sample)

        if verification and scratch.sample_hidden.shape == scratch.hidden.shape:
            scratch = replace(scratch, sample_hidden=scratch.hidden)
        self.weights, self.scratch, self.state = weights, scratch, state
        self.record_layout = None if gdn_records is None else (
            gdn_records.layers, gdn_records.delta.shape[1], gdn_records.keys.shape[2],
            gdn_records.delta.shape[2], gdn_records.delta.shape[3])
        self.verification = verification
        if residual_output is not None and (
                residual_output.shape != scratch.residual.shape
                or residual_output.dtype != torch.bfloat16
                or residual_output.device != scratch.residual.device):
            raise ValueError("target feedback requires persistent BF16 [tokens,10240] storage")
        self.residual_output = residual_output
        if layer_outputs is not None and (layer_outputs.shape != (48, state.tokens, 10240)
                or layer_outputs.dtype != torch.bfloat16 or layer_outputs.device != scratch.hidden.device):
            raise ValueError("layer trace requires BF16 [48,token_capacity,10240] on the model device")
        self.layer_outputs = layer_outputs
        self.context_capture, self.context_layers = None, ()
        self.embedding_injection = embedding_injection
        self.workspaces, self.ple_table, self.qsa_states = workspaces, ple_table, qsa_states
        language = weights.tree.model.language_model
        sharded_hc = language.layers[0].attn_hyper_connection.input_mix_weight_down.weight.shape[0] < 320
        half = state.tokens // 2
        shard = getattr(weights, "shard", None)
        row_sharded_qsa_fits = (
            qsa_shard_mode is None
            or (qsa_shard_mode == "row" and shard is not None and half % (32 * shard.world_size) == 0)
        )
        self.pipelined_prefill = (
            state.mode == "prefill"
            and state.rows == 1
            and getattr(state, "gdn_partitions", None) is None
            and gdn_records is None
            and not sharded_hc
            and isinstance(workspaces, ScratchArena)
            and workspaces.overlap
            and _is_dma_runtime(runtime)
            and state.tokens >= _PIPELINED_PREFILL_MIN_TOKENS
            and state.tokens % 2 == 0
            and row_sharded_qsa_fits
            and runtime.should_allreduce(scratch.hidden[:half])
        )
        self.tokens = scratch.residual.shape[0]
        self.half_tokens = half if self.pipelined_prefill else 0
        self._name = name
        self._runtime = runtime
        self._barrier_runtime = barrier_runtime
        self._coordinator = coordinator
        from .decoder_state import PreparedDecoderState

        is_single_prefill_state = (
            isinstance(state, PreparedDecoderState) and state.mode == "prefill" and state.rows == 1
        )
        self._single_row_final_mixer = bool(
            self.sample and scoring is None and is_single_prefill_state and self.tokens > 1
        )
        needs_any_residual = bool(
            getattr(state, "maintain_draft_state", False)
            or residual_output is not None
            or verification
        )
        self._single_row_residual = False
        self._needs_full_residual = needs_any_residual
        self.row_shard_dma = False
        self.split_att_rs_1 = False
        self.world_size = 1 if shard is None else shard.world_size
        self.shard_tokens = 0
        self.shard_rows_a = self.shard_rows_b = None
        self.shard_offset_a = self.shard_offset_b = None
        self.attn_quantizer_plan = None
        self.attn_local_shapes = None
        self.attn_vals_bf16_cols = 0
        self.attn_scale_bf16_cols = 0
        self.attn_pad_bf16_cols = 0
        self.layer_attn_mxfp8 = ()
        self.final_res_gather_a = self.final_res_gather_b = None
        self.final_hidden_gather_a = self.final_hidden_gather_b = None
        self.final_row_reduce = None
        self.context_capture_reduce = None
        self.sharded_context_layers = ()
        ffn_local_packed = ffn_full_packed = None
        if self.pipelined_prefill and shard is not None and shard.world_size > 1 and layer_outputs is None and len(language.layers) > 1:
            from ignition.components.collectives import (
                PreparedChunkedGatherShards,
                PreparedGatherShards,
                PreparedReduceScatter,
                row_shard_surfaces,
            )

            if row_shard_surfaces(runtime) and half % shard.world_size == 0:
                st = half // shard.world_size
                dev = scratch.hidden.device
                # Pack BF16 hidden (2560) + RouteRecord for top_k=10 (40 BF16 cols) + 24-col padded
                # shared gate = 2624 BF16 cols (32-element / 64-byte aligned so Inductor keeps rows
                # strictly contiguous without stride padding).
                cand_ffn_local = torch.zeros((st, 2624), dtype=torch.bfloat16, device=dev)
                cand_ffn_full = torch.zeros((half, 2624), dtype=torch.bfloat16, device=dev)
                if (
                    PreparedReduceScatter.fits(runtime, scratch.hidden[:half], scratch.hidden[:st])
                    and PreparedGatherShards.fits(runtime, scratch.hidden[:st], scratch.hidden[:half])
                    and PreparedGatherShards.fits(runtime, cand_ffn_local, cand_ffn_full)
                    and (not self._needs_full_residual
                         or PreparedChunkedGatherShards.fits(runtime, scratch.residual[:st], scratch.residual[:half]))
                ):
                    self.row_shard_dma = True
                    self.shard_tokens = st
                    self.split_att_rs_1 = (
                        st >= 1024
                        and st % 2 == 0
                        and PreparedReduceScatter.fits(runtime, scratch.hidden[:half // 2], scratch.hidden[:st // 2])
                    )
                    if self.split_att_rs_1:
                        warmup_triton(
                            _aux_copy_rows_2560_kernel.fn,
                            (scratch.hidden[:st // 2], scratch.hidden[:st], st // 2),
                            (_AUX_COPY_CTAS, 1, 1),
                            num_warps=_AUX_COPY_CONFIG.num_warps,
                            num_stages=_AUX_COPY_CONFIG.num_stages,
                            NUM_CTAS=_AUX_COPY_CTAS,
                        )
                    ffn_local_packed, ffn_full_packed = cand_ffn_local, cand_ffn_full
                    from ignition.preparation.persistent import persistent_copy

                    s0 = shard.rank * st
                    self.shard_rows_a = persistent_copy(torch.arange(s0, s0 + st, dtype=torch.int64, device=dev))
                    self.shard_rows_b = persistent_copy(torch.arange(half + s0, half + s0 + st, dtype=torch.int64, device=dev))
                    self.shard_offset_a = persistent_copy(torch.tensor([s0], dtype=torch.int64, device=dev))
                    self.shard_offset_b = persistent_copy(torch.tensor([half + s0], dtype=torch.int64, device=dev))
                    if (
                        is_single_prefill_state
                        and residual_output is None
                        and not verification
                        and getattr(state, "maintain_draft_state", False)
                    ):
                        self._single_row_residual = True
                        self._needs_full_residual = False
        if self.pipelined_prefill:
            layer_scratch = scratch.rows(half, state.rows)
            shard_scratch = scratch.rows(self.shard_tokens, state.rows) if self.row_shard_dma else None
            layer_gdn_meta = DeltaMetadata.allocate(state.rows, scratch.hidden.device)
            layer_qsa_meta = QSAMetadata.allocate(
                tokens=half, batch=state.rows,
                context=state.qsa.main_block_table.shape[1] * 64,
                device=scratch.hidden.device,
            )
        else:
            layer_scratch = scratch
            shard_scratch = None
            layer_gdn_meta = state.gdn
            layer_qsa_meta = state.qsa
        source = layer_scratch.hyperconnection.block_input
        attn_full_encoded = None
        attn_local_packed = attn_full_packed = None
        mxfp8_gather_eligible = False
        if self.row_shard_dma:
            from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
            from b12x.preparation import PreparedCall
            from b12x.quantization import mxfp8
            from hyperflare_recipes.components import sm103
            from ignition.components.collectives import PreparedGatherShards
            from ignition.preparation.tuning import declare

            st = self.shard_tokens
            dev = source.device
            if dev.type == "cuda" and st % 128 == 0 and half % 128 == 0 and half > 28 and not sm103.is_sm103(dev):
                local_shapes = mxfp8.dense_rows_storage_shapes(st, 2560)
                full_shapes = mxfp8.dense_rows_storage_shapes(half, 2560)
                local_vals = torch.zeros(local_shapes[0], dtype=torch.float8_e4m3fn, device=dev)
                local_srows = torch.full(local_shapes[1], 127, dtype=torch.uint8, device=dev)
                local_smma = torch.full(local_shapes[2], 127, dtype=torch.uint8, device=dev)
                full_vals = torch.zeros(full_shapes[0], dtype=torch.float8_e4m3fn, device=dev)
                full_srows = torch.full(full_shapes[1], 127, dtype=torch.uint8, device=dev)
                full_smma = torch.full(full_shapes[2], 127, dtype=torch.uint8, device=dev)
                vals_bf16_cols = 2560 // 2
                scale_bf16_cols = local_srows.view(st, -1).view(torch.bfloat16).shape[1]
                # Pad packed row width to a multiple of 32 BF16 elements (64 bytes) so Inductor
                # keeps torch.cat outputs strictly contiguous for DMA GatherShards.
                pad_bf16_cols = (32 - (vals_bf16_cols + scale_bf16_cols) % 32) % 32
                packed_bf16_cols = vals_bf16_cols + scale_bf16_cols + pad_bf16_cols
                cand_local_packed = torch.zeros((st, packed_bf16_cols), dtype=torch.bfloat16, device=dev)
                cand_full_packed = torch.zeros((half, packed_bf16_cols), dtype=torch.bfloat16, device=dev)
                if PreparedGatherShards.fits(runtime, cand_local_packed, cand_full_packed):
                    mxfp8_gather_eligible = True
                    attn_local_packed, attn_full_packed = cand_local_packed, cand_full_packed
                    attn_local_encoded = mxfp8.dense_rows_from_bases(local_vals, local_srows, local_smma)
                    attn_full_encoded = mxfp8.dense_rows_from_bases(full_vals, full_srows, full_smma)
                    self.attn_local_shapes = local_shapes
                    self.attn_vals_bf16_cols = vals_bf16_cols
                    self.attn_scale_bf16_cols = scale_bf16_cols
                    self.attn_pad_bf16_cols = pad_bf16_cols
                    src_shard = shard_scratch.hyperconnection.block_input
                    self.attn_quantizer_plan = mxfp8.plan(mxfp8.query_from_call(
                        src_shard, attn_local_encoded.values, attn_local_encoded.scale_rows,
                        attn_local_encoded.scale_mma, min_amax=0.0))
                    declare(session, (self.attn_quantizer_plan.request(
                        name=f"{name}.row_shard.attn_quant",
                        prepare_call=lambda state, src=src_shard, enc=attn_local_encoded: PreparedCall(
                            run=lambda: state.run(src, enc.values, enc.scale_rows, enc.scale_mma),
                            owners=(src, enc))),))
        self.embedding = PreparedEmbedding(language.embed_tokens.weight, state.ple.token_ids,
            scratch.embedding, num_rows=state.gdn.num_tokens, session=session, name=f"{name}.embedding")
        last_index = len(language.layers) - 1
        # A non-terminal chunk without classic MTP draft-residual retention never consumes the final
        # layer's residual stream. When the final layer is QSA and not a DFlash2 feature tap, only
        # its KV and selector-cache writes persist; sparse attention, o_proj, MLP, and final-layer
        # collectives are skipped.
        self.last_layer_cache_only = (
            not self.sample
            and not getattr(state, "maintain_draft_state", False)
            and residual_output is None
            and layer_outputs is None
            and last_index >= 0
            and last_index % 4 == 3
        )
        self.layers, self.reductions, self.pipelined_reductions = [], [], []
        layer_attn_mxfp8 = []
        for index, (module, layer_weights) in enumerate(zip(language.layers, weights.layers, strict=True)):
            prefix = f"{name}.layer{index}"
            is_cache_only = self.last_layer_cache_only and index == last_index
            layer_sharded = self.row_shard_dma and index >= 1
            hc_scratch = shard_scratch if layer_sharded else layer_scratch
            use_attn_mxfp8 = (
                layer_sharded
                and mxfp8_gather_eligible
                and all(
                    isinstance(w, MXFP8LinearWeight) and w.padded_in_features == 2560
                    for key, w in layer_weights.attention.items()
                    if key not in ("out_proj", "o_proj")
                )
            )
            layer_attn_mxfp8.append(use_attn_mxfp8)
            layer_input_encoded = attn_full_encoded if use_attn_mxfp8 else None
            attn_hc = PreparedHyperconnection(module.attn_hyper_connection, hc_scratch.hyperconnection,
                priming_state=hc_scratch.residual, session=session, name=f"{prefix}.attn_hc",
                runtime=runtime, coordinator=coordinator)
            if is_cache_only:
                attention = PreparedAttention(module.self_attn, layer_weights.attention, source,
                    layer_scratch.qsa, qsa_states[index], layer_qsa_meta, rotary, session=session, name=f"{prefix}.qsa",
                    workspace=workspaces, projection_workspace=workspaces,
                    max_cache_pages=max_cache_pages, draft_tokens=state.pool.layout.speculative_tokens,
                    cache_only=True, shard=shard, coordinator=coordinator,
                    input_encoded=layer_input_encoded)
                self.layers.append((attn_hc, attention, None, None))
                if layer_sharded:
                    from ignition.components.collectives import ConcurrentGatherShards, PreparedGatherShards

                    ag_in = attn_local_packed if use_attn_mxfp8 else shard_scratch.hyperconnection.block_input
                    ag_out = attn_full_packed if use_attn_mxfp8 else source
                    ag_0 = PreparedGatherShards(ag_in, ag_out, runtime=runtime, session=session,
                        coordinator=coordinator, name=f"{prefix}.attn_gather.0")
                    ag_1 = PreparedGatherShards(ag_in, ag_out, runtime=runtime, session=session,
                        coordinator=coordinator, name=f"{prefix}.attn_gather.1")
                    attn_hc_plans = _hc_plans(attn_hc) + ((self.attn_quantizer_plan,) if use_attn_mxfp8 else ())
                    attn_plans = _attention_plans(attention, is_qsa=True)
                    self.pipelined_reductions.append((
                        ConcurrentGatherShards(ag_0, workspace=workspaces, session=session, parallel_plans=attn_hc_plans),
                        ConcurrentGatherShards(ag_1, workspace=workspaces, session=session, parallel_plans=attn_plans),
                        None, None, None, None, None, None, None,
                    ))
                elif self.pipelined_prefill:
                    self.pipelined_reductions.append((None, None, None, None))
                elif runtime is not None:
                    self.reductions.append((None, None))
                if progress is not None:
                    progress(f"prepared {name} layer {index + 1}/48")
                continue
            mlp_hc = PreparedHyperconnection(module.mlp_hyper_connection, hc_scratch.hyperconnection,
                priming_state=hc_scratch.residual, session=session, name=f"{prefix}.mlp_hc",
                runtime=runtime, coordinator=coordinator)
            if index % 4 == 3:
                attention = PreparedAttention(module.self_attn, layer_weights.attention, source,
                    layer_scratch.qsa, qsa_states[index], layer_qsa_meta, rotary, session=session, name=f"{prefix}.qsa",
                    workspace=workspaces, projection_workspace=workspaces,
                    max_cache_pages=max_cache_pages, draft_tokens=state.pool.layout.speculative_tokens,
                    shard=shard, qsa_gather_runtime=qsa_gather_runtime,
                    qsa_shard_mode=qsa_shard_mode, coordinator=coordinator,
                    input_encoded=layer_input_encoded)
            else:
                attention = PreparedGDN(module.linear_attn, layer_weights.attention, source,
                    layer_scratch.gdn, state.pool.gdn[index], layer_gdn_meta, mode=state.mode,
                    session=session, name=f"{prefix}.gdn", checkpoint_export=False,
                    workspace=workspaces, projection_workspace=workspaces,
                    records=None if gdn_records is None else gdn_records[index], in_place=gdn_in_place,
                    partitions=getattr(state, "gdn_partitions", None),
                    input_encoded=layer_input_encoded)
            moe = PreparedQwenMoe(source, layer_weights.moe, session=session, name=f"{prefix}.moe",
                scratch=layer_scratch.moe, workspace=workspaces,
                projection_workspace=workspaces, prefill=state.mode == "prefill",
                shard_source=shard_scratch.hyperconnection.block_input if layer_sharded else None)
            self.layers.append((attn_hc, attention, mlp_hc, moe))
            if layer_sharded:
                from ignition.components.collectives import (
                    ConcurrentGatherShards,
                    ConcurrentReduceScatter,
                    PreparedGatherShards,
                    PreparedReduceScatter,
                )

                ag_in = attn_local_packed if use_attn_mxfp8 else shard_scratch.hyperconnection.block_input
                ag_out = attn_full_packed if use_attn_mxfp8 else source
                attn_gather_0 = PreparedGatherShards(ag_in, ag_out, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"{prefix}.attn_gather.0")
                attn_gather_1 = PreparedGatherShards(ag_in, ag_out, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"{prefix}.attn_gather.1")
                att_rs_0 = PreparedReduceScatter(attention.scratch.output, shard_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_rs.0")
                if self.split_att_rs_1:
                    half_in = half // 2
                    half_st = self.shard_tokens // 2
                    att_rs_1 = PreparedReduceScatter(attention.scratch.output[:half_in], shard_scratch.hidden[:half_st],
                        runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_rs.1a")
                    att_rs_1b = PreparedReduceScatter(attention.scratch.output[:half_in], shard_scratch.hidden[:half_st],
                        runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_rs.1b")
                else:
                    att_rs_1 = PreparedReduceScatter(attention.scratch.output, shard_scratch.hidden,
                        runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_rs.1")
                    att_rs_1b = None
                ffn_gather_0 = PreparedGatherShards(ffn_local_packed, ffn_full_packed,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.ffn_gather.0")
                ffn_gather_1 = PreparedGatherShards(ffn_local_packed, ffn_full_packed,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.ffn_gather.1")
                mlp_rs_0 = PreparedReduceScatter(moe.output, shard_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.moe_rs.0")
                mlp_rs_1 = PreparedReduceScatter(moe.output, shard_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.moe_rs.1")
                attn_hc_plans = _hc_plans(attn_hc) + ((self.attn_quantizer_plan,) if use_attn_mxfp8 else ())
                attn_plans = _attention_plans(attention, is_qsa=index % 4 == 3)
                shard_prep_plans = _hc_plans(mlp_hc) + _moe_shard_plans(moe)
                moe_compute_plans = _moe_compute_plans(moe)
                self.pipelined_reductions.append((
                    ConcurrentGatherShards(attn_gather_0, workspace=workspaces, session=session,
                        parallel_plans=attn_hc_plans),
                    ConcurrentGatherShards(attn_gather_1, workspace=workspaces, session=session,
                        parallel_plans=attn_plans),
                    ConcurrentReduceScatter(att_rs_0, workspace=workspaces, session=session,
                        parallel_plans=attn_plans),
                    ConcurrentReduceScatter(att_rs_1, workspace=workspaces, session=session,
                        parallel_plans=moe_compute_plans),
                    ConcurrentGatherShards(ffn_gather_0, workspace=workspaces, session=session,
                        parallel_plans=attn_plans),
                    ConcurrentGatherShards(ffn_gather_1, workspace=workspaces, session=session,
                        parallel_plans=_hc_plans(mlp_hc)),
                    ConcurrentReduceScatter(mlp_rs_0, workspace=workspaces, session=session,
                        parallel_plans=shard_prep_plans),
                    ConcurrentReduceScatter(mlp_rs_1, workspace=workspaces, session=session,
                        parallel_plans=_hc_plans(mlp_hc)),
                    None if att_rs_1b is None else ConcurrentReduceScatter(att_rs_1b, workspace=workspaces,
                        session=session, parallel_plans=moe_compute_plans),
                ))
            elif self.pipelined_prefill:
                from ignition.components.collectives import ConcurrentAllReduce, PreparedAllReduce

                att_reduce_0 = PreparedAllReduce(attention.scratch.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_reduce.0")
                att_reduce_1 = PreparedAllReduce(attention.scratch.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.attention_reduce.1")
                mlp_reduce_0 = PreparedAllReduce(moe.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.moe_reduce.0")
                mlp_reduce_1 = PreparedAllReduce(moe.output, layer_scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.moe_reduce.1")
                attn_plans = _hc_plans(attn_hc) + _attention_plans(attention, is_qsa=index % 4 == 3)
                moe_plans = _hc_plans(mlp_hc) + _moe_plans(moe)
                self.pipelined_reductions.append((
                    ConcurrentAllReduce(att_reduce_0, workspace=workspaces, session=session,
                        parallel_plans=attn_plans),
                    ConcurrentAllReduce(att_reduce_1, workspace=workspaces, session=session,
                        parallel_plans=moe_plans),
                    ConcurrentAllReduce(mlp_reduce_0, workspace=workspaces, session=session,
                        parallel_plans=moe_plans),
                    ConcurrentAllReduce(mlp_reduce_1, workspace=workspaces, session=session,
                        parallel_plans=_hc_plans(mlp_hc)),
                ))
            elif runtime is not None:
                from ignition.components.collectives import PreparedAllReduce

                self.reductions.append(tuple(PreparedAllReduce(value, scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.{label}_reduce")
                    for label, value in (("attention", attention.scratch.output), ("moe", moe.output))))
            if progress is not None:
                progress(f"prepared {name} layer {index + 1}/48")
        self.layer_attn_mxfp8 = tuple(layer_attn_mxfp8)
        module = language.layers[1].ple
        self.ple = PreparedPLE(module,
            {key: getattr(module, key).weight for key in ("key_proj", "value_proj")},
            scratch.residual, scratch.ple, state.pool.ple_convolution, state.ple, ple_table,
            mode=state.mode, session=session, name=f"{name}.ple", workspace=workspaces,
            partitions=getattr(state, "ple_partitions", None),
            runtime=runtime if barrier_runtime is None else barrier_runtime, coordinator=coordinator)
        if (
            not self.row_shard_dma
            and not self.pipelined_prefill
            and len(self.layers) > 1
            and hasattr(self.ple, "prepare_residual_norm")
        ):
            layer1_hc = self.layers[1][0]
            self.ple.prepare_residual_norm(layer1_hc.norm_gain, eps=layer1_hc.eps)
        if self._single_row_final_mixer:
            final_hc_scratch = scratch.rows(1, 1)
            final_mixer_runtime = barrier_runtime if barrier_runtime is not None else runtime
        else:
            final_hc_scratch = shard_scratch if self.row_shard_dma else scratch
            final_mixer_runtime = runtime
        self.final_mixer = (
            PreparedHyperconnection(language.hyper_connection_mixer,
                replace(final_hc_scratch.hyperconnection, block_input=final_hc_scratch.hidden),
                priming_state=final_hc_scratch.residual,
                session=session, name=f"{name}.final_mixer", runtime=final_mixer_runtime, coordinator=coordinator)
            if self.sample else None
        )
        if self.row_shard_dma:
            from ignition.components.collectives import PreparedAllReduce, PreparedChunkedGatherShards, PreparedGatherShards

            if self.sample and not self._single_row_final_mixer:
                self.final_hidden_gather_a = PreparedGatherShards(
                    shard_scratch.hidden, layer_scratch.hidden, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"{name}.final_hidden_gather.a")
                self.final_hidden_gather_b = PreparedGatherShards(
                    shard_scratch.hidden, layer_scratch.hidden, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"{name}.final_hidden_gather.b")
            if self._needs_full_residual:
                self.final_res_gather_a = PreparedChunkedGatherShards(
                    shard_scratch.residual, layer_scratch.residual, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"{name}.final_res_gather.a")
                self.final_res_gather_b = PreparedChunkedGatherShards(
                    shard_scratch.residual, layer_scratch.residual, runtime=runtime, session=session,
                    coordinator=coordinator, name=f"{name}.final_res_gather.b")
            elif self._single_row_final_mixer or self._single_row_residual:
                dev = scratch.residual.device
                row_in = torch.zeros((1, 10240), dtype=torch.bfloat16, device=dev)
                row_out = torch.empty_like(row_in)
                row_rt = barrier_runtime if barrier_runtime is not None else runtime
                self.final_row_reduce = PreparedAllReduce(
                    row_in, row_out, runtime=row_rt, session=session,
                    coordinator=coordinator, name=f"{name}.final_row_reduce",
                )
        for index, (attn_hc, _, mlp_hc, moe) in enumerate(self.layers):
            if mlp_hc is None:
                continue
            hc_scratch = shard_scratch if (self.row_shard_dma and index >= 1) else layer_scratch
            attn_hc.prepare_combine_norm(mlp_hc, priming_state=hc_scratch.residual,
                session=session, name=f"{name}.layer{index}.attn_combine_norm")
            if index != 0 and ((self.sample and not self._single_row_final_mixer) or index + 1 < len(self.layers)):
                following = self.layers[index + 1][0] if index + 1 < len(self.layers) else self.final_mixer
                fuse_mlp = (
                    not self.row_shard_dma
                    and not self.pipelined_prefill
                    and not self.reductions
                    and moe is not None
                    and not getattr(getattr(moe, "experts", None), "reduces_into_shared", False)
                )
                mlp_hc.prepare_combine_norm(following, priming_state=hc_scratch.residual,
                    session=session, name=f"{name}.layer{index}.mlp_combine_norm", fuse_mlp=fuse_mlp)
        if self.row_shard_dma:
            for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
                if index == 0:
                    _, cr_att_1, cr_mlp_0, _ = self.pipelined_reductions[0]
                    if mlp_hc is not None and "combine_norm" in attn_hc.plans:
                        attn_cn = attn_hc.plans["combine_norm"]
                        workspaces.register_parallel(cr_att_1.reduce.plan, attn_cn)
                        workspaces.register_parallel(cr_mlp_0.reduce.plan, attn_cn)
                    continue
                (cr_ag_0, _, _, _, cr_ffn_0, cr_ffn_1, cr_mlp_0, cr_mlp_1, _) = self.pipelined_reductions[index]
                if mlp_hc is not None:
                    if "combine_norm" in attn_hc.plans:
                        attn_cn = attn_hc.plans["combine_norm"]
                        workspaces.register_parallel(cr_mlp_0.reduce.plan, attn_cn)
                        workspaces.register_parallel(cr_ffn_0.gather.plan, attn_cn)
                    if "combine_norm" in mlp_hc.plans:
                        workspaces.register_parallel(cr_ffn_1.gather.plan, mlp_hc.plans["combine_norm"])
                        workspaces.register_parallel(cr_mlp_1.reduce.plan, mlp_hc.plans["combine_norm"])
                if index >= 2:
                    prev_cr_ffn_1 = self.pipelined_reductions[index - 1][5]
                    prev_cr_mlp_1 = self.pipelined_reductions[index - 1][7]
                    prev_moe = self.layers[index - 1][3]
                    attn_hc_plans = _hc_plans(attn_hc) + (
                        (self.attn_quantizer_plan,) if self.layer_attn_mxfp8[index] else ()
                    )
                    for plan in attn_hc_plans:
                        workspaces.register_parallel(prev_cr_ffn_1.gather.plan, plan)
                    for plan in _moe_compute_plans(prev_moe):
                        workspaces.register_parallel(cr_ag_0.gather.plan, plan)
                    for plan in _attention_plans(attention, is_qsa=index % 4 == 3):
                        workspaces.register_parallel(prev_cr_mlp_1.reduce.plan, plan)
        elif self.pipelined_prefill:
            for index, (attn_hc, attention, mlp_hc, _) in enumerate(self.layers):
                cr_att_0, cr_att_1, cr_mlp_0, cr_mlp_1 = self.pipelined_reductions[index]
                if mlp_hc is not None:
                    if "combine_norm" in attn_hc.plans:
                        attn_cn = attn_hc.plans["combine_norm"]
                        workspaces.register_parallel(cr_att_1.reduce.plan, attn_cn)
                        workspaces.register_parallel(cr_mlp_0.reduce.plan, attn_cn)
                    if index > 0 and "combine_norm" in mlp_hc.plans:
                        workspaces.register_parallel(cr_mlp_1.reduce.plan, mlp_hc.plans["combine_norm"])
                if index >= 2:
                    prev_cr_mlp_1 = self.pipelined_reductions[index - 1][3]
                    prev_mlp_cn = self.layers[index - 1][2].plans.get("combine_norm")
                    if cr_att_0 is not None and prev_mlp_cn is not None:
                        workspaces.register_parallel(cr_att_0.reduce.plan, prev_mlp_cn)
                    for plan in _hc_plans(attn_hc) + _attention_plans(attention, is_qsa=index % 4 == 3):
                        workspaces.register_parallel(prev_cr_mlp_1.reduce.plan, plan)
        if self.sample:
            if self._single_row_final_mixer:
                state.prepare_hidden(scratch.hidden[:1], scratch.sample_hidden, single_row=True)
            else:
                state.prepare_hidden(scratch.hidden, scratch.sample_hidden)
        else:
            state.prepare_rope_delta()
        if self._single_row_residual:
            state.prepare_residual(scratch.residual[:1], single_row=True)
        else:
            state.prepare_residual(scratch.residual)
        if not self.sample:
            self.lm_head = None
        elif vocabulary_runtime is None:
            self.lm_head = UnquantizedProjection(scratch.sample_hidden, weights.tree.lm_head.weight,
                scratch.logits, session=session, name=f"{name}.lm_head")
        else:
            from hyperflare_recipes.components.vocabulary import PreparedVocabularyProjection

            self.lm_head = PreparedVocabularyProjection(scratch.sample_hidden, weights.tree.lm_head.weight,
                scratch.local_logits, scratch.logits, runtime=vocabulary_runtime, session=session,
                coordinator=coordinator, name=f"{name}.lm_head")
        self.scoring = None if scoring is None else scoring.prepare(scratch.hidden, weights.tree.lm_head.weight,
            vocabulary=248320, session=session, coordinator=coordinator, name=name + ".prompt")
        self.sampler = None
        self.completion_recorder = None
        if not verification:
            state, inputs = self.state, self.state.inputs
            if self.sample:
                from ignition.components.sampling import PreparedSampling

                self.sampler = PreparedSampling(self.scratch.logits, state.requests,
                    inputs.row_slot_ids[:state.rows], inputs.row_generations[:state.rows],
                    inputs.live_rows, completion, frame_index, submission_sequence,
                    sample_enabled=inputs.sample_enabled[:state.rows], workspace=sampling_workspace)
                self.sampler.prepare()
                self.sampler.prepare_broadcast(runtime=vocabulary_runtime if result_runtime is None else result_runtime,
                    session=session,
                    coordinator=coordinator, name=name + ".sampling")
            else:
                from ignition.components.sampling import PreparedCompletion

                self.completion_recorder = PreparedCompletion(state.requests,
                    inputs.row_slot_ids[:state.rows], inputs.row_generations[:state.rows],
                    inputs.live_rows, completion, frame_index, submission_sequence)
                self.completion_recorder.prepare()

    def capture_context(self, metadata, draft_config, *, compact_prefill=False):
        from hyperflare_recipes.components.dflash2 import PreparedContextCapture

        if getattr(self, "_functional_requested", False):
            raise RuntimeError("context capture must be configured before functional conversion")
        if self.last_layer_cache_only and (len(self.layers) - 1) in draft_config.target_layers:
            raise ValueError("cannot capture context from a cache-only final layer")
        self.context_layers = draft_config.target_layers
        self.context_capture = PreparedContextCapture(metadata,
            2560 * len(self.context_layers), draft_config.attention.window,
            compact_prefill=compact_prefill)
        self.auxiliary = torch.empty(self.context_capture.shape, dtype=torch.bfloat16,
                                     device=self.context_capture.device)

    def _capture_context_layer(self, capture, layer_index: int, res_a: torch.Tensor, res_b: torch.Tensor | None = None):
        column = self.context_layers.index(layer_index) * 2560
        _, indices, _ = capture
        if indices is None:
            if res_b is None:
                features = res_a.view(-1, 4, 2560).mean(dim=1)
            else:
                features = torch.cat(
                    (res_a.view(-1, 4, 2560).mean(dim=1), res_b.view(-1, 4, 2560).mean(dim=1)),
                    dim=0,
                )
            self.context_capture(capture, features, column)
            return
        if res_b is None:
            selected = res_a[indices]
        else:
            half = res_a.shape[0]
            selected = torch.where(
                (indices < half)[:, None],
                res_a[indices.clamp(max=half - 1)],
                res_b[(indices - half).clamp(min=0)],
            )
        self.context_capture.write_selected(capture, selected.view(-1, 4, 2560).mean(dim=1), column)

    def _capture_context_layer_sharded(self, capture, sharded_slots: dict[int, torch.Tensor],
                                       layer_index: int, res_a: torch.Tensor, res_b: torch.Tensor) -> None:
        _, indices, live = capture
        st = self.shard_tokens
        if indices is None:
            local = torch.zeros((2 * self.half_tokens, 2560), dtype=torch.bfloat16, device=res_a.device)
            local.index_copy_(0, self.shard_rows_a, res_a.view(-1, 4, 2560).mean(dim=1))
            local.index_copy_(0, self.shard_rows_b, res_b.view(-1, 4, 2560).mean(dim=1))
            sharded_slots[layer_index] = local
            return
        s0 = self.shard_offset_a
        s1 = self.shard_offset_b
        in_a = (indices >= s0) & (indices < s0 + st)
        in_b = (indices >= s1) & (indices < s1 + st)
        idx_a = (indices - s0).clamp(0, st - 1)
        idx_b = (indices - s1).clamp(0, st - 1)
        selected = torch.where(in_a[:, None], res_a[idx_a], res_b[idx_b])
        owned = (in_a | in_b) if live is None else ((in_a | in_b) & live)
        mean = selected.view(-1, 4, 2560).mean(dim=1)
        sharded_slots[layer_index] = torch.where(owned[:, None], mean, 0)

    def _finish_sharded_context_capture(self, capture, sharded_slots: dict[int, torch.Tensor]):
        if self.context_capture_reduce is None or not self.sharded_context_layers:
            return None
        ordered = [sharded_slots[layer_index] for layer_index in self.sharded_context_layers]
        reduced = self.context_capture_reduce(torch.cat(ordered, dim=0))
        rows = ordered[0].shape[0]
        output = capture[0]
        for slot, layer_index in enumerate(self.sharded_context_layers):
            column = self.context_layers.index(layer_index) * 2560
            output[:, column:column + 2560].copy_(reduced[slot * rows:(slot + 1) * rows])
        return reduced

    def functional(self, *, session):
        if getattr(self, "_functional_requested", False):
            return self
        self._functional_requested = True
        self.embedding.functional(session=session, dynamic_inputs=True)
        for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
            attn_hc.functional(session=session)
            if index % 4 != 3:
                attention.functional(session=session, dynamic_metadata=True)
            else:
                attention.functional(session=session)
            if mlp_hc is not None:
                mlp_hc.functional(session=session)
            if moe is not None:
                moe.functional(session=session)
        for pair in self.reductions:
            for reduction in pair:
                if reduction is not None:
                    reduction.functional(session=session)
        for group in self.pipelined_reductions:
            for reduction in group:
                if reduction is not None:
                    reduction.functional(session=session)
        self.ple.functional(session=session)
        if self.final_mixer is not None:
            self.final_mixer.functional(session=session)
        if self.row_shard_dma:
            if self.residual_output is not None and self.final_res_gather_a is None:
                from ignition.components.collectives import PreparedChunkedGatherShards

                if self._single_row_residual:
                    self._single_row_residual = False
                    self.state.prepare_residual(self.scratch.residual, single_row=False)
                self.final_res_gather_a = PreparedChunkedGatherShards(
                    self.scratch.residual[:self.shard_tokens], self.scratch.residual[:self.half_tokens],
                    runtime=self._runtime, session=session, coordinator=self._coordinator,
                    name=f"{self._name}.final_res_gather.a")
                self.final_res_gather_b = PreparedChunkedGatherShards(
                    self.scratch.residual[:self.shard_tokens], self.scratch.residual[:self.half_tokens],
                    runtime=self._runtime, session=session, coordinator=self._coordinator,
                    name=f"{self._name}.final_res_gather.b")
                self.final_row_reduce = None
            for gather in (self.final_hidden_gather_a, self.final_hidden_gather_b,
                           self.final_res_gather_a, self.final_res_gather_b, self.final_row_reduce):
                if gather is not None:
                    gather.functional(session=session)
            if self.context_capture is not None:
                self.sharded_context_layers = tuple(l for l in self.context_layers if l >= 1)
                if self.sharded_context_layers:
                    from ignition.components.collectives import PreparedChunkedAllReduce

                    rows_total = len(self.sharded_context_layers) * self.context_capture.shape[0]
                    dev = self.context_capture.device
                    ctx_in = torch.zeros((rows_total, 2560), dtype=torch.bfloat16, device=dev)
                    ctx_out = torch.empty_like(ctx_in)
                    probe = ctx_in[:min(rows_total, self.half_tokens)]
                    ctx_rt = (
                        self._runtime
                        if self._runtime is not None and self._runtime.should_allreduce(probe)
                        else self._barrier_runtime
                    )
                    self.context_capture_reduce = PreparedChunkedAllReduce(
                        ctx_in, ctx_out, runtime=ctx_rt, session=session,
                        coordinator=self._coordinator, name=f"{self._name}.context_capture_reduce",
                    ).functional(session=session)
        if self.lm_head is not None:
            self.lm_head.functional(session=session)
        if self.sampler is not None:
            self.sampler.functional(session=session)
        if self.context_capture is not None:
            self.context_capture = self.context_capture.functional()
        if self.embedding_injection is not None:
            self.embedding_injection.functional()
        return self

    def release_outputs(self):
        self.state.functional()
        self.scratch = self.residual_output = None
        self.auxiliary = None
        self._runtime = self._barrier_runtime = self._coordinator = None

    def _prepare_attn_shard(self, layer_index: int, mixed: torch.Tensor) -> torch.Tensor:
        if not self.layer_attn_mxfp8[layer_index]:
            return mixed
        from b12x.quantization import mxfp8

        local_vals = torch.empty(self.attn_local_shapes[0], dtype=torch.float8_e4m3fn, device=mixed.device)
        local_srows = torch.empty(self.attn_local_shapes[1], dtype=torch.uint8, device=mixed.device)
        local_smma = torch.empty(self.attn_local_shapes[2], dtype=torch.uint8, device=mixed.device)
        mxfp8.quantize_dense_rows(mixed, local_vals, local_srows, local_smma, plan=self.attn_quantizer_plan)
        srows_bf16 = local_srows.view(mixed.shape[0], -1).view(torch.bfloat16)
        if self.attn_pad_bf16_cols:
            return torch.cat(
                (local_vals.view(torch.bfloat16), srows_bf16, srows_bf16[:, :self.attn_pad_bf16_cols]),
                dim=1,
            )
        return torch.cat((local_vals.view(torch.bfloat16), srows_bf16), dim=1)

    def _unpack_attn_gathered(self, layer_index: int, packed_full: torch.Tensor):
        if not self.layer_attn_mxfp8[layer_index]:
            return packed_full, None
        full_rows = packed_full.shape[0]
        full_vals = packed_full[:, :self.attn_vals_bf16_cols].contiguous().view(torch.float8_e4m3fn)
        hidden = full_vals.shape[1]
        srows_end = self.attn_vals_bf16_cols + self.attn_scale_bf16_cols
        full_srows_2d = packed_full[:, self.attn_vals_bf16_cols:srows_end].contiguous().view(torch.uint8)
        full_srows = full_srows_2d.view(1, full_rows, hidden // 32)
        full_smma = _mxfp8_scale_mma(full_srows_2d, hidden)
        return full_vals, (full_vals, full_srows, full_smma)

    @staticmethod
    def _prepare_ffn_shard(moe, mixed_shard: torch.Tensor) -> torch.Tensor:
        from hyperflare_recipes.components.moe import RouteRecord

        route_ids, route_weights, gate = moe.route_shard_runtime(mixed_shard)
        packed_routes = RouteRecord.empty(route_ids.shape[0], route_ids.shape[1], route_ids.device)
        packed_routes.write(route_ids, route_weights)
        # Pad [st, 1] BF16 shared gate to 24 BF16 columns so total width is 2624 (32-aligned).
        return torch.cat((mixed_shard, packed_routes.packed, gate.expand(-1, 24)), dim=1)

    @staticmethod
    def _run_ffn_gathered(moe, packed_full: torch.Tensor) -> torch.Tensor:
        from hyperflare_recipes.components.moe import RouteRecord

        full_rows = packed_full.shape[0]
        mixed_full = packed_full[:, :2560].contiguous()
        route_words = packed_full[:, 2560:2600].contiguous().view(torch.int32)
        route_ids = torch.empty((full_rows, 10), dtype=torch.int32, device=packed_full.device)
        route_weights = torch.empty((full_rows, 10), dtype=torch.float32, device=packed_full.device)
        RouteRecord(route_words).read(route_ids, route_weights)
        gate_full = packed_full[:, 2600:2601]
        return moe.run_gathered_runtime(mixed_full, route_ids, route_weights, gate_full)

    def _run_row_shard_dma(self, runtime, residual, capture):
        h = self.half_tokens
        gdn_a, gdn_b, qsa_a, qsa_b = _split_prefill_metadata(runtime.gdn, runtime.qsa, h)
        # Layer 0 stays un-sharded so PLE runs on the full [tokens, 10240] residual before row sharding.
        attn_hc0, attention0, mlp_hc0, moe0 = self.layers[0]
        cr_att_0, cr_att_1, cr_mlp_0, cr_mlp_1 = self.pipelined_reductions[0]
        res_a, res_b = residual[:h], residual[h:]
        mixed_a, attn_inj_a = attn_hc0.mix_runtime(res_a, None)
        out_a = attention0(mixed_a, metadata=gdn_a)
        p_att_a = cr_att_0.start(out_a)
        mixed_b, attn_inj_b = attn_hc0.mix_runtime(res_b, None)
        out_b = attention0(mixed_b, metadata=gdn_b)
        att_a = cr_att_0.finish(p_att_a, out_b)
        p_att_b = cr_att_1.start(out_b, dependencies=(att_a,))
        res_a, norm_a = attn_hc0.combine_norm_runtime(res_a, att_a, attn_inj_a)
        mixed_a, mlp_inj_a = mlp_hc0.mix_runtime(res_a, norm_a)
        moe_a = moe0(mixed_a)
        att_b = cr_att_1.finish(p_att_b, moe_a)
        p_mlp_a = cr_mlp_0.start(moe_a, dependencies=(att_b,))
        res_b, norm_b = attn_hc0.combine_norm_runtime(res_b, att_b, attn_inj_b)
        mixed_b, mlp_inj_b = mlp_hc0.mix_runtime(res_b, norm_b)
        moe_b = moe0(mixed_b)
        mlp_a = cr_mlp_0.finish(p_mlp_a, moe_b)
        p_mlp_b = cr_mlp_1.start(moe_b, dependencies=(mlp_a,))
        res_a = mlp_hc0.combine_runtime(res_a, mlp_a, mlp_inj_a)
        mlp_b = cr_mlp_1.finish(p_mlp_b, res_a)
        res_b = mlp_hc0.combine_runtime(res_b, mlp_b, mlp_inj_b)
        residual = torch.cat((res_a, res_b), dim=0)
        if 0 in self.context_layers:
            self._capture_context_layer(capture, 0, residual)

        if hasattr(self.ple, "add_to_residual"):
            residual, _ = self.ple.add_to_residual(residual, runtime.ple, getattr(runtime, "ple_partitions", None))
        else:
            residual.add_(self.ple(residual, runtime.ple, getattr(runtime, "ple_partitions", None)))
        res_a = torch.index_select(residual, 0, self.shard_rows_a)
        res_b = torch.index_select(residual, 0, self.shard_rows_b)
        sharded_ctx_slots: dict[int, torch.Tensor] = {}
        last_layer = len(self.layers) - 1

        # Prologue for Layer 1: gather half-A attention input while preparing half-B so the
        # wavefront loop enters every layer with attn_a_full ready.
        attn_hc1 = self.layers[1][0]
        cr_ag_0_1 = self.pipelined_reductions[1][0]
        mixed_a_sh, attn_inj_a = attn_hc1.mix_runtime(res_a, None)
        attn_a_send = self._prepare_attn_shard(1, mixed_a_sh)
        p_ag_a = cr_ag_0_1.start(attn_a_send)
        mixed_b_sh, attn_inj_b = attn_hc1.mix_runtime(res_b, None)
        attn_b_send = self._prepare_attn_shard(1, mixed_b_sh)
        attn_a_full = cr_ag_0_1.finish(p_ag_a, attn_b_send)
        pending_prev_b = None

        for index in range(1, len(self.layers)):
            attn_hc, attention, mlp_hc, moe = self.layers[index]
            (_, cr_ag_1, cr_att_0, cr_att_1,
             cr_ffn_0, cr_ffn_1, cr_mlp_0, cr_mlp_1, cr_att_1b) = self.pipelined_reductions[index]
            is_qsa = index % 4 == 3
            has_row_gather = is_qsa and attention.sharded is not None and attention.sharded.mode == "row"
            src_a_att, enc_a_att = self._unpack_attn_gathered(index, attn_a_full)

            if mlp_hc is None:
                attention(src_a_att, qsa_a, input_encoded_bases=enc_a_att)
                if pending_prev_b is not None:
                    prev_cr_mlp_1, prev_mlp_hc, prev_p_mlp_b, prev_res_b, prev_inj_b = pending_prev_b
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, attention.state.compressed_k_cache)
                    res_b, norm_b = prev_mlp_hc.combine_norm_runtime(prev_res_b, prev_mlp_b, prev_inj_b)
                    if (index - 1) in self.context_layers:
                        self._capture_context_layer_sharded(capture, sharded_ctx_slots, index - 1, res_a, res_b)
                    mixed_b_sh, _ = attn_hc.mix_runtime(res_b, norm_b)
                    attn_b_send = self._prepare_attn_shard(index, mixed_b_sh)
                    ag_b_deps = (prev_mlp_b, attn_b_send)
                else:
                    ag_b_deps = (attn_a_full, attn_b_send)
                p_ag_b = cr_ag_1.start(attn_b_send, dependencies=ag_b_deps)
                attn_b_full = cr_ag_1.finish(p_ag_b, attn_b_send)
                src_b_att, enc_b_att = self._unpack_attn_gathered(index, attn_b_full)
                attention(src_b_att, qsa_b, input_encoded_bases=enc_b_att)
                self._finish_sharded_context_capture(capture, sharded_ctx_slots)
                return residual, None

            if not is_qsa:
                # Step 1: overlap half-A input projections with previous layer's half-B MoE RS.
                proj_a = attention.project_input_runtime(src_a_att, input_encoded_bases=enc_a_att)
                if pending_prev_b is not None:
                    prev_cr_mlp_1, prev_mlp_hc, prev_p_mlp_b, prev_res_b, prev_inj_b = pending_prev_b
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, proj_a[0])
                    res_b, norm_b = prev_mlp_hc.combine_norm_runtime(prev_res_b, prev_mlp_b, prev_inj_b)
                    if (index - 1) in self.context_layers:
                        self._capture_context_layer_sharded(capture, sharded_ctx_slots, index - 1, res_a, res_b)
                    mixed_b_sh, attn_inj_b = attn_hc.mix_runtime(res_b, norm_b)
                    attn_b_send = self._prepare_attn_shard(index, mixed_b_sh)
                    ag_b_deps = (prev_mlp_b, attn_b_send)
                    pending_prev_b = None
                else:
                    ag_b_deps = (attn_a_full, attn_b_send)
                # Step 2: overlap half-B attention gather with half-A GDN sequence & out_proj.
                p_ag_b = cr_ag_1.start(attn_b_send, dependencies=ag_b_deps)
                out_a = attention.finish_sequence_runtime(
                    *proj_a[:4], qkvz=proj_a[4], ba=proj_a[5], metadata=gdn_a,
                )
                attn_b_full = cr_ag_1.finish(p_ag_b, out_a)
                # Step 3: overlap half-A attention RS with half-B input projections.
                p_att_a = cr_att_0.start(out_a, dependencies=(attn_b_full,))
                src_b_att, enc_b_att = self._unpack_attn_gathered(index, attn_b_full)
                proj_b = attention.project_input_runtime(src_b_att, input_encoded_bases=enc_b_att)
                att_a = cr_att_0.finish(p_att_a, proj_b[0])
                # Step 4: prepare half-A FFN shard, then overlap half-A FFN gather with half-B GDN sequence.
                res_a, norm_a = attn_hc.combine_norm_runtime(res_a, att_a, attn_inj_a)
                mixed_a_sh, mlp_inj_a = mlp_hc.mix_runtime(res_a, norm_a)
                ffn_a_send = self._prepare_ffn_shard(moe, mixed_a_sh)
                p_ffn_a = cr_ffn_0.start(ffn_a_send, dependencies=(att_a, ffn_a_send))
                out_b = attention.finish_sequence_runtime(
                    *proj_b[:4], qkvz=proj_b[4], ba=proj_b[5], metadata=gdn_b,
                )
                if cr_att_1b is not None:
                    out_b_chunks = out_b.view(self.world_size, 2, self.shard_tokens // 2, 2560).transpose(0, 1).contiguous()
                    ffn_a_full = cr_ffn_0.finish(p_ffn_a, out_b_chunks)
                else:
                    out_b_chunks = None
                    ffn_a_full = cr_ffn_0.finish(p_ffn_a, out_b)
            elif has_row_gather:
                # Step 1: overlap half-A QKV/selection with previous layer's half-B MoE RS, then gather
                # QSA group IDs with zero DMA in flight to avoid PCIe contention.
                stage_a = attention.pre_gather_runtime(src_a_att, qsa_a, input_encoded_bases=enc_a_att)
                if pending_prev_b is not None:
                    prev_cr_mlp_1, prev_mlp_hc, prev_p_mlp_b, prev_res_b, prev_inj_b = pending_prev_b
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, stage_a.send)
                    gathered_a = attention.gather_row_runtime(stage_a, gather_dependencies=(prev_mlp_b,))
                    res_b, norm_b = prev_mlp_hc.combine_norm_runtime(prev_res_b, prev_mlp_b, prev_inj_b)
                    if (index - 1) in self.context_layers:
                        self._capture_context_layer_sharded(capture, sharded_ctx_slots, index - 1, res_a, res_b)
                    mixed_b_sh, attn_inj_b = attn_hc.mix_runtime(res_b, norm_b)
                    attn_b_send = self._prepare_attn_shard(index, mixed_b_sh)
                    ag_b_deps = (gathered_a, attn_b_send)
                    pending_prev_b = None
                else:
                    gathered_a = attention.gather_row_runtime(stage_a, gather_dependencies=(attn_a_full,))
                    ag_b_deps = (gathered_a, attn_b_send)
                # Step 2: overlap half-B attention gather with half-A sparse attention & o_proj.
                p_ag_b = cr_ag_1.start(attn_b_send, dependencies=ag_b_deps)
                out_a = attention.attend_project_runtime(stage_a, gathered_a)
                attn_b_full = cr_ag_1.finish(p_ag_b, out_a)
                # Step 3: overlap half-A attention RS with half-B QKV/selection, then gather group IDs.
                p_att_a = cr_att_0.start(out_a, dependencies=(attn_b_full,))
                src_b_att, enc_b_att = self._unpack_attn_gathered(index, attn_b_full)
                stage_b = attention.pre_gather_runtime(src_b_att, qsa_b, input_encoded_bases=enc_b_att)
                att_a = cr_att_0.finish(p_att_a, stage_b.send)
                gathered_b = attention.gather_row_runtime(stage_b, gather_dependencies=(att_a,))
                # Step 4: prepare half-A FFN shard, then overlap half-A FFN gather with half-B sparse attention.
                res_a, norm_a = attn_hc.combine_norm_runtime(res_a, att_a, attn_inj_a)
                mixed_a_sh, mlp_inj_a = mlp_hc.mix_runtime(res_a, norm_a)
                ffn_a_send = self._prepare_ffn_shard(moe, mixed_a_sh)
                p_ffn_a = cr_ffn_0.start(ffn_a_send, dependencies=(gathered_b, ffn_a_send))
                out_b = attention.attend_project_runtime(stage_b, gathered_b)
                if cr_att_1b is not None:
                    out_b_chunks = out_b.view(self.world_size, 2, self.shard_tokens // 2, 2560).transpose(0, 1).contiguous()
                    ffn_a_full = cr_ffn_0.finish(p_ffn_a, out_b_chunks)
                else:
                    out_b_chunks = None
                    ffn_a_full = cr_ffn_0.finish(p_ffn_a, out_b)
            else:
                out_a = attention(src_a_att, qsa_a, input_encoded_bases=enc_a_att)
                if pending_prev_b is not None:
                    prev_cr_mlp_1, prev_mlp_hc, prev_p_mlp_b, prev_res_b, prev_inj_b = pending_prev_b
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, out_a)
                    res_b, norm_b = prev_mlp_hc.combine_norm_runtime(prev_res_b, prev_mlp_b, prev_inj_b)
                    if (index - 1) in self.context_layers:
                        self._capture_context_layer_sharded(capture, sharded_ctx_slots, index - 1, res_a, res_b)
                    mixed_b_sh, attn_inj_b = attn_hc.mix_runtime(res_b, norm_b)
                    attn_b_send = self._prepare_attn_shard(index, mixed_b_sh)
                    ag_b_deps = (prev_mlp_b, attn_b_send)
                    pending_prev_b = None
                else:
                    ag_b_deps = (attn_a_full, attn_b_send)
                p_ag_b = cr_ag_1.start(attn_b_send, dependencies=ag_b_deps)
                attn_b_full = cr_ag_1.finish(p_ag_b, attn_b_send)
                p_att_a = cr_att_0.start(out_a, dependencies=(attn_b_full,))
                src_b_att, enc_b_att = self._unpack_attn_gathered(index, attn_b_full)
                out_b = attention(src_b_att, qsa_b, input_encoded_bases=enc_b_att)
                att_a = cr_att_0.finish(p_att_a, out_b)
                res_a, norm_a = attn_hc.combine_norm_runtime(res_a, att_a, attn_inj_a)
                mixed_a_sh, mlp_inj_a = mlp_hc.mix_runtime(res_a, norm_a)
                ffn_a_send = self._prepare_ffn_shard(moe, mixed_a_sh)
                p_ffn_a = cr_ffn_0.start(ffn_a_send, dependencies=(att_a, ffn_a_send))
                ffn_a_full = cr_ffn_0.finish(p_ffn_a, ffn_a_send)
                out_b_chunks = (
                    out_b.view(self.world_size, 2, self.shard_tokens // 2, 2560).transpose(0, 1).contiguous()
                    if cr_att_1b is not None else None
                )

            # Step 5: overlap half-B attention RS with half-A MoE.
            # When split_att_rs_1 is active, splitting out_b into two half-size reduce-scatters
            # lets chunk-0 piece-0 DMA finish before A4_fc1 (grid=188) launches, and staging
            # chunk-0's output prefix on the auxiliary stream keeps chunk-1's flag-wait kernel
            # behind A4_fc2 (grid=188), avoiding CWD head-of-line starvation on both MoE GEMMs.
            if cr_att_1b is not None:
                half_in = self.half_tokens // 2
                half_st = self.shard_tokens // 2
                att_b = torch.empty((self.shard_tokens, 2560), dtype=torch.bfloat16, device=out_b.device)
                p_att_b0 = cr_att_1.start(out_b_chunks[0].view(half_in, 2560), dependencies=(ffn_a_full,))
                _aux_copy_prefix(p_att_b0.output, att_b, cr_att_1.stream_handle)
                p_att_b1 = cr_att_1b.start(out_b_chunks[1].view(half_in, 2560),
                                           dependencies=(ffn_a_full, att_b))
                moe_a = self._run_ffn_gathered(moe, ffn_a_full)
                att_b_0 = cr_att_1.finish(p_att_b0, moe_a)
                att_b_1 = cr_att_1b.finish(p_att_b1, att_b_0)
                # Step 6: overlap half-A MoE RS with half-B FFN shard preparation.
                p_mlp_a = cr_mlp_0.start(moe_a, dependencies=(att_b_0, att_b_1))
                att_b[half_st:].copy_(att_b_1)
            else:
                p_att_b = cr_att_1.start(out_b, dependencies=(ffn_a_full,))
                moe_a = self._run_ffn_gathered(moe, ffn_a_full)
                att_b = cr_att_1.finish(p_att_b, moe_a)
                # Step 6: overlap half-A MoE RS with half-B FFN shard preparation.
                p_mlp_a = cr_mlp_0.start(moe_a, dependencies=(att_b,))
            res_b, norm_b = attn_hc.combine_norm_runtime(res_b, att_b, attn_inj_b)
            mixed_b_sh, mlp_inj_b = mlp_hc.mix_runtime(res_b, norm_b)
            ffn_b_send = self._prepare_ffn_shard(moe, mixed_b_sh)
            mlp_a = cr_mlp_0.finish(p_mlp_a, ffn_b_send)

            if index < last_layer:
                next_attn_hc = self.layers[index + 1][0]
                next_cr_ag_0 = self.pipelined_reductions[index + 1][0]
                # Step 7: overlap half-B FFN gather with next layer's half-A attention shard prep.
                p_ffn_b = cr_ffn_1.start(ffn_b_send, dependencies=(mlp_a,))
                res_a, norm_a = mlp_hc.combine_norm_runtime(res_a, mlp_a, mlp_inj_a)
                next_mixed_a_sh, next_attn_inj_a = next_attn_hc.mix_runtime(res_a, norm_a)
                next_attn_a_send = self._prepare_attn_shard(index + 1, next_mixed_a_sh)
                ffn_b_full = cr_ffn_1.finish(p_ffn_b, next_attn_a_send)

                # Step 8: overlap next layer's half-A attention gather with current half-B MoE.
                p_next_ag_a = next_cr_ag_0.start(next_attn_a_send, dependencies=(ffn_b_full,))
                moe_b = self._run_ffn_gathered(moe, ffn_b_full)
                attn_a_full = next_cr_ag_0.finish(p_next_ag_a, moe_b)

                p_mlp_b = cr_mlp_1.start(moe_b, dependencies=(attn_a_full,))
                attn_inj_a = next_attn_inj_a
                pending_prev_b = (cr_mlp_1, mlp_hc, p_mlp_b, res_b, mlp_inj_b)
            else:
                p_ffn_b = cr_ffn_1.start(ffn_b_send, dependencies=(mlp_a,))
                if self.sample and not self._single_row_final_mixer:
                    res_a, norm_a = mlp_hc.combine_norm_runtime(res_a, mlp_a, mlp_inj_a)
                else:
                    res_a = mlp_hc.combine_runtime(res_a, mlp_a, mlp_inj_a)
                ffn_b_full = cr_ffn_1.finish(p_ffn_b, res_a)
                moe_b = self._run_ffn_gathered(moe, ffn_b_full)
                p_mlp_b = cr_mlp_1.start(moe_b, dependencies=(ffn_b_full, moe_b))
                mlp_b = cr_mlp_1.finish(p_mlp_b, moe_b)
                if self.sample and not self._single_row_final_mixer:
                    res_b, norm_b = mlp_hc.combine_norm_runtime(res_b, mlp_b, mlp_inj_b)
                else:
                    res_b = mlp_hc.combine_runtime(res_b, mlp_b, mlp_inj_b)
                if index in self.context_layers:
                    self._capture_context_layer_sharded(capture, sharded_ctx_slots, index, res_a, res_b)

        ctx_reduced = self._finish_sharded_context_capture(capture, sharded_ctx_slots)
        last_dep = () if ctx_reduced is None else (ctx_reduced,)
        last_res = None
        if self.final_res_gather_a is not None:
            full_res_a = self.final_res_gather_a(res_a, dependencies=last_dep)
            full_res_b = self.final_res_gather_b(res_b, dependencies=(full_res_a,))
            residual = torch.cat((full_res_a, full_res_b), dim=0)
            last_dep = (full_res_b,)
            if self._single_row_final_mixer:
                last_idx = (runtime.gdn.query_start_loc[1:2] - 1).clamp(0, self.tokens - 1).to(torch.int64)
                last_res = torch.index_select(residual, 0, last_idx)
        elif self.final_row_reduce is not None:
            last_idx = (runtime.gdn.query_start_loc[1:2] - 1).clamp(0, self.tokens - 1).to(torch.int64)
            s0, s1, st = self.shard_offset_a, self.shard_offset_b, self.shard_tokens
            in_a = (last_idx >= s0) & (last_idx < s0 + st)
            in_b = (last_idx >= s1) & (last_idx < s1 + st)
            row_a = torch.index_select(res_a, 0, (last_idx - s0).clamp(0, st - 1))
            row_b = torch.index_select(res_b, 0, (last_idx - s1).clamp(0, st - 1))
            local_row = torch.where(
                in_a[:, None], row_a, torch.where(in_b[:, None], row_b, torch.zeros_like(row_a)),
            )
            last_res = self.final_row_reduce(local_row, dependencies=last_dep)
            if self._single_row_residual:
                residual = last_res
        if not self.sample:
            return residual, None
        if self._single_row_final_mixer:
            hidden, _ = self.final_mixer.mix_runtime(last_res, None)
            return residual, hidden
        hid_a_sh, _ = self.final_mixer.mix_runtime(res_a, norm_a)
        hid_b_sh, _ = self.final_mixer.mix_runtime(res_b, norm_b)
        hid_a = self.final_hidden_gather_a(hid_a_sh, dependencies=last_dep)
        hid_b = self.final_hidden_gather_b(hid_b_sh, dependencies=(hid_a,))
        return residual, torch.cat((hid_a, hid_b), dim=0)

    def _run_pipelined(self, runtime, residual, capture):
        h = self.half_tokens
        gdn_a, gdn_b, qsa_a, qsa_b = _split_prefill_metadata(runtime.gdn, runtime.qsa, h)
        res_a, res_b = residual[:h], residual[h:]
        norm_a = norm_b = None
        normalized = None
        pending_prev = None
        last_layer = len(self.layers) - 1
        for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
            cr_att_0, cr_att_1, cr_mlp_0, cr_mlp_1 = self.pipelined_reductions[index]
            is_qsa = index % 4 == 3
            has_row_gather = is_qsa and attention.sharded is not None and attention.sharded.mode == "row"
            if index == 1:
                if hasattr(self.ple, "add_to_residual"):
                    residual, _ = self.ple.add_to_residual(residual, runtime.ple, getattr(runtime, "ple_partitions", None))
                else:
                    residual.add_(self.ple(residual, runtime.ple, getattr(runtime, "ple_partitions", None)))
                res_a, res_b = residual[:h], residual[h:]
                norm_a = norm_b = None
            if mlp_hc is None:
                if pending_prev is not None:
                    prev_cr_mlp_1, prev_mlp_hc, prev_p_mlp_b, prev_res_a, prev_mlp_a, prev_inj_a, prev_res_b, prev_inj_b = pending_prev
                    res_a, norm_a = prev_mlp_hc.combine_norm_runtime(prev_res_a, prev_mlp_a, prev_inj_a)
                mixed_a, _ = attn_hc.mix_runtime(res_a, norm_a)
                attention(mixed_a, qsa_a)
                if pending_prev is not None:
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, attention.state.compressed_k_cache)
                    res_b, norm_b = prev_mlp_hc.combine_norm_runtime(prev_res_b, prev_mlp_b, prev_inj_b)
                    if (index - 1) in self.context_layers:
                        self._capture_context_layer(capture, index - 1, res_a, res_b)
                    pending_prev = None
                mixed_b, _ = attn_hc.mix_runtime(res_b, norm_b)
                attention(mixed_b, qsa_b)
                return residual, None
            if pending_prev is not None:
                prev_cr_mlp_1, prev_mlp_hc, prev_p_mlp_b, prev_res_a, prev_mlp_a, prev_inj_a, prev_res_b, prev_inj_b = pending_prev
                res_a, norm_a = prev_mlp_hc.combine_norm_runtime(prev_res_a, prev_mlp_a, prev_inj_a)
                if self.layer_outputs is not None:
                    self.layer_outputs[index - 1][:h].copy_(res_a)
            mixed_a, attn_inj_a = attn_hc.mix_runtime(res_a, norm_a)
            if not is_qsa:
                out_a = attention(mixed_a, metadata=gdn_a)
                if pending_prev is not None:
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, out_a)
                    att0_deps = (prev_mlp_b,)
                else:
                    att0_deps = ()
            elif has_row_gather:
                stage_a = attention.pre_gather_runtime(mixed_a, qsa_a)
                if pending_prev is not None:
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, stage_a.send)
                    gather_deps_a = (prev_mlp_b,)
                else:
                    gather_deps_a = ()
                out_a = attention.post_gather_runtime(stage_a, gather_dependencies=gather_deps_a)
                att0_deps = (out_a,)
            else:
                out_a = attention(mixed_a, qsa_a)
                if pending_prev is not None:
                    prev_mlp_b = prev_cr_mlp_1.finish(prev_p_mlp_b, out_a)
                    att0_deps = (prev_mlp_b,)
                else:
                    att0_deps = ()

            p_att_a = cr_att_0.start(out_a, dependencies=att0_deps)
            if pending_prev is not None:
                res_b, norm_b = prev_mlp_hc.combine_norm_runtime(prev_res_b, prev_mlp_b, prev_inj_b)
                if self.layer_outputs is not None:
                    self.layer_outputs[index - 1][h:].copy_(res_b)
                if (index - 1) in self.context_layers:
                    self._capture_context_layer(capture, index - 1, res_a, res_b)
                pending_prev = None
            mixed_b, attn_inj_b = attn_hc.mix_runtime(res_b, norm_b)
            if not is_qsa:
                out_b = attention(mixed_b, metadata=gdn_b)
                att_a = cr_att_0.finish(p_att_a, out_b)
            elif has_row_gather:
                stage_b = attention.pre_gather_runtime(mixed_b, qsa_b)
                att_a = cr_att_0.finish(p_att_a, stage_b.send)
                out_b = attention.post_gather_runtime(stage_b, gather_dependencies=(att_a,))
            else:
                out_b = attention(mixed_b, qsa_b)
                att_a = cr_att_0.finish(p_att_a, out_b)

            p_att_b = cr_att_1.start(out_b, dependencies=(att_a,))
            res_a, norm_a = attn_hc.combine_norm_runtime(res_a, att_a, attn_inj_a)
            mixed_a, mlp_inj_a = mlp_hc.mix_runtime(res_a, norm_a)
            moe_a = moe(mixed_a)
            att_b = cr_att_1.finish(p_att_b, moe_a)

            p_mlp_a = cr_mlp_0.start(moe_a, dependencies=(att_b,))
            res_b, norm_b = attn_hc.combine_norm_runtime(res_b, att_b, attn_inj_b)
            mixed_b, mlp_inj_b = mlp_hc.mix_runtime(res_b, norm_b)
            moe_b = moe(mixed_b)
            mlp_a = cr_mlp_0.finish(p_mlp_a, moe_b)

            p_mlp_b = cr_mlp_1.start(moe_b, dependencies=(mlp_a,))
            if index == 0:
                res_a = mlp_hc.combine_runtime(res_a, mlp_a, mlp_inj_a)
                mlp_b = cr_mlp_1.finish(p_mlp_b, res_a)
                res_b = mlp_hc.combine_runtime(res_b, mlp_b, mlp_inj_b)
                residual = torch.cat((res_a, res_b), dim=0)
                if self.layer_outputs is not None:
                    self.layer_outputs[0].copy_(residual)
                if 0 in self.context_layers:
                    self._capture_context_layer(capture, 0, residual)
            elif index < last_layer:
                pending_prev = (cr_mlp_1, mlp_hc, p_mlp_b, res_a, mlp_a, mlp_inj_a, res_b, mlp_inj_b)
            else:
                if self.sample and not self._single_row_final_mixer:
                    res_a, norm_a = mlp_hc.combine_norm_runtime(res_a, mlp_a, mlp_inj_a)
                    mlp_b = cr_mlp_1.finish(p_mlp_b, res_a)
                    res_b, norm_b = mlp_hc.combine_norm_runtime(res_b, mlp_b, mlp_inj_b)
                    residual = torch.cat((res_a, res_b), dim=0)
                    normalized = torch.cat((norm_a, norm_b), dim=0)
                else:
                    res_a = mlp_hc.combine_runtime(res_a, mlp_a, mlp_inj_a)
                    mlp_b = cr_mlp_1.finish(p_mlp_b, res_a)
                    res_b = mlp_hc.combine_runtime(res_b, mlp_b, mlp_inj_b)
                    residual = torch.cat((res_a, res_b), dim=0)
                    normalized = None
                if self.layer_outputs is not None:
                    self.layer_outputs[index].copy_(residual)
                if index in self.context_layers:
                    self._capture_context_layer(capture, index, residual)
        return residual, normalized

    def __call__(self, runtime):
        from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords
        self.ple.embedding_lookup.publish(runtime.ple)
        embedding = self.embedding(token_ids=runtime.ple.token_ids, num_rows=runtime.gdn.num_tokens)
        if self.embedding_injection is not None:
            self.embedding_injection.inject(embedding, runtime.injection_errors)
        residual = torch.empty((embedding.shape[0], 10240), dtype=embedding.dtype, device=embedding.device)
        residual.view(-1, 4, 2560).copy_(embedding[:, None, :])
        records = None
        if self.record_layout is not None:
            layers, rows, key_heads, value_heads, width = self.record_layout
            records = GDNLayerRecords.empty(layers, rows, embedding.device,
                key_heads=key_heads, value_heads=value_heads, width=width)
        capture = None if self.context_capture is None else self.context_capture.begin(runtime.capture_metadata)
        hidden = None
        if self.row_shard_dma:
            residual, hidden = self._run_row_shard_dma(runtime, residual, capture)
            normalized = None
        elif self.pipelined_prefill:
            residual, normalized = self._run_pipelined(runtime, residual, capture)
        else:
            normalized = None
            prefetched_mix = None
            for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
                if index == 1:
                    if hasattr(self.ple, "add_to_residual"):
                        residual, normalized = self.ple.add_to_residual(
                            residual, runtime.ple, getattr(runtime, "ple_partitions", None),
                        )
                    else:
                        residual.add_(self.ple(residual, runtime.ple, getattr(runtime, "ple_partitions", None)))
                        normalized = None
                    prefetched_mix = None
                if prefetched_mix is not None:
                    mixed, injection = prefetched_mix
                    prefetched_mix = None
                else:
                    mixed, injection = attn_hc.mix_runtime(residual, normalized)
                if mlp_hc is None:
                    attention(mixed, runtime.qsa)
                    normalized = None
                    break
                if index % 4 != 3:
                    layer_records = None if records is None or index not in records.layers else records[index]
                    block = attention(mixed, records=layer_records, metadata=runtime.gdn,
                                      partitions=getattr(runtime, "gdn_partitions", None))
                else:
                    block = attention(mixed, runtime.qsa)
                if self.reductions:
                    block = self.reductions[index][0](block)
                if attn_hc.has_combine_mix:
                    residual, mixed, injection = attn_hc.combine_mix_runtime(residual, block, injection)
                    normalized = None
                else:
                    residual, normalized = attn_hc.combine_norm_runtime(residual, block, injection)
                    mixed, injection = mlp_hc.mix_runtime(residual, normalized)
                if index == 0 or ((not self.sample or self._single_row_final_mixer) and index + 1 == len(self.layers)):
                    block = moe(mixed)
                    if self.reductions:
                        block = self.reductions[index][1](block)
                    residual = mlp_hc.combine_runtime(residual, block, injection)
                    normalized = None
                elif not self.reductions and mlp_hc.has_combine_mix_mlp:
                    routed, shared_proj, shared_gate = moe.parts_runtime(mixed)
                    residual, next_mixed, next_inj = mlp_hc.combine_mix_mlp_runtime(
                        residual, routed, shared_proj, shared_gate, injection,
                    )
                    normalized = None
                    prefetched_mix = (next_mixed, next_inj)
                elif mlp_hc.has_combine_mix:
                    block = moe(mixed)
                    if self.reductions:
                        block = self.reductions[index][1](block)
                    residual, next_mixed, next_inj = mlp_hc.combine_mix_runtime(residual, block, injection)
                    normalized = None
                    prefetched_mix = (next_mixed, next_inj)
                elif not self.reductions and getattr(mlp_hc, "_triton_combine_norm_mlp", None) is not None:
                    routed, shared_proj, shared_gate = moe.parts_runtime(mixed)
                    residual, normalized = mlp_hc.combine_norm_mlp_runtime(
                        residual, routed, shared_proj, shared_gate, injection,
                    )
                else:
                    block = moe(mixed)
                    if self.reductions:
                        block = self.reductions[index][1](block)
                    residual, normalized = mlp_hc.combine_norm_runtime(residual, block, injection)
                if self.layer_outputs is not None:
                    self.layer_outputs[index].copy_(residual)
                if index in self.context_layers:
                    self._capture_context_layer(capture, index, residual)
            if prefetched_mix is not None and self.sample and hidden is None and not self._single_row_final_mixer:
                hidden = prefetched_mix[0]
        if not self.sample:
            self.state.finish_runtime(runtime, None, residual)
            if self.completion_recorder is not None:
                self.completion_recorder()
            return None, None, DecoderAuxiliary(residual, None if capture is None else capture[0]), records
        if hidden is None:
            if self._single_row_final_mixer:
                last_idx = (runtime.gdn.query_start_loc[1:2] - 1).clamp(0, self.tokens - 1).to(torch.int64)
                last_res = torch.index_select(residual, 0, last_idx)
                hidden, _ = self.final_mixer.mix_runtime(last_res, None)
            else:
                hidden, _ = self.final_mixer.mix_runtime(residual, normalized)
        if self.scoring is not None:
            self.scoring(hidden)
        sample = self.state.finish_runtime(runtime, hidden, residual)
        logits = self.lm_head(sample)
        if self.sampler is not None:
            self.sampler.run(logits)
        return sample, logits, DecoderAuxiliary(residual, None if capture is None else capture[0]), records
