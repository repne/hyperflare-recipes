# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity tests for SM120/SM121 Triton hyperconnection gate kernels."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("FLARE_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="requires FLARE_RUN_CUDA_TESTS=1 and a CUDA device",
)


def _reference_gate(projected_down: torch.Tensor, weight: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
    from b12x.norm.hyperconnection import reference

    bottleneck = reference.scaled_silu(projected_down.contiguous(), streams=4)
    logits = torch.nn.functional.linear(bottleneck, weight)
    return reference.gate_mean(normalized, logits, streams=4)


def _make_hc_module(device: torch.device, *, has_injection: bool, seed: int) -> torch.nn.Module:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    module = torch.nn.Module()
    module.hc_count = 4
    module.hidden_size = 2560
    module.hc_norm = torch.nn.Module()
    module.hc_norm.weight = torch.nn.Parameter((torch.randn(10240, generator=gen) * 0.05).to(device), requires_grad=False)
    module.hc_norm.eps = 1e-6
    module.input_mix_weight_down = torch.nn.Module()
    module.input_mix_weight_down.weight = torch.nn.Parameter(
        (torch.randn(320, 10240, generator=gen) * 0.01).bfloat16().to(device), requires_grad=False
    )
    module.input_mix_weight_up = torch.nn.Module()
    module.input_mix_weight_up.weight = torch.nn.Parameter(
        (torch.randn(10240, 320, generator=gen) * 0.05).bfloat16().to(device), requires_grad=False
    )
    if has_injection:
        module.block_inject_weight = torch.nn.Module()
        module.block_inject_weight.weight = torch.nn.Parameter(
            (torch.randn(4, 10240, generator=gen) * 0.01).bfloat16().to(device), requires_grad=False
        )
    else:
        module.block_inject_weight = None
    return module


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 16, 32, 64, 100, 128, 2048, 4096])
@pytest.mark.parametrize("strided", [True, False])
def test_prepared_hyperconnection_gate_matches_reference(rows: int, strided: bool):
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.components.hyperconnection_gate import PreparedHyperconnectionGate

    device = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(1000 + rows + int(strided))
    weight = (torch.randn((10240, 320), generator=gen) * 0.05).bfloat16().to(device).contiguous()
    packed_proj = (torch.randn((rows, 336 if strided else 320), generator=gen) * 0.5).bfloat16().to(device)
    proj_down = packed_proj[:, :320]
    normalized = (torch.randn((rows, 10240), generator=gen) * 0.5).bfloat16().to(device).contiguous()

    from b12x.norm.hyperconnection import reference

    bottleneck = reference.scaled_silu(proj_down.contiguous(), streams=4)
    expected = _reference_gate(proj_down, weight, normalized)

    with torch.inference_mode(), RecordedPreparationSession(
        device=device,
        checkpoint=f"test-hc-gate-{rows}-{int(strided)}",
        workload_schema="test-hc-gate",
        numerical_policy="bf16",
        autotune_effort="off",
    ) as session:
        src_example = torch.zeros((rows, 320), dtype=torch.bfloat16, device=device)
        proj_example = torch.zeros((rows, 336), dtype=torch.bfloat16, device=device)[:, :320]
        norm_example = torch.zeros((rows, 10240), dtype=torch.bfloat16, device=device)
        out_example = torch.zeros((rows, 2560), dtype=torch.bfloat16, device=device)
        ws_example = torch.zeros((rows, 10240), dtype=torch.bfloat16, device=device)
        gate = PreparedHyperconnectionGate(
            src_example, weight, norm_example, out_example, ws_example,
            session=session, name="gate", projected_down=proj_example,
        )
        gate.functional(session=session)
        if gate.plan is not None:
            session.tune(show_progress=False)

        cap = torch.cuda.get_device_capability(device)
        if cap in ((12, 0), (12, 1)):
            if rows == 1:
                assert gate.implementation == "triton_m1_gemv" and gate.fuses_scaled_silu
            elif rows >= 64:
                assert gate.implementation == "triton_bptr_4acc" and not gate.fuses_scaled_silu
            else:
                assert gate.plan is not None and not gate.fuses_scaled_silu

        gate_in = proj_down if gate.fuses_scaled_silu else bottleneck
        eager_out = gate(gate_in, normalized)
        torch.testing.assert_close(eager_out, expected, rtol=1e-2, atol=2e-2)

        def run():
            return gate(gate_in, normalized)

        compiled = aot_forward(run)
        compiled_out = compiled(run)
        torch.testing.assert_close(compiled_out, eager_out, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("rows", [1, 8, 16, 128, 2048])
@pytest.mark.parametrize("has_injection", [True, False])
def test_prepared_hyperconnection_mix_runtime_compiled_parity(rows: int, has_injection: bool):
    from b12x.norm.hyperconnection import reference
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.qwen38.hyperconnection import (
        HyperconnectionScratch,
        PreparedHyperconnection,
    )

    device = torch.device("cuda:0")
    module = _make_hc_module(device, has_injection=has_injection, seed=2000 + rows + int(has_injection))
    gen = torch.Generator(device="cpu").manual_seed(3000 + rows)
    residual = (torch.randn((rows, 10240), generator=gen) * 0.25).bfloat16().to(device).contiguous()

    expected_norm = reference.grouped_rmsnorm(residual, module.hc_norm.weight, streams=4, eps=1e-6)
    expected_down = torch.nn.functional.linear(expected_norm, module.input_mix_weight_down.weight)
    expected_bottleneck = reference.scaled_silu(expected_down, streams=4)
    expected_logits = torch.nn.functional.linear(expected_bottleneck, module.input_mix_weight_up.weight)
    expected_mixed = reference.gate_mean(expected_norm, expected_logits, streams=4)
    expected_inj = (
        torch.nn.functional.linear(expected_norm, module.block_inject_weight.weight)
        if has_injection
        else None
    )

    with torch.inference_mode(), RecordedPreparationSession(
        device=device,
        checkpoint=f"test-hc-mix-{rows}-{int(has_injection)}",
        workload_schema="test-hc-mix",
        numerical_policy="bf16",
        autotune_effort="off",
    ) as session:
        scratch = HyperconnectionScratch.allocate(rows, device)
        priming = torch.zeros_like(scratch.normalized)
        hc = PreparedHyperconnection(module, scratch, priming_state=priming, session=session, name="hc")
        hc.functional(session=session)
        session.tune(show_progress=False)

        out_buf = torch.empty((rows, 2560), dtype=torch.bfloat16, device=device)
        inj_buf = torch.empty((rows, 4), dtype=torch.bfloat16, device=device) if has_injection else None

        def run():
            mixed, inj = hc.mix_runtime(residual)
            out_buf.copy_(mixed)
            if inj is not None:
                inj_buf.copy_(inj)

        compiled = aot_forward(run)
        compiled(run)
        torch.testing.assert_close(out_buf, expected_mixed, rtol=2e-2, atol=2e-2)
        if has_injection:
            torch.testing.assert_close(inj_buf, expected_inj, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "rows,fuse_mlp",
    [
        (8, True),
        (16, True),
        (32, True),
        (128, True),
        (1024, False),
        (1024, True),
        (2048, False),
        (2048, True),
        (4096, False),
        (4096, True),
    ],
)
def test_triton_combine_norm_compiled_parity(rows: int, fuse_mlp: bool):
    from b12x.norm.hyperconnection import reference
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.qwen38.hyperconnection import (
        HyperconnectionScratch,
        PreparedHyperconnection,
    )

    device = torch.device("cuda:0")
    first_mod = _make_hc_module(device, has_injection=True, seed=4000 + rows)
    next_mod = _make_hc_module(device, has_injection=True, seed=5000 + rows)
    gen = torch.Generator(device="cpu").manual_seed(6000 + rows + int(fuse_mlp))
    residual = (torch.randn((rows, 10240), generator=gen) * 0.25).bfloat16().to(device).contiguous()
    routed = (torch.randn((rows, 2560), generator=gen) * 0.25).bfloat16().to(device).contiguous()
    shared = (torch.randn((rows, 2560), generator=gen) * 0.25).bfloat16().to(device).contiguous()
    sgate = (torch.randn((rows, 1), generator=gen) * 0.5).bfloat16().to(device).contiguous()
    inj = (torch.randn((rows, 336), generator=gen) * 0.5).bfloat16().to(device)[:, 320:324]

    block_ref = routed + shared * torch.sigmoid(sgate) if fuse_mlp else routed
    expected_comb = reference.combine(residual, block_ref, inj, streams=4)
    expected_norm = reference.grouped_rmsnorm(expected_comb, next_mod.hc_norm.weight, streams=4, eps=1e-6)

    with torch.inference_mode(), RecordedPreparationSession(
        device=device,
        checkpoint=f"test-cn-{rows}-{int(fuse_mlp)}",
        workload_schema="test-cn",
        numerical_policy="bf16",
        autotune_effort="off",
    ) as session:
        scratch = HyperconnectionScratch.allocate(rows, device)
        priming = torch.zeros_like(scratch.normalized)
        first = PreparedHyperconnection(first_mod, scratch, priming_state=priming, session=session, name="first")
        following = PreparedHyperconnection(next_mod, scratch, priming_state=priming, session=session, name="next")
        first.prepare_combine_norm(following, priming_state=priming, session=session, name="cn", fuse_mlp=fuse_mlp)
        first.functional(session=session)
        following.functional(session=session)
        session.tune(show_progress=False)

        if torch.cuda.get_device_capability(device) in ((12, 0), (12, 1)):
            if fuse_mlp:
                assert first._triton_combine_norm_mlp is not None
            else:
                assert first._triton_combine_norm is not None

        comb_buf = torch.empty_like(residual)
        norm_buf = torch.empty_like(residual)

        def run():
            if fuse_mlp:
                c, n = first.combine_norm_mlp_runtime(residual, routed, shared, sgate, inj)
            else:
                c, n = first.combine_norm_runtime(residual, routed, inj)
            comb_buf.copy_(c)
            norm_buf.copy_(n)

        compiled = aot_forward(run)
        compiled(run)
        torch.testing.assert_close(comb_buf, expected_comb, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(norm_buf, expected_norm, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("has_next_injection", [True, False])
@pytest.mark.parametrize("fuse_mlp", [False, True])
def test_triton_m1_combine_mix_compiled_parity(has_next_injection: bool, fuse_mlp: bool):
    from b12x.norm.hyperconnection import reference
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.qwen38.hyperconnection import (
        HyperconnectionScratch,
        PreparedHyperconnection,
    )

    device = torch.device("cuda:0")
    first_mod = _make_hc_module(device, has_injection=True, seed=7000 + int(has_next_injection))
    next_mod = _make_hc_module(device, has_injection=has_next_injection, seed=8000 + int(fuse_mlp))
    gen = torch.Generator(device="cpu").manual_seed(9000 + int(has_next_injection) * 2 + int(fuse_mlp))
    residual = (torch.randn((1, 10240), generator=gen) * 0.25).bfloat16().to(device).contiguous()
    routed = (torch.randn((1, 2560), generator=gen) * 0.25).bfloat16().to(device).contiguous()
    shared = (torch.randn((1, 2560), generator=gen) * 0.25).bfloat16().to(device).contiguous()
    sgate = (torch.randn((1, 1), generator=gen) * 0.5).bfloat16().to(device).contiguous()
    inj = (torch.randn((1, 336), generator=gen) * 0.5).bfloat16().to(device)[:, 320:324]

    block_ref = routed + shared * torch.sigmoid(sgate) if fuse_mlp else routed
    expected_comb = reference.combine(residual, block_ref, inj, streams=4)
    expected_norm = reference.grouped_rmsnorm(expected_comb, next_mod.hc_norm.weight, streams=4, eps=1e-6)
    expected_down = torch.nn.functional.linear(expected_norm, next_mod.input_mix_weight_down.weight)
    expected_bottle = reference.scaled_silu(expected_down, streams=4)
    expected_logits = torch.nn.functional.linear(expected_bottle, next_mod.input_mix_weight_up.weight)
    expected_mixed = reference.gate_mean(expected_norm, expected_logits, streams=4)
    expected_next_inj = (
        torch.nn.functional.linear(expected_norm, next_mod.block_inject_weight.weight)
        if has_next_injection
        else None
    )

    with torch.inference_mode(), RecordedPreparationSession(
        device=device,
        checkpoint=f"test-m1-cm-{int(has_next_injection)}-{int(fuse_mlp)}",
        workload_schema="test-m1-cm",
        numerical_policy="bf16",
        autotune_effort="off",
    ) as session:
        scratch = HyperconnectionScratch.allocate(1, device)
        priming = torch.zeros_like(scratch.normalized)
        first = PreparedHyperconnection(first_mod, scratch, priming_state=priming, session=session, name="first")
        following = PreparedHyperconnection(next_mod, scratch, priming_state=priming, session=session, name="next")
        first.prepare_combine_norm(following, priming_state=priming, session=session, name="cn", fuse_mlp=fuse_mlp)
        first.functional(session=session)
        following.functional(session=session)
        session.tune(show_progress=False)

        if torch.cuda.get_device_capability(device) in ((12, 0), (12, 1)):
            if fuse_mlp:
                assert first._triton_m1_combine_mix_mlp is not None
            else:
                assert first._triton_m1_combine_mix is not None

        comb_buf = torch.empty_like(residual)
        mix_buf = torch.empty((1, 2560), dtype=torch.bfloat16, device=device)
        inj_buf = torch.empty((1, 4), dtype=torch.bfloat16, device=device) if has_next_injection else None

        def run():
            if fuse_mlp:
                c, m, ni = first.combine_mix_mlp_runtime(residual, routed, shared, sgate, inj)
            else:
                c, m, ni = first.combine_mix_runtime(residual, routed, inj)
            comb_buf.copy_(c)
            mix_buf.copy_(m)
            if ni is not None:
                inj_buf.copy_(ni)
            else:
                assert inj_buf is None

        compiled = aot_forward(run)
        compiled(run)
        torch.testing.assert_close(comb_buf, expected_comb, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(mix_buf, expected_mixed, rtol=2e-2, atol=2e-2)
        if has_next_injection:
            torch.testing.assert_close(inj_buf, expected_next_inj, rtol=2e-2, atol=2e-2)


def test_prepared_decoder_m1_combine_mix_compiled_parity():
    """4-layer M=1 compiled decode with fused combine_mix matches unfused combine_norm + mix across PLE and final_mixer."""
    from hyperflare_recipes.components.parallel import TensorParallelShard
    from hyperflare_recipes.recipes.qwen38.attention import QSAState, RotaryTables
    from hyperflare_recipes.recipes.qwen38.decoder import DecoderScratch, PreparedDecoder, PreparedDecoderForward
    from hyperflare_recipes.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState
    from hyperflare_recipes.recipes.qwen38.sampling import DeviceRequestState
    from hyperflare_recipes.recipes.qwen38.state import HybridStatePool
    from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
    from ignition.components.readiness import ReadinessTimings
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from tests.recipes.test_qwen38_pipelined_prefill_cuda import (
        _make_mini_model_weights,
        _make_synthetic_ple_table,
    )

    device = torch.device("cuda:0")
    shard = TensorParallelShard(0, 1)
    weights = _make_mini_model_weights(device, shard, num_layers=4)
    context, num_pages, slots = 4096, 64, 4
    table = _make_synthetic_ple_table(device, tokens=1)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    table.sequence = sequence
    table.timings = ReadinessTimings.allocate(frame, 1)
    rotary = RotaryTables.allocate(weights.tree.config.text_config, context, device)

    def build_env():
        pool = HybridStatePool.allocate(slots + 1, device, speculative_tokens=7, world_size=1)
        requests = DeviceRequestState.allocate(device, capacity=slots)
        inputs = DecoderInputs.allocate(device, context=context, tokens=1, capacity=slots)
        cache = {3: QSAState.allocate(pages=num_pages, slots=slots + 1, device=device, draft_tokens=7, kv_heads=2)}
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 1), device=device)
        return pool, requests, inputs, cache, completion

    def init_env(req, pool, qsa_st, comp):
        pool.backing.zero_()
        qsa_st.main_k_cache.zero_()
        qsa_st.main_v_cache.zero_()
        qsa_st.compressed_k_cache.zero_()
        qsa_st.raw_k_ring.zero_()
        qsa_st.raw_logical_positions.fill_(-1)
        qsa_st.raw_rope_positions.fill_(-1)
        qsa_st.raw_interval_start_positions.fill_(-1)
        comp.records.zero_()
        req.generation.zero_()
        req.status.zero_()
        req.error_code.zero_()
        req.consumed_tokens.zero_()
        req.generated_tokens.zero_()
        req.pending_token.fill_(-1)
        req.ple_history.fill_(248044)
        req.rng_counter.zero_()
        req.seed.fill_(12345)
        req.parameters.zero_()
        req.parameters[0].copy_(torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], device=device))
        req.top_k.fill_(1)
        req.max_new_tokens.fill_(16)
        req.eos_count.zero_()
        req.histogram.zero_()
        req.grammar_enabled.zero_()
        req.status[0] = 1
        req.generation[0] = 7
        req.consumed_tokens[0] = 1
        req.pending_token[0] = 42

    def stage_step(inp, *, active: bool):
        inp.request_state_rows.zero_()
        inp.request_state_rows[0, 0] = 1
        inp.row_slot_ids.fill_(-1)
        inp.row_slot_ids[0] = 0
        inp.row_generations.zero_()
        inp.row_generations[0] = 7
        inp.live_rows[0] = 1 if active else 0
        inp.query_lengths.zero_()
        inp.query_lengths[0] = 1 if active else 0
        inp.sample_enabled.zero_()
        inp.sample_enabled[0] = 1 if active else 0
        inp.token_ids.fill_(248044)
        inp.prefill_positions.zero_()
        inp.page_tables.fill_(-1)
        if active:
            inp.page_tables[0, 0] = 0

    try:
        ref_pool, ref_req, ref_in, ref_cache, ref_comp = build_env()
        fused_pool, fused_req, fused_in, fused_cache, fused_comp = build_env()
        with RecordedPreparationSession(
            device=device,
            checkpoint="test_qwen38_m1_combine_mix",
            workload_schema="test_qwen38_m1_combine_mix",
            numerical_policy="default",
            autotune_effort="off",
        ) as session:
            arena = ScratchArena(device, overlap=True)
            session.register_scratch(arena)
            ref_scratch = DecoderScratch.allocate(1, device, sample_rows=1, world_size=1)
            ref_state = PreparedDecoderState(ref_in, ref_req, ref_pool, tokens=1, rows=1, mode="decode")
            ref_dec = PreparedDecoder(
                weights, ref_scratch, ref_state, ref_cache, rotary, table, arena,
                ref_comp, frame, sequence, session=session, name="ref_m1",
                max_cache_pages=num_pages, sample=True,
            )
            fused_scratch = DecoderScratch.allocate(1, device, sample_rows=1, world_size=1)
            fused_state = PreparedDecoderState(fused_in, fused_req, fused_pool, tokens=1, rows=1, mode="decode")
            fused_dec = PreparedDecoder(
                weights, fused_scratch, fused_state, fused_cache, rotary, table, arena,
                fused_comp, frame, sequence, session=session, name="fused_m1",
                max_cache_pages=num_pages, sample=True,
            )
            ref_dec.functional(session=session)
            ref_dec.release_outputs()
            fused_dec.functional(session=session)
            fused_dec.release_outputs()
            session.tune(show_progress=False)

            # Disable fused combine_mix only on ref_dec to compare against unfused combine_norm + mix_runtime.
            for attn_hc, _, mlp_hc, _ in ref_dec.layers:
                attn_hc._triton_m1_combine_mix = None
                attn_hc._triton_m1_combine_mix_mlp = None
                mlp_hc._triton_m1_combine_mix = None
                mlp_hc._triton_m1_combine_mix_mlp = None

            if torch.cuda.get_device_capability(device) in ((12, 0), (12, 1)):
                assert fused_dec.layers[0][0]._triton_m1_combine_mix is not None
                assert getattr(fused_dec.layers[0][2], "_triton_m1_combine_mix_mlp", None) is None
                assert fused_dec.layers[1][2]._triton_m1_combine_mix_mlp is not None
                assert fused_dec.layers[3][2]._triton_m1_combine_mix_mlp is not None

            ref_fwd = PreparedDecoderForward(ref_dec)
            fused_fwd = PreparedDecoderForward(fused_dec)
            ref_compiled = compile_forward(ref_fwd)
            fused_compiled = compile_forward(fused_fwd)

            stream = torch.cuda.Stream(device=device)
            ref_graph = torch.cuda.CUDAGraph()
            fused_graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.stream(stream):
                    ref_compiled()
                    fused_compiled()
                stream.synchronize()
                with torch.cuda.graph(ref_graph, stream=stream):
                    ref_compiled()
                with torch.cuda.graph(fused_graph, stream=stream):
                    fused_compiled()
                stream.synchronize()

                init_env(ref_req, ref_pool, ref_cache[3], ref_comp)
                init_env(fused_req, fused_pool, fused_cache[3], fused_comp)
                sequence.fill_(1)
                stage_step(ref_in, active=True)
                stage_step(fused_in, active=True)
                with torch.cuda.stream(stream):
                    ref_graph.replay()
                    fused_graph.replay()
                stream.synchronize()

                assert torch.equal(fused_req.pending_token, ref_req.pending_token)
                assert torch.equal(fused_comp.records, ref_comp.records)
                # FUSE_MLP contracts routed + shared * sigmoid(gate) via fma.rn.bf16x2 (1 BF16 ULP vs
                # separate BF16 mul + add), which propagates across 4 random synthetic layers.
                rel_hidden = (
                    (fused_pool.last_hidden[1].float() - ref_pool.last_hidden[1].float()).norm()
                    / ref_pool.last_hidden[1].float().norm().clamp_min(1e-6)
                ).item()
                assert rel_hidden < 3e-2, f"last_hidden diverged: rel_l2={rel_hidden:.6g}"
                torch.testing.assert_close(fused_pool.last_hidden[1], ref_pool.last_hidden[1], rtol=5e-2, atol=1e-1)

                # Inactive replay must preserve persistent state bit-for-bit.
                snap_backing = fused_pool.backing.clone()
                snap_pending = fused_req.pending_token.clone()
                stage_step(fused_in, active=False)
                with torch.cuda.stream(stream):
                    fused_graph.replay()
                stream.synchronize()
                assert torch.equal(fused_pool.backing, snap_backing)
                assert torch.equal(fused_req.pending_token, snap_pending)

                # With FUSE_MLP disabled (the TP>1 path), combine_mix matches unfused bit-for-bit
                # across all 4 layers, PLE boundary, and final_mixer.
                for _, _, mlp_hc, _ in fused_dec.layers:
                    if getattr(mlp_hc, "_triton_m1_combine_mix_mlp", None) is not None:
                        cn_grid, cn_opts, gate_grid, gate_opts, out_rows = mlp_hc._triton_m1_combine_mix_mlp
                        plain_opts = dict(cn_opts, FUSE_MLP=False)
                        mlp_hc._triton_m1_combine_mix = (cn_grid, plain_opts, gate_grid, gate_opts, out_rows)
                        mlp_hc._triton_m1_combine_mix_mlp = None
                init_env(fused_req, fused_pool, fused_cache[3], fused_comp)
                sequence.fill_(1)
                stage_step(fused_in, active=True)
                fused_fwd()
                torch.cuda.synchronize(device)
                assert torch.equal(fused_pool.last_hidden[1], ref_pool.last_hidden[1])
                assert torch.equal(fused_req.pending_token, ref_req.pending_token)
            finally:
                stream.synchronize()
                ref_graph.reset()
                fused_graph.reset()
    finally:
        table.close()


@pytest.mark.parametrize("tokens", [1, 16, 128, 1024])
def test_packed_router_gate_moe_compiled_parity(tokens):
    """Packed [528, 2560] router+shared_gate matches separate BF16 projections and feeds strided combine_norm."""
    from b12x.preparation import PreparationSession
    from hyperflare_recipes.components.hyperconnection_combine import launch_combine_norm, warmup_combine_norm
    from hyperflare_recipes.recipes.qwen38.moe import PreparedQwenMoe, packed_router_gate
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.scratch import ScratchArena
    from tests.recipes.test_qwen38_pipelined_prefill_cuda import _make_nvfp4_moe_weights

    device = torch.device("cuda:0")
    weights = _make_nvfp4_moe_weights(device, world=1, seed=77 + tokens)
    packed_w = packed_router_gate(weights)
    assert packed_w.shape == (528, 2560)
    assert torch.equal(packed_w[:512], weights.router)
    assert torch.equal(packed_w[512:513], weights.shared.gate)
    assert not packed_w[513:].any()
    # Second call reuses the cached packed storage.
    assert packed_router_gate(weights) is packed_w

    gen = torch.Generator(device="cpu").manual_seed(900 + tokens)
    source = (torch.randn((tokens, 2560), generator=gen) * 0.2).bfloat16().to(device)
    expected_logits = torch.nn.functional.linear(source, weights.router)
    expected_gate = torch.nn.functional.linear(source, weights.shared.gate)

    arena = ScratchArena(device, overlap=True)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        moe = PreparedQwenMoe(
            source, weights, session=session, name=f"moe_rg_{tokens}",
            workspace=arena, projection_workspace=arena, prefill=tokens >= 1024,
        )
        if moe.concurrent_shared is not None:
            # Shared gate is computed in router_gate_projection, not on the auxiliary stream.
            assert len(moe.concurrent_shared.weights) == 2
        moe.functional(session=session)
        arena.freeze()
        arena.finalize()
        session.freeze()

        routed_buf = torch.empty_like(source)
        shared_buf = torch.empty_like(source)
        gate_buf = torch.empty((tokens, 1), dtype=torch.bfloat16, device=device)
        full_buf = torch.empty_like(source)

        def run():
            routed, shared_proj, gate = moe.parts_runtime(source)
            routed_buf.copy_(routed)
            shared_buf.copy_(shared_proj)
            gate_buf.copy_(gate)
            full_buf.copy_(moe(source))

        compiled = aot_forward(run)
        compiled(run)
        torch.cuda.synchronize(device)

        # Router adapter and gate adapter match separate linear projections.
        actual_logits = moe.router_projection(source)
        actual_gate = moe.shared.gate_projection(source)
        assert actual_logits.shape == (tokens, 512)
        assert actual_logits.stride() == (528, 1)
        assert actual_gate.shape == (tokens, 1)
        assert actual_gate.stride() == (528, 1)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_gate, expected_gate, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(gate_buf, expected_gate, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            full_buf,
            routed_buf + shared_buf * torch.sigmoid(gate_buf),
            rtol=2e-2,
            atol=2e-2,
        )

        # Verify strided gate (stride=(528, 1)) feeds _combine_norm_5x512 without copy when tokens >= 1024.
        if tokens >= 1024 and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1)):
            res = torch.randn((tokens, 10240), dtype=torch.bfloat16, device=device) * 0.2
            inj = torch.randn((tokens, 4), dtype=torch.bfloat16, device=device) * 0.2
            gain = torch.ones((10240,), dtype=torch.float32, device=device)
            grid, opts = warmup_combine_norm(
                res, routed_buf, inj, gain, torch.empty_like(res), torch.empty_like(res),
                eps=1e-6, fuse_mlp=True,
            )
            c_strided, n_strided = launch_combine_norm(
                res, routed_buf, inj, gain, grid=grid, options=opts,
                shared_projected=shared_buf, shared_gate=actual_gate,
            )
            c_contig, n_contig = launch_combine_norm(
                res, routed_buf, inj, gain, grid=grid, options=opts,
                shared_projected=shared_buf, shared_gate=actual_gate.contiguous(),
            )
            assert torch.equal(c_strided, c_contig)
            assert torch.equal(n_strided, n_contig)


@pytest.mark.parametrize(("tokens", "width"), [(512, 640), (512, 320), (300, 320), (512, 160), (4096, 640), (4100, 640)])
def test_prepared_mlp_fused_swiglu_mxfp8_compiled_parity(tokens, width):
    """Fused SwiGLU + zero-pad + MXFP8 quantize matches unfused b12x SwiGLU + MXFP8 down_proj bit-for-bit."""
    from b12x.norm import hyperconnection
    from b12x.preparation import PreparedCall
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession, declare
    from hyperflare_recipes.components import sm103
    from hyperflare_recipes.components.linear import QuantizedProjection
    from hyperflare_recipes.components.mlp import MLPScratch, PreparedMLP
    from hyperflare_recipes.components.weights import MLPWeights, quantize_bf16_linear_as_mxfp8

    device = torch.device("cuda:0")
    if sm103.is_sm103(device):
        pytest.skip("SM12x-only fused SwiGLU MXFP8 test")
    hidden = 2560
    gen = torch.Generator(device=device).manual_seed(91 + tokens + width)
    w_gu = quantize_bf16_linear_as_mxfp8((torch.randn((2 * width, hidden), generator=gen, device=device) * 0.02).bfloat16())
    w_down = quantize_bf16_linear_as_mxfp8((torch.randn((hidden, width), generator=gen, device=device) * 0.02).bfloat16())
    weights = MLPWeights(w_gu, w_down)

    source = (torch.randn((tokens, hidden), generator=gen, device=device) * 0.25).bfloat16()
    source[-16:].zero_()  # Exercise inactive/zero rows alongside live rows.
    scratch_fused = MLPScratch.allocate(tokens, hidden, width, device)
    scratch_ref = MLPScratch.allocate(tokens, hidden, width, device)
    out_buf = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    ws = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint=f"test_swiglu_mx_{tokens}_{width}",
        workload_schema=f"test_swiglu_mx_{tokens}_{width}",
        numerical_policy="default", autotune_effort="off",
    ) as session:
        session.register_scratch(ws)
        mlp_fused = PreparedMLP(
            source, weights, scratch_fused.gate_up, scratch_fused.intermediate, scratch_fused.output,
            session=session, name="fused", workspace=ws, activation_mode="auto",
        )
        assert mlp_fused.activation_plan is None and mlp_fused._down_encoded_bases is not None
        if tokens >= 4096 and 2 * width >= 1280:
            assert mlp_fused.up_projection.quantizer_plan is not None
        else:
            assert mlp_fused.up_projection.quantizer_plan is None
        ref_up = QuantizedProjection(
            source, w_gu, scratch_ref.gate_up, session=session, name="ref.gate_up",
            workspace=ws, activation_mode="auto",
        )
        ref_act = hyperconnection.plan(
            hyperconnection.Caps(device=device, max_tokens=tokens, hidden_size=width),
            invocation={"operation": "swiglu", "limit": float("inf"), "round_silu": True,
                        "clamp_after_silu": False, "swiglu_alpha": 1.0, "swiglu_beta": 0.0},
        )
        declare(session, (ref_act.request(
            name="ref.swiglu",
            prepare_call=lambda st: PreparedCall(run=lambda: st.launch(scratch_ref.gate_up, scratch_ref.gate_up, scratch_ref.intermediate)),
        ),))
        ref_down = QuantizedProjection(
            scratch_ref.intermediate, w_down, scratch_ref.output, session=session, name="ref.down",
            workspace=ws, activation_mode="auto",
        )
        session.tune(show_progress=False)

    # 1. Bound execution parity.
    bound_fused = mlp_fused(source).clone()
    ref_up(source)
    hyperconnection.run_swiglu(scratch_ref.gate_up, limit=float("inf"), round_silu=True,
                               out=scratch_ref.intermediate, plan=ref_act)
    bound_ref = ref_down(scratch_ref.intermediate).clone()
    assert torch.equal(bound_fused, bound_ref)
    assert torch.equal(bound_fused[-16:], torch.zeros_like(bound_fused[-16:]))

    # 2. Functional compiled CUDA graph execution parity.
    ws_fn = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint=f"test_swiglu_mx_fn_{tokens}_{width}",
        workload_schema=f"test_swiglu_mx_fn_{tokens}_{width}",
        numerical_policy="default", autotune_effort="off",
    ) as session:
        session.register_scratch(ws_fn)
        mlp_fn = PreparedMLP(
            source, weights, MLPScratch.allocate(tokens, hidden, width, device).gate_up,
            MLPScratch.allocate(tokens, hidden, width, device).intermediate,
            MLPScratch.allocate(tokens, hidden, width, device).output,
            session=session, name="fused_fn", workspace=ws_fn, activation_mode="auto",
        ).functional(session=session)
        session.tune(show_progress=False)

    def run():
        out_buf.copy_(mlp_fn(source))

    compiled = compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    out_buf.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    graph.reset()
    assert torch.equal(out_buf, bound_ref)


@pytest.mark.parametrize("tokens", [128, 200, 4096])
@torch.no_grad()
def test_hc_gate_fused_mxfp8_compiled_parity(tokens):
    """Fused BH=32 MXFP8 hyperconnection gate matches unfused b12x quantization bitwise."""
    from b12x.quantization import mxfp8
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.recipes.qwen38.hyperconnection import HyperconnectionScratch, PreparedHyperconnection

    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        pytest.skip("requires SM120/SM121 fused Triton hyperconnection gate")

    module = _make_hc_module(device, has_injection=True, seed=303 + tokens)
    gen = torch.Generator(device=device).manual_seed(303 + tokens)
    residual = (torch.randn((tokens, 10240), generator=gen, device=device) * 0.25).bfloat16()
    residual[-8:].zero_()

    shapes = mxfp8.dense_rows_storage_shapes(tokens, 2560)
    enc_bases_prep = (
        torch.zeros(shapes[0], dtype=torch.float8_e4m3fn, device=device),
        torch.full(shapes[1], 127, dtype=torch.uint8, device=device),
        torch.full(shapes[2], 127, dtype=torch.uint8, device=device),
    )
    enc_prep = mxfp8.dense_rows_from_bases(*enc_bases_prep)

    with RecordedPreparationSession(
        device=device, checkpoint=f"test_hc_mx_{tokens}",
        workload_schema=f"test_hc_mx_{tokens}",
        numerical_policy="default", autotune_effort="off",
    ) as session:
        hc_ref = PreparedHyperconnection(
            module, HyperconnectionScratch.allocate(tokens, device),
            priming_state=torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device),
            session=session, name="ref",
        ).functional(session=session)
        hc_mx = PreparedHyperconnection(
            module, HyperconnectionScratch.allocate(tokens, device),
            priming_state=torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device),
            session=session, name="mxfp8", encoded_output=enc_prep,
        ).functional(session=session)
        assert hc_mx.emits_mxfp8
        session.tune(show_progress=False)

    ref_bf16, ref_inj = hc_ref.mix_runtime(residual)
    ref_v = torch.empty(shapes[0], dtype=torch.float8_e4m3fn, device=device)
    ref_sr = torch.empty(shapes[1], dtype=torch.uint8, device=device)
    ref_sm = torch.full(shapes[2], 127, dtype=torch.uint8, device=device)
    ref_enc = mxfp8.dense_rows_from_bases(ref_v, ref_sr, ref_sm)
    qplan = mxfp8.plan(mxfp8.query_from_call(
        ref_bf16, ref_enc.values, ref_enc.scale_rows, ref_enc.scale_mma, min_amax=0.0,
    ))
    mxfp8.quantize_dense_rows(ref_bf16, ref_v, ref_sr, ref_sm, plan=qplan)

    out_v = torch.empty_like(ref_v)
    out_sr = torch.empty_like(ref_sr)
    out_sm = torch.empty_like(ref_sm)
    out_inj_mx = torch.empty_like(ref_inj)

    def run():
        (v, sr, sm), inj_mx = hc_mx.mix_runtime(residual)
        out_v.copy_(v)
        out_sr.copy_(sr)
        out_sm.copy_(sm)
        out_inj_mx.copy_(inj_mx)

    compiled = compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    out_v.zero_()
    out_sr.zero_()
    out_sm.zero_()
    graph.replay()
    torch.cuda.synchronize()
    graph.reset()

    assert torch.equal(out_inj_mx, ref_inj)
    assert torch.equal(out_v.view(torch.uint8), ref_v.view(torch.uint8))
    assert torch.equal(out_sr, ref_sr)
    assert torch.equal(out_sm, ref_sm)


@pytest.mark.parametrize(("tokens", "fuse_mlp", "has_next_inj"), [
    (4100, False, True),
    (4100, True, True),
    (4100, True, False),
])
@torch.no_grad()
def test_tiled_combine_down_silu_compiled_parity(tokens, fuse_mlp, has_next_inj):
    """Fused tiled combine_norm + down + ScaledSilu matches unfused combine_norm + down + ScaledSilu."""
    from b12x.norm.hyperconnection import api as hc
    from b12x.preparation import FrozenMapping, PreparedCall
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession, declare
    from hyperflare_recipes.components.hyperconnection_combine import (
        launch_combine_norm,
        launch_tiled_combine_down,
        supports_tiled_combine_down,
        warmup_combine_norm,
        warmup_tiled_combine_down,
    )
    from hyperflare_recipes.components.linear import UnquantizedProjection

    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        pytest.skip("requires SM120/SM121 fused tiled combine_norm + down kernel")

    gen = torch.Generator(device=device).manual_seed(707 + tokens + int(fuse_mlp) * 10 + int(has_next_inj))
    state = (torch.randn((tokens, 10240), generator=gen, device=device) * 0.2).bfloat16()
    block = (torch.randn((tokens, 2560), generator=gen, device=device) * 0.2).bfloat16()
    shared_proj = (torch.randn((tokens, 2560), generator=gen, device=device) * 0.2).bfloat16()
    packed_router_gate = (torch.randn((tokens, 528), generator=gen, device=device) * 0.2).bfloat16()
    shared_gate = packed_router_gate[:, 512:513]
    packed_inj = (torch.randn((tokens, 336), generator=gen, device=device) * 0.2).bfloat16()
    inj = packed_inj[:, 320:324]
    gain = (1.0 + torch.randn((10240,), generator=gen, device=device) * 0.05).float()
    down_rows = 336 if has_next_inj else 320
    down_w = (torch.randn((down_rows, 10240), generator=gen, device=device) * 0.02).bfloat16()

    assert supports_tiled_combine_down(
        tokens, tokens, 320, down_w, device, following_fuses_scaled_silu=False,
    )

    comb_ref_buf = torch.empty_like(state)
    norm_ref_buf = torch.empty_like(state)
    down_ref_buf = torch.empty((tokens, down_rows), dtype=torch.bfloat16, device=device)
    bn_ref_buf = torch.empty((tokens, 320), dtype=torch.bfloat16, device=device)
    with RecordedPreparationSession(
        device=device, checkpoint=f"test_tiled_cn_{tokens}_{int(fuse_mlp)}_{int(has_next_inj)}",
        workload_schema=f"test_tiled_cn_{tokens}_{int(fuse_mlp)}_{int(has_next_inj)}",
        numerical_policy="default", autotune_effort="off",
    ) as session:
        cn_grid, cn_opts = warmup_combine_norm(
            state, block, inj, gain, comb_ref_buf, norm_ref_buf, eps=1e-6, fuse_mlp=fuse_mlp,
        )
        down_proj = UnquantizedProjection(norm_ref_buf, down_w, down_ref_buf, session=session, name="ref.down")
        caps = hc.Caps(device=device, max_tokens=tokens, hidden_size=2560, streams=4, lowrank=320)
        silu_plan = hc.plan(caps, invocation=FrozenMapping({"operation": "scaled_silu"}))
        declare(session, (silu_plan.request(
            name="ref.silu",
            prepare_call=lambda st: PreparedCall(run=lambda: st.launch(down_ref_buf[:, :320], bn_ref_buf)),
        ),))
        spec = warmup_tiled_combine_down(
            state, block, inj, gain, down_w, comb_ref_buf, norm_ref_buf, bn_ref_buf,
            eps=1e-6, has_next_injection=has_next_inj, fuse_mlp=fuse_mlp,
        )
        session.tune(show_progress=False)

    ref_comb, ref_norm = launch_combine_norm(
        state, block, inj, gain, grid=cn_grid, options=cn_opts,
        shared_projected=shared_proj if fuse_mlp else None,
        shared_gate=shared_gate if fuse_mlp else None,
    )
    down_proj(ref_norm)
    torch.ops.b12x.hyperconnection_scaled_silu(down_ref_buf[:, :320], bn_ref_buf, silu_plan.handle)
    ref_inj_out = down_ref_buf[:, 320:324].clone() if has_next_inj else None

    out_comb = torch.empty_like(ref_comb)
    out_norm = torch.empty_like(ref_norm)
    out_bn = torch.empty_like(bn_ref_buf)
    out_inj = torch.empty((tokens, 4), dtype=torch.bfloat16, device=device) if has_next_inj else None

    def run():
        c, n, bn, n_inj = launch_tiled_combine_down(
            state, block, inj, gain, down_w, spec=spec,
            shared_projected=shared_proj if fuse_mlp else None,
            shared_gate=shared_gate if fuse_mlp else None,
        )
        out_comb.copy_(c)
        out_norm.copy_(n)
        out_bn.copy_(bn)
        if has_next_inj:
            out_inj.copy_(n_inj)

    compiled = compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    out_comb.zero_()
    out_norm.zero_()
    out_bn.zero_()
    if has_next_inj:
        out_inj.zero_()
    graph.replay()
    torch.cuda.synchronize()
    graph.reset()

    assert torch.equal(out_comb, ref_comb)
    norm_rel = (out_norm.float() - ref_norm.float()).norm() / ref_norm.float().norm().clamp(min=1e-8)
    bn_rel = (out_bn.float() - bn_ref_buf.float()).norm() / bn_ref_buf.float().norm().clamp(min=1e-8)
    assert norm_rel < 1e-4, f"norm_rel={norm_rel}"
    assert bn_rel < 2e-3, f"bn_rel={bn_rel}"
    if has_next_inj:
        inj_rel = (out_inj.float() - ref_inj_out.float()).norm() / ref_inj_out.float().norm().clamp(min=1e-8)
        assert inj_rel < 2e-3, f"inj_rel={inj_rel}"





