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
