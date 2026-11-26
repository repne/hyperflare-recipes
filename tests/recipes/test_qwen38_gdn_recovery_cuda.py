# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recorded Qwen GDN verification and commit against b12x's per-token checkpointing decode."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")

KEY_HEADS, VALUE_HEADS = 16, 48


def _reference(tensors, columns):
    """b12x decode: per-token checkpoints in columns, gated output norm applied in place."""
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.sequence import gdn_decode

    device = tensors["mixed_qkv"].device
    caps = gdn_decode.Caps(device=device, max_tokens=tensors["mixed_qkv"].shape[0],
                           max_seqs=tensors["query_start_loc"].numel() - 1,
                           max_state_slots=tensors["recurrent_state"].shape[0], key_heads=KEY_HEADS,
                           value_heads=VALUE_HEADS, state_index_columns=columns, state_dtype=torch.float32,
                           gate_activation="sigmoid", null_state_index=0)
    plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **tensors))

    original_state, original_output = tensors["recurrent_state"].clone(), tensors["output"].clone()

    def restore():
        # Preparation primes the call; the comparison starts from the original state.
        tensors["recurrent_state"].copy_(original_state)
        tensors["output"].copy_(original_output)

    def prepare(state):
        if "scratch" not in tensors:
            (spec,) = state.layout.scratch_specs()
            tensors["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        bound = state.bind(**tensors)
        return PreparedCall(run=lambda: state.run(bound), restore=restore)

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name="gdn", prepare_call=prepare),))
    restore()
    gdn_decode.run(gdn_decode.bind(plan, **tensors), eps=1e-6)


def _backing(initial, prefix=4096, padding=8192):
    # Recurrent state as a pool would hold it: one region of a wider byte row per slot.
    slots = initial.shape[0]
    region = initial[0].numel() * 4
    backing = torch.zeros((slots, region + padding), dtype=torch.uint8, device=initial.device)
    state = backing[:, prefix:prefix + region].view(torch.float32).view_as(initial)
    state.copy_(initial)
    return backing, state


def _case(lengths, packed_inputs=False):
    """Random packed verification inputs and b12x's checkpointed decode of them."""
    from hyperflare_recipes.components.recurrent import DeltaMetadata

    generator = torch.Generator(device="cuda").manual_seed(sum(lengths))
    rows, tokens, columns = len(lengths), sum(lengths), 8
    slots = 1 + rows * columns
    mixed = (torch.randn((tokens, 2 * KEY_HEADS * 128 + VALUE_HEADS * 128), generator=generator,
                         device="cuda") * 0.5).bfloat16()
    if packed_inputs:
        projected = torch.randn((tokens, 16480), generator=generator, device="cuda").bfloat16()
        ba = projected[:, 16384:]
        z = projected[:, 10240:16384].view(tokens, VALUE_HEADS, 128)
    else:
        ba = torch.randn((tokens, 2 * VALUE_HEADS), generator=generator, device="cuda").bfloat16()
        z = torch.randn((tokens, VALUE_HEADS, 128), generator=generator, device="cuda").bfloat16()
    a, b = ba[:, VALUE_HEADS:], ba[:, :VALUE_HEADS]
    a_log = torch.rand(VALUE_HEADS, generator=generator, device="cuda") * 0.5
    dt_bias = torch.randn(VALUE_HEADS, generator=generator, device="cuda") * 0.1
    weight = (1 + 0.2 * torch.randn(128, generator=generator, device="cuda")).bfloat16()
    initial = torch.randn((slots, VALUE_HEADS, 128, 128), generator=generator, device="cuda") * 0.05
    initial[0].zero_()
    metadata = DeltaMetadata.allocate(rows, torch.device("cuda"), verification_columns=columns)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    metadata.query_start_loc.copy_(starts)
    model_slots = torch.tensor([1 + row * columns if lengths[row] else 0 for row in range(rows)], device="cuda")
    metadata.state_indices.copy_(torch.where(model_slots[:, None] > 0,
        model_slots[:, None] + torch.arange(columns, device="cuda")[None, :], 0))
    metadata.num_seqs.fill_(rows)
    metadata.num_tokens.fill_(tokens)

    expected = torch.full((tokens, VALUE_HEADS, 128), float("nan"), device="cuda").bfloat16()
    reference_state = initial.clone()
    _reference(dict(mixed_qkv=mixed, a=a, b=b, A_log=a_log, dt_bias=dt_bias, recurrent_state=reference_state,
                    z=z, norm_weight=weight, query_start_loc=metadata.query_start_loc,
                    num_accepted_tokens=metadata.accepted_tokens, state_indices=metadata.state_indices,
                    num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens, output=expected), columns)
    return dict(mixed=mixed, a=a, b=b, z=z, a_log=a_log, dt_bias=dt_bias, weight=weight, initial=initial,
                metadata=metadata, model_slots=model_slots, expected=expected, reference_state=reference_state,
                tokens=tokens, slots=slots)


def _recurrence(case, state, records):
    from hyperflare_recipes.components.gated_norm import PreparedGatedRMSNorm
    from hyperflare_recipes.components.gdn_recovery import PreparedRecoveredRecurrence

    tokens = case["tokens"]
    recurrence = torch.full((tokens, VALUE_HEADS * 128), float("nan"), device="cuda").bfloat16()
    gated = torch.full_like(recurrence, float("nan"))
    PreparedRecoveredRecurrence(mixed_qkv=case["mixed"], a=case["a"], b=case["b"], A_log=case["a_log"],
                                dt_bias=case["dt_bias"], state=state, metadata=case["metadata"], output=recurrence,
                                records=records, key_heads=KEY_HEADS, value_heads=VALUE_HEADS)()
    PreparedGatedRMSNorm(recurrence.view(tokens, VALUE_HEADS, 128), case["z"], case["weight"],
                         gated.view(tokens, VALUE_HEADS, 128), case["metadata"].num_tokens, eps=1e-6,
                         round_normalized=True)()
    return gated.view_as(case["expected"])


@pytest.mark.parametrize(("lengths", "accepted"), [
    ((4,), (3,)), ((6, 6, 0, 6), (1, 6, 0, 2)), ((5, 2), (5, 1)),
    *[((8,), (accepted,)) for accepted in range(1, 9)],
])
@pytest.mark.parametrize(("prefix", "padding", "alignment"), [(4096, 8192, 4), (4100, 8192, 1), (4096, 8196, 1)])
@pytest.mark.parametrize("packed_inputs", [False, True])
def test_recorded_verification_and_commit_match_checkpointing_decode(lengths, accepted, prefix, padding, alignment, packed_inputs):
    from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit

    case = _case(lengths, packed_inputs)
    initial, metadata, model_slots = case["initial"], case["metadata"], case["model_slots"]
    expected, reference_state, slots = case["expected"], case["reference_state"], case["slots"]
    backing, state = _backing(initial, prefix, padding)
    stacked = GDNLayerRecords.allocate((3,), slots, torch.device("cuda"), key_heads=KEY_HEADS, value_heads=VALUE_HEADS)
    gated = _recurrence(case, state, stacked[3])
    # Verification leaves committed state untouched.
    assert torch.equal(state, initial)
    counts = torch.tensor(accepted, dtype=torch.int32, device="cuda")
    commit = PreparedGDNCommit(backing, {3: state}, stacked, metadata, counts, key_heads=KEY_HEADS,
                               value_heads=VALUE_HEADS)
    assert commit.options["STATE_ALIGNMENT"] == alignment
    commit()

    torch.testing.assert_close(gated, expected, rtol=2**-6, atol=2e-2)
    assert (gated == expected).float().mean() > 0.99
    for row, (length, count) in enumerate(zip(lengths, accepted)):
        slot = int(model_slots[row])
        if not length:
            continue
        committed = reference_state[slot + count - 1]
        torch.testing.assert_close(state[slot], committed, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("lengths", [(1,), (1, 0, 1, 1)])
def test_single_token_rounds_advance_committed_state_in_place(lengths):
    """Without drafts the token is accepted when emitted, so verification commits it directly."""
    case = _case(lengths)
    initial, model_slots, reference_state = case["initial"], case["model_slots"], case["reference_state"]
    _, state = _backing(initial)
    gated = _recurrence(case, state, None)

    torch.testing.assert_close(gated, case["expected"], rtol=2**-6, atol=2e-2)
    assert (gated == case["expected"]).float().mean() > 0.99
    live = {int(slot) for slot in model_slots if slot}
    for slot in range(state.shape[0]):
        if slot in live:
            torch.testing.assert_close(state[slot], reference_state[slot], rtol=1e-5, atol=1e-6)
        else:
            assert torch.equal(state[slot], initial[slot])


def test_compilation_sees_every_recovery_store():
    """Forward compilation keeps a user kernel only for the operands its stores reach."""
    from torch._higher_order_ops.triton_kernel_wrap import get_mutated_tensors, kernel_side_table

    from hyperflare_recipes.components import gdn_recovery

    def mutated(kernel, names, options):
        constants = {name: value for name, value in options.items()}
        kwargs = {name: torch.empty(1, device="cuda") for name in names}
        return set(get_mutated_tensors(kernel_side_table.add_kernel(kernel),
                                       kernel_side_table.add_constant_args(constants), kwargs, {}))

    geometry = {"KEY_HEADS": KEY_HEADS, "VALUE_HEADS": VALUE_HEADS, "WINDOW": 8, "RECORD_WIDTH": 8, "ROWS": 16}
    for in_place, precomputed, stores in ((False, False, {"Output", "Delta", "Keys", "Decay"}),
                                         (False, True, {"Output", "Delta"}),
                                         (True, False, {"Output", "State"})):
        verify = mutated(gdn_recovery._verify, ("Mixed", "A", "B", "ALog", "DtBias", "State", "Starts", "Slots",
            "NumSeqs", "Output", "Delta", "Keys", "Decay", "Queries", "Beta"), {**geometry, "SCALE": 0.1, "mixed_stride": 1,
            "a_stride": 1, "b_stride": 1, "state_stride": 1, "slot_stride": 1, "IN_PLACE": in_place, "PRECOMPUTED": precomputed})
        assert verify == stores
    prepare = mutated(gdn_recovery._prepare_inputs,
        ("Mixed", "A", "B", "ALog", "DtBias", "Starts", "Slots", "NumSeqs", "Queries", "Keys", "Decay", "Beta"),
        {"KEY_HEADS": KEY_HEADS, "VALUE_HEADS": VALUE_HEADS, "WINDOW": 8, "RECORD_WIDTH": 8,
         "GROUP": 4, "SCALE": 0.1, "mixed_stride": 1, "a_stride": 1, "b_stride": 1, "slot_stride": 1})
    assert prepare == {"Queries", "Keys", "Decay", "Beta"}
    commit = mutated(gdn_recovery._commit, ("State", "LayerOffsets", "Delta", "KeyRecords", "Decay", "Slots",
        "Counts", "NumSeqs"), {**geometry, "ROWS_CAPACITY": 3, "state_stride": 1, "slot_stride": 1, "STATE_ALIGNMENT": 1})
    # The analysis may conservatively report every operand; it must include the state backing.
    assert "State" in commit
    verify_wy = mutated(gdn_recovery._verify_wy, ("Mixed", "State", "Starts", "Slots", "NumSeqs",
        "Output", "Delta", "Keys", "Decay", "Queries", "Beta"),
        {"KEY_HEADS": KEY_HEADS, "VALUE_HEADS": VALUE_HEADS, "WINDOW": 8, "RECORD_WIDTH": 8,
         "ROWS": 64, "BK": 16, "EVICT_STATE": True, "mixed_stride": 1, "state_stride": 1, "slot_stride": 1})
    assert verify_wy == {"Output", "Delta"}
    commit_stream = mutated(gdn_recovery._commit_streaming, ("State", "LayerOffsets", "Delta", "KeyRecords",
        "Decay", "Slots", "Counts", "NumSeqs"),
        {**geometry, "ROWS": 32, "ROWS_CAPACITY": 3, "state_stride": 1, "slot_stride": 1, "STATE_ALIGNMENT": 1})
    assert "State" in commit_stream


@pytest.mark.parametrize("width", [13, 16])
def test_wide_recovery_all_prefixes_with_poisoned_record_tail(width):
    from hyperflare_recipes.components.gated_norm import PreparedGatedRMSNorm
    from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit, PreparedRecoveredRecurrence
    from hyperflare_recipes.components.recurrent import DeltaMetadata

    base = _case((8,))
    mixed, a, b, z = (torch.cat((base[name], base[name]), dim=0)[:width].contiguous()
                       for name in ("mixed", "a", "b", "z"))
    second_state = base["reference_state"].clone()
    second_state[1].copy_(base["reference_state"][8])
    tail = width - 8
    tail_metadata = DeltaMetadata.allocate(1, torch.device("cuda"), verification_columns=8)
    tail_metadata.query_start_loc.copy_(torch.tensor([0, tail], dtype=torch.int32, device="cuda"))
    tail_metadata.state_indices[0].copy_(torch.arange(1, 9, device="cuda"))
    tail_metadata.num_seqs.fill_(1)
    tail_metadata.num_tokens.fill_(tail)
    tail_output = torch.empty((tail, VALUE_HEADS, 128), dtype=torch.bfloat16, device="cuda")
    _reference(dict(mixed_qkv=mixed[8:], a=a[8:], b=b[8:], A_log=base["a_log"],
        dt_bias=base["dt_bias"], recurrent_state=second_state, z=z[8:], norm_weight=base["weight"],
        query_start_loc=tail_metadata.query_start_loc, num_accepted_tokens=tail_metadata.accepted_tokens,
        state_indices=tail_metadata.state_indices, num_seqs=tail_metadata.num_seqs,
        num_tokens=tail_metadata.num_tokens, output=tail_output), 8)
    expected = torch.cat((base["expected"], tail_output))
    expected_states = torch.cat((base["reference_state"][1:9], second_state[1:tail + 1]))
    metadata = DeltaMetadata.allocate(1, torch.device("cuda"), verification_columns=width)
    metadata.query_start_loc.copy_(torch.tensor([0, width], dtype=torch.int32, device="cuda"))
    metadata.state_indices[:, 0].fill_(1)
    metadata.num_seqs.fill_(1)
    metadata.num_tokens.fill_(width)
    backing, state = _backing(base["initial"])
    records = GDNLayerRecords.allocate((3,), 1, torch.device("cuda"),
                                       key_heads=KEY_HEADS, value_heads=VALUE_HEADS, width=16)
    recurrence = torch.empty((width, VALUE_HEADS * 128), dtype=torch.bfloat16, device="cuda")
    gated = torch.empty_like(recurrence)
    verify = PreparedRecoveredRecurrence(mixed_qkv=mixed, a=a, b=b, A_log=base["a_log"],
        dt_bias=base["dt_bias"], state=state, metadata=metadata, output=recurrence,
        records=records[3], key_heads=KEY_HEADS, value_heads=VALUE_HEADS)
    norm = PreparedGatedRMSNorm(recurrence.view(width, VALUE_HEADS, 128), z, base["weight"],
        gated.view(width, VALUE_HEADS, 128), metadata.num_tokens, eps=1e-6, round_normalized=True)
    counts = torch.ones(1, dtype=torch.int32, device="cuda")
    commit = PreparedGDNCommit(backing, {3: state}, records, metadata, counts,
                               key_heads=KEY_HEADS, value_heads=VALUE_HEADS)
    verify(); norm(); commit()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify(); norm(); commit()
    for accepted in range(width + 1):
        state.copy_(base["initial"])
        counts.fill_(accepted)
        for tensor in (records.delta, records.keys, records.decay, gated):
            tensor.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(gated.view_as(expected), expected, rtol=2**-6, atol=2e-2)
        reference = base["initial"][1] if accepted == 0 else expected_states[accepted - 1]
        torch.testing.assert_close(state[1], reference, rtol=1e-5, atol=1e-6)
        if width < 16:
            assert torch.isnan(records.delta[..., width:, :]).all()
            assert torch.isnan(records.keys[..., width:, :]).all()
            assert torch.isnan(records.decay[..., width:]).all()
    graph.reset()


def test_reused_graph_width_transitions_preserve_recurrent_history():
    from hyperflare_recipes.components.gated_norm import PreparedGatedRMSNorm
    from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit, PreparedRecoveredRecurrence
    from hyperflare_recipes.components.recurrent import DeltaMetadata

    base = _case((8,), packed_inputs=True)
    device = torch.device("cuda")
    backing, state = _backing(base["initial"])
    mixed = torch.empty((16, 10240), device=device, dtype=torch.bfloat16)
    projected = torch.empty((16, 16480), device=device, dtype=torch.bfloat16)
    a, b = projected[:, 16432:], projected[:, 16384:16432]
    z = projected[:, 10240:16384].view(16, VALUE_HEADS, 128)
    output = torch.empty((16, VALUE_HEADS, 128), device=device, dtype=torch.bfloat16)
    records = GDNLayerRecords.allocate((3,), 1, device, key_heads=KEY_HEADS, value_heads=VALUE_HEADS, width=16)
    counts = torch.ones(1, device=device, dtype=torch.int32)
    variants = {}
    mixed.normal_(std=0.5)
    projected.normal_()
    for width in (1, 8, 16):
        metadata = DeltaMetadata.allocate(1, device, verification_columns=width)
        metadata.query_start_loc[1].fill_(width)
        metadata.state_indices[:, 0].fill_(1)
        metadata.num_seqs.fill_(1)
        metadata.num_tokens.fill_(width)
        recurrence = torch.empty((16, VALUE_HEADS * 128), device=device, dtype=torch.bfloat16)
        verify = PreparedRecoveredRecurrence(mixed_qkv=mixed, a=a, b=b, A_log=base["a_log"],
            dt_bias=base["dt_bias"], state=state, metadata=metadata, output=recurrence,
            records=None if width == 1 else records[3], key_heads=KEY_HEADS, value_heads=VALUE_HEADS)
        norm = PreparedGatedRMSNorm(recurrence.view_as(output), z, base["weight"], output,
                                   metadata.num_tokens, eps=1e-6, round_normalized=True)
        commit = None if width == 1 else PreparedGDNCommit(backing, {3: state}, records, metadata, counts,
                                                          key_heads=KEY_HEADS, value_heads=VALUE_HEADS)
        verify(); norm()
        if commit is not None:
            commit()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify(); norm()
            if commit is not None:
                commit()
        variants[width] = (metadata, graph, verify, norm, commit)
    state.copy_(base["initial"])
    reference_state = base["initial"].clone()
    reference_metadata = DeltaMetadata.allocate(1, device, verification_columns=8)
    reference_metadata.state_indices[0].copy_(torch.arange(1, 9, device=device))
    reference_metadata.num_seqs.fill_(1)
    for width, length, accepted in ((1, 1, 1), (16, 16, 7), (8, 3, 2), (1, 0, 0),
                                     (8, 8, 8), (16, 5, 0), (1, 1, 1), (16, 16, 16)):
        mixed.normal_(std=0.5)
        projected.normal_()
        before = state.clone()
        reference_state[1].copy_(state[1])
        expected_chunks, checkpoints = [], []
        for start in range(0, length, 8):
            count = min(8, length - start)
            reference_metadata.query_start_loc[1].fill_(count)
            reference_metadata.num_tokens.fill_(count)
            expected = torch.empty((count, VALUE_HEADS, 128), device=device, dtype=torch.bfloat16)
            _reference(dict(mixed_qkv=mixed[start:start + count], a=a[start:start + count],
                b=b[start:start + count], A_log=base["a_log"], dt_bias=base["dt_bias"],
                recurrent_state=reference_state, z=z[start:start + count], norm_weight=base["weight"],
                query_start_loc=reference_metadata.query_start_loc,
                num_accepted_tokens=reference_metadata.accepted_tokens,
                state_indices=reference_metadata.state_indices, num_seqs=reference_metadata.num_seqs,
                num_tokens=reference_metadata.num_tokens, output=expected), 8)
            expected_chunks.append(expected)
            checkpoints.extend(reference_state[1:count + 1].clone().unbind())
            reference_state[1].copy_(reference_state[count])
        metadata, graph, _, norm, _ = variants[width]
        recurrence = norm.args[0]
        metadata.query_start_loc[1].fill_(length)
        metadata.num_seqs.fill_(int(length > 0))
        metadata.num_tokens.fill_(length)
        counts.fill_(accepted)
        for tensor in (recurrence, output, records.delta, records.keys, records.decay):
            tensor.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.equal(state[0], before[0])
        assert torch.equal(state[2:], before[2:])
        expected_state = checkpoints[accepted - 1] if accepted else before[1]
        torch.testing.assert_close(state[1], expected_state, rtol=1e-5, atol=1e-6)
        if length:
            expected = torch.cat(expected_chunks)
            torch.testing.assert_close(output[:length], expected, rtol=2**-6, atol=2e-2)
            assert (output[:length] == expected).float().mean() > 0.99
        assert torch.isnan(recurrence[length:]).all()
        assert torch.equal(output[length:], torch.zeros_like(output[length:]))
        if width > 1:
            assert torch.isnan(records.delta[..., length:, :]).all()
            assert torch.isnan(records.keys[..., length:, :]).all()
            assert torch.isnan(records.decay[..., length:]).all()
    for _, graph, *_ in variants.values():
        graph.reset()


@pytest.mark.parametrize(("window", "lengths", "accepted"), [
    (8, (6, 6, 0, 6), (1, 6, 0, 2)),
    (5, (5, 3, 0, 4), (5, 2, 0, 4)),
])
def test_wy_verification_and_streaming_commit_match_reference(window, lengths, accepted):
    """WY tf32x3 verify and multi-layer streaming commit match b12x across partial windows and null slots."""
    from hyperflare_recipes.components import gdn_recovery
    from hyperflare_recipes.components.gated_norm import PreparedGatedRMSNorm
    from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit, PreparedRecoveredRecurrence
    from hyperflare_recipes.components.recurrent import DeltaMetadata

    device = torch.device("cuda")
    # 16 layers x 4 requests x 48 value heads x 64 KiB = 192 MiB -> triggers streaming commit on SM12x.
    num_layers = 16
    layer_ids = tuple(range(num_layers))
    base = _case(lengths)
    rows, tokens = len(lengths), sum(lengths)
    metadata = DeltaMetadata.allocate(rows, device, verification_columns=window)
    metadata.query_start_loc.copy_(base["metadata"].query_start_loc)
    metadata.state_indices.copy_(base["metadata"].state_indices[:, :window])
    metadata.num_seqs.copy_(base["metadata"].num_seqs)
    metadata.num_tokens.copy_(base["metadata"].num_tokens)

    slots = base["slots"]
    layer_bytes = VALUE_HEADS * 128 * 128 * 4
    backing = torch.zeros((slots, num_layers * layer_bytes + 4096), dtype=torch.uint8, device=device)
    states = {}
    for idx, layer in enumerate(layer_ids):
        off = 4096 + idx * layer_bytes
        st = backing[:, off:off + layer_bytes].view(torch.float32).view(slots, VALUE_HEADS, 128, 128)
        st.copy_(base["initial"])
        states[layer] = st

    records = GDNLayerRecords.allocate(layer_ids, rows, device, key_heads=KEY_HEADS, value_heads=VALUE_HEADS, width=8)
    for tensor in (records.delta, records.keys, records.decay):
        tensor.fill_(float("nan"))

    recurrence = torch.full((tokens, VALUE_HEADS * 128), float("nan"), device=device, dtype=torch.bfloat16)
    gated = torch.full_like(recurrence, float("nan"))
    verify = PreparedRecoveredRecurrence(
        mixed_qkv=base["mixed"], a=base["a"], b=base["b"], A_log=base["a_log"],
        dt_bias=base["dt_bias"], state=states[0], metadata=metadata, output=recurrence,
        records=records[0], key_heads=KEY_HEADS, value_heads=VALUE_HEADS,
    )
    if torch.cuda.get_device_capability(device)[0] >= 12:
        assert verify.use_wy
    verify()
    # Copy layer-0 records to the remaining layers to validate multi-layer streaming commit.
    for idx in range(1, num_layers):
        records.delta[idx].copy_(records.delta[0])
        records.keys[idx].copy_(records.keys[0])
        records.decay[idx].copy_(records.decay[0])
    assert torch.equal(states[0], base["initial"])
    if window < 8:
        assert torch.isnan(records.delta[0, :, :, window:, :]).all()
        assert torch.isnan(records.keys[0, :, :, window:, :]).all()
        assert torch.isnan(records.decay[0, :, :, window:]).all()

    PreparedGatedRMSNorm(
        recurrence.view(tokens, VALUE_HEADS, 128), base["z"], base["weight"],
        gated.view(tokens, VALUE_HEADS, 128), metadata.num_tokens, eps=1e-6, round_normalized=True,
    )()
    torch.testing.assert_close(gated.view_as(base["expected"]), base["expected"], rtol=2**-6, atol=2e-2)
    assert (gated.view_as(base["expected"]) == base["expected"]).float().mean() > 0.99

    counts = torch.tensor(accepted, dtype=torch.int32, device=device)
    commit = PreparedGDNCommit(
        backing, states, records, metadata, counts, key_heads=KEY_HEADS, value_heads=VALUE_HEADS,
    )
    if torch.cuda.get_device_capability(device)[0] >= 12:
        assert commit.streaming
    assert commit.options["STATE_ALIGNMENT"] == 4
    commit()
    for layer in (0, num_layers - 1):
        for row, (length, count) in enumerate(zip(lengths, accepted)):
            slot = int(base["model_slots"][row])
            if not length:
                continue
            torch.testing.assert_close(states[layer][slot], base["reference_state"][slot + count - 1],
                                       rtol=1e-5, atol=1e-6)
        assert torch.equal(states[layer][0], base["initial"][0])

    # Also validate functional() dynamic_metadata execution of both WY verify and streaming commit.
    for st in states.values():
        st.copy_(base["initial"])
    verify.functional(dynamic_metadata=True)
    commit.functional(dynamic_metadata=True)
    func_rec = verify(base["mixed"], base["a"], base["b"], records=records[0], metadata=metadata)
    torch.testing.assert_close(func_rec, recurrence, rtol=0, atol=0)
    commit(counts, records=records, metadata=metadata)
    for row, (length, count) in enumerate(zip(lengths, accepted)):
        slot = int(base["model_slots"][row])
        if not length:
            continue
        torch.testing.assert_close(states[0][slot], base["reference_state"][slot + count - 1],
                                   rtol=1e-5, atol=1e-6)


def test_prepared_gdn_verify_fused_output_encoding_parity():
    """Large verification rounds (>28 tokens) fuse gated RMSNorm with MXFP8 out_proj quantization."""
    from types import SimpleNamespace
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from hyperflare_recipes.components import sm103
    from hyperflare_recipes.components.gated_delta import GDNScratch, GDNState, PreparedGDN
    from hyperflare_recipes.components.gdn_recovery import GDNLayerRecords
    from hyperflare_recipes.components.linear import QuantizedProjection
    from hyperflare_recipes.components.recurrent import DeltaMetadata
    from hyperflare_recipes.components.weights import quantize_bf16_linear_as_mxfp8

    device = torch.device("cuda")
    if sm103.is_sm103(device):
        pytest.skip("SM12x-only fused MXFP8 gated RMSNorm test")
    rows, window, tokens = 4, 8, 32
    gen = torch.Generator(device=device).manual_seed(17)
    conv_w = (torch.randn((10240, 1, 4), generator=gen, device=device) * 0.1).bfloat16()
    module = SimpleNamespace(
        num_k_heads=KEY_HEADS, num_v_heads=VALUE_HEADS, head_k_dim=128, head_v_dim=128,
        conv_dim=10240, conv_kernel_size=4, hidden_size=2560, activation="silu",
        conv1d=SimpleNamespace(weight=conv_w, bias=None),
        A_log=torch.rand(VALUE_HEADS, generator=gen, device=device) * 0.5,
        dt_bias=torch.randn(VALUE_HEADS, generator=gen, device=device) * 0.1,
        norm=SimpleNamespace(weight=(1 + 0.1 * torch.randn(128, generator=gen, device=device)).bfloat16(),
                             activation="sigmoid", variance_epsilon=1e-6),
    )
    projections = {
        "in_proj_qkvz": quantize_bf16_linear_as_mxfp8((torch.randn((16384, 2560), generator=gen, device=device) * 0.02).bfloat16()),
        "in_proj_ba": quantize_bf16_linear_as_mxfp8((torch.randn((96, 2560), generator=gen, device=device) * 0.02).bfloat16()),
        "out_proj": quantize_bf16_linear_as_mxfp8((torch.randn((2560, 6144), generator=gen, device=device) * 0.02).bfloat16()),
    }
    source = (torch.randn((tokens, 2560), generator=gen, device=device) * 0.2).bfloat16()
    state_fused = GDNState.allocate(rows + 1, device, speculative_tokens=window - 1)
    state_ref = GDNState.allocate(rows + 1, device, speculative_tokens=window - 1)
    state_fused.recurrent[1:].normal_(std=0.05)
    state_fused.convolution[1:].normal_(std=0.1)
    state_ref.recurrent.copy_(state_fused.recurrent)
    state_ref.convolution.copy_(state_fused.convolution)
    meta_zero = DeltaMetadata.allocate(rows, device, verification_columns=window)
    rec_fused = GDNLayerRecords.allocate((0,), rows, device, key_heads=KEY_HEADS, value_heads=VALUE_HEADS, width=window)
    rec_ref = GDNLayerRecords.allocate((0,), rows, device, key_heads=KEY_HEADS, value_heads=VALUE_HEADS, width=window)
    ref_gated = torch.empty((tokens, 6144), dtype=torch.bfloat16, device=device)
    ref_out = torch.empty((tokens, 2560), dtype=torch.bfloat16, device=device)
    ws = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="test_gdn_verify_mx", workload_schema="test_gdn_verify_mx",
        numerical_policy="default", autotune_effort="off",
    ) as session:
        session.register_scratch(ws)
        gdn_fused = PreparedGDN(
            module, projections, source, GDNScratch.allocate(tokens, device), state_fused, meta_zero,
            mode="decode", session=session, name="gdn_fused", workspace=ws, projection_workspace=ws,
            records=rec_fused[0],
        )
        assert gdn_fused.output_encoding is not None
        gdn_fused.functional(session=session, dynamic_metadata=True)
        # Reference: unfused PreparedGatedRMSNorm (BF16) -> MXFP8 QuantizedProjection (auto/quantized).
        gdn_ref = PreparedGDN(
            module, projections, source, GDNScratch.allocate(tokens, device), state_ref, meta_zero,
            mode="decode", session=session, name="gdn_ref", workspace=ws, projection_workspace=None,
            records=rec_ref[0], prepare_projections=False,
        )
        assert gdn_ref.output_encoding is None
        gdn_ref.functional(session=session, dynamic_metadata=True)
        ref_out_proj = QuantizedProjection(
            ref_gated, projections["out_proj"], ref_out, session=session, name="ref_out_proj",
            workspace=ws, activation_mode="auto", allow_bf16_atomic_reduction=False,
        ).functional(session=session)
        session.tune(show_progress=False)

    # Partial live count (18 of 32 tokens live, row 2 empty) tests active parity and inactive-row zeroing.
    live_meta = DeltaMetadata.allocate(rows, device, verification_columns=window)
    live_meta.query_start_loc.copy_(torch.tensor([0, 6, 12, 12, 18], dtype=torch.int32, device=device))
    live_meta.convolution_slots.copy_(torch.tensor([1, 2, -1, 4], dtype=torch.int64, device=device))
    live_meta.state_indices[:, 0].copy_(torch.tensor([1, 2, 0, 4], dtype=torch.int64, device=device))
    live_meta.num_seqs.fill_(4)
    live_meta.num_tokens.fill_(18)
    projected, z, a, b, _, _ = gdn_fused.project_input_runtime(source)
    out_fused = gdn_fused.finish_sequence_runtime(projected, z, a, b, records=rec_fused[0], metadata=live_meta)
    gated_ref, encoded_ref = gdn_ref._functional_sequence(projected, z, a, b, records=rec_ref[0], metadata=live_meta)
    assert encoded_ref is None
    out_ref = ref_out_proj(gated_ref)
    torch.testing.assert_close(out_fused[:18], out_ref[:18], rtol=2e-2, atol=2e-2)
    assert (out_fused[:18] == out_ref[:18]).float().mean() > 0.99
    assert torch.equal(out_fused[18:], torch.zeros_like(out_fused[18:]))
    torch.testing.assert_close(rec_fused.delta, rec_ref.delta, rtol=0, atol=0)
