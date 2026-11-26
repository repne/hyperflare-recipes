"""Prepared BF16 head gates with explicit gate row strides."""

import torch
import triton
import triton.language as tl
from b12x.gemm.blockscaled._quantize import _scale_offset
from b12x.norm.hyperconnection import api as hyperconnection
from b12x.preparation import PreparedCall

from hyperflare_recipes.fixtures.hyperconnection import HyperconnectionBenchmarkInputs
from ignition.preparation.tuning import benchmark_resource, declare


@triton.jit
def _qsa_output_gate(
    ATT, GATE, REQ_IDS, OUT,
    GATE_ROW_STRIDE, GATE_HEAD_STRIDE,
    HEADS: tl.constexpr, DIM: tl.constexpr, HEAD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)[:, None]
    col = tl.arange(0, DIM)[None, :]
    mask = head < HEADS
    live = tl.load(REQ_IDS + row) >= 0
    live_mask = mask & live
    att_off = row.to(tl.int64) * (HEADS * DIM) + head * DIM + col
    gate_off = row.to(tl.int64) * GATE_ROW_STRIDE + head * GATE_HEAD_STRIDE + col
    a = tl.load(ATT + att_off, mask=live_mask, other=0.0).to(tl.float32)
    g = tl.load(GATE + gate_off, mask=live_mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    val = tl.where(live, (a * sig).to(tl.bfloat16), 0.0).to(tl.bfloat16)
    tl.store(OUT + att_off, val, mask=mask)


@triton.jit
def _qsa_output_gate_mxfp8(
    ATT, GATE, REQ_IDS, VALUES, SCALE_ROWS, SCALE_MMA,
    GATE_ROW_STRIDE, GATE_HEAD_STRIDE,
    HEADS: tl.constexpr, DIM: tl.constexpr, HEAD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)[:, None]
    col = tl.arange(0, DIM)[None, :]
    mask = head < HEADS
    live = tl.load(REQ_IDS + row) >= 0
    live_mask = mask & live
    att_off = row.to(tl.int64) * (HEADS * DIM) + head * DIM + col
    gate_off = row.to(tl.int64) * GATE_ROW_STRIDE + head * GATE_HEAD_STRIDE + col
    a = tl.load(ATT + att_off, mask=live_mask, other=0.0).to(tl.float32)
    g = tl.load(GATE + gate_off, mask=live_mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    val = tl.where(live, (a * sig).to(tl.bfloat16), 0.0).to(tl.bfloat16)

    groups: tl.constexpr = HEADS * DIM // 32
    local_groups: tl.constexpr = HEAD_BLOCK * DIM // 32
    x = tl.reshape(val.to(tl.float32), (local_groups, 32))
    amax = tl.max(tl.abs(x), 1)
    ratio = tl.where(amax > 0, amax / 448.0, 1.0)
    exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(ratio)), -127.0), 127.0)
    scale = tl.exp2(exponent)
    quantized = (x / scale[:, None]).to(tl.float8e4nv)
    tl.store(VALUES + att_off, tl.reshape(quantized, (HEAD_BLOCK, DIM)), mask=mask)
    group = tl.program_id(1) * local_groups + tl.arange(0, local_groups)
    byte = (exponent + 127).to(tl.uint8)
    tl.store(SCALE_ROWS + row.to(tl.int64) * groups + group, byte, group < groups)
    tl.store(SCALE_MMA + _scale_offset(row, group, groups), byte, group < groups)


def qsa_output_gate(
    attended: torch.Tensor,
    gate: torch.Tensor,
    request_ids: torch.Tensor,
    *,
    encoded_bases: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply strided sigmoid output gating and zero inactive rows (request_ids < 0).

    When ``encoded_bases`` is supplied, quantizes directly to MXFP8
    ``(values, scale_rows, scale_mma)`` and returns ``values`` without writing
    an intermediate BF16 tensor.
    """
    if (
        attended.ndim != 3
        or attended.dtype != torch.bfloat16
        or not attended.is_contiguous()
        or gate.shape != attended.shape
        or gate.dtype != torch.bfloat16
        or gate.stride(2) != 1
        or request_ids.shape != (attended.shape[0],)
        or not request_ids.is_contiguous()
        or request_ids.dtype not in (torch.int32, torch.int64)
        or gate.device != attended.device
        or request_ids.device != attended.device
    ):
        raise ValueError(
            "qsa_output_gate requires contiguous BF16 attended [tokens, heads, dim], "
            "unit-inner-stride BF16 gate, and contiguous 1D int32/int64 request_ids"
        )
    tokens, heads, dim = attended.shape
    if dim <= 0 or (dim & (dim - 1)) != 0 or dim % 32 != 0:
        raise ValueError("qsa_output_gate requires a 32-aligned power-of-two head_dim")
    head_block = 4 if heads % 4 == 0 else (2 if heads % 2 == 0 else 1)
    grid = (tokens, triton.cdiv(heads, head_block))
    if encoded_bases is not None:
        values, scale_rows, scale_mma = encoded_bases
        if (
            values.shape != (tokens, heads * dim)
            or values.dtype != torch.float8_e4m3fn
            or not values.is_contiguous()
            or values.device != attended.device
        ):
            raise ValueError("qsa_output_gate encoded_bases values must match [tokens, heads * dim] FP8")
        torch.library.wrap_triton(_qsa_output_gate_mxfp8)[grid](
            attended,
            gate,
            request_ids,
            values,
            scale_rows,
            scale_mma,
            gate.stride(0),
            gate.stride(1),
            HEADS=heads,
            DIM=dim,
            HEAD_BLOCK=head_block,
            num_warps=4,
        )
        return values
    out = torch.empty((tokens, heads * dim), dtype=torch.bfloat16, device=attended.device)
    torch.library.wrap_triton(_qsa_output_gate)[grid](
        attended,
        gate,
        request_ids,
        out,
        gate.stride(0),
        gate.stride(1),
        HEADS=heads,
        DIM=dim,
        HEAD_BLOCK=head_block,
        num_warps=4,
    )
    return out


class PreparedHeadGate:
    def __init__(self, values, gates, *, session, name):
        if (values.ndim != 3 or values.dtype != torch.bfloat16 or not values.is_contiguous()
                or gates.shape != values.shape[:2] or gates.dtype != values.dtype
                or gates.device != values.device or gates.stride(1) != 1
                or gates.stride(0) < gates.shape[1]):
            raise ValueError("head gates require contiguous BF16 values and matching strided gate rows")
        value_start, gate_start = values.data_ptr(), gates.data_ptr()
        gate_bytes = ((gates.shape[0] - 1) * gates.stride(0) + gates.shape[1]) * gates.element_size()
        if value_start < gate_start + gate_bytes and gate_start < value_start + values.numel() * values.element_size():
            raise ValueError("head gate logits must not overlap mutable values")
        self.values, self.gates = values, gates
        self.plan = hyperconnection.plan(hyperconnection.Caps(device=values.device,
            max_tokens=values.shape[0], streams=values.shape[1], hidden_size=values.shape[2],
            lowrank=1, dtype=values.dtype),
            invocation={"operation": "head_gate", "gate_stride": gates.stride(0)})

        def benchmark(state):
            operands = benchmark_resource(session, ("head_gate", name),
                lambda: HyperconnectionBenchmarkInputs("head_gate", (values, gates, values)))
            v, g, _ = operands.arguments
            return PreparedCall(run=lambda: state.launch(v, g, v), output=v,
                                produce=operands.produce, owners=(operands,))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.launch(values, gates, values)),
            benchmark_call=benchmark),))

    def functional(self, *, session):
        """Consume forward-local values and gates with the prepared row strides."""
        del session
        self.values = self.gates = None
        return self

    def __call__(self, values=None, gates=None):
        values = self.values if values is None else values
        gates = self.gates if gates is None else gates
        if values is None or gates is None:
            raise ValueError("head gate requires values and gates")
        torch.ops.b12x.hyperconnection_head_gate(values, gates, self.plan.handle)
        return values
