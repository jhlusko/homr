"""§4/§7.4 Stage C's inference-time two-pass decode: `training/transformer/
train_staff_context.py`'s `two_pass_forward`, ported from teacher-forced training to
real autoregressive inference, which has no ground truth to fall back on and no
batch-of-fixed-length-sequences to pool over - every staff's first-pass decode runs to
its own natural length.

Not yet wired into `parse_staffs`'s live pipeline (`homr/staff_parsing.py`) - built and
tested standalone first, the same "mechanism before wiring" discipline this project
used for every other Stage C piece (the module itself, the batching loader, the
training script). Wiring this into the default pipeline, behind its own opt-in flag
the way `enable_phase1_rerank` already is, is a distinct next step.
"""

from dataclasses import dataclass

import numpy as np
import torch

import homr.staff_parsing_tromr as staff_parsing_tromr
from homr.model import Staff
from homr.staff_parsing_tromr import parse_staff_tromr_greedy_with_margins
from homr.transformer.configs import Config
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from training.architecture.transformer.staff_context import StaffContextTransformer


#: `train_staff_context.py` saves every trainable parameter under this prefix
#: (`decoder.staff_context.*`, matching the full model's own attribute path) -
#: stripped here since the standalone module loaded for inference has no such
#: parent to qualify it.
_CHECKPOINT_PREFIX = "decoder.staff_context."


def load_staff_context(weights_path: str, dim: int) -> StaffContextTransformer:
    """Loads a `train_staff_context.py`-produced checkpoint (e.g. this project's own
    `phase24-staff-context-weights` release) for CPU inference - a small module (one
    attention layer over at most a system's few staves), so unlike the encoder/decoder
    it is not worth exporting to ONNX for this."""
    module = StaffContextTransformer(dim=dim)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    stripped = {
        (k[len(_CHECKPOINT_PREFIX) :] if k.startswith(_CHECKPOINT_PREFIX) else k): v
        for k, v in state.items()
    }
    module.load_state_dict(stripped)
    module.eval()
    return module


def pool_hidden(hidden_states: NDArray) -> NDArray:
    """Mean over the sequence dimension. Unlike training's fixed-length, padded
    batches (`masked_mean_pool` in `train_staff_context.py`), every row here is a real
    decoded step - inference has nothing to mask out - so this is a plain mean, with
    the same all-zero fallback for the degenerate empty-decode case (an immediate EOS)
    that masked pooling already falls back to for an all-padded staff.
    """
    if hidden_states.shape[0] == 0:
        return np.zeros((hidden_states.shape[-1],), dtype=np.float32)
    return hidden_states.astype(np.float32).mean(axis=0)


@dataclass(frozen=True)
class SystemDecodeResult:
    """One system's two-pass result - `first_pass`/`second_pass` are each a list of
    per-staff filtered greedy decodes, aligned with the caller's own `staffs` order.
    Keeping both (not just `second_pass`) lets a caller compare them directly, the
    same before/after shape `evaluate`'s with/without ablation already uses in
    `train_staff_context.py`."""

    first_pass: list[list[EncodedSymbol]]
    second_pass: list[list[EncodedSymbol]]


def decode_system_with_staff_context(
    staffs: list[Staff],
    staff_images: list[NDArray],
    config: Config,
    staff_context: StaffContextTransformer,
) -> SystemDecodeResult:
    """Full two-pass decode for one system's staves.

    First pass: an ordinary greedy decode per staff (no `staff_context_emb`, so this
    is bit-identical to today's single-pass pipeline), pooling each staff's own hidden
    states as it goes. `StaffContextTransformer` then attends across the system's own
    pooled vectors - a lone staff (no real siblings) still runs correctly, self-
    attending onto itself (see the module's own test coverage), so a system of size 1
    needs no special-casing here, though nothing meaningful is expected to change for
    it. Second pass: the same decode again, now with each staff's own gated context
    vector - identical to the first pass again until `staff_context`'s gate has moved
    off zero, the same guarantee every Stage C call site relies on.
    """
    first_pass_raw = [
        parse_staff_tromr_greedy_with_margins(staff=staff, staff_image=image, config=config)
        for staff, image in zip(staffs, staff_images, strict=True)
    ]
    first_pass = [filtered for filtered, *_rest in first_pass_raw]
    pooled = np.stack(
        [pool_hidden(hidden_states) for *_rest, hidden_states in first_pass_raw]
    )

    with torch.no_grad():
        mask = torch.ones(1, len(staffs), dtype=torch.bool)
        context = staff_context(torch.from_numpy(pooled).float().unsqueeze(0), mask)
    context_np = context.squeeze(0).numpy()

    # The ONNX decoder's dtype (fp16 on the GPU path, fp32 on CPU) - `staff_parsing_
    # tromr.inference` is only populated once the first pass above has actually run,
    # which it has by this point.
    fp16 = staff_parsing_tromr.inference is not None and staff_parsing_tromr.inference.decoder.fp16
    context_np = context_np.astype(np.float16 if fp16 else np.float32)

    second_pass_raw = [
        parse_staff_tromr_greedy_with_margins(
            staff=staff,
            staff_image=image,
            config=config,
            staff_context_emb=context_np[i : i + 1],
        )
        for i, (staff, image) in enumerate(zip(staffs, staff_images, strict=True))
    ]
    second_pass = [filtered for filtered, *_rest in second_pass_raw]

    return SystemDecodeResult(first_pass=first_pass, second_pass=second_pass)
