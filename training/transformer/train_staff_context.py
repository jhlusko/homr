"""
Train §4/§7.4 Stage C's cross-staff `StaffContextTransformer` over a frozen core.

Same frozen-core probe shape `train_profile_context.py`/`train_structured_heads.py`
already used, applied to the one genuinely new mechanism here: this run is a true
*two-pass decode* per system, not a single forward pass conditioned on precomputed
fields. For each system batch:

  1. First pass: decode every staff independently and keep the shared decoder's own
     pooled hidden state per staff (`mixed_first_pass_hidden`, masked-mean-pooled
     over the sequence dimension). `--sampling-prob` (default 0.5) controls how much
     of this pass is teacher-forced vs. the model's own greedy prediction, mixed
     position-by-position - see `mixed_first_pass_hidden`'s docstring for why a fully
     teacher-forced first pass (the original `sampling_prob=1.0` version of this
     mechanism) turned out to make DECODER_RHYTHM_ACCURACY_DESIGN.md §7.4's real
     Stage A/B benchmark measurably worse, not better, and why this fixes it without
     needing genuine step-by-step autoregressive decoding.
  2. `StaffContextTransformer` attends across a system's own staves' pooled vectors
     (masking out padded staff slots via `staff_mask`), producing a per-staff context
     vector - zero at initialization, per its own zero-init gate.
  3. Second pass: decode every staff again, this time with `staff_context_emb` added
     to the decoder's input embedding, and train against the model's own existing
     `loss` from that second pass.

The frozen core (everything except `decoder.staff_context`) never adapts; this asks
the same narrow question `train_profile_context.py` asked of its own module - can
*some* assignment of the new module, backpropagated through a frozen network, move
the model's own existing loss - before committing to unfreezing the decoder to let it
adapt to cross-staff signal as well.

`evaluate`'s with/without ablation does NOT reuse the same first-pass hidden state
for both numbers: "without" is a separate, always-fully-teacher-forced call
(`model(**flat, sampling_prob=1.0)`, no cross-staff context, no exposure-bias
mixing - a stable "no Stage C at all" baseline this fix has no reason to touch),
"with" is the second pass built from `mixed_first_pass_hidden`'s pooled hidden state.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

from training.architecture.transformer.checkpoint_loading import load_checkpoint

#: Parameters the pretrained checkpoint cannot be expected to contain.
NEW_PARAMETER_PREFIXES = ("decoder.staff_context.",)


def set_probe_mode(model: nn.Module) -> None:
    """Frozen core in eval mode, the staff context module in train mode - same
    reasoning as `train_profile_context.py`'s own `set_probe_mode`: a frozen core's
    dropout would put a gap between what it trains on and what it is scored on,
    without regularising anything, since nothing about the core is being learned here.
    """
    model.eval()
    staff_context = getattr(getattr(model, "decoder", None), "staff_context", None)
    if staff_context is not None:
        staff_context.train()


def staff_context_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if name.startswith(NEW_PARAMETER_PREFIXES)]


def load_pinned(model: nn.Module, checkpoint: Path) -> None:
    """Load the pretrained weights, allowing only the new module to be missing."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report = load_checkpoint(model, state, NEW_PARAMETER_PREFIXES)
    print(f"checkpoint: {report.describe()}")


def flatten_staff_dim(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int, int]:
    """`(batch, staff, ...)` -> `(batch * staff, ...)` for every field except
    `staff_mask` itself, which stays `(batch, staff)` - `StaffContextTransformer`'s own
    input shape, not the shared decoder's.
    """
    staff_count = batch["staff_mask"].shape[1]
    sample_count = batch["staff_mask"].shape[0]
    flat = {
        key: value.reshape(sample_count * staff_count, *value.shape[2:])
        for key, value in batch.items()
        if key != "staff_mask"
    }
    return flat, sample_count, staff_count


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-staff summary vector for `StaffContextTransformer`'s input - the mean of
    the shared decoder's own hidden state over every real (unmasked) token. A staff
    with no real tokens at all (a fully-padded slot from `SystemBatchDataset`) pools
    to an all-zero vector rather than dividing by zero - harmless, since
    `StaffContextTransformer`'s own `staff_mask` excludes exactly those slots from
    attention anyway.
    """
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return summed / denom


def mixed_first_pass_hidden(
    model: nn.Module, flat: dict[str, torch.Tensor], sampling_prob: float
) -> torch.Tensor:
    """DECODER_RHYTHM_ACCURACY_DESIGN.md §7.4's exposure-bias fix: the pooled hidden
    state Stage C conditions on, computed the same way `ScoreDecoder.forward`'s own
    built-in scheduled-sampling branch already does (`training/architecture/
    transformer/decoder.py`, `if self.training and sampling_prob < 1.0`) - one
    exploratory forward pass, greedy-argmax-sampled at every position, mixed
    per-position with ground truth by `sampling_prob`, then a second real forward over
    that mixed input. That branch only fires when `model.training` is True, and
    `set_probe_mode` deliberately keeps the frozen core in `.eval()` (this module's
    own docstring: dropout on an untrained core regularizes nothing, it just adds a
    train/eval mismatch) - so this calls `model.decoder.net` directly instead, the
    same way `ScoreDecoder.generate` already does, reproducing only the substitution
    logic without ever touching the model's train/eval mode.

    The real, measured problem this exists to fix: the old first pass was always
    `sampling_prob=1.0` (fully teacher-forced - every position sees ground truth for
    every earlier position, never its own prediction), so Stage C learned to pool
    hidden states real inference never produces. `sampling_prob < 1.0` here closes
    that gap at the cost of one extra forward pass, not a full autoregressive
    step-by-step decode - the model already predicts every position in parallel
    given a fixed input, so "one exploratory pass, sample, mix, one real pass" is
    O(1) extra forwards, not O(L) or O(L^2).
    """
    net = model.decoder.net
    context = model.encoder(flat["inputs"])

    rhythmsi = flat["rhythms"][:, :-1].clone()
    pitchsi = flat["pitchs"][:, :-1].clone()
    liftsi = flat["lifts"][:, :-1].clone()
    articulationsi = flat["articulations"][:, :-1].clone()
    slursi = flat["slurs"][:, :-1].clone()

    mask = flat["mask"]
    if mask.shape[1] == flat["rhythms"].shape[1]:
        mask = mask[:, :-1]
    mask = mask.bool()

    r_logits, p_logits, l_logits, _pos, a_logits, s_logits, _, _, _ = net(
        rhythms=rhythmsi, pitchs=pitchsi, lifts=liftsi,
        articulations=articulationsi, slurs=slursi,
        context=context, mask=mask, cache=None, return_center_of_attention=False,
    )
    r_sample = r_logits[:, :-1].argmax(dim=-1)
    p_sample = p_logits[:, :-1].argmax(dim=-1)
    l_sample = l_logits[:, :-1].argmax(dim=-1)
    a_sample = a_logits[:, :-1].argmax(dim=-1)
    s_sample = s_logits[:, :-1].argmax(dim=-1)

    # rand() > sampling_prob: sampling_prob=1.0 keeps ground truth everywhere (fully
    # teacher-forced, matching the old first pass exactly); lower values substitute
    # the model's own greedy prediction at that position more often.
    mix_mask = (torch.rand(r_sample.shape, device=rhythmsi.device) > sampling_prob).long()
    rhythmsi[:, 1:] = (1 - mix_mask) * rhythmsi[:, 1:] + mix_mask * r_sample
    pitchsi[:, 1:] = (1 - mix_mask) * pitchsi[:, 1:] + mix_mask * p_sample
    liftsi[:, 1:] = (1 - mix_mask) * liftsi[:, 1:] + mix_mask * l_sample
    articulationsi[:, 1:] = (1 - mix_mask) * articulationsi[:, 1:] + mix_mask * a_sample
    slursi[:, 1:] = (1 - mix_mask) * slursi[:, 1:] + mix_mask * s_sample

    _, _, _, _, _, _, x, _, _ = net(
        rhythms=rhythmsi, pitchs=pitchsi, lifts=liftsi,
        articulations=articulationsi, slurs=slursi,
        context=context, mask=mask, cache=None, return_center_of_attention=False,
    )
    return masked_mean_pool(x, mask)


def two_pass_forward(
    model: nn.Module, batch: dict[str, torch.Tensor], device: str, sampling_prob: float = 1.0
) -> dict[str, Any]:
    flat, sample_count, staff_count = flatten_staff_dim(batch)
    flat = {k: v.to(device) for k, v in flat.items()}
    staff_mask = batch["staff_mask"].to(device)

    with torch.no_grad():
        # Still computed - its own "loss" is `evaluate`'s "without staff context"
        # baseline, a fair "no Stage C at all" number this fix has no reason to touch.
        first_pass = model(**flat, sampling_prob=1.0)
        pooled = mixed_first_pass_hidden(model, flat, sampling_prob)
    pooled = pooled.view(sample_count, staff_count, -1)

    context = model.decoder.staff_context(pooled, staff_mask)
    context_flat = context.reshape(sample_count * staff_count, -1)

    second_pass = model(**flat, staff_context_emb=context_flat, sampling_prob=1.0)
    return {"first_pass": first_pass, "second_pass": second_pass}


def train_epoch(
    model: nn.Module,
    batches: object,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cpu",
    sampling_prob: float = 1.0,
) -> dict[str, Any]:
    set_probe_mode(model)
    total = 0.0
    count = 0
    for raw in batches:  # type: ignore[attr-defined]
        outputs = two_pass_forward(model, raw, device, sampling_prob=sampling_prob)
        loss = outputs["second_pass"]["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        count += 1

    mean = total / max(count, 1)
    print(f"epoch {epoch}: mean loss {mean:.4f} over {count} system batch(es)")
    return {"epoch": epoch, "loss": mean, "batches": count}


@torch.no_grad()
def evaluate(
    model: nn.Module, batches: object, device: str = "cpu", sampling_prob: float = 1.0
) -> tuple[float, float]:
    """`(mean loss with staff context, mean loss without)` over the same held-out
    systems and the same first-pass hidden state either way - see this module's own
    docstring for why this ablation, not just the training-loss trend, is what
    actually answers whether cross-staff context helped.
    """
    model.eval()
    total_with = 0.0
    total_without = 0.0
    count = 0
    for raw in batches:  # type: ignore[attr-defined]
        outputs = two_pass_forward(model, raw, device, sampling_prob=sampling_prob)
        total_without += float(outputs["first_pass"]["loss"].item())
        total_with += float(outputs["second_pass"]["loss"].item())
        count += 1
    denom = max(count, 1)
    return total_with / denom, total_without / denom


def build_batches(
    index: Path,
    config: Any,
    batch_size: int,
    workers: int,
    shuffle: bool = True,
    validation: bool = False,
    min_staves: int = 2,
) -> tuple[TorchDataLoader, int]:
    """Systems (every staff of one system, padded/stacked), not individual staves -
    `system_batch_loader.py`'s own reasoning for why this is the first mechanism in
    this codebase that needs it.
    """
    from training.transformer.data_loader import load_dataset
    from training.transformer.system_batch_loader import build_system_batches

    samples = index.read_text(encoding="utf-8").splitlines()
    datasets = load_dataset([line for line in samples if line.strip()], config,
                             val_split=1.0 if validation else 0.0)
    key = "validation" if validation else "train"
    system_dataset = build_system_batches(
        datasets[f"{key}_list"], datasets[key], min_staves=min_staves
    )
    loader = TorchDataLoader(
        system_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers
    )
    return loader, len(system_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="Dataset index.txt.")
    parser.add_argument(
        "--valid-index", type=Path, help="Held-out index.txt for the with/without ablation."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned .pth to start from.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the history JSON.")
    parser.add_argument("--weights", type=Path, help="Where to write the trained module's weights.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Systems per batch, not individual staves."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-staves", type=int, default=2,
                         help="Systems with fewer real parts than this are excluded.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sampling-prob", type=float, default=0.5,
        help="Probability the first pass keeps ground truth per-position rather than "
        "its own greedy prediction (1.0 = fully teacher-forced, the old behavior; "
        "lower closes the exposure-bias gap - see mixed_first_pass_hidden's docstring).",
    )
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR

    config = Config()
    config.enable_staff_context = True

    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    trainable = model.freeze_core_for_staff_context()
    print(f"training {len(trainable)} tensor(s), everything else frozen")
    model.to(args.device)

    batches, examples = build_batches(
        args.index, config, args.batch_size, args.workers, min_staves=args.min_staves
    )
    print(f"{examples} system(s) from {args.index}")

    valid_batches = None
    if args.valid_index:
        valid_batches, valid_examples = build_batches(
            args.valid_index, config, args.batch_size, args.workers,
            shuffle=False, validation=True, min_staves=args.min_staves,
        )
        print(f"{valid_examples} validation system(s) from {args.valid_index}")

    optimizer = torch.optim.Adam(staff_context_parameters(model), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        report = train_epoch(
            model, batches, optimizer, epoch, device=args.device,
            sampling_prob=args.sampling_prob,
        )
        if valid_batches is not None:
            with_context, without_context = evaluate(
                model, valid_batches, device=args.device, sampling_prob=args.sampling_prob
            )
            report["valid_loss_with_staff_context"] = with_context
            report["valid_loss_without_staff_context"] = without_context
            print(
                f"  valid: with staff context {with_context:.4f}, "
                f"without {without_context:.4f}, delta {without_context - with_context:+.4f}"
            )
        history.append(report)

        # Checkpoint every epoch, not just at the end - same fix train_detector.py/
        # train_profile_context.py already applied for the same reason.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"history": history, "trainable": trainable}, indent=2), encoding="utf-8"
        )
        if args.weights:
            args.weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    name: param.detach().cpu()
                    for name, param in model.named_parameters()
                    if name.startswith(NEW_PARAMETER_PREFIXES)
                },
                args.weights,
            )


if __name__ == "__main__":
    main()
