"""
Train §4/§7.4 Stage C's cross-staff `StaffContextTransformer` over a frozen core.

Same frozen-core probe shape `train_profile_context.py`/`train_structured_heads.py`
already used, applied to the one genuinely new mechanism here: this run is a true
*two-pass decode* per system, not a single forward pass conditioned on precomputed
fields. For each system batch:

  1. First pass: decode every staff independently (teacher-forced, no cross-staff
     information at all) and keep the shared decoder's own pooled hidden state per
     staff (`ScoreDecoder.forward`'s `"hidden"` output, masked-mean-pooled over the
     sequence dimension).
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

`evaluate`'s with/without ablation reuses the *same* first-pass hidden state either
way: "without" is exactly the first pass's own loss (no cross-staff context at all,
since sampling_prob=1.0 disables scheduled sampling and no staff_context_emb is
passed), "with" is the second pass built from that same first pass's pooled hidden
state - the real comparison this experiment exists to produce, not just whether
training loss went down.
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


def _pool_mask(flat: dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
    """The token mask aligned to `hidden`'s own sequence length - `ScoreDecoder.
    forward` trims `mask` to `rhythms[:, :-1]`'s length internally (see its own
    `if mask.shape[1] == rhythms.shape[1]` check) before producing `"hidden"`; this
    mirrors that same trim here rather than re-deriving it from `hidden`'s shape
    alone, since a shorter caller-supplied mask must be left untouched exactly as the
    decoder itself would.
    """
    mask = flat["mask"]
    if mask.shape[1] == flat["rhythms"].shape[1]:
        mask = mask[:, :-1]
    return mask.bool()


def two_pass_forward(
    model: nn.Module, batch: dict[str, torch.Tensor], device: str
) -> dict[str, Any]:
    flat, sample_count, staff_count = flatten_staff_dim(batch)
    flat = {k: v.to(device) for k, v in flat.items()}
    staff_mask = batch["staff_mask"].to(device)

    with torch.no_grad():
        first_pass = model(**flat, sampling_prob=1.0)
    hidden = first_pass["hidden"]
    pooled = masked_mean_pool(hidden, _pool_mask(flat, hidden))
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
) -> dict[str, Any]:
    set_probe_mode(model)
    total = 0.0
    count = 0
    for raw in batches:  # type: ignore[attr-defined]
        outputs = two_pass_forward(model, raw, device)
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
def evaluate(model: nn.Module, batches: object, device: str = "cpu") -> tuple[float, float]:
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
        outputs = two_pass_forward(model, raw, device)
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
        report = train_epoch(model, batches, optimizer, epoch, device=args.device)
        if valid_batches is not None:
            with_context, without_context = evaluate(model, valid_batches, device=args.device)
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
