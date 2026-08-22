"""
Unfrozen follow-up to `train_profile_context.py`'s frozen-core probe.

phase20 (ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md §3) found real, held-out signal training
only the 8 `decoder.profile_context.*` tensors over an otherwise-frozen model: 10/10
epochs positive, mean delta +0.0615 nats. That frozen-core result answers a narrower,
cheaper question - can *some* assignment of the new embedding, backpropagated through
weights that never move, help at all - and it came back yes. This script asks the more
expensive follow-up the frozen core's own docstring names as the natural next step: does
letting the decoder itself adapt to profile context do better than only training the
embedding that feeds it.

Deliberately still not a full-model fine-tune: `TrOMR.unfreeze_decoder_for_profile_
context` freezes the visual encoder explicitly and only unfreezes the decoder. The
encoder never sees profile context - it only processes staff image crops - so training
it would widen risk (a much bigger, harder-to-attribute change) without being the
variable this experiment tests.

Same corpus, same held-out validation split, and the same with/without-profile ablation
methodology as phase20 - deliberately unchanged, so unfreezing is the only variable that
differs from that result and any change in the ablation delta can be attributed to it,
not to a different corpus or a different measurement.

Unlike the frozen-core run, this one *can* regress the model's core competence - nothing
here still holds the "this cannot make the baseline worse" guarantee `freeze_core_for_
profile_context` gave phase20. Watch the `without`-profile column specifically each
epoch: if it gets worse than phase20's own `without` numbers at the same epoch, that is
the core eroding, not profile context helping, and is the signal to stop rather than
keep training. A conservative learning rate (default 1e-5, two orders of magnitude below
phase20's 1e-3 probe) and per-epoch full checkpoints (not just profile_context's 8
tensors - the whole decoder can have moved) are both deliberate given that risk.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

from training.transformer.train_profile_context import (
    build_batches,
    evaluate,
    load_pinned,
)


def train_epoch(
    model: nn.Module,
    batches: object,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Unlike the frozen-core run's `set_probe_mode`, the whole model (decoder included)
    trains in `train()` mode here - its dropout is part of what is being fine-tuned, not
    a mismatch between training and scoring conditions the way it would be for a core
    that never moves.
    """
    model.train()
    total = 0.0
    count = 0
    for raw in batches:  # type: ignore[attr-defined]
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw.items()}
        outputs = model(**batch)
        loss = outputs["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        count += 1

    mean = total / max(count, 1)
    print(f"epoch {epoch}: mean loss {mean:.4f} over {count} batch(es)")
    return {"epoch": epoch, "loss": mean, "batches": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="Dataset index.txt.")
    parser.add_argument(
        "--valid-index", type=Path, required=True,
        help="Held-out index.txt for the with/without ablation - required here (not "
        "optional the way it is for the frozen-core script): this run's only "
        "justification for existing is that ablation, and it must not silently skip it.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="OSSQ corpus root (score_profile_pairing.py's dataset_root).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned .pth to start from.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the history JSON.")
    parser.add_argument(
        "--checkpoint-out-dir", type=Path, required=True,
        help="Directory to write a full model checkpoint after every epoch "
        "(unlike the frozen-core script's --weights, the whole decoder can have moved, "
        "not just profile_context's 8 tensors, so the full state dict is saved).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Two orders of magnitude below phase20's 1e-3 frozen-core probe - the whole "
        "decoder can move here, so a much smaller step size is the conservative default.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR

    config = Config()
    config.enable_profile_context = True

    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    trainable = model.unfreeze_decoder_for_profile_context()
    print(f"training {len(trainable)} decoder tensor(s), encoder frozen")
    model.to(args.device)

    batches, examples = build_batches(
        args.index, config, args.batch_size, args.workers, str(args.dataset_root)
    )
    print(f"{examples} example(s) from {args.index}")

    valid_batches, valid_examples = build_batches(
        args.valid_index, config, args.batch_size, args.workers, str(args.dataset_root),
        shuffle=False, validation=True,
    )
    print(f"{valid_examples} validation example(s) from {args.valid_index}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    history = []
    args.checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        report = train_epoch(model, batches, optimizer, epoch, device=args.device)
        with_profile = evaluate(model, valid_batches, device=args.device)
        without_profile = evaluate(model, valid_batches, device=args.device, force_no_profile=True)
        report["valid_loss_with_profile"] = with_profile
        report["valid_loss_without_profile"] = without_profile
        print(
            f"  valid: with profile {with_profile:.4f}, without {without_profile:.4f}, "
            f"delta {without_profile - with_profile:+.4f}"
        )
        history.append(report)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"history": history, "trainable": trainable}, indent=2), encoding="utf-8"
        )
        # Full checkpoint every epoch, not just at the end - the whole decoder can have
        # moved here, unlike the frozen-core script where only 8 tensors ever change.
        torch.save(
            model.state_dict(), args.checkpoint_out_dir / f"epoch_{epoch}.pth"
        )


if __name__ == "__main__":
    main()
