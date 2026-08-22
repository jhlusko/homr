"""
Train §7.3's score-profile context embedding over a frozen core.

The question this run answers, mirroring `train_structured_heads.py`'s own reasoning:
can profile context - injected as an additive term into the decoder's existing input
embedding, gated at zero initialization - move the model's own existing loss at all,
without letting the rest of the model adapt? A frozen core answers a narrower, cheaper
question first: is there *any* assignment of the embedding tables and gate,
backpropagated through weights that never move, that helps. If this finds nothing,
unfreezing the core and letting it adapt to the signal is the natural next, more
expensive experiment; if it does, that is real evidence the signal is usable before
paying for the bigger one.

Unlike the structured heads, profile context does not add a new loss channel - it
conditions the input to the *existing* rhythm/pitch/... objective, so this trains
against the model's own `loss` directly. No separate structured-loss machinery, and no
custom collate function either: `context_to_batch_fields` (profile_context.py) already
gives every sample in a `--dataset-root` run the same keys and shapes (unlike the
structured heads' per-sample-optional notation sidecars), so the default collator
handles a batch on its own.

`evaluate`'s `force_no_profile` ablation is the actual answer this run measures: the
same held-out staves, the same trained gate, the only difference is whether the model
was told anything about the part it is looking at. A lower loss with profile context
than without, on the same examples, is the result this experiment exists to produce -
not just "the loss went down while training," which a frozen core's only two knobs
(the gate and the sub-embeddings) could technically do by memorising training-set
quirks that do not generalise to the held-out comparison.
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
NEW_PARAMETER_PREFIXES = ("decoder.profile_context.",)


def set_probe_mode(model: nn.Module) -> None:
    """Frozen core in eval mode, the profile context module in train mode - the same
    reasoning `train_structured_heads.py`'s own `set_probe_mode` gives: a frozen core's
    dropout would put a gap between what the module trains on and what it is scored on,
    without regularising anything, since nothing about the core is being learned here.
    """
    model.eval()
    profile_context = getattr(getattr(model, "decoder", None), "profile_context", None)
    if profile_context is not None:
        profile_context.train()


def profile_context_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if name.startswith(NEW_PARAMETER_PREFIXES)]


def load_pinned(model: nn.Module, checkpoint: Path) -> None:
    """Load the pretrained weights, allowing only the new module to be missing."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report = load_checkpoint(model, state, NEW_PARAMETER_PREFIXES)
    print(f"checkpoint: {report.describe()}")


def train_epoch(
    model: nn.Module,
    batches: object,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cpu",
) -> dict[str, Any]:
    set_probe_mode(model)
    total = 0.0
    total_duration_adherence = 0.0
    count = 0
    for raw in batches:  # type: ignore[attr-defined]
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw.items()}
        outputs = model(**batch)
        loss = outputs["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        # §7.3's duration-adherence loss - visible separately even though it's folded
        # into "loss" above, so a run with the weight turned on can actually see
        # whether this term is moving, not just the combined total. .get(...) rather
        # than a bare index: this script's own tests exercise a lightweight fake model
        # whose outputs predate this key, and a real model with duration_adherence_
        # weight left at its 0.0 default still returns it (a zero tensor - see
        # calDurationAdherenceLoss's own docstring), so only a caller with neither of
        # those needs the fallback.
        duration_adherence = outputs.get("loss_duration_adherence")
        if duration_adherence is not None:
            total_duration_adherence += float(duration_adherence.item())
        count += 1

    mean = total / max(count, 1)
    mean_duration_adherence = total_duration_adherence / max(count, 1)
    print(
        f"epoch {epoch}: mean loss {mean:.4f} "
        f"(duration_adherence {mean_duration_adherence:.4f}) over {count} batch(es)"
    )
    return {
        "epoch": epoch,
        "loss": mean,
        "loss_duration_adherence": mean_duration_adherence,
        "batches": count,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, batches: object, device: str = "cpu", force_no_profile: bool = False
) -> float:
    """Mean loss over `batches`. `force_no_profile` re-runs the same batches with every
    `profile_present` flag zeroed - see this module's own docstring for why this
    ablation, not just the training-loss trend, is what actually answers whether
    profile context helped.
    """
    model.eval()
    total = 0.0
    count = 0
    for raw in batches:  # type: ignore[attr-defined]
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw.items()}
        if force_no_profile and "profile_present" in batch:
            batch["profile_present"] = torch.zeros_like(batch["profile_present"])
        outputs = model(**batch)
        total += float(outputs["loss"].item())
        count += 1
    return total / max(count, 1)


def build_batches(
    index: Path,
    config: Any,
    batch_size: int,
    workers: int,
    dataset_root: str,
    shuffle: bool = True,
    validation: bool = False,
) -> tuple[TorchDataLoader, int]:
    """The loader, with `--dataset-root` set so every sample carries the same
    `profile_*` keys (`score_profile_pairing.py` resolves what it can; an unresolvable
    or non-OSSQ sample gets the uniform "missing" case, not a different shape) - which
    is exactly what makes the default collator usable here with no custom `collate_fn`,
    unlike `train_structured_heads.py`'s own `build_batches`.
    """
    from training.transformer.data_loader import load_dataset

    samples = index.read_text(encoding="utf-8").splitlines()
    datasets = load_dataset(
        [line for line in samples if line.strip()],
        config,
        val_split=1.0 if validation else 0.0,
        dataset_root=dataset_root,
    )
    wrapped = datasets["validation" if validation else "train"]
    loader = TorchDataLoader(
        wrapped, batch_size=batch_size, shuffle=shuffle, num_workers=workers
    )
    return loader, len(wrapped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="Dataset index.txt.")
    parser.add_argument(
        "--valid-index", type=Path, help="Held-out index.txt for the with/without ablation."
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="OSSQ corpus root (score_profile_pairing.py's dataset_root).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned .pth to start from.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the history JSON.")
    parser.add_argument("--weights", type=Path, help="Where to write the trained module's weights.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--duration-adherence-weight",
        type=float,
        default=0.0,
        help=(
            "DECODER_RHYTHM_ACCURACY_DESIGN.md §7.3's ground-truth-supervised "
            "measure-duration adherence loss weight - 0.0 (default) leaves the "
            "existing objective untouched, exactly as before this flag existed."
        ),
    )
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR

    config = Config()
    config.enable_profile_context = True
    config.duration_adherence_weight = args.duration_adherence_weight

    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    trainable = model.freeze_core_for_profile_context()
    print(f"training {len(trainable)} tensor(s), everything else frozen")
    model.to(args.device)

    batches, examples = build_batches(
        args.index, config, args.batch_size, args.workers, str(args.dataset_root)
    )
    print(f"{examples} example(s) from {args.index}")

    valid_batches = None
    if args.valid_index:
        valid_batches, valid_examples = build_batches(
            args.valid_index, config, args.batch_size, args.workers, str(args.dataset_root),
            shuffle=False, validation=True,
        )
        print(f"{valid_examples} validation example(s) from {args.valid_index}")

    optimizer = torch.optim.Adam(profile_context_parameters(model), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        report = train_epoch(model, batches, optimizer, epoch, device=args.device)
        if valid_batches is not None:
            with_profile = evaluate(model, valid_batches, device=args.device)
            without_profile = evaluate(model, valid_batches, device=args.device, force_no_profile=True)
            report["valid_loss_with_profile"] = with_profile
            report["valid_loss_without_profile"] = without_profile
            print(
                f"  valid: with profile {with_profile:.4f}, without {without_profile:.4f}, "
                f"delta {without_profile - with_profile:+.4f}"
            )
        history.append(report)

        # Checkpoint every epoch, not just at the end - 27.91's gap, same fix
        # train_detector.py already applied this session.
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
