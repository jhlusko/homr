"""Re-measure a trained Stage C checkpoint, independently of the run that chose it.

The trainer keeps the epoch with the best ablation delta, which is the right thing to keep
and the wrong number to report. Epoch-to-epoch spread in that delta is the same size as
the delta itself - v2's first two epochs scored +0.0037 and +0.0009 at identical training
loss - so taking the maximum over eight epochs reports the upper tail of the noise, not
the effect. Selection and estimation from the same number is the circularity this project
already documented for evaluation sets filtered by agreement with the model.

So: load the kept weights, run the with/without ablation again as its own measurement, and
report that. Repeated over several passes, because the ablation's first pass is
teacher-forced and the second mixes in the model's own predictions, which makes the figure
stochastic - one pass would substitute a different noise for the one being avoided.
"""

# flake8: noqa: T201

import argparse
import statistics
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="The pinned core .pth.")
    parser.add_argument("--weights", type=Path, required=True, help="The trained Stage C .pt.")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.transformer.train_staff_context import (
        build_batches,
        evaluate,
        load_pinned,
    )

    config = Config()
    # The module only exists on the model when this is on, and these are its weights -
    # loading them into a model without it would silently do nothing.
    config.enable_staff_context = True
    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    model.freeze_core_for_staff_context()

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state)
    gate = float(state["decoder.staff_context.gate"].item())
    print(f"loaded {loaded} Stage C tensors, gate {gate:+.6f}")
    if unexpected:
        raise SystemExit(f"weights do not belong to this model: {unexpected[:3]}")
    model.to(args.device)
    model.eval()

    batches, count = build_batches(
        args.index, config, args.batch_size, args.workers, shuffle=False, validation=True
    )
    print(f"{count} validation system(s)")

    deltas = []
    for attempt in range(1, args.passes + 1):
        with_context, without_context = evaluate(model, batches, device=args.device)
        delta = without_context - with_context
        deltas.append(delta)
        print(
            f"  pass {attempt}: with {with_context:.4f}, without {without_context:.4f}, "
            f"delta {delta:+.4f}"
        )

    mean = statistics.mean(deltas)
    spread = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    print(f"\nmean delta {mean:+.4f}, sd {spread:.4f} over {len(deltas)} passes")
    print(f"as a share of the without-context loss: {mean / max(without_context, 1e-9):.2%}")
    if spread >= abs(mean) / 2:
        print(
            "\nThe spread is comparable to the effect. Report this as indistinguishable\n"
            "from noise rather than as a gain."
        )


if __name__ == "__main__":
    main()
