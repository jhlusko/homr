"""Targeted rare-metre continuation from a completed, scored checkpoint.

This is intentionally a continuation, not another warm start from the pinned release:
the selected starting checkpoint must already be a model that holds the desired domain
trade-off.  The rare replay is added to the same scanned-OSSQ/Lieder/PDMX mixture that
protects the rest of the vocabulary from a narrowly targeted update.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

from training.transformer.train import train_transformer
from training.transformer.train_scans import (
    IMSLP_COUNT,
    IMSLP_TRAIN_INDEX,
    IMSLP_VAL_INDEX,
    OSSQ_COUNT,
    OSSQ_SCANNED_INDEX,
    PDMX_REPLAY_COUNT,
)
from training.omr_datasets.convert_pdmx import pdmx_train_index

RARE_NUMERATOR_INDEX = "/workspace/b0/lieder-rebuild/rare_numerator_replay_index.txt"
RARE_NUMERATOR_COUNT = 796
EPOCHS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Completed .pth checkpoint to continue.")
    parser.add_argument("--rare-index", default=RARE_NUMERATOR_INDEX)
    parser.add_argument("--rare-count", type=int, default=RARE_NUMERATOR_COUNT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-folder", default="current_training_rare_numerators")
    args = parser.parse_args()

    required = [
        ("checkpoint", args.checkpoint),
        ("OSSQ index", OSSQ_SCANNED_INDEX),
        ("Lieder index", IMSLP_TRAIN_INDEX),
        ("validation index", IMSLP_VAL_INDEX),
        ("PDMX index", pdmx_train_index),
        ("rare-numerator index", args.rare_index),
    ]
    missing = [f"{name}: {path}" for name, path in required if not Path(path).is_file()]
    if missing:
        raise SystemExit("required input missing:\n" + "\n".join(missing))
    if args.rare_count <= 0 or args.epochs <= 0:
        raise SystemExit("--rare-count and --epochs must be positive")

    counts = [OSSQ_COUNT, IMSLP_COUNT, PDMX_REPLAY_COUNT, args.rare_count]
    total = sum(counts)
    print(
        "mix: OSSQ scanned %d, Lieder %d, PDMX replay %d, balanced rare numerators %d "
        "(%.1f%%), total %d" % (*counts, 100 * args.rare_count / total, total)
    )
    print(f"continuing from: {args.checkpoint}")
    train_transformer(
        warm_start=True,
        checkpoint=args.checkpoint,
        dataset_index=[OSSQ_SCANNED_INDEX, IMSLP_TRAIN_INDEX, pdmx_train_index, args.rare_index],
        dataset_weights=[float(count) for count in counts],
        number_of_files=total,
        number_of_epochs=args.epochs,
        validation_index=IMSLP_VAL_INDEX,
        seed=args.seed,
        checkpoint_folder=args.checkpoint_folder,
    )


if __name__ == "__main__":
    main()
