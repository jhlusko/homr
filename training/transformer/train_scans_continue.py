"""Continue the corrected-scan run on the clef-corrected corpus.

`train_scans.py` is the run that first trained on OSSQ with its pagination fixed
(`phase7fix`). This continues from *that run's own checkpoint* rather than starting again
from the pinned one, because the only thing changing underneath is the clef repair:
2.4% of staves arrived with no clef token at all, and `convert_ossq.py` now carries the
clef forward from the previous segment of the same part (`ensure_clef`). Everything else
about the corpus is identical, so throwing away the epochs already spent on it would buy
nothing.

Two things this therefore needs, and both are easy to get silently wrong:

- **The base checkpoint must be the finished run's own weights**, not the pinned
  checkpoint `train.py` warm starts from by default. `--checkpoint` names it explicitly;
  the pinned file is left alone, since it is what production and every other run loads.
- **The corpus must be `phase7clef`, not `phase7fix`.** Pointing at the previous corpus
  would train more epochs on the labels this run exists to replace, and nothing in the
  output would say so.

Validation stays the mixed Lieder+OSSQ set for the same reason `train_scans.py` adopted
it: a Lieder-only held-out set never measures the OSSQ half at all, which is how 56.7%
of that half being mislabeled stayed invisible.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

from training.omr_datasets.convert_pdmx import pdmx_train_index
from training.transformer.train import train_transformer

OSSQ_SCANNED_INDEX = "/workspace/b0/phase7clef/train/index.txt"
IMSLP_TRAIN_INDEX = "/workspace/b0/imslp_train_index.txt"
MIXED_VAL_INDEX = "/workspace/b0/mixed_valid_clef_index.txt"

IMSLP_COUNT = 3353
PDMX_REPLAY_COUNT = 6700

#: The previous run converged slowly rather than plateauing (0.9457 -> 0.9666 over five
#: epochs), so this is not the "six epochs is plenty" case the Lieder run was. Early
#: stopping still applies.
EPOCHS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="The previous run's saved weights to continue from.",
    )
    args = parser.parse_args()

    ossq_count = sum(
        1 for line in Path(OSSQ_SCANNED_INDEX).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    counts = [ossq_count, IMSLP_COUNT, PDMX_REPLAY_COUNT]
    total = sum(counts)
    print(
        f"mix: OSSQ scanned (clef-corrected) {ossq_count}, IMSLP scans {IMSLP_COUNT}, "
        f"pdmx replay {PDMX_REPLAY_COUNT} ({100 * PDMX_REPLAY_COUNT / total:.1f}%) = {total}"
    )
    print(f"continuing from {args.checkpoint}")

    train_transformer(
        warm_start=True,
        dataset_index=[OSSQ_SCANNED_INDEX, IMSLP_TRAIN_INDEX, pdmx_train_index],
        dataset_weights=[float(c) for c in counts],
        number_of_files=total,
        number_of_epochs=EPOCHS,
        validation_index=MIXED_VAL_INDEX,
        checkpoint=str(args.checkpoint),
    )


if __name__ == "__main__":
    main()
