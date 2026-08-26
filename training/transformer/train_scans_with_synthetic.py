"""Add the OSSQ synthetic track to the corrected-scan mixture.

The scan runs so far train on OSSQ's *scanned* track only (34,510 staves) plus Lieder
scans and pdmx replay. The synthetic track - the same music rendered by MuseScore rather
than photographed, 42,089 staves - has never been in the mixture at all. It is the
largest untapped source available, larger than the scanned track it would join.

**Why this is a real decision and not free data.** §23 warns that adapting on new-domain
data specialises a model at the expense of everything else, and the entire point of the
scanned track is that scans are the domain that matters. Synthetic renders are easier and
more numerous, so adding them at full weight makes the majority of the mixture the domain
we are *not* trying to fit, and a model that drifts back toward synthetic would show up as
a *rising* aggregate while scan performance fell - the aggregate would hide it, because
the validation set would contain both.

So the run is set up to make that visible rather than to assume it will not happen:

- `--synthetic-weight` scales the synthetic contribution. 1.0 is the full track, which is
  what "add the synthetic track" means literally; lower values keep scans in the
  majority. The default is 1.0 and the choice is recorded in the log.
- Validation stays the mixed Lieder+OSSQ *scanned* set. Synthetic staves are deliberately
  **not** added to it: the question this run has to answer is what happens to scan
  accuracy, and a validation set that grew easier alongside the training set could not
  answer it.

Continues from the clef-corrected run's checkpoint rather than starting over, for the
same reason `train_scans_continue.py` does.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

from training.omr_datasets.convert_pdmx import pdmx_train_index
from training.transformer.train import train_transformer

OSSQ_SCANNED_INDEX = "/workspace/b0/phase7clef/train/index.txt"
OSSQ_SYNTHETIC_INDEX = "/workspace/b0/phase2clef/train/index.txt"
IMSLP_TRAIN_INDEX = "/workspace/b0/imslp_train_index.txt"
MIXED_VAL_INDEX = "/workspace/b0/mixed_valid_clef_index.txt"

IMSLP_COUNT = 3353
PDMX_REPLAY_COUNT = 6700
EPOCHS = 10


def _count(path: str) -> int:
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--synthetic-weight", type=float, default=1.0,
        help="Fraction of the synthetic track to include. 1.0 adds all of it.",
    )
    args = parser.parse_args()

    scanned = _count(OSSQ_SCANNED_INDEX)
    synthetic = int(_count(OSSQ_SYNTHETIC_INDEX) * args.synthetic_weight)
    counts = [scanned, synthetic, IMSLP_COUNT, PDMX_REPLAY_COUNT]
    total = sum(counts)
    scan_share = (scanned + IMSLP_COUNT) / total
    print(
        f"mix: OSSQ scanned {scanned}, OSSQ synthetic {synthetic} "
        f"(weight {args.synthetic_weight}), IMSLP scans {IMSLP_COUNT}, "
        f"pdmx replay {PDMX_REPLAY_COUNT} = {total}"
    )
    print(f"  real-scan share of the mixture: {scan_share:.1%}")
    print("  validation is scans only - synthetic staves are deliberately not in it")

    train_transformer(
        warm_start=True,
        dataset_index=[
            OSSQ_SCANNED_INDEX, OSSQ_SYNTHETIC_INDEX, IMSLP_TRAIN_INDEX, pdmx_train_index
        ],
        dataset_weights=[float(c) for c in counts],
        number_of_files=total,
        number_of_epochs=EPOCHS,
        validation_index=MIXED_VAL_INDEX,
        checkpoint=str(args.checkpoint),
    )


if __name__ == "__main__":
    main()
