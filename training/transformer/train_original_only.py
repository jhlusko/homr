"""Train the current architecture on homr's ORIGINAL corpora only - none of our data.

Every checkpoint since 426 has been trained on data this project built: rebuilt Lieder
labels, consensus arbitration, pseudo-labels. Every one of those is a place a data
defect could hide, and several have been found. What has never been tested is the
architecture on its own, against corpora nobody here relabelled.

That is what this is. It is a control, not a product: if the current model reaches or
beats the pinned checkpoint on data neither of them has any special relationship with,
the architecture is exonerated and the problem is upstream of it, in what we feed it.

**homr's original mix is five corpora**, not one - `train.py` defaults to lieder
(synthetic renders, NOT our scan corpus), grandstaff, primus, pdmx and musetrainer.
Only pdmx is built on this machine, so `--index` takes whichever of the five exist and
defaults to pdmx alone.

**This is a fine-tune, matched to the runs it is a control for.** 447 and 448 warm
started from the pinned checkpoint, mixed ~8,300 samples and ran 12 epochs. So does
this - same start, same size, same schedule. The only thing that differs is where the
data comes from: theirs is our rebuilt Lieder corpus, this is PDMX. That is what makes
the comparison mean something. Training from scratch instead would answer a different
question at ten times the cost, and would confound "is our data the problem" with "is
one corpus enough to train from nothing".

**Read a weak result carefully.** PDMX alone is one fifth of the original mix, so a
run that underperforms has two available explanations - the data quantity or the
architecture - and this design cannot separate them. A run that holds up on the
independent benchmarks where 448 regressed has one: our corpus is what hurt. That is
the informative outcome.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

from training.omr_datasets.convert_pdmx import pdmx_train_index, pdmx_valid_index
from training.transformer.train import train_transformer

#: Matched to the Lieder fine-tunes this controls for: 12 epochs, and the same total
#: sample count they mixed (6,989 Lieder + 1,300 replay). Early stopping, patience 5,
#: still applies.
EPOCHS = 12
MATCHED_SAMPLES = 8289


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", nargs="+", default=[pdmx_train_index],
                        help="training index/indices from homr's original corpora")
    parser.add_argument("--weights", nargs="+", type=float,
                        help="sampling weights; default is equal")
    parser.add_argument("--val-index", default=pdmx_valid_index)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--number-of-files", type=int, default=MATCHED_SAMPLES,
                        help="samples per epoch; the default matches the Lieder runs. "
                             "-1 uses every row.")
    parser.add_argument("--from-scratch", action="store_true",
                        help="train from random init instead of warm starting. Answers a "
                             "different question - whether the architecture can learn from "
                             "one corpus alone - and costs roughly ten times as much.")
    args = parser.parse_args()

    missing = [p for p in args.index if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"training index does not exist: {missing}")
    if not Path(args.val_index).is_file():
        raise SystemExit(f"validation index does not exist: {args.val_index}")

    # The PDMX splits are disjoint (tests/test_pdmx_split_disjoint.py), and this run
    # exists to produce a number that can be trusted, so check rather than assume:
    # a contaminated validation set would make the control look better than it is,
    # in exactly the direction that would mislead.
    val_rows = {line.strip() for line in Path(args.val_index).read_text().splitlines() if line.strip()}
    for path in args.index:
        train_rows = {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}
        overlap = train_rows & val_rows
        if overlap:
            raise SystemExit(
                f"{len(overlap)} row(s) of {path} are also in {args.val_index}; "
                "refusing to train a control on a contaminated split"
            )
        print(f"train {Path(path).name}: {len(train_rows):,} rows, 0 shared with validation")
    print(f"valid {Path(args.val_index).name}: {len(val_rows):,} rows")
    print(f"{'FROM SCRATCH' if args.from_scratch else 'warm start from the pinned checkpoint'}, "
          f"{args.epochs} epochs max, {args.number_of_files} samples per epoch")
    print("none of our rebuilt Lieder data is in this mix - that is the point")

    train_transformer(
        warm_start=not args.from_scratch,
        dataset_index=list(args.index),
        dataset_weights=args.weights or [1.0] * len(args.index),
        number_of_files=args.number_of_files,
        number_of_epochs=args.epochs,
        validation_index=args.val_index,
    )


if __name__ == "__main__":
    main()
