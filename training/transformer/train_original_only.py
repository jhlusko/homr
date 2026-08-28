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

**Read a weak result carefully.** PDMX alone is one fifth of the original mix, so a
run that underperforms the pinned checkpoint has two available explanations - the
architecture, or simply less and less varied data - and this experiment cannot separate
them. A run that MATCHES or BEATS the pinned checkpoint has only one, which is why the
informative outcome here is the positive one.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

from training.omr_datasets.convert_pdmx import pdmx_train_index, pdmx_valid_index
from training.transformer.train import train_transformer

#: From scratch, so the default is train.py's own from-scratch epoch count rather than
#: the short schedule the fine-tunes use. Early stopping still applies.
EPOCHS = 35


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", nargs="+", default=[pdmx_train_index],
                        help="training index/indices from homr's original corpora")
    parser.add_argument("--weights", nargs="+", type=float,
                        help="sampling weights; default is equal")
    parser.add_argument("--val-index", default=pdmx_valid_index)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--number-of-files", type=int, default=-1,
                        help="-1 uses every row; a small value is for smoke tests")
    parser.add_argument("--warm-start", action="store_true",
                        help="start from the pinned checkpoint instead of from scratch. "
                             "Off by default: a warm start inherits the pinned model's "
                             "own data history, which is the thing being controlled for.")
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
    print(f"{'warm start from pinned' if args.warm_start else 'FROM SCRATCH'}, "
          f"{args.epochs} epochs max")
    print("none of our rebuilt Lieder data is in this mix - that is the point")

    train_transformer(
        warm_start=args.warm_start,
        dataset_index=list(args.index),
        dataset_weights=args.weights or [1.0] * len(args.index),
        number_of_files=args.number_of_files,
        number_of_epochs=args.epochs,
        validation_index=args.val_index,
    )


if __name__ == "__main__":
    main()
