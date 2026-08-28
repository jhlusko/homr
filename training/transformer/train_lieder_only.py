"""Fine-tune on the verified Lieder scans only, with pdmx replay - no OSSQ scanned data.

`train_scans.py` mixes OSSQ scanned crops (32,982), Lieder scans (3,353) and pdmx replay
(6,400). The OSSQ scanned half of that is not trustworthy: `convert_ossq.py` pairs every
segment's symbolic content with a crop by `(page, system)`, taking the symbols from
`musicxml/unaligned` for *both* tracks - but that directory is keyed to the synthetic
pagination, and the scanned pages paginate differently (24 synthetic pages against 22
scanned ones for `sq8907120`). Measured over 900 staves across all 9 validation scores,
**56.7% of scanned staves are paired with the wrong music**, and the per-score collapse
rate runs from 63% to 95% - it affects every score, in proportion to how far its two
paginations drift apart.

That makes ~77% of `train_scans.py`'s mixture mislabeled, so the checkpoint it produced
should be treated as suspect rather than as a base to build on. This run therefore warm
starts from the *pinned* checkpoint, not from that one.

The Lieder pairs are in a different position entirely: they were content-verified by
fingerprinting against the piece's own MusicXML note stream (§7), the recovered ones
were re-checked the same way, and a human has reviewed them through
`stage2_pair_review_server.py`. They are small - 3,353 training pairs against OSSQ's
32,982 - but they are known-good, and a small correct corpus is worth more than a large
one that is more than half wrong.

Replay stays for the same reason it existed before: §23 warns that adapting on new-domain
data alone specialises the model at the expense of everything else, and the smaller the
domain corpus the more that matters.
"""

# flake8: noqa: T201

from training.omr_datasets.convert_pdmx import pdmx_train_index
from training.transformer.train import train_transformer

IMSLP_TRAIN_INDEX = "/workspace/b0/imslp_train_index.txt"
IMSLP_VAL_INDEX = "/workspace/b0/imslp_val_index.txt"

IMSLP_COUNT = 3353
#: ~15% of the mix, matching the fraction `train_scans.py` used - deliberately the same
#: replay ratio so this run differs from that one in *which* scan data it trains on, not
#: in how much general data it retains.
PDMX_REPLAY_COUNT = 600

#: Last run converged by epoch 5 and its epochs 5-13 sat within +/-0.0006, so the 15-epoch
#: default was ten epochs of nothing. Early stopping (patience 5) still applies.
EPOCHS = 12


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-index", default=IMSLP_TRAIN_INDEX)
    parser.add_argument("--val-index", default=IMSLP_VAL_INDEX)
    parser.add_argument(
        "--imslp-count", type=int, default=IMSLP_COUNT,
        help="Lieder pairs in the mix; sets the replay ratio against PDMX_REPLAY_COUNT.",
    )
    parser.add_argument("--replay-count", type=int, default=PDMX_REPLAY_COUNT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--checkpoint-folder",
        help="Where the trainer keeps per-epoch checkpoints. Defaults to a name derived "
             "from the seed so concurrent runs cannot delete each other's; pass "
             "current_training explicitly to reproduce the old single-run behaviour.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Trainer seed. Repeating a corpus at two seeds measures the noise floor, "
             "without which a 1pp corpus difference cannot be called a result.",
    )
    args = parser.parse_args()

    # Paths are arguments rather than constants so a rebuilt corpus can be trained on
    # without editing the module and silently changing what an earlier run meant.  The
    # defaults still point at the original indices, which read from
    # `stage2_pairs_out/` - the model-recovered pairs later quarantined as circular.
    counts = [args.imslp_count, args.replay_count]
    total = sum(counts)
    print(
        f"mix: IMSLP Lieder scans {args.imslp_count}, pdmx replay {args.replay_count} "
        f"({100 * args.replay_count / total:.1f}%), total {total}"
    )
    print(f"train index: {args.train_index}")
    print(f"val index:   {args.val_index}")
    print("OSSQ scanned deliberately excluded - see this module's docstring")
    train_transformer(
        warm_start=True,
        dataset_index=[args.train_index, pdmx_train_index],
        dataset_weights=[float(c) for c in counts],
        number_of_files=total,
        number_of_epochs=args.epochs,
        validation_index=args.val_index,
        seed=args.seed,
        checkpoint_folder=args.checkpoint_folder
        or (f"current_training_s{args.seed}" if args.seed is not None else "current_training"),
    )


if __name__ == "__main__":
    main()
