"""Fine-tune on the verified Lieder scans only, with general replay - no OSSQ scanned data.

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

`--replay CORPUS=COUNT` draws that replay from any of pdmx, grandstaff, musetrainer and
primus rather than pdmx alone. Two reasons beyond breadth for its own sake. The measured
metre gap is a supply gap: the numerators v4 never predicts are the ones a 1,300-pair PDMX
replay showed it roughly twenty times (RUNLOG IV.15.1), and grandstaff carries a different
metre distribution. And every one of these corpora now states the numerator - grandstaff
only since its reconversion, primus only since its parser was fixed - so mixing them no
longer teaches the token and its absence at the same time, which is what made the OSSQ
scanned track unusable alongside Lieder until `phase7num`.

Sizes differ by two orders of magnitude (musetrainer holds a few hundred staves against
PDMX's 32k), so a count borrowed from one corpus oversamples another; the run says so
rather than leaving it to be inferred from the mixture line.
"""

# flake8: noqa: T201

import argparse

from pathlib import Path

from training.omr_datasets.convert_grandstaff import grandstaff_train_index
from training.omr_datasets.convert_musetrainer import musetrainer_train_index
from training.omr_datasets.convert_pdmx import pdmx_train_index
from training.omr_datasets.convert_primus import primus_train_index
from training.transformer.train import train_transformer

#: Replay corpora this run can draw general data from, by name.
#:
#: Replay existed to stop a small domain corpus specialising the model away from
#: everything else (§23), and PDMX alone was a narrow reading of "everything else".
#: Widening it also widens metre coverage, which is the measured gap: the numerators v4
#: never predicts are the ones the 1,300-pair PDMX replay showed it roughly twenty times
#: (RUNLOG IV.15.1). Every corpus here now states the numerator - grandstaff only since
#: its reconversion, primus only since its parser was fixed - so a mixture no longer
#: teaches the token and its absence at once.
REPLAY_CORPORA = {
    "pdmx": pdmx_train_index,
    "grandstaff": grandstaff_train_index,
    "musetrainer": musetrainer_train_index,
    "primus": primus_train_index,
}

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


def _replay_pair(value: str) -> tuple[str, int]:
    """Parse one CORPUS=COUNT pair."""
    name, _, count = value.partition("=")
    if not count.isdigit() or int(count) <= 0:
        raise argparse.ArgumentTypeError(
            f"expected CORPUS=COUNT with a positive count, got {value!r}"
        )
    return name, int(count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-index", default=IMSLP_TRAIN_INDEX)
    parser.add_argument("--val-index", default=IMSLP_VAL_INDEX)
    parser.add_argument(
        "--imslp-count", type=int, default=IMSLP_COUNT,
        help="Lieder pairs in the mix; sets the replay ratio against PDMX_REPLAY_COUNT.",
    )
    parser.add_argument(
        "--replay-count", type=int, default=PDMX_REPLAY_COUNT,
        help="PDMX-only replay size. Kept so existing invocations mean what they did; "
        "ignored once --replay is given.",
    )
    parser.add_argument(
        "--replay", action="append", metavar="CORPUS=COUNT", type=_replay_pair, default=None,
        help="Replay corpus and how many staves to draw, repeatable: "
        "--replay pdmx=1300 --replay grandstaff=1300. Known corpora: "
        + ", ".join(sorted(REPLAY_CORPORA))
        + ". Replaces --replay-count entirely when given, so the mixture is stated in "
        "one place rather than split across two flags.",
    )
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
    replay = dict(args.replay) if args.replay else {"pdmx": args.replay_count}
    for name in replay:
        if name not in REPLAY_CORPORA:
            raise SystemExit(
                f"unknown replay corpus {name!r}; known: {', '.join(sorted(REPLAY_CORPORA))}"
            )
    missing = [n for n in replay if not Path(REPLAY_CORPORA[n]).exists()]
    if missing:
        # A silently absent corpus would train a different mixture than the one reported,
        # and the log would still claim it was included.
        raise SystemExit(
            "replay corpus not built: "
            + ", ".join(f"{n} ({REPLAY_CORPORA[n]})" for n in missing)
        )

    replay_names = sorted(replay)
    counts = [args.imslp_count, *(replay[n] for n in replay_names)]
    total = sum(counts)
    replay_total = total - args.imslp_count
    described = ", ".join(f"{n} {replay[n]}" for n in replay_names)
    print(
        f"mix: IMSLP Lieder scans {args.imslp_count}, replay [{described}] "
        f"({100 * replay_total / total:.1f}%), total {total}"
    )
    for name in replay_names:
        available = sum(1 for line in Path(REPLAY_CORPORA[name]).read_text(
            encoding="utf-8").splitlines() if line.strip())
        if replay[name] > available:
            # Asking for more than a corpus holds is oversampling, not an error - but it
            # is worth saying out loud, since musetrainer holds a few hundred staves and
            # a count borrowed from PDMX would repeat each of them many times over.
            print(
                f"  NOTE {name}: asked {replay[name]} of {available} available "
                f"- each stave repeats ~{replay[name] / max(available, 1):.1f}x"
            )
    print(f"train index: {args.train_index}")
    print(f"val index:   {args.val_index}")
    print("OSSQ scanned deliberately excluded - see this module's docstring")
    train_transformer(
        warm_start=True,
        dataset_index=[args.train_index, *(REPLAY_CORPORA[n] for n in replay_names)],
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
