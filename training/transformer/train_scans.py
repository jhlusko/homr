"""Stage 2 domain-adaptation run: real scans, with a small general-data replay.

Launcher rather than logic - every mechanism it uses lives in `train.py`. It exists
so the run's configuration is a reviewable artifact instead of a shell command
typed once at 3am, since three of its choices are easy to get silently wrong:

- **`number_of_files` must be positive.** `mix_training_sets` short-circuits to
  "concatenate everything" when it is negative and ignores `dataset_weights`
  entirely, so the replay ratio below would silently become "all 35,800 pdmx
  files", i.e. the opposite of a small replay. Weights are set to the desired
  per-source counts and the total to their sum, which makes each source contribute
  exactly its target.
- **The validation split is score-disjoint and built ahead of time**
  (`split_pairs_by_score.py`), not `load_dataset`'s positional `val_split`, which
  slices an already-shuffled list and would put systems of one score on both sides
  - forbidden by ENSEMBLE_TRANSCRIPTION_DESIGN.md §13.5.
- **Naming the indexes explicitly suppresses auto-download.** The default path
  would notice Lieder/grandstaff/primus/musetrainer are missing here and start
  hours of downloading and re-rendering.

Replay is pdmx only, at roughly 15% - "only enough to prevent catastrophic
forgetting" per the user, against §23's warning that adapting on new-domain data
alone specialises the model at the expense of everything else.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

from training.omr_datasets.convert_pdmx import pdmx_train_index
from training.transformer.train import train_transformer
from training.transformer.train_lieder_only import REPLAY_CORPORA, _replay_pair

#: The *current* scanned conversion, three builds on from the original. `phase7` took
#: every scanned crop's symbols from `musicxml/unaligned` - the synthetic pagination -
#: and paired 56.7% of staves with the wrong music (§7). `phase7fix` corrected that but
#: predates the clef repair, and `phase7clef` predates the metre numerator: it carries
#: `timeSignatureBeats_*` on 0 staves while Lieder and PDMX, mixed in below, carry it
#: throughout - so training on it taught the model to both state and omit the numerator
#: for the same notation (RUNLOG IV.15). All three earlier builds are deliberately not
#: referenced any more: re-running an old mixture is the single easiest way to undo a
#: fix, and a path constant is where that happens.
OSSQ_SCANNED_INDEX = "/workspace/b0/phase7num/train/index.txt"
#: The v4 boundary-safe Lieder corpus, not the original. The original is what v4 exists
#: to replace: IV.10 found grouped-boundary displacement fabricating individual cuts in
#: it, confirmed on six human-reviewed systems. Mixing corrected OSSQ with those labels
#: would put back on one side what was just removed from the other.
IMSLP_TRAIN_INDEX = "/workspace/b0/lieder-rebuild/imslp_train_index_v4_boundary_safe.txt"
#: Both domains, not just Lieder. The previous run validated on the 362 Lieder
#: staves alone, so nothing in its reported accuracy ever measured the OSSQ half -
#: which is part of why 56.7% of that half being mislabeled did not show up as a
#: falling number. A mixed held-out set makes the scanned track answerable.
IMSLP_VAL_INDEX = "/workspace/b0/mixed_valid_index.txt"

OSSQ_COUNT = 34510
IMSLP_COUNT = 3622  # the v4 consensus split
#: ~15% of the total mix. Deliberate and adjustable - a replay fraction in the
#: 10-20% band is the usual range for mitigating forgetting; nothing here measured
#: the right value for this model.
PDMX_REPLAY_COUNT = 6700


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ossq-count", type=int, default=OSSQ_COUNT)
    parser.add_argument("--imslp-count", type=int, default=IMSLP_COUNT)
    parser.add_argument("--train-index", default=IMSLP_TRAIN_INDEX)
    parser.add_argument("--val-index", default=IMSLP_VAL_INDEX)
    parser.add_argument(
        "--replay", action="append", metavar="CORPUS=COUNT", type=_replay_pair, default=None,
        help="Replay corpus and count, repeatable. Defaults to pdmx alone at "
        f"{PDMX_REPLAY_COUNT}. Known: " + ", ".join(sorted(REPLAY_CORPORA)),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-folder", default=None)
    args = parser.parse_args()

    replay = dict(args.replay) if args.replay else {"pdmx": PDMX_REPLAY_COUNT}
    unknown = [n for n in replay if n not in REPLAY_CORPORA]
    if unknown:
        raise SystemExit(f"unknown replay corpus: {', '.join(unknown)}")
    missing = [n for n in replay if not Path(REPLAY_CORPORA[n]).exists()]
    if missing:
        raise SystemExit(
            "replay corpus not built: "
            + ", ".join(f"{n} ({REPLAY_CORPORA[n]})" for n in missing)
        )

    replay_names = sorted(replay)
    counts = [args.ossq_count, args.imslp_count, *(replay[n] for n in replay_names)]
    total = sum(counts)
    replayed = total - args.ossq_count - args.imslp_count
    described = ", ".join(f"{n} {replay[n]}" for n in replay_names)
    print(f"mix: OSSQ scanned {args.ossq_count}, IMSLP scans {args.imslp_count}, "
          f"replay [{described}] ({100 * replayed / total:.1f}%) = {total} files")
    print(f"  ossq index:  {OSSQ_SCANNED_INDEX}")
    print(f"  lieder index:{args.train_index}")
    print(f"  val index:   {args.val_index}")
    extra = {}
    if args.epochs is not None:
        extra["number_of_epochs"] = args.epochs
    if args.seed is not None:
        extra["seed"] = args.seed
    if args.checkpoint_folder:
        extra["checkpoint_folder"] = args.checkpoint_folder
    train_transformer(
        warm_start=True,
        dataset_index=[
            OSSQ_SCANNED_INDEX,
            args.train_index,
            *(REPLAY_CORPORA[n] for n in replay_names),
        ],
        dataset_weights=[float(c) for c in counts],
        number_of_files=total,
        validation_index=args.val_index,
        **extra,
    )


if __name__ == "__main__":
    main()
