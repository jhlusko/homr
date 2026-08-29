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

from training.omr_datasets.convert_pdmx import pdmx_train_index
from training.transformer.train import train_transformer

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
IMSLP_TRAIN_INDEX = "/workspace/b0/imslp_train_index.txt"
#: Both domains, not just Lieder. The previous run validated on the 362 Lieder
#: staves alone, so nothing in its reported accuracy ever measured the OSSQ half -
#: which is part of why 56.7% of that half being mislabeled did not show up as a
#: falling number. A mixed held-out set makes the scanned track answerable.
IMSLP_VAL_INDEX = "/workspace/b0/mixed_valid_index.txt"

OSSQ_COUNT = 34510
IMSLP_COUNT = 3353
#: ~15% of the total mix. Deliberate and adjustable - a replay fraction in the
#: 10-20% band is the usual range for mitigating forgetting; nothing here measured
#: the right value for this model.
PDMX_REPLAY_COUNT = 6700


def main() -> None:
    counts = [OSSQ_COUNT, IMSLP_COUNT, PDMX_REPLAY_COUNT]
    total = sum(counts)
    print(f"mix: OSSQ scanned {OSSQ_COUNT}, IMSLP scans {IMSLP_COUNT}, "
          f"pdmx replay {PDMX_REPLAY_COUNT} ({100 * PDMX_REPLAY_COUNT / total:.1f}%) "
          f"= {total} files")
    train_transformer(
        warm_start=True,
        dataset_index=[OSSQ_SCANNED_INDEX, IMSLP_TRAIN_INDEX, pdmx_train_index],
        dataset_weights=[float(c) for c in counts],
        number_of_files=total,
        validation_index=IMSLP_VAL_INDEX,
    )


if __name__ == "__main__":
    main()
