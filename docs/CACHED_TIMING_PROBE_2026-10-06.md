# Resumable cached timing-head probe — 2026-10-06

> **2026-10-07 semantic correction:** the validation ratio-2:1 examples previously
> called “duplets” are tremolo-associated (all 135 targets), including unmarked
> members of tremolo-bearing chords. These results do not qualify ordinary duplets.
> The piano part does have a 4/4 declaration; its vocal peer supplies independent
> bar-length evidence. A legacy paired-tremolo rule outperforms the head policy on
> the inspected complete page. See [the staff-lane investigation](STAFF_LANE_TREMOLO_TIMING_2026-10-07.md).


Implemented `training/transformer/train_cached_timing_heads.py`: hash-bound
dataset preparation, resumable teacher-forced feature extraction, and training
of four separate linear timing projections over a frozen core. Full extraction
and six epochs are running locally in persistent tmux `timing-probe-full`.
Twenty focused tests pass. No runtime/model/corpus pin or deployment changes.

## Inputs and scope

Run root:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-probe-20261006/`.

| Source | Train crops | Validation crops |
| --- | ---: | ---: |
| Lieder, unchanged qualified labels | 3,409 | 291 |
| OSSQ, qualified Borodin training / corrected Haydn+d’Indy validation | 529 | 8 |
| Total | 3,938 | 299 |

All 4,237 inputs were preflighted: image/token/timing hashes, loader/vocabulary,
sequence length and disjoint score sets. No PDMX. Changed Lieder labels remain
excluded. The previous OSSQ subsets are replaced, not concatenated. Eight OSSQ
validation crops cover only two source scores; this is a bounded probe, not an
OSSQ-wide qualification. OSSQ training is also one score and oversampling cannot
create more score diversity.

Base: accepted September core, `old-core-20260918.pth`, SHA-256
`1b2855ce3facea9aa8b0017f6e30143abd9aa43569a2f9ce1ab46bf8bf818fa3`.
It loads strictly with structured heads disabled, stays in eval mode and has
every parameter frozen. Feature extraction uses float32 CPU and four threads,
the existing normalization/canvas path, and existing audit libraries through
`PYTHONPATH`; no environment rebuild/install. The core file hash is rechecked
after extraction. V7 notation projections do not affect the shared features.

Features are teacher-forced: hidden position `t` predicts reference token `t+1`.
Timing targets receive exactly the same one-position shift; control/EOS/grace
positions remain masked. Packed caches retain original token indices and only
rows supervised by at least one timing head. Ground-truth preceding symbols
make this a representation probe; its metrics are not free-decoding accuracy.

## Objective and resume guarantees

Four projections: elapsed duration, written duration, tuplet ratio, full-measure
rest. They share no trainable parameters. Core recognition loss is never used
for updates. Four explicit loss weights of 1 mean independent mean cross-entropy
objectives; there is no competition with pitch/rhythm or the legacy 1/12 joint
notation weight. No focal loss or class weighting in the initial probe.

Each epoch cycles through all Lieder training crops and oversamples OSSQ to the
same number of crop draws. Batches draw four crops per source; per-head losses
are averaged within source, then the two source losses are averaged equally.
Thus longer/more numerous Lieder examples do not dominate the objective.
AdamW starts at learning rate 0.001. Logs retain per-source/per-head training
loss/support, validation loss/accuracy and per-class support/correct counts.
Frozen target dimensions are 51/21/22/2; unsupported rare classes remain reported,
not removed or discovered from validation.

After **every optimizer update**, `latest.pth` atomically saves head parameters,
AdamW state, epoch, next batch, accumulated loss/support and history. The batch
order is deterministic by seed+epoch. Epoch bundles/history are also saved.
Resuming is automatic in the same directory. Increasing `--epochs` is allowed;
changing data, feature hashes, learning rate, batch size, code or identity is
refused. Feature extraction atomically saves each crop and verifies prior
cache identities when resumed. A partial feature cache requires the explicit
`--allow-partial` smoke option. Smoke and full training have separate outputs.

Probe bundles use `trainingStatus=probe-trained`, which the existing inference
loader rejects. Finishing six epochs does not activate inference or imply
promotion. No trained timing capability manifest is published by this runner.

The interrupted/resumed optimizer test matches uninterrupted weights and full
history **bit-for-bit**, then extends to another epoch. Changed cache tensors
and changed learning rate are refused. A real 16-crop smoke deliberately stopped
after one update, resumed, and finished two epochs; probe inference refusal
was also checked.

## Benchmark, monitoring and next steps

The 16-crop benchmark contains four crops per source/split. Mean extraction:
Lieder 0.764 seconds/crop, OSSQ 0.644 seconds/crop. All cached hidden tensors are
finite float32. Pilot cache is about 1 MB. Initial full-cache estimate: 45–70
minutes; watcher updates ETA from actual progress. Training ETA will be measured
once features finish. Four-thread full extraction is already making progress.

Persistent job and monitor:

- `control/full.sh`: extraction then six head epochs, failing closed on extraction
  error; `control/full.EXIT` records final exit code.
- `control/watch.py`: separate watcher inside tmux, writing `control/status.json`
  every 15 seconds with phase, crop progress, extraction ETA and completed epochs.
- `control/extract-full.log`, `control/train-full.log`: actual work logs.
- `features/features.json`: current cache manifest; `features-pilot16.json` is
  frozen for the separate two-epoch smoke.
- `full/latest.pth`, `full/history.json`: full training state and per-head curves
  once training starts. The watcher alone does not assert training success.

Resume feature extraction with the same command in `control/full.sh`; existing
crops are checked and reused. Resume/add epochs using:

```bash
export PYTHONPATH=/home/jhlusko/workspace/homr-artifacts/tools/audit-python:/home/jhlusko/workspace/homr
TIMING_RUN_ROOT=/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-probe-20261006
.venv/bin/python -m training.transformer.train_cached_timing_heads train \
  --dataset "$TIMING_RUN_ROOT/dataset.json" \
  --features "$TIMING_RUN_ROOT/features/features.json" \
  --out "$TIMING_RUN_ROOT/full" --epochs 10 --threads 4
```

Run this from `/home/jhlusko/workspace/homr`, after the current process finishes;
do not start competing writers in the same directory.

Next read six-epoch loss curves and per-source/per-class validation, especially
triplet recall and full-measure rests versus majority-class behavior. If the
common classes learn, evaluate on real free-decoded features and expand held-out
OSSQ coverage before duration/export integration. Unsupported rare classes,
the small OSSQ validation set and single-score training are explicit limitations.

## Completion and sparse-mask fix — 2026-10-07

All4237 feature crops and six epochs finished. A false assertion stopped the first
training attempt at next_batch16: no full-rest labels in either source meant a
legitimately absent gradient. Fixed to check supervision before requiring a
gradient; nonfinite gradients still fail. Explicit hash-bound migration retained
optimizer/weights/batch position and running losses; original run archived in
`full-first-attempt`, receipt in `full/resume-migration.json`. Cached extraction
identity stays bound to its original source; reuse it for training without
rerunning extraction under the changed code. Nine focused tests pass.
Lieder epoch6 accuracy97.60% elapsed,99.19% written,98.88% ratio,99.56% full-rest.
Ratio minorities remain64.44% duplet/60.61% triplet; free-decoding evidence pending.
See docs/C1_SEMANTIC_BASELINE_AND_TIMING_RESULTS_2026-10-07.md for class support,
local report paths and next gates.
