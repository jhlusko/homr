# C1 semantic baseline and completed timing probe — 2026-10-07

## Timing probe completed

All 4,237 frozen-core feature crops and six head epochs completed. Live curves:
http://localhost:8765/timing-heads-20261006/ . These are teacher-forced results;
heads remain marked `probe-trained` and rejected by the inference loader.

| Lieder head, epoch 6 | Accuracy | Constant training-majority baseline | Macro class recall |
|---|---:|---:|---:|
| Elapsed duration | 97.60% | 45.91% | 83.05% |
| Written duration | 99.19% | 47.40% | 99.03% |
| Tuplet ratio | 98.88% | 97.15% | 75.00% |
| Full-measure-rest flag | 99.56% | 90.84% | 99.76% |

Lieder duplet `2:1` recall is87/135=64.44%; triplet `3:2` recall100/165=60.61%.
All63 positive full-measure-rest flags are correct, with three false positives
among625 negative rest flags. The98.88% overall ratio score is dominated by ordinary
`1:1` notes. Written-duration and full-rest validation losses reached their minima
at epoch4; elapsed and ratio losses continued improving through epoch6. Do not
promote solely on headline accuracy or concatenate per-head best epochs without
an independent qualification.

OSSQ elapsed/written/ratio accuracy is100%, but only121 events from two validation
scores, including six triplets. Its19 rest flags are all negative, so this says
nothing about OSSQ full-measure-rest-positive recognition. Training is one OSSQ
score. This probe shows that the frozen core contains learnable timing information;
it does not demonstrate a free-decoding improvement over the accepted stack.

Final report:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-probe-20261006/teacher-forced-summary-v2/report.md`.
Machine-readable class support, trajectories and receipts: adjacent `report.json`.
Epoch bundles, exact optimizer checkpoint and curves: adjacent `../full/`.

### Sparse-head failure and exact resumption

The first attempt stopped before update17. Both sources had zero full-rest labels
in that batch. The full-rest projection legitimately had no gradient; the other
three heads had finite gradients. The prior assertion incorrectly required a
gradient for every head on every batch.

The fixed check requires gradients for supervised heads and rejects any nonfinite
gradient. Unsupervised parameters remain untouched by AdamW, including weight decay.
A regression test checks this and verifies no optimizer state is created for a
never-supervised head. Exact-resume and cache-tamper tests still pass.

The first run is archived in `timing-probe-20261006/full-first-attempt/`; the exact
old trainer is in `control/train_cached_timing_heads-first-attempt.py`. An explicit
migration validated its source hash and retained weights, AdamW state, epoch0,
next_batch16 and accumulated losses; only trainer code identity changed. Receipt:
`timing-probe-20261006/full/resume-migration.json` under the artifact root above.
The full cache was reused and all six epochs then finished successfully. Do not
rerun extraction into that same cache with the changed trainer: its extraction
identity deliberately still binds the original source. Training reads the frozen
cache normally; its unchanged identity is recorded in the migrated run.

## Fixed lyric-selection baseline

http://localhost:8765/c1-semantic-baseline-20261007/ now serves the completed
side-by-side results. All84 crop/context assets and report.json were HTTP verified.

There is no trained semantic-role classifier in the current implementation. This
pass measures the existing pinned-e2 region/OCR gate, not a newly trained classifier.
Full scans and OCR are its only inputs; the42 separately inspected references enter
only evaluation. Fixed detector area200 and10% OCR-region union overlap are unchanged.
Reference/OCR matching uses at least50% coverage of the inspected box, after inference.
No threshold was tuned against these controls.

Results separate lyric candidates, gate abstentions and OCR misses. Unsupported
rows remain `Unknown`, never retrospectively relabeled as correct directions.
A selected direction is a false lyric candidate, not yet a measured false note
attachment. Regions can share a whole OCR line; the report explicitly lists those
dependencies. Positive-only references cannot support whole-page precision.

Run:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-baseline-20261007/`.
Persistent tmux sessions: `c1-semantic-baseline` and `c1-semantic-baseline-monitor`.
The watcher validated frozen input/code hashes, scored once, and published the
local report. Both detector and scoring exited0; all eight pages finished05:54UTC.

| Inspected role | Regions | Selected as lyrics | Gate abstentions | OCR misses |
|---|---:|---:|---:|---:|
| Lyrics | 26 | 26 | 0 | 0 |
| DirectionText | 15 | 2 | 12 | 1 |
| Credit / other | 1 | 0 | 1 | 0 |

The two false lyric candidates are IMSLP11784 `Etwas geschwind.` (18.87% detector
coverage of its OCR rectangle) and IMSLP435483 `Assez lent` (83.58%). No reference
regions share an OCR row in this packet. Selection recall is26/26 for this small
inspected sample, not corpus-wide recall; direction false selection is2/15.
No inference threshold was altered. These are region candidates, not final
attachments: this pass does not test whether later staff/voice routing would reject
them. The previously confirmed Bizet in-row `a tempo.` attachment error remains
independent evidence that geometry alone does not identify semantics.

Final machine-readable report:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-baseline-20261007/scored/report.json`.

## Priorities

1. Read the semantic baseline, keeping direction/credit false lyric decisions and
   lyric misses separate. Build a dedicated semantic discriminator if this confirms
   the gap; it needs a separate training/calibration pool, explicit abstention and
   independent checks. Do not train and qualify on these same42 reviewed regions.
2. Freeze the timing probe results and run a research-only free-decoding benchmark:
   use image-only predicted prefixes, score matched and unmatched events separately,
   preserve pitch/rhythm/accidental recognition and measure actual timing changes.
   Broaden OSSQ validation, including positive full-measure rests and tuplets, before
   drawing release conclusions. The existing runtime has no timing-head integration;
   this benchmark must not silently activate unqualified heads in production.
3. Automatic lyric fusion and incapable e4 direction routing remain disabled.
   No publishing, model pin, deployment or production defaults changed.

Nine focused tests passed, including sparse masking and optimizer/code migration.
Black/Ruff checks cover the changed source. Completed timing epoch6 still fails
inference activation, as intended. Both reviews and all84 semantic comparison
images load successfully. No jobs from these two experiments remain running.
