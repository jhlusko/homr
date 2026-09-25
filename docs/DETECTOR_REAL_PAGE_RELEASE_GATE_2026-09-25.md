# Detector real-page release gate — frozen 2026-09-25

This is the B3 criterion in `OurTextScores/docs/private/OTS_HOMR_RELEASE_SEQUENCE_2026-09-25.md`.
It is fixed before another detector training run. The five-class `DirectionText` scheme and
per-model class order are in `223f012`. The selected 2026-09-19 adaptation parent, not its
rejected fine-tune, is the training baseline. The seven-class released `e4` is the
`Dynamic` release baseline. Neither the 2026-09-24 seven-class retrain nor the rejected
2026-09-19 fine-tune is eligible to ship.

## Frozen data and scoring

`training/ocr/real_page_release_split_20260925.json` lists every score and support count.
The split is by **score**, shared by `lieder_boxes` and `lieder_lyrics`; no page from a
score can cross roles. All these scores are in the 09-19 fine-tune's `heldout_scores`,
none in its `train_scores`. All future training, including synthetic replay from the same
source scores, must exclude the listed selection and test scores. Selection is 145 OSSQ
pages (467 direction, 1,841 dynamic boxes), 12 Lieder direction pages (11 direction boxes),
and 20 Lieder lyric pages (374 lyric, 78 dynamic boxes). Test is 154 OSSQ pages (421
direction, 1,660 dynamic boxes), 14 Lieder direction pages (11 direction boxes), and 20
Lieder lyric pages (407 lyric, 79 dynamic boxes). The manifest records SHA-256 of its
source page-row files and the prior fine-tune audit so changes are detectable.

These pages were already evaluated in aggregate in September; this is a **prospective
split for the next checkpoint**, not a pristine external test. Freeze it now and do not
choose hyperparameters, thresholds, or data augmentation from test results. One final
test evaluation per chosen checkpoint. If it fails, report the failure and design a new
experiment without tuning to these test pages. A new untouched corpus would be needed
for a fully independent release estimate.

Score full pages with the same inference path and greedy one-to-one box matching used
for the parent. IoU 0.5 is only the threshold for saying a predicted box found a
reference box; **recall (`matched / ground_truth`) is the checkpoint metric**. Map
`Tempo`, `StaffText`, `Expression`, and `SystemText` references to
`DirectionText` for five-class models, with **re-matching after folding** for seven-class
diagnostics. Confirm checkpoint class metadata and channel count before scoring. Report
per-corpus, per-class `matched / ground_truth` recall and `matched / predicted` precision
lower bound, plus predicted-box counts and mean foreground fraction. The OCR-confirmed
annotations make precision a lower bound; neither precision nor patch IoU selects a model.
Do not pool OSSQ and Lieder, or hide zero-support classes in an aggregate.

## Checkpoint selection — selection role only

Evaluate the parent and every completed epoch on the selection scores. An epoch is
eligible only if OSSQ `DirectionText` recall improves over the parent by at least 3
percentage points, OSSQ `Dynamic` recall is no worse than 2 points below the parent,
and Lieder `DirectionText` loses at most one of the parent's matched boxes. Among
eligible epochs choose the highest OSSQ `DirectionText` recall; break ties by higher
OSSQ `Dynamic` recall, then earlier epoch. The parent remains the fallback. Predicted
counts and foreground fraction are reported for each epoch; investigate a large rise,
but never reward fewer detections by using precision or F1 to rank epochs. A synthetic
validation gate can veto a broken class, never override a real-page failure. Do not
change this selection rule after seeing any new checkpoint's test score.

## Release decision — test role, once

The selected checkpoint is eligible to replace the released non-lyric detector only if
all of these hold on the frozen test scores, comparing baselines on the **same pages**:

1. OSSQ `DirectionText` recall exceeds the 09-19 parent by at least 5 percentage points.
2. Lieder `DirectionText` matches at least as many boxes as the parent. With only 11 test
   boxes, even one lost match is a material regression; show raw counts.
3. OSSQ `Dynamic` recall is no more than 5 points below released seven-class `e4`, and
   no lower than the 09-19 parent. This prevents shipping a direction improvement that
   undoes the dominant existing detector class.
4. On OSSQ, the number of `DirectionText` predictions is no higher than the parent's
   and the number of `Dynamic` predictions is no higher than `e4`'s on the same pages.
   This is an explicit over-prediction guard, not a claim of measured precision. On
   Lieder direction pages, direction predictions must not exceed the parent's count.
5. The five-class checkpoint's history gate passes for all supported in-scope classes;
   `Fingering` is excluded. Real-page class metrics and model digest accompany any pin.

Failing any one condition means retain the existing release and record the failed
comparison. The Lieder lyric pages are a diagnostic for false direction predictions
and collateral `Lyrics`/`Dynamic` behavior; a lyric release still needs its separate
§5 decision and must not be inferred from this gate.

Before B4, run the parent and `e4` on both roles, archive per-page rows, exact model
digests and split-manifest digest, and verify the train-score exclusion. These baseline
runs fill in the numerical bars without changing this criterion. Keep `/workspace/venv`
on the GPU instance intact; run only pushed code from a clean checkout.

## Baseline fill-in — completed 2026-09-25, criterion unchanged

Both baselines ran from pushed `e1223f4` in persistent tmux. The watcher marked
completion. Reports, including each page's counts, are archived under
`homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/detector-b3-baselines/`.
The parent digest is `a03cd63177353360faf7c8a377e061d5130c3eeeab10775fd2c980a08096520c`;
released `e4` is `fcb8ca529fa355b0427112bda1c4edd95b2d6cc6c369be0c32f3313987a4ec82`.
The split-manifest digest is `43f922bc24179a2a182888df54587514db9da4ac0a806ad242c0e8e361fad72f`.

| role/corpus | parent direction matched/predicted/GT | parent dynamic matched/predicted/GT | `e4` dynamic matched/predicted/GT |
| --- | ---: | ---: | ---: |
| selection OSSQ | 94 / 512 / 467 | 367 / 3,262 / 1,841 | 1,169 / 10,023 / 1,841 |
| test OSSQ | 83 / 367 / 421 | 465 / 2,533 / 1,660 | 1,101 / 5,728 / 1,660 |
| selection Lieder direction | 8 / 38 / 11 | no reference boxes | no reference boxes |
| test Lieder direction | 11 / 87 / 11 | no reference boxes | no reference boxes |

Thus the selection floors are **109/467** OSSQ direction matches, **331/1,841**
OSSQ dynamic matches, and **7/11** Lieder direction matches. The test release floors
are **105/421** OSSQ direction matches, **1,018/1,660** OSSQ dynamic matches, and
**11/11** Lieder direction matches, with the prediction-count caps already stated
above: OSSQ direction ≤367, OSSQ dynamic ≤5,728, Lieder direction ≤87. These are
raw-count translations of the predeclared percentage rules, not newly chosen bars.
`e4` had zero matched direction boxes after folding on these pages, consistent with
its known direction failure. The parent's perfect 11/11 Lieder test recall makes that
part of the release bar demanding; it must not be relaxed after seeing a candidate.

## Split-system amendment — 2026-09-25 (decision 13)

B4 calibrated to this criterion (`results/b4_calibrated_selection.md`): epoch 4 wins
OSSQ direction text at every epoch tried (+15.4pp over the parent, at the parent's box
count) but cannot clear the OSSQ `Dynamic` floor at any epoch, and Dynamic gets worse
with more training rather than better. The owner resolved this as decision 13: the
non-lyric detector splits into two models. B4's calibrated epoch 4 supplies
`DirectionText` only; a separate `dynamics` detector supplies `Dynamic`, starting as the
released `e4` **unchanged** - so there is nothing new to test for `Dynamic` under this
gate. This section amends, and does not replace, everything above: the split, scoring
method, and selection procedure are unchanged except as stated here.

**Selection (unchanged split/scores, amended rule).** The `Dynamic` selection floor
(originally the 09-19 parent's matched count minus 2pp, `training/ocr/select_direction_epoch.py`)
is dropped. An epoch is no longer disqualified, or preferred over another, by a class it
will never supply in production. The `Dynamic` and Lieder-direction eligibility
conditions, and the OSSQ `DirectionText` eligibility condition, are otherwise unchanged.
Ranking among eligible epochs is by OSSQ `DirectionText` matched count alone (previously
a tie-break on `Dynamic` came second; with `Dynamic` no longer gating anything, dropping
it from ranking too avoids preferring an epoch for a channel this checkpoint will not
ship). Implemented as `select_direction_epoch.select(..., direction_only=True)`
(`--direction-only` on the CLI).

**Release decision (unchanged split/scores, amended rule).** Condition 3 (OSSQ `Dynamic`
recall within 5pp of `e4`) and the `Dynamic` half of condition 4 (prediction-count cap)
are dropped from the test gate. The candidate is tested on direction text only:

1. OSSQ `DirectionText` recall exceeds the 09-19 parent by at least 5 percentage points
   (frozen numeric floor: **≥ 105/421** matched, from the baseline fill-in above).
2. Lieder `DirectionText` matches at least as many boxes as the parent (frozen numeric
   floor: **11/11**).
3. On OSSQ, `DirectionText` predictions do not exceed the parent's count on the same
   pages (**≤ 367**). On Lieder direction pages, predictions do not exceed the parent's
   count (**≤ 87**).
4. The five-class checkpoint's synthetic history gate still passes for all in-scope
   classes, **including `Dynamic`** - this is a sanity check that the head did not break
   during training, independent of whether production uses its output. `Fingering` stays
   excluded, as before.

Implemented as `decide_direction_release.decide(..., direction_only=True)`
(`--direction-only` on the CLI); the returned decision object carries `"direction_only":
true` so a report cannot be mistaken for a combined-gate result. `Dynamic`'s release
path is now `e4`'s existing, already-shipped behavior - not re-tested here, and not
re-opened by this amendment.

Everything else in this document (the frozen split, scoring method, precision-as-lower-
bound stance, one-test-read rule, and the requirement to freeze this before any test
page is read) is unchanged.
