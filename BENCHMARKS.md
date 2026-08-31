# Checkpoint results ledger

## SMB (PRAIG Sheet Music Benchmark)

SMB is a 748 MB, gated full-page benchmark.  Its target is each page's own
``**kern`` transcription—not a concatenation of the auxiliary region labels.
On the GPU instance, the project venv needs the isolated music21 install:

```bash
HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' /path/to/.env | head -n1)" \
PYTHONPATH=/workspace/b0/m21libs \
  /workspace/b0/homr/.venv/bin/python -m validation.smb \
  --tool homr --workers 1 --kern-parser music21 --xml-parser native \
  --output /workspace/b0/lieder-rebuild/smb_homr.db
```

Use the same environment and arguments when recomputing scores with `--update`.
The dataset requires an account that has accepted PRAIG/SMB's access conditions; never
commit its token.  Record the model graph/checkpoint and command alongside each result.

### Upstream-426 SMB baseline — qualified result (2026-08-31)

The first end-to-end `homr` run completed all **685/685** gated SMB test pages on the
instance checkout `997b70a`. The command did not set `HOMR_MODEL_NAME`; the production
CLI therefore loaded its configured default, the **upstream checkpoint 426**
(`pytorch_model_426-b6fd…`), not Arm A, `rareNum`, or any other fine-tuned export. The
reported baseline is the **651 pages with a non-empty reference token sequence after the
configured `music21` `**kern` parse**:

| pages | overall NED | rhythm | pitch | lift | articulation | slur |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 651 | **19.97%** mean / **18.88%** median | 9.52% | 7.90% | 7.72% | 8.37% | 12.43% |

This is a full-page, end-to-end **upstream baseline**: segmentation and all
decoding/repair stages in the current code are enabled through the normal `homr` CLI,
but the core model weights are 426. It establishes an external baseline only. No
fine-tuned checkpoint has yet been scored on SMB, so this table cannot support a claim
that Arm A/`rareNum` improves the external benchmark.

**Why 34 pages are omitted.** All 685 SMB rows have non-empty page-level `**kern`
text; 34 valid rows (`2, 25, 130, 142, 170, 187, 206, 212, 247, 260, 317, 318, 353,
362, 363, 376, 377, 384, 385, 406, 419, 420, 428, 437, 448, 499, 535, 550, 566, 567,
573, 602, 621, 678`) produced zero tokens from the chosen `music21` reference parser.
They are evaluator-unscoreable under this protocol, not blank ground truth and not
HOMR failures. Including them mechanically assigns 100% NED and changes the raw mean
to 23.94%, which is not a valid quality estimate. The native parser can parse 33 of
these 34 pages; do not merge native and music21 per-page scores in a single headline
metric. A future parser-fallback protocol must first be applied uniformly and then
re-run for all comparison checkpoints.

Every paired comparison run during the Lieder corpus work, in one place. Companion to
`CORPUS_CHANGELOG.md`, which traces what changed in the corpus; this records what those
changes scored.

## Read this first: the noise floor is 4.06pp

Two runs of the **same corpus**, same recipe, differing only in the trainer seed:

| | final `eval_accuracy` | final `train_loss` | OSSQ |
|---|---|---|---|
| 452, seed 42 | 0.99684 | 1.0868 | **93.12** |
| 454, seed 1234 | 0.99669 | 1.0919 | **89.06** |

Both runs are healthy and indistinguishable in-corpus - 0.00015 apart on
`eval_accuracy` - and **4.06pp apart on OSSQ**.

**Every corpus effect in this document is smaller than that.** Removing the
model-derived half (+0.69), restoring the overfull pairs (−1.31), 447 over 449 (−1.60),
448 eroding 447 (−1.23), even 447 over the baseline (+2.32): all inside the range two
identical runs produce by chance. The monotonic "smaller corpus scores better" pattern
- 3,548 > 6,989 > 7,369 pairs - is three points, each step under 1.3pp, and is
consistent with three draws from a distribution this wide.

**The confidence intervals below do not license those claims, and saying they did was
an error.** They bootstrap over *staves*, which is the right uncertainty for "would
this difference hold on other staves" and the wrong one for "was this difference caused
by the corpus". The second question needs replication over training runs, which nothing
here had until 454.

What survives the floor: 450's −26.82pp collapse and its epoch bisect; 450's +5.56pp on
PDMX; and the qualitative pattern that every checkpoint wins on its own training
distribution, which spans four distributions rather than resting on a margin.

**Protocol from here.** A corpus comparison needs several seeds per condition and a
comparison of means. A single run is a draw, not a measurement.

## Alignment-provenance correction (2026-08-29)

The historical v6 clean manifest accepted a multi-scan measure-count group when only
its **total** matched the reference total, then fabricated individual system boundaries
by cumulatively splitting that total. Human review found six explicit shifted/truncated
labels of this form; every one had already disagreed with reverse fingerprinting. The
coverage-safe v4 alignment now quarantines these as `boundary_ambiguous` rather than
emitting them as `aligned`: 2,371 model-free aligned systems remain and 480 are
quarantined. Its rebuilt manifest has 4,343 pairs (versus v6's 4,543), and the
reverse-aware consensus manifest has 3,922 evaluation-admissible pairs. A separate
333-pair reverse-derived manifest is explicitly **training-only**, never evaluation.

No table below has been retroactively changed: OSSQ and PDMX scores are still correct
historical observations because neither uses these Lieder labels. The old Lieder column,
however, is not evidence that a checkpoint handles this corrected corpus: it measures
project-built labels, and v6-trained checkpoints were trained on the raw manifest that
contained this now-quarantined class. Do not claim a performance effect for v4 until a
fresh, replicated training comparison uses its manifest and reports OSSQ as the primary
independent outcome.

### First v4-boundary-safe arm (in progress)

The first clean replication is running on the instance as
`current_training_v4_boundary_safe_s42`: **3,622** model-free consensus pairs from
202 score-disjoint training scores, plus **1,300** PDMX `index_train.txt` replay pairs,
for 12 epochs from the pinned checkpoint (seed 42). Its selection validation is a new,
disjoint **300-pair / 13-score** v4 consensus split. It includes **none** of the 333
reverse-derived boundary-ambiguous pairs; those remain pending human review and are not
eligible for evaluation. Once training selects a checkpoint, score it on OSSQ,
`index_valid.txt` PDMX, and this v4 Lieder holdout, then add the resulting row below as
a single preliminary draw—not a corpus-effect claim until matched seeds replicate it.

## What kind of errors, not how many

`error_taxonomy.py` partitions each staff by the *kind* of its worst error, because
token accuracy scores a dropped barline and a wrong pitch identically. On the 148 dense
OSSQ staves:

| checkpoint | exact | pitch-only | rhythm | **length** | **structural** |
|---|---|---|---|---|---|
| 426 base | 50.7% | 2.7% | 29.7% | 13.5% | 3.4% |
| 447 | 60.1% | 1.4% | 25.0% | 10.1% | 3.4% |
| **456 (v6)** | **60.8%** | 2.0% | 26.4% | **6.8%** | 4.1% |
| 459 (single-staff only) | 52.7% | 2.7% | 28.4% | 12.8% | 3.4% |

* **Fine-tuning buys exactly-correct staves**: 50.7% to 60.8%, a fifth more transcriptions
  needing no correction. That is the figure a user would notice.
* **The gain is concentrated in length errors** - notes invented or dropped inside a
  correct bar grid. 456 more than halves them, 20 staves to 10. Rhythm and pitch hardly
  move.
* **Structural errors are untouched by everything.** 5 staves at baseline and 5-6 after
  every fine-tune. Losing the bar grid is what makes a transcription unusable rather
  than merely wrong, and no corpus change made here has moved it. That points at
  alignment or architecture, not at data quantity or label quality.
* **459 confirms the grand-staff result independently**: +3 exact staves against 456's
  +15, from the same corpus minus its grand staves.

## Four corpus fixes, all correct, none consequential

Each was a real defect, established from the data and verified in the built pairs. None
moved the model measurably.

| fix | size | effect |
|---|---|---|
| overfull rule guarded to single staves | 371 pairs restored | neutral (dense mean 94.04 vs 94.19) |
| stale numerator tokens dropped | 41 pairs | folded into the above, neutral |
| unison duplicate notes removed | **490 tokens, 0.31%** | **null, and the targeted metric moved against it** |
| overfull pairs restored (arm B) | 380 pairs | **worse** - OSSQ 92.43 -> 91.12 |

The duplicate result is the sharpest, because it had a mechanism and a prediction. The
corpus writes two note tokens for one notehead where parts are in unison - verified
against double-stemmed noteheads in the crops - which is a direct instruction to
over-emit, and over-emission inside a correct bar grid is the dominant failure. Removing
them should have cut length errors.

| dense cut | exact | length |
|---|---|---|
| v6, two seeds | 91, 90 | **10, 10** |
| dedup, two seeds | 88, 92 | **11, 12** |

Length errors rose in both draws. Aggregate 92.15 -> 92.30, inside noise.

**The pattern is the conclusion.** This corpus's label defects are real, individually
under 1% of tokens, and individually too small to move a model measured at 0.23-0.69pp of
dense-cut noise. Hunting further sub-1% defects is not where the remaining accuracy is;
the one lever found with a large enough mechanism is tuplet SUPPLY, where the corpus
carries 1.78% tuplet notes against OSSQ's 6.58%.

## PDMX: nothing has improved since checkpoint 448

Scored on the PDMX dense cut (2,492 of 3,349 staves, 45+ symbols), baselined on 448 - the
last checkpoint anything had scored there before this session:

| checkpoint | PDMX dense | vs 448 |
|---|---|---|
| 426 base | 83.91 | −1.78 *(significant)* |
| 447 | 85.79 | +0.11 *(ns)* |
| 448 | 85.68 | — |
| **456 (v6, our best on OSSQ)** | **85.52** | **−0.17 *(ns)*** |
| 459 (single-staff) | 82.36 | −3.33 *(significant)* |

**456 is statistically indistinguishable from 448 on PDMX, and nominally below it.** Eight
checkpoints of corpus work produced no measurable movement on 3,349 staves of different
repertoire. Calling 456 "the best checkpoint" is an OSSQ-only claim and should be stated
that way.

456 is also **worse than 448 on PDMX structural errors** - 135 staves against 121 - despite
carrying 7 more exact. Whatever the OSSQ gain is, it did not bring the bar grid with it.

## The grand-staff effect holds on every corpus

Removing grand staves from the training corpus, matched seed where possible:

| corpus (dense cut) | grand staves in | grand staves out | delta |
|---|---|---|---|
| OSSQ (148) | 94.37 / 93.68 | 92.46 / 92.32 | **−1.9 to −2.1pp** |
| PDMX (2,492) | 85.52 | 82.36 | **−3.16pp** *(CI −4.34 to −2.01)* |
| Lieder (122) | 90.52 | 75.35 / 74.39 | −15.2 to −16.1pp |

Larger on PDMX than on OSSQ, and largest on Lieder - which is piano/voice repertoire and
so the most grand-staff-dependent. The Lieder row pairs different checkpoints as well as
different corpora and rests on 14 scores, so read it as direction, not measurement.

OSSQ carries two training seeds per arm, which pins the noise directly: within-corpus
dense spread 0.69pp (v6) and 0.14pp (single-staff), against a smallest cross-corpus gap of
1.22pp, with both single-staff seeds below both v6 seeds. That seed replication is what
licenses treating the single-seed PDMX gap as real.

**The aggregate would have said the opposite.** v6's own two seeds are 92.65 and 88.28 - a
4.37pp within-corpus spread straddling both single-staff runs - and `compare_checkpoints`
calls that same-corpus pair "significant". The dense cut is what makes the finding visible
at all.

## The end-on-a-divider constraint: structural fix, no correct transcriptions

Checkpoint 456, same index, decode-only difference:

| | structural | length | exact | overall |
|---|---|---|---|---|
| off | 63 | 67 | 389 | 92.65 |
| on | **41** | 89 | **389** | 92.35 |

It repairs 22 of 23 invariant violations and never damages a correct grid - but the
structural-to-length movement is exactly 1:1 and **not one staff becomes exact**. Producing
those 22 barlines took 90 extra tokens, 68 of them spurious, including time and key
signatures at the end of a system where they are nonsense. Accuracy fell 0.31pp against a
predicted 0.03pp: the simulation appended exactly one divider, where the real model needs
~3.9 tokens to reach one.

Left **off by default**. It converts a broken grid into a correct grid plus noise, and
whether that is a win depends on what consumes the output.

## Use the dense-staff cut, not the aggregate

Six checkpoints, five of them seeded replicates, on 792 staves and on the 148 holding
45+ symbols:

| checkpoint | aggregate | **dense 45+** | dense vs 426 |
|---|---|---|---|
| 426 base | 91.70 | 91.32 | — |
| 447 | **94.03** | 93.97 | +2.65 |
| 452 (v8 clean, s42) | 93.12 | 94.08 | +2.76 |
| 454 (v8 clean, s1234) | **89.06** | **94.31** | +2.99 |
| 455 (v6, s42) | 91.64 | 94.07 | +2.75 |
| **456 (v6, s7)** | 92.65 | **94.37** | **+3.05** |
| 457 (v6, s99) | **88.28** | 93.68 | +2.36 |

| corpus | aggregate mean | spread | dense mean | spread |
|---|---|---|---|---|
| v6, 3 seeds | 90.86 | **4.37** | 94.04 | **0.69** |
| v8 clean, 2 seeds | 91.09 | **4.06** | 94.19 | **0.23** |

**The dense cut is six to nineteen times more precise on 19% of the staves.** The
aggregate has been measuring seed noise; the dense cut measures the model.

Three consequences:

1. **Every fine-tune is a real improvement** - +2.4 to +3.1pp over baseline, six for
   six, against spreads under 0.7. 454 scores 89.06 on the aggregate and **94.31**
   dense, second best of all; the run that looked like a disaster is one of the better
   models.
2. **447's lead was a fortunate draw.** On the reliable metric it is 93.97, mid-pack,
   below 452, 454 and 456. Much of this document's earlier narrative exists to explain
   a gap that was noise.
3. **The corpus fixes are neutral, and that is now a measurement.** v6 dense mean 94.04
   against v8 clean's 94.19, a 0.15 difference against spreads of 0.23 to 0.69. The
   fixes are correct and verified in the data; they do not move the model.

**Best checkpoint by the metric that can resolve one: 456**, v6 corpus at seed 7, 94.37.

## The signal the aggregate was hiding

Noise on OSSQ is strongly inhomogeneous, and stratifying by staff density separates a
real effect from the variance that was burying it. Same-corpus seed pairs give the
yardstick in each bucket.

| bucket | staves | 447 | 455 (v6 s42) | 456 (v6 s7) | **same-corpus seed noise** |
|---|---|---|---|---|---|
| < 15 symbols | 23 | −1.78 | −0.02 | −1.79 | **1.77** |
| 30+ symbols | 446 | +2.41 | +0.60 | +1.22 | **0.62** |
| **45+ symbols** | 148 | **+2.65** | **+2.75** | **+3.05** | **0.29** |
| **60+ symbols** | 45 | **+3.30** | **+3.43** | **+3.62** | **0.19** |

All figures against the 426 baseline.

**On dense staves the fine-tuning gain is real and replicated** - three independently
trained checkpoints across two corpora, all gaining 2.7 to 3.6pp, against a noise
yardstick an order of magnitude smaller. On sparse staves nothing is measurable: two
runs of the *same* corpus differ by 1.77pp there, which is larger than any effect.

The reason is mechanical. A sparse staff holds few tokens, so a single error moves its
accuracy several points; a dense staff averages over many. The 792-staff headline mixes
both and the sparse tail contributes variance out of all proportion to its token count.

**Consequence.** The aggregate is the wrong statistic for comparing checkpoints, and
most of this document's earlier conclusions were drawn from it. A dense-staff cut, or
any weighting that reflects tokens rather than staves, is both more stable and more
informative. **Always include a same-corpus seed pair in a stratified comparison** - it
is the only thing that says whether a bucket can resolve the effect being claimed.

## How these numbers are produced

`training/transformer/base_predictions.py` scores a checkpoint to a `.jsonl` of per-staff
reference and prediction vectors; `training/transformer/compare_checkpoints.py` compares
them. Three properties matter for reading the tables:

- **Paired.** Checkpoints are compared on identical staves, as per-staff differences,
  with a 2,000-resample bootstrap over staves (seed 0). Two aggregate percentages hide
  how a difference is distributed - a 1pp gap is a different fact when it is 8 staves
  collapsing than when it is every staff drifting - so every row carries a
  better/worse/tied split.
- **Only staves scored by every run** are compared, so a checkpoint that failed on some
  crop is never credited with an easier subset.
- **Padded.** Reference and prediction are padded to a common width, so a free-running
  model that diverges in length is penalised rather than scored on the overlap it
  happened to get right.

`overall` pools every branch's positions. Per-branch figures live in the
`cmp_*.json` reports.

## The benchmarks

| name | staves | what it is | exposure to this work |
|---|---|---|---|
| **OSSQ** | 792 | real scanned string-quartet pages | **none** - never trained on, never relabelled |
| **PDMX** | 3,349 | public corpus, synthetic renders | split predates this work; ~3.6% replay contamination, see below |
| **Lieder** | 292 | our own held-out Lieder split | labels built by this project |

Only OSSQ is fully independent. Lieder measures our labels as much as the model, and
every checkpoint that improved on it has done worse on OSSQ.

## Master table

Overall token accuracy. Dashes are runs not scored on that benchmark.

| checkpoint | trained on | OSSQ | PDMX | Lieder |
|---|---|---|---|---|
| **426** | five original corpora (all synthetic) | 91.70 | 84.71 | 83.25 |
| **447** | v6 Lieder corpus + pdmx replay | **94.03** | 86.34 | 91.81 |
| **448** | v7 corpus + replay | 92.80 | 86.34 | 92.69 |
| **449** | v8 corpus + replay (arbitration ablation) | 92.43 | — | **94.53** |
| **450** | PDMX only, 12 epochs | 64.89 | **90.27** | 41.06 |
| **451** | v8 corpus, 2 epochs | 91.83 | — | 87.86 |
| **452** | v8 clean only, model-derived removed (3,548) | 93.12 | — | 91.67 |
| **453** | v8 + 380 overfull pairs restored (7,369) | 91.12 | — | 93.94 |

**Every checkpoint is best on its own training distribution.** 447 is the only one that
improved on all three.

## Paired comparisons

### OSSQ, against the 426 baseline

| comparison | delta | 95% CI | better / worse / tied |
|---|---|---|---|
| 447 vs 426 | **+2.32** | +1.36 to +3.25 | 252 / 103 / 437 |
| 448 vs 426 | +1.10 | +0.08 to +2.10 | 236 / 130 / 426 |
| 449 vs 426 | +0.73 *(ns)* | −0.53 to +1.89 | 243 / 127 / 422 |
| 451 vs 426 | +0.13 *(ns)* | −0.43 to +0.71 | 137 / 121 / 534 |
| 450 vs 426 | **−26.82** | −29.17 to −24.44 | 90 / 480 / 222 |

### OSSQ, against 447 - the erosion

| comparison | delta | 95% CI | better / worse / tied |
|---|---|---|---|
| 448 vs 447 | **−1.23** | −1.89 to −0.52 | 67 / 116 / 609 |
| 449 vs 447 | **−1.60** | −2.54 to −0.71 | 89 / 118 / 585 |

### Lieder (our own labels), against 447

| comparison | delta | 95% CI | better / worse / tied |
|---|---|---|---|
| 448 vs 447 | +0.88 *(ns)* | −0.61 to +2.68 | 16 / 24 / 252 |
| 449 vs 447 | **+2.72** | +0.80 to +5.25 | 26 / 19 / 247 |
| 451 vs 447 | **−3.95** | −5.77 to −2.20 | 14 / 81 / 197 |

Note 448: our own metric reported +0.88pp, and a paired test says that is noise **with
more staves worse than better**. The same checkpoint lost 1.23pp on OSSQ. An earlier
write-up reported roughly +3% for exactly this shape of result.

### The corpus bisect

Both arms vary the corpus and hold the recipe fixed - warm start, ~8k samples, 12
epochs, replay 1300, same validation index.

| comparison | OSSQ | Lieder |
|---|---|---|
| 452 vs 449 - remove the model-derived half | **+0.69** *(ns, CI −0.20 to +1.67)* | **−2.86** *(sig, CI −5.41 to −0.76)* |
| 453 vs 449 - restore the overfull pairs | **−1.31** | +... *(see table above)* |
| 452 vs 426 | +1.42 *(sig, CI +0.53 to +2.33)* | — |
| 453 vs 426 | −0.58 *(ns, CI −4.70 to +2.10)* | — |

Two independent additions to the corpus, the same direction each time: **more pairs,
worse on the independent corpus, better on ours.**

| pairs trained on | OSSQ |
|---|---|
| 3,548 (452, clean only) | **93.12** |
| 6,989 (449, + model-derived) | 92.43 |
| 7,369 (453, + overfull) | 91.12 |

453 falls below the 426 baseline of 91.70.

### PDMX, against 426

| comparison | delta | 95% CI | better / worse / tied |
|---|---|---|---|
| 447 vs 426 | +1.63 | +0.80 to +2.47 | 780 / 364 / 2205 |
| 448 vs 426 | +1.63 | +0.75 to +2.56 | 752 / 388 / 2209 |
| 450 vs 426 | **+5.56** | +4.68 to +6.51 | 953 / 269 / 2127 |

**Caveat.** Replay for these runs drew from an index containing the PDMX validation
rows - roughly 3.6% of the benchmark. It inflates every checkpoint equally, so
comparisons hold and absolute figures do not. Fixed in `1c51174`; no clean absolute
PDMX number exists until a checkpoint trains without it.

## Where 450 fell apart: a bisect over training time

450 warm started from 426 and fine-tuned 12 epochs on PDMX alone. Each epoch checkpoint
was scored on OSSQ.

| | OSSQ | vs start | in-domain `eval_accuracy` |
|---|---|---|---|
| 426, the starting point | 91.70 | — | — |
| after epoch 1 | 85.64 | −6.06 | 0.99009 |
| after epoch 2 | 73.53 | **−18.17** | 0.99073 |
| after epoch 4 | 71.26 | −20.44 | 0.99269 |
| after epoch 8 | 65.53 | −26.17 | 0.99334 |
| after epoch 12 (=450) | 64.89 | −26.82 | 0.99337 |

Two epochs did 68% of the damage. Every delta is significant.

**Epochs 2-12 bought +0.33pp in-domain and cost −20.75pp out-of-domain** - about 60:1
against us, run for eleven epochs while loss stayed flat, the learning rate decayed
normally, and in-domain accuracy rose monotonically. Nothing malfunctioned.

`metric_for_best_model="eval_accuracy"` on the training corpus's own validation rises
throughout, so early stopping selects epoch 11 or 12 - the worst checkpoint by the
measure that matters.

## In-corpus accuracy, and why it is not usable as a verdict

| checkpoint | in-corpus `eval_accuracy` | OSSQ |
|---|---|---|
| 448 | 0.99718 | 92.80 |
| 449 | 0.99726 | 92.43 |
| 451 | 0.99549 | 91.83 |
| 450 | 0.99337 *(on PDMX)* | 64.89 |

448 and 449 differ by 0.00008 in-corpus and by 0.37pp on OSSQ. The metric is
teacher-forced and computed against labels this project built; it has been directionally
wrong or silent at every decision point in this work.

## Reproducing any row

```
python -m training.transformer.base_predictions \
  --index <benchmark index> --checkpoint <path> --out scored.jsonl

python -m training.transformer.compare_checkpoints \
  --name OSSQ --run 447=general_mid.jsonl --run 448=general_new.jsonl
```

Scored `.jsonl` files, `cmp_*.json` reports and the checkpoints are in
`homr-artifacts/instance-2026-08-28/`.

## v4 boundary-safe, replicated (2026-08-29)

Two matched seeds against the pinned base checkpoint (426), numerator-neutral - see
RUNLOG IV.15 for why a raw comparison across this vocabulary change is meaningless in
both directions.

| benchmark | base426 | v4 s42 | v4 s7 | delta (s42 / s7) |
| --- | --- | --- | --- | --- |
| OSSQ | 91.49 | 93.21 | 93.19 | +1.71 / +1.70 |
| PDMX held-out | 83.97 | 87.46 | 87.30 | +3.48 / +3.33 |
| Lieder v4 holdout | 87.23 | 95.38 | 95.44 | +8.15 / +8.21 |

All six significant; seed spread at most 0.15pp. OSSQ is the independent read, PDMX the
held-out general corpus, and the Lieder holdout is score-disjoint but shares the v4
consensus lineage with training - it is the least independent of the three and should not
carry a corpus claim on its own.

**These OSSQ numbers use the regenerated `phase7num/` references.** The older
`phase7fix/` set contains no `timeSignatureBeats_*` at all and understates any checkpoint
that states a metre numerator; it is not comparable with the rows above.


## Vocabulary-controlled attribution (2026-08-29)

The rows above compare against checkpoint 426, which predates the numerator *and* the
naturals; only the numerator is neutralised. Against `nat42` - a naturals-era checkpoint
on the older Lieder corpus, so vocabulary is roughly held fixed and the corpus varies:

| benchmark | vs 426 (s42/s7) | vs nat42 (s42/s7) |
| --- | --- | --- |
| OSSQ | +1.71 / +1.70 | +2.55 / +2.53 (significant) |
| PDMX | +3.48 / +3.33 | +1.54 / +1.39 (significant) |
| Lieder v4 | +8.15 / +8.21 | +0.51 / +0.56 (**not significant**) |

Read the second column, not the first, for any claim about the corpus. The Lieder gain is
a vocabulary gap once controlled; the OSSQ and PDMX gains are real and replicate.

Both baseline seeds are scored, and the spread differs sharply by benchmark:

| benchmark | baseline spread | v4 vs nat42 | v4 vs nat7 |
| --- | --- | --- | --- |
| PDMX | -0.05pp | +1.54 / +1.39 | +1.59 / +1.44 |
| OSSQ | +1.46pp | +2.55 / +2.53 | +1.09 / +1.08 |
| Lieder v4 | +0.70pp | +0.51 / +0.56 (ns) | -0.20 / -0.14 (ns) |

Quote **PDMX +1.4 to +1.6pp** (stable baseline, all four pairings significant) and **OSSQ
+1.1 to +2.6pp** (baseline noisy, positive everywhere). Quote **no effect on the Lieder
holdout**: `nat7` outscores both v4 seeds, so v4 straddles the baseline seeds there. See
RUNLOG IV.17.


## scans_v4: OSSQ gain, general cost (2026-08-29)

Numerator-neutral, against `v4_s42`:

| benchmark | v4_s42 | scans_v4 | delta |
| --- | --- | --- | --- |
| OSSQ | 93.21 | **95.14** | **+1.93pp** (significant) |
| PDMX | 87.46 | 86.98 | -0.48pp (not significant) |
| Lieder v4 | 95.38 | 94.40 | -0.98pp (not significant, 85 staves worse / 22 better) |

Do not quote `scans_v4` as "the best model" without a domain. It is the best on scanned
OSSQ by a wide margin and slightly behind on the other two. Four variables differ between
these runs (OSSQ added, replay composition, replay fraction, epochs), so no single one is
attributed. See RUNLOG IV.20.

## scans_v4 attribution arms (2026-08-30)

Two deliberately different continuations separate the two large changes bundled into
`scans_v4`. Arm A keeps the scanned OSSQ and v4 Lieder mixture but uses the PDMX-only
replay recipe. Arm B removes scanned OSSQ entirely and uses PDMX + grandstaff replay with
v4 Lieder. Both are seed 42, six epochs from the pinned starting checkpoint, and every
comparison below drops `timeSignatureBeats_` on both sides.

| benchmark | scans_v4 | Arm A | Arm B |
| --- | ---: | ---: | ---: |
| OSSQ | **95.14** | 95.05, -0.09pp (CI -0.28 to +0.04, ns) | 91.26, **-3.89pp** (CI -5.12 to -2.79) |
| PDMX held-out | 86.98 | **88.08**, +1.10pp (CI +0.30 to +1.90) | **88.02**, +1.04pp (CI +0.33 to +1.80) |
| Lieder v4 holdout | 94.40 | 94.54, +0.13pp (CI -1.03 to +1.37, ns) | 94.59, +0.19pp (CI -1.35 to +1.73, ns) |

The PDMX gains and Arm B OSSQ loss are significant paired bootstrap intervals; the other
two deltas are not. This is still one seed per condition, so it establishes direction and
mechanism rather than a seed-replicated corpus estimate.

**What it separates.** Arm A recovers the PDMX cost while leaving OSSQ unchanged: replay
composition is a material contributor to the `scans_v4` general-domain trade-off. Arm B
also recovers PDMX and holds Lieder, but loses 3.89pp on OSSQ: the real scanned OSSQ crops
are what supply `scans_v4`'s OSSQ gain. The useful production candidate before further
work is therefore Arm A, not Arm B.

**Rare metres are not measured by this table.** The fair cross-vocabulary comparison
intentionally removes numerator tokens. On raw PDMX predictions `scans_v4` emits
numerator 5 on only 6/75 reference staves and 12 on 0/113; Arm B emits neither 5 nor 12.
That is a supply tail, not evidence of aggregate accuracy. The next, separately named
continuation starts from Arm A and adds a fixed 796-row training-only replay manifest
(401 rows with 5; 400 with 12) while retaining the Arm A mixture. It must report these
per-numerator raw counts alongside the same three neutral benchmarks.

**Operational note.** Arm A training completed normally, but a Bash `set -u` bug in the
scoring wrapper stopped it before scoring. The saved Arm A checkpoint was scored and the
chain resumed without retraining; the wrapper has a regression-safe resume mode. Thus the
table represents the intended checkpoints, not a second Arm A draw.

## Rare-numerator continuation from Arm A (2026-08-30)

The aggregate tables above drop metre numerators by design, so they cannot answer whether
the known 5/12 supply gap was repaired. Starting from Arm A, a three-epoch continuation
kept its OSSQ/Lieder/PDMX mixture and added the fixed training-only rare replay manifest:
796 unique rows (401 carrying `timeSignatureBeats_5`, 400 carrying `_12`; one row carries
both). This is the direct held-out PDMX result, where "exact position" means the predicted
numerator equals the reference token at its reference sequence position:

| PDMX numerator | Arm A exact position | rare continuation exact position | emits target anywhere |
| --- | ---: | ---: | ---: |
| 5 (75 reference staves) | 7 / 75 | **43 / 75** | 52 / 75 |
| 12 (113 reference staves) | 0 / 113 | **64 / 113** | 66 / 113 |

The arm therefore repairs the named capability gap; a numerator emitted elsewhere is not
counted as correct, which is why the final column is reported separately. `scans_v4` was
only 5/75 and 0/113 on the same exact-position measure, so this is not a minor movement.

The normal three benchmarks remain numerator-neutral and are safety reads against Arm A:

| benchmark | Arm A | rare continuation | delta / 95% paired CI |
| --- | ---: | ---: | --- |
| OSSQ | 95.05 | 95.03 | -0.19pp, -0.33 to +0.26 (ns) |
| PDMX | 88.08 | 87.76 | -0.33pp, -1.06 to +0.45 (ns) |
| Lieder v4 | 94.54 | 94.08 | -0.46pp, -1.13 to +0.02 (ns) |

This is a single targeted continuation and one seed, so the intervals establish no
detectable aggregate regression rather than proving absence of one. The direct numerator
gain is large enough to make this the correct candidate for a repeat/production review;
future reports must retain both the raw 5/12 table and the neutral safety reads.

### Seed-7 replication

A matched seed-7 continuation (the same Arm-A start, mixture, fixed 796-row replay
manifest, and three epochs) reproduces the targeted repair.  It improves the direct
held-out PDMX counts slightly over seed 42:

| PDMX numerator | seed 42 exact / anywhere | seed 7 exact / anywhere |
| --- | ---: | ---: |
| 5 (75 reference staves) | 43 / 52 | **45 / 54** |
| 12 (113 reference staves) | 64 / 66 | **69 / 71** |

The neutral safety reads remain close to Arm A.  PDMX and OSSQ are non-significant;
seed 7's Lieder delta is significant but small and negative.  That makes the repair
repeatable, while setting a real compatibility cost to weigh before promotion rather
than calling the result regression-free.

| benchmark | Arm A | seed 42 | seed 7 | seed-7 delta from Arm A / 95% paired CI |
| --- | ---: | ---: | ---: | --- |
| OSSQ | 95.05 | 95.03 | **95.19** | +0.14pp, -0.02 to +0.34 (ns) |
| PDMX held-out | 88.08 | 87.76 | 87.98 | -0.10pp, -0.80 to +0.63 (ns) |
| Lieder v4 holdout | 94.54 | 94.08 | 93.88 | **-0.65pp, -1.48 to -0.09** |

These are two runs of the targeted continuation, not two independent Arm-A baselines;
they establish replication of the rare-class effect, not a corpus-wide estimate of its
mean trade-off.  Retain seed 42 as the structured-head base already trained, and use the
seed-7 checkpoint only if the next production comparison explicitly values its modestly
better numerator counts over the Lieder decrement.

## Mixed-source structured notation sidecar (2026-08-31)

The earlier promoted `rareNum` structured-head sidecar was trained and scored on
GrandStaff/**kern alone. That split genuinely validates beams, ties, and onset/advance,
but has no directional-stem, phrase-slur, or dynamic targets. Zero support there is a
property of that evaluation split, not a claim about the wider corpus.

This follow-up freezes the same pinned `rareNum` core and trains only its 24
`decoder.structured_heads` tensors for 12 epochs. Its score-disjoint training mixture is
5,959 GrandStaff crops, 3,622 rebuilt-Lieder crops, and a deterministic 6,000-crop PDMX
training sample (seed `464_20260831`). The aggregate held-out index contains the pinned
GrandStaff (1,807), Lieder (300), and PDMX (3,349) splits; 5,278 of 5,456 entries remain
after the existing excessive-ledger-line filter.

| head / measure | held-out support | result |
| --- | ---: | ---: |
| beam level 1 / 2 / 3 / 4 macro F1 | 219,289 / 77,353 / 7,218 / 301 | .852 / .786 / .721 / .434 |
| exact beam vector | 219,289 note positions | .831 (182,288 / 219,289) |
| hooks F1 | 8,433 | .777 |
| ties macro F1 | 16,693 | .844 |
| stem direction F1 (up/down only) | 217,631 | .811 |
| slur spans F1 | 3,400 | .772 |
| slur side macro F1 | 3,121 | .723 |
| advance, nontrivial macro F1 | 166,624 | .751 |
| dynamics macro F1 | 935 | .105 |

Two compact source reads guard against the aggregate concealing a source distinction:
all 298 usable Lieder validation crops, and 475 usable crops from a deterministic
500-row PDMX validation sample (the same seed). They show usable support for the missing
heads rather than a GrandStaff-only proxy:

| source | ties macro F1 (support) | stem direction F1 | slur span F1 (support) | slur-side macro F1 (support) | dynamics macro F1 (support) |
| --- | --- | ---: | --- | --- | --- |
| Lieder | .729 (525) | .832 | .686 (474) | .621 (45) | .124 (158) |
| PDMX sample | .810 (1,964) | .797 | .777 (486) | .743 (494) | .106 (112) |

**Promotion boundary.** The figures validate the mixed sidecar as an experimental
source of beam, tie, stem, slur, and advance predictions. They do **not** justify
emitting dynamics: the apparent 99.7% dynamic micro accuracy is almost entirely the
`none` class and the macro result is .105. At this point the renderer exports beams and
advance; its existing slur path only uses structured values to pair/place a slur already
emitted by the core. Ties and stems require explicit MusicXML integration and visual
round-trip review before the mixed sidecar can be called a production bundle. Artifacts,
including prediction sidecars and exact input manifests, are synced locally under
`artifacts-instance/rareNum-heads-mixed/`.

### Export-review implementation (2026-08-31)

For review only, the renderer now has an isolated structured-export switch
(`HOMR_STRUCTURED_EXPORT=none|tie|stem|slur|...`). It makes a same-crop, same-core-decode
MusicXML A/B possible without fabricating alternate predictions. Predicted stems write
`<stem>` on pitched notes; predicted ties write both direct `<tie>` and notational
`<tied>` elements; a structured slur emits endpoints only when the core did not already
emit a flat slur token. Existing core markings win, so this cannot double-engrave them.
Twenty targeted tests covering the renderer/decoder seam pass locally and on the instance.

The embedded review gallery uses five PDMX crops with visible tie changes, five with stem
changes, and four with additional structured-slur endpoints. The fourth count is factual:
the other high-scoring candidates already had a core slur and correctly remain no-ops in
this fallback mode. This is direct transformer-crop export evidence, not deployment of a
new default sidecar or a substitute for the full detector pipeline.
