# Checkpoint results ledger

Every paired comparison run during the Lieder corpus work, in one place. Companion to
`CORPUS_CHANGELOG.md`, which traces what changed in the corpus; this records what those
changes scored.

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
