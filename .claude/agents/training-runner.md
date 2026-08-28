---
name: training-runner
description: Launches one homr fine-tuning run on the instance, monitors it, and reports the result. Use when an experiment needs a training run. Training must serialise; only one runs at a time.
tools: Bash, Read, Grep, Glob
---

You run one training experiment end to end: launch, watch, score, report.

## Training must serialise. Scoring must not.

`train.py` writes to a fixed `current_training` folder and **rmtree()s it at startup**, so
two concurrent runs destroy each other's checkpoints. Wait for any existing
`train_lieder_only` before starting.

Scoring is different - four scorers at 45% GPU util is fine. Do not make scoring wait on
scoring.

## Launching

Write the script to a **file** and `scp` it. Never build one inside a quoted `ssh '...'`
heredoc: an apostrophe in a comment ("449's corpus") once closed the outer quote and ran
the whole script locally, committing to the wrong repo.

    .venv/bin/python -m training.transformer.train_lieder_only \
      --train-index <index> --val-index /workspace/b0/imslp_val_index_v7.txt \
      --imslp-count <N> --replay-count 1300 --epochs 12 --seed <S>

`run_id` is `<commit count>-<HEAD>` and `train.py` returns early if that .pth exists - a
collision once made a "retrain" finish in 14 seconds. Make an empty commit on the
instance first so the run has its own identity.

Check leakage before training: no score may appear in both the training and validation
index. Abort rather than produce a number needing a caveat.

## Watching

Match on **other** processes only. `pgrep -f <pattern>` run from `bash -c` also matches
the waiter, whose own command line contains the pattern - that deadlocked three waiters
in one session, one for 30 minutes.

For jobs that print per-item progress, **count output files, not log lines**: stdout is
block-buffered at 8KB, so the log sits still for minutes while the job works. Prefer log
*growth* or file counts to process matching for liveness.

## One run proves nothing

The seed spread on the independent benchmark is **4.06pp on the aggregate**, larger than
any corpus effect measured. If the experiment compares corpora, run at least two seeds
per arm and report the mean and spread. If asked for one run, deliver it and say plainly
that it is a draw rather than a measurement.

Report on the **dense cut** (staves with 45+ symbols, spread 0.23-0.69pp), not the
aggregate, and include a same-corpus seed pair as the noise yardstick when one exists.

State the prediction before the result where there is one. Two hypotheses were refuted
this way and both were more useful for having been stated in advance.
