# Handoff — OSSQ scanned v2 released; v2-stem training active (2026-09-24)

Read `HANDOFF_2026-09-24.md` for the preceding build and experiment history, and
`HANDOFF_2026-09-23.md` for the instance setup. The critical instruction still holds:
**train and evaluate with `/workspace/venv`; never rebuild it.** The isolated
`/workspace/venv-ossqbuild-26` was used only for the OSSQ corpus build.

This snapshot was checked at **05:27 UTC on 2026-09-24**. The instance is
`root@88.207.87.60` on SSH port `40097` (A100 40 GB). The live training ETA below is an
estimate, not a completion marker.

## 1. Live job: mixed training with restored OSSQ stems

The persistent tmux session `mixed_v2` is running `/root/train_mixed_v2.sh`. Its companion
`watch_mixed_v2` checks progress every two minutes and records it in
`/root/watch_mixed_v2.log`. At 05:27 UTC, training had been active for 16 minutes: the main
process and data workers were consuming CPU, and GPU utilization was 82% with 2,002 MiB
allocated. No epoch had completed yet; the trainer logs only at epoch boundaries.

The run is `heads-mixed-v2stems-v7` under
`/workspace/train-20260924/heads-mixed-v2stems-v7/`. It trains 3 epochs on the same
63,931-row mixed index (62,677 after filtering), replacing only the 35,226 OSSQ training
paths with the verified v2 package. PDMX and Lieder rows, the frozen core checkpoint,
hyperparameters, and `/workspace/venv` match mixed v6. The index is
`/workspace/train-20260924/eval-indexes/mixed_train_v2.txt`; its construction script is
`/root/build_mixed_v2_index.py`. The launcher chains separate PDMX, Lieder, and OSSQ-v2
validation evaluations after training, in that order, and exits on any failure.

Expected training finish: **06:20–06:40 UTC**. Expected three evaluations finished:
**07:00–07:20 UTC**. These are based on the preceding run; use the markers and reports to
establish actual completion. Avoid starting another GPU job while this one runs.

```bash
ssh -p 40097 root@88.207.87.60
tmux ls
tail -n 30 /root/watch_mixed_v2.log
tail -n 30 /workspace/train-20260924/heads-mixed-v2stems-v7/train.log
tail -n 30 /root/train_mixed_v2.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Done means `MIXED_V2_ALL_DONE` appears in `/root/train_mixed_v2.log` **and** all three
`eval-{pdmx,lieder,ossq}-valid/report.json` files exist under the run directory. Inspect
the epoch-1 `stem.direction` support in `train.log`: it should exceed mixed v6's 1,738,605
if v2 OSSQ labels contribute the intended gradient. The monitor's `epochs=0` before the
first epoch is expected, not a stall; corroborate with GPU and process activity.

When complete, copy the checkpoint, manifest, logs, and reports to the matching tree under
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/`, then
compare **each validation source separately** against mixed v6 and PDMX-only. Do not
collapse these domains into an aggregate score. In particular, check OSSQ stem accuracy
and macro-F1 against the existing v6 checkpoint on v2 labels (83.45% direction accuracy),
and look for PDMX/Lieder regression before choosing a stem head or arbiter.

## 2. OSSQ scanned v2 is published and pinned

The source fix is `ossq-omr-data` commit `c10725b`: write per-system MusicXML before the
in-place stem strip used for LMXE. `/root/rebuild_stems.py` finished at 03:26 UTC. The
**corrected** `/root/compare_builds.py` report is
`/workspace/ossq-build/build-20260924-stems/compare-with-20260923.json`. Its first
version accidentally compared new files with themselves because it treated old relative
paths as new absolute paths; that bug was fixed before accepting the rebuild. The corrected
comparison shows identical ordered train/valid rows, **0 differing image bytes, 0 differing
token files, 0 non-stem sidecar field changes**, and 889,429 restored stem fields
(806,762 train; 82,667 valid). All 11,572 LMXE files are byte-identical; segment count is
unchanged, and all 11,572 new MusicXML segments contain `<stem>`.

The materialized package passed `tools/package_dataset.py --verify-only`: 35,226 train and
3,911 valid rows, with every reference resolved and no symlinks or absolute paths. It was
sharded into six zstd archives (five train, one valid; 887,780,539 bytes total). An
instance-side extraction of the validation shard resolved every image, token, and sidecar
for its 3,911 rows.

All ten release objects (six shards, two indexes, manifest, checksums) were uploaded
**directly from the instance** to the `archives` bucket under `ossq-scanned-v2/`. All
object heads returned the expected sizes. Public URLs on
`https://pub-d31b33903cd04fec8cc87ee0667c1981.r2.dev/ossq-scanned-v2/` returned 200
with the corpus fetcher's user agent. A separate local consumer fetched the 85 MB public
validation shard via `make fetch-corpus SPLIT=valid`, checked its pinned SHA-256, and passed
`make verify-corpus`. The existing `ossq-scanned-v1/` remains accessible.

`ossq-omr-data` commit **`3b2c045` is pushed to main**: its lockfile, README, Makefile, and
fetcher now pin v2 and explain that v1 lacks usable stem labels. Small release metadata is
also local at `/home/jhlusko/workspace/ossq-release/ossq-scanned-v2/`; the large shards
were not downloaded locally. The instance package is
`/workspace/ossq-build/build-20260924-stems/package/out/ossq-scanned-v2/`.

Do not interpret `/root/publish_ossq_v2.rc = 1` as an upload failure. The uploader finished
every S3 object and size check; its final public request used Python's default user agent,
which Cloudflare rejected with 403. The corpus user agent and the independent fetch above
proved public access. The temporary private rclone config was removed; credentials are
not in this handoff.

## 3. Completed mixed v6 result, to use as the comparator

Mixed v6 trained on **v1 OSSQ**, so its OSSQ stem result on v1 is meaningless: applicable
labels were `unknown` and masked from loss. Training and all three source evaluations
completed at 04:09 UTC. The checkpoint and reports are under
`/workspace/train-20260923/heads-mixed-v6/`, with local copies under the artifact tree
named above. Values below are PDMX-only baseline → mixed v6 on the same source.

| Metric | PDMX | Lieder | OSSQ scanned |
|---|---:|---:|---:|
| Exact beam vector | .862 → .844 | .710 → .820 | .900 → .932 |
| Slur-side macro-F1 | .818 → .800 | .570 → .511 | .840 → .899 |
| Tie macro-F1 | .860 → .831 | .705 → .753 | .746 → .811 |
| Slur-span F1 | .761 → .733 | .482 → .499 | .851 → .881 |
| Nontrivial advance macro-F1 | .790 → .771 | .853 → .855 | .808 → .879 |

Mixed training helped Lieder beams and ties, and OSSQ beams, ties, slur span, and advance;
it regressed several PDMX scores. Lieder slur side has only 45 head-scored sides, so that
metric alone cannot carry a selection decision. Dynamics macro-F1 stays around .10 on all
sources; keep it disabled. The v2-stem run should be judged against these per-source
tradeoffs, not selected solely for improving OSSQ stems.

The v6 checkpoint was also evaluated against restored **OSSQ v2 labels** without
retraining: `/workspace/train-20260923/heads-mixed-v6/eval-ossq-v2-valid/`. Stem
direction-only accuracy is **83.45%** (macro-F1 .6632, micro-F1 .8592); exact beams .9314,
slur side .8972, ties .8091. This is the proper before value for the new v2-stem run.

## 4. Rules, paired comparisons, and the beam release gate

The restored stems unlocked OSSQ rule measurements. On **exactly the same slur endpoints**,
the opposite-stem rule beats the mixed v6 head on scans:

| Source | Paired sides | Rule macro-F1 | Mixed head macro-F1 |
|---|---:|---:|---:|
| OSSQ v2 | 5,998 | .9463 | .8996 |
| PDMX | 3,575 | .8192 | .7996 |
| Lieder | 45 | .3211 | .5109 |

The paired scorer is `/root/paired_slur_all.py`. Reports are under
`/workspace/train-20260924/rules/ossq-v2/slur_side_paired.json` and
`/workspace/train-20260923/rules/{pdmx,lieder}/slur_side_paired.json`, copied into the
local artifact tree. The broader OSSQ rule score is .9464 macro-F1 on 5,999 scorable
sides; one side has no paired head slot.

The beam-derived stem rule scored 94.4% on 38,229 joined OSSQ notes (2,070 staves skipped).
On the held-out stem arbiter's **19,539 matched reporting notes**, the v6 head scored
83.75%, rule 93.86%, and tuned confidence switch 93.83%. This favors the rule for now,
but v7 is explicitly testing whether v2 stem supervision improves the head. Arbiter output
is `/workspace/train-20260923/heads-mixed-v6/eval-ossq-v2-valid/stem_arbiter.txt`.

On OSSQ scans the earlier PDMX-only **beam head beat the beam rule on paired notes**,
90.8% versus 81.9% (+8.9 points), contrary to the roadmap's old default-to-rule
assumption. That is **not a ship decision**. The roadmap's named gate is precision on
rest-spanning beams, and it remains unmeasured. The audit
`/workspace/train-20260924/rules/ossq-v2/rest_spanning_gate.json` found 7,527 flagged
rest tokens in 3,904 held-out crops, but reference and prediction beam vectors omit those
tokens. In 3,886 crops, the missing vector count equals the flagged-rest count. Across
3,564 validation crops joined to MusicXML (13,329 rests) and all 122 cleaned source
MusicXML scores (185,942 rests), no rest carries a `<beam>` tag. There are 340 validation
crops without a source segment join. Thus current labels and evaluator cannot measure the
rest-spanning gate; do not infer that the head passed or failed it, and do not ship a
beam-rule hybrid based on the old heuristic. One inspected source `.mscx` file has
`<Rest><BeamMode>begin32</BeamMode>` plus beam-group markers, a possible avenue for
recovering ground truth; it has **not** been validated or joined to predictions.

## 5. Next actions and records

1. Let `mixed_v2` finish; use `watch_mixed_v2`, the actual epoch log, GPU activity, and
   `MIXED_V2_ALL_DONE` to distinguish progress from completion. Recalculate ETA if epochs
   run slower than expected. Do not disturb `/workspace/venv`.
2. Sync the v7 checkpoint and all three reports locally. Compare PDMX, Lieder, and OSSQ-v2
   separately with the tables above; report support counts for stem and slur metrics.
   Record whether v2 labels improved OSSQ stems enough to justify the mixed checkpoint,
   and whether other domains regressed.
3. Decide stem and slur-side inference policy from paired results, with Lieder's small
   slur sample called out. Update `RUNLOG.md`, the private roadmap, and artifact `STATUS.md`
   with the actual v7 outcome. Keep dynamics disabled.
4. Recover and validate rest-spanning beam ground truth (possibly from MSCX), join it to
   the held-out crop predictions, and score the named gate before any beam shipping decision.

For broader release criteria, read
`OurTextScores/docs/private/OTS_HOMR_PUBLIC_RELEASE_ROADMAP.md`,
`homr/docs/HANDOFF_DETECTOR_RETRAIN.md`, `homr/docs/TIE_LABEL_INVESTIGATION.md`,
`homr/RUNLOG.md`, `pdmx-omr-data/docs/BUILD_PIPELINE.md`, and
`homr-artifacts/gpu-roadmap-20260919/STATUS.md`. The roadmap is gitignored; the artifact
directory is not a git repo. `homr` has pre-existing uncommitted edits, including RUNLOG;
`ossq-omr-data` retains unrelated dirty build-doc/packager files. Do not stage or discard
those while committing this handoff or future release notes.
