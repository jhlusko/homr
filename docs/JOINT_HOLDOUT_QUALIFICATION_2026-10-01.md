# Joint epoch-27 training-held-out comparison — 2026-10-01

## Completed local comparison

The full raw crop comparison completed **8,422/8,422 decodes with zero failures**. Candidate symbol error improved by 1.455 percentage points on OSSQ and 0.418 points on Lieder. The first 3,501 accepted-model decodes used two CPU threads; the remainder used one at the owner's request. See `execution-history.json` for the preserved resource provenance. No GPU or remote instance was used.

**4,211 crops × 2 model pairings = 8,422 decodes**: OSSQ 3,911 crops / 9 score identities; Lieder 300 / 13. This completed job has no remaining ETA. Runtime/export qualification is a separate gate; the first ONNX pilot matched recognition on 21/22 crops and did not pass strict parity.

Status: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-qualification-20261001/FULL_WATCH_STATUS.json` and `FULL_STATUS.md`.
Logs: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-qualification-20261001/logs/full-{accepted_oldcore_v7,public_joint27}.log`.
Results on completion: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-qualification-20261001/holdout/comparison.{md,json}` plus both prediction JSONL files.
Browser status: http://localhost:8765/joint-qualification-20261001/FULL_STATUS.md .

The watcher records completion/failure and result deltas into local `homr/RUNLOG.md` and the private release sequence. A nonzero exit, missing coverage, or decode failure is not a qualification pass.

## What is being compared

- **Accepted local pairing:** September-18 core (same core SHA as provider pins) plus September-24 v7 heads. Core SHA `1b2855ce3facea9aa8b0017f6e30143abd9aa43569a2f9ce1ab46bf8bf818fa3`; v7 SHA `d78c0f0b4ea9e5584ca80920d09ad8759aaf53f36b02924f302016385b7d7b4e`. This is not the deployed provider's older heads, nor the public rareNum + real-only frozen-head diagnostic reference.
- **Candidate:** `public-joint-core-heads-20260928`, epoch27, containing its own jointly trained core and heads. SHA `6b146d0d7d2b5a7e4ecfe6d2d004bfd49ef5eef9a5843251b3696b466d1255cc`. Comparison folder dates identify when comparisons ran; September28 is the training run's recorded identity/date.

No head transplantation, pin change, deployment or new training. Eager float32 greedy math-SDPA decoding, final-divider enforcement off; no post-decode runtime policies. This compares complete core/head weight pairings, **not the complete production pipeline**. Torch is local `2.14.0+cpu`, versus historical GPU `2.6.0+cu124`; both current arms use the same local environment and hash-recorded code. All six current token-ID mappings match the archived pinned-core scoring vocabulary. Do not combine these results with GPU results as if they were one unchanged protocol.

Published OSSQ-v2 validation shard (85,431,911 bytes) was downloaded and SHA-verified against its archived lock (`68519b83959c122481687386a976080518eecf3b5cf1b90f9e3535fcae68b284`). It is extracted separately under this run's `corpus/`; v1 is unchanged. Lieder uses local published v8 crop/label/sidecar files. Both arms verify image, token and sidecar hashes before decoding and compare actual pixel/reference hashes before reporting.

## Split and gate limits

`split-audit.json` checks all validation score identities against the public core's OSSQ/Lieder training indexes and the joint run's frozen training index: **zero score overlap**. Historical old-core/v7 training membership is not completely proven from these indexes. Published validation was previously evaluated and used for model selection: it is training-held-out, **not an untouched new test set**. No independent new-score release claim is justified.

Owner authorized **Lieder symbol error at most +0.25 percentage points** versus the accepted pairing. Report separate OSSQ/Lieder symbol, rhythm, pitch and accidental errors with score-clustered 2,000-resample intervals, plus correct-recognition-and-notation GT-note coverage. Keep point deltas and intervals visible; model promotion remains a separate evidence/review decision. Tie/slur-event totals include NONE and are not span F1. PDMX remains a separate diagnostic and is not included in this real-source run.

## Completed pilot

All **44/44 decodes succeeded** on one deterministic SHA/seed-selected crop per score. Exact paired pixel/reference/sidecar checks pass; a deliberately incomplete result set was rejected by the reporter.

| Source | Symbol error old core + v7 | Joint27 | Difference |
|---|---:|---:|---:|
| OSSQ | 17.373% | 2.119% | -15.254pp |
| Lieder | 2.018% | 1.468% | -0.550pp |

The OSSQ result is driven by **one crop** (`sq8806881_0008_0005_3.txt`) where the old core produces sixteenth-note triplets (`note_24`) for ordinary sixteenths (`note_16`): old41 edits vs joint5. Other eight sampled OSSQ crops have no symbol edits in either arm. This is a diagnostic example, not a representative magnitude of improvement. Pilot support is only236 OSSQ /545 Lieder reference events. OSSQ stem coverage 79.23→97.27%, beam coverage71.55→95.69%; Lieder stems89.21→98.37%, beams82.26→100%. Confidence intervals are in `pilot/comparison.json`; no promotion from this small pilot.

Pilot report: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-qualification-20261001/pilot/comparison.md`.

## After full crop results

### 2026-10-02 — candidate export and parity audit

The public-joint-epoch-27 checkpoint was exported in an isolated directory with one CPU thread. The encoder, decoder and structured-head ONNX graphs passed ONNX model checks and contain embedded weights; the shipping cache and provider pins were not modified. Export receipts and hashes are in `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-runtime-qualification-20261002/exports.json`.

On 22 deterministic pilot crops, encoder outputs were within `7.63e-6` maximum absolute error and the 12 structured-head projection checks had matching argmaxes. Recognition tokens matched on 21/22 crops and cached eager predictions matched 22/22. The one recognition divergence is a low-margin Lieder rhythm at `IMSLP89797`, step 130: eager `note_2.` vs ONNX `note_4`; margins were 0.0232 and 0.0075. This is a real float32 eager-versus-ONNX greedy boundary difference, so strict parity is **not passed**. The candidate must not be promoted on this evidence.

Notation-only differences occur mainly on chord/non-note tokens; XML serialization was identical on 21/22 crops. The retained audit is `semantic-parity-audit.json`, with the low-margin investigation in `attention-isolation.json`. Full-page functional smoke is running separately on one Lieder and one OSSQ original page; it checks segmentation, candidate graphs, MusicXML validity, and policy behavior, but cannot close recognition parity or accuracy gates.

1. Inspect score-level recognition regressions against the +0.25pp Lieder margin; retain all failed/missing cases and review divergent real-source crops.
2. Prepare an isolated local candidate export/pin, preserving rollback. Test eager-versus-ONNX parity on fixed crops before treating candidate exports as production-equivalent.
3. Run full-page detection, decode and MusicXML-export checks with matched detectors and explicit runtime policy settings. **Do not automatically reuse the v7 stem-confidence cutoff for jointly trained heads:** its calibration applies to v7, and candidate confidence/policy behavior needs separate validation. Full-page recognition ground-truth alignment and genuinely new real-score tests remain open.

### 2026-10-02 — full-page functional smoke completed

The exported public-joint epoch-27 graphs ran through the local full HOMR pipeline on one original Lieder page (`IMSLP152191`) and one original OSSQ page (`sq10414906`). Both completed with one CPU thread, valid MusicXML, nonempty parts/notes, and zero automatic dynamics. The Lieder page took 148.7 seconds (4 parts, 92 measures, 512 notes); the OSSQ page took 124.8 seconds (4 parts, 51 measures, 331 notes). Outputs, decoded staves, teasers, hashes and exact policy identity are under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-runtime-qualification-20261002/full-page-{0,1}/`.

This is a functional export/pipeline smoke, not an accuracy result: no matched full-page reference scorer was run. The measured times are CPU-only and cannot be compared with the 280-second production GPU/HTTP budget. The full-page script is `training/onnx/qualify_joint_pages.py`; it disables uncalibrated stem/slur arbitration policies and records that choice.
4. Promote or write public model-advantage claims only after those remaining gates; this local run alone does not close release qualification.

Reproducible runner: `homr/training/transformer/qualify_joint_holdout.py`, committed locally as `f452d9a` (not pushed). Local supplementary import packages are isolated in `homr-artifacts/tools/audit-python`; neither training environment nor remote venv was rebuilt.


### 2026-10-01 — owner-requested single-thread comparison resume

At20:15 UTC, stopped the decoder and resumed its saved3,501 successful accepted-oldcore/v7 crops using **one CPU thread**; candidate public_joint27 had not started. Torch intra-op/interop, OpenCV and BLAS/OpenMP are limited to1. Both remaining arms run in tmux `joint_full_holdout_20261001`; watcher restarted in `joint_full_holdout_watch_20261001`. Pre-change results/identities are archived under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-qualification-20261001/thread-change-20261001/`. `execution-history.json` records the3501 two-thread crops versus subsequent single-thread decodes; the final comparison report will include this resource provenance. Inputs/models/scorer are unchanged; no completed results lost. The watcher's interruption/failure entry was generated by the requested stop, not a decoding failure; the run is resumed. ETA is recalculated from recent single-thread samples, with pre-change pilot fallbacks until sufficient new samples exist. Laptop/WSL must remain awake.
