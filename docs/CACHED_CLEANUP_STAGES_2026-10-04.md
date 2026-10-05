# Isolated real-source cleanup stages — 2026-10-04

Tuplet removal and pitch deduplication account for the aggregate recognition damage in standalone OSSQ/Lieder crops. Both heuristics also repair some errors, so switching them off globally is not qualified. The next experiment should constrain when they run using predicted context, then verify complete voice/page behavior.

## Controlled results

All4,211 paired crops are retained:3,911 OSSQ crops/9 scores and300 Lieder crops/13 scores, accepted September18 core/September24 v7 heads versus public-initialized joint epoch27. No recognition rerun. Raw33,688 metrics replay exactly; standard cleanup and no-pitch controls exactly reproduce the previous audit across all16 source/arm/metric aggregates. Each ablation restores a fresh stream and holds the other stages fixed. Combined variants are explicitly identified.

Symbol errors (same denominators OSSQ102,219/Lieder11,255):

|Source / stack|Raw|Standard cleanup|Disable only tuplets|Disable only pitch dedup|Disable both|
|---|---:|---:|---:|---:|---:|
|OSSQ / accepted|4,067|4,185|4,078|4,174|4,067|
|OSSQ / joint27|2,580|2,773|2,595|2,758|2,580|
|Lieder / accepted|333|380|373|340|333|
|Lieder / joint27|286|327|326|287|286|

Disabling lower-staff reassignment changes no cached crop here. Disabling repeated-header removal changes3 Lieder crop/model streams but none of these recognition metrics; the scorer excludes clef/time-signature declarations, so this is not proof of correct meter/header export. Disabling both tuplets and pitch dedup recovers the raw aggregate metrics, as does disabling all stages. Pitch/rhythm/accidental counts and every changed stream are retained in the report.

Tuplet-only crop outcomes: OSSQ accepted33 improved/17 worsened/3 tied; OSSQ joint27 24 improved/2 worsened/1 tied; Lieder accepted2 improved/1 worsened; Lieder joint27 1 improved/1 worsened. Therefore aggregate improvements do not imply every edit is safe. Disabling tuplets alone also raises OSSQ pitch errors by2 in each arm through differing event alignment; pitch/rhythm/accidental gates must remain separate.

## Exact-recognition positive and negative witnesses

The heuristic computes the median duration of measures within its input stream and removes all tuplets from any shorter measure. It ignores explicit meter, changes of meter, incomplete measure context and independent voices. Its name suggests a mean, but the implementation uses a median.

There are23 crop/model pairs where raw recognition exactly matches all legacy symbol labels, and this heuristic introduces errors:21 OSSQ and2 Lieder. These are strong label-relative witnesses, not independently certified full-page XML examples.

* Accepted OSSQ `sq10307350_0016_0003_4.txt`: symbol errors0→24; disabling tuplets returns0.
* Joint27 OSSQ `sq7354505_0047_0002_1.txt`: errors0→18; crop-relative measures1–3 each contain six eighth-note-triplet tokens. Their predicted duration is1/2 whole note versus a local median1, triggering conversion to ordinary eighth notes.
* Both stacks, Lieder `IMSLP621832-sys6-v1.tokens`: errors0→6; crop-relative measure3 has duration3/4 against median1, with six eighth-note-triplet tokens changed to ordinary eighth notes. Disabling tuplets returns0.

The negative controls matter equally:9 accepted OSSQ crops become exactly correct because tuplets are removed. For example `sq8885571_0006_0002_2.txt` has16 raw rhythm errors corrected to0. In its crop-relative measures6–7, eight sixteenth-triplet tokens per measure are converted to ordinary sixteenths. Disabling tuplets restores all16 errors. These examples were selected after scoring for diagnosis; they cannot serve as a new untouched qualification set.

Retained `tuplet-witnesses.json` records exact counts, crop-relative measures, model-predicted rational durations, rhythm transitions and input/script hashes. Its companion inspection script is retained locally for reproducibility. Corpus IDs retain historical remote prefixes as identifiers; all file inputs and evidence paths are local.

## Complete Grieg page/voice control

Replayed the instrumented pre-cleanup complete-voice streams for both models on the previously certified Grieg opening page. Standard XML and measured scores exactly reproduce the original page outputs. All14 arm/variant XML exports succeed.

Disabling tuplets, lower-staff reassignment or header removal separately leaves the measured event/pitch/rhythm/accidental scores unchanged. Disabling pitch dedup, alone or with tuplets, yields pitch2→0/372, accidental2→0/372 and event/rhythm46→44/380 for each model, reproducing the earlier two-unison finding. The remaining44 timing errors concern source tremolo/elapsed-duration representation, already documented separately.

This demonstrates a difference between standalone-crop and full-voice behavior on one page. It does not establish that tuplet cleanup is safe on all complete pages. XML may differ in unscored notation/header details even when these event metrics tie. Voice identity and wider positive/negative coverage remain required before enabling a default correction.

## Next bounded work

1. Develop an opt-in, predicted-context tuplet guard that handles explicit meter changes, pickups/incomplete bars, multivoice duration and crop boundaries. Retain the successful false-tuplet corrections as negative controls. Never use reference timing at inference.
2. Qualify unison preservation versus genuine repeated noise and differing accidental/duration events; inspect the existing longest-duration dictionary-key defect independently. Preserve voice ownership as a separate export gate.
3. Replay candidate policies on complete page/voice traces, then resume broader final-policy page qualification and independent text-fusion calibration. Existing defaults and promotion decisions remain unchanged. No training is needed to investigate these runtime heuristics.

## Local evidence

Crop report, all changed streams, source snapshots and witness inspection:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/cached-cleanup-stages-20261004/`

Complete-page ablations:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/page-cleanup-stages-20261004/`

One-thread tmux launch, logs, EXIT0 and completion watcher:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/cached-cleanup-stages-20261004-launch/`

Tools: `training/transformer/audit_cached_cleanup_stages.py` and `training/onnx/ablate_page_cleanup_stages.py`. They refuse existing output, require the previous control report/qualified page traces, and record hashes. Eleven focused tests and Ruff/Black pass. Complete, ETA0; no inference-default change, GPU job, upload, push, deployment or model promotion.
