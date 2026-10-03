# Reference-derived rhythmic advance: OSSQ/Lieder, 2026-10-03

Completed correction qualification and cached predicted-advance diagnostic: see `TIMING_FIX_QUALIFICATION_2026-10-03.md`. The initial reference improvements below are retained unchanged; the follow-up finds preservation blockers and keeps predicted advance disabled.

Reference-derived advance recovers substantial Lieder timing lost by legacy grouping. This tests canonical MusicXML → current parser labels → XML, **not model predictions or recognition accuracy**. Parser-derived sidecars contain errors themselves.

| Same inventory-preserved staff measures | OSSQ | Lieder |
|---|---:|---:|
| Successful indexed references / attempted | 75 / 85 | 212 / 212 |
| Common eligible measures | 120,218 | 39,915 |
| Common eligible events | 525,443 | 253,664 |
| Moved-onset events: disabled → enabled | 470 → 412 | 15,855 → 5,252 |
| Net reduction | 12.3% | 66.9% |
| Improved exact-onset-count measures | 22 | 1,945 |
| Worse exact-onset-count measures | 0 | 39 |
| Newly lost correct inventories | 0 | 0 |

Ten OSSQ detachedLegato failures remain explicit. Every successful disabled-advance control reproduces the previous roundtrip XML hash, metrics and timing exactly. Enablement uses the current runtime allowlist and fallback. Three Lieder measures recover inventory and are excluded from common-measure comparisons. Order changes remain 106 / 145 measures. Whole-score event edits change 8,364 → 8,330 and 14,057 → 14,051; these are not model symbol errors.

The one-thread A/B completed in approximately 438 seconds in persistent tmux, EXIT=0. Its completion watcher finished. No compute remains, no PDMX was read, and no model pin or production behavior changed.

## Regression investigation

Exact-onset mismatches do not measure distance. Of 23 retained regression examples (capped at three per score), 19 improve minimum summed absolute onset distance despite worse exact-match counts; four worsen it. This selected sample does not cover all 39 measures. Matching identical pitch/duration/grace events ignores voice identity.

The four worse examples are Lieder IMSLP154220 (physical staff 2, measure indices 8 and 28) and IMSLP22647 (staff 2, indices 28 and 56). An isolated chord-cursor hypothesis changes neither score's metrics. An isolated grace-cursor hypothesis removes all onset errors on their common eligible measures: enabled-advance moved events fall 18 → 0 and 14 → 0, with no exact-count regressions. Neither probe edits production source.

Cause: `_process_note` calls `append_note(..., max(duration, 1), ...)` for pitched notes, including grace notes with absent duration. This advances the label parser one tick and corrupts advance labels. Restoring the pre-call cursor for grace notes fixes these two examples. This supports a future-build correction, not qualification of every grace sequence or trained predictions.

## Separate export precision blocker

A synthetic mixed-duration group demonstrates that advance can change XML divisions from 6 to 2 and round a printed triplet from one-third quarter to zero duration. Division selection considers consumed group duration; it also needs to preserve printed note/rest durations. `precision-probe.json` retains both event lists. This is a duration-corruption blocker even though the real-source A/B introduces no new loss of previously correct inventories.

Next: narrowly test division allocation against printed durations and cursor advances; correct grace cursor handling for future corpus builds. Repeat real-source qualification after each change before evaluating predicted advance. Preserve original results and published pins; corpus corrections require versioning. Earlier whole-bar-rest and reciprocal-tuplet representation losses remain separate issues.

## Local evidence

All results are under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/real-reference-advance-ablation-20261003/`: `report.json`, all 297 outcomes in `scores.jsonl`, XML examples, reference/source hashes, logs, `EXIT`, watcher status, `regression-distance-audit.json`, `precision-probe.json`, and `chord-cursor-probe.json` with twelve selected-score XML exports. The latter records both cursor hypotheses and requires exact original A/B replay.

Reproduce with `training/onnx/ablate_real_reference_advance.py --baseline <local real-timing-label-roundtrip-20261003> --out <new directory>`, then `training/onnx/probe_real_advance_regressions.py --run <new directory>`. Use the existing HOMR virtual environment and audit dependency overlay with OMP/OpenBLAS/MKL threads set to one. Source snapshots retain tested versions.

This diagnostic preserves inventory within physical staff/measure boundaries. It does not establish voice-aware equivalence, scan alignment, untouched-page generalization or a promotion gate.
