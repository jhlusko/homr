# OSSQ / Lieder timing-label census — 2026-10-03

Scope is OSSQ and Lieder only. This audit counts canonical-reference encoding losses and reads the recovered public-core train/validation label files. It runs on one CPU thread without model inference, rendering, training or corpus changes.

## References for scores included in the published corpora

| Category | OSSQ — 85 scores | Lieder — 212 scores |
|---|---:|---:|
| Printed timed note/rest events | 1,026,004 | 274,652 |
| Full-measure rests | 16,507 | 2,737 |
| Full-measure-rest duration losses | **356 in 15 scores** | **132 in 18 scores** |
| Of those, six-quarter full-measure rests | 246 | 41 |
| Explicit time-modification events | 63,811 | 15,035 |
| Consistent-source reciprocal rounding losses | **2,031 in 13 scores** | **46 in 8 scores** |
| Scores with either category | **21 / 85 (24.7%)** | **22 / 212 (10.4%)** |

Full-measure-rest losses affect 2.16% of such rests in OSSQ and 4.82% in Lieder. Rounding losses affect 3.18% and 0.31% of explicitly time-modified events respectively. Combined, the two categories are 0.233% and 0.065% of all printed timed events. They are concentrated rhythm problems; these counts alone are unlikely to explain a widespread recognition regression.

These are whole canonical score references for identities included in the published indexes, not a claim that every listed event occurs in a retained scanned crop. No image-to-reference coverage assumption or model error subtraction is made. Grace/invisible notes are excluded from the canonical timing denominator; frozen token counts below have a different denominator.

## What loses timing

The full-measure-rest helper falls back to `rest_1` for durations absent from its table. In indexed OSSQ references, lost quarter durations are 6 (246 events), 9/2 (39), 5/2 (36), 8 (21), 5 (11) and 79/16 (3). In Lieder they are 9/2 (58), 6 (41), 9/4 (26), 5/2 (4) and 11/4 (3). Adding only the six-quarter case would address about 69% of this OSSQ category and 31% of Lieder's; it would leave the other cases unresolved. The unusual source durations should be examined rather than automatically normalized to a meter.

For explicit tuplets, the current encoding formula rounds `written reciprocal × actual / normal` to an integer. The census calls a rounding loss confirmed only when the reference `<duration>` agrees exactly with the explicit ratio, type and dots, and decoding the rounded token changes that duration. This is an arithmetic check, not a full parser/bracket-state or alignment roundtrip. Hidden or state-suppressed ratios may take another path; token spellings alone cannot distinguish correct quintuplets from wrongly encoded duplets.

Of 2,031 indexed OSSQ rounding losses, 1,257 use ratio 2:3 and another 429 use equivalent 4:6 or 8:12 ratios. Lieder's 46 are 34 at 2:3, eight at 4:3 and four at 5:3. OSSQ also has 446 explicit ratio/source-duration disagreements and 12 unsupported tuplet-base events; those are reported separately, not added to confirmed rounding losses. Indexed Lieder has none of those disagreements. Exact elapsed-time preservation and preservation of the written tuplet glyph/ratio are different requirements: a dotted-duration substitute can preserve some duplet timings while losing their notation meaning.

## Frozen label census

All **39,137 OSSQ** and **3,867 Lieder** indexed label files were read successfully, with per-file SHA-256 receipts. They contain 1,032,094 and 146,502 note/rest tokens. Overlapping crop windows repeat events, so these are token exposures, not unique source events.

There are 15 OSSQ sole-`rest_1` occurrences under an explicit 12/8 header, across seven windows/four score identities; Lieder has three occurrences across two windows/two scores. These are review flags, not automatically certified errors: a crop boundary or incomplete bar can make a sole rest ambiguous. Locations are retained in `frozen-labels.jsonl`. Headerless crops are not counted by this conservative flag.

Frozen OSSQ has 767 `note_5`, 68 `note_5.`, 12 `rest_5`, 31 `note_10`, 120 `note_20` and 65 `note_40` exposures. Lieder has two `note_5`, ten `note_20` and 20 `note_40` exposures. These tokens can describe legitimate tuplets and must not be globally rewritten or excluded. Neither corpus has `rest_0` in the audited files; repairing its current runtime handling alone would not fix these existing labels.

OSSQ label bytes were read from the local v1 materialization. The locked v2 rebuild receipt explicitly records zero token-file differences; its change restored stem sidecars. This audit hashes the local files but does not freshly verify every one against the published archive manifest. Lieder uses the local v8 build selected by the archived published PD212 indexes, rather than all 4,187 locally built candidate windows.

## Larger canonical collections

All 122 OSSQ and 330 Lieder references were readable; no reference read failed. The larger collections contain 1,413,771 and 429,357 printed timed events. Full-rest losses are 374 across 20 OSSQ scores and 395 across 27 Lieder scores; rounding losses are 2,713 across 18 OSSQ scores and 150 across 13 Lieder scores. These larger collections include score identities outside the current published scanned corpora. Keep those counts separate from the 85/212-score table above.

## Historical parser checks

Recovered OSSQ build revisions `555d7c2e4093d2b817a42632ae9b146f9dff5236` and `cc3dcd9fda492e13d2dfbeb5d6ee287a0649a5b7` have the same `_measure_rest_rhythm`, `TupletState`, `_process_note` and `kern_to_symbol_duration` function syntax trees as current source. Their exact files and hashes are archived. The current Lieder vendor pin `1b3d6253590cb42d36a7bc4df12865486e6d9389` also matches these functions; that current pin does not by itself prove the original v8 invocation/build environment.

## Whole-score parser/export roundtrip

The additional audit completed all 452 reference attempts in **809.8 seconds**, one CPU thread, with `EXIT=0`. Its export allows beam/stem/tie/slur fields, **without rhythmic-advance output**, matching the paired page policy; no late repairs or model inference run. Measures are paired by fixed physical staff and measure index. Onset/order counts below are conditional on an unchanged pitch/duration/grace event multiset in the measure. Other measures can also have timing errors; matching inventory does not establish voice or engraving correctness.

| Published-score reference scope | OSSQ | Lieder |
|---|---:|---:|
| Successful roundtrips / attempts | **75 / 85** | **212 / 212** |
| Inventory-preserved physical staff measures | 120,218 | 39,915 |
| Such measures with event-order changes | **106 in 22 scores** | **145 in 48 scores** |
| Such measures with onset changes | **141 in 30 scores** | **2,735 in 160 scores** |
| Moved-onset events within those measures | **470** | **15,855** |
| Ordered event edits, all successful-score measures | 8,364 / 869,287 | 14,057 / 275,863 |

The last row is a reference-label roundtrip diagnostic, not model accuracy, crop token SER, missing-note count or a promotion gate. It includes order effects; a moved sustained note can incur deletion/insertion edits despite an unchanged note inventory. Whole-score grace events are included here, unlike the timed-event census denominator above. Empty staff measures without events in either output are not counted in the onset/order totals. Fixed measure pairing is not an alignment proof and can be affected by altered measure structure.

All ten failures in the indexed OSSQ subset are `Unsupported articulation detachedLegato` at export. Across the larger collections, OSSQ succeeds on 110/122 (11 such export failures and one parser `KeyError: '256th'`); Lieder succeeds on 313/330 (16 backup-position errors and one parser `IndexError`). No failed reference is silently scored as correct or removed from coverage. These full-score failures do not establish that every retained crop from the score fails.

The larger successful collections have 143 order-changed / 195 onset-changed inventory-preserved OSSQ staff measures, and 236 / 4,021 for Lieder. Their moved-event counts are 634 and 22,763. These aggregate onset diagnostics establish scope, not the cause of every change. A bounded test with ground-truth rhythmic advance enabled would distinguish lost timing in legacy grouping from what the sidecar can recover; it would not by itself qualify a model's predicted advance head.

## Local artifacts and reproduction

Census: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/real-timing-label-census-20261003/`.

It contains `report.json`, `report.md`, all located `reference-issues.jsonl`, `references.jsonl` with canonical hashes, `frozen-labels.jsonl` with indexed label hashes, logs/exit receipt, historical parser snapshots and source snapshots. Tool: `training/onnx/census_real_timing_labels.py`. Inputs are the local reference-audit manifests and recovered published train/valid indexes recorded in the report. No PDMX index or reference is read.

Whole-score parser/export roundtrip: sibling `real-timing-label-roundtrip-20261003/`, using `training/onnx/roundtrip_real_timing_labels.py`. It separately reports parser/export failures and onset/order changes in inventory-preserved physical staff measures. Those conditional measures are not a replacement for recognition scoring or a voice-aware musical-equivalence judgment.

Both persistent tmux jobs finished (`EXIT=0`); no remaining ETA. Eleven focused tests pass, covering exact timing-loss classification, source-ratio disagreement, conservative frozen-rest flags and conditional onset/order counting. Source snapshots and input hashes preserve the tested implementation, including relevant historical parser files.

Next: inspect representative six-quarter/rest and duplet cases in the retained scanned crops, then implement narrowly tested future-build duration handling and an explicit policy for ratio preservation. Strengthen onset/meter/voice-aware full-page scoring. Keep published corpus pins and original model metrics unchanged; a blanket token rewrite or new GPU training is not justified by this census.
