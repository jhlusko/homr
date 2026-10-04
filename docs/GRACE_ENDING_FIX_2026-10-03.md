# Preserve grace notes beside ending markers — 2026-10-03

The MusicXML label merger now emits `voltaStart`, `voltaStop` and `voltaDiscontinue` as standalone structural symbols before the note chord. Ending markers consume no time and no longer hide pitched members of their group. Other marker ordering, grace cursor handling, duration allocation and the advance allowlist remain unchanged. This is a future-build label correction; published corpora and model/provider pins are unchanged, and predicted advance remains disabled.

Previously, a zero-time grace note could share a source onset with an ending marker. `create_chord_over_two_staffs` treated that marker as a note-chord member; the XML exporter handled the leading marker and ignored the chord's notes. Correcting the grace cursor exposed this issue in OSSQ Beethoven sq9396439.

## Bounded verification

Eighty-two focused tests pass, including end-to-end parser/export cases for start, stop and discontinue endings and a test preventing an ending marker from taking the note's advance target. They preserve two zero-time grace notes, the following quarter note and the ending marker. The located real-score preflight replays the prior corrected advance-off/on XML hashes, metrics and timing exactly before applying the merger correction.

| Score | Prior correction | Ending-marker correction |
|---|---:|---:|
| OSSQ sq9396439 emitted grace notes | 63 | 75 |
| OSSQ sq9396439 ordered event edits | 40 | 28 |
| OSSQ sq9396439 rhythm edits | 12 | 0 |
| Lieder IMSLP89026 event edits | 4 | 4 |

All twelve grace notes in the four located ending-marker groups are restored, including four already hidden before the grace-cursor correction. On common inventory-preserved measures, no onset counts regress and no previously correct inventory is lost. The Lieder duplicate-rest limitation remains unchanged. This two-score preflight is not the full qualification or a model-prediction gate.

## Full qualification complete

All 297 indexed OSSQ/Lieder canonical references were compared in four arms: exact prior corrected controls with advance off/on, then ending-marker correction with advance off/on. The driver parses once per merger version and isolates exports with deep copies. Unchanged parser/generator source hashes match the previous qualification. Both prior controls reproduce exact XML hashes, metrics and timing; all ten OSSQ detachedLegato failures replay unchanged. Success coverage is 75/85 OSSQ and 212/212 Lieder in all arms. No PDMX or recognition/model inference is involved.

The ending-marker correction qualifies against the prior corrected-label controls: no newly lost correct inventory, no worse common-measure onset counts, no new failure, and no per-score regression in event, pitch, rhythm or accidental edits. Both advance-off/on arms restore nineteen grace notes across three OSSQ scores and recover three canonical staff/measure inventories (thirteen events). Lieder outputs and metrics remain unchanged.

| OSSQ score | Prior event edits | Corrected event edits | Restored grace notes |
|---|---:|---:|---:|
| sq8561633 | 1,513 | 1,510 | 3 |
| sq8806134 | 4 | 0 | 4 |
| sq9396439 | 40 | 28 | 12 |

OSSQ aggregate edits decrease: event 8,370 → 8,351; pitch 2,199 → 2,180; rhythm 6,211 → 6,192; accidental 918 → 899. Common eligible onset counts remain identical versus each matching prior control. Recovered inventories increase eligibility from 120,218 measures / 525,443 events to 120,221 / 525,456. No timing errors are waived. Event-edit counts are reference roundtrip diagnostics, not model recognition error rates.

One CPU thread, persistent tmux `grace_ending_qualification_20261003` and watcher `grace_ending_watch_20261003`; both finished EXIT=0. Runtime: 1,057.5 seconds (17.6 minutes). All jobs complete, no remaining ETA. Source snapshots and hash receipts retain the tested code and original merger.

This qualifies the narrow ending-marker preservation change, not the combined parser/representation policy or model-predicted advance. The three duplet-duration inventory regressions and one duplicate-rest inventory regression from the preceding audit remain unresolved. Next: explicitly preserve simultaneous duplicate rests as separate voices/backup events and qualify that correction; address future-build exact tuplet/ratio representation before publishing new corpus versions. Predicted advance remains disabled pending broader OSSQ/Lieder model tests. No published corpus or provider pin changes.

Local parent: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `grace-ending-preflight-20261003/`: two-score report, eight XML exports, source snapshots/hashes and reference provenance.
- `grace-ending-qualification-20261003/`: full comparison, original merger snapshot/receipt, tested source snapshots, launch script and progress/completion receipt.

Tooling: `training/onnx/qualify_grace_ending_fix.py`. Use the existing HOMR environment and audit dependency overlay with OMP/OpenBLAS/MKL threads set to one. Underlying duplet rounding, duplicate-rest/voice preservation and broader predicted-advance qualification remain separate follow-ups. See `TIMING_FIX_QUALIFICATION_2026-10-03.md` for the original blockers.
