# Cached page timing and label audit — 2026-10-03

The Lorelei replay reproduces both original XML outputs exactly. Disabling late tuplet, beam, tie or slur-crossing repair, separately or together, leaves the ordered pitch/rhythm/event edit counts unchanged. These shared timing errors are already present in cached symbols; late tuplet repair does not cause them on this page.

One CPU thread; persistent tmux `page_xml_ablation_20261003` completed with `EXIT=0`. No model inference, GPU, training, publication or deployment was performed. Six focused reference/scoring/ablation tests pass.

## Results

| Diagnostic | Accepted event errors / 389 | Candidate event errors / 389 |
|---|---:|---:|
| Original policy, exact XML replay | 14 | 17 |
| All late repairs disabled | 14 | 17 |
| Sole whole-rest duration hypothesis | 11 | 14 |

Pitch edits remain 7 versus 5. The rest hypothesis changes only the three sole vocal whole rests in measures 6–8, using each output's predicted 12/8 meter rather than reference durations. It leaves the candidate's three-error rhythm disadvantage intact. A lone whole rest can occur in an incomplete measure, so this is **not an approved general runtime repair**.

Disabling rejection of a contradictory stated numerator changes candidate final meter from 12/8 to its raw 2/8 on all three staves. The ordered note metrics do not change because they do not score meter/onsets. Keep this raw recognition mistake visible; metric equality does not imply equivalent musical output.

## Current training labels lose reference timing

A separate canonical-reference → current label parser → XML roundtrip reproduces all five shared vocal duration errors without using model predictions:

* Reference full-measure rests have six quarters. `_measure_rest_rhythm` lacks the six-quarter case and falls back to `rest_1`, which exports four quarters. A dotted-whole duration is representable (`rest_1.`), but the current parser does not emit it here. A build fix would change future corpus labels; it would not repair existing checkpoints or published datasets automatically.
* Measure 12 contains two explicitly marked duplet eighth notes: two in the time of three. Each lasts 3/4 of a quarter. The parser rounds the reciprocal duration, 8 × 2/3, to integer 5, yielding `note_5`; the duration decoder emits 4/5 of a quarter. The models' matching tokens therefore reproduce a label limitation. These are duplets, not dotted glyphs, even though dotted-eighth timing is equivalent. Replacing them with dotted labels would discard the explicit ratio/engraving meaning.

The piano roundtrip preserves the unordered note/rest event multiset, but changes the ordering of two sustained bass D2 notes relative to other voices in measures 7 and 11. The ordered scorer charges four pitch/event edits (two deletions/two insertions) despite matching event inventory. This demonstrates a representation/onset-order limitation, not four missing or wrongly pitched notes. Original onset-aware MusicXML remains authoritative. Do not subtract these diagnostic counts from the original model scores as a corrected release metric.

The first-page vocal and piano parts parse successfully when bounded to the certified 12-measure window. Parsing the entire 73-measure reference also exposed a later `Backup duration is too long 22 -24` exception; that full-score failure was not used to filter the page benchmark or change its reference.

## Why `rest_0` is not a shortcut

An isolated probe confirms `kern_to_symbol_duration('0')` currently returns fraction 1, despite its special full-measure-rest comment. The XML generator only marks a rest as a measure rest when its duration fraction is zero. The probe thus produces an ordinary whole rest with four-quarter duration. Furthermore, `state.beats` stores the meter numerator rather than duration in XML divisions. Making only the parser return zero would reach a duration-unit mismatch in the generator. A fix needs coordinated duration, grouping and XML behavior with meter changes, polyphony and partial-bar tests. It should not be enabled as a quick correction to these three measures.

## Next work

The requested OSSQ/Lieder quantification is complete: see `REAL_TIMING_LABEL_CENSUS_2026-10-03.md`. It reads no PDMX data and reports canonical versus published-score scope, frozen label exposures, historical parser checks, full roundtrip failure coverage and conditional onset/order counts.

1. Audit the frequency of six-quarter measure rests and non-integral reciprocal tuplets in canonical real-source references and frozen labels. Verify which archived parser built each published pin before claiming a historical label defect.
2. Make a narrowly tested future-build correction for representable full-bar durations, and design explicit ratio support or a documented rejection path for durations the integer reciprocal vocabulary cannot preserve. Retain published corpus pins; any regenerated labels require a new version. No broad whole-rest runtime rewrite or new training is justified yet.
3. Strengthen full-page evaluation to report onset/meter/voice-aware failures separately from ordered event edits, then certify genuinely new page references and OSSQ scan alignment. Continue text-fusion calibration/C1 independently.

## Local evidence

XML replay, all 16 variant outputs, post-repair symbols, raw rhythm counts, exact-replay checks, source hashes, rest-zero probe and log:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/full-page-xml-ablation-20261003/` (`report.json`, `report.md`, `run.log`, `EXIT`).

Canonical reference-part excerpts, parsed labels, regenerated XML, ordered differences and unordered-event checks:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/full-page-label-roundtrip-20261003/` (`report.json`, `part-{1,2}/`, `run.log`). These diagnose current source, not the unknown historical parser version of every dataset pin. All original matched model outputs and metrics remain intact in sibling `full-page-lorelei-20261003/`.

Scripts: `training/onnx/ablate_cached_page_xml.py` and `training/onnx/audit_page_label_roundtrip.py`. Hash-verified source snapshots are retained alongside both reports, including pre-existing dirty runtime source.
