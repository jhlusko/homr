# Full-page reference audit and paired diagnostic — 2026-10-03

The corrected public-joint-core-heads-20260928 epoch-27 candidate is now compared with the accepted September 18 core + September 24 v7 heads on one original scanned page. Results are mixed. This is a diagnostic, not a promotion decision or untouched-test result.

## Reference readiness

The audit checks current public train/validation indexes, recovered historical indexes, frozen page/system alignment, complete contiguous physical-measure boundaries, canonical reference MusicXML and local page availability. Historical index membership is conservative evidence of possible exposure, not proof that every listed crop was trained on. Absence from recovered indexes cannot establish complete historical non-exposure.

Of 330 Lieder scores and 1,926 page entries, 42 score identities have no known index exposure. Four locally available fully boundary-certified pages pass all other checks, all from IMSLP323995 (Clara Schumann, *Lorelei*). RUNLOG IV.9 explicitly records earlier bar-count evaluation of this score. Thus **zero verified untouched pages** are available through this audit. Nine page entries fail reference-count consistency; their boundaries were not relaxed.

OSSQ has 122 local canonical score references and 2,986 original scan pages, but zero scores with aligned scanned-system metadata. Existing metadata describes synthetic engravings and must not be used as scan alignment.

Audit evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/full-page-reference-audit-20261003-final/` (`audit.json`, `pilot-pages.json`, page/score manifests and the hashed historical-evaluation excerpt).

## Matched original-page comparison

Original local IMSLP PDF was re-rasterized with the existing 250-DPI/autocrop helper. The page was selected by score hash before recognition, not selected for favorable model results. Page 1 covers reference physical measures 0–12, end excluded. Both stacks use isolated float32 exports, the corrected continuation-context runtime, identical post-decode policies, one CPU thread and no text detectors/title OCR. This is not the deployed quantized/provider pipeline or production latency measurement.

| Ordered MusicXML edit error | Accepted | Joint epoch 27 |
|---|---:|---:|
| Pitch | 1.955% (7/358) | 1.397% (5/358) |
| Rhythm | 3.599% (14/389) | 4.370% (17/389) |
| Combined event | 3.599% (14/389) | 4.370% (17/389) |
| Accidental projection | 0.838% (3/358) | 1.397% (5/358) |

Accepted and candidate took 118.84 and 118.04 CPU seconds respectively. Both found three staves. No inference failures or automatic dynamics occurred. All persistent tmux jobs completed; no remaining ETA.

The scorer compares ordered per-physical-staff note/rest sequences. It preserves fractional quarter-note durations, backup/forward cursor behavior, chords, grace events and inherited divisions. Extra/missing staves incur edits. Pitch includes alteration. Invisible notes are omitted. Onset, barline, notation and lyrics accuracy are not scored; polyphonic ordering and optimal sequence alignment limit interpretation. These event edits are **not crop token SER** and cannot implement the accepted +0.25 percentage-point Lieder token gate.

The accidental increase consists entirely of deletion edits: three versus five, with **no accidental substitutions**. It is not evidence of a worse accidental head. Pitch edits are five deletions plus two insertions for accepted, versus five deletions for candidate. Rhythm edits are seven replacements/seven deletions versus ten replacements/seven deletions. `edit-audit.json` records one optimal alignment with original measure numbers and explicitly quarter-note beat units.

## Shared timing issues to investigate next

Follow-up completed: `PAGE_TIMING_LABEL_AUDIT_2026-10-03.md`. Exact cached replay and repair ablations locate these shared timing errors in the current label representation, not late repair on this page. The two fractional events are explicitly marked duplets (2:3 eighths), not dotted glyphs. Original comparison scores below remain intact; the follow-up does not waive their errors or promote the candidate.

Both XML outputs encode the vocal whole-bar rests in measures 6–8 as four quarters, while the 12/8 reference requires six. The XML generator gives positive-duration rests their decoded duration and only treats a zero-duration rest as a full-measure rest. This identifies a representation/export boundary to audit; it does not yet establish whether recognition or export should change. Both outputs also emit 4/5-quarter durations for two reference 3/4-quarter events in measure 12. Investigate cached symbols and tuplet-repair policy before attributing those changes to training.

Candidate raw headers predict 2/8, while accepted raw headers predict 12/8; both final XML outputs contain 12/8. The final meter correction must not hide that raw recognition difference. Additional candidate rhythm changes remain and cannot be excused by the shared errors.

Next: bounded cached-symbol/XML-only ablations for rest and tuplet handling, with separate artifacts and unchanged original scores. Then acquire or certify genuinely new scanned-page references and OSSQ scan alignment before a release accuracy gate. Detector-fusion calibration and C1 lyric attachment remain independent work. No new GPU training, promotion, provider pin or deployment follows from this one-page result.

## Local reproduction evidence

Paired run: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/full-page-lorelei-20261003/`.

It contains the original PDF, rendered pages and render receipt; accepted export/checkpoint identities; model/page manifests; per-arm source identities, XML, cached decoded symbols and result receipts; `run-pair.sh`, `pair.log`, `comparison.json`, `comparison.md`, and `edit-audit.json`. Source snapshot and SHA manifest preserve the tested tooling. Candidate export/runtime qualification remains archived separately in `joint-integrated-runtime-20261003/` under the same parent. Those runtime snapshots include pre-existing dirty pipeline work; a tooling commit alone does not reproduce all runtime policies from a clean checkout.

Tooling: `training/onnx/audit_full_page_references.py`, `export_accepted_page_stack.py`, `score_full_page_musicxml.py`, `audit_page_event_differences.py`. Three focused reference/scoring tests cover boundary completeness, inherited divisions/polyphony/invisible notes and extra/missing staff penalties.
