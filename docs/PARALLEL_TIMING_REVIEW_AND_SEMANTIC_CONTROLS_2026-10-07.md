# Live timing curves and broader semantic controls — 2026-10-07

Two local reviews now run alongside the frozen-core timing probe.

## Timing dashboard

http://localhost:8765/timing-heads-20261006/

The dashboard refreshes every 15 seconds. It separates each timing head and source,
shows training/validation losses, class support and recall, and compares accuracy
with the training-derived majority-class baseline. Missing validation classes are
shown explicitly. Epoch zero is initialized-head validation, not a trained model.
No tiny smoke-run measurements are mixed into the full-run results.

Run artifacts: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-probe-20261006/`.
Dashboard artifacts: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-probe-review-20261006/`.

At 05:33 UTC, 2,753/4,237 feature crops were complete. Extraction ETA was
05:45 UTC, followed automatically by six head-training epochs. Training duration
has not yet been measured. The original wall-time ETA included a long clock gap;
the watcher now estimates from recent progress and resets after gaps over 120s.
Training stays in `timing-probe-full`; its replacement watcher is in
`timing-probe-monitor`. The trainer and cached identity were not modified.

These remain teacher-forced representation measurements, not free-decoding or
release qualification. OSSQ training is one score and validation only two scores.
Next inspect per-class learning, then run free decoding and broader OSSQ checks.

## Broader lyric/direction reference gallery

http://localhost:8765/c1-semantic-expanded-20261006/

Eight Lieder calibration pages from distinct scores exclude all 14 previous C1
calibration scores and the reserved examples. Five pages have both lyric and
direction positives; three supply additional directions. Selection is deterministic
and independent of OCR quality. Formal test pages were not read. Being new to these
C1 calibration runs does not establish independence from detector/core training.

Full-page OCR received scans only. References were retained separately for review.
The gallery contains all 11 positive direction boxes, 15 sampled positive lyric
boxes, and 16 deterministically selected unannotated OCR regions. Unannotated
regions were never treated automatically as negatives.

Codex inspected all 42 original-pixel crops and four additional non-lyric context
bands: 26 lyric regions, 15 directions and one dedication/credit. This is Codex
role inspection, not user adjudication, classifier performance, transcription or
note-attachment truth. Candidate evidence remains visible and the adjudication
binds each decision to image, box and crop/context hashes. Written `cresc.` is
marked as a textual direction subtype here; glyph-dynamics routing is unchanged.

Inputs and adjudication: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-expansion-20261006/`.
Final gallery: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-expansion-review-20261006-v2/`.
The input directory preserves the originally executed selector source, before
formatting, to match its receipt hash. The first gallery is retained separately.

Next evaluate a fixed classifier using scans/OCR/context without references as
inputs; report lyric recall, false lyric decisions on directions/credits and
abstentions, grouped by page/score. Geometry alone is insufficient. This small
calibration set does not qualify promotion; use independent held-out checks after
any tuning. Automatic lyric fusion remains off.

## Verification

Five focused tests passed: deterministic score-disjoint selection, OCR/reference
overlap semantics, balanced timing sampling, exact optimizer resume/epoch extension,
and cache-tamper refusal. Black, Ruff and dashboard JavaScript syntax checks passed.
HTTP checks verified all 84 text/context images and four live dashboard endpoints.
No model/corpus pins, publication, deployment or environment changes.
