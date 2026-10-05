# Intact OCR lyric selection — 2026-10-05

Cached full-page OCR regions were selected by union coverage of their bounding rectangles against pinned e2 Lyrics rectangles. Selection uses no reference text, manual row geometry or source lyrics. Six settings were declared before scoring. OCR input remains intact; no inference or detector rerun was needed.

| Detector component minimum area | Minimum coverage | Character edits / 408 |
|---|---|---|
| 200 | 0.1 | 10 |
| 200 | 0.25 | 31 |
| 200 | 0.5 | 46 |
| 4 | 0.1 | 11 |
| 4 | 0.25 | 31 |
| 4 | 0.5 | 45 |

Full-page OCR has 5/408 edits; the 10% overlap settings have 10–11/408, versus 56/408 for nested OCR inside area-200 detector crops. Stricter overlap loses more lyric fragments. Two calibration pages and twelve manually transcribed rows are insufficient to select a release threshold. Conditional character error excludes outside-row regions; it does not measure semantic lyric precision, verse ordering, syllable attachment or production latency. Detector rectangles can also select titles and notation.

Review original rows beside full-page and the predeclared area-200/10% setting: http://localhost:8765/c1-intact-lines/. No individual votes required.

Next: broaden frozen calibration coverage and independently audit retained non-lyric regions and missed lyric fragments, then certify verse ordering and note attachment. Keep fusion/checkbox disabled.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-intact-line-gate-20261005/`; frozen references in sibling `c1-lyric-row-reference-20261005/`. Executed source preserved before formatting.
