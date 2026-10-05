# Lyric selection coverage review — 2026-10-05

Review: http://localhost:8765/c1-selection/ (17 original-scan context crops). No individual voting is required. The setting remains the predeclared area-200/10% rectangle-union overlap; no reference text or row rectangles were used to select OCR regions.

Across the two calibration pages, full-page OCR supplies75 regions centered in the12 certified lyric rows and74 outside. Selection retains70 inside and12 outside, dropping5 inside and62 outside. Retention is70/75 (93.3%) of existing lyric-row OCR fragments, not recall of all printed lyrics. The five dropped fragments read `a_`, `A`, `va -`, `ne` and `al.`; the last is an OCR error on the printed lyric `a`.

All17 context crops were visually inspected. The12 retained outside-row regions are two Chaminade titles, its composer/tempo, two Franz titles plus title punctuation, author/composer credits, piece number, tempo and instrument label. None is a sung lyric row. `visual-adjudication.json` records these classifications and crop identities. Within this small reviewed packet,70/82 retained regions lie in certified lyric rows; this85.4% region-level proportion is not pixel precision, word accuracy, note attachment or a held-out release gate.

Stricter area-200 overlap thresholds trade away lyric fragments: at25%,66 retained inside /7 retained outside /9 dropped inside; at50%,62 /7 /13. This confirms that raising overlap alone does not solve semantic selection while preserving recall. No threshold was promoted.

Next: preserve full-page OCR and test score/system/voice geometry for semantic selection on a broader frozen calibration set, with titles and bilingual verses as negative/positive controls. Avoid filtering by recognized title or lyric text. Certify verse ownership and note pixels separately before note attachment or checkbox activation.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-lyric-selection-audit-20261005/` (report, visual adjudication, executed sources). The review is served from the existing persistent loopback tmux server; all audit compute is complete.
