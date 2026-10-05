# Staff-gap lyric selection experiment — 2026-10-05

Review: http://localhost:8765/c1-geometry/ (original twelve lyric-row crops beside overlap-only, staff-gap and row-recovery results). This exploratory rule was designed after examining the calibration errors; it is not a predeclared held-out gate.

Input is the exact cached full-page OCR and pinned-e2 area200/10% overlap selection. Staff geometry comes only from original scan pixels: grayscale threshold130, horizontal morphology of width page_width/8, row support above25% of page width, contiguous line runs grouped with maximum35px gap. Only five-line groups with spacing ratio below1.25 are accepted. OCR centers must lie between successive accepted staves, at least half a staff-line spacing from either staff. No source text, manual lyric rectangles, verse map or native note geometry enters selection.

The first variant keeps seeded OCR regions in these gaps. The second also recovers rejected OCR regions in the same gap whose vertical intervals overlap a retained region by at least half the shorter height; recovery follows connected rows until stable. It does not establish vocal voice ownership or syllable-to-note alignment.

| Variant | Lyric-row regions retained / 75 | Retained outside rows | Character edits / 408 |
|---|---|---|---|
| Detector overlap only | 70 | 12 | 10 |
| Staff gaps | 70 | 0 | 10 |
| Staff gaps plus row recovery | 75 | 0 | 5 |

The final variant removes all12 reviewed title/credit/tempo/instrument regions and recovers all5 dropped lyric fragments on these two pages. Its conditional character error equals full-page OCR, including the original transcription errors; it is not perfect recognition. Chaminade yields9 accepted staves; Franz11 accepted staves and2 excluded groups, including an incomplete line group. Missing/skewed staff lines and false grouping can merge gaps. Piano interstaff text, lyrics above staves, more vocal parts, tightly packed bilingual verses and scan resolution changes remain untested. Region retention is relative to cached OCR, not recall of all printed lyrics.

Next: freeze a broader calibration packet before further fitting, including those failure controls; replace this raster prototype with hash-identified runtime system/staff geometry where available. Verify explicit verse ownership and certified note pixels before attachment. Keep checkbox/fusion disabled; do not promote the two-page heuristic.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-staff-geometry-20261005/` (per-page selected/rejected regions, staff groups, parent hashes, scores and executed sources). All cached computations are complete; the review server remains in persistent tmux.
