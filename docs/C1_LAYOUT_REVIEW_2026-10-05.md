# Expanded lyric calibration review — 2026-10-05

The eight frozen pages are visually suitable for the next calibration pass. The packet includes bilingual German/English pages, single-verse English pages, opening pages with titles and credits, continuation pages, and a French page with dense direction text. The score identity and page boundaries are still pending formal manual certification; no OCR or model output influenced selection.

Review the packet at http://localhost:8765/c1-layouts/. The thumbnails are for layout review only. The original, hash-verified scan paths and native source identities are recorded in `c1-layout-expansion-20261005/selection.json`.

The next scoring pass should run full-page OCR, pinned-e2 regions, and the staff-gap geometry diagnostic on these pages, then create independent visible-row references. Report results by layout stratum. Preserve title, tempo, credit, bilingual verse, and incomplete-staff failure cases rather than filtering them out.

I completed a page-level identity/full-page/layout review for all eight pages. The checklist is in the local artifact `c1-layout-expansion-20261005/visual-certification.json`; every page is marked identity and complete-page checked, while row transcription and note-pixel attachment remain explicitly pending.

The transcription worksheet is at http://localhost:8765/c1-row-worksheet/. It lists OCR candidates by vertical position beside the original scan. OCR is only a prompt for review; any reference row must be transcribed from the scan.

This packet does not yet qualify lyric precision, verse ordering, note attachment, or production fusion. Keep the lyrics checkbox and fusion disabled.
