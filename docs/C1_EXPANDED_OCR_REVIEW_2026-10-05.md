# Expanded lyric OCR review — 2026-10-05

Review the eight expanded calibration scans with their complete-page OCR at http://localhost:8765/c1-layout-ocr/. The gallery is intentionally unscored: it shows original pages and OCR regions without pretending that native lyric metadata is pixel-level ground truth.

Use it to inspect the failure controls before selecting a semantic policy: title and credit text, tempo/dynamics, bilingual rows, continuation systems, sparse opening pages, dense piano accompaniment and OCR fragments split across staff lines. The earlier staff-gap heuristic failed on three pages in this packet, so those pages must remain visible during future selector work.

The next measurement requires independently certified visible lyric rows on a subset of these pages. Those references should be selected from the scan, with native metadata used only for identity and verse ordering. Note pixels and lyric-to-note attachment remain separate gates.
