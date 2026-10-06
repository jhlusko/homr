# C1 note-anchor pilot — 2026-10-05

Review: http://localhost:8765/c1-note-anchors/.

The linked Chaminade `IMSLP154060` and Franz `IMSLP89794` calibration scans were
replayed with public-joint-core-heads-20260928 epoch27, fixed local float32 ONNX
graphs, one CPU thread, and the existing page-diagnostic policy. This policy does
not enable stem/slur-side arbitration or detector fusion. Neither native lyrics
nor source note locations were passed to inference.

The diagnostic captures both crop origins, the actual canvas resize/padding,
staff scaling, the exact piecewise-dewarp mesh, and the original-page resize
ratio. Its inverse uses the same triangles as the forward mesh and rejects
uncovered, ambiguous, nonfinite, or padded coordinates.

| page | first-pass notes | coordinates mapped |
|---|---:|---:|
| Chaminade | 207 | 207 |
| Franz | 247 | 247 |

The initial two replays finished in 108 seconds; the final source-snapshotted repeat
finished in 86 seconds and reproduced both XML outputs and anchor arrays exactly.
All compute is complete, ETA 0. Approximate attention positions
are orange; segmentation noteheads are green. Median distance to the nearest
detected notehead is 23.1 original-page pixels on Chaminade and 15.9 on Franz
(90th percentiles 35.4 and 37.8). These are localization diagnostics, not verified
same-note errors: proximity can select another note, and segmentation itself is
imperfect. Visual inspection shows some attention points toward stems/beams.

## Native ownership and conditional scoring

All six lyric-bearing vocal systems have exact count, written-pitch, and note
duration agreement with their source sequences: 25 Chaminade and 45 Franz notes.
The first Chaminade system has no vocal notes. A linkage proposal is made only
when the whole system's sequence matches; missing/extra notes cannot silently
shift source ownership by index. This does not independently certify accidentals,
bar/rest timing, the note pixels, or ownership after final export/repair. Measure
and beat tooltips are explicitly source-linkage proposals.

Three selected single-syllable OCR boxes were visually checked against the scans
and native lyrics: Chaminade “If” (measure 6, beat 1, native verse 2), Franz “Nun”
(measure 1, beat 1, verse 1), and Franz “Wein,” (measure 3, beat 1, verse 1).
With fixed reviewed system/verse routing, nearest-x gives **3/3 correct text and
note ownership**. OCR predictions are frozen by SHA-256; references are checked
against native lyrics. This is a smoke test of the contract/evaluator, not corpus
accuracy, automatic verse selection, pinned-e2 recall, or the production path.
No unreviewed note-pixel links are certified by this result.

Full-page RapidOCR also groups multiple syllables into a box: “I were”,
“star - ling gay”, and “Si jé - tais”. One box cannot be attached wholesale to one
note. The three smoke cases exclude these grouped boxes deliberately.

## Next work

1. Extract individual syllable positions from grouped OCR regions, keeping the
   better full-line recognition and original scan geometry. Do not divide a box
   uniformly by character count or use source text as an OCR prompt.
2. Preserve anchor provenance through selected candidates, cross-staff repairs,
   voice cleanup and XML export. Current overlays are first-pass coordinates.
3. Qualify automatic lyric-system/verse routing against native ground truth,
   including reversed bilingual verse order, melisma and missed notes. Score a
   broader frozen calibration partition before considering runtime integration.

The scripts are `pilot_lyric_note_anchors.py`, `run_lyric_anchor_pilot.py`,
`lyric_anchor_geometry.py`, `link_lyric_anchor_sequences.py`,
`score_lyric_anchor_witnesses.py`, and `review_lyric_note_anchors.py` under
`training/onnx/`. Four geometry/linkage tests and fourteen OTS contract/policy/
evaluator tests pass. The persistent tmux runner writes per-page logs, progress,
completion receipts and a source snapshot. Production settings remain unchanged.

Canonical local packet:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-note-anchor-pilot-20261005-final/`.
The first successful packet is preserved at the sibling
`c1-note-anchor-pilot-20261005-v2/`. The original attempt lacked the local
`rapidfuzz` import required by the reused page runner; installing RapidFuzz 3.14.1
in the laptop's existing `.venv` resolved that dependency without rebuilding it.
