# C1 positioned syllables and final-note provenance — 2026-10-05

Review: http://localhost:8765/c1-syllables/. Sixteen original-scan examples show
the unchanged cached text beside approximate CTC-positioned syllable candidates.
Six Chaminade boxes additionally show native measure/beat ownership comparisons.

## Positioned recognition

The installed RapidOCR 3.9.2 and the same pinned local OCR graphs can expose
character boxes from CTC decoding. The pilot reruns unprompted full-page OCR,
verifies each parent text and quadrilateral against the frozen cache, and splits
at recognized spaces/hyphens. Syllable centers use the actual recognized character
positions rather than equal-width subdivisions. Apostrophes, punctuation and
OCR errors are preserved. Missing/mismatched character streams are skipped.

Both calibration scans reproduce **149/149 parent texts and boxes** exactly.
All 149 rows have complete positioned character streams; 46 grouped regions
split into multiple candidates (20 Chaminade, 26 Franz). These counts include
titles and other text: they are not lyric recall. The initial replay took
14.3 seconds with one ONNX thread per pool; the canonical repeat additionally
limits OpenCV to one thread and records that setting explicitly. That repeat
finishes in 16.3 seconds with identical counts and attachment results. All jobs
are complete, ETA 0; the persistent review server remains available.

## Conditional comparison

Six previously reviewed Chaminade boxes contain 17 native lyric syllables,
including melisma cases and bilingual verses. System and verse routing are fixed
from scan review. Source linkage requires an exact written-pitch/duration sequence;
the notehead arm additionally requires count-exact, strictly monotonic vocal
noteheads. These are calibration proposals, not a generally qualified notehead
association policy. Ground-truth text is never passed to OCR or syllable splitting.

| metric, out of 17 syllables | attention anchors | detected-notehead anchors |
|---|---:|---:|
| literal text correct | 15 | 15 |
| note ownership correct | 14 | 17 |
| both correct | 13 | 15 |

Attention sends “ling”, “l'oi”, and “ly_” to the following melisma note. Detected
notehead centers resolve these three selected cases. Literal OCR differences
remain `jé` versus native `j'é`, and `ly_` versus native `ly`; no text cleanup was
fit to those references. Scores use the existing OTS attachment contract/evaluator.
This is a small conditional diagnostic, not C1 corpus accuracy, automatic verse
selection, e2 detector recall, or the deployed fusion route.

## Final decoded/exported notes

The diagnostic now records first-pass provenance on symbols and captures retained
coordinates after candidate selection, cross-staff repairs and voice cleanup.
Symbols that lose provenance are explicitly skipped rather than assigned a
geometry by guessed token index. No normal inference code path is enabled by
this diagnostic metadata.

A 66.3-second one-thread replay preserves coordinates for **414/414 final decoded
notes**, including **70/70 vocal notes**. Both exported XML files and decoded
symbol caches match the earlier uninstrumented-metadata pilot exactly. Vocal
note sequences match their exported XML pitch/duration sequences; all 70 vocal
XML anchors uniquely agree with native staff, physical measure, onset, pitch
(including accidental), duration and grace status. No native/predicted vocal
anchor keys are missing. This verifies the two selected pages; it does not prove
coverage for every fork/repair, polyphonic note ownership or pixel localization.

## Remaining work

1. Test fused syllables with missing separators, such as Franz `nemÁb`, using
   character geometry and scan evidence. Spaces/hyphens alone cannot resolve them.
2. Qualify association between decoded notes and segmented noteheads on count
   mismatches, chords, recognition errors and missed/extra noteheads. The current
   count-exact monophonic mapping must not be generalized by index.
3. Qualify automatic text-role, system and verse routing, including bilingual
   native verse order, then score a broader frozen calibration partition.
4. Integrate an off-by-default producer after those checks. Lyrics fusion and
   production defaults remain unchanged.

Local packets:

- `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-syllable-positions-20261005-single-thread/`
  (OCR source snapshot, model hashes, positioned text, grouped score, review).
- `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-final-anchor-provenance-20261005/`
  (source snapshot, model hashes, final notes, completion receipt, `audit.json`).

Code: `lyric_syllable_positions.py`, `pilot_lyric_syllable_positions.py`,
`score_grouped_lyric_positions.py`, `review_lyric_syllable_positions.py`,
`audit_final_lyric_anchors.py`, and the updated `pilot_lyric_note_anchors.py` under
`training/onnx/`. Three syllable-position tests cover nonuniform CTC geometry,
punctuation/hyphen handling and failure on missing positions; the existing four
geometry/linkage tests and fourteen OTS contract/policy/evaluator tests also apply.
