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

## Missing-separator calibration follow-up

An opt-in spacing hypothesis splits a token only when each resulting fragment has
at least two characters and the gap exceeds 2.5 times the token's median positive
CTC character-center spacing. It preserves every recognized character. Threshold
2.5 was selected on the `nemÁb` calibration example; it is not a frozen production
rule or an independent test result.

The Franz row changes `mei | nemÁb` to `mei | nem | Áb`. Its reviewed native targets
are `mei | nem | Ab`: all three note owners are correct with fixed system/verse
routing and count-exact noteheads; literal text is correct for two of three.
The OCR accent is deliberately retained. The other 148 rows on the initial pages
are unchanged.

The same OCR graphs replayed the eight previously frozen expanded calibration
pages in 69.8 seconds, using one OpenCV thread and one ONNX thread per pool.
All 594 parent texts/boxes reproduce exactly; the same spatial rule makes no
additional proposals. Across ten calibration scans, only one of 743 regions
changes. Unchanged regions are stability controls, not 742 independently certified
negative examples, and no recall/false-positive rate is inferred from these counts.

The repaired example is at http://localhost:8765/c1-syllables/; sixteen unchanged
word/phrase controls are at http://localhost:8765/c1-gap-controls/. The controls
include non-lyric text and are selected for inspecting long character sequences.
Five syllable-position tests pass. All compute is complete, ETA 0.

Local evidence: sibling directories under the artifact workspace named
`c1-missing-separators-20261005-final/`, `c1-gap-expanded-positions-20261005/`, and
`c1-gap-expanded-proposals-20261005/`. The first contains the native witness score;
the latter two contain the expanded OCR receipt, changed-region report and controls.
The driver is `training/onnx/pilot_missing_lyric_separators.py`.

## Remaining work

1. Qualify the spatial-gap hypothesis against broader native-linked missing-
   separator and ordinary-word controls. Small gaps, short fragments and OCR
   character errors remain unresolved; the successful single witness is not a
   general syllable-segmentation gate.
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
