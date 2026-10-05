# C1 lyric attachment baseline — 2026-10-05

The existing rendered-source benchmark is the baseline for lyric-to-note attachment. It uses
MuseScore's note geometry and the corresponding ground-truth lyric boxes, then assigns each
syllable to the nearest note horizontally within the voice staff. The result is:

| measure | result |
|---|---:|
| all scored syllables | 26,503 / 26,998 (98.2%) |
| one-note syllables | 98.7% |
| syllables held across notes (melisma) | 93.0% |
| within one note either side | 99.3% |

It scored 2,584 systems and skipped 327 systems. The skipped cases are a coverage limitation
(mostly systems with multiple lyric lines or no verse-1 syllables), not attachment errors.
The earlier single-line subset was 3,079 / 3,113 syllables (98.9%). These measurements are
recorded in `RUNLOG.md` sections 27.52 and 27.57.

The OTS service now contains the same policy as an off-by-default diagnostic baseline in
`services/ots-homr-modal/lyric_attachment_policy.py`. Its output is validated by the
`LyricAttachment` contract, but it is not wired into production inference and has not been
claimed as a detector/OCR result.

The next measurement must separate the pipeline stages: detector misses, OCR text errors,
and attachment errors. To do that, a producer must expose note anchors (part, note ordinal,
onset, staff band and geometry) together with lyric detector boxes and OCR text. The native
source ground truth is the primary evaluator; the MuseScore nearest-x result remains the
cheap floor and regression check.
