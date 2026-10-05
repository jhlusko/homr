# Two-page calibration lyric-row reference

`rows.json` contains12 manually transcribed visible lyric rows from IMSLP154060 p001
and IMSLP89794 p001. Rectangles use original-image pixels. Source verse IDs are retained
separately from printed vertical ordering: Chaminade English is native verse2 above
French verse1; Franz German verse1 is above English verse2. No note pixel attachment
is certified. The reference is calibration-only and review was not blinded to prior
OCR. Original scans and every row crop were visually checked; OCR was not used to
produce or choose the reference text/rectangles. `receipt.json` freezes its identity
and normalization policy.

Use `training/onnx/score_lyric_rows.py` with the frozen references and saved inference
outputs. Character normalization removes layout connectors/whitespace and normalizes
Unicode/case/apostrophes, preserving diacritics and other punctuation. This deliberately
does not establish word boundaries, syllable accuracy or note attachment. Rows link to
the separately certified page/system/native-verse manifest; they do not replace native
source note ownership.

Original image SHA-256 and row crop SHA-256 are retained per row. Regenerate crops from
the original image and `box` using the independently linked page manifest's score/image
mapping; check both hashes. Images/crops and scored outputs remain in the local artifact
archive described in `homr/docs/C1_LYRIC_ROW_QUALIFICATION_2026-10-05.md`.
