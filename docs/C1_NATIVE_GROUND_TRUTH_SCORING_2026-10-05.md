# Native lyric ground-truth scoring — 2026-10-05

The expanded pages are now scored against the existing native MusicXML lyric ground truth, restricted to each page's independently aligned physical measure window. This uses source text, verse identity and source note/onset ownership already present in `c1-source-lyrics-20261005/source-lyrics.jsonl`; no new manual transcription is required for this diagnostic.

The current score is an ordered character coverage diagnostic between page-window source text and complete-page OCR. It is useful for checking whether OCR contains the expected lyric characters, but it does not separate lyrics from titles/credits, assign OCR regions to verses, or attach syllables to notes. Those are separate fusion gates. The score must not be read as OCR precision or a production acceptance score.

| Score | Source characters | Ordered source coverage |
|---|---:|---:|
| IMSLP89287 | 268 | 52.6% |
| IMSLP89290 | 297 | 40.1% |
| IMSLP617260 | 218 | 35.8% |
| IMSLP89796 | 273 | 54.6% |
| IMSLP435041 | 98 | 100.0% |
| IMSLP286659 | 152 | 96.1% |
| IMSLP83318 | 100 | 100.0% |
| IMSLP83314 | 93 | 97.8% |

The low scores on dense bilingual/continuation pages reflect concatenated OCR order and non-lyric text interleaving; they are not evidence that the native source is wrong. The high scores on sparse single-verse pages are encouraging but still do not certify region precision or attachment.

Local report: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-layout-expansion-20261005/native-text-diagnostic-v3.json`. The scorer is `training/onnx/score_native_lyric_text.py`. Next use native source targets to evaluate verse/note attachment directly in the OMR output and use scan galleries only for detector coverage and false-positive adjudication.
