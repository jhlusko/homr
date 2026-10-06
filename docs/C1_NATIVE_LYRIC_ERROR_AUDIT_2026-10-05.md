# Native lyric error decomposition — 2026-10-05

The frozen v6 result remains **311/480 exact targets (64.8%)**. An evaluation-only
audit now partitions all 169 unrecovered targets, without shifting notes, changing
predictions or feeding reference text into inference.

| Target outcome | Count |
| --- | ---: |
| Exact literal text at exact native note and oracle-mapped verse | 311 |
| No attachment at a native identity present in decoded vocal XML | 100 |
| Different text at exact note and oracle-mapped verse | 60 |
| Native note identity absent from decoded vocal XML | 5 |
| Ambiguous native identity | 4 |
| Total | 480 |

These are output outcomes, not independently certified causes. The 100 can include
missing OCR, row/verse grouping errors, segmentation or association abstentions.
The 60 can include recognition errors or a neighboring syllable attached to the
wrong note. The five use strict measure/onset/pitch/alteration/duration identity;
they do not necessarily mean five missing pitches. Four references remain
ambiguous rather than being credited to an arbitrary native note.

| Score | Exact | No attachment at decoded identity | Different text | Absent decoded identity | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jagdlied | 46 | 27 | 8 | 0 | 0 |
| Schlummerlied | 57 | 15 | 12 | 2 | 4 |
| In Welschland | 52 | 11 | 3 | 2 | 0 |
| Tanzlied im Mai | 62 | 16 | 10 | 0 | 0 |
| It was a Lover and his Lass | 31 | 0 | 2 | 0 | 0 |
| He doesn't love me | 40 | 7 | 1 | 1 | 0 |
| Rose d'Amour | 15 | 12 | 11 | 0 | 0 |
| À une Fleur | 8 | 12 | 13 | 0 | 0 |

Seven text differences disappear with NFC plus trimming outer whitespace only.
Thirty-one disappear after broader case-folding and removal of all nonalphanumeric
characters, but that diagnostic also erases meaningful differences: native
`we're` versus predicted `were`. Do not change the literal metric or production
text using this broad rule. Other examples include native `be!` versus `be.`,
predicted trailing underscores, and native nonbreaking spaces around syllables.

Next inspect the 100 missing attachments first, starting with Jagdlied's 27 and
the two pages with missing outputs despite few/no unresolved OCR candidates
(In Welschland and Tanzlied im Mai). Trace each against native measure/beat,
selected OCR, physical row and nearest segmented head; do not assume an unresolved
candidate is a known missing lyric. Then inspect the 60 literal differences to
separate visible recognition errors from shifted syllables. Keep the four native
ambiguities and five strict note-identity failures separate. A detector/model
change is premature until these causes are established.

Local canonical report:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-native-lyric-error-audit-v2-20261005/report.json`.
Per-score JSON in that directory includes every target's native measure, onset,
pitch, duration, verse, text, category and candidate text where applicable.
The initial audit is preserved in `c1-native-lyric-error-audit-20261005/` alongside
it. All frozen scoring input hashes were checked. The page-global verse mapping
comes from evaluation only; native verse IDs and independent pixel ownership
remain unqualified. Duplicate capacity is capped, exact matches are reserved
before mismatch allocation, and allocation within duplicate keys is bookkeeping.

Four new focused tests pass, covering duplicate capacity, exact-match reservation,
fixed offsets, missing decoded versus missing attachment identity, ambiguous
native notes and the danger of broad normalization. All jobs done, ETA 0; runtime
fusion remains disabled.
