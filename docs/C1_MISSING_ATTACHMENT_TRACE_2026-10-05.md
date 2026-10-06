# Missing lyric attachment trace — 2026-10-05

Constrained follow-up now gains 20 exact targets (331/480) while preserving every
baseline attachment and adding no outputs without native lyric-note identity.
The bounded row rule rejects distant directions but needs held-out/in-row controls;
see `docs/C1_ESTABLISHED_ROW_RECOVERY_2026-10-05.md`.

All 100 missing attachments at exact decoded native identities now have cached
traces. Instrumentation leaves every baseline prediction and unresolved candidate
unchanged across eight pages; literal recovery stays **311/480 (64.8%)**.

| Missing-output stage | Targets |
| --- | ---: |
| No accepted decoded-note/notehead association | 41 |
| Accepted head selected by syllables only in other physical lyric rows | 30 |
| No routed syllable selects the accepted head | 29 |

The latter two categories do not prove missing OCR or a verse-routing bug. They
describe source-free candidate geometry evaluated against the native verse via
the same evaluation-only page-global permutation. No native label enters routing
or matching; no source-guided reassignment is performed.

| Score | No accepted head match | Only other rows | No routed syllable |
| --- | ---: | ---: | ---: |
| Jagdlied | 20 | 4 | 3 |
| Schlummerlied | 8 | 7 | 0 |
| In Welschland | 0 | 9 | 2 |
| Tanzlied im Mai | 0 | 10 | 6 |
| It was a Lover and his Lass | 0 | 0 | 0 |
| He doesn't love me | 2 | 0 | 5 |
| Rose d'Amour | 5 | 0 | 7 |
| À une Fleur | 6 | 0 | 6 |

All 27 missing targets in In Welschland and Tanzlied im Mai already have accepted
head matches. Inspecting the rejected cached OCR identifies lyric fragments
excluded by the area200/coverage0.1 detector selection. In Welschland includes
`un`, `ran`, `Zi -`, `ver`, `träu`; Tanzlied includes `sin - gén`, `bel`, `ge`,
`Rei -`, `ex. -`, `love -`, `Juch`, `hei`. These are evaluation findings, not a
whitelist of source words to add to inference.

## All-OCR diagnostic control

Using all cached page OCR rows, with identical head matching, routing and literal
scoring, yields **336/480 (70.0%)**, versus baseline 311. There are 434 attachments
versus 392, and 25 lack exact native lyric-note identity versus 15. Thus the
control gains 25 exact targets but also adds ten unanchored-to-native-lyric-note
outputs. This does not certify ten detector false positives: semantic role and
independent pixel ownership are still unmeasured. Extra full-OCR direction text
such as `legato` and `ben legato e cantando` can enter routed physical rows.

| Score | Baseline exact | All-OCR exact |
| --- | ---: | ---: |
| Jagdlied | 46 | 48 |
| Schlummerlied | 57 | 59 |
| In Welschland | 52 | 58 |
| Tanzlied im Mai | 62 | 73 |
| It was a Lover and his Lass | 31 | 31 |
| He doesn't love me | 40 | 41 |
| Rose d'Amour | 15 | 17 |
| À une Fleur | 8 | 9 |

Next test a source-free constrained recovery for detector-rejected OCR fragments
in established lyric rows, retaining non-lyric controls and reporting both gains
and regressions. Do not remove the detector gate wholesale. Then inspect the 41
association abstentions; do not loosen matching just to maximize this calibration
score. Production recommendation remains unchanged; runtime fusion stays off.

Local evidence:

- Baseline instrumented output:
  `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-attachment-trace-20261005/`.
- Missing-target trace, including native measure/onset and candidate boxes:
  `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-missing-attachment-trace-20261005/report.json`.
- Broad diagnostic control:
  `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-all-ocr-attachment-control-20261005/report.json`.

Frozen input hashes checked; no new model decoding or GPU work. Three new trace
tests pass; 22 scoped tests and lint pass. All jobs complete, ETA 0.
