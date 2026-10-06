# Constrained lyric-row recovery diagnostic — 2026-10-05

Reserved fixed-rule validation subsequently gains only one exact target (50→51
of133 frozen-window targets), with none lost. It exposes first-voice and omitted
piano-introduction measure-context assumptions plus incomplete native page windows.
Those prerequisites now take priority; see `docs/C1_FIXED_RULE_VALIDATION_2026-10-05.md`.

The bounded diagnostic recovers **331/480 exact native lyric targets (69.0%)**,
20 above the frozen 311/480 baseline. Every baseline attachment is preserved.
Twenty-one detector-rejected OCR boxes add 21 attachments; all 21 have exact native
lyric-note identity and 20 have exact literal text under the evaluation-only
page-global verse mapping. Outputs without native lyric-note identity remain 15.
No runtime activation or production qualification is claimed.

| Score | Baseline exact | Bounded recovery exact |
| --- | ---: | ---: |
| Jagdlied | 46 | 47 |
| Schlummerlied | 57 | 59 |
| In Welschland | 52 | 58 |
| Tanzlied im Mai | 62 | 70 |
| It was a Lover and his Lass | 31 | 31 |
| He doesn't love me | 40 | 41 |
| Rose d'Amour | 15 | 17 |
| À une Fleur | 8 | 8 |
| Total | 311 | 331 |

## Source-free rule

Only existing detector-selected OCR rows seed recovery. A candidate must:

- Preserve its cached box and text, with some alphabetic content.
- Fit wholly inside the unpadded gap below an assumed vocal staff.
- Overlap every seed member of exactly one existing physical row by at least half
  the larger box height.
- Stay inside the row's horizontal span, extended by four median seed box heights
  on either side. This is an uncalibrated diagnostic parameter, not a production
  default.

Recovered boxes never seed more recovery, create a new physical row or change any
existing staff/row assignment. The first runtime voice remains the vocal-part
assumption. Native notes/text/verse IDs do not enter selection; they are used only
after prediction for evaluation. No direction-word whitelist was used.

The initial unbounded row rule yields 335/480 but admits `legato` far beyond the
established lyric span in In Welschland. Bounded recovery rejects it and the new-row
`ben legato e cantando` control. It also forgoes valid edge fragments, including
`sin - gén` and `sing - ing` in Tanzlied and Bizet's `le,`/`rê`. The all-OCR
diagnostic remains 336/480 but adds ten outputs without native lyric-note identity.
The bound is a conservative response to a calibration control, not proof of
general lyric/direction separation. A direction placed inside an established lyric
row can still pass; existing selected non-lyric text is not removed by this rule.

## Evidence and next steps

Candidate selected rows, accepted/rejected boxes and input hashes:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-established-row-selection-v3-20261005/`.
Candidate native score:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-established-row-score-v2-20261005/report.json`.
Unbounded selection/score are preserved in sibling directories
`c1-established-row-selection-v2-20261005/` and `c1-established-row-score-20261005/`.
The first failed selection attempt produced only an empty output directory;
selected rows include detector metadata, so input equality correctly compares
cached text and box while preserving baseline metadata.

Next qualify this rule on independently selected held-out scans and explicitly
check in-row non-lyric controls. Keep the baseline and candidate separate and
measure lost prior targets as well as gains. Then inspect the 41 baseline
notehead-association abstentions instead of relaxing association on these same
eight calibration pages. Detector training/runtime changes remain premature.

Three new tests enforce bounded/nonchaining recovery, no new rows, staff intrusion
refusal, numeric-only refusal, vocal-staff scope, cached text identity and metadata
preservation. Six focused tests and scoped lint pass. All jobs complete, ETA 0;
fusion remains disabled.
