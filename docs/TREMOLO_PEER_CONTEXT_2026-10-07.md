# Prediction-only tremolo context experiment — 2026-10-07

Runtime follow-up: four new negative controls pass, and a default-off exporter
implementation matches all 17 unique frozen page exports exactly. See
[TREMOLO_RUNTIME_2026-10-07.md](TREMOLO_RUNTIME_2026-10-07.md).

Two separate frozen replays completed all 13 previously selected pages. Independent
predicted bar spans unblock three Lieder scores; accepting identical complete
repeated meter declarations additionally unblocks OSSQ sq9631717. The final
candidate changes 93 note symbols across 20 measures in four scores. Rhythm errors
decrease by 93 in total, with no native-measure regressions. Pitch, accidentals,
untargeted events, meter notation and voice assignments are unchanged.

This is an **adaptive diagnostic on the inspected cohort**, not an untouched test
or a release decision. The accepted runtime is unchanged. The original zero-edit
batch and both candidates remain separately frozen.

## Results against the accepted stack

| Selected page | Changed notes / measures | Rhythm errors, accepted → candidate | Strict measure/onset errors, accepted → candidate |
|---|---:|---:|---:|
| Lieder IMSLP589008 p008 | 26 / 6 | 42 → 16 / 306 | 68 → 42 / 306 |
| Lieder IMSLP589118 p007 | 3 / 1 | 43 → 40 / 353 | 242 → 242 / 353 |
| Lieder IMSLP634835 p004 | 20 / 2 | 30 → 10 / 357 | 41 → 21 / 357 |
| Lieder IMSLP83318 p006 | 0 / 0 | 39 → 39 / 372 | 65 → 65 / 372 |
| OSSQ sq9631717 | 44 / 11 | 46 → 2 / 380 | 46 → 2 / 380 |
| OSSQ sq8071278, single-note negative | 0 / 0 | 21 → 21 / 455 | 21 → 21 / 455 |

Four new Lieder pages together: rhythm 154/1,388 (11.10%) → 105/1,388 (7.56%);
strict measure/onset 416 → 370. Two new OSSQ pages: rhythm and strict-onset
67/835 (8.02%) → 23/835 (2.75%). These are selected-page results, not whole-corpus
claims. Unmatched source events remain in support; source pages and predictions
are not silently trimmed to agree. IMSLP589118 has substantial existing pitch and
bar-alignment errors; its small rhythm gain does not establish general recognition
quality, and its strict-onset errors stay unchanged.

All seven cached controls make zero edits. No native measure regresses in event,
pitch, rhythm, accidental or strict-onset errors across any of the 13 cases. No
XML voice IDs change. Unsupported lanes, ambiguous chords and incomplete pairs
still abstain. The single-note tremolo negative remains unchanged.

## Context policy and limits

If any meter tokens exist in a target part, peer-span fallback is forbidden.
With complete repeated declarations, the optional second ablation accepts only an
exact repeated sequence of the same numerator/denominator pair. Partial declarations,
reordered pairs and genuinely different signatures remain rejected. sq9631717
contains two identical 2/2 declarations in each decoded part; the old exact-count
check rejected them as “missing-or-changing”, although no actual change occurred.
Its existing independent peer-bar confirmation still applies.

When all target meter tokens are absent, examine other decoded single-staff parts
with the same predicted bar count. Exclude the candidate measure, grace bars and
bars without notes. Only clean serialized single-voice timelines spanning their
whole computed length contribute votes. Require at least three such bars and 75%
modal agreement; qualifying peers must agree on one span. Require a peer's candidate
measure itself to span that length cleanly. A quarter-note span establishes consumed
duration context, **not a numerator/denominator**, so preserve meter notation.

The existing complete paired-tremolo lane, equal-duration chord, exact-fit,
no-overlap, no-overflow, bracket isolation and protected-notation guards remain.
Legacy rhythm/articulation drives decisions; research timing-head values do not.
Freeze proposals and both XML exports before source references enter scoring.
No source meter is read by either context function.

First candidate (peer spans only) makes 49 edits across nine Lieder measures;
OSSQ still abstains. Second candidate adds identical-repeat handling and makes
44 additional OSSQ edits. Candidate iterations and their frozen hashes are recorded
separately; adapting to these diagnostics prevents calling the cohort untouched.

Final policy SHA: `45f85bf66576fcc460cf69dffa45f12641e13e3811aa5337c9fdc89101d7c76e`.
Peer-span module SHA: `1f56734a5114efd4f44747bb6bf3ad9109ee08fcc46c076a8a85681b5d5b9ab0`.

## Remaining release work

1. Freeze this candidate and validate it on additional ordinary and single-note
   tremolo controls not used for these context decisions. Include repeated and
   changing meter, grace notes, mixed voices/chords and incomplete tremolo pairs.
2. Extract the qualified export rule into runtime behind a default-off flag,
   preserving exact replay of the accepted path; parity-test against frozen exports.
   Do not expose trusted-pickle or reference-reading research scripts in production.
3. Keep timing-head model promotion separate: these gains require no new head model.
   Ordinary duplet qualification still needs real ordinary-duplet examples.
   Full-rest metadata and C1 semantic lyric routing remain separate workstreams.

Reproduction uses `training/onnx/replay_tremolo_validation.py` with `--source-run`
pointing to the original completed batch, a new `--run` directory and optional
`--repeat-meter-context`. It saves policy/module/runner snapshots and hashes,
retains all failures and chains the native-measure auditor in persistent tmux.
Focused policy/export/audit tests: 24 pass; scoped Ruff passes.

Local artifacts:
- Original batch: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-score-validation-20261007/`.
- Peer-span-only: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-peer-span-validation-20261007/`.
- Final candidate: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-peer-span-validation-20261007-v2/`.
- Each candidate contains `selection-receipt.json`, policy/module snapshots,
  `completion.json`, `audit-report.json`, per-case frozen XML/proposals and native
  measure audits. All replay/audit jobs have finished.
