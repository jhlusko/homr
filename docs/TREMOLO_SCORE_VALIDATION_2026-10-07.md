# Broader paired-tremolo validation — 2026-10-07

The frozen research rule completed all 13 selected pages: six new scores and seven
cached controls. Every page made zero edits; no native measure regressed, and all
reported event, pitch, rhythm and accidental scores stayed identical. This tests
safe abstention and replay, **not additional repair effectiveness**. The promotion
gate remains open; the accepted runtime is unchanged.

## Frozen selection and source-specific results

Four new Lieder pages were selected from distinct available certified scores with
paired tremolos: IMSLP589008 p008, IMSLP589118 p007, IMSLP634835 p004, IMSLP83318 p006.
OSSQ adds paired-positive sq9631717 and single-note-tremolo negative sq8071278.
Selection used source annotations before inference; proposals used only decoded
symbols and predicted context. These scores are disjoint from the policy-development
score, but recognition/model exposure is tracked: this is not an untouched test set.
Seven cached controls are reported separately.

| Cohort | Pages | Rhythm errors / reference events, both arms | Strict measure/onset errors, both arms |
|---|---:|---:|---:|
| New Lieder | 4 | 154 / 1,388 | 416 / 1,388 |
| New OSSQ | 2 | 67 / 835 | 67 / 835 |
| Cached Lieder controls | 5 | 49 / 1,582 | 172 / 1,582 |
| Cached OSSQ controls | 2 | 6 / 684 | 13 / 684 |

All 13 attempted cases completed. Unmatched reference events remain in denominators.
No changed symbols, changed bars or targeted voice-ID changes occurred. The frozen
policy SHA-256 is `e43b3365d44c4513175d685b92672028933523917a70d85ce3299834091a9baa`.

## Why the rule did not fire

All four new Lieder decoded streams have no meter declarations. The OSSQ paired
positive is also blocked by missing-or-changing target meter. The single-note OSSQ
control has sufficient meter context but fails complete-pair/lane requirements;
leaving it unchanged is the intended negative behavior. Cached pages are blocked
by missing meter, unavailable peer support, or incomplete/ambiguous tremolo lanes.

A subsequent **prediction-only diagnostic** found modal peer-bar lengths:

| Score | Peer part | Modal length in quarter notes | Support / eligible bars |
|---|---:|---:|---:|
| IMSLP589008 | 1 | 3 | 16 / 18 |
| IMSLP589118 | 1 | 3 | 12 / 14 |
| IMSLP634835 | 1 | 4.5 | 10 / 11 |
| IMSLP83318 | 1 | 4 | 8 / 10 |
| sq9631717 | 1–4 | 4 | 24/28, 22/28, 22/28, 21/28 |

These are descriptive whole-page votes, not an inferred time signature or approved
repair context. For any candidate, exclude that bar from support; require matching
predicted bar counts and inspect the serialized timelines. IMSLP589118 has 14 bars
in two parts and 13 in another, so alignment cannot be assumed. A 4.5-quarter span
does not uniquely identify a meter. Never introduce a default 4/4 or consult source
meter to choose an export edit.

## Next priorities

1. Freeze a separate prediction-only context experiment that uses independent
   peer-bar spans where declarations are absent, leaving meter notation unchanged.
   Preserve candidate-excluded votes, bar alignment, full-pair and interval guards.
2. Re-run the six new pages and all cached controls under that separate candidate;
   audit every native measure against ground truth, preserving this fixed-policy
   result. Diagnose incomplete lanes separately rather than deleting guards.
3. Require actual positive repairs plus controls before a flagged runtime patch.
   The Der Zwerg development gain remains encouraging but insufficient alone.

Scripts: `prepare_tremolo_validation_pages.py`, `run_tremolo_validation.py`,
`audit_tremolo_validation.py`, and `audit_tremolo_peer_context.py` under
`training/onnx/`. Legacy cached-window certification is now checked from aligned,
exact, contiguous certified systems rather than requiring a newer root-level flag.

All artifacts are local:
- `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-score-validation-20261007/selection-receipt.json`
- `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-score-validation-20261007/completion.json`
- `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-score-validation-20261007/audit-report.json`
- `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-score-validation-20261007/peer-context-audit.json`
- Per-case `tremolo-rule/planning.json` and `per-native-measure-audit.json`.
