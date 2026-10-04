# Exact timing source grid — 2026-10-04

The exact timing label contract now retains the source tick grid, allowing the experimental target builder to distinguish bounded MusicXML rounding from contradictory duration notation. This is an additive dataset change, not a trained model or recognition improvement.

## Contract and compatibility

`homr.exact-timing-label.v3` adds two rational strings: `source_divisions` (ticks per quarter) and `source_duration_ticks` (the original duration). Their ratio must equal the retained raw `elapsed_quarters`. Readers reject inconsistent provenance before attaching any labels. The surrounding `.timing.json` file schema remains v1 because its pairing/hash envelope is unchanged.

Older v2 labels remain readable, including nested tuplet container ownership and visibility. Their missing source grid is not guessed. Strict raw-duration targets can still use them; bounded tick-quantized targets require the new provenance. Legacy token vocabulary, token durations, notation labels, model weights and runtime output are unchanged.

XML reconstruction can use a different divisions value while preserving elapsed duration and written notation. Re-extracting that XML naturally records its new grid. The codec audit therefore compares semantic timing fields separately from source provenance, rather than requiring different source grids to be equal. Reconstruction does not mutate the original payload.

For an explicit 9:4 sixteenth stored as 53 ticks at divisions 480, raw metadata stays 53/480 quarter. The frozen experimental target policy supplies 1/9 quarter only because the intended tick count is non-integral and the stored value is its floor or ceiling. Larger disagreements remain masked; missing source markings are not repaired automatically.

## Qualification

Twenty-three focused tests pass, covering old-label loading, nested container preservation, source-grid rejection before mutation, different-grid reconstruction, serialized 9:4 normalization, decoder alignment and existing checkpoint/head guards. Two-score Lieder preflight preserves both legacy export controls and verifies 14 serialized windows / 383 timed symbols with the frozen train-only target manifest.

The broader suite passes 94 tests, including parser, merger, notation sidecars, legacy/training vocabulary and structured target tests. Selected-field JSON/XML reconstruction passes all 2,565 located events (2,387 OSSQ / 178 Lieder) in 12.6 seconds. This preserves semantic fields across a different exact tick grid, not original source-grid identity or full MusicXML schema/order validation. All jobs finish successfully; remaining ETA zero.

The full 297-reference qualification completes in 1,111.2 seconds (18.5 minutes) on one CPU thread, in persistent tmux with a completion watcher. Both legacy XML/metrics/timing arms reproduce exactly for 75 OSSQ and 212 Lieder scores; all ten known OSSQ detachedLegato export failures reproduce unchanged.

| Source | Verified serialized target windows | Timed symbols | Elapsed supervised / unseen | Written supervised / unseen | Ratio supervised / unseen | Rest flag supervised / unseen |
|---|---:|---:|---:|---:|---:|---:|
| OSSQ | 1,235 | 16,381 | 16,370 / 11 | 15,749 / 0 | 15,738 / 11 | 3,019 / 0 |
| Lieder | 1,137 | 27,426 | 27,426 / 0 | 26,291 / 0 | 26,291 / 0 | 3,434 / 0 |

Twenty-nine selected OSSQ window target occurrences normalize under the bounded rule. Raw sidecars remain unchanged except for the version and added source-grid fields. Eleven unseen elapsed and ratio occurrences are in validation score sq7354505. Unseen classes stay masked; the frozen training-only class lists are not expanded. These selected-window counts are not the earlier whole-score inventory: the latter still has 29 unresolved Lieder events and additional OSSQ validation classes absent from these windows. No unresolved Lieder event is selected here; this does not resolve those source contradictions.

All 2,372 regenerated token windows are byte-identical to retained v2 windows. Their 43,807 timed records retain every old semantic field and gain valid source provenance; all 347 null records preserve alignment. The two unsupported OSSQ windows remain explicit gaps: sq12772795 merged part 1 measure index 214 (`note_72`) and sq8928855 merged part 1 index 20 (`note_19`). No Lieder selected-window gap.

The main scoring process exits successfully. A launcher substitution accidentally appended the target-manifest arguments to its completion receipt; the follow-up guard refused that malformed receipt before running. The original receipt/launcher are retained, the receipt is repaired from its recorded zero exit plus completed report, and final comparisons run separately. This was a completion-wrapper error, not a changed score or discarded model failure.

Local evidence parent: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `exact-timing-source-grid-preflight-20261004/`: bounded preflight and exact control hashes.
- `exact-timing-source-grid-qualification-20261004/`: full controls, serialized windows, source snapshots, target support, run log and watcher receipt.
- `timing-target-inventory-quantized-20261004/target-manifest.json`: unchanged training-only classes and bounded duration policy.
- `exact-timing-representation-source-grid-20261004/`: selected-field reconstruction with source-grid provenance retained in the original payload.

These windows come from canonical scores and the real label writer/reader. They are not a rebuilt image-crop corpus or actual training-crop frequency census. Published OSSQ/Lieder snapshots must explicitly vendor a qualified HOMR commit before new builds can carry this metadata. No public corpus/model/provider pins or deployment change.

## Next work

Investigate the 29 unresolved Lieder events against source notation/images, establish support on actual training crops, and freeze both raw-source and bounded notation-duration scoring views before fitting. Then wire opt-in, resumable frozen-core training for the four new heads. Measure shared-core gradients before choosing joint loss weights; retain pitch/rhythm/accidental reporting and the agreed Lieder +0.25 percentage-point symbol-error ceiling, with OSSQ reported separately. Predicted advance remains disabled.

Background: [target design](EXACT_TIMING_MODEL_TARGETS_2026-10-04.md) and [previous dataset qualification](EXACT_TIMING_DATASET_INTEGRATION_2026-10-04.md).
