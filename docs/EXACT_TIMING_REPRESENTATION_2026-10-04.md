# Exact timing representation audit — 2026-10-04

Follow-up: `EXACT_TIMING_DATASET_INTEGRATION_2026-10-04.md` records the additive parser/merger/crop/sidecar implementation and its real-source qualification. The isolated prototype findings below remain the audit baseline; the integration uses a new label schema to retain notation-container ownership and visibility.

The remaining tuplet and whole-measure-rest errors require a versioned representation change. Changing `note_5` globally, or substituting dotted notes solely to match elapsed time, would corrupt valid quintuplets or printed duplets. Current runtime, vocabulary, training inputs and published pins remain unchanged.

## Indexed OSSQ/Lieder findings

The audit re-locates every indexed canonical issue from the previous census, verifies reference hashes and source duration, and evaluates exact elapsed-time alternatives in the existing rhythm vocabulary. It includes canonical references whose full parser fails; this is not a new 297-score parser/export qualification.

| Source | Tuplet losses | Existing tokens match duration only | Whole-measure-rest losses | Existing tokens match duration only |
|---|---:|---:|---:|---:|
| OSSQ | 2,031 | 1,871 | 356 | 246 |
| Lieder | 46 | 42 | 132 | 41 |

These issues occur in 21 OSSQ and 22 Lieder scores overall. Tuplets affect 13/8 scores respectively, whole rests 15/18. Overlapping categories are not added as distinct scores. Counts are canonical events, not repeated crop labels or model error rates. No PDMX involved.

None of the alternative tuplet tokens matches all three of the canonical written base, dots and reduced actual:normal ratio. For example, a 2:3 eighth-note duplet lasts 3/4 quarter; current reciprocal rounding emits `note_5` lasting 4/5 quarter. `note_8.` has the right elapsed time but adds a dot and removes the duplet ratio/bracket. The current integer reciprocal scheme cannot retain both meanings.

The three remaining inventory regressions are corroborated at original XML locations: IMSLP598858 physical/printed measure 22, part 1 staff 1 notes 2–3 and part 2 staff 2 notes 13–14; IMSLP622488 physical/printed measure 4, part 2 staff 2 notes 8–9. They are six 2:3 eighth notes, all 3/4 quarter in the reference versus 4/5 quarter in the label. Derived merged-staff indices in earlier diagnostics differ from original XML part/staff indices.

All 246/41 full-rest matches above are six-quarter rests that can numerically match `rest_1.`. That still draws a dotted ordinary rest unless full-measure-rest identity is carried separately. Other missing full-rest lengths include 9/2, 5/2, 8, 5 and 79/16 quarters in OSSQ, and 9/2, 9/4, 5/2 and 11/4 in Lieder. Inferring a generic whole-rest duration from meter alone would also mishandle unusual/incomplete measures.

## Prototype and verification

`training/omr_datasets/exact_timing_labels.py` provides an isolated `exact-timing-proposal-v1` sidecar codec. It keeps rational elapsed quarters, printed type/dots, full-measure-rest status, explicit time-modification XML (including normal-type/dots), and tuplet notation elements separately. The source note XML is retained in the audit output. Serialization normalizes indentation; application refuses tick grids that would round durations.

All 2,565 located events (2,387 OSSQ, 178 Lieder) pass JSON serialization and reconstruction of these selected fields at a different exact tick resolution. Six focused tests cover the dotted-alias distinction, full-rest glyph and arbitrary duration, duplet/nested-bracket metadata, unrelated slur preservation, and rejection of inadequate divisions. Audit runtime: 8.7 seconds on one CPU thread. Persistent tmux and watchers finish EXIT=0.

This is a selected-field sidecar roundtrip, not a complete MusicXML schema/order validation, a production exporter, a new recognition head, or proof that the current merger/trainer preserves sidecars. Container placement, visibility attributes, nested notation ownership, crop alignment and trainer consumption still need integration. The full original note is retained so these fields can be recovered during that work.

The MusicXML specification describes time-modification as the cumulative sounding change, while detailed/nested notation also requires tuplet elements. Normal-type can differ from the printed note type, so the prototype preserves it rather than guessing from the ratio. See [MusicXML time-modification](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/time-modification/) and [normal-type](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/normal-type/).

## Next implementation sequence

1. Design a versioned label contract that carries exact elapsed duration and written notation independently, including ownership/visibility of tuplet containers and full-measure-rest identity. Keep legacy token meanings and old model loading stable.
2. Wire that contract through parser, merging, crop selection and serialization; prove alignment and survival with real OSSQ/Lieder fixtures before changing any published labels.
3. Define how the model predicts the new fields (append-only vocabulary and/or notation heads), then implement checkpoint migration explicitly. A reference sidecar alone cannot fix inference from old models.
4. Repeat full canonical qualification with exact prior controls, then qualify model decoding on OSSQ/Lieder and package new corpus versions. Continue keeping predicted advance disabled until its separate model gate passes.

No GPU needed for steps 1–2. Training/model qualification follows only after the representation contract and dataset changes are tested.

Local artifacts: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/exact-timing-representation-20261004-v3/`: report, all located events, original XML notes, diagnostic sidecars, reference/code hashes, source snapshots and completion receipts. Sibling initial run contains the token-only analysis; v2 retains a failed whitespace-sensitive prototype check, corrected by canonical serialization in v3. All jobs complete; remaining ETA zero.

Tools: `training/onnx/audit_exact_timing_representation.py`, `training/omr_datasets/exact_timing_labels.py`, `tests/test_exact_timing_representation.py`. See `SIMULTANEOUS_REST_FIX_2026-10-03.md` for the preceding qualified rest-preservation fix.
