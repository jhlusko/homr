# Versioned note anchors and written-meter qualification — 2026-10-04

The source-note annotation policy and written-meter guard are implemented and qualified locally. Existing serialization remains the default. All jobs are complete.

## Annotation contract

`legacy-staff-hoist-v1` retains the existing staff-union behavior. Explicit opt-in `note-anchors-v2` preserves each note’s articulation and combined tie/slur field. Vocabulary IDs and recognition channel order do not change.

`write_label_bundle` validates vocabulary and note identities, requires a policy, refuses overwrites, and writes token, notation/timing sidecars and a `.labels.json` manifest. The manifest binds the policy to token SHA-256. Files are staged and the token is exposed last. `read_tokens(..., expected_annotation_policy=...)` rejects unknown manifests, modified token bytes and policy mismatches. Existing unmarked labels are legacy. Published datasets, checkpoint manifests and training configuration have not migrated.

The final110 changed Lieder crops preserve7,653 symbols and6,573 timed symbols through serialization and sidecar reattachment. Legacy export removes all source notation/timing sidecars before checking: combined tie/slur endpoints have zero errors, with no missing/duplicate events or wrong physical measures. Exact raw onset errors remain286. Literal phrase endpoint differences remain578:481 tie-only aliases,30 added tie endpoint types and67 lost phrase multiplicities. The original97-crop check had514 such differences; the extra13 meter-only crops add64. Zero combined-field errors does not establish separate tie/phrase fidelity, concurrent pairing, complete articulation export or model prediction quality.

## Written-meter fix

The first candidate counted repeated denominator declarations, but the actual IMSLP618722-sys4-v1 crop omits its inherited common-time signature. It has only the later2/4 declaration. Its first replay made no change and was interrupted during full-score controls, retained with EXIT130 and an explanation.

The qualified guard also refuses modal-duration rejection when the first signature appears after notes/rests have begun. Earlier bars can belong to an inherited meter and cannot safely disprove the later declaration. Constant incompatible opening meters retain the existing narrow check.

| Check | Result |
| --- | --- |
| Full indexed Lieder replay | 3,867 crops;3,757 frozen-identical,110 changed;189.2s |
| Meter-only changes versus previous replay | 14 crops across8 scores;all retained numerators match source |
| Non-numerator crop strings, original image hashes, timing support | Identical across all3,867;14 native-stretch masks retained;no unseen targets |
| Actual affected-crop runtime exports, same current labels in both arms | All14 incorrect declarations repaired to source meters |
| Non-time XML and event/onset audits on those14 | Exactly unchanged |
| Broad old/new runtime guard controls | 85 OSSQ +212 Lieder;784.8s,one CPU thread |
| Broad output differences/new failures | Zero;the same10 OSSQ detachedLegato failures;75/85 OSSQ and212/212 Lieder export |
| Final compatibility tests | 113 pass;new-file Ruff/Black checks pass |

One of the14 affected crops belongs to the previous97 regrouping changes;13 are newly affected. All110 remain outside metadata-only migration. Whole-score controls alone would miss the crop defect: the inherited opening signature exists in the full source but is omitted in these crop labels. The paired crop runtime check is therefore essential. These are label/export diagnostics, not recognition error rates. The runtime guard changes local code; default annotation serialization remains legacy.

## Remaining onset regression

The reproducible trace of IMSLP592442-sys5-v2 identifies source part3 physical measure21. At quarter-note onset1/2, upper-staff quarter notes last one quarter, but the next lower-staff note starts at onset1. The legacy exporter advances by its minimum member duration (one quarter), while the next onset gap is half a quarter. Another overlap occurs at onset2 before a lower-staff note at5/2. Correct source grouping cannot supply these shorter gaps through the six legacy fields. The former artificial grace offset hid some errors; restoring it would corrupt source grouping.

Exact own-duration supervision preserves the note’s duration, independently of next-onset advance. This work does not enable the experimental advance head, inject source onset targets into inference, shorten printed notes to hide the problem, or change timing to repair meters.

## Local evidence and next work

All final evidence is under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/label-policy-meter-qualification-20261004-v2/`:

- `anchor-bundles-final/report.json`: final110 versioned audit bundles and roundtrips.
- `full-lieder/report.json`, `replay-comparison.json`: original certified crop boundaries and image hashes.
- `crop-meter-gate-release.json`: source-specific written numerator checks.
- `crop-runtime-meter-gate-release.json`: paired actual-label runtime meters and exact non-time/event parity.
- `meter-controls/report.json`, `runtime-meter-gate.json`: all297 old/new score controls and known failure accounting.
- `onset-progression-final.json`: independently computed source gaps versus legacy advances; no inference injection.
- `tests-final.log`, `finalize.EXIT`, `qualified-final-source/`, `qualified-final-source-receipt.json`: tested final code and successful completion.

The initial candidate remains archived in sibling `label-policy-meter-qualification-20261004/`. Earlier unformatted probes are retained; critical parser/exporter/serializer snapshots bind the long replay, and final policy/crop checks bind the formatted code. Qualification, completion watchers and finalizer ran in persistent tmux with one CPU thread. All jobs are complete; remaining ETA zero.

Reproducible tools: `training/onnx/qualify_annotation_policy.py`, `qualify_meter_guard.py`, `qualify_crop_meters.py`, `qualify_crop_runtime_meter.py`, `audit_onset_progression.py`.

Next: qualify separate ties/phrase slurs and articulation ownership through structured export, recover certified OSSQ crop alignment, then wire resumable frozen-core timing-head training on qualified labels. Public corpus/model/provider pins and deployment remain unchanged. No GPU needed for the next label-contract checks.
