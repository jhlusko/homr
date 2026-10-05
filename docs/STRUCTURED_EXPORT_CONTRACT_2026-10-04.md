# Structured export ownership and native evidence — 2026-10-04

This is a proposed opt-in contract, not an activated runtime policy. The accepted stack and default export remain unchanged. Ground-truth sidecars and native IDs are diagnostics, never predicted capabilities.

## Native source check

A reproducible collector preserves the six unresolved phrase endpoint cases from the 110-crop qualification, source/native hashes, raw MusicXML notes and neighboring native chord/slur XML. Exact-measure/pitch candidates exist for all six, but pitch/measure alone does not certify event identity or complete span correspondence.

| Crop/event | Native evidence |
|---|---|
| IMSLP154140, part2 physical measure4 note2 | Grace B-flat6 has native slur start ID33; the following chord has stop ID33. |
| IMSLP435485, part2 measure58 note1 | C-sharp4-containing upper chord has two previous-location slur anchors, both pointing across staff/voice to the prior measure. |
| IMSLP55434, part2 measure61 note11 | Lower G4 has two previous-location anchors, one explicitly pointing to another voice in the prior measure. |
| IMSLP589004, part2 measure36 note7 | Lower chord containing E4 has a native start and next-chord stop. |
| IMSLP89026, part2 measure35 note7 | E4 has a next-location slur with a negative voice offset; the upper E5 has the complementary previous-location anchor. MusicXML serialization orders the upper stop before the lower start. |
| IMSLP89026, part2 measure13 note1 | Upper chord containing D4 has native start and a later same-measure stop. |

These observations support retaining all six endpoints. They do not certify every source span or make all MusicXML number scopes equivalent. The duplicated anchors in IMSLP435485 are preserved as multiplicity evidence, not silently deduplicated or declared engraving errors. A source-voice-only pairing rule is insufficient for cross-voice spans; part-level pairing can also encounter serialized stop-before-start and reused numbers. Native timeline/event mapping is needed before changing extraction globally.

Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/native-slur-evidence-20261004/evidence.json`. Tool: `training/onnx/collect_native_slur_evidence.py`. It refuses replacement, validates the source hash against the prior audit, and does not run recognition or modify labels. Three tests preserve duplicate/cross-voice anchors, alteration/rest distinctions and neighboring-measure uncertainty; the seven prior structured-export tests also pass.

## Proposed opt-in ownership contract

1. **Availability is explicit per field.** Record schema, producer (prediction/reference), model/checkpoint identity, supported classes/slots and label policy. Missing fields are unavailable, not known NONE. Old sidecars and a non-null NoteNotation alone cannot authorize replacing legacy output.
2. **One owner per semantic field.** Separately designate sounding ties, written ties and phrase slurs. Structured ownership suppresses the combined legacy tie/slur channel only when the necessary fields are available together. Otherwise retain the entire legacy behavior or reject the requested opt-in; never partially suppress aliases and lose missing ties/slurs.
3. **Span identity is independent of voice.** An explicit per-span identity connects anchors across voices/staves; placement and voice are properties of anchors. Raw source IDs are reference-only. Current predicted within-staff voice classes and two trained slur slots do not implement this contract. Six representation slots are not six trained slots.
4. **Boundary stops cannot steal open spans.** An unmatched crop-boundary stop is emitted/recorded with an isolated identity or explicitly omitted with a finding. It never closes the last unrelated staff/global span. Report complete interior pairing, boundary behavior, multiplicity and extra spans separately.
5. **Articulation fidelity needs detailed fields.** Preserve tremolo single/start/stop type and stroke count, and notation-block visibility. The four omitted source tremolos are single one-stroke glyphs; the existing generic token's alternating three-stroke export is unsuitable. Preserve glyph aliases/arpeggio propagation as separately declared normalization policies.
6. **Activation requires predicted-output qualification.** Assert no duplicated ties/slurs, note/rhythm/meter/voice controls, and measured OSSQ/Lieder predicted accuracy. Source-oracle success is an export check only. Keep default behavior and rollback until the manifest, producer capabilities and paired evaluations qualify an opt-in.

Implementing this richer span representation or training new heads is a separate improvement lane. It is not an unconditional prerequisite for releasing the existing accepted/candidate stack if its declared current behavior passes the release gates.

## Full-page evaluation input checklist

The existing frozen audit is the starting manifest, not an independent-test claim: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/full-page-reference-audit-20261003-final/audit.json` and neighboring page/score manifests.

- Lieder: 330 scores, 1,926 page entries; 42 identities without known index exposure. Four boundary-certified local candidate pages all belong to Lorelei, already used for historical bar-count evaluation. Verified untouched pages: zero. Classify any further Lorelei comparison as a diagnostic/model-selection result.
- OSSQ: 122 canonical scores and 2,986 original scan pages; zero scores with aligned scanned-system metadata. Synthetic engraving metadata cannot certify original scan windows.
- Before scoring: certify score edition, original page hash, independently obtained reference hash, physical part/staff/measure correspondence, complete page boundaries and score-level training/selection exposure. Missing historical provenance must stay unknown.
- Freeze selection before recognition. Include exclusions/failure denominators and retain source-specific results. Start OSSQ alignment with a small identity-selected scan subset; certify against visible source measures without either model's predictions. Archive signed-off mappings and uncertainty before broad inference.
- Compare accepted September18 core/September24 v7 heads with public-joint-core-heads-20260928 epoch27 under identical final policies. Report pitch, rhythm, accidentals, onset/meter/voice and notation separately. Current page event edits cannot substitute for the agreed Lieder token-SER regression ceiling of +0.25 percentage points.

No additional broad decoding was started: its eligible-input manifest is not ready. No running job or remaining job ETA. The next bounded task is certifying a small OSSQ scan-to-reference page mapping and documenting whether it is diagnostic or genuinely independent; broad-run ETA follows the frozen page count and a measured one-thread pilot. Text-fusion calibration remains an independent release lane.
