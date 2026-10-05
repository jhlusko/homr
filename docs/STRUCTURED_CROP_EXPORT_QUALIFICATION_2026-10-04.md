# Structured crop export qualification — 2026-10-04

The bounded source-oracle audit is complete on all110 changed real Lieder crops. It identifies the changes a future structured-ownership contract needs; it does not qualify predicted heads or change the runtime’s ownership policy. Source ties can be preserved exactly, but current phrase-slur export still lets the legacy combined field dominate. Voice/slot identity and boundary fallback also affect pairing independently of endpoint accuracy.

## Protocol and source uncertainty

The versioned note-anchor bundles are hash-checked and reattached to their notation sidecars, then independently aligned to canonical source notes through parser/crop sorting. All6,811 retained source events survive every arm without duplication. Original image/reference hashes are checked. Exact timing sidecars are removed; advance, beam, stem and dynamic structured exports are disabled. Source tie/slur data are enabled only in the oracle arms. Non-span XML (after removing direct ties and notation ties/slurs) is byte-identical in all five arms, preserving pitches, written rhythms, cursor timing, meter, rests, voices and articulations. Existing286 raw onset errors remain a separate known limitation, not corrected by this audit.

Source phrase spans are independently paired two ways: within the source voice, matching the existing extractor’s scope, and within the whole part. These views disagree or encounter duplicate/unmatched identities for51 interior span exposures. They remain explicitly unqualified rather than blamed on malformed source files. The shared391 interior spans have matching identities in both views and no endpoint findings. Crop-boundary spans, extra output pairings and raw source findings remain in the report; matching the391 measures interior-span recall, not complete span precision or boundary fidelity. Technical/fingering content is outside this audit.

## Five-arm results

Counts below are affected notes or missing complete source spans, not symbol-error percentages. All are ground-truth-label export diagnostics.

| Arm | Phrase endpoint mismatch notes | Notational/sounding tie mismatch notes | Missing matches among391 agreed interior spans |
| --- | --- | --- | --- |
| Legacy note-anchor labels, no source sidecars | 578 | 549 /549 | 85 |
| Current structured export with source tie/slur labels | 578 | 0 /0 | 78 |
| Diagnostic structured phrase ownership | 6 | 0 /0 | 19 |
| Diagnostic ownership plus source voice identity | 6 | 0 /0 | 6 |
| Diagnostic identity with unmatched-stop fallback suppressed | 6 | 0 /0 | 0 |

`build_slurs` uses the flat legacy field whenever it is nonempty. That field merges ties and phrase slurs and cannot preserve concurrent endpoint multiplicity. Source structured ties are therefore added correctly while many false/merged phrase endpoints remain. Clearing the flat field only in the diagnostic structured-ownership arms exposes the representation’s separate phrase data.

The existing allocator keys spans by staff/slot. Canonical source slots are independently assigned within voices, so different voices can share a staff/slot. Supplying the actual source voice identity in a diagnostic arm repairs13 additional agreed span matches. Six remaining failures are caused by fallback closing another span when a stop’s own identity is not open, including a crop-boundary stop. An isolated strict-identity arm preserves the orphan endpoint without consuming another open span and recovers all391 agreed matches.

The strict arm still misses3 of the442 voice-scoped interior span exposures; they belong to the51 uncertain exposures. Six raw phrase endpoint mismatches also remain. Five affected crops have clean whole-part numbering but problematic voice-scoped numbering, pointing to the extractor’s scope as a suspect, not proving bad source notation; a sixth has findings under both views. Native-source adjudication is needed before changing the extractor’s scope or targets.

The oracle uses the full six-slot representation. Current heads train two slots, and their voice classes are within-staff identities, not the part-global source voice identities supplied here. Consequently the oracle fix cannot simply be activated as a predicted-head policy or presented as a model improvement. The safe runtime contract must specify ownership, supported fields/classes, unknown/boundary behavior and span identity with backward compatibility.

## Articulation findings

All arms retain the same36 notes with raw source/articulation XML differences:

- 27 have only known glyph aliases: strong-accent to accent, inverted-mordent to trill, or wavy-line markup reduced to a trill.
- 4 have only chord-level arpeggio propagation to an additional note.
- 1 has both an accent alias and arpeggio propagation.
- 4 omit a visible single-stroke tremolo in a later notation block of IMSLP622487 physical measure25 (source part2, notes7/13/19/26). The first notation block contains hidden tuplets; `_collect_articulation` reads only that block and skips the subsequent visible ornaments.

These categories distinguish intentional legacy projection/propagation from actual omitted source content; not every raw XML difference is a new regression. Tremolo recovery also needs its type/stroke count represented: blindly adding the legacy generic token would render three strokes with alternating start/stop behavior instead of the source’s single-stroke `single` tremolo. This audit checks element ownership/counts; it does not establish complete glyph-attribute fidelity.

## Completion and local evidence

Final run:22.0 seconds, one CPU thread, persistent tmux with a completion watcher.120 focused tests pass; Ruff/Black checks pass for the new auditor/tests. Tests cover tie/phrase separation, concurrent spans with identical endpoints but wrong partners, independent voice identities, competing source scopes, crop-boundary stops stealing another span, articulation ownership and non-span timing controls. All jobs finished, EXIT0; remaining ETA zero.

Final source/hash snapshots, report, summary, logs, tests and completion receipts are under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/structured-note-ownership-20261004-v3/`. First three-arm and four-arm results are retained in siblings `structured-note-ownership-20261004/` and `structured-note-ownership-20261004-v2/`. The latter also retains source XML locations and label traces for the six fallback failures, disproving the initial possibility that they were merely co-onset ordering. Reproducible CLI: `training/onnx/audit_structured_crop_export.py`.

Coverage is actual Lieder crops; there is no new OSSQ crop or model-prediction result. Published corpora, checkpoint/provider pins, inference policies and deployment are unchanged. Existing110 changed crops remain outside metadata-only migration.

## Next bounded work and release priorities

1. Verify the six endpoint/scope cases against native source notation; freeze a source span-identity contract instead of unconditionally switching from voice to part numbering. Specify an opt-in runtime ownership contract and explicit field/class availability; preserve legacy behavior for older manifests.
2. Preserve source tremolo type/strokes and visibility across notation blocks in a future label contract; do not widen the parser blindly into a misleading legacy token. Qualify any correction on OSSQ/Lieder before new corpus publication or training.
3. Start full-page reference/score-identity and OSSQ scan-alignment certification, and continue detector-fusion calibration independently. Those remain the main model-release gates; complete notation research or new timing-head training is not an unconditional release prerequisite.
4. Compare final stacks on real pages, select the candidate, qualify final production latency, then pin/package and regenerate public examples. No GPU needed for the immediate contract/reference work.
