# Ordinary onset and annotation audit — 2026-10-04

All 97 remaining changed Lieder crops group ordinary notes/rests at the correct MusicXML source onsets. None mixes distinct source onsets or splits a source simultaneity. The frozen labels have mixed-onset groups in 18 crops and split source simultaneities in 95. This independently qualifies the regrouping; it does not qualify the complete legacy export or justify a corpus/model promotion.

## Protocol

The auditor computes exact rational source onsets independently from divisions, note durations, chord flags, backups and forwards. Source event IDs are carried through parsing, certified crop slicing, natural stripping and token chord sorting in diagnostic copies only. The source and frozen reference hashes are checked. Current crop strings reproduce the prior qualified replay exactly; an isolated former-grace-cursor replay reproduces every one of the 97 original token files, so the export comparison uses the actual frozen labels rather than a guessed baseline.

Both exports use legacy token strings without source notation/timing sidecars or predicted heads. Event IDs are carried onto exported XML notes and matched back to their source events. The audit checks onset within the measure, physical measure relative to the crop, missing/duplicated events and per-note slur start/stop anchors. Printed measure numbers do not substitute for physical measure indexing. Concurrent slur numbering/pairing, full articulation semantics, glyph layout and recognition-model quality are outside this check.

## Paired timing results

Incorrect exported within-measure onsets fall from 2,072 source events with the frozen labels to 286 with corrected grouping. Across crops, 94 improve, two are unchanged and one worsens. The number of crops containing any onset error falls from 95 to 22. These compare raw MusicXML ticks exactly; they are not symbol-error scores or a quantization-tolerant/native-staff-stretch benchmark.

The regression is `IMSLP592442-sys5-v2.tokens`: 25 incorrect onsets become 29. Event-by-event comparison repairs 1,792 onset errors, leaves 280, and introduces six, all in physical measure 21 of this one crop (two old errors in it are also repaired). Correct grouping combines notes at their true source onsets, but the legacy exporter’s inferred rhythmic progression is still wrong in this passage. Do not revert the correct source grouping to hide this exporter limitation, or claim that exact own-duration heads alone resolve gaps to the next onset.

## Paired slur results

Literal phrase-slur endpoint mismatch records are 790 before / 795 after. The legacy field intentionally collapses ties and phrase slurs, so these include known tie-to-slur aliasing; they are not all regressions against the existing six-branch target contract. Eight crops improve, seven worsen, and 82 have unchanged counts. Source anchors can move when serialization unions staff annotations and puts them on the first sorted chord member. Preserving the union is insufficient to preserve the note the slur actually starts or stops on. Existing errors dominate both exports; the regression counts distinguish changes from those existing limitations, but equal counts do not guarantee identical affected endpoints.

The seven crops with increased literal phrase-slur mismatch counts are:

- `IMSLP476183-sys7-v1.tokens`: 16 → 18.
- `IMSLP55434-sys5-v1.tokens`: 12 → 13.
- `IMSLP588852-sys9-v1.tokens`: 8 → 10.
- `IMSLP597597-sys5-v1.tokens`: 18 → 20.
- `IMSLP617260-sys16-v1.tokens`: 0 → 2.
- `IMSLP617260-sys17-v1.tokens`: 5 → 9.
- `IMSLP97202-sys9-v1.tokens`: 13 → 14.

The existing serializer moves annotations within 79 of these crops even when the published versus corrected flat strings do not show an annotation-field difference. The earlier 21 changed-slur-field count was a comparison of the two serializations, not an absolute source-anchor fidelity count.

## Existing contract and source-anchor prototype

The per-note combined tie/slur contract gives a different, complementary result: 820 endpoint mismatches with frozen labels, 866 after corrected grouping, and **zero** with the audit-only note-anchor-preserving serializer. Against frozen labels, the default corrected serialization improves one crop, worsens 18, and leaves 78 counts unchanged; three old endpoint errors are repaired and 49 new ones appear. Source grouping and annotation placement therefore need separate qualification.

The prototype keeps each parsed note’s flat articulation/slur fields on that note through sorting. It does not alter the default serializer, write a new corpus, enable predicted heads, or import source timing/notation sidecars into the export. All 97 exports succeed with no missing/duplicated notes or misplaced measures. Its onset errors remain 286: annotation anchoring does not change rhythmic progression.

Literal phrase-slur differences fall to 514 in the prototype: 420 records are ties aliased to a phrase endpoint, 28 add a tie endpoint type to an existing phrase endpoint, and 66 lose phrase endpoint multiplicity through the flat field’s deduplication. Those remain explicit representation limitations. Zero combined-field errors does not demonstrate faithful separate ties/phrases, concurrent slur pairing, model prediction quality or full notation export.

## Written time signatures

Three crops change numerator-token retention: IMSLP434340-sys18-v2 adds 9; IMSLP580410-sys6-v1 adds 3; IMSLP618722-sys4-v1 removes 2. Source evidence confirms 9/8 and 3/4 for the two added tokens. The removed 2 belongs to a genuine change: physical measures 10–11 are 4/4, and measure 12 is 2/4. The initial common-time token does not supply a separate numerator, so the heuristic sees only the later “2” and mistakes the crop for a constant-meter stream. Source meter evidence and the existing modal-rendered-duration decision are retained per crop. A rendered modal duration is not ground truth for what meter was written. The source numerator must be checked separately from whether legacy rendering fills the bars, rather than treating numerator removal as a proven repair.

## Local artifacts and next work

Final evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lieder-onset-anchor-contract-20261004/`, including per-score source locations, crop grouping records, annotation moves, paired export errors, source snapshots, logs and receipts. Earlier paired checks remain in `lieder-ordinary-onset-final-20261004/`, including event-level timing transitions; earlier exploratory/prototype runs also remain archived. Final `arm-summary.json`, `paired-summary.json` and `prototype-literal-difference-causes.json` separate annotation policies explicitly.

The reproducible tool is `training/onnx/audit_crop_onsets.py`. No published reference, crop, image, model/provider pin, runtime serializer or exporter is changed. This audit is on actual Lieder crops; it makes no OSSQ crop-alignment or model-performance claim. The earlier separate grace audit covers both OSSQ and Lieder.

Keep the 97 outside metadata-only migration. Next, turn the successful audit-only anchor experiment into a versioned, opt-in dataset policy and qualify separate ties/phrase slurs with the structured representation. Check written meter against source evidence and keep source metadata separate from rendering heuristics. Investigate the single timing regression through the exporter’s onset progression. Recover certified OSSQ crop alignment before expanding actual timing-head crop support; then wire resumable frozen-core training on qualified labels.

Final run finishes in 32.8 seconds on one CPU thread in persistent tmux. Audit and test receipts are both zero; 31 focused tests and three subtests pass. Tests cover mixed/split onsets, grace/backup timing, note-anchor relocation, crop-relative measure indexing, staff-specific meter inheritance, tie aliases and the earlier grace/arpeggio/advance guards. The watcher tracks both receipts and finishes. Source snapshots match the tested code. All jobs are complete, remaining ETA zero.


Follow-up: [versioned policy and written-meter qualification](LABEL_POLICY_METER_QUALIFICATION_2026-10-04.md) implements the opt-in note-anchor contract and qualifies the runtime guard. The revised full replay has110 changed crops, including13 additional meter-only changes; all110 remain outside metadata-only migration. The single onset regression is traced to gaps shorter than minimum member duration. The earlier prototype totals in this document remain scoped to its original97 crops.
