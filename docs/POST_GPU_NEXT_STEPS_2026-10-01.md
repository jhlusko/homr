# Local follow-up after the 2026-09-29 random-start experiment

## Current release priorities — 2026-10-04

This order supersedes the historical status table and earlier running-job entries below. The accepted September18 core plus September24 v7 heads remains the local comparison/rollback stack; it is not asserted to equal the deployed provider pin. Public-joint-core-heads-20260928 epoch27 is the leading qualification candidate: the completed paired real-source crop comparison improves symbol error by1.455pp on OSSQ and0.418pp on Lieder. Corrected ONNX/runtime parity passes, but functional page smoke and crop scores do not establish full-page accuracy or promotion. One matched Lieder page improves pitch while worsening rhythm.

1. **Finish the current label/export checks locally.** Independently measure separate tied/sounding-tie elements, phrase-slur endpoints and span pairing, and articulation ownership against source XML. Preserve note/timing controls and report crop-boundary spans and unsupported source notation explicitly. The110 changed Lieder crops remain outside metadata-only migration. This bounded correctness work must not turn into an unconditional requirement to train new timing heads before release.
2. **Certify full-page evaluation inputs and compare complete stacks.** Audit score identities against training/model-selection history; recover trustworthy OSSQ scan-to-source alignment and page boundaries. Compare accepted and epoch27 stacks under identical detector/decoder/post-decode policies, with pitch/rhythm/accidentals, onsets/meter/voices, failed pages and source-specific results. Keep the authorized Lieder symbol-error ceiling of+0.25pp and report OSSQ separately. D1 still needs OSSQ real-page verification and provider identity/pin integration. Revisit ordinary-group beam substitution separately; the rest-spanning hybrid remains rejected.
3. **Continue text fusion independently of model selection.** Freeze detector postprocessing after the component-area/recall sweep; verify training exclusions and partially labelled region handling; fit fusion calibration and evaluate its held-out partition. Lyrics use e2, directions use e4, and review-only dynamics share e4 physical inference. C1 must establish acceptable transcription, verse ordering and note attachment. Checkbox/fusion remain disabled pending those gates.
4. **Choose and qualify the release stack.** Select epoch27 only if complete-stack evidence passes; retain the accepted stack if it does not. Measure the final production GPU/HTTP/OCR/attachment path, including cold overhead and failure handling, against the280s per-page timeout. The earlier warm22-page benchmark does not qualify the changed final path. Prepare candidate manifests, immutable pins and rollback; activation remains a separate release decision.
5. **Package and present the final release.** Regenerate E3 examples against the selected models, update the OurTextScores/ots-homr page with supported public-reproducible advantages and limitations, publish reproducibility instructions and qualified corpus updates as needed, then complete appropriate repository consolidation and release review.

**Separate improvement lane:** finish certified timing-label support, raw/normalized/native-staff scoring policy and an opt-in resumable frozen-core trainer before renting GPU for the four additional timing heads. They need not block a release whose existing stack passes. Any later joint loss weighting must use measured shared-core gradients and preserve pitch/rhythm/accidental gates. Random/ImageNet-start training remains paused because its decoded recognition underperformed; no automatic resume or new multi-day run is scheduled.

All prior qualification jobs are complete. Immediate steps1–3 are local work and do not require a GPU rental. A full-page evaluation manifest and calibration subset must be frozen before giving trustworthy broad-run ETAs. See `homr/docs/LABEL_POLICY_METER_QUALIFICATION_2026-10-04.md` and the dated runtime/text-fusion reports below for completed evidence. No deployment, pin promotion, corpus upload or training is authorized by recording this order.

### 2026-10-04 — bounded structured-export audit complete; release priorities retained

Five source-oracle arms on110 real Lieder crops retain all6,811 source events and exact non-span XML. Structured ties match both source `<tied>` and `<tie>` fields exactly; phrase endpoints still have578 mismatched notes under current export because the combined legacy field wins. Diagnostic structured phrase ownership reduces that to6; source voice identity repairs13 additional agreed spans, and suppressing unmatched-stop fallback repairs the final6. All391 interior spans agreed by independent voice/part source views match in the strict diagnostic arm.51 other interior span exposures and6 endpoint mismatches remain unqualified; differing source scope does not prove bad source notation.

Articulations:36 raw XML differences partition into27 glyph-alias-only,4 arpeggio-propagation-only,1 mixed,4 visible single-stroke tremolos omitted from later notation blocks. Source tremolo type/strokes need preservation before a parser fix; the generic legacy token would misrender them. Oracle identity/full6 slots are not the currently predicted within-staff voice/two-slot capability. No default inference/export policy changed and no model advantage is claimed.

120 focused tests and new-file Ruff/Black pass. Final audit22.0s, one CPU thread, persistent tmux/watchers, EXIT0; all jobs complete, ETA zero. Local write-up: `/home/jhlusko/workspace/homr/docs/STRUCTURED_CROP_EXPORT_QUALIFICATION_2026-10-04.md`. Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/structured-note-ownership-20261004-v3/` (report, summary, tests, logs, source snapshots and receipt). Earlier candidate analyses/source locations stay in sibling first-pass/v2 directories.

Next bounded label work: native adjudication of source span scope, explicit opt-in ownership/identity/field-availability contract, detailed tremolo labels. Independently proceed with full-page reference identity/OSSQ scan certification and text-fusion calibration. Full-page accuracy → final-stack choice → production latency → pins/package/examples remains the release sequence. New timing-head training and comprehensive notation redesign are not unconditional release prerequisites. No GPU/new training, corpus upload, pin change, push or deployment.

### 2026-10-04 — versioned note anchors and meter guard qualified

Opt-in `note-anchors-v2` bundles preserve per-note articulation/combined tie-slur fields and carry hash-bound policy manifests plus notation/timing sidecars. Existing/default/unmarked labels remain `legacy-staff-hoist-v1`; vocabulary IDs and channel order are unchanged. All110 changed Lieder crops round-trip7,653 symbols/6,573 timed symbols, with zero combined endpoint errors and no missing/duplicated/misplaced events in legacy export without source sidecars. Remaining286 onset errors and578 literal phrase differences (481 tie-only aliases,30 added tie endpoint types,67 lost phrase multiplicities) are explicit representation limits, not model recognition scores or complete structured-slur fidelity.

Revised meter guard protects repeated declarations and a first declaration after notes/rests have begun, including crops omitting their inherited opening signature. Full3,867-crop replay restores14 numerator tokens across8 scores; all match source meters and all non-numerator strings/image hashes/timing support remain unchanged. Of these14,13 are newly affected:3,757 frozen-identical crops and110 changed crops. Keep all110 outside metadata-only migration; retain the14 native-stretch timing masks.

Runtime check on the actual14 crops repairs all14 exported meters, with byte-identical non-time XML and identical event/onset audits. Broad paired runtime guard check on85 OSSQ/212 Lieder finishes in13.1 minutes with exact unchanged outputs and the same10 known OSSQ detachedLegato failures; no new failures or regressions. Coverage75/85 OSSQ,212/212 Lieder. Whole-score controls do not substitute for the affected-crop check.113 focused compatibility tests, new-file Ruff and Black checks pass. Persistent qualification/watchers/finalizer completed successfully; no heavy job remains, ETA zero.

The one prior timing regression is explained: source part3 physical measure21 of IMSLP592442 contains interleaved staff notes whose next onset gap is shorter than the minimum member duration. Legacy advancement overshoots. Preserve correct grouping; own-duration heads alone cannot encode the missing gap. Predicted advance remains off and source timing is not injected into inference.

Write-up: `/home/jhlusko/workspace/homr/docs/LABEL_POLICY_METER_QUALIFICATION_2026-10-04.md`. All final evidence is local under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/label-policy-meter-qualification-20261004-v2/`: anchor-bundles-final/report.json, crop-meter-gate-release.json, crop-runtime-meter-gate-release.json, runtime-meter-gate.json, full-lieder/report.json, onset-progression-final.json, tests-final.log, finalize.EXIT and qualified-final-source snapshots/receipt. First failed candidate remains archived in sibling label-policy-meter-qualification-20261004 with interruption EXIT130; it is not a passed gate.

Next: independently qualify separate ties/phrase slurs and articulation ownership in structured export; recover certified OSSQ crop alignment/support; then wire resumable frozen-core timing-head training using qualified labels. No GPU, corpus/model/provider pin change, upload, push or deployment in this work.


## Current local follow-up — 2026-10-01

### 2026-10-04 — remaining Lieder onset groups qualified; source-anchor prototype succeeds

Independently traced all 97 changed real Lieder crops to rational MusicXML onsets. Current grouping mixes/splits no source onset; frozen labels mix onsets in 18 crops and split source simultaneities in 95. All original/corrected crop strings replay exactly, all three export arms retain every source event in its correct measure without duplication. This qualifies grouping, not complete XML/model promotion.

Paired legacy-only export: incorrect within-measure onsets 2,072 → 286; 94 crops improve, two unchanged, one worsens (IMSLP592442-sys5-v2, 25 → 29). Event transitions: 1,792 repaired, 280 remain, six newly wrong, all in that crop’s physical measure21. Those comparisons use literal raw source ticks, not a normalized staff-stretch view or model SER.

Default serializer hoisting regresses the existing combined tie/slur field: per-note endpoint mismatches 820 → 866 (one crop improves, 18 worsen, 78 unchanged). Audit-only per-note annotation preservation reduces these to zero across all97, with timing unchanged. Literal phrase-slur mismatches are separately 790 frozen /795 corrected /514 prototype; the prototype’s remaining514 are420 tie-only aliases,28 added tie endpoint types,66 lost phrase endpoint multiplicities. No true-phrase fidelity, concurrent pairing or predicted-head/model advantage claim.

The removed numerator2 in IMSLP618722-sys4-v1 is a real 4/4→2/4 change (source physical measures10–11 vs12). Initial common time has no separate numerator, so the single-numerator/modal-duration heuristic discards the later genuine meter. Added9 in IMSLP434340-sys18-v2 and3 in IMSLP580410-sys6-v1 agree with source9/8 and3/4. Dataset cleanup and runtime meter heuristics both need a qualified change; restoring a token alone is insufficient.

No default parser/serializer/exporter, published corpus/model/provider pin, training job or deployment changed. Keep the97 outside metadata-only migration. Next: versioned opt-in source-anchor dataset policy; common-time/meter-change guard qualification; the one rhythmic-progression regression; certified OSSQ crop alignment/support; resumable frozen-core timing-head trainer on qualified labels. No GPU needed for immediate work.

Write-up: /home/jhlusko/workspace/homr/docs/ORDINARY_ONSET_CROP_AUDIT_2026-10-04.md. Final evidence: /home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lieder-onset-anchor-contract-20261004/. Earlier event transitions: sibling lieder-ordinary-onset-final-20261004/paired-event-transitions.json. Reproducible CLI: training/onnx/audit_crop_onsets.py --anchor-prototype. Final run32.8s, one CPU thread, persistent tmux/watchers; audit/tests EXIT=0;31 tests and3 subtests pass. All jobs complete; ETA zero.


### 2026-10-04 — grace serialization repaired and qualified on OSSQ/Lieder

Controlled old-cursor ablation reproduces all 167 changed Lieder crops byte-for-byte, isolating the earlier grace-cursor correction as the cause. Fixed two exposed regressions: consecutive zero-duration graces were merged/reordered as a chord, and principal-chord arpeggios leaked into grace groups. Source grace order/chord membership now passes across 85 OSSQ / 212 Lieder references, 6,728 / 1,156 visible pitched grace notes and 46 explicit grace chords, with zero mismatches.

Final actual crop replay: 3,867 Lieder crops, 3,770 byte-identical (3,473 train / 297 valid), 97 remaining differences (94 / 3). Seventy regain parity; no new differences among previously identical crops. All note/rest identity inventories, original image hashes, certified boundaries and splits remain unchanged. Remaining categories: 45 group-boundary-only, 37 note-order changes with the same inventory, 15 same-order/other-field changes. Twenty-one crops change slur attribution, one accent/tenuto attribution, three control inventories; counts overlap. Their ordinary onset/export/numerator semantics remain unqualified, so keep them outside metadata-only migration. Fourteen native-stretch elapsed/ratio masks remain; no unseen mixed-source timing classes.

47 focused tests and three subtests pass. Full replay 386.4s; source check 301.3s, sequential one-CPU-thread persistent tmux/watchers. All five final receipts EXIT=0; all jobs complete, ETA zero. No published corpus/model/provider pin or runtime inference change. Next: independently qualify the 97 ordinary-onset/slur/numerator differences; recover certified OSSQ crop alignment/support; explicit stretch scoring policy; resumable frozen-core exact-timing head trainer.

Write-up: /home/jhlusko/workspace/homr/docs/GRACE_SERIALIZATION_AUDIT_2026-10-04.md. Final evidence: /home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lieder-grace-and-arpeggio-qualification-20261004/. Ablation: sibling lieder-grace-cursor-ablation-20261004/ (167 crops/71 scores, 22.4s), with pre-arpeggio-scope source snapshots. Failed intermediate diagnostic receipts are retained; final source clock handling accounts for valid voice backups. Existing original-scan viewer stays at http://localhost:8765/; no per-case decisions requested.


### 2026-10-04 — native rest stretch explained; full Lieder crop replay finds a migration gate

All 29 duration conflicts are rests whose raw XML duration × explicit native staff stretch equals the written value. Fourteen reach training in three IMSLP624193 crops; the other 15 are outside accepted crop boundaries. Original scans agree with the written rests. Keep elapsed/ratio masks until staff-time/export semantics are qualified, rather than inventing dots/tuplets or excluding entire scores.

Full actual Lieder replay finishes in 358.6 seconds on one CPU thread: 212 scores, 3,867 crops. 3,700 are byte-identical (3,409 train / 291 valid); 167 change serialization (158 / 9). They preserve the note/rest identity inventory, but include 47 group-boundary changes, 95 order changes and 25 other changes with the same note order. Musical equivalence is not established; keep changed crops separate. The identical training subset supplies 128,120 timed crop occurrences and supports every target value in identical validation, with 14 elapsed/ratio masks and no unseen mixed-source classes.

Next: controlled parser/merger/serializer audit of the 167 changes, especially successive grace notes; recover OSSQ crop alignment/support; explicit scoring views; resumable frozen-core head training. [Write-up and evidence](LIEDER_REST_STRETCH_AND_CROP_SUPPORT_2026-10-04.md). Twenty-eight focused tests pass. All audits finished, ETA zero; the original-scan review remains at http://localhost:8765/ in persistent tmux.

### 2026-10-04 — source-grid timing labels and normalized targets qualified

Timing label v3 retains original divisions and duration ticks, with positive/consistent provenance checks before attachment and backward reading of v2 labels. Missing older grids are not guessed. The full 297-reference control check takes 18.5 minutes on one CPU thread: 75 OSSQ / 212 Lieder exact controls, ten unchanged OSSQ export failures. All 2,372 selected token windows remain byte-identical; 43,807 timing records retain their old semantics. Normalized target construction and decoder shifts pass; 29 selected OSSQ target occurrences normalize, eleven elapsed and eleven ratio targets remain unseen/masked. Two unsupported legacy OSSQ windows remain explicit gaps.

Next: audit the 29 unresolved Lieder source events, establish actual training-crop support and freeze raw-source/bounded-duration scoring views, then wire opt-in resumable frozen-core head training. This is dataset/target qualification, not a trained-model or recognition improvement. See [source-grid evidence](EXACT_TIMING_SOURCE_GRID_2026-10-04.md). Published pins and runtime remain unchanged; predicted advance stays off.

All 2,565 located issue events also pass selected-field reconstruction at a different exact tick grid; 94 related tests pass. Final comparisons complete successfully. A completion-receipt launcher bug is repaired with originals retained; scoring itself completed normally. All jobs finished, remaining ETA zero.

### 2026-10-04 — exact timing targets, source quantization and additive checkpoint prototype

Implemented four experimental output-only heads: elapsed note/rest duration, written type/dots, cumulative sounding ratio and full-measure-rest flag. These are independent of advance-to-next-onset and printed rareNum labels. Targets mask BOS/EOS/padding/grace/missing/unseen values; rest identity is supervised only on rests. Existing forward/recognition/notation objectives and runtime remain unchanged.

Train-only canonical inventory covers 76/9 OSSQ train/valid scores (928,090/97,914 events) and 199/13 Lieder (256,383/18,269). Strict raw ticks yield 52/21/18/2 classes. Bounded floor/ceil normalization of explicit ratio durations yields 51/21/22/2, classifying all 446 OSSQ conflicts as source tick quantization, including nested 9:4 tuplets stored as 53/480 rather than 1/9 quarter. It masks 29 unresolved Lieder elapsed/ratio targets (IMSLP622105 3, IMSLP624193 21, IMSLP632172 5). Whole-source frequencies are not actual training-crop support.

Mixed training classes cover all Lieder validation values; OSSQ retains 15 unseen elapsed, one written and 23 ratio targets, explicitly masked/reported without adding validation classes. The normalized policy requires source divisions, absent from current v2 sidecars; sequence target building refuses that missing provenance. Next dataset step: a versioned source-grid field and requalification, not a training launch from old sidecars.

An explicitly untrained separate addon binds the accepted Sept18 core checksum and the exact target contract; 512 feature dimensions from accepted v7 ONNX, 49,248 parameters. Saved weights reload/shape-check, base file remains checksum-identical, and inference loading refuses untrained heads. Twenty focused tests pass, including core/logit/RNG invariance in a small fixture, wrong-base/class-order guards, head-only gradients, source masking and decoder alignment. No real decoder or training gate ran.

Loss weights must be explicit; the new heads do not silently change recognition + notation/12. Suggested sequence: source-grid metadata and crop support; opt-in frozen-core head warm-up with resumable auxiliary optimizer; per-group shared-core gradient measurement before selecting joint weights; OSSQ/Lieder prediction-only XML qualification, including nested bracket treatment. Retain agreed Lieder +0.25 pp symbol-error ceiling and separate pitch/rhythm/accidental and OSSQ reporting. No GPU needed for the next metadata/trainer wiring work.

Write-up: `/home/jhlusko/workspace/homr/docs/EXACT_TIMING_MODEL_TARGETS_2026-10-04.md`. Local evidence under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-target-inventory-20261004/`, sibling `timing-target-inventory-quantized-20261004/` and `timing-addon-init-20261004/`, with class manifests, source/split hashes, located source examples, initialized addon and tested snapshots. Inventories finish in 74.5/142.9 seconds on one CPU thread with persistent tmux/watchers, EXIT=0. All jobs finished; ETA zero. No published corpus/model/provider pin, deployment or website advantage claim changed; predicted advance remains off.


### 2026-10-04 — exact timing dataset integration qualified with explicit gaps

Current HOMR MusicXML parsing attaches exact reference timing outside legacy token identity/duration. Staff merging, crop selection and the shared notation writer/reader preserve it in token-hash-guarded `.timing.json` files, using serialized symbol order. Label schema v2 retains tuplet notation-container ownership and visibility. Existing vocabulary, legacy token/notation targets, runtime export and checkpoint shapes are unchanged; training losses do not consume the new metadata.

Full 297-reference check completes in 733.0 seconds on one CPU thread, persistent tmux/watchers EXIT=0. Both prior XML/metrics/timing controls replay exactly for 75 OSSQ and 212 Lieder scores; all ten OSSQ detachedLegato export failures reproduce unchanged. Selected windows verify OSSQ 1,235 / 16,381 timed symbols, Lieder 1,137 / 27,426. Two OSSQ windows are explicit unsupported legacy-token gaps: sq12772795 merged part 1 index 214 (note_72), sq8928855 part 1 index 20 (note_19). Lieder has no selected-window gap. This qualifies additive metadata for tested serializable windows, not a full image/corpus rebuild or model improvement.

The updated v2 contract additionally reconstructs all 2,565 located canonical problem events, including tuplet container ownership/visibility, in 9.1 seconds. Forty-eight focused tests and three subtests pass. Three-score Lieder preflight verifies 18 windows / 499 timed symbols. All jobs complete; remaining ETA zero.

Write-up: `/home/jhlusko/workspace/homr/docs/EXACT_TIMING_DATASET_INTEGRATION_2026-10-04.md`. Local artifacts under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/exact-timing-pipeline-qualification-20261004-v2/`, sibling `exact-timing-pipeline-preflight-20261004/` and `exact-timing-representation-20261004-v4/`, with tested snapshots, source/reference/control hashes, label windows, gaps and completion receipts. Initial failed checker run is retained separately.

Next: derive model targets for exact elapsed duration and written notation, including rare/unrepresentable legacy rhythms, and define explicit checkpoint/runtime migration. Reference sidecars must not be injected into production decoding. Then train/qualify on OSSQ/Lieder and publish new corpus versions. Public OSSQ/Lieder vendored snapshots remain unchanged; future builds must explicitly update that pin and retain timing files in packaging. No published corpus/model/provider pin or deployment changes. Predicted advance remains disabled. Target-design work can proceed locally without GPU.


### 2026-10-04 — exact timing representation audit and isolated prototype

Indexed canonical OSSQ/Lieder issues re-located with source hashes: OSSQ 2,031 tuplet losses and 356 whole-rest losses; Lieder 46 and 132. Existing tokens match duration for 1,871/42 tuplets and 246/41 full rests, but no tuplet alternative preserves written base, dots and ratio together. All full-rest alternatives are six-quarter dotted-whole aliases that lose full-measure-rest identity. Six located duplet notes corroborate the three remaining inventory regressions. No blanket note_5 or rest_1 rewrite.

An isolated versioned exact-timing sidecar codec preserves rational elapsed duration, written type/dots, full-rest status, time-modification and tuplet elements. Selected-field JSON/reconstruction checks pass all 2,565 located events; six tests pass. It is not consumed by current training/runtime and is not a complete XML exporter or model gate. Full source notes are retained. One CPU thread, persistent tmux/watchers, EXIT=0; runtime 8.7 seconds, all jobs finished.

Write-up: `/home/jhlusko/workspace/homr/docs/EXACT_TIMING_REPRESENTATION_2026-10-04.md`. Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/exact-timing-representation-20261004-v3/`. Original token-only analysis and failed whitespace-sensitive v2 probe are retained separately. Next: define and integrate the versioned label contract through parser/merger/crop serialization, retaining tuplet ownership/visibility and checkpoint compatibility; then model-prediction migration, full OSSQ/Lieder qualification and new corpus versions. No GPU needed for the label contract work. Predicted advance remains off; existing corpus/model/provider pins unchanged.


### 2026-10-03 — simultaneous-rest preservation qualified

The exporter preserves co-onset duplicate rests using explicit backups and separate voices. Full 297-reference qualification finishes in 17.8 minutes on one CPU thread; exact prior controls and all ten OSSQ failures replay. Coverage remains 75/85 OSSQ and 212/212 Lieder. Restored rests: OSSQ 3 across 3 scores; Lieder 117 across 29 scores. Event edits improve 8,351 → 8,348 and 13,578 → 13,461; rhythm improves by the same amounts, pitch/accidental unchanged. Correct inventories recover in 3 OSSQ and 108 Lieder measures. No new failures, correct-inventory losses, common-onset or per-score edit regressions.

Cached accepted/candidate Lorelei outputs remain unchanged; this previously evaluated page has no affected predicted groups and is not a generalization gate. Seventy-seven focused tests and three subtests pass. Both persistent jobs/watchers complete EXIT=0; remaining ETA zero.

Write-up: `/home/jhlusko/workspace/homr/docs/SIMULTANEOUS_REST_FIX_2026-10-03.md`. Local artifacts under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/simultaneous-rest-qualification-20261003/`, sibling `simultaneous-rest-preflight-20261003/` and `cached-page-rests-20261003/`, including reports, controls, XML, source/hash and completion receipts. These are reference roundtrip diagnostics, not recognition rates.

Next: exact future-build tuplet/ratio and whole-measure-rest representation audit against OSSQ/Lieder canonical XML. The three duplet inventory regressions remain; duplicate-rest blocker is resolved. No blanket rhythm-token rewrite, corpus publication, model/provider pin change or deployment. Predicted advance remains disabled; broader model checks, text-fusion/C1 and untouched-page references remain separate work. No GPU needed for the next audit.


### 2026-10-03 — grace/ending-marker preservation correction qualified

The label merger emits ending markers separately from note chords. Eighty-two focused tests pass, including all three ending types and advance-target ownership. The full 297-reference OSSQ/Lieder comparison completes in 17.6 minutes on one CPU thread with exact prior controls and unchanged failure coverage. Nineteen grace notes are restored across three OSSQ scores, aggregate event edits improve 8,370 → 8,351, and three canonical inventories recover. Lieder is unchanged. No new inventory loss, common-onset regression or per-score event/pitch/rhythm/accidental edit regression. See `GRACE_ENDING_FIX_2026-10-03.md` for local results. All jobs complete, no remaining ETA.

Next: preserve simultaneous duplicate rests with explicit voice/backup semantics and qualify that correction; resolve exact future-build tuplet/ratio representation before publishing new corpus versions. This narrow preservation fix does not qualify the combined label changes or predicted advance, which remains disabled. Published corpora/model/provider pins are unchanged; text-fusion/C1 and new full-page references remain independent work.

### 2026-10-03 — timing qualification complete; predicted advance remains off

Five-arm reference qualification completes in 22.2 minutes on one CPU thread, with exact archived controls and unchanged failure coverage. Matched old/corrected advance onsets improve OSSQ 412 → 354 and Lieder 5,248 → 2,169, with zero worse exact-count measures. Overall event edits change 8,330 → 8,370 and 14,051 → 13,578. Four Lieder inventories regress: three expose existing duplet-label rounding, one collapses co-onset duplicate rests. OSSQ grace notes can now collide with a voltaStart marker and disappear in already lossy measures. These are explicit blockers to a clean qualification.

Cached accepted/candidate Lorelei advance A/B also completes: accepted eligible onsets 0 → 1 moved event; candidate 0 → 0, event/pitch/rhythm/accidental edits unchanged. Keep predicted advance disabled. See `TIMING_FIX_QUALIFICATION_2026-10-03.md` for counts, located cases, local paths and next steps. All jobs finished, no remaining ETA. Next: protect grace notes in ending-marker groups, test/requalify real sources, resolve rest/voice preservation and future-build tuplet representation before publishing new labels or training. Text-fusion/C1 and new full-page references remain open.

### 2026-10-03 — timing corrections under qualification

Local fixes preserve the parser cursor for grace notes and include every printed note/rest duration when selecting XML divisions, while keeping meter inference based on consumed group time. Seventy focused tests pass, including a mixed-duration triplet regression and grace sequence/chord cursor coverage. No published corpus or model pins changed.

Five-arm OSSQ/Lieder qualification runs on one CPU thread in tmux `real_timing_fixes_20261003`, with watcher `real_timing_fixes_watch_20261003`. It exactly replays archived disabled/enabled controls, then evaluates precision-only enabled output and both fixes with advance disabled/enabled. Results are local at `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/real-timing-fixes-20261003-v2/`. Read `watch-status.json`, `run.log`, `scores.jsonl`, and eventually `EXIT`/`report.json`. An initial input-format failure is retained in sibling `real-timing-fixes-20261003/`. Qualification is pending; do not infer promotion or regenerate published corpora from these local source changes.

### 2026-10-03 — reference advance A/B complete

Reference advance reduces moved-onset events 470 → 412 on OSSQ and 15,855 → 5,252 on Lieder, on common inventory-preserved measures. No newly lost correct inventories. There are 39 Lieder exact-count regressions; the count does not measure distance. A grace-cursor probe removes four genuinely worse cases in two scores. A separate synthetic probe exposes zero-duration rounding when advance changes XML divisions. See `REFERENCE_ADVANCE_ABLATION_2026-10-03.md`. All jobs complete on one CPU thread; no remaining ETA.

Next: narrowly tested printed-duration division allocation and future-build grace cursor corrections; repeat reference qualification before evaluating predicted advance. No model promotion or published pin change. Text-fusion/C1 and new full-page references remain open.

### 2026-10-03 — OSSQ/Lieder timing census complete

No PDMX was read. All 39,137 OSSQ and 3,867 published-index Lieder labels were available. Among canonical references for indexed score identities (85 OSSQ / 212 Lieder), the current formula loses full-rest timing on 356 / 132 events and non-integral reciprocal-tuplet timing on 2,031 / 46 consistent-source events. These categories affect 21 / 22 score identities and are 0.233% / 0.065% of printed timed events; full canonical score events are not interchangeable with retained crop exposures. Relevant historical OSSQ parser functions match current source. See `REAL_TIMING_LABEL_CENSUS_2026-10-03.md` for precise definitions, larger-collection counts and local artifacts.

Whole-score reference roundtrips complete in 809.8 seconds on one CPU thread. Published-score success coverage is 75/85 OSSQ and 212/212 Lieder; ten OSSQ exports reject detachedLegato. With advance output disabled, onset changes occur in inventory-preserved measures in 30/75 successful OSSQ scores and 160/212 Lieder scores. These fixed-staff/measure diagnostics are not voice-aware equivalence or model error rates. Eleven tests pass; both tmux jobs complete, no remaining ETA. Next: bound a ground-truth advance A/B to locate lost grouping timing; inspect representative real-source rests/duplets and make narrowly tested future-build representation corrections before changing corpus pins or training. Text-fusion/C1 remain independent work.

### 2026-10-03 — cached timing ablation complete

Exact XML replay passes for both stacks. Late tuplet/beam/tie/slur-crossing repairs do not change pitch/rhythm edits on Lorelei. The canonical-reference label roundtrip reproduces the five shared vocal duration errors: six-quarter whole-bar rests collapse to `rest_1`, and 2:3 eighth-note duplets round to `note_5`. Piano ordering changes also show that ordered edits can penalize unchanged event inventory. No runtime or published label policy changed. See `PAGE_TIMING_LABEL_AUDIT_2026-10-03.md` for local evidence, limitations and the `rest_0` duration-unit warning.

Next: measure these representation problems across frozen labels and canonical references, establish historical parser provenance, then implement a narrowly tested future-build correction and strengthen onset/meter-aware page scoring. Do not enable a blanket sole-rest rewrite or retrain before that audit. All jobs complete on one CPU thread; six tests pass, no remaining ETA. Full-page untouched references/OSSQ alignment and text-fusion/C1 still remain open.

### 2026-10-03 — reference audit and paired original-page diagnostic complete

See `FULL_PAGE_DIAGNOSTIC_2026-10-03.md` for the current result and local reproduction paths. The reference audit finds no verified untouched locally scoreable pages: four certified Lieder pages are from Lorelei, explicitly previously bar-count evaluated; OSSQ scan alignment is missing. A deterministically selected Lorelei page was scored with matched accepted/candidate float32 exports. Candidate pitch edit error improves 1.955% → 1.397%, while rhythm/event edits worsen 3.599% → 4.370%. Accidental projection differences are deletions, not substitutions. This one-page diagnostic does not apply the crop-token Lieder regression gate or justify promotion.

Both stacks complete in about 118 CPU seconds on one thread. All tmux jobs are complete. Next local task: cached-symbol/XML-only ablations of shared 12/8 whole-bar-rest and fractional-duration errors, retaining original results. Then certify genuinely new full-page references and OSSQ scan alignment. Detector-fusion calibration/C1 lyrics remain parallel release work; no new training or production change.

### 2026-10-03 — integrated runtime correction qualified locally

New decoder exports explicitly support empty continuation context and carry `homr.decoder.empty_continuation_context=1`. The runtime selects that behavior for greedy, margin and prefix decoding; existing unmarked graphs retain the legacy path. Actual runtime qualification, without the diagnostic monkeypatch, passes all 22 strict crop parity checks and all 12 head projection checks. Margin/prefix/fork checks pass with a constant 1,280-token cross-attention cache across 616 steps. Eight focused unit tests pass.

Repeated full-page functional smoke passes: Lieder 116.2 CPU seconds, OSSQ 108.3, both with nonempty well-formed XML and zero automatic dynamics. One CPU thread; all tmux jobs completed, `FOLLOWUP_EXIT=0`. Evidence and hash-verified tested source are at `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-integrated-runtime-20261003/`. Shipping graphs and provider pins were not changed. Local source snapshots include pre-existing uncommitted pipeline work, so the correction commit is not by itself a clean-checkout reproduction of every tested policy.

Next: audit a full-page ground-truth evaluation manifest against training and model-selection score identities, then compare complete accepted/candidate pipelines. Continue detector-fusion area/recall calibration and C1 lyrics independently. Untouched full-page accuracy, GPU/HTTP timing, final example regeneration and promotion remain open; no additional training is justified by the resolved export mismatch.

### 2026-10-03 — corrected diagnostic passes strict export parity

The ONNX one-token continuation workaround changes cross-attention cache semantics. Eager decoding sends empty continuation context, but the exported graph rejected its zero-length head-split reshape. An isolated graph correction makes the eight-head dimension explicit and permits literal zero dimensions; a diagnostic runtime supplies empty context. All 22 pilot crops now match eager recognition and complete serialized notation exactly, with all encoder, cached-eager and head-projection checks passing. No tolerance was relaxed.

Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-context-full-parity-20261003/`. Reproduce with `training/onnx/audit_joint_context_cache.py --repair-empty-reshape --full-parity` and the source/export/output directories recorded in the release sequence. Job complete on one CPU thread. Next: integrate the correction with graph capability metadata so older graphs remain compatible, cover greedy/margin/fork paths, then repeat full-page smoke. The shipping runtime is unchanged; the original page smoke does not qualify the corrected graph. New-score accuracy, GPU/HTTP latency and text-fusion/C1 remain open.

### 2026-10-02 — export parity is a release blocker

The epoch-27 public-joint checkpoint has isolated ONNX exports and a bounded parity audit. Encoder and head logits are numerically close, but greedy recognition differs on one of 22 pilot crops at a low-margin rhythm decision. Keep the accepted stack as the runtime baseline and do not publish candidate advantages as deployment results until the cause is understood or the release gate explicitly accepts this behavior. Full-page functional smoke completed on one Lieder and one OSSQ page; it is a validity check, not an accuracy benchmark. Both produced valid MusicXML and no automatic dynamics: Lieder 148.7 seconds CPU-only, OSSQ 124.8 seconds CPU-only. This closes basic export/pipeline wiring, but not strict parity, full-page accuracy, GPU latency, detector fusion, or C1 lyric attachment.

The training-held-out real-source comparison completed all 8,422 decodes without failures. Candidate symbol error improved by 1.455 percentage points on OSSQ and 0.418 points on Lieder, passing the accepted +0.25-point Lieder regression margin. The first 3,501 accepted-model decodes used two CPU threads; remaining decodes used one. This uses previously evaluated validation scores and raw eager crop decoding, not untouched/full-page qualification. See `JOINT_HOLDOUT_QUALIFICATION_2026-10-01.md` for exact local paths, limitations and next gates. No training or GPU job launched; no model promoted.

## Archived experiment state

The ImageNet-backed random music decoder + joint notation-head run is paused at **epoch 81 / update 102,060**. It did not complete the planned 160 epochs. Epoch 80 is the last complete three-source validation. No model or production pin was promoted, and instance destruction has not been confirmed by the owner.

Prioritized preservation completed at 2026-10-01 04:01 UTC: **29,024 files / 21.23 GB logical size** and code/environment records passed SHA-256 checks. The owner explicitly chose a smaller salvage over waiting for all intermediate checkpoints. **125 missing intermediate model/optimizer files / 44.58 GB** were omitted, along with redundant supplementary core-training states/context. Some other epochs were already local and retained. Published corpora and rebuildable environments/tools were not copied.

Local archive: `/home/jhlusko/workspace/homr-artifacts/instance-salvage-20261001/`.
Read `SAFE_TO_DESTROY.md`, `PRIORITY_VERIFIED.json`, `priority-manifest.json` and `intentionally-not-pulled.json` for the precise scope. The original full inventory is not a claim that every inventoried file was copied.

Retained checkpoints include epochs **44, 73, 80, 81** and the full epoch-81 resume with model, optimizer, scheduler and RNG. Resume:
`/home/jhlusko/workspace/homr-artifacts/instance-salvage-20261001/files/runs/random-core-heads-pdmx-v2-audited-20260930/full/resume.pt`.
SHA-256: `f2cfffbddd3780472a402e901dbd9b5475f1f1c36eaf9ec8cf8adaa9f76a9a29`.
Archived files may share hard links with existing immutable artifacts; copy a checkpoint to a working directory before modifying it.

## What actual decoding found

Both epochs decoded all **3,909 paired crops** successfully: 2,871 OSSQ, 429 Lieder, 609 PDMX. The reference is **public-rareNum-20260927 core + the 8,400-update real-only frozen-core heads**. It is not necessarily the strongest accepted stack or the deployed Modal model. This was float32/math-SDPA eager greedy decoding, without runtime repair policies.

| Symbol edit error; lower is better | Frozen reference | Random-start epoch 73 | Random-start epoch 80 |
|---|---:|---:|---:|
| OSSQ | 2.372% | 2.681% | 2.538% |
| Lieder | 3.542% | 5.784% | 5.738% |
| PDMX | 2.865% | 8.702% | 8.493% |

Epoch 80 is slightly better overall than 73, but OSSQ pitch and accidental errors still worsen versus the reference, with paired score-bootstrap intervals above zero. Lieder symbol/pitch/accidental regressions also have intervals above zero. OSSQ overall symbol/rhythm differences have intervals crossing zero.

There are useful notation gains: on the same correctly recognized notes, epoch 80 improves stem-direction accuracy by approximately 6.3 percentage points on OSSQ and 20.6 on Lieder, and beam-vector accuracy by 6.9 and 20.3 respectively. These conditional accuracies cannot be substituted for recognition quality or compared directly with teacher-forced macro-F1. Wrong or missing notes must count in the end-to-end assessment.

Reports and complete predictions are local:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/train-20261001/random-core-decode-epochs73-80/epoch-{73,80}/comparison.md` (and `.json`, `part-*/{frozen,joint}.jsonl`).

The public-core-initialized **joint epoch-27 candidate** is a different experiment, not the poorly performing frozen-backbone E1 run or the random-start run. Its cached decoding is at:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/train-20260929/joint-vs-frozen-20260929/decoded/`.
It had symbol errors OSSQ 1.862%, Lieder 3.728%, PDMX 3.222% on this same benchmark. These are reference points; recompute matched comparisons rather than subtracting separate confidence intervals.

The benchmark contains some core-seen score identities and is an internal diagnostic, not an untouched real-score/full-page release gate. These results establish that neither random-start candidate should be promoted now. They do not prove that the remaining 79 planned epochs could never improve it.

## Completed local comparison and review — 2026-10-01

Steps 1–3 below are complete. Four-way identity matching and all 62,544 stored recognition metrics pass recomputation. The 17-case MuseScore gallery includes original pixel-verified crops, readable crop-local measure/beat labels, and explicit warnings for reconstruction or engraving changes. It requires no individual accept/reject decisions.

Report: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/cached-decoding-audit-20261001/comparison.md`.
Audit: same directory, `error-audit.md`; gallery: `review/index.html`.
Browser: http://localhost:8765/cached-decoding-audit-20261001/review/.

Public-initialized joint **2026-09-29 epoch 27** has better recognition and end-to-end stem/beam coverage than random73/80 on all sources. It is the preferred joint candidate to qualify against the accepted full stack, not a newly promoted model. Frozen/public27/random73/random80 symbol errors are OSSQ 2.372/1.862/2.681/2.538%, Lieder 3.542/3.728/5.784/5.738%, PDMX 2.865/3.222/8.702/8.493%.

Only eight independent OSSQ score identities underlie the 2,871 crops. Large failures are concentrated, but Lieder/PDMX still degrade after removing their worst 5% (a diagnostic, not an evaluation filter). Random80 emits 149 invalid-pitch note symbols on PDMX versus 32 frozen and 15 public27. Invalid combinations motivate a bounded decoding-constraint experiment but do not explain the whole real-source gap. EOS causes were not logged, and no library/optimizer cause has been demonstrated.

Next: resolve export parity, then qualify the full public27 stack on untouched real-score/full-page inputs; for the random branch, test decoder consistency and termination logging before spending on more training. The random run remains paused at epoch81; the separate local real-source crop comparison is complete. See `error-audit.md` for limits and the detailed experiment recommendation.

## Original sequence, with steps 1–3 now completed

1. **Build a paired comparison from cached predictions — no GPU.** Include the frozen reference, public-initialized joint epoch 27, random-start epochs 73 and 80. Verify crop IDs, pixel/reference/sidecar hashes and scoring settings before matching. Report source-specific symbol, pitch, rhythm and accidental errors, score-clustered intervals, common-note head metrics and correctly-recognized-and-notated GT-note coverage. Prioritize OSSQ/Lieder; retain PDMX as a separate diagnostic.
2. **Audit the decoding regressions locally — no GPU for the initial audit.** Rank changes by score, separate widespread small errors from a few catastrophic crops, and inspect deletions/insertions, repeated passages, early termination, output/reference lengths, pitch/octave errors and accidental/key-signature handling. Check training/validation versus inference image preparation, including the Lieder pre-resize cap, canvas geometry, vocabulary order, checkpoint configuration and archived/runtime code differences. The teacher-forced/free-running gap motivates this audit; its cause is not established.
3. **Generate a focused review gallery — no GPU for stored predictions.** Show original crops beside the reference and divergent model outputs, emphasizing real-source regressions and notation gains. Use measure/beat/pitch labels. Use MuseScore for reconstruction and audit its output; keep original images authoritative and do not infer encoded stem/beam differences solely from renderer output. The review should explain representative failures, not require hundreds of individual accept/reject decisions.
4. **Choose a bounded follow-up only after the audit.** Keep the existing recognition baseline and accepted runtime policies in place. If the audit finds a code/protocol defect, fix it and re-decode the affected retained checkpoints before drawing training conclusions. If the regression is model behavior, select a targeted experiment on the public-initialized stack or a limited resume of the random-start run, with actual decoded OSSQ/Lieder recognition gates. Do not automatically spend another ~15 GPU-hours or transplant heads between incompatible core feature representations.
5. **Return to release qualification after selecting a complete stack.** Measure untouched real-score/full-page performance with the selected core, heads, runtime policies and detector fusion. Then regenerate final side-by-side examples (E3), finalize pins/calibration/provenance and update the public OTS-HOMR model description to claims supported by those results. C1 lyric-recognition/attachment quality and the lyrics/directions/dynamics fusion plan remain separate release work; random-start scoring does not close them.

No GPU is needed for steps 1–3. Random-start training/decoding is inactive. The comparison, initial error audit, gallery and full raw crop comparison are complete; export/runtime qualification remains open.
