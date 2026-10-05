# Local follow-up after the 2026-09-29 random-start experiment

The off-by-default nearest-x lyric attachment baseline is implemented and tested in OTS-HOMR. It requires explicit vocal staff/note geometry and is not enabled. Next score it against native targets before considering a producer integration.

The off-by-default `LyricAttachment` contract is now implemented and tested in the OTS-HOMR service. It validates native target identity, verse, note ordinal/onset and confidence without producing records. Next wire a producer only after its attachment policy is defined.

The C1 interface audit found that native attachment targets exist, but the current service emits no predicted lyric-to-note attachment object. Implement that off-by-default output contract before measuring verse/note attachment; do not score a proxy.

The native-ground-truth C1 text scorer now restricts source lyrics to each selected page's physical measure window. It is a coverage diagnostic only; verse assignment and note attachment remain to be wired against native targets.

Native lyric ground truth is now the primary evaluation source for the expanded pages. The ordered source-text coverage diagnostic is complete; manual scan rows are optional visual checks, not a prerequisite. See `homr/docs/C1_NATIVE_GROUND_TRUTH_SCORING_2026-10-05.md`.

An expanded lyric-row transcription worksheet is available at http://localhost:8765/c1-row-worksheet/. It keeps OCR candidates separate from any future scan-derived reference labels.

The eight expanded pages now have a page-level identity/full-page/layout checklist. Row transcription and note-pixel attachment remain pending; see the local `c1-layout-expansion-20261005/visual-certification.json` artifact.

The expanded packet now also has a crop review aid at http://localhost:8765/c1-layout-crops/. It preserves full-page fallback on staff-detection failures and does not score or label crops.

### 2026-10-05 — expanded OCR review gallery available

The eight additional pages now have a scan/OCR inspection gallery at http://localhost:8765/c1-layout-ocr/. It is intentionally unscored and includes no guessed lyric attachment. Use it to choose independently certified visible-row references while preserving the geometry-failure pages.

### 2026-10-05 — expanded lyric OCR and geometry failure controls

The eight frozen calibration layouts completed full-page OCR (594 regions, 54s) and pinned-e2 OCR (8m02s). The raster staff-gap diagnostic found valid five-line groups on only five pages and failed its minimum precondition on three pages. No lyric score or production filter was produced. See `homr/docs/C1_EXPANDED_SCORING_2026-10-05.md`.

### 2026-10-05 — isolated alias export and expanded layout review

The confidence-gated XML experiment removed only structurally owned duplicate phrase spans from cached outputs: 3 Franz candidate, 15 `sq7358579` candidate and 3 `sq9146376` candidate spans. All tie and non-slur invariants passed. This remains a source-free diagnostic, not a runtime policy or promotion gate. See `homr/docs/ALIAS_XML_EXPERIMENT_2026-10-05.md`.

The eight additional lyric layouts are visually suitable for calibration review, with identity and boundary certification still pending. Review http://localhost:8765/c1-layouts/. Next run the existing OCR/geometry diagnostics and manually certify visible rows by layout stratum. Keep fusion disabled.

### 2026-10-05 — alias-confidence controls and broader lyric packet prepared

All8 cached arms replay exactly with both endpoint provenance. Conservative NONE-event/owned-tie confidence diagnostic at0.90 removes21/44 joint source-tie-only duplicates while preserving the known Franz phrase;0.80 removes30 but loses it. Accepted at0.90 removes4/38. No universal calibration or XML/runtime edit. Next opt-in cached owned-span export experiment with independent phrase/multiplicity/boundary controls. See `homr/docs/ALIAS_CONFIDENCE_DIAGNOSTIC_2026-10-05.md`.

Eight additional lyric calibration scores frozen by native metadata/hash, two each single/multiple verse and opening/continuation. No OCR, predictions or held-out selection. Review http://localhost:8765/c1-layouts/. Manual edition/page/row/verse certification and fixed OCR/e2/geometry scoring remain next; this does not yet qualify new layouts or attachment. See `homr/docs/C1_LAYOUT_EXPANSION_2026-10-05.md`.

Twenty focused tests pass. All current jobs complete, ETA0; persistent review server remains. Defaults, pins, disabled fusion/checkbox unchanged; no training or deployment. Final provider/production latency work remains after selected-stack acceptance.

### 2026-10-05 — tie-alias and staff-gap experiments completed

Exact predicted tie/phrase duplicates include38 accepted and44 joint source-tie-only spans. Joint also has one source-correct Franz phrase duplicated by an erroneous predicted tie, so unconditional deduplication would lose a correct phrase. Start-only suppression remains unsafe for longer phrases beginning on tied notes. Next recover both endpoint confidences before an opt-in cached suppression experiment, with simultaneous/true-phrase/boundary controls. See `homr/docs/TIE_PHRASE_ALIAS_AUDIT_2026-10-05.md`.

Scan-only staff-gap filtering removes all12 reviewed non-lyric regions; same-row recovery restores all5 dropped lyric fragments. On these two calibration pages:75/75 existing lyric-row OCR regions,0 outside,5/408 character edits (same as full-page OCR). This rule was designed after reviewing calibration and does not certify voice ownership, held-out precision/recall or attachment. Review: http://localhost:8765/c1-geometry/. Next freeze broader layouts and use identified runtime system/staff geometry rather than promote the raster prototype. See `homr/docs/C1_STAFF_GEOMETRY_2026-10-05.md`.

Seventeen focused tests pass; all current compute complete, ETA0. Defaults, model pins, disabled fusion and checkbox unchanged. Provider/production latency qualification follows selected-stack acceptance; no new training/deployment scheduled.

### 2026-10-05 — complete-span and lyric-fragment audits completed

Conditional complete-span counts now distinguish sounding ties, engraved ties and phrase slurs, with simultaneous-event, tie-chain and boundary tests. Joint improves supported Glazunov ties26→30/30 and Franz11→14/24; phrase matches improve on three pages, unchanged Haydn. Both stacks still have extra phrase spans. Next identify exact tie-alias phrase pairs with simultaneous true-phrase controls before any exporter change. See `homr/docs/NOTATION_SPAN_AUDIT_2026-10-05.md`.

Pinned-e2 area200/10% selection retains70/75 existing lyric-row OCR fragments but also12 visually confirmed non-lyric regions; drops5 lyric fragments. Raising overlap loses more lyrics and leaves titles/credits. Seventeen original-scan examples: http://localhost:8765/c1-selection/. Next broaden frozen calibration and test system/voice geometry while preserving OCR context; verse/note attachment remains separate. See `homr/docs/C1_LYRIC_SELECTION_AUDIT_2026-10-05.md`.

All current audit compute complete, ETA0;13 focused tests pass. Current defaults/pins and disabled fusion remain unchanged. Final provider identity and production latency/failure qualification still follow selected-stack acceptance; no training/deployment scheduled.

### 2026-10-05 — cached slur ownership and intact OCR selection completed

Eight exact XML replays trace all155 unspecified starts to preserved core endpoints with no matching structured start slot. Legacy tokens conflate ties and phrases; source evidence includes75 tie-only and7 simultaneous tie/phrase starts. No blanket fallback/deletion is justified. See `homr/docs/SLUR_EVENT_OWNERSHIP_2026-10-05.md`.

Intact full-page OCR selected by pinned Lyrics-box overlap gives10–11/408 character edits at predeclared10% coverage, versus56/408 for nested detector crops and5/408 for full-page OCR. Higher coverage loses more fragments. Calibration only, not semantic precision/attachment or release qualification. Review: http://localhost:8765/c1-intact-lines/. See `homr/docs/C1_INTACT_LINE_SELECTION_2026-10-05.md`.

Next priorities: independent tie/phrase span ownership with simultaneous-event controls; broader frozen semantic lyric calibration and missed-fragment audit; then verse/note attachment and final provider/production latency qualification. Defaults/pins remain unchanged; direction fusion still blocked. All current jobs complete, ETA0; no new training or deployment.

### 2026-10-05 — matched notation policies and scan-row lyric diagnostics completed

All8/8 real-page policy arms succeeded in9m47s, one CPU thread with completed tmux watcher. Recognition and strict timing/meter metrics reproduce the earlier four-page baseline exactly. On unique shared source/prediction timing/pitch anchors, OSSQ stems: accepted18→12 errors/543 after reviewed policy, joint6→6/543. Lieder: legacy stem policy accepted128→130/652, joint6→6/652. These conditional results favor joint stems, while not qualifying universal policy or promotion. OSSQ slur-start errors: accepted14→7/88, joint12→13/88. Joint Glazunov includes11 exported starts without explicit side and one start-count mismatch, not12 wrong-side predictions; the OSSQ rule adds one Haydn wrong side. Next audit preserved core endpoints versus structured slots/export before filling unknown sides. Candidate use of the v7-calibrated0.8 cutoff remains provisional. Report: `homr/docs/PAGE_NOTATION_POLICY_2026-10-05.md`; local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/page-notation-policy-20261005/`.

C1 now has12 manually checked scan-row transcriptions/rectangles on the two calibration pages, committed for reproducibility. Review was not blinded to prior OCR; no source text prompts or text-selected alignment. Conditional normalized character edits: full-page OCR5/408 (1.23%), pinned-e2 direct crops270/408 at area4 and269/408 at area200; OCR line detection inside the same boxes57/408 and56/408. Tightly cropped input clips syllables and mixes verse rows; area filtering does not resolve it. This is not lyric precision, syllable/note attachment, a held-out score or production-path latency. Compare original row crops and outputs at http://localhost:8765/c1-rows-final/. Report: `homr/docs/C1_LYRIC_ROW_QUALIFICATION_2026-10-05.md`; local refs/scores: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-lyric-row-reference-20261005/`; model/region evidence in sibling `c1-e2-region-ocr-20261005/` and `c1-e2-nested-ocr-20261005/`.

Next priorities: (1) audit unknown slur sides/core-structured endpoint compatibility and retain joint raw-side controls; (2) test full-page OCR line preservation plus independent semantic lyric selection rather than tight crop recognition, on broader frozen calibration data, then qualify verse ordering and note attachment with independently certified note pixels; (3) complete final selected-stack provider identity and production GPU/HTTP/OCR latency/failure gates, while keeping meter improvements separate. Direction fusion remains blocked by B4's failed gate. Keep current defaults, pins and disabled lyric checkbox; no training, promotion, upload, push or deployment. All compute/watchers complete, ETA0. Sixteen focused tests and scoped Ruff/Black pass; final gallery plus12 original row crops return HTTP200.


### 2026-10-05 — raw meter attribution and initial lyric OCR pilot completed

All four Sonntag page/model traces reproduce saved XML and decoded symbols exactly, in217s total with one CPU thread and completed tmux watcher. Incorrect meter declarations are already present in raw decoding; cross-staff repairs and voice cleanup preserve their values/order. Joint epoch27 recognizes p001's printed initial6/4, then incorrectly declares4/4 on later piano systems; on p002 it declares initial4/4 despite inherited6/4. Do not weaken the exporter guard based on this witness. Next meter hypothesis must retain genuine changes, printed-correct-signature and pickup/mixed-meter controls. Report: `homr/docs/METER_PROVENANCE_2026-10-05.md`; local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/franz-meter-provenance-20261005-v2/`.

The two linked calibration scans completed unprompted full-page RapidOCR in26.24s, one CPU thread, fixed local model hashes, no source lyric prompt/download. Chaminade41 and Franz108 regions include plausible bilingual lyrics, titles and notation false positives; counts are not lyric accuracy. Inspection page: http://localhost:8765/c1-ocr/ (click text for original crop). Source ownership is displayed separately; no guessed note alignment. This is standalone OCR, not the pinned-e2 fusion route or production latency qualification. Report: `homr/docs/C1_OCR_PILOT_2026-10-05.md`; local predictions: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-raw-ocr-20261005/`.

Next priorities: (1) complete identical final-policy OSSQ/Lieder stack verification, with the scoped meter investigation kept separate; (2) freeze scan-visible lyric row regions and literal transcription independently of OCR, then quantify pinned-e2 region selection/OCR and native verse mapping; certify note pixel anchors before attachment scoring; (3) continue conservative lyrics/dynamics calibration, with direction fusion blocked by the failed B4 gate. All compute/watchers finished, ETA0. Eleven focused meter/context/certification tests pass, scoped Ruff/Black and review JavaScript/HTTP checks pass. No inference default, pin, training, promotion or deployment change.


### 2026-10-05 — previous-page meter and C1 linkage experiments completed

The independently reviewed preceding Sonntag page was decoded by both stacks in90s, one CPU thread, persistent tmux with completed exit0 watcher. Guarded predicted-meter carryover and disabling stated-numerator rejection produce **no improvement** on its affected second page: accepted event/strict/meter edits remain8/26/51; joint epoch27 remains9/27/42. Accepted prior meter is wrong4/4; candidate vocal prior6/4 is correct, but its current cached output already declares4/4, so guarded carry does not replace it. Accepted carry adds incorrect4/4 where XML previously had no meter. Keep defaults unchanged. Next audit raw per-system meter predictions versus same-page repairs, then continue final-policy complete-stack verification; current diagnostics still omit stem/slur-side arbitration and OCR/provider latency. Report: `homr/docs/PREDICTED_PAGE_METER_2026-10-05.md`; canonical local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/franz-meter-carry-20261005-v3/`.

Direction pin naming is now resolved: **B4 DirectionText epoch4** is a five-class-plus-background candidate, distinct from **Aug30 seven-class dynamics epoch4**. B4's already-recorded single held-out test rejected it: OSSQ221/421 direction matches but788 predictions exceeds367 cap, and Lieder8/11 falls below11/11 floor. The intended split amendment was conditional on that gate; B4 cannot be renamed/swapped into the eight-channel provider. No qualified replacement direction detector is available. Continue lyrics e2 and review-only dynamics Aug30 e4 calibration independently; direction fusion stays disabled pending a capable independently qualified model.

C1 has a hash-bound calibration-only page/system linkage pilot: two original scans, seven independently certified systems,128 native lyric entries and eight reviewed visible phrase/verse anchors. Native verse numbers do **not** reliably follow scan vertical order (Chaminade English is above French despite native verse2/1). Page/system ownership is certified; per-note pixel attachment is not. Literal/native transcription also needs normalization/adjudication before exact-match scoring. Report: `homr/docs/TEXT_FUSION_C1_LINKAGE_2026-10-05.md`; evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-page-linkage-20261005/linked/`.

Next priorities: (1) meter provenance audit and final-policy OSSQ/Lieder stack qualification; (2) a calibration-only lyric OCR/verse-order/attachment pilot on the two linked pages with independently adjudicated transcription; (3) conservative lyric/dynamics component filtering while keeping direction capability unresolved. No pin/default change, training, push, deployment or model promotion. Eleven meter/context/certification tests and six C1 tests pass; scoped formatting checks pass. All experiment jobs complete, ETA0.


### 2026-10-05 — parallel real-page and text-fusion work completed

Four additional original music pages are independently reviewed for edition/part/system/measure boundaries: Haydn sq9146376, Glazunov sq7358579, Chaminade IMSLP154086 and Franz IMSLP89797. All 8/8 matched accepted/epoch27 decodes succeeded in 12m49s; tmux jobs and watchers complete, ETA 0. Haydn rhythm edits improve 5→3 while accidentals worsen 3→4; Glazunov and Chaminade tie; Franz event edits worsen 8→9. These remain diagnostic pages with recorded historical/validation exposure, not new untouched tests or token-SER promotion gates. Strict timing scoring finds shared missing inherited 6/4 on Franz; epoch27 handles its later return to 6/4 better. Keep the candidate/rollback decision open.

The six-page calibration-only text audit reproduces e4 direction failure: 0/22 OSSQ and 0/2 Lieder known direction matches at IoU≥0.5. Exact checkpoint/class order matches the pinned Aug30 seven-class model; its selected loss targets Lyrics/Dynamic. This is distinct from the later DirectionText model and is not channel mislabelling. No eligible replacement pin was found. The historical “directions use e4” route is not a qualified capability: keep it disabled pending a capable detector. Dynamics can remain review-only on e4; lyrics use e2.

Conservative next filtering pilot: Lyrics area 200 / Dynamic area 50, with area 4 controls, on broader calibration-only inputs. This is a proposal, not a frozen production threshold. Area200 loses weak dynamic overlap coverage; partial annotations do not authorize treating unmatched boxes as negatives. Detector source-order training indexes were not recovered, so training independence remains unknown. C1 now has 26,662 native lyric targets across 124 calibration scores, preserving verse and source note/onset ownership; scan linkage and transcription/attachment accuracy are still uncertified. Held-out test annotations were not used to fit these diagnostics.

Next priorities: (1) test predicted previous-page meter context with meter-change/strict-onset controls and continue complete final-policy stack qualification; (2) resolve direction-detector capability before three-route fusion calibration, while separately extending conservative lyrics/dynamics calibration; (3) certify native-lyric-to-scan linkage, then quantify transcription, verse ordering and note attachment against independent references. No new training, pin/default change, upload, push or deployment.

Reports: `homr/docs/REAL_PAGE_EXTENSION_2026-10-05.md` and `homr/docs/TEXT_FUSION_QUALIFICATION_2026-10-05.md`. All evidence is local under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`: `real-page-selection-20261005/`, `real-page-pair-20261005/`, `text-fusion-sweep-expanded-20261005/`, `c1-source-lyrics-20261005/`. Page track 18 tests, text track 56 tests pass; Black/Ruff pass. No compute remains; the loopback review server remains available.


### 2026-10-05 — complete-voice context recovered

All 32 selected witnesses were replayed with available neighboring same-page systems. Current cleanup preserves 22/23 correct raw predictions and repairs 8/9 false-tuplet controls; the rejected guard achieves 23/23 and 7/9 respectively. These note/rest metrics exclude repeated key-signature headers and are diagnostic, not full-page release scores. All nine repair controls have earlier-page predicted meter evidence; three have same-page declarations (one in the target itself). Cross-page meter is not currently carried by HOMR. The review at http://localhost:8765/ includes recovered context and original full-page scan links.

An actual Mendelssohn page repeat confirms candidate reranking corrects all 16 reviewed raw triplet errors before voice cleanup; output XML matches the first run exactly. See `homr/docs/TUPLET_VOICE_CONTEXT_2026-10-05.md`. Canonical local results: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/voice-context-20261005-v2/`. Independent native OSSQ measure-boundary certification remains open. Keep current inference defaults; next qualify identical complete stacks with certified page inputs, and test any meter-local cleanup against both control groups.


## Current release priorities — 2026-10-04

This order supersedes the historical status table and earlier running-job entries below. The accepted September18 core plus September24 v7 heads remains the local comparison/rollback stack; it is not asserted to equal the deployed provider pin. Public-joint-core-heads-20260928 epoch27 is the leading qualification candidate: the completed paired real-source crop comparison improves symbol error by1.455pp on OSSQ and0.418pp on Lieder. Corrected ONNX/runtime parity passes, but functional page smoke and crop scores do not establish full-page accuracy or promotion. One matched Lieder page improves pitch while worsening rhythm.

1. **Finish the current label/export checks locally.** Independently measure separate tied/sounding-tie elements, phrase-slur endpoints and span pairing, and articulation ownership against source XML. Preserve note/timing controls and report crop-boundary spans and unsupported source notation explicitly. The110 changed Lieder crops remain outside metadata-only migration. This bounded correctness work must not turn into an unconditional requirement to train new timing heads before release.
2. **Certify full-page evaluation inputs and compare complete stacks.** Audit score identities against training/model-selection history; recover trustworthy OSSQ scan-to-source alignment and page boundaries. Compare accepted and epoch27 stacks under identical detector/decoder/post-decode policies, with pitch/rhythm/accidentals, onsets/meter/voices, failed pages and source-specific results. Keep the authorized Lieder symbol-error ceiling of+0.25pp and report OSSQ separately. D1 still needs OSSQ real-page verification and provider identity/pin integration. Revisit ordinary-group beam substitution separately; the rest-spanning hybrid remains rejected.
3. **Continue text fusion independently of model selection.** Freeze detector postprocessing after the component-area/recall sweep; verify training exclusions and partially labelled region handling; fit fusion calibration and evaluate its held-out partition. Lyrics use e2; review-only dynamics use Aug30 seven-class e4. Direction fusion stays disabled: the intended separate B4 DirectionText epoch4 failed its single held-out gate. C1 must establish acceptable transcription, verse ordering and note attachment. Checkbox/fusion remain disabled pending those gates.
4. **Choose and qualify the release stack.** Select epoch27 only if complete-stack evidence passes; retain the accepted stack if it does not. Measure the final production GPU/HTTP/OCR/attachment path, including cold overhead and failure handling, against the280s per-page timeout. The earlier warm22-page benchmark does not qualify the changed final path. Prepare candidate manifests, immutable pins and rollback; activation remains a separate release decision.
5. **Package and present the final release.** Regenerate E3 examples against the selected models, update the OurTextScores/ots-homr page with supported public-reproducible advantages and limitations, publish reproducibility instructions and qualified corpus updates as needed, then complete appropriate repository consolidation and release review.

**Separate improvement lane:** finish certified timing-label support, raw/normalized/native-staff scoring policy and an opt-in resumable frozen-core trainer before renting GPU for the four additional timing heads. They need not block a release whose existing stack passes. Any later joint loss weighting must use measured shared-core gradients and preserve pitch/rhythm/accidental gates. Random/ImageNet-start training remains paused because its decoded recognition underperformed; no automatic resume or new multi-day run is scheduled.

All prior qualification jobs are complete. Immediate steps1–3 are local work and do not require a GPU rental. A full-page evaluation manifest and calibration subset must be frozen before giving trustworthy broad-run ETAs. See `homr/docs/LABEL_POLICY_METER_QUALIFICATION_2026-10-04.md` and the dated runtime/text-fusion reports below for completed evidence. No deployment, pin promotion, corpus upload or training is authorized by recording this order.

### 2026-10-04 — nine-case tuplet review available locally

Review: http://localhost:8765/ (persistent `tuplet-review-server` tmux session, loopback only). All nine successful-repair negative controls show original hash-verified OSSQ crops beside reference labels, raw accepted-stack predictions, current cleanup and rejected guard. Crop-relative measure labels describe triplet-to-ordinary-note changes. Case8 clearly shows16 raw rhythm errors,0 after current cleanup,8 remaining under the guard; case9 is the sole explicit predicted2/2 example.

“Eight have no predicted meter” means no time-signature declaration was emitted in these standalone crop decodes; it does not establish absent printed or inherited full-page meter. Reconstructed display meters are explicitly identified as XML-export context. MuseScore4.6.5 outputs were audited for visible notes AND rests, timing, onset and staff;20/36 panels have changes (including inserted rests), flagged on the page. Original scans and stored labels remain authoritative. This gallery requires no individual voting and is not a new promotion gate.

Local site, all images/XML and provenance: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/`. Generator: `homr/training/transformer/review_guarded_tuplets.py`; guarded renderer reuse verifies inputs, binary and exact reconstructed XML. Nine cards/81 unique linked assets return HTTP200. Rendering/watcher complete, ETA0; the serving tmux remains active. No inference/default change, promotion, push, upload or deployment.

### 2026-10-04 — first predicted-context tuplet guard tested, not accepted

Implemented a diagnostic-only exact-fill guard using predicted meter or strongly supported local bar length, with crop-boundary and staff/chord ambiguity safeguards. All4,211 paired real-source caches and10 Grieg complete-voice XML variants complete; raw33,688 metrics and prior controls replay. It protects23/23 otherwise-perfect predictions but retains0/9 complete successful-repair controls (one partial), so do not activate it. Eight negative controls lack predicted meter; context/boundary safeguards are too restrictive for standalone crops.

Guard-only symbol errors: OSSQ accepted4,185→4,070, joint27 2,773→2,595; Lieder380→373 and327→326. These aggregate improvements do not override the failed negative controls or OSSQ pitch regressions. Grieg guard-only scores are unchanged; its combined unison-preserving arm reproduces the prior two-note recovery.

Next recover predicted meter/staff/voice and true measure-boundary context on complete-voice controls rather than weaken safeguards to fit known crops. Keep unison/noise qualification independent, and continue complete-stack page qualification and text fusion; this optional guard is not a new mandatory release blocker. No new training needed.

Write-up: `/home/jhlusko/workspace/homr/docs/GUARDED_TUPLET_EXPERIMENT_2026-10-04.md`. Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/` and `page-guarded-tuplet-20261004/` alongside it. Twenty-two focused tests and Ruff/Black pass. One-thread/core tmux job/watcher complete, ETA0; defaults unchanged, no promotion, push, upload or deployment.

### 2026-10-04 — cleanup stages isolated; context guard is next

Completed all4,211 OSSQ/Lieder cached pairs and14 Grieg complete-voice XML ablations. Raw33,688 metrics replay; standard/no-pitch controls match the previous audit. Tuplet removal explains the remaining standalone-crop damage: disabling it reduces OSSQ symbol errors by107 accepted/178 joint27 and Lieder7/1. Both tuplets and pitch dedup disabled recover raw aggregate scores. Lower-staff/header ablations do not improve measured recognition here; header/meter export is not covered by those metrics.

The tuplet heuristic damages23 otherwise exact crop/model pairs, but fully repairs9 accepted OSSQ crops. It uses local median measure duration without explicit meter/context; switching it off globally is not qualified. The complete Grieg trace shows no measured tuplet-only benefit, while unison preservation still repairs both pitch errors in each model. These are diagnostics on existing validation/evaluated scores, not new untouched gates.

Next: opt-in predicted-context tuplet safeguards plus successful-repair negative controls, unison/noise and complete-voice/page qualification; continue broader final-policy page and independent text-fusion work. Defaults remain unchanged; no new training is needed for these runtime checks.

Write-up: `/home/jhlusko/workspace/homr/docs/CACHED_CLEANUP_STAGES_2026-10-04.md`. Local crop/page reports: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/{cached-cleanup-stages-20261004,page-cleanup-stages-20261004}/`. Eleven tests and Ruff/Black pass. One-thread tmux job/watcher complete, ETA0; no promotion, push, upload or deployment.

### 2026-10-04 — real-source cached cleanup ablation complete

All4,211 paired OSSQ/Lieder crops (9/13 score identities), all33,688 stored recognition metrics and paired input hashes pass. Disabling only pitch dedup improves symbol errors OSSQ accepted4,185→4,174 / joint27 2,773→2,758 and Lieder380→340 /327→287. Of74 changed crop/model pairs,60 improve,7 regress,7 tie. Preserved collisions include unwanted repetitions, rests and malformed note/rest fields, so defaults remain unchanged.

Other cleanup still increases standalone-crop errors relative to raw predictions; production cleans concatenated voices, so isolate these stages before inferring a full-page fix. Next: stage-wise cached cleanup audit and full-voice/page positive/negative controls, then continue final-policy page qualification and independent text fusion. No additional training is needed for this investigation.

Details: `/home/jhlusko/workspace/homr/docs/CACHED_PITCH_DEDUP_AUDIT_2026-10-04.md`. Local reports/cases/hashes: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/cached-pitch-dedup-20261004/`. Seven tests and Ruff/Black pass. Complete, ETA0; no running job, promotion, default change, push, upload or deployment.

### 2026-10-04 — cached timing audit locates a runtime cleanup defect

Exact four-arm cached XML replay and8/8 source-part round-trips complete. Late repair/meter ablations leave original timing scores unchanged. Beethoven source labels preserve elapsed timing, but encode whole-bar rests as dotted-half tokens; models choose ordinary whole-rest tokens. A predicted-meter sole-rest hypothesis repairs18 durations (accepted rhythm21→3; epoch27 18→0), but incomplete-bar safety remains unqualified.

Grieg's44 paired-tremolo duration errors reproduce in current source-label round-trip: written half/generic tremolo loses elapsed1-quarter, 2:1 ratio and one-stroke details. This is not failed learning of the legacy half-note target. More of the same training cannot restore missing distinctions.

Instrumented Grieg repeats match original XML/cache hashes for both models and prove both recognize two D4 unisons in measures9–10; pitch deduplication deletes one in each. Disabling only that cleanup in cached replay yields pitch2→0/372, accidental2→0/372, rhythm/event46→44/380, with no extra pitched events elsewhere on this page. Source labels preserve multiplicity; distinct XML voice ownership and broad duplicate-noise safety remain unqualified. Defaults and original evaluation scores unchanged.

17 focused tests pass; new-tool Ruff/Black pass. All one-thread tmux jobs finished, EXIT0/TRACE_EXIT0, ETA zero. Write-up: `/home/jhlusko/workspace/homr/docs/OSSQ_CACHED_TIMING_AUDIT_2026-10-04.md`. Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-cached-timing-audit-20261004/` (cached/label reports, cleanup traces, cleanup-ablation-final, executed source snapshots, logs).

Next quantify cached pitch-dedup ablation against real OSSQ/Lieder crops; broaden rest positive/incomplete-bar checks; continue reviewed full-page/final-policy coverage and independent text-fusion calibration. No new GPU/training, model promotion, pin change, upload, push or deployment.

### 2026-10-04 — OSSQ paired page diagnostic complete

Four decodes succeeded,2/2 paired pages, EXIT0,6m46s sequential one-thread wall time; tmux and completion watcher finished, ETA zero. Epoch27 improves Beethoven pitch3→1/383 and rhythm21→18/455; accidental totals remain1/383 but change from an insertion to a substitution. Grieg is identical: pitch2/372, rhythm/event46/380, accidental2/372. Both scores were historically evaluated; diagnostic only, not untouched testing, crop SER or final-policy promotion.

Shared errors are concrete:18 Beethoven full-bar rests export4 quarters instead of3;44 Grieg paired-tremolo half glyphs have source elapsed1 quarter/2:1 time modification but export2; both stacks collapse simultaneous D4 voices in Grieg measures9–10. Original metrics remain intact. Earlier Lorelei epoch27 pitch improves7→5 but rhythm worsens14→17. These three pages modestly favor epoch27 on OSSQ without settling promotion.

All evidence local: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-opening-page-pair-20261004/` (`summary.json`, `completion-report.md`, per-score `comparison.json`/`edit-audit.json`, Grieg `source-duration-audit.json`, environment/source snapshots). Write-up: `/home/jhlusko/workspace/homr/docs/OSSQ_OPENING_PAGE_DIAGNOSTIC_2026-10-04.md`. Ten focused tests pass. Next bounded cached-label/export checks for these timing/voice cases, broader reviewed real-page coverage and final-policy verification; text fusion independent. No push, promotion, upload, deployment or training.

### 2026-10-04 — two OSSQ opening pages certified; paired diagnostic starting

Beethoven sq8071278 page1 maps to physical measures1–36 (systems8/10/9/9); Grieg sq9631717 page1 maps to1–28 (7/9/6/6). Agent visual edition/boundary/part review and all-four-part source-layout checks pass; original/source hashes and a17-score selection pool are frozen before recognition. Both scores were already evaluated in RUNLOG27.81, so this is diagnostic evidence, not untouched testing. Corpus-wide scanned-system metadata remains absent.

Accepted September18/v7 versus public-joint-core-heads-20260928 epoch27 uses prior float32 exports and identical Lorelei diagnostic policies, without text/OCR, reference injection or D1 rule activation. Four sequential one-thread page decodes start in persistent tmux with a completion watcher; initial ETA20–40 minutes, revised after first result. Failures remain failures; completed arms are resumable with input/code identity guards. Nine focused tests and new-file Ruff/Black pass.

Write-up: `/home/jhlusko/workspace/homr/docs/OSSQ_OPENING_PAGE_DIAGNOSTIC_2026-10-04.md`. Certification: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-page-certification-20261004/`. Run/status: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-opening-page-pair-20261004/` (`progress.json`, `EXIT`, per-score comparison, completion-report.md when finished). No promotion, training, upload, push or deploy.

<!-- ossq-opening-pair-20261004-status-begin -->
OSSQ paired diagnostic completed 2026-10-05T03:31:13.289459+00:00: 4/4 successful arms, 2/2 fully paired pages; EXIT0. ETA zero. Results: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-opening-page-pair-20261004/completion-report.md`. No promotion/deployment.
<!-- ossq-opening-pair-20261004-status-end -->

### 2026-10-04 — native slur evidence and export contract

Archived all six unresolved phrase endpoints with source/native hashes and verbatim native anchors. Exact-measure/pitch evidence supports retaining all six, including multiple stops and cross-voice spans; complete native event/span mapping is still uncertified. Do not replace extraction with a voice-only pairing rule. Proposed opt-in contract requires explicit producer/field availability, separate tie/phrase ownership, span identity independent of voice, non-stealing boundary stops, detailed tremolo type/strokes and predicted-output qualification. Default inference unchanged.

Ten focused tests and new-file Ruff/Black pass. Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/native-slur-evidence-20261004/evidence.json`. Contract and full-page input checklist: `/home/jhlusko/workspace/homr/docs/STRUCTURED_EXPORT_CONTRACT_2026-10-04.md`.

Full-page inputs remain the gating work: zero verified untouched Lieder pages; OSSQ122 scores/2,986 scan pages have no certified scanned-system metadata. Next certify a small identity-selected OSSQ page/reference mapping before broad paired inference; distinguish diagnostic/model-selection from independent testing. Text-fusion calibration proceeds independently. No running job, ETA zero; broad-run ETA requires a frozen eligible manifest and one-thread pilot. No training, upload, pin change, push or deployment.

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
# 2026-10-05 attachment baseline

The C1 attachment contract is now implemented in the OTS service as an off-by-default
diagnostic policy. The existing MuseScore/rendered-source nearest-x benchmark is the baseline:
98.2% overall (26,503/26,998 syllables), 93.0% on melismas, and 98.7% on one-note syllables.
See `docs/C1_ATTACHMENT_BASELINE_2026-10-05.md`. The next release measurement is a staged
score using native ground truth, with detector misses, OCR errors, and attachment errors
reported separately after a producer exposes note anchors and lyric boxes.
