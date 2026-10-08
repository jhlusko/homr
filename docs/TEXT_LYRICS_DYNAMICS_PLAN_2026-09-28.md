# Lyrics, direction text and dynamics plan — 2026-09-28

### 2026-10-05 — parallel real-page and text-fusion work completed

Four additional original music pages are independently reviewed for edition/part/system/measure boundaries: Haydn sq9146376, Glazunov sq7358579, Chaminade IMSLP154086 and Franz IMSLP89797. All 8/8 matched accepted/epoch27 decodes succeeded in 12m49s; tmux jobs and watchers complete, ETA 0. Haydn rhythm edits improve 5→3 while accidentals worsen 3→4; Glazunov and Chaminade tie; Franz event edits worsen 8→9. These remain diagnostic pages with recorded historical/validation exposure, not new untouched tests or token-SER promotion gates. Strict timing scoring finds shared missing inherited 6/4 on Franz; epoch27 handles its later return to 6/4 better. Keep the candidate/rollback decision open.

The six-page calibration-only text audit reproduces e4 direction failure: 0/22 OSSQ and 0/2 Lieder known direction matches at IoU≥0.5. Exact checkpoint/class order matches the pinned Aug30 seven-class model; its selected loss targets Lyrics/Dynamic. This is distinct from the later DirectionText model and is not channel mislabelling. No eligible replacement pin was found. The historical “directions use e4” route is not a qualified capability: keep it disabled pending a capable detector. Dynamics can remain review-only on e4; lyrics use e2.

Conservative next filtering pilot: Lyrics area 200 / Dynamic area 50, with area 4 controls, on broader calibration-only inputs. This is a proposal, not a frozen production threshold. Area200 loses weak dynamic overlap coverage; partial annotations do not authorize treating unmatched boxes as negatives. Detector source-order training indexes were not recovered, so training independence remains unknown. C1 now has 26,662 native lyric targets across 124 calibration scores, preserving verse and source note/onset ownership; scan linkage and transcription/attachment accuracy are still uncertified. Held-out test annotations were not used to fit these diagnostics.

Next priorities: (1) test predicted previous-page meter context with meter-change/strict-onset controls and continue complete final-policy stack qualification; (2) resolve direction-detector capability before three-route fusion calibration, while separately extending conservative lyrics/dynamics calibration; (3) certify native-lyric-to-scan linkage, then quantify transcription, verse ordering and note attachment against independent references. No new training, pin/default change, upload, push or deployment.

Reports: `homr/docs/REAL_PAGE_EXTENSION_2026-10-05.md` and `homr/docs/TEXT_FUSION_QUALIFICATION_2026-10-05.md`. All evidence is local under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`: `real-page-selection-20261005/`, `real-page-pair-20261005/`, `text-fusion-sweep-expanded-20261005/`, `c1-source-lyrics-20261005/`. Page track 18 tests, text track 56 tests pass; Black/Ruff pass. No compute remains; the loopback review server remains available.


## 2026-10-01 implementation authorized

Owner authorized the routing and calibration-data preparation now, while the core/head comparison runs. Three logical routes reuse two physical checkpoints/passes initially. Dynamics remains review-only, fusion/checkbox enablement waits for calibrated region handling, C1 text read/attachment quality and final-path latency. The local implementation now assigns Dynamic only to the dynamics route, while sharing the e4 physical pass. Scanner validates all three logical identities. Preparation must distinguish positive labels from fully adjudicated negatives and preserve unknown regions.


The owner agreed to a region-detection/fusion architecture with specialized dynamics
recognition. This document separates the agreed target from currently implemented code.

## Agreed architecture

| Logical component | Responsibility | Starting model / status |
|---|---|---|
| Music recognition core and notation heads | Notes, rhythm and structural notation | Public core plus joint heads under evaluation |
| Lyrics detector | Locate lyric regions when the user requests lyrics | Released e2 / calibrated lyrics-only-v1 route |
| Direction detector | Locate non-lyric written directions | Released e4; collapsed DirectionText successor not promoted |
| Dynamics detector | Locate musical dynamic glyph regions | Reuse e4's Dynamic channel initially; separate logical identity |
| Fusion | Class allowlists, calibrated overlap handling, provenance and ambiguity handling | Local fusion-v1 box-review implementation; disabled |

Three logical detector routes do not necessarily mean three physical model passes:
while direction text and dynamics both use e4, reuse one inference result and route its
classes independently. A separate checkpoint/pass is needed when either route adopts a
specialized model. Measure its full-page incremental cost before promotion.

Tempo, StaffText and Expression share the DirectionText semantic family. The collapsed
training experiment already exists, but its candidate failed the release gate. Current
e4 still exposes its legacy classes. Never change training channel order independently of
runtime CLASS_ORDER; checkpoint metadata and calibration must identify the exact order.

## Recognition and musical attachment

Detection answers where a region is. It does not establish what it says or where it belongs
in the score. These subsequent stages have distinct quality requirements:

- Lyrics: OCR the syllables/text, retain verse identity, and associate syllables with the
  correct staff, notes and musical positions. The checkbox remains gated on C1 read/attach
  quality and C2 final-path latency.
- Direction text: read the words, preserve their text, and place them at the appropriate
  staff/measure/beat. Semantic interpretation of tempo/expression is a later decision.
- Dynamics: recognize p/pp/mp/mf/f/ff etc. with a music-glyph-specific classifier or
  recognizer rather than assuming generic prose OCR is sufficient. Associate the glyph
  with the correct staff and musical position, then emit the appropriate MusicXML marking.
  Written directions such as crescendo/diminuendo remain DirectionText; graphical hairpins
  require a separate notation/geometry path.

Dynamics initially remain review-only. The joint model's dynamic.mark head is experimental,
not an automatic output authority. It remains prohibited in the current service policy.
Do not route its present approximately 10–11% macro-F1 output into released MusicXML.
Training inclusion is separate from inference permission; audit its flat behavior before
choosing its role in the next training objective.

## Required wiring and gates

1. Add the dynamics detector identity to pins, fusion class allowlists, calibration metadata,
   provenance and tests. Route only Dynamic from this identity; prevent duplicate candidates
   when e4 supplies both direction and dynamics. Retain lyrics-only filtering for e2.
2. Keep thresholds/calibrated confidence specific to checkpoint and class. Raw pixel
   confidences from different detectors cannot decide cross-model conflicts.
3. Fail closed for explicitly requested fusion if a required model/calibration is missing.
   Ambiguous or uncalibrated candidates remain review items.
4. Define source-disjoint gates for region recall/false positives, recognized glyph/text,
   and correct note/staff/verse attachment. Dynamics strength confusion matters separately
   from box localization. Do not collapse these into a single score.
5. Measure production-path latency including core decoding, detector passes, OCR/recognition,
   fusion, attachment and cold/HTTP overhead within the 280-second per-page timeout.
6. Promote lyrics/directions/dynamics independently after their gates pass; deployment and
   production pin changes remain separate decisions.

## Current implementation and evidence

### 2026-10-01 — text-fusion preparation and real-checkpoint preview complete

Routing/calibration tooling committed locally in OurTextScores as `bee8a18` (not pushed). Preparation retained **1,236 pages / 224 scores**: calibration803/133 scores, test433/91 scores, zero score overlap. OSSQ299 pages, Lieder937. Positive boxes: Lyrics11,071, Dynamic6,153, DirectionText986; evidence preserved and duplicates merged. Input problems explicitly include364 missing page images and85 oversized originals deferred above30MP. No detector-training-disjoint claim yet; exact released e2/e4 training IDs are still unverified.

Actual production-detector smoke completed2/2 calibration pages, four physical calls total, using exact pinned weights/original pixels. CPU page times112.771s (OSSQ) and64.316s (Lieder), detector-only, excluding model load and core/HTTP/OCR/attachment. Three logical routes share the e4 call; no Dynamic duplicate owner. Admissible candidates: Lyrics36, directions1,153, dynamics499. Preview: http://localhost:8765/text-fusion-calibration-20261001/review/ — original-page overlays with known positives beside raw, uncalibrated candidates; no per-case user decisions requested. Browser check passed: two cases, four original images, no page errors. Results locally in `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/text-fusion-calibration-20261001/`; see `audit.json`, `input-problems.json`, `smoke/{identity.json,detections.jsonl}`, `SMOKE_REPORT.md`, `review/index.html`. Completion watcher finished; no fusion job remains running.

**New prerequisite before broad calibration:** provider `detector_runtime/detector_inference.py` defaults to foreground-component area4, while current HOMR `training/ocr/detector_inference.py` defaults to200. The smoke yields1,688 admissible boxes;1,022 bounding boxes are no larger than10x10. This is a real postprocessing-protocol divergence, not evidence for accepting noisy raw scores or blindly setting200. Next measure a component-area sweep with masks computed once, reporting known-positive recall separately for lyrics/directions/dynamics and false positives only where absence is verified. Then freeze the chosen protocol, verify exact detector-training exclusions, adjudicate unknown overlaps/regions, fit calibration and evaluate the unused test partition. C1 read/verse/note attachment and final full-path latency still gate automatic output. Fusion flags remain disabled, immutable lyrics-only calibration unchanged; no model promotion/deploy.

81 focused Python tests and eight Scanner contract tests pass; backend build passes. Local `.env` includes the two public dynamics pin expectations equal to e4; no enabled flags/provider-revision change. Reference plan: `/home/jhlusko/workspace/homr/docs/TEXT_LYRICS_DYNAMICS_PLAN_2026-09-28.md`.



As of 2026-10-01, `OurTextScores/services/ots-homr-modal/{pins,detector_routes,bootstrap,ots_engine,fusion_v1}.py` implements three logical identities over two physical models. e2 owns Lyrics; e4 supplies Tempo/StaffText/Expression to directions and Dynamic only to dynamics. DirectionText is the shared semantic family without changing checkpoint channel order. The e4 model is loaded/called once, with independent logical calibration/provenance entries. Missing routes, incompatible pins or class ownership fail closed. Dynamic stays review-only even if a future manifest supplies high calibrated confidence; `dynamic.mark` remains prohibited. The specialized glyph reader and C1 read/attachment remain pending.

Scanner now requires independent `SCANNER_EXPECTED_OTS_HOMR_DYNAMICS_{REVISION,SHA256}` expectations in fusion responses. Local `.env` and `.env.example` include these; the current values equal e4 because it supplies both routes. No new model repository/pin is required for this wiring. Existing lyrics-only calibration is unchanged; its historical shadow diagnostics keep their own manifest allowlist. `fusion_v1.py` produces box decisions, not MusicXML text attachment.

Reproducible preparation and production-detector collection: `OurTextScores/services/ots-homr-modal/{prepare_fusion_calibration,collect_fusion_calibration,build_fusion_review}.py`. Outputs and completion status are local under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/text-fusion-calibration-20261001/`. A score-hash split separates calibration and test; exact released-detector training exclusions still need verification before an independent accuracy claim. Existing positives retain source evidence; partially annotated unmatched regions stay unknown. Instrumental OSSQ provides known no-lyrics negatives. Oversized originals above the 30-megapixel preparation budget are deferred and listed without resizing or changing label coordinates. Collection records exact original bytes, production pixel hashes, checkpoint/runtime hashes and raw scores. No fitted thresholds or fake runtime manifest.

Verification: 81 focused Python tests pass, eight Scanner contract tests pass, and the backend builds. The two-page calibration-only real-checkpoint smoke and original-image preview run in persistent tmux `text_fusion_smoke_20261001`; status `SMOKE_STATUS.json`, results `smoke/detections.jsonl`, report `SMOKE_REPORT.md`. Completion automatically appends its result to RUNLOG and the release sequence. Preview on completion: http://localhost:8765/text-fusion-calibration-20261001/review/ . No test-set detections are collected in this smoke. CPU timings are diagnostic, not a replacement for the final GPU/HTTP latency gate.

The fixed warm paired latency run passed 22/22 pages: median 114.604 s, conservative p95
195.091 s, maximum 249.589 s. This leaves only 30.411 s on its slowest page for unmeasured
final-path overhead, so that result alone does not authorize release.

The master's decision 13 (2026-09-25) already approved separate dynamics ownership. B4's
later test failure means e4 remains released. This plan preserves that result rather than
assuming the DirectionText candidate was promoted.
