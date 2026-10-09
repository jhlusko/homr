# Handoff: review-assisted lyrics, fusion-v1 and note anchoring (2026-10-08)

Written for an agent with no prior context. Nothing here is deployed or pushed. The owner's rules still
hold: commit locally, never push OurTextScores (its pre-push hook tears down the local Docker stack),
never deploy or change production pins/env without being asked, and say plainly what is a measured
result versus a development number.

## 1. What exists now

A scanned Lieder page can go through the **local OTS scanner** and come out as reviewable lyric text with
proposed note placements; a person confirms; a copy of the MusicXML gets lyrics. Nothing is automatic.

```
page image -> [GPU test provider: core OMR + e2/e4 text detectors + fusion-v1 + RapidOCR + semantic gate]
           -> backend stores text-fusion artifact (boxes, OCR rows, gate verdicts)
           -> backend builds the lyric candidate inventory (+ staff / printed line / measure / note anchors)
           -> review page /scanner/:job/pages/:n/lyric-review (accept/edit/exclude text, see proposed placement)
           -> build_scanner_placement_inventory.py (tokens <-> MusicXML notes, exact or refuse)
           -> placement confirmation page (owner confirms text, line, note)
           -> import_reviewed_lyric_copy.py -> NEW MusicXML copy with lyrics only
```

All of this is on the OTS branch `c2-fusion-local`, **local commits only**. The shipping Modal provider
image and its three deployed apps are untouched; only a separate test app was ever deployed.

## 2. Repositories and key commits

OTS = `/home/jhlusko/workspace/OurTextScores` (branch `c2-fusion-local`); homr = `/home/jhlusko/workspace/homr`.
Artifacts live under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`
(called `$W` below). The homr `RUNLOG.md` tail (entries 2026-10-07 and 2026-10-08) is the detailed record.

| Area | Where | Notes |
| --- | --- | --- |
| Review page (placement stage 1) | `frontend/app/scanner/lyric-review/` (static inventory) and `frontend/app/scanner/[jobId]/pages/[pageNumber]/lyric-review/` (per job) | flag `NEXT_PUBLIC_OTS_HOMR_LYRIC_REVIEW_ENABLED=1` |
| Backend routes | `backend/src/scanner/scanner.service.ts` `pageLyricCandidates`, `pageLyricCandidateCrop`; `scanner-lyric-candidates.ts`; `scanner-lyric-anchors.ts` | flag `SCANNER_OTS_HOMR_LYRIC_REVIEW_ENABLED=1` |
| fusion-v1 policy | `services/ots-homr-modal/fusion_v1.py`, `fusion_policy.py` (size gate), `calibration/fusion-v1.json` | revision `fadb96af…` (raw 0.7 accept line) |
| Calibration pipeline | `select_lyric_calibration_pages.py`, `collect_lyric_calibration.py`, `lyric_calibration_sample.py`, `fit_lyric_calibration.py`, `build_fusion_v1_manifest.py`, `evaluate_lyric_calibration.py`, `audit_detector_training_membership.py` | fit + eval reports are in `calibration/` |
| OCR + gate | `lyric_ocr.py`, `semantic_gate.py`, `models/semantic-lyrics-v1.json`, `ocr-requirements.txt` | test provider only |
| Test provider | `modal_fusion_test.py` (Modal app `ourtextscores-ots-homr-fusion-localtest`, revision `ots-homr-fusion-ocr-localtest-20261008-v1`) | deploy/stop by hand, see section 6 |
| Placement -> copy | `build_scanner_placement_inventory.py`, `import_reviewed_lyric_copy.py`, `reviewed_lyric_xml.py`, `lyric_placement_handoff.py`, `lyric_placement_review.html` | copy writer refuses on any mismatch |
| Evaluation scripts | `qualify_lyric_ocr.py`, `evaluate_lyric_anchors.cjs`, `evaluate_ocr_row_anchors.cjs`, `build_anchor_judgement_review.py` | |
| Training-membership audit | `calibration/fusion-v1-training-membership-audit.json` | |

Latest commits: OTS `f9122ea` (fragments, ellipsis splits, credit-like), `23cf00f` (far-from-notes + fresh
evaluation record); homr RUNLOG `2433617`. Run `git log --oneline -30` in OTS for the whole chain.

## 3. Measured results (and what each does NOT show)

- **fusion-v1 lyrics calibration** (pinned e2): accept line raw 0.7 = calibrated 0.9315. Bins [0.5,0.7,0.8,0.9] ->
  [0.861, 0.9315, 0.9315, 0.9869]. 218 calibration pages, 150 owner-judged unmatched boxes (128 lyric / 21 not /
  1 unsure). Held-out partition: matched share 53.8%, instrumental false positives 0.80/page (12 on 15 pages,
  small sample). **The "held-out" partition was mostly inside the detectors' training pool** (99/124 and
  77/88 Lieder scores); on the 29 verifiably clean scores matched share 57.8% vs 51.3% (no sign of inflation, but
  only 19 clean judged boxes). e4 directions/dynamics are pinned and uncalibrated: they always stay in review.
  The owner noted that judged boxes sometimes span several lyrics/lines; precision means "contains lyric text".
- **Size gate** (manifest `geometry`): min area 200 (Dynamic 50), max 25% of page. Needed because uncalibrated e4
  emits thousands of specks and a page-sized box that chain every lyric into one conflict (zero accepted).
- **Provider response bounds**: rejected candidates are tallied (`rejected*`), not listed; review list capped at
  200 (`reviewOmitted`). Backend verifies `regions == accepted + review - reviewOmitted`.
- **OCR + semantic gate vs source lyrics** (173 pages, 89 scores): proposed rows lyric-like 87.6%; gate retention
  of lyric-like rows 96.7%; 94.0% exact tokens. Titles/credits pass the gate (handled by rules below).
- **Anchoring**: on the 97 owner-confirmed placements, from syllable positions: measure 97/97, exact note 96/97;
  through the scanner's own OCR rows: exact note 62/67 matched syllables (64% of all 97 end to end).
  **Fresh-page check #1** (11 pages, 132 placements, owner judged): 124 correct, 8 "unsure" which the owner says are
  WRONG (3 OCR fragments/splits; 5 non-lyric credits) -> 93.9% (90% interval 0.896-0.966).
- **Rules added after seeing those results (development, not held out)**: syllable > 100 raster px from every note
  of its measure gets no note, row withheld if none placed (`far-from-notes`); row under no staff withheld
  (`no-staff`); clipped single consonant at a row edge gets no note; runs of 3+ dots split words; imprint
  text withheld (`credit-like`, EN/DE/FR vocabulary + 3+ digit runs). On the 132 judged items these remove
  `n`, `f`, `B.&`, `H.20327`, `PAUL` and change none of the 124 correct placements; `jetzt` moves to the likely
  right note. `DUPONT`/`Bouloi.` were left to the credit-like rule. Width-weighting letters was tried and
  DROPPED (it moved confirmed placements).

## 4. Second fresh sample: DONE (owner judged: 109/120 = 90.8%); see RUNLOG 2026-10-08 'Second fresh placement sample'

(The text below is kept for the record of how it was run. Results, failure types and the new `footer` rule are in the
RUNLOG entry and `calibration/lyric-anchor-fresh-evaluation-2.json`; the next sample, if wanted, is a THIRD one.)

14 new pages were submitted (`$J/fresh2-pages.txt`, 5 clean-score and 9 possibly-trained scores; anchoring does not
depend on detector training). **10 succeeded; 4 never became jobs.** IMSLP80521 (3.2 MB) and IMSLP637242 (2.9 MB)
were rejected with HTTP 413 (the scanner's upload size limit, not a code bug); IMSLP83316 and IMSLP19490 are small and
also produced no job (cause not investigated; check `$J/be.log` around 22:30-23:10 UTC for their POSTs). The test GPU
app was stopped by the script (verify with `modal app list`).

The judgement page for the 10 jobs is built: 120 syllable placements (<= 12 per page), current rules
(no-staff, far-from-notes 100 px, clipped consonant, ellipsis split, credit-like):
`http://localhost:8765/lyric-anchor-judgement-2-20261008/` (key beside the served dir:
`$W/guarded-tuplet-20261004/review/lyric-anchor-judgement-2-20261008-key.json`). Owner convention: C = correct, W = wrong,
U = unsure **= wrong**. When the owner exports (`lyric-anchor-judgements.json` lands in `homr-artifacts/`, overwriting
the first sample's export of the same name, so copy the first one aside first - it is also summarised in
`calibration/lyric-anchor-fresh-evaluation.json`): score with the same arithmetic as that file (accuracy and a 90%
Wilson interval), report failures by type, and do NOT retune the rules on this sample and then call it held out.
The per-page rows/withheld counts from the build: 61112 21/22 proposed, 43630 40/42, 12778 23/23, 16259 21/21,
622104 10/12 (1 no-staff, 1 credit-like), 542017 18/23 (3 no-staff, 1 far-from-notes), 154289 20/21, 44341 51/56,
575893 7/8 (1 no-staff), 97202 18/20.

## 5. Open items, in priority order

1. A third fresh sample to confirm the footer rule and the other post-hoc rules; decide with the owner how lone punctuation pieces (`.`) should be treated (judged correct twice in sample 1, wrong three times in sample 2).
2. Owner decisions pending: merge/deploy of any of this (never assumed); whether the raw 0.7 accept line stays;
   whether to build per-syllable x from the e2 boxes (needs the provider to return ALL e2 boxes; today the stored
   artifact lists only accepted and the top review regions, so counts match the OCR tokens in only 35/91 rows).
3. OCR splits of adjacent words (`jetzt`-type) and merged tokens remain; OCR accuracy on rows is only measured as
   agreement with source syllables, not as character error rate.
4. Verse identity is never inferred (printed line only); the placement page still needs the owner to confirm lines.
5. Real calibration of e4 (directions/dynamics) is untouched; the e4 over-prediction problem is unresolved.
6. The detectors' training membership is an upper bound reconstructed from a hash split; the exact pool is not
   recoverable beyond that. OSSQ's presence in the synthetic bank is unverified.
7. Production path: the shipping `modal_app.py` image has no OCR; wiring it there, adding the manifest to the
   shipping provider and setting `SCANNER_EXPECTED_OTS_HOMR_FUSION_CALIBRATION_REVISION` are deploy decisions.

## 6. How to run things locally

Environment: WSL, Docker infra up (`ourtextscores_mongo` :27018, `_minio` :9002, `_meili` :7700). Python with the
deps: `/home/jhlusko/workspace/homr/.venv/bin/python` with `PYTHONPATH=/home/jhlusko/workspace/homr-artifacts/tools/audit-python:.`
from `services/ots-homr-modal`. Node 24 for the backend; jest in `backend/` and `frontend/`.

Servers (tmux, scripts in `$J`): `lyric-be` = `run-backend.sh` (:4000), `lyric-worker` = `run-worker.sh`,
`lyric-fe` = `run-frontend.sh` (:3000). They set `SCANNER_ENABLED`, the lyric-review flags, and point the local
backend at the TEST provider URL and revision (`gpu-env.sh` holds `SCANNER_EXPECTED_OTS_HOMR_PROVIDER_REVISION` and
`..._FUSION_CALIBRATION_REVISION`). After backend code changes: `npm --prefix backend run build`, then restart `lyric-be`
and `lyric-worker`. Auth for scripts: `bash $J/tok.sh` mints a short-lived HS256 bearer for `jhlusko@gmail.com` from the local
`NEXTAUTH_SECRET` (local stack only). A static review server serves
`$W/guarded-tuplet-20261004/review/` on :8765 (tmux `tuplet-review-server`). `ENOTFOUND mailpit` errors in the
backend log are harmless (host run, no Docker network).

GPU test app: `cd services/ots-homr-modal && PYTHONPATH=. modal deploy modal_fusion_test.py`; stop with
`modal app stop --yes ourtextscores-ots-homr-fusion-localtest`. It costs money only while containers run
(L4, scaledown 60 s) and uses the existing Modal volume/secret; it never touches the three shipping apps. Submit a page
with `bash $J/submit.sh <image>` (creates a job via the API, starts it, polls to terminal state).

Tests: `cd services/ots-homr-modal && pytest -q` (many files; `test_provider.py` needs `fastapi`, absent in that venv);
`cd backend && npx jest src/scanner`; `cd frontend && npx jest lyric`. Backend lint is enforced (`npx eslint --fix`);
three pre-existing prettier errors in `scanner.service.ts` are not ours.

## 7. Cleanup list (nothing here is needed by production)

- tmux: `lyric-be`, `lyric-worker`, `lyric-fe` (kill when done), plus any leftover `lyrcal*`/`lyrtest*`/`lyrocr*`.
- Modal app `ourtextscores-ots-homr-fusion-localtest` (stop it).
- Mongo `scanner_jobs`: roughly 50 local test jobs titled `LYRIC-REVIEW ...` or named after page files, all owned by
  `jhlusko@gmail.com`; MinIO objects under `scores-aux/scanner/416bba7c.../<jobId>/`. Delete only on request.
- Worktree and branch `lyric-review-integration` were removed after the merge.
- Large local data under `$W` (`fusion-lyrics-*`, `lyric-ocr-qualification-*`, `c1-reviewed-import-*`) is evidence; keep.
- The frontend has unrelated uncommitted edits (`ots-homr/page.tsx`, settings); leave them alone.

## 8. Things that bit this session

- `eslint --fix` reformats code, so string `replace` edits made by a script silently miss afterwards; check with grep.
- A shell variable used inside an unquoted heredoc is expanded by bash, not python (it once made four dumps identical).
- The backend job JWT from `tok.sh` lives ~50 minutes; long loops must call it per request (submit.sh does).
- A mid-session API error interrupted the owner's flow once; resume from this document, not from memory.

## Follow-up: upload failures investigated (2026-10-08, 23:28 UTC)

Reproduced local job creation for IMSLP83316 (446,975 bytes) and IMSLP19490 (202,639 bytes), without starting processing. Both returned HTTP 413 with the exact message `This upload would exceed the 50-page rolling 24-hour quota`. No jobs were created and no GPU app was restarted. This is a page quota, not file size. The original four error bodies were discarded by submit.sh, so this reproduction does not conclusively prove their original responses. The original claim that the larger two failed specifically on file size is not established by the retained log; the backend default file limit is 25 MiB.

The indicated historical backend log was overwritten on restart: its current first entry is Oct8 14:03 local. The retained submission log only contains parser KeyErrors for all four missing pages. Durable evidence: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-fresh2-upload-audit-20261008/, including full retry responses, original submission log/scripts/page list, and copied fresh2 inventories/reviews. These are now outside the temporary agent directory.

The available homr-artifacts/lyric-anchor-judgements.json still contains the first 132-item export (items SHA 118da216d74ff84ca9a6a2bcf25e2d7eadf0828d4ec0a73b20a0fe57d4fbce8d); do not score it as the second sample. The second review has 120 cases on ten pages. Next: collect that review export. Retry the missing four only after the rolling quota has capacity or the owner explicitly requests a local quota change; no quota change has been made.


### 2026-10-08 — Second fresh lyric-anchor review scored

Owner export lyric-anchor-judgements-v2.json exactly matches the second sample inventory (120/120 IDs and ordered-ID SHA). Frozen rules: 109 correct, 7 wrong, 4 unsure; unsure counts wrong, yielding 109/120 = 90.83% with descriptive 90% Wilson interval 85.55–94.31% (does not account for page clustering). Ten pages, 12 sampled proposed placements/page; this is conditional correctness, not end-to-end lyric recall, and the four failed uploads are outside the result. First sample remains separately recorded at 124/132 = 93.94%; different samples do not establish a regression or rule improvement.

Failure triage: three standalone dots, three publisher-credit fragments (J., England, Philadelphia), five lyric-text/placement failures (Da, be, trunk', her, sé). Local complete result and raw export: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-anchor-fresh2-scored-20261008/REPORT.md and lyric-anchor-fresh-evaluation-v2.json. Scoring code and two provenance/completeness tests committed in OTS. No retuning on this frozen result.

Next priorities: punctuation-only suppression; whole-row publisher-credit diagnosis; exact text/ownership checks for the five lyric failures. Any replayed improvement on these judged pages is development evidence and needs a new frozen-rule sample. Review-assisted attachment remains appropriate; unattended attachment remains unqualified. GPU app remains stopped; no production changes.


### 2026-10-08 — Punctuation filtering and both-sample development replay

Added source-free punctuation-only syllable suppression in scanner-lyric-anchors.ts. Preserves punctuation attached to words and all original character widths, so omission cannot shift neighbouring anchors. The preceding agent already committed a 92%-page-height footer guard (2f45795); replay confirms it withholds the three publisher-credit errors from sample2. No publisher-name vocabulary added.

Sample2 development replay: six incorrect proposals withheld (three standalone dots, three footer credits), all 109 owner-correct proposals retained with unchanged anchors, five original errors remain. Retained conditional correctness 109/114 = 95.61%, a development result; frozen fresh evaluation remains 109/120 = 90.83%. First sample: 121 of 124 owner-correct original proposals remain unchanged; two dots judged correct are withheld (crops show printed hyphens), and the previously accepted `ziehn,......wir` proposal is split by the earlier ellipsis rule. Seven original errors withheld; `jetzt` remains but its position changes, so do not claim it fixed without new judgement. Original owner labels preserved throughout.

Receipts, source snapshots and durable first-sample inputs: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-anchor-development-replay-20261008/sample1-v2.json and sample2-v2.json. The unsuffixed first replay mistakenly treated moved text as removed; v2 matches unchanged geometry first and then uniquely matching text, explicitly marking changes. Thirty-six focused backend tests and scoped lint pass. No running backend restart, GPU run, production update or new inference occurred.

Next: diagnose the five remaining sample2 lyric-text/placement failures against complete OCR rows and source/reference context; distinguish text mistakes from note ownership. Avoid moving anchors on accepted controls. Then freeze and gather a new sample when quota permits. These two judged samples are now development evidence for the changes.


### 2026-10-08 — Five remaining failures diagnosed; review display ambiguity

Built focused page http://localhost:8765/lyric-anchor-five-diagnostics-20261008/ with all five original crops, full OCR rows, retained owner judgements, and diagnostic source context. `Da` has a visibly displaced decoder-attention marker; `trunk'.` contains an extra OCR period. `be`, `her` and `sé` agree with matching source lyric/note context (D5 at source measure60 ordinal1; A4 at measure6 ordinal2; A-flat4 at measure3 ordinal3 respectively). This is diagnostic passage comparison, not certified scan/native linkage and not grounds to change owner labels or the frozen 109/120 score. Asked owner whether those three failures refer to text, blue marker placement, or target identification inside the broad orange band.

The blind judgement page did not show the recognized syllable. A fixed +/-45px orange band can cover multiple syllables or a printed hyphen; the blue line is decoder attention/interpolation rather than a verified notehead. Updated the generator/template for future reviews to show the proposed syllable, pitch and coordinate provenance and clarify the band. Existing frozen reviews remain unchanged. Diagnostic page/report/all five images verified over localhost; scoped Python formatting/lint pass.

No new inference policy or vocabulary fix added for these five: moving source-agreeing anchors or stripping punctuation broadly would risk accepted lyrics. Next: resolve the three ambiguous judgements, then investigate genuine coordinate errors with detected notehead/character geometry instead of tuning on misleading display markers. No GPU or production action occurred.


2026-10-08 owner clarification: IMSLP16259 `trunk'.` concern was that the orange marker covered only `unk`; after confirming the actual OCR row captured the whole word, owner says the case is fine. Saved clarification separately at homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-anchor-fresh2-scored-20261008/owner-clarification-trunk.json; original export and frozen 109/120 result unchanged. Owner also says some other cases have the same explanation; asked which of be/her/sé rather than assuming all three. Recognition text still includes an extra period; do not present this clarification as exact-token accuracy.

2026-10-08 further owner clarification: `her` explicitly accepted. `sé` might be fine; concern is orange band covering neighbouring words. `be` concern is included printed hyphen. Verified the frozen exported syllable strings are exactly `sé` and `be`, respectively; the hyphen and neighbouring syllables belong only to the fixed-width marker overlap, not these syllable strings. Preserve be/sé as pending clarification rather than silently relabel. Separate receipt: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-anchor-fresh2-scored-20261008/owner-clarification-her-se-be.json.


### 2026-10-08 — Owner accepts marker-ambiguity cases; next priorities

Owner accepted trunk', her, be and sé after clarifying that orange bands are display markers, not OCR boundaries. Clarified review: 113 acceptable/120, seven remaining errors (three standalone dots, three footer-credit fragments, Da's displaced blue marker). This is post-review clarification, not new blind validation or exact-token accuracy; original frozen 109/120 export retained unchanged. Current development filters withhold the six punctuation/footer cases; Da remains. Separate local receipt: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-anchor-fresh2-scored-20261008/owner-clarified-summary.json.

Priority order: (1) replace the fixed-width orange band with a clearly labelled position marker and show actual OCR-row bounds separately; (2) investigate Da using actual detected notehead geometry rather than arbitrary attention shifts, checking accepted controls; (3) freeze the revised pipeline and qualify new Lieder and OSSQ samples, reporting conditional placement correctness and end-to-end lyric recall separately; (4) complete local review-to-MusicXML-copy integration and provider latency qualification before any promotion. No new GPU app or production change authorized by this next-steps discussion.


### 2026-10-08 — Clear review markers and source-free Da notehead diagnosis

New review page: http://localhost:8765/lyric-anchor-clear-markers-20261008/. Replaces fixed-width orange boxes with narrow estimated-position ticks and separately outlines complete OCR rows in grey. Recognized syllable, pitch and coordinate provenance remain visible. Same 120 case IDs/order; original reviewed page/export preserved. Browser verified 120 labelled targets and corrected legend. No need to redo the whole review.

Local CPU segmentation (four threads, ~2.4 s inference) on IMSLP43630 first-staff crop detected the Da head at raster x323.32, 3.18 px from lyric x326.50. Old decoder attention x401.48 is displaced 78.16 px. A diagnostic shape-filtered head set and locally tracked five staff lines yield diatonic positions [3,3,3,8,8,7,1], matching all seven decoded notes; existing forced-sequence association returns seven unambiguous pairs without native text or note labels. Staff-line tracking and component thresholds are development diagnostics, not qualified runtime policy.

Focused visual: http://localhost:8765/lyric-da-notehead-check-20261008/. Local source/model/image receipts, masks and exact executed scripts: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-da-noteheads-20261008-v2/report.json and sequence-report.json. First initial extraction failed on omitted API arguments; the suffixed directory is the completed run. Sequence helper initially refused beam fragments/long-span skew, then tracked local staff-line continuation with slope extrapolation; preserve this limitation for qualification. No score rewrite, accepted-control remapping, provider/GPU start or production change.

Next: qualify detected-head associations on accepted Lieder/OSSQ controls, preserving ambiguity refusal and separating detected-head positioning from lyric OCR-character positioning. Then wire only qualified associations into local scanner review. This selected Da result is development evidence, not 100% independent accuracy.


2026-10-08: Added optional Correct syllable textbox to the active clear-marker judgement page (http://localhost:8765/lyric-anchor-clear-markers-20261008/). Stores corrections by frozen item ID, preserves browser judgements, exports original/edited text in a separate syllableCorrections map. Typing cannot trigger C/W/U or navigation shortcuts. Verified typing isolation, reload persistence, JSON export and clearing in an isolated browser; no test decisions imported as user approvals. Text edits do not automatically change note placement or judgement. Saved pre-textbox HTML/receipt beside the current page; original frozen reviews remain unchanged.


### 2026-10-08 — Corrected-syllable export validated

Received lyric-anchor-judgements-w-corrections.json: exact 120-ID inventory/ordered-ID hash; five explicit corrections, each originalText matches the frozen key: IMSLP575893 pourton→ton and eon→con; IMSLP43630 Da→du and a→zen; IMSLP16259 .→ge. All stored as reviewed text requiring position refresh, not confirmed attachments. New scorer validates correction membership/original text/nonempty replacement; five tests and lint pass. Separate durable raw export and scored report: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-anchor-fresh2-scored-20261008/review-with-corrections-report-v2.json. Unsuffixed report predates final lint change; v2 binds final source.

Raw new export has 108 correct, 10 wrong, 2 unsure (90% if unsure counts wrong). It still carries trunk'/her unsure and sé wrong despite earlier verbal clarifications; keep explicit verbal clarifications separate rather than silently overriding the newly exported labels. This is a revisited development review, not a new fresh sample.

The .→ge correction reveals a genuine OCR-lost syllable, so punctuation filtering must withhold automatic placement but retain the original row/fragment for manual recovery. Do not claim every punctuation rejection is a true negative or delete the row. Da also needs the user-confirmed du text, in addition to fixing displaced attention. Next: reconstruct positions for these five edited syllables from fresh character/glyph geometry; retain ambiguous replacements for manual placement rather than applying string swaps at old centers. Frozen evaluations, source scores and production unchanged; GPU stopped.


### 2026-10-08 — Five corrected syllable positions recovered locally

Focused review: http://localhost:8765/lyric-corrected-syllables-20261008/. Shows only the five explicitly edited syllables (ton, du, con, zen, ge), fresh orange text positions, grey original OCR-row bounds and proposed blue note positions. Confirmations default pending and export separately; isolated browser test confirmations are not user approvals. No need to repeat the 120-case review.

Fresh CPU OCR character boxes recover ton, con, zen and ge at unique positions stable across 64/96-pixel crop paddings. Narrow crops had missed zen and ge because the old OCR regions clipped their text. OCR still reads du incorrectly (De/Da): du uses the owner's explicit correction plus a single visible word-ink region, matched to the independently detected head. Do not describe du as successfully recognized by OCR. All five refreshed centers retain their original decoded measure/note/pitch identities; du's displayed note coordinate now follows the detected head rather than displaced attention.

Local evidence: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-corrected-crop-geometry-wide-20261008/report.json, placement-comparison.json and du-word-ink-report.json. The focused review folder contains the generator, images, items and SHA-256 receipt. Narrow-run evidence remains in lyric-corrected-crop-geometry-20261008. These are selected development diagnostics, not independent accuracy results or certified character boundaries. No MusicXML edits, runtime promotion, provider changes or paid GPU starts.

Next priorities: owner checks the five revised positions; validate/import those explicit confirmations before any reviewed MusicXML copy; qualify detected-head association on accepted controls and new Lieder/OSSQ samples; then complete local copy integration and provider latency gates. Preserve withheld punctuation/row context for recovery: the period-to-ge correction demonstrates that punctuation suppression can hide a genuine lyric.


### 2026-10-08 — Owner confirms all five corrected placements

Owner explicitly stated "all the updates are correct" after the focused five-case review. Recorded all five text/note confirmations separately, bound to the exact items.json SHA-256: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/lyric-corrected-syllables-20261008/owner-confirmations.json. This is a chat-derived confirmation, not a browser export or independent evaluation; original pending inventory, frozen scores and review receipts remain unchanged. No further owner review of these five is required.

Next: connect this corrected-placement confirmation format to the validated local score-copy importer, preserving source hashes, explicit printed-line identity and all non-lyric content. The existing importer accepts a different placement-handoff schema; do not bypass it or treat the new confirmation JSON as already importable. Then qualify geometry on accepted controls and fresh Lieder/OSSQ samples before runtime promotion. No MusicXML written, GPU started or production changed.


### 2026-10-08 — Five owner-confirmed corrections imported into local score copies

Completed corrected-placement adapter and applied all five explicit owner confirmations through the existing validated review/copy writer. Retrieved original page MusicXML read-only from the three existing local scanner jobs; retained originals and retrieval SHA-256 receipts at homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-corrected-copy-inputs-20261008/. No new inference or GPU app was started.

Copies and exact source/code/confirmation hashes: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-corrected-musicxml-copies-20261008/. Each score subdirectory contains score.musicxml, validated-plan.json and receipt.json. IMSLP575893: ton at generated measure 3, note 9; con at measure 8, note 6. IMSLP43630: du at measure 1, note 1; zen at measure 15, note 5. IMSLP16259: ge at measure 7, note 2. These are generated page measure indices, not native source numbering. Printed lyric line 1 is retained for four corrections and line 2 for ge; labels remain printed-line-N rather than inferred native verse numbers. Original accidentals, durations, notes and all other non-lyric content are unchanged, verified by the writer.

Adapter binds the exact reviewed items hash, complete unique IDs, explicit boolean confirmations, corrected text, unchanged decoded note identity, original OCR row/staff/printed line and token identity. Aligns the entire relevant decoded part to its MusicXML note/rest sequence, then verifies selected pitch and records exact part/measure/ordinal/onset/duration/alter. Only relevant parts are mapped: unrelated piano parts can fail alignment and are neither attached nor modified. Refuses mismatches and unconfirmed placements; writes only new exclusive output directories. All 26 focused adapter/handoff/copy tests passed; scoped formatting/lint passed. Test decisions are not owner approvals.

This closes the selected five-case local copy-import step, not overall lyric qualification. Next: qualify detected-head geometry on previously accepted controls, then freeze and evaluate new Lieder/OSSQ samples for both conditional placement accuracy and end-to-end lyric recall. Provider latency and runtime/UI integration remain gates before promotion. Original frozen scores and evaluations unchanged; production and GPU app remain untouched.


### 2026-10-08 — Earlier accepted lyric judgements are provisional

Owner warns that previously accepted examples were judged using imperfect bounding boxes. Preserve exports and historical scores, but do not use those labels as a reliable correctness gate, regression oracle or calibration truth. Clear-marker reinspection or exact source-ground-truth linkage is needed before correctness claims; geometry/sequence consistency alone is not placement accuracy. The five corrected placements explicitly accepted on the clarified page remain separately confirmed.

Continue with source-free geometry checks on the existing 120-case development sample, refusing ambiguous staff/sequence associations and reporting coverage/displacement without accuracy claims. Then qualify on fresh Lieder/OSSQ examples with clear OCR-row bounds, narrow text position markers and actual notehead targets; keep conditional correctness separate from end-to-end lyric recall. No broad redo of the old review requested.


### 2026-10-08 — Source-free head/attention audit completed; provisional labels excluded from gates

Audited all 120 second-sample placements across 41 decoded staves on local CPU in persistent tmux session lyric-geometry-audit-20261008. Completed v3 in 70.2 seconds: 28 staves sequence-checked, 13 refused; 75 placements have forced equal-pitch sequence associations, 31 unresolved due to geometry and 14 due to sequence ambiguity. Coverage is 75/120, not accuracy or end-to-end recall. Of matched placements, median old-attention/head displacement is 7.1 raster pixels; 26 are at least 10 px and two at least 30 px (largest 44.5 px). No automatic note reassignment or score changes.

Review: http://localhost:8765/lyric-geometry-audit-20261008-v3/ shows the 20 largest differences with green detected-head circles, blue old-attention lines and narrow orange original lyric-position ticks. Text is original OCR, including errors, not newly corrected text. Two largest (own, tes) visually show the old attention shifted away from the head above the syllable. This is agent visual inspection, not owner approval or new ground truth. No broad rerun of human judgement is required at this stage; existing acceptances remain provisional per owner warning. The explicitly accepted five corrected placements remain separately recorded.

Evidence and executed source/model/input hashes: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-notehead-batch-audit-20261008-v3/report.json and summary.json. v2 preserved (22 associations before inheriting clefs); unsuffixed initial attempt failed on a helper variable before completing. v3 inherits clefs only from preceding decoded staves of the same part; refuses within-staff clef changes. Twelve staves have missing/ambiguous local five-line geometry and one has an off-grid head. Component/line thresholds are conservative development settings, not qualified runtime defaults. Script, reproducible review builder and six geometry tests added; 21 geometry/association tests and scoped lint passed. All 20 review images served HTTP 200 and the two largest differences were visually inspected. CPU audit finished; tmux retains an idle shell, no GPU job running.

Next: investigate the 13 refused staves with better staff-line context, retaining ambiguity refusal; then collect fresh Lieder/OSSQ qualification with clear markers and exact ground-truth linkage where available. Do not gate on old acceptance labels or count sequence matches as correct placements. Frozen review exports/statistics and original scores preserved; no runtime/provider/pin change or paid app start.


### 2026-10-08 — Broader staff-context experiment and adjacent-staff contamination diagnosed

Completed local geometry experiments while treating all older acceptance labels as provisional. Wider 40/80/120-pixel half-windows with at least two agreeing line sets yield v4 coverage 76/120. Long-range line support plus consistent diatonic position across windows yields v5 75/120, recovering 12 and refusing 12 relative to v3; no retained head coordinate moves by more than 1 px. Broader windows alone do not establish an improvement. Masks/source crops saved in v4 permit exact cached replay without further inference.

Visual inspection of IMSLP43630 staff 3 shows clipped noteheads from the neighbouring staff above the actual vocal lines. v6 excludes a component outside the unique long-range central staff frame plus three staff spacings only when that frame is unambiguous, then still requires agreement across windows and forced pitch sequence association. Eight components excluded across successfully checked staves. Final v6: 79/120 forced associations, 27 unresolved geometry, 14 unresolved sequence; 29 of 41 staves checked and 12 refused. Relative to v3, 16 gained and 12 lost; no shared association moves by more than 1 px. These are coverage/stability counts, not accuracy, recall or a passing qualification gate. Do not combine the runs' matched sets to inflate coverage.

Remaining refusals: seven insufficient-window-support staves, two off-grid heads, two competing staff-line sets, one diatonic disagreement across windows. Whole-staff refusal remains conservative; no weakened match threshold or arbitrary neighbour reassignment. Latest diagnostics: http://localhost:8765/lyric-geometry-audit-20261008-v6/ shows 20 matched examples and all 12 unresolved source staves. Green=associated detected head; blue=old attention; orange=old estimated text position; purple=raw components (possibly artifacts), not confirmed notes. No new human review is required for this diagnostic page.

Durable local evidence: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-notehead-batch-audit-20261008-v6/report.json, comparison-v3.json and mask-reuse-receipt.json. v4 holds inference masks; v5/v6 replay those masks in ~3 seconds with original input/model checks and mask hashes. Previous reports/reviews preserved. Updated audit and review builder; all 26 geometry/association tests plus scoped formatting/lint passed. Latest HTML and all 32 images return HTTP 200. All CPU jobs finished; persistent tmux retains an idle shell. No score copies rewritten, runtime/pins changed or GPU app started.

Next priority: establish exact scan-to-native ground-truth links for lyric text and note identity on fresh Lieder/OSSQ qualification cases before more threshold tuning or runtime adoption. Diagnose unsupported/ambiguous staff frames separately rather than treating previous human acceptance as truth. Keep conditional placement correctness, association coverage and end-to-end lyric recall separate. The five explicitly accepted corrected local score copies remain unchanged.


### 2026-10-08 — Native lyric references recovered; corpus roles corrected

Audited the fixed fusion-lyrics-test-20261008 pool without recognizer predictions: 143 pages (128 Lieder, 15 OSSQ). Forty Lieder pages from 27 scores have unique local native MusicXML and complete count-exact page/system windows, yielding 1,605 native lyric targets. Seventy Lieder pages have incomplete/ambiguous page boundaries; 18 have missing/ambiguous source identity and remain excluded. Native references retain exact part, physical measure, note ordinal, voice/staff, onset/duration, pitch/alter, verse, text and lyric segments/elisions. These are genuine native ground-truth identities, but page-count linkage does not certify syllable/note pixel ownership or edition fidelity.

Correction to prior next-step wording: OSSQ is instrumental quartet data, so it is a no-lyrics false-positive control, not a positive lyric-recall corpus. All three native OSSQ sources represented by the 15 pages contain zero lyric targets. Scan edition/page linkage remains unverified; native absence does not itself adjudicate every printed box. Existing cached lyrics-detector outputs cover exactly all 143 frozen page IDs with matching image hashes; OSSQ has 109 raw detector proposals. This is before final fusion/OCR, not the final false-positive rate. Older projected precision derived from imperfect box judgements is not a promotion gate.

Fixed a deterministic ten-page Lieder follow-up, one per score, outside both older anchor-review score inventories, with 398 native lyric targets. One selected score (IMSLP173063) has two lyric-bearing parts; never assume one vocal part or collapse its identities. All selected scores have known index exposure; zero pages are certified untouched. Call these new review/evaluation cases, not untouched model tests. Native sources are evaluation-only and were not passed to inference.

Final formatted-source-bound local artifacts: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-native-reference-readiness-20261008-v3/README.md, report.json, candidate-pages.json, native-page-lyrics.json and ossq-cached-proposal-ledger.json. v1/v2 retained; v3 includes elision text and binds final formatted code. Initial direct-file invocation failed before output due module search path; successful command is .venv/bin/python -m training.onnx.audit_lyric_reference_readiness. Reproducible tool and two native-identity/polyphony/elision tests added; formatting/lint/tests passed. All local jobs complete, persistent reference-audit tmux shell idle; GPU app remains stopped.

Next: verify scan/native edition and complete-page/system correspondence for the fixed ten and OSSQ controls, freeze those links, then score the frozen pipeline against native note/timing/accidental/text identities. Do not optimize native offsets or verse permutations to maximize a claimed release score; if used diagnostically, label them oracle-only. Independently establish pixel ownership where needed and report conditional placement correctness separately from end-to-end recovery. Existing 120-case geometry results remain development coverage; five owner-confirmed MusicXML copies unchanged. No runtime/pins/env/prod changes.


### 2026-10-08 — Native scan/source passage review rendered and inspected

Built reproducible local review with MuseScore 4.6.5: http://localhost:8765/lyric-native-linkage-20261008-v2/ . All 39 system fragments across the fixed ten Lieder pages rendered successfully; all 74 HTML/image URLs returned HTTP 200. Verified every fragment preserves all native note XML, including lyrics, across every part. Six reference/certification tests pass. Sources and scan hashes remain unchanged; no recognizer predictions were used to establish reference passage linkage.

Visually inspected all 32 system pairs on eight pages with matching cached geometry: musical passages, lyric sequences and first/last measure anchors correspond, including both vocal parts in IMSLP173063. This is agent passage/boundary evidence, not owner approval, exhaustive notation accuracy, or syllable pixel certification. Exact-text caveat: IMSLP60848 system 2/native physical measure 116 prints historical “Thränen” while native source spells “Tränen”; retain an explicit scoring exception rather than silently modernizing scan ground truth. IMSLP588989 native title also has “Le hemin du ciel”; metadata left unchanged.

Seven systems on IMSLP508574 and IMSLP621598 remain pending geometry verification. The former scan is 2481×3508 while cached geometry is 1866×2575; the latter is 2644×3533 versus 2644×3532. Review deliberately shows the full scan for these cases, without assumed scaling. Initial unsuffixed preparation stopped before rendering on this mismatch; v2 completed and preserves warnings. Rendering alone is not certification.

Local evidence: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/lyric-native-linkage-20261008-v2/report.json, visual-inspection.json, verification script, native fragments/PDFs/logs and 32 inspection contact sheets. Tool: homr/training/onnx/build_lyric_native_linkage_review.py. Render tmux session lyric-native-render-20261008 is now an idle shell; no running GPU/Modal task.

Next, in order: compare the cached build-page pixels to the two frozen scans and establish any exact coordinate transform from evidence; finish their seven passage checks; verify the three OSSQ instrumental source/scan controls; freeze the admissible reference links and score the frozen pipeline with text exceptions reported separately. Native timing/part/note identities are evaluation-only. Historical imperfect-box approvals remain provisional; five separately confirmed corrections unchanged. No release gate passed, runtime/provider/pins/env/prod unchanged.


### 2026-10-08 — Owner reviewed first half of native linkage page

Owner reports: “i reviewed the first half found no mistakes”. Recorded against the rendered report SHA in homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/lyric-native-linkage-20261008-v2/owner-review-first-half.json. Exact last reviewed case was unspecified, so this is partial review evidence, not individual-case or whole-set approval. Remaining review and reference qualification stay open.

Compared both mismatched frozen scans to the corresponding cached build-page files. IMSLP621598 differs in actual pixels under both top and bottom alignment, despite only one pixel of height difference; dimensions alone do not justify reusing coordinates. IMSLP508574 also has different dimensions/aspect ratio. Saved cached file hashes and comparison results in cached-page-pixel-comparison.json beside the review. Seven system links remain pending verified registration or fresh scan geometry; do not silently rescale.


### 2026-10-08 — Whole native-linkage review approved; qualification and integration parallel

Owner gives whole-set approval for the displayed ten-page native scan/source correspondence review and authorizes proceeding with frozen text-fusion qualification and local OTS review-assisted integration in parallel. SHA-bound approval: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/lyric-native-linkage-20261008-v2/owner-review-whole-set.json. Supersedes partial review scope; does not certify unknown pixel coordinates, unseen-test independence, exact source text where edition spelling differs, or release gate success.

Active tracks: (1) freeze/evaluate actual lyric detector/OCR/filter behavior with Lieder positives and OSSQ negative controls, preserving omissions and text exceptions; (2) wire corrected syllable text and explicit printed-line/note confirmation to validated MusicXML-copy output locally behind the flag. Root resolves remaining scan geometry/reference prerequisites and consolidates results. Unattended attachment, production deployment/push and paid GPU starts remain unauthorized. No need for further owner review of the approved linkage set.


### 2026-10-08 — Remaining seven passage checks and OSSQ no-lyrics scan controls complete

Resolved display-context prerequisite without scaling stale cached coordinates: marked broad system contexts directly on the two frozen original scans, checked all seven proposed native passages and first/last anchors. All 39 systems now have visual passage/boundary evidence; original report and partial inspection receipt remain immutable. Additive manual-system-context.json and visual-inspection-complete.json bind crop/inspection/source/owner-approval hashes. Context supplement: http://localhost:8765/lyric-native-linkage-20261008-v2/manual-context.html . This does not certify exact syllable/note pixels. Additional native text exception IMSLP621598 physical measure 14: printed Thal versus native Tal. Both this and Thränen/Tränen remain explicit exceptions, not silent source edits.

Independently inspected all 15 frozen OSSQ full scans across three scores: instrumental four-string layouts, no sung lyric rows. SHA-bound selection/contacts/visual-review.json at homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-lyric-negative-scan-check-20261008/. This establishes scan no-lyrics control status, not exact native measure alignment or exhaustive notation fidelity.

Parallel qualification discovers no cached OCR for the selected ten or OSSQ controls: old OCR cache covers another 173-page pool. Fresh pinned production lyric OCR plus frozen semantic replay is running locally against exact cached e2 proposals, without source labels entering inference. Non-lyric/dynamics detector routes and generated note/verse ownership are not supplied by these caches, so complete fusion/end-to-end attachment qualification must remain separate. Local OTS integration is implementing explicit recognised-note choices and a confirmed corrected-copy download behind the existing lyric-review flag. No original score overwrite, unattended attachment, paid app start, deployment or push.


### 2026-10-08 — Approved native links frozen; fresh OCR stage complete

Reproducible freeze tool refuses partial approvals, changed source/scans/inspection pixels, differing window coverage and output overwrite. Final formatted-code-bound manifest: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-approved-native-links-20261008-v2/{pages.json,receipt.json}: 10 Lieder pages, 39 systems, 398 native lyric targets. Five guard tests pass. Owner approval establishes passage correspondence; exact syllable pixel ownership and unseen-test independence are explicitly false. Both historical spelling exceptions retained. Earlier unsuffixed manifest preserved; v2 binds final lint-clean code.

Fresh production OCR/semantic replay completed EXIT 0 in 226.3 seconds: Lieder 235 e2 proposals at raw confidence >=0.5, 219 OCR rows, 211 semantic lyric suggestions; OSSQ 24 proposals, 22 OCR rows, six false semantic lyric suggestions on five scan-verified no-lyrics pages. These are before backend staff/credit/footer filtering, not final false attachments or recall. Native targets are not OCR rows; do not divide counts to invent accuracy. Artifacts: homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/frozen-lyric-stage-qualification-20261008/{report.json,negative-control-addendum.json,README.md}; production detector checkpoint SHAs verified separately.

Complete e2/e4 detector -> logical routes -> fuse_page replay runs in persistent tmux frozen-fusion-cpu-20261008 with frozen-fusion-watch-20261008 completion watcher; result pages.jsonl and input-binding.json at sibling frozen-fusion-cpu-20261008/, log at frozen-fusion-cpu-20261008.log. First ETA 15–30 minutes provisional until a page completes. A parallel supervised private-FP16 CPU recognition context job is being prepared for the fixed ten Lieder pages to supply generated staves/notes for the real backend filters. This is explicitly CPU evidence, not production GPU latency. No production pin/env/deploy/push changes.



### 2026-10-08 — Local reviewed lyric-copy workflow validated; full fusion still running

Local OTS now offers page-specific lyric review, exact recognised MusicXML note choices, editable syllables/printed lyric lines and three independent confirmations before downloading a corrected score copy. Accepted, unchanged OCR rows can propose aligned syllables/notes, but proposals remain unconfirmed. A hash-bound draft cannot overwrite newly added rows; proposals wait until the note inventory loads. The scanner page links to review behind the existing lyric-review flag. The download preserves the stored score and all non-lyric XML. It adds text-only lyrics: hyphenation, melismas and native verse-number mapping still require later editing. Implementation/workflow: OurTextScores/docs/LYRIC_REVIEW_LOCAL_WORKFLOW.md. Local example: http://localhost:3000/scanner/a2c06b70-0a51-4f11-830f-b270d8a24675/pages/1/lyric-review . No additional owner review is requested now.

Validation: three focused backend suites (14 tests), two frontend suites (13 tests), and the authenticated Playwright corrected-copy smoke passed. Browser test checks the downloaded syllable, unchanged stored XML SHA, scan access and unauthenticated rejection. Docker frontend compilation and public-link guard passed; image assembly is still running at this entry and is not yet counted as a completed Docker build. Core integration commits: 60001da and 2255c7a in OurTextScores. No deployment, push, provider pin change or .env edits.

Qualification: all ten Lieder recognition contexts succeeded. OSSQ controls are running (6/15 successful at this checkpoint; roughly 15–25 minutes remaining). Full pinned e2/e4 detector/fusion replay is 5/25; revised ETA 2–4 hours, superseding the initial optimistic 15–30 minutes. Persistent tmux sessions: frozen-fusion-cpu-20261008, lyric-recognition-ossq-20261008, lyric-final-filter-watch-20261008. The final watcher waits for fusion and both recognition sets, records prerequisite failures, then reruns production OCR from the fresh e2 detections and applies actual backend filters. It writes local artifacts below homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/: fresh-fusion-lyric-ocr-20261008/ and lyric-backend-filter-replay-20261008/. Completed Lieder contexts: lyric-recognition-context-20261008/results/complete.json. Its earlier watcher misrecorded exit 127 after losing the original status; additive watcher-exit-correction.json preserves that limitation and points to the ten verified successful outputs.

Release priorities remain: complete and assess those filtered outputs on Lieder and OSSQ; measure meaningful ground-truth syllable/note ownership separately from passage approval; finish production GPU/HTTP latency and final pinned-stack checks; then refresh public examples/claims and obtain deployment approval. Native whole-set approval covers passage correspondence, not exact syllable pixels or an independent unseen test. Review-assisted lyrics are the accepted release scope; unattended attachment remains unqualified.

