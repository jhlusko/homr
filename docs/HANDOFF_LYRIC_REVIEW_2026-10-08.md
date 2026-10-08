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

## 4. Second fresh sample: built, awaiting the owner's judgement

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

1. Collect and score the owner's judgements of the second fresh sample (above).
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
