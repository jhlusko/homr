# C1 OCR-text pilot and external controls — 2026-10-07

## 2026-10-07 — C1 text pilot fails useful recall; private-fp16 smoke running

Parallel continuation completed a fixed OCR-text pilot and separate real-scan
control evaluation. Training-only character2–5gram TF-IDF, balanced logistic loss,
L2lambda0.01; fixed threshold0.95, scores uncalibrated, empty OCR alwaysUnknown.
152 nonempty training regions (118 lyrics,34 directions) from73scores; original
26 empty training lyrics still count in coverage. Internal calibration27scores:
all63regions Unknown, including52lyrics (7empty) and11directions. Zero false
lyric selections is vacuous with zero recall; automatic fusion stays disabled.

External frozen574fullpage OCR rows from10scores have no overlap with any100
development scores. All574Unknown. Expanded42controls retain0/26lyrics versus
26/26 prior e2 selections; directions0/15 false versus2/15 previously, one OCRmiss
preserved. Original6 human-confirmed directions allUnknown. Conditional veto of
old a tempo2tokens/2notes and Revenez4tokens/2notes is not fresh attachment accuracy;
rejecting every lyric is not a useful repair. Unmatched fullpage rows unadjudicated.

Exploratory internal thresholds were reported, never selected:0.5 selects45/52
lyrics but3/11 false directions;0.6 gives42/52 and1/11;0.7 gives25/52 and0/11.
These small development results do not establish a release operating point.
Next C1 work: use score-grouped training-only cross-validation to choose model
regularization and an abstaining operating point before another frozen check;
expand real annotated training evidence and title/credit/OSSQ controls, preserving
score exclusions. Do not simply lower the threshold against these known controls.
Actual final note-attachment precision/recall remains a separate gate.

Review: http://localhost:8765/c1-semantic-pilot-20261007/ (48cases,90assets verified).
Pilot: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-text-pilot-20261007/`.
External: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-external-controls-20261007-v4/`.

Private provider preflight: all8 pinned model/checkpoint/manifest/detector inputs
match hashes; all3 recognition graphs load CPU sessions. CPU-only local providers;
no CUDA available. Checked recipe calibration path lacks fusion-v1.json, not a
claim that no calibration exists elsewhere. Separate private-fp16 positive on/off
HTTP smoke now running, true plain/head masks, known LiederIMSLP634835 and OSSQsq9631717.
Persistent tmux tremolo-private-fp16 and tremolo-private-fp16-watch; no retry on
failure, automatic comparison/native audit after successful scans. FirstoffLieder
HTTP20028.39s; launchETA8–15min, currentremaining4–8min. No final fp16 gains claimed.
Artifact: `/home/jhlusko/workspace/homr-artifacts/local-tremolo-private-fp16-20261008/`
(dirname is reserved ID, work dateOct7). Watch-status.json is the live progress
receipt. Full OCR/fusion/GPU280-second qualification still pending. No shipping
pin, backend environment, production deployment, runtime classifier or push.
Nineteen focused semantic/OCR/preflight tests pass; new code scoped lint passes.


## 2026-10-07 — Private fp16 positive smoke passed; negative controls next

Exact private pinned recognition graphs now tested through actual supervised
plain and heads workers on both timing states. All four HTTP scans succeeded;
actual shell exit0, distinct head routing and native audits passed. Both candidates
show Lieder30→10 rhythm errors/357 and OSSQ46→2/380. Pitch and accidental errors
unchanged at3/333 and2/372 respectively, zero regressed native measures; tokens,
written-note inventories and measure counts unchanged. These known development
pages demonstrate compatibility, not independent corpus-wide accuracy.
Local CPU page seconds off→on: Lieder28.39→29.67, OSSQ51.55→55.59; both workers,
OCR/text detectors/fusion disabled. No GPU/production280-second claim.

Final receipts: `/home/jhlusko/workspace/homr-artifacts/local-tremolo-private-fp16-20261008/comparison.json`,
`native-score-report.json`, `watch-status.json`, `run-exit.json` alongside it.
Positive job and watcher finished. Next fresh private-fp16 negatives use the same
five predetermined controls already tested on experimental fp32, with their own
isolated arms and persistent watcher. Model-loaded GPU and full text-path latency
qualification still pending. C1 pilot remains rejected (zero retained lyrics).
No production/pin/environment/runtime classifier changes or push.


## 2026-10-07 — Training-only grouped C1 sweep finds no useful safe policy

Ran20 fits: five deterministic score-disjoint training folds for each fixed
L2lambda0.0001/0.001/0.01/0.1. Character TF-IDF vocabulary/IDF and logistic model
are learned only from each fold's nonempty training OCR; no held-score text in fit.
Every configuration has178 unique out-of-fold predictions (144lyrics,34directions),
including26emptylyrics asUnknown. Calibration/external controls not used to select.

Fixed development criterion: zero observed direction-to-lyric errors while retaining
at least one lyric, then maximize full-support lyric recall across fixed thresholds
0.5/0.6/0.7/0.8/0.9/0.95/0.99. No configuration meets it. At0.7, lambdas0.0001/0.001/
0.01 select115/112/60lyrics with3/2/1 false directions respectively;0.1 selectsnone.
No useful policy/model selected; no further calibration or external rescoring in
this attempt. Zero observed errors is an experimental constraint, not an accepted
numeric release threshold or statistically established precision.

Artifacts: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-grouped-pilot-20261007-v2/`.
report.json, out-of-fold.json, selected-policy.json(null), model-receipt.json and
independent verification.json bind code/input hashes, packages and disjoint folds.
Earlier first attempt kept in adjacent unsuffixed directory: it selected only the
all-abstaining fallback; v2 explicitly rejects that vacuous operating point, without
changing feature/model/grid candidates. No threshold chosen against reviewed pages.
Five focused pilot/selection tests pass, source lint/format pass; code committed locally.

Next C1 priority is more varied annotated non-lyric training evidence (performance
directions, title/credit and real OSSQ), not another threshold sweep on these controls.
The current available pool has45 directions across37scores total;34 are training.
More of its4,771 lyric boxes alone does not fill this non-lyric coverage gap. Partial
annotations and unannotated regions must not be treated as negatives. Preserve24
score exclusions and100-score internal split; keep actual final attachment gates
separate. Automatic lyric fusion remains disabled.

Private-fp16 negative audit still running in existing persistent tmux/watchers;
allfiveoffpages HTTP200, onarm underway; ETA2–3min at this update. Receipt:
`/home/jhlusko/workspace/homr-artifacts/local-tremolo-private-fp16-negatives-20261007/watch-status.json`.
No production/pin/environment/push changes. Inspect finalnativeaudit next, then
qualify full calibrated OCR/text path and model-loaded GPU latency.

