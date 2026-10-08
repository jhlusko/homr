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


## 2026-10-07 — Existing OSSQ direction GT expands C1; full-page recall still fails

Correction to the prior data-gap wording: existing original calibration manifest
already contains800 positive OSSQ DirectionText boxes across9scores. The earlier
training-pool audit explicitly filtered to Lieder; it did not establish that OSSQ
non-lyric ground truth was absent. No new manual labels were requested. Titles and
credits still lack training coverage; partial annotations remain partial positives.

Prepared deterministic bounded sample99 OSSQ regions acrossall9scores, up to12per
score with source-label diversity:35Tempo,36Expression,28StaffText, collapsed to
DirectionText per accepted policy. Original241 Lieder inventory records/split and
24prior exclusions unchanged; formal-test manifest not read. New8training scores
(87regions),1internal-calibration score sq7103818(12regions). Existing annotations
used as positive roles, not newly fully adjudicated; one tempo crop spot-checked.
Review: http://localhost:8765/c1-semantic-gt-packet-20261007/ossq-expansion-20261007/
All198crop/contextassets+index verified overHTTP.

Four one-thread OCR workers completed99newcrops inabout90seconds, exit0. Original
241OCR results retained byte-for-byte, same model/package/parameter identities.
Total340regions:139/144directions nonempty,5empty;163/196lyrics nonempty,33empty.
No exceptions. Missing OCR remainsUnknown/counts in full support. Currenttraining
265regions (144lyrics,121directions), calibration75regions (52lyrics,23directions).
No source roles/text entered OCR or inference; source annotation strings not used.

Training-score grouped sweep20fits, prespecified same model/grid/zero-observed-error
criterion, now selectslambda0.0001/threshold0.99:14/144out-of-fold lyrics retained,
0/121direction false lyric selections. Already-inspected calibration8/52lyrics,
0/23directions falselyselected. All23direction positives include12boxes fromone
OSSQscore; box counts do not establish independent precision. Lieder/OSSQ curves
reported separately; OSSQ has no positive lyrics and must not dominate a pooled
claim. All109development scores disjoint from10external control scores.

Frozen fullpage574OCR-row control inference then retains only1/26reviewedlyrics,
versus26/26 prior e2 selections.0/15 expanded directions and0/6human-confirmed
directions selectedaslyrics; one existingOCRmiss preserved. OtherText credit1
abstained. This is not useful automatic fusion; no model promoted/runtime wired.
No thresholds were changed after external results. These controls/calibration were
previously inspected, not untouched qualification; unmatched lines unadjudicated.
No new final note-attachment accuracy claimed.
Review: http://localhost:8765/c1-semantic-ossq-pilot-20261007/

Artifacts: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-ossq-expanded-20261007/`
contains inventory/receipt/verification, exact launched source, OCRshards+receipt,
grouped-pilot/report.json, source-report.json and frozenmodel/model-receipt.json.
External: adjacent `c1-semantic-ossq-expanded-external-20261007/report.json`.
Source/tests committed locally; sevenfocused expansion/grouped/pilot tests pass,
scopedlint passes. Jobs/watchers finished; only local reviewserver remains.

Next: broaden direction phrase diversity using remaining calibration-only annotated
OSSQ positives, and improve training representation of actual fullpage OCR rows
versus GT-selected crops; preserve score-level separation and positive-only label
matching. Titles/credits remain a gap. More threshold fitting on already-reviewed
controls is not a qualification strategy. Full calibrated text-path/GPU280second
latency/provider image qualification remains separate from C1. No deployment,
modelpin,backend.env,push or automatic lyric attachment change. No running-job ETA.



## 2026-10-07 — Approved semantic retention target; recall-oriented pilot result

Owner accepted the proposed semantic-filter target: retain at least95% of genuine
lyric candidates supplied by e2. Denominator is unique fullpage OCR candidate IDs
selected by e2 and matched to true lyric references; numerator is those retained
aslyrics after the semantic gate. This is conditional retention, not whole-page
lyric detection recall, OCR transcription accuracy or correct note-attachment recall.
OCR/detection misses still count in separate end-to-end metrics. Numeric acceptable
false-direction attachment rate and end-to-end read/attach bar remain unresolved.
Prior zero-observed-direction-error rule was an experimental constraint, not the
owner-approved release requirement. Classifier thresholds .95/.99 are uncalibrated
scores, not95/99% recall or demonstrated precision.

New training-only selection policy uses at least95% nonempty GT-crop lyric retention
as DEVELOPMENT PROXY, minimizing false direction selections among eligible fixed
model/grid candidates. Training GTcrops do not establish actual e2/fullpage gate.
Twenty score-grouped fits from expanded241Lieder+99OSSQ packet chooselambda0.001,
threshold0.5:115/118 usable training lyrics retained(97.46%),115/144 full-support
lyrics(79.86%) including26OCRempties;23/121 directions falselyselected. Already
inspected calibration43/45 usablelyrics(95.56%),43/52 fullsupport(82.69%),6/23
false-direction selections. No calibration/control data used for model selection.
Historical zero-error mode stays available for exact reproduction of prior pilots.

Frozen external fullpage574-row inference now retains25/26 genuine e2 lyric
candidates(96.15%), meeting the OBSERVED approved95%retention target. Only26lyric
candidates from reviewed development pages: not independent qualification or a
confidence-bound guarantee. One true lyric lost: IMSLP12778 OCR `an.`. Expanded
15directionrefs combined false selections2→1 (`Etwas geschwind.` survives; Assez
lent blocked); one preexistingOCRmiss. Original6human-confirmed directions e2false
selections4→1: `Revenez au 1. mouv!` survives; `a tempo.` blocked. The1credit ref
remains unselected by combined e2+semanticgate despite semantic-only misclassification.
Unmatched text remains unadjudicated. Conditional vetoes of old traces are not new
note-attachment results. Fusion staysdisabled; no model or runtime promotion.

Review: http://localhost:8765/c1-semantic-retain95-20261007/ (48cases,90assets verified).
Pilot `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-retain95-20261007/`.
External adjacent `c1-semantic-retain95-external-20261007/`, report.json and
approved-retention-gate.json. Frozen source/input/model hashes and expanded score
exclusion checks preserved.14focused grouped/pilot/external tests pass; lintpasses.

Next: preserve this recall-oriented candidate as the research baseline; use actual
fullpage OCR on training-only annotations and broader existing direction phrases,
then score retention plus false directions bycorpus/score on independent evidence.
Focus actual finalattachment checks on remainingperformance-instruction controls;
no exact-string exceptions or tuning against these already-reviewed failures.
Full calibrated detector/text fusion, model-loaded GPU/280second latency and capable
direction routing remain separate gates. No backend.env,modelpin,push,deployment or
automaticlyrics change. Allcompute done; no running-job ETA.



## 2026-10-07 — Owner accepts review-assisted lyric candidate; local tool prepared

Owner accepted the observed tradeoff for a REVIEW-ASSISTED feature: recall-oriented
semantic candidate retains25/26 genuine e2 lyric candidates(96.15%), while `Etwas
geschwind.` and `Revenez au 1. mouv!` remainfalse lyric suggestions and `an.` is
withheld. This authorizes the review-assisted direction; it does not accept those
errors for unattended MusicXML attachment or authorize production deployment.
95%conditional semantic retention target remains; final verse/note attachment,
calibrated fusion and production latency remain separate gates.

Built local text-suggestion review using ALLactual e2-selected OCR rows on10original
control pages, rather than only48GT-selected regions:310unique candidates,293
proposedlyrics and17withheld. Those are unadjudicated suggestion counts, not
precision/recall. Suggestions contain original scan crops/context, actualOCRtext,
model-based proposed/withheld status and no source reference text/GTrole fields.
All620crop/contextassets,index anddata verifiedoverHTTP. Existing reviews untouched.

Tool: http://localhost:8765/lyric-suggestions-20261007/
Editable text, explicitAccept/Exclude/Reset, explicitbatchaccept of unreviewed
proposals on currentpage, model+pageidentity-scoped browser persistence and JSON
review-record download. Editing previouslyacceptedtext resets it to pending;
empty text cannot beaccepted. Bulk does not overwrite prior edits/exclusions.
Export contains originalOCR,correctedtext,explicitpending/accept/excludestate,
sourcepagehash/box/ID andmodelhash. No importer, MusicXML write, guessed note/verse
attachment or new server-side account state. Textconfirmation alone is not note
placement confirmation. No request for310newmanual labels; existingGTalready
establishes measured errors. This is a functional local preparation for laterOTS
integration, not deployed/scanner-enabled fusion.

Browser smoke passes editing,acceptance,reloadpersistence,emptyrejection,bulk
preserving edits andexport; noJSerrors. Source generator/template committed locally,
scopedlint passes. Model `82da7a641985d4c2baf75e57efe06dbe9bf57f5a49c9186cfd30f50a8887d5d1`.
Output/data/receipt:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/lyric-suggestions-20261007/`.

Next: validate source-free verse/note placement for explicitlyreviewed lyric rows,
then integrate the review record with localOTS behindthefeatureflag. Automatic
attachment remainsdisabled; no productionpin,.env,deployment/push or frontend
integration performed. Existing unrelated frontend edits preserved. Only persistent
local reviewserver isrunning; no compute-job ETA.



## 2026-10-07 — Review decision handoff and explicit verse-number audit

Added local review-record validator/CLI lyric_review_handoff.py. Checks schema,
model/page image identity, exact candidate membership/no duplicates, originalOCR,
box andsemanticproposal evidence, explicitpending/accept/excludestate and nonempty
acceptedtext. Accepting withheldlyrics is allowed via explicitreview. Onlyaccepted
text advances; excluded/pending text doesnot. Editedtext markspositionRefreshRequired.
Allacceptedtext stayspending-verse-note-placement, verse/noteconfirmedfalse,
attachments empty andMusicXMLunchanged. No actual userdecision export imported;
fixturetests are not userapprovals. Ten meaningful stale/forged/duplicate/empty,
withheld-recovery andpending/exclusion tests pass; lintpasses. No runtimewiring.

Audited existing fourpage exact-export-traced sourcefree predictions with fixed
physical-row-order=nativeverse-number DIAGNOSTIC assumption, no oraclemapping:
50/145 literalnote/text/verse matches versus82/145 with bestpage-globalverse
permutation(also82ignoringverse). ChaminadeIMSLP154059 contributes0/42fixedversus
32/42oracle; otherpagesremain5/48,25/29,20/26. This mayreflect nativeedition verse
IDs versus printedroworder; it doesnotprove32visuallywrong placements. NativeGT
used only for evaluation, never toselect reverseverse numbering for inference.
Ninetyfive cachedattachments have source-free exporter/head association evidence;
independentpixelownership and nativeverseidentity stillunqualified. These earlier
pages/predictions are not a freshrun ofacceptedsemanticcandidate or reviewedinserts.

Audit: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-review-verse-audit-20261007-v2/report.json`.
Originalscoringinputs/nativehashes reverified; cachedpredictionhashes bound. One
regressiontest checks explicitverse mismatch andduplicatecapping; lintpasses.
Earlier preformat report/source retained in adjacentunsuffixeddirectory.

Placementreview: http://localhost:8765/lyric-placement-20261007-v2/
95originalscancrops: blue associateddetectednotehead, orange positionedsyllable.
Labels generatedmeasure,pitch,quarter-noteonsetoffset andphysicalprintedroworder;
nativeverseidentity unconfirmed. Source-free cachedoutputs only; noGTreverse mapping,
manualtextapproval,modelpromotion orMusicXMLwrite. All95images+index HTTPverified;
one Chaminadecrop visuallychecked. Generatorandreceipt/inputoutputinventory retained
under `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/lyric-placement-20261007-v2/`.

Next: explicitprintedverse/note-confirmation controls, linking reviewedtext to
verifiedruntimeanchors while keepingeditedsyllablegeometry unresolved untilfresh
positioning. Then localOTS integrationbehindflag. The ownerapprovedreviewassisted
texttradeoff and95%retentionbar, not unattendedverse/note insertion. No frontend,
backend.env,pin,deployment,push orproductiondefaults changed. Allcompute done;
only reviewserverpersists, no runningjobETA.



### 2026-10-07 — Explicit lyric placement review

Added a local 95-case placement confirmation page at http://localhost:8765/lyric-placement-confirmation-20261007-v3/. This is the four-page cached calibration cohort, separate from the 310-candidate semantic review. Text, user-entered verse label, and proposed note require independent confirmation; printed row order never supplies verse identity. Edited text requires fresh positioning. Exports are validated against original page, generated MusicXML, prediction hashes and exact runtime note anchors. The result is a review plan only: no MusicXML writes, automatic attachment remains disabled, and no user decisions have been imported. Six placement validator tests and ten existing text-handoff tests pass. Browser checks cover all 95 cards, separate confirmations, edit invalidation, persistence and an exported fixture validated as one confirmed/94 pending (test fixture only, not user approvals). The receipt binds generator, HTML template and inventory hashes. Next: connect the validated review plan to the local attachment path, then exercise explicit approvals on a copied score before any provider qualification.
