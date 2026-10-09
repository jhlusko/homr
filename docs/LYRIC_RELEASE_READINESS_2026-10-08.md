# Review-assisted lyrics: release evidence and draft claims

Updated 2026-10-08. This is a reviewable release checklist and copy draft, not a deployment approval or assertion that the current hosted service implements the local candidate.

## Draft product copy

> Review lyric suggestions beside the original scan, correct the text, and choose which notes receive each syllable before adding them to your score. The suggestions can miss lyrics or confuse printed instructions with sung text, so your review matters. Lyrics are added only after you accept them.

Publish only after the final pinned service and local acceptance/rerender flow are verified. Direction text belongs in a separate review flow; the other agent owns that implementation. Do not imply that a DirectionText classification guarantees a printed instruction: confirmed lyrics receive that label too.

## Measured results suitable for a dated evaluation note

- Fresh scanner-prepared replay: 10 Lieder vocal pages and 15 instrumental OSSQ pages; all 25 completed recognition, both detectors, OCR and corrected backend filtering. The baseline retained 157 vocal suggestions and no instrumental suggestions. These are suggestion counts, not lyric recall or exact note-placement scores.
- An offline lyric-line context pilot restored four scan-confirmed lyric rows, raising vocal suggestions to 161 with zero retained instrumental suggestions. All 157 original retained rows and anchors were unchanged. No context rule is shipped from this evidence.
- On eight additional pages with previously inspected partial labels, selected OCR rows covered by lyric references improved from 19/20 retained to 20/20 under the one-peer pilot. Both covered direction rows remained excluded. These pages previously evaluated the semantic classifier; they are new to context-policy development, not an untouched end-to-end test. Thirteen other direction references were outside that selected-row denominator.
- The two-page/six-owner-confirmed-direction replay completed. Only two reference-covered selected OCR rows entered the scored denominator: one existing false lyric suggestion (Revenez au 1. mouv!) remains under baseline and both variants; tremolando. stays excluded. The context rule restores the genuine lyric syllable ro without new labelled direction leakage. This does not test exclusion of all six reference directions.

Do not advertise 95% end-to-end recall, 100% lyric recognition, automatic lyric attachment, exact syllable ownership, or a qualified final GPU/provider stack on these results. The 398 native targets have approved passage correspondence; ambiguous or unassigned pixel/candidate links remain unscored.

## Package and source audit

Read-only local audit passed: model artifacts match their pins, candidate graph copies and runtime sources match their recorded hashes, calibration detector identities match pins, and both completed fusion replays have no binding drift. The source archive also matches its declared SHA. The identity/calibration/fusion suites passed 53 tests.

The completed recognition qualification used the private CPU FP16 plain core with notation heads disabled. It is not a final structured-head or GPU HTTP qualification.

The calibration manifest currently binds model weights and its policy body, but does not bind the code that extracts raw confidence. Component confidence extraction has been corrected. A five-control audit found 12 changed e2 Lyrics scores and zero calibration-band transitions; this does not certify calibration on the corrected source. Before promoting that fix, bind confidence-extraction source identity and requalify the calibration evidence. Do not silently claim that unchanged weight pins establish unchanged score semantics.

## Remaining release work (2026-10-09)

1. DONE locally: combined lyric/direction/unplaced-mark review through save, XML, PDF and same-browser reopen/resave; prior edits survive. See the dated verification evidence below.
2. Diagnose part alignment failures and recognition-core crashes; keep uncertain placement explicitly reviewer-chosen.
3. Freeze and audit the actual candidate, including dynamics weights/direction code. Bind/requalify confidence-extraction source semantics before shipping that fix.
4. Final active-heads pinned-stack GPU HTTP latency/timeouts using actual scanner-prepared inputs.
5. Refresh public examples and measured claims, then seek deployment approval.

Direction review, dynamics retraining, unplaced rows and the reviewed-XML PDF renderer now exist locally/test-provider-side. Further classifier training and the offline lyric-context pilot are optional follow-ups. The 95% conditional lyric-retention measurement is distinct from direction availability and remains unclosed; publish no claim that it passed. Shipping provider changes remain unperformed.

## Local evidence

- [Package/source audit](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-release-package-audit-20261008/report.json)
- [Prepared-page baseline](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-prepared-full25-20261008/filtered/report.json)
- [Context development assessment](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-line-context-single-peer-20261008/ASSESSMENT.md)
- [Additional-page assessment](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-context-additional8-20261008/ASSESSMENT.md)
- [Calibration-band stability](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/detector-component-calibration-drift-20261008/report.json)
- [Owner-confirmed direction watcher](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-context-direction6-20261008/watch-status.json)

### 2026-10-09 — priority 1 combined review flow verified

PASS on a disposable copy of IMSLP44645-p017. Browser edits accepted a lyric (`verifica`), placed direction (`dolce`) and an unplaced dynamic (`mf`, requiring explicit note selection). Save generated reviewed MusicXML and a regenerated PDF containing the edited lyric/direction. Reopening in the same browser preserved decisions; resaving `molto dolce` preserved the lyric/dynamic, produced exactly two review directions (no duplicates), and regenerated the PDF again. Parsed XML with review additions removed was unchanged; original user-job XML SHA was unchanged. Headless Chromium has no embedded PDF viewer; PDF downloads/content were verified separately. This is one mixed-content end-to-end smoke test, not corpus recall qualification or cross-device decision persistence.

Fixed three defects discovered during verification: explicit engine XML downloads ignored reviewed copies; explicit engine PDF downloads lacked review-input signature validation; an older reassembly could overwrite a newer reviewed XML locator. Reassembly now guards reviewed checksum snapshots before publishing pages, releases a rejected snapshot's lease, and review saves renew a pending rebuild even during rendering. Service/worker regression suites: 124 passing; earlier copy/candidate/worker suites: 92 passing; frontend lyric/direction suites: 27 passing. Backend build passed. Test renders used the production worker method on a reserved local clone and the existing Docker MuseScore wrapper; no paid inference, production changes or pin/env changes.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/review-flow-verification-20261009/report.json`, with both MusicXML/PDF revisions, browser script and screenshots beside it. The disposable Mongo job is removed directly after verification to protect shared original inputs from scanner cleanup. Next priority: alignment/recognition failures, then actual-candidate freeze/calibration binding, final pinned GPU HTTP/latency, public examples and deployment approval.
