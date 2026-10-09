# Review-assisted lyrics: release evidence and draft claims

Updated 2026-10-08. This is a reviewable release checklist and copy draft, not a deployment approval or assertion that the current hosted service implements the local candidate.

## Draft product copy

> Review lyric suggestions beside the original scan, correct the text, and choose which notes receive each syllable before adding them to your score. The suggestions can miss lyrics or confuse printed instructions with sung text, so your review matters. Lyrics are added only after you accept them.

Publish only after the final pinned service and local acceptance/rerender flow are verified. Direction text belongs in a separate review flow; the other agent owns that implementation. Do not imply that a DirectionText classification guarantees a printed instruction: confirmed lyrics receive that label too.

## Measured results suitable for a dated evaluation note

- Fresh scanner-prepared replay: 10 Lieder vocal pages and 15 instrumental OSSQ pages; all 25 completed recognition, both detectors, OCR and corrected backend filtering. The baseline retained 157 vocal suggestions and no instrumental suggestions. These are suggestion counts, not lyric recall or exact note-placement scores.
- An offline lyric-line context pilot restored four scan-confirmed lyric rows, raising vocal suggestions to 161 with zero retained instrumental suggestions. All 157 original retained rows and anchors were unchanged. No context rule is shipped from this evidence.
- On eight additional pages with previously inspected partial labels, selected OCR rows covered by lyric references improved from 19/20 retained to 20/20 under the one-peer pilot. Both covered direction rows remained excluded. These pages previously evaluated the semantic classifier; they are new to context-policy development, not an untouched end-to-end test. Thirteen other direction references were outside that selected-row denominator.
- A separate two-page/six-owner-confirmed-direction replay is running. Do not publish a result before its completion and inspection.

Do not advertise 95% end-to-end recall, 100% lyric recognition, automatic lyric attachment, exact syllable ownership, or a qualified final GPU/provider stack on these results. The 398 native targets have approved passage correspondence; ambiguous or unassigned pixel/candidate links remain unscored.

## Package and source audit

Read-only local audit passed: model artifacts match their pins, candidate graph copies and runtime sources match their recorded hashes, calibration detector identities match pins, and both completed fusion replays have no binding drift. The source archive also matches its declared SHA. The identity/calibration/fusion suites passed 53 tests.

The completed recognition qualification used the private CPU FP16 plain core with notation heads disabled. It is not a final structured-head or GPU HTTP qualification.

The calibration manifest currently binds model weights and its policy body, but does not bind the code that extracts raw confidence. Component confidence extraction has been corrected. A five-control audit found 12 changed e2 Lyrics scores and zero calibration-band transitions; this does not certify calibration on the corrected source. Before promoting that fix, bind confidence-extraction source identity and requalify the calibration evidence. Do not silently claim that unchanged weight pins establish unchanged score semantics.

## Remaining release work

1. Finish and inspect the owner-confirmed direction controls; keep the context rule offline unless broader labelled evidence supports its use.
2. Bind the confidence-extraction implementation to the calibration/release identity and requalify score semantics. Any change must preserve compatibility with existing pinned manifests deliberately, rather than silently altering their meaning.
3. Complete lyric acceptance/PDF rerender and OCR-first direction review integration with the other agent; test accepted text in exported XML and rendered output.
4. Match genuine lyric candidates to references to score the conditional 95% retention gate; measure OCR text and note ownership separately. Review-assisted acceptance does not require inventing automatic certainty for ambiguous rows.
5. Run final pinned-stack GPU HTTP and per-page timeout checks, including active notation heads and the actual scanner preparation path.
6. Update public examples and copy from that final evidence; keep current historical metrics tied to their specific detector, input preparation and evaluation scope. Publishing/deployment remains a separate action.

## Local evidence

- [Package/source audit](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-release-package-audit-20261008/report.json)
- [Prepared-page baseline](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-prepared-full25-20261008/filtered/report.json)
- [Context development assessment](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-line-context-single-peer-20261008/ASSESSMENT.md)
- [Additional-page assessment](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-context-additional8-20261008/ASSESSMENT.md)
- [Calibration-band stability](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/detector-component-calibration-drift-20261008/report.json)
- [Owner-confirmed direction watcher](/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/lyric-context-direction6-20261008/watch-status.json)
