# Predicted-context tuplet guard, first experiment — 2026-10-04

The first guard protects all23 otherwise-perfect crop/model predictions damaged by the old tuplet heuristic. It fails the successful-repair controls:0/9 remain fully corrected, and only one retains a partial correction. **Do not activate this version.** Aggregate improvements alone are insufficient.

The implementation is diagnostic-only in `training/transformer/guarded_tuplet_cleanup.py`; it is not wired into inference. It uses predicted symbols only, never reference timing. Existing runtime defaults and original scores remain unchanged.

## Frozen first hypothesis

* Rewrite tuplets only when their removal makes the bar exactly fill a known predicted numerator/denominator, or a local bar length supported by at least3 closed interior bars without tuplets and at least two-thirds of eligible samples.
* Estimate within regions separated by predicted meter declarations. An unpaired numerator or denominator does not inherit the previous complete meter.
* Protect first/last crop measures, unterminated bars, and meter declarations appearing after material within a measure.
* Decline streams spanning both staff positions or containing unequal-duration chord members. These guards detect some ambiguity; they cannot recover missing voice ownership or certify all other streams as monophonic.

The thresholds and guards are one diagnostic hypothesis, not learned calibration or independently qualified inference policy. Existing validation controls informed the design. Later tuning on these cases must remain labelled diagnostic.

## Paired OSSQ/Lieder comparison

All4,211 paired crops:3,911 OSSQ/9 scores and300 Lieder/13 scores. Both retained stacks use identical cached inputs. All33,688 raw recognition metrics replay; standard, no-tuplet and no-pitch controls match the prior stage audit in all source/arm/metric totals. No model inference or GPU use.

|Source / stack|Raw symbol errors|Standard cleanup|Guard only|Guard + unison preservation|
|---|---:|---:|---:|---:|
|OSSQ / accepted|4,067|4,185|4,070|4,059|
|OSSQ / joint27|2,580|2,773|2,595|2,580|
|Lieder / accepted|333|380|373|333|
|Lieder / joint27|286|327|326|286|

Denominators remain OSSQ102,219/Lieder11,255. The joint unison-preserving arm is a combined counterfactual, not a separately qualified policy or promotion result. Pitch/rhythm/accidental counts are in the local report. Guard-only OSSQ pitch errors rise287→289 accepted and229→231 joint27 even while symbol errors improve; alignments can change when rhythms change, so each primary metric remains a separate gate.

Relative to standard cleanup, guard-only symbol crop outcomes are OSSQ accepted33 improved/17 worsened/3 tied; joint27 24 improved/2 worsened/1 tied; Lieder accepted2 improved/1 worsened, joint27 1 improved/1 worsened. These are diagnostic crop/model pairs, not independent score-level test samples.

## Why the negative controls fail

Eight of nine successful-repair crops have no predicted meter declaration. Two include unequal-duration chords and are declined as ambiguous clocks. Several lack3 plain interior bars from which to infer a bar length. Some necessary corrections occur at the final crop measure, which this hypothesis protects regardless of its ending divider.

The guard performs only one actual tuplet-removal change relative to disabling tuplets outright: accepted `sq8885571_0006_0002_2.txt`. It fixes8 of16 raw rhythm errors in crop-relative measure6; measure7 remains protected at the boundary. Thus this version is mostly equivalent to disabling tuplet removal. It does not preserve enough of the old heuristic's useful repairs.

The one successful-repair crop with a complete predicted meter, `sq8885571_0015_0004_3.txt`, needs correction in its final crop-relative measure10. The boundary guard declines it. Relaxing that condition without actual page/voice boundary evidence could also damage a legitimate partial bar or pickup; do not select such a relaxation solely because it rescues this known example.

Retained `controls.json` and its reproducible inspection script record all23 positive and9 negative controls, model-predicted meter declarations, staff positions, unequal-duration chords, rational bar lengths, boundaries and errors. They do not establish native source-XML measure/voice truth.

## Complete-voice/page control

Both Grieg opening-page arms replay standard XML and measured scores exactly; all10 arm/policy XML exports succeed. Guard-only leaves event/rhythm46/380, pitch2/372 and accidental2/372 unchanged for both models. Guard plus unison preservation yields event/rhythm44/380 and pitch/accidental0/372, matching the previous unison-only finding. This page supplies a useful complete-voice control but does not qualify broader tuplet behavior. The remaining44 timing errors concern tremolo representation, not this heuristic.

## Recommendation and next steps

Keep this version as a reproducible rejected hypothesis. Do not keep weakening crop safeguards to maximize known validation-case repairs. Recover predicted meter, physical-staff/voice context and actual measure-boundary state from complete voice streams for representative valid-tuplet and false-tuplet cases, then test a bounded revised rule there. Keep unison/noise qualification independent. No new training is needed for these runtime investigations.

This failure does not invalidate the candidate core's raw decoding gains or change the release's primary OSSQ/Lieder gates. Continue complete-stack page qualification and independent text-fusion work; this optional guard is not a new unconditional release blocker.

## Local artifacts and verification

Crop reports, changed streams, controls and source snapshots:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/`

Complete-voice XML variants:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/page-guarded-tuplet-20261004/`

One-thread/core tmux launch, logs, EXIT0 and completion watcher:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004-launch/`

Tools: `training/transformer/audit_guarded_tuplet_cleanup.py` and `training/onnx/ablate_page_guarded_tuplets.py`. Both require new output directories and preserve input/source hash receipts. Eleven new guard tests cover explicit/missing/changed meter, exact fills, short bars, boundaries, staff/chord ambiguity and default restoration. With existing cleanup controls,22 tests pass; Ruff/Black pass. Complete, ETA0; no job running, inference change, promotion, push, upload or deployment.

## Visual review

### 2026-10-04 — nine-case tuplet review available locally

Review: http://localhost:8765/ (persistent `tuplet-review-server` tmux session, loopback only). All nine successful-repair negative controls show original hash-verified OSSQ crops beside reference labels, raw accepted-stack predictions, current cleanup and rejected guard. Crop-relative measure labels describe triplet-to-ordinary-note changes. Case8 clearly shows16 raw rhythm errors,0 after current cleanup,8 remaining under the guard; case9 is the sole explicit predicted2/2 example.

“Eight have no predicted meter” means no time-signature declaration was emitted in these standalone crop decodes; it does not establish absent printed or inherited full-page meter. Reconstructed display meters are explicitly identified as XML-export context. MuseScore4.6.5 outputs were audited for visible notes AND rests, timing, onset and staff;20/36 panels have changes (including inserted rests), flagged on the page. Original scans and stored labels remain authoritative. This gallery requires no individual voting and is not a new promotion gate.

Local site, all images/XML and provenance: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/guarded-tuplet-20261004/review/`. Generator: `homr/training/transformer/review_guarded_tuplets.py`; guarded renderer reuse verifies inputs, binary and exact reconstructed XML. Nine cards/81 unique linked assets return HTTP200. Rendering/watcher complete, ETA0; the serving tmux remains active. No inference/default change, promotion, push, upload or deployment.
