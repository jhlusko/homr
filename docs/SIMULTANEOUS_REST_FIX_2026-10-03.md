# Preserve simultaneous rests — 2026-10-03

The MusicXML exporter now preserves every positive-duration rest in a simultaneous group, using explicit backups to retain their shared onset. Voice rebalancing assigns overlapping rests separate voices; rests are not emitted as chord notes. Printed durations and the group's rhythmic advance remain unchanged. Previously, only the first rest survived.

## Qualification

All 297 indexed canonical OSSQ/Lieder references were compared with exact archived controls and advance off/on. Coverage remains 75/85 OSSQ and 212/212 Lieder. The ten OSSQ detachedLegato failures reproduce unchanged. Both old arms exactly replay the preceding ending-marker qualification's XML hashes, metrics and timing.

| Source | Rest events restored | Scores improved | Event edits before → after | Rhythm edits before → after | Correct measure inventories recovered |
|---|---:|---:|---:|---:|---:|
| OSSQ | 3 | 3 | 8,351 → 8,348 | 6,192 → 6,189 | 3 |
| Lieder | 117 | 29 | 13,578 → 13,461 | 9,829 → 9,712 | 108 |

Counts apply independently to both advance settings; they are not doubled across arms. Pitch and accidental edits are unchanged. No new failures, lost correct inventories, worse common-measure onset counts, per-score edit regressions or zero-duration non-grace notes occur. The previously located IMSLP89026 duplicate-rest case is restored: event edits 4 → 3, with both co-onset quarter rests retained.

Lieder's advance-enabled total moved-event count grows 2,169 → 2,173 because recovered inventories admit additional measures into the timing diagnostic. On the identical common measures the count stays 2,169 → 2,169. This is an eligibility change, not evidence of a timing regression. One of 109 changed Lieder inventories remains imperfect; it still improves rather than newly losing a correct inventory.

These are canonical reference-label/XML roundtrip diagnostics, not recognition error rates. Runtime was 1,070.6 seconds (17.8 minutes), one CPU thread, persistent tmux and completion watchers. Both reference and cached jobs finished EXIT=0; remaining ETA is zero.

## Cached model check and tests

Hash-verified accepted (September 18 core plus September 24 v7 heads) and candidate (public joint core/heads September 28, epoch 27) predictions on the previously evaluated Lorelei page reproduce the preceding cached controls exactly. Rest preservation changes neither model's XML event inventory, pitch/rhythm/accidental edits nor timing, with advance off or on. This page contains no affected predicted duplicate-rest groups, so it provides a replay check, not a broad model generalization gate.

Seventy-seven focused tests pass in the final two test groups (48 plus 29), including new independent XML-timeline checks for duplicate rests, separate voices, simultaneous pitched notes and an advance shorter than the printed duration. Three additional subtests pass. Existing parser, grace, ending-marker, tuplet, beam, octave, slur-placement and timing diagnostic coverage remains green.

## Local evidence and next steps

Local parent: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `simultaneous-rest-qualification-20261003/`: complete report, all 297 outcomes, original generator, source/hash receipts, selected XML, launch/log/EXIT and watcher records.
- `simultaneous-rest-preflight-20261003/`: located IMSLP89026 controls and corrected XML.
- `cached-page-rests-20261003/`: eight model XML exports, hash-verified cached comparison, source receipts and completion records.

Tools: `training/onnx/qualify_simultaneous_rest_fix.py` and `training/onnx/ablate_cached_page_rests.py`.

The narrow simultaneous-rest correction qualifies against the preceding corrected-label controls. It resolves the duplicate-rest blocker, but the three Lieder duplet-duration inventory regressions remain. Next, audit exact future-build tuplet/ratio and whole-measure-rest representation against canonical OSSQ/Lieder XML, with versioned label changes and no blanket rhythm-token rewrite. No GPU is required for that audit. Broader model-predicted advance remains unqualified and disabled. Published corpora, model/provider pins and deployment are unchanged.

