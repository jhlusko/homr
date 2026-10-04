# Timing fixes: OSSQ/Lieder qualification, 2026-10-03

Follow-up: `GRACE_ENDING_FIX_2026-10-03.md` qualifies standalone ending-marker separation and restores nineteen OSSQ grace notes without regressions versus the corrected-label controls. The earlier results below remain intact. Duplicate-rest and duplet-representation blockers remain unresolved.

The five-arm reference qualification completes with exact archived control replay. Grace cursor correction substantially improves Lieder timing and ordered event edits, but **the fixes do not yet pass a clean inventory-preservation gate**. Keep predicted advance disabled: the cached accepted/candidate page diagnostic finds no eligible timing gain and one accepted-stack onset regression.

The local corrections are committed as `7ef93bc`: grace notes preserve the parser cursor; XML division selection preserves every printed note/rest duration as well as cursor advances. Meter inference remains based on consumed group time. Seventy focused tests pass. Published corpora, checkpoint identities, model/provider pins and production were not changed.

## Complete reference results

All 297 indexed references were attempted on one CPU thread. Every arm succeeds on 75/85 OSSQ and 212/212 Lieder; the ten OSSQ detachedLegato failures replay unchanged. The archived advance-off/on controls match their earlier XML hashes and metrics exactly. Runtime: 1,332.8 seconds (22.2 minutes), persistent tmux and watcher, EXIT=0.

| Reference roundtrip arm | OSSQ moved onsets | Lieder moved onsets | OSSQ event edits | Lieder event edits |
|---|---:|---:|---:|---:|
| Archived control, advance off | 470 | 15,855 | 8,364 | 14,057 |
| Archived advance on | 412 | 5,252 | 8,330 | 14,051 |
| Precision correction only, advance on | 412 | 5,252 | 8,364 | 14,057 |
| Both corrections, advance off | 412 | 12,341 | 8,370 | 13,578 |
| Both corrections, advance on | 354 | 2,169 | 8,370 | 13,578 |

The table's onset counts are conditional on each arm preserving canonical event inventory within the physical staff/measure. Eligibility changes slightly on Lieder, so do not subtract those columns as a strictly paired comparison. On measures eligible in **both** old and corrected advance arms, OSSQ is 412 → 354 (14.1% reduction; 120,218 measures / 525,443 events), Lieder 5,248 → 2,169 (58.7%; 39,914 measures / 253,655 events). There are six / 460 improved exact-onset-count measures and zero worse ones. Lieder order-changed measures on this common subset fall 144 → 0. This is canonical reference-label/XML timing, not model recognition accuracy or voice-aware equivalence.

Both corrected advance-off/on arms have the same eligibility: OSSQ 412 → 354; Lieder 12,341 → 2,169, with 22 / 1,857 improved exact-count measures and zero worse ones. The previous 39 Lieder count regressions do not recur in this corrected-label comparison. Inventory and duration issues below remain explicit.

## Located limitations

**Precision exposes reciprocal-label rounding.** Three previously correct Lieder staff measures become duration-incorrect: IMSLP598858 staff 1 and 3, physical measure index 21; IMSLP622488 staff 3, index 3. All six affected duplet notes require three-quarters of a quarter note but are encoded as `note_5`, or four-fifths quarter. The old advance-selected grid accidentally truncated them back to the source duration. Preserving the printed label's exact fraction removes that accidental compensation. This does not justify retaining truncation or globally rewriting legitimate quintuplets. Future-build ratio/duration representation remains required.

In OSSQ Nielsen sq12536479, precision-only ordered event edits increase 62 → 96, across 12 changed event-sequence measures. Located printed durations change three-quarters → four-fifths quarter; those measures already fail canonical inventory preservation in both arms. This accounts for 34 of the net 40 additional OSSQ event edits. All located changes and complete metrics are retained, including changes on already lossy measures.

**Grace realignment exposes structural-marker grouping.** In Beethoven sq9396439, grace correction increases event edits 32 → 40 and removes eight previously emitted grace notes. Four groups in parts indexed 1/2, source-measure indices 435/436, begin with `voltaStart` and contain grace notes. With all grace notes correctly at zero elapsed time, they collide with the ending marker; the exporter handles the marker and ignores the group's notes. The earlier one-tick error separated some of those notes. This is a real remaining preservation bug. Next correction should keep ending markers out of note chords and explicitly test grace notes at ending boundaries. An inventory-preserved-only timing metric would miss these losses because the affected measures were already lossy.

OSSQ sq8913219 improves event edits 47 → 45 from event reordering after grace correction. Lieder IMSLP89026 improves 7 → 4 overall but loses one previously correct inventory measure (staff 3, physical index 69): two co-onset quarter rests are reduced to one. `build_note_chord` intentionally emits only the first of equal-duration rests; correcting their onset exposes that existing voice-collapse policy. Both pitches and timing improve there, but exact inventory still regresses. Preserve this result rather than waiving it as musical equivalence; a voice-aware decision is still needed.

Thus versus the old advance arm there are **zero OSSQ / four Lieder newly lost correct inventories**, plus real grace-note loss in already lossy OSSQ measures. Overall Lieder event edits improve by 473, while OSSQ worsens by 40. Neither conditional onset improvement nor net aggregate improvement constitutes a clean pass.

## Predicted advance: cached original-page diagnostic

Accepted stack: September 18 core + September 24 v7 heads. Candidate: public-initialized joint core/heads, September 28 epoch 27. Replayed the existing Lorelei page predictions, verifying cached hashes and exact original XML/metric controls. Late repairs remain constant; no recognition rerun. The page was previously bar-count evaluated, so it is not an untouched test.

| Stack | Eligible measures / events | Moved events, advance off → on | Whole-page event edits |
|---|---:|---:|---:|
| Accepted | 27 / 297 | 0 → 1 | 14 → 14 |
| Candidate epoch 27 | 25 / 261 | 0 → 0 | 17 → 17 |

No new correct-inventory loss occurs on this page. Advance changes nine / ten groups' consumed durations, with 15 / 17 fallback groups. Pitch, rhythm and accidental edit metrics remain unchanged, and zero non-grace zero-duration notes are emitted. This is no evidence to enable predicted advance. Changes in measures whose recognition inventory is already wrong are outside the conditional onset metric; this small diagnostic does not establish all-source head quality.

## Next actions and local artifacts

First fix preservation of grace notes in groups containing ending markers, test the located Beethoven cases, and repeat the real-source qualification. Resolve duplicate-rest/voice preservation explicitly. Retain the exact-duration allocator; correct future-build reciprocal/tuplet representation rather than depending on truncation. Do not rebuild published corpora or train new checkpoints until those label changes are versioned and qualified. Then broaden predicted-advance tests across OSSQ/Lieder pages/crops with explicit exposure/coverage records. Text-fusion/C1 and new untouched page references remain separate release work.

All jobs complete. Local parent directory:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`

- `real-timing-fixes-20261003-v2/`: five-arm report, all 297 outcomes, source snapshots/hashes, run/EXIT/watcher receipts. Initial input-format failure is retained in sibling `real-timing-fixes-20261003/`.
- `cached-page-advance-20261003/`: per-stack three-variant XML, cached/source hashes, report, sequential launch script and completion receipt.
- `timing-fix-event-inspection-20261003-v2/`: exact replay of three selected OSSQ scores and IMSLP89026, 12 XML exports, located inventory/event changes and the grace/ending-marker collision probe.

Tooling: `qualify_reference_timing_fixes.py`, `ablate_cached_page_advance.py`, `inspect_timing_fix_event_changes.py`, `probe_grace_metadata_collision.py` under `training/onnx/`. Source receipts preserve the tested versions and the old functions used to replay archived controls. Use the existing HOMR environment and audit dependency overlay with OMP/OpenBLAS/MKL threads set to one. No GPU is needed for these follow-ups.
