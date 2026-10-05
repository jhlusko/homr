# Grace serialization audit — 2026-10-04

Follow-up: [Ordinary onset and annotation audit](ORDINARY_ONSET_CROP_AUDIT_2026-10-04.md) qualifies all 97 onset groups and records paired export limitations plus the successful annotation-anchor prototype. The migration gate remains.

The 167 changed indexed Lieder crops originate in the zero-duration grace-cursor correction (`7ef93bc`), rather than the exact-timing metadata. A controlled ablation restores the old grace cursor under otherwise current sequence-repair code, before arpeggio scoping and reproduces all 167 published token files exactly. This isolates the cause; it is not a full replay of the historical environment.

## Regression and repair

Correcting the cursor exposed a merger regression: successive grace notes share an onset and were collapsed into one chord, whose pitch sorting could change their source order. The parser now records a grace sequence ordinal separately from elapsed position. An explicit MusicXML `<chord>` inherits the preceding grace ordinal; successive grace notes receive separate ordinals. The merger uses these as a secondary ordering key. Grace notes still consume zero elapsed time and do not displace following notes or their advance targets. Existing kern callers retain their default ordering.

A second regression affected five crops: arpeggio propagation used elapsed position alone, spreading a principal chord’s marking into preceding grace groups. Propagation now includes the actual group ordering key and staff, so it stays within the written chord. A focused test checks both separation of successive grace groups and propagation inside an explicit grace chord.

The full replay reuses all 212 indexed scores / 3,867 certified crop boundaries and original images, on one CPU thread. After both repairs it takes 386.4 seconds, with persistent tmux and a completion watcher, EXIT=0. Published references, tokens and image hashes remain unchanged.

| Split | Byte-identical after repair | Still changed |
|---|---:|---:|
| Training | 3,473 | 94 |
| Validation | 297 | 3 |
| Total | 3,770 | 97 |

Seventy previously changed crops regain exact parity. No previously identical crop develops a new difference. All 3,867 preserve the multiset of note/rest rhythm, pitch, accidental and staff-position identities. Replayed notation and timing sidecars pass the existing token-hash/order guards. The 14 native-stretch elapsed/ratio masks remain, with no unseen values in the frozen mixed-source target classes.

## Remaining migration gate

The remaining 97 comprise 45 group-boundary-only changes, 37 note-order changes with the same identity inventory, and 15 changes to other fields with the same note order. Examples join ordinary upper/lower simultaneities that the old artificial grace duration separated; chord serialization then sorts members and hoists slur fields. One inspected case also changes numerator retention through the existing modal-duration check.

The ablation explains their cause, but does not independently qualify ordinary cross-staff onsets, slur/export semantics or numerator decisions. Keep these crops separate from byte-identical metadata migration. Do not revert the zero-duration cursor simply to force token parity, silently replace frozen labels, or claim a trained-model improvement. A versioned corpus change needs source-onset and export checks for this subset. Field comparisons identify 21 crops with changed slur attribution, one with changed articulation attribution (accent/tenuto hoisting, not residual arpeggio contamination), and three with changed control inventories. These counts overlap the categories above.

## Local evidence and tools

Artifact parent: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `lieder-grace-and-arpeggio-qualification-20261004/`: final complete replay, per-score records, regenerated audit labels/sidecars, source snapshots, logs and completion receipts.
- `lieder-grace-order-qualification-20261004/`: intermediate sequence-only repair, `replay-comparison.json`, and retained source-audit failure evidence.
- `lieder-grace-cursor-ablation-20261004/`: the 167 frozen crops across 71 scores, filtered immutable inputs, ablation replay and EXIT=0; finishes in 22.4 seconds.

Reproducible tools: `training/onnx/audit_grace_cursor_ablation.py` wraps the existing crop replay with the former cursor behavior; `training/onnx/audit_source_grace_order.py` checks visible pitched grace sequences and explicit grace-chord membership against source XML through token grouping. Diagnostic attributes stay in memory; no source/corpus writes occur.

An intermediate source-order check also produced three false positives when different voices were compared in document order across distinct onsets. The corrected audit computes source onsets independently from divisions, note durations, backups and forwards, and compares grace sequences within each source onset. A voice-backup regression test guards this distinction. Ordinary onset grouping remains outside this grace-only qualification.

The first source-order attempt stopped on unsupported legacy `note_24.` during vocabulary sorting. Its failure receipt/log are retained. The corrected diagnostic uses the same chord partition as the serializer without looking up unrelated unsupported duration aliases; within-chord pitch sorting cannot change group membership or sequence. This avoids hiding a source-order defect, and does not make that unsupported token serializable.

Final source-order qualification covers 85 OSSQ / 212 Lieder references: 6,728 / 1,156 visible pitched grace notes, 6,713 / 1,117 grace groups and 12 / 34 explicit grace chords, with zero mismatches. It finishes in 301.3 seconds. All 47 focused tests and three subtests pass, including sequence/chord order, arpeggio isolation, advance targets, voice backups, unsupported unrelated duration aliases and existing sidecar/target/checkpoint guards.

All five final receipts are zero: crop replay, source-order audit, tests, replay comparison and class-support inventory. Source snapshots match the tested code. Every original image hash, crop boundary and split matches the earlier frozen replay; immutable input hashes are identical. The byte-identical training subset now supplies 129,216 timed crop occurrences (validation 10,719); changed crops remain separately inventoried (5,861 / 197). All qualification jobs are finished, remaining ETA zero. The original scan evidence viewer remains idle at http://localhost:8765/.

## Next work

Independently qualify the 97 remaining Lieder changes, including ordinary onset groups and associated slur/numerator behavior. Recover certified OSSQ crop-to-source alignment before claiming actual OSSQ training support. Keep the native staff-stretch duration policy explicit and retain the 14 masks. Then wire opt-in, resumable frozen-core exact-timing head training against qualified labels. No GPU is needed for these next audits and wiring.
