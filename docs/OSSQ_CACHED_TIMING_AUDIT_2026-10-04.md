# OSSQ timing and unison audit — 2026-10-04

The shared page errors have three different causes. Late XML repair ablations do not remove them. An instrumented repeat proves that both models recognize Grieg's duplicate D4 noteheads and the runtime then deletes them. Original paired evaluation results remain unchanged; all counterfactual results below are separate diagnostics.

## Evidence and controls

Eight cached XML variants per arm on two pages reproduce all four original baseline XML outputs exactly, with identical scores. Disabling tuplet, beam, tie, slur-crossing, all late repairs or stated-meter rejection leaves pitch/rhythm scores unchanged. The caches are after `parse_staffs` cleanup, not raw decoder outputs.

Canonical-source label round-trips succeed for all8/8 parts. Beethoven has zero event/pitch/rhythm errors across all455 visible events. Grieg's Violin2 round-trip has44 duration errors; the other three parts have zero, including the duplicate D4 events. This tests the current parser, not the historical parser version used to build either trained checkpoint.

The Grieg page was repeated once per model with a diagnostic wrapper around `staff_parsing.remove_duplicated_symbols`. Both repeats reproduce the original XML and cache SHA256 exactly. Incoming/outgoing symbols, original identities and source hashes are archived. Recognition behavior was not changed by instrumentation.

## Whole-bar rests: elapsed duration and glyph identity differ

Both models predict18 `rest_1` tokens on Beethoven's page. These decode as ordinary four-quarter whole rests; the source marks full-measure rests with three quarters in3/4. The current source parser instead produces `rest_2.` (three-quarter dotted half), preserving elapsed timing but losing the full-measure-rest glyph/semantic marker on export. Thus the zero-error source timing round-trip does **not** establish engraving fidelity.

The existing diagnostic sole-whole-rest hypothesis uses only the predicted meter and sole-note structure, not reference timing. It changes18 durations from4 to3 quarters:

| Beethoven variant | Accepted pitch / rhythm errors | Epoch27 pitch / rhythm errors |
|---|---:|---:|
| Original, unchanged | 3 / 21 | 1 / 18 |
| Sole-whole-rest hypothesis | 3 / 3 | 1 / 0 |

This is positive evidence for a meter-aware export alternative, not a generally safe rule. A sole whole rest can also be an incomplete bar. Existing negative fixtures reject missing meters, dots, polyphony, backups and already-correct durations, but do not establish pickup/incomplete-bar safety. Keep the hypothesis outside default inference until broader positive/negative real-page checks. Explicit full-rest availability remains preferable to pretending every ordinary whole-rest token is authoritative.

## Paired tremolos: source timing lost before recognition

Grieg Violin2 measures18–28 contains44 written half notes, with source duration480/divisions480 (one quarter), 2:1 time modification and alternating one-stroke tremolo start/stop. Current label parsing produces `note_2` plus the generic `tremolo` articulation. Both model streams match those legacy target fields. Label round-trip and model export both emit two-quarter elapsed durations; generic articulation export additionally changes the source's one stroke to three.

The parser's tuplet state consumes time modification only when a visible numbered tuplet is open. These paired tremolos have no such tuplet marker. Simply treating every ratio as an ordinary visible tuplet would conflate different notation and is not a qualified fix. The flat tremolo field also loses single/start/stop and stroke count, so it cannot safely infer alternating-pair duration from every generic tremolo token.

These44 errors are a label/export representation limit in the tested current path, not evidence of failed learning of its half-note target. More training on the same legacy targets cannot supply the missing timing/type/stroke distinctions. Preserve written value, elapsed duration, ratio and detailed tremolo ownership independently in the optional improvement lane. No new head training is an unconditional release requirement.

## Unisons: cleanup deletes correct predictions

Grieg Violin1 measures9 and10 each contains two simultaneous D4 whole notes in different source voices. Current source labels preserve both with a chord separator. Both actual incoming predicted streams also contain both. `_remove_duplicated_piches` groups by pitch spelling and physical staff position, ignoring source/model voice identity, and retains one. This causally accounts for the two missing-note errors; they are not recognition failures in these runs.

Disabling only pitch deduplication on the observed pre-cleanup streams, while retaining other cleanup and late repairs, reproduces the original control exactly and gives:

| Grieg variant, both models | Pitch | Rhythm / event | Accidental |
|---|---:|---:|---:|
| Original, unchanged | 2/372 | 46/380 | 2/372 |
| Pitch-dedup ablation | 0/372 | 44/380 | 0/372 |

Only measures9 and10 change in the six legacy fields. Other cleanup metadata changes remain recorded. No extra pitched events appear elsewhere on this page. Preserving two notes as a chord does not itself restore distinct XML voice ownership. This one-page success does not establish that globally retaining duplicate model predictions is safe; broader repeated-note/noise cases remain required.

## Local reproduction

All results: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-cached-timing-audit-20261004/`.

- Per-score `cached/report.json` and `report.md`: exact replay controls and eight variants per model.
- Per-score `labels/report.json`, `part-*/{reference.musicxml,labels.json,roundtrip.musicxml}`: all source-label round-trips, including failures if any.
- Grieg `cleanup-trace/{accepted,candidate}/`: observed pre/post cleanup streams, XML/cache, identities and exact-match receipts.
- Grieg `cleanup-ablation-final/report.json`: final source-hash-guarded ablation; first exploratory ablation remains in sibling `cleanup-ablation/`.
- `cleanup-symbol-differences.json`, `executed-source/sha256.json`, logs, `run.sh`, `trace-run.sh`, `EXIT` and `TRACE_EXIT`: source snapshots and execution evidence. The executed trace script snapshot predates a later keyword-only API/lint cleanup; its hash matches both actual trace receipts.

Tools: `audit_page_label_roundtrip.py`, `ablate_cached_page_xml.py`, `trace_page_duplicate_cleanup.py`, `ablate_cached_duplicate_cleanup.py`. The two existing tools now accept canonical plain MusicXML as well as MXL; completed label reports cannot be overwritten.17 focused tests pass, including preservation of unison multiplicity without changing default cleanup, continued redundant-header cleanup and trace identity/overwrite guards. New-tool Ruff/Black checks pass. All jobs finished with EXIT0, single-thread CPU; no remaining ETA.

## Next bounded work

1. Run cached pitch-dedup counterfactuals on the existing paired OSSQ/Lieder real-crop outputs to quantify legitimate unison recovery versus erroneous duplicate retention. If those caches are before this cleanup, no new recognition is needed. Include error changes and supported-case denominators; do not infer XML voice identity from event multiplicity.
2. Broaden the sole-whole-rest hypothesis checks to positive and incomplete-bar/pickup cases on independently reviewed real-page windows. Keep original scores and the alternative side by side. Decide whether a narrow optional runtime policy can pass without training new timing heads.
3. Continue real-page coverage/final-policy qualification and text-fusion calibration independently. The candidate's crop gains and modest OSSQ page gain remain, but Lorelei rhythm regression and these common pipeline limitations still need honest release reporting. No model promotion, provider pin change, upload, push, deployment or training occurred.
