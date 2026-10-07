# Guarded paired-tremolo runtime export — 2026-10-07

The qualified paired-tremolo timing rule is now available in the HOMR exporter
behind a **default-off flag**. It reproduces the frozen research candidates exactly
on all 17 unique cached pages, while the disabled path reproduces every accepted
control exactly. No new timing model, weights or model pin is required.

## Additional controls

Four pages from distinct scores outside the context-development cohort were selected
by source semantics before new inference. None contains source paired tremolos.
Use original scans and certified native measure windows; keep historical model
exposure in receipts, so this is a scope check rather than an untouched model test.

| Control | Score / page | Rhythm errors / source events, both arms | Timing edits |
|---|---|---:|---:|
| Changing 2/4 → 3/4 meter | IMSLP89026 p002 | 22 / 290 | 0 |
| Single-note tremolo with grace notes | IMSLP617260 p006 | 15 / 559 | 0 |
| Grace notes plus ordinary triplets | IMSLP154289 p001 | 0 / 143 | 0 |
| Ordinary chords | IMSLP558609 p006 | 19 / 385 | 0 |

All four new pages and seven repeated cached controls completed (11/11 jobs).
All made zero edits; no native measure regressed. New-control totals remain rhythm
56/1,377, pitch24/1,301, accidental12/1,301 and strict-onset249/1,377. These existing
recognition errors are visible, not counted as repairs. Source categories describe
what is present on the page; they do not assert every such feature was decoded
correctly. Separate runtime guard tests exercise explicit decoded meter changes,
partial declarations, grace notes and mixed-duration chords.

## Runtime entry point and enablement

`homr.music_xml_generator.generate_xml` selects the rule only when
`HOMR_PAIRED_TREMOLO_TIMING=1`, or when the caller explicitly supplies:

```python
XmlGeneratorArguments(paired_tremolo_timing=True)
```

The unset environment and omitted/None argument preserve the accepted path.
Explicit `paired_tremolo_timing=False` overrides an enabled environment. This flag
is separate from `HOMR_STRUCTURED_EXPORT`; it does not enable experimental advance
or timing heads. Do not change production defaults to run a comparison.

`homr/tremolo_timing.py` contains only runtime symbol/context/export logic, with
no training imports, reference reads, trusted-pickle loaders or model loads.
Per-render state traces original symbol identities and restores per-note tuplet
marks; no global monkeypatch is used. Work on copied input symbols, preserve the
caller's objects and support concurrent flag-on and flag-off exports. Complete
paired lanes, candidate-excluded independent context, constant complete meter
repeats, exact fit, chord/voice/interval guards and protected notation remain.

After regeneration, require unchanged emitted inventory, unchanged untargeted
durations, precisely halved targeted consumed durations, unchanged written
notation and meter, and clean final voice timelines. Preserve the accepted export
if a protected invariant fails. A regression test specifically verifies abstention
when shortening a wholly overfilled part would also rewrite its inferred meter.

## Parity and validation

Final runtime parity: **17/17 pages**, both flag states exactly match their frozen
research XML. This includes four positive edited scores, all seven cached controls,
two further unchanged positive/negative pages and the four new control pages.
Input-cache hashes and frozen XML hashes are checked before comparison. Caller
objects remain unchanged during enabled export. Positive edits/metrics are exactly
those in [TREMOLO_PEER_CONTEXT_2026-10-07.md](TREMOLO_PEER_CONTEXT_2026-10-07.md):
93 note symbols,20 measures,four selected scores; no native-measure regressions.

One local export-only timing pass adds median0.244s and maximum1.066s across17 pages.
This excludes inference, OCR and detector execution; it is not a production latency
benchmark or a percentile estimate. The default-off path performs no context work.

Checks:71 focused export/policy/audit tests pass, full Ruff checks pass on owned
files, and scoped mypy passes (`--follow-imports=silent --disable-error-code unused-ignore`
for the pre-existing generator ignore). No source/runtime changes occurred after
final parity apart from research-script lint formatting; snapshots are retained.

Local artifacts:
- Additional controls: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-additional-controls-20261007/`.
- Final runtime parity: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-runtime-parity-20261007-v2/report.json`.
- Final parity directory also contains the exact runtime and qualifier source
  snapshots matching receipt hashes, per-case flag-off/on XML and timings.
- Earlier parity iteration: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-runtime-parity-20261007/`.

## Remaining release steps

1. Run a fresh full-page pipeline smoke comparison with the flag on, on at least
   one positive Lieder page and the positive OSSQ page, using the same accepted pins.
   Cached final-export parity establishes the rule's output; fresh smoke covers its
   integration with current pipeline execution and measures end-to-end overhead.
2. Include the runtime revision/flag policy in the local release candidate and
   decide whether to enable it; no production change, push or deployment has occurred.
3. Keep timing-head promotion, ordinary-duplet qualification and full-rest metadata
   separate. C1 lyric-versus-direction semantic routing remains unresolved and still
   blocks automatic lyric fusion. Continue E3 examples only against a settled stack.

All control, watcher and parity jobs have finished. The flag remains off by default.
