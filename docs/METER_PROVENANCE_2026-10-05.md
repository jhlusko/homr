# Sonntag meter provenance — 2026-10-05

The wrong meter declarations originate in raw model decoding, rather than being introduced by cross-staff repair or voice cleanup. Four instrumented page/model repeats reproduce all original decoded symbols and MusicXML exactly. The preceding carryover experiment's negative result therefore does not justify disabling exporter safeguards.

The accepted September18/v7 stack and public joint20260928 epoch27 were repeated on independently reviewed IMSLP89797 original p001/p002. Source meter was never passed into recognition. Instrumentation records raw and filtered greedy declarations, decoded per-system state before/after cross-staff repair, and voice declarations before/after cleanup. Every arm passed exact XML and cached-symbol equality against its saved baseline; meter values/order were unchanged by those repair stages. Token indices shift when unrelated symbols are removed and are not treated as meter corrections.

| Page/model | Raw meter observations |
|---|---|
| p001 accepted | Initial vocal and piano4/4, despite printed6/4; piano repeats4/4 on systems3/4 |
| p001 joint epoch27 | Initial vocal and piano6/4 correct; piano incorrectly adds4/4 at each subsequent system opening |
| p002 accepted | Vocal initial meter absent; predicts4/4 at the real middle change. Piano begins with4/4 and ends with3/4 where native returns to6/4 |
| p002 joint epoch27 | Vocal and piano initially predict4/4 where native inherits6/4. Vocal and piano correctly predict the final6/4 return |

Staff-call mapping is voice-major, then scan-system order, as recorded in the SystemPlan slots and repair snapshots. The piano is a two-staff logical part; two decoded logical voices represent all three physical staves. No matching was chosen from reference note accuracy. p001 has four systems and p002 five; no reference meter was injected. Historical/validation exposure remains recorded, so these are diagnostic witnesses, not untouched release tests.

All4/4 statements at continuation system openings are model declarations, even when the printed page has no new signature there. Carrying a previous-page meter while refusing to overwrite existing declarations cannot repair those. Blindly overriding them would risk genuine changes such as p002's native4/4 change and return to6/4. The default exporter may still affect XML meter representation, but the separate carry/no-rejection ablation found no score benefit; provenance and outcome are distinct claims.

Next investigate a meter-specific hypothesis with controls containing both correct printed signatures and inherited context, genuine changes, pickup bars and mixed-meter music. Before any runtime change, preserve pitch/rhythm/onset metrics and verify signature visibility or another independent evidence source; a whole-page duration mode is insufficient. This does not require new training to document, nor establish that a new timing head is a release prerequisite. Continue matched final-policy stack qualification separately.

Canonical local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/franz-meter-provenance-20261005-v2/`, containing four `meter-trace.json` and exact `trace-receipt.json`, model/runtime receipts, XML and exact executed source. Initial launcher attempt failed before decoding because setup tried to set PyTorch inter-op threads twice; v2 performs setup once and is canonical. No environment was rebuilt. Total217.05s, one CPU thread, tmux and watcher exit0, ETA0.

Reproducible tracer: `training/onnx/trace_page_meter.py`. Runtime defaults and existing qualification files are untouched. Eleven focused carry/context/certification tests pass; new diagnostic tools pass Ruff/Black. See [the meter carryover report](PREDICTED_PAGE_METER_2026-10-05.md) for the original scores.
