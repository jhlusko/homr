# Predicted previous-page meter experiment — 2026-10-05

Carrying explicit predicted meter from the preceding Sonntag scan page does **not** improve the affected second-page scores. Disabling the exporter's stated-numerator rejection also does not improve them. Keep runtime defaults unchanged; the missing native context cannot safely be replaced by a blanket previous-page fallback.

The preceding original IMSLP89797-p001 was independently checked against native measures0–18 (including pickup), four systems with4/5/5/5 measures, voice plus two piano staves. All three printed staves start in6/4. The source window is contiguous with the previously evaluated p002, native measures19–43. Native material was used to certify boundaries and score outputs, never to initialize inferred meter. This score has known historical/validation exposure; this is a diagnostic, not a new untouched test.

Accepted September18 core/September24 v7 heads and public-joint-core-heads-20260928 epoch27 used the same one-thread CPU runtime/policies/model graphs as the completed page comparison. Both preceding-page decodes succeeded; total90.06 seconds. Three physical staves are exported as two logical parts, with the piano part containing two staves, on both pages. No staff was silently dropped. The persistent tmux jobs/watchers finished with exit0.

The accepted preceding-page cached voices both end with explicit4/4, which disagrees with native6/4. The candidate ends with explicit6/4 for the vocal part and4/4 for the piano part. These are cached predictions **after** parse_staffs processing, not isolated raw decoder predictions; raw recognition versus same-page repair attribution remains to be audited.

The diagnostic only inherits a complete explicit numerator/denominator pair from the same arm, policy and adjacent independently certified score window. It rejects changed runtime/source/model identity or physical staff grouping, and never replaces an existing initial target declaration. Later declarations remain in place; incomplete later declarations invalidate earlier carry state. Accepted vocal output has no initial declaration and receives predicted4/4; its piano part already declares4/4. Both candidate parts already have initial declarations, so no signature is injected. In particular, the candidate vocal part's prior6/4 cannot override its current cached4/4 under this guard.

Baseline XML replays reproduce both original outputs and original metrics exactly. Four variants were tested: baseline, carry only, numerator-rejection disabled only, and carry plus disabled rejection. All four give the following scores:

| Arm | Ordered event edits | Pitch edits | Rhythm edits | Accidental edits | Strict measure/onset/pitch edits | Meter-measure edits |
|---|---:|---:|---:|---:|---:|---:|
| Accepted |8/486|6/439|8/486|6/439|26/486|51/75|
| Joint epoch27 |9/486|7/439|9/486|7/439|27/486|42/75|

Accepted carry changes XML by adding a wrong explicit4/4 where meter was unknown; unchanged error counts do not mean unchanged XML or correctness. All candidate variants reproduce its original XML. There is no evidence here to disable the meter guard or change promotion policy. Preceding-page ordered event edits are8/372 accepted versus7/372 candidate; these additional diagnostics do not change the original page results.

Canonical local evidence:

- Certified preceding-page inputs/review/native previews: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/franz-preceding-page-20261005/`.
- Matched preceding-page recognition/model/runtime receipts: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/franz-preceding-page-pair-20261005/`.
- Final export experiment, predicted meter streams and XML variants: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/franz-meter-carry-20261005-v3/report.json`. Earlier unformatted diagnostic runs are retained, superseded by v3.
- Reproducible tool: `training/onnx/ablate_predicted_page_meter.py`; tests: `tests/test_predicted_page_meter.py`. Eleven focused carry/context/certification tests pass; Ruff/Black pass.

Next: inspect raw per-system predictions and same-page repair provenance for the incorrect initial signatures before proposing any scoped meter correction. Continue identical **final-policy** complete-stack verification separately: this page protocol still disables stem arbitration/slur-side decisions and omits text/OCR/provider latency. A meter-specific experiment is not a substitute for those release gates.
