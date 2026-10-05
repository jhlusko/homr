# Matched real-page notation policy diagnostic — 2026-10-05

The joint epoch27 model has substantially fewer stem errors on this four-page diagnostic. The reviewed notation policies improve accepted-v7 OSSQ stems and slur sides, but do not establish a universal improvement for joint heads or Lieder. Recognition/timing metrics remain exactly unchanged. Keep promotion and policy selection separate.

Eight one-thread CPU decodes succeeded on the four previously certified original pages: Haydn sq9146376, Glazunov sq7358579, Chaminade IMSLP154086, Franz IMSLP89797. Native boundaries, image/source hashes, model identities and exposure receipts are unchanged from the earlier paired diagnostic. Historical/validation exposure remains; these are not new untouched tests.

OSSQ uses `HOMR_STEM_ARBITRATION=ossq-v7` (0.8 cutoff) and `HOMR_SLUR_SIDE=ossq-rule`; Lieder uses legacy stem cutoff0.9 and slur-side off. Beam, tie, crossing, tuplet and cross-staff settings remain identical between arms and to the baseline. Structured head confidence is retained in `pre-policy-with-confidence.json`, rather than discarded by the old cached serializer. The OSSQ0.8 cutoff was calibrated for accepted-v7 heads: applying it to joint heads is explicitly provisional, not a calibrated final joint policy. Unknown-corpus inference cannot assume the OSSQ route from instrumentation.

Total runtime586.86s (9m47s). All jobs/watchers completed exit0. Ordered event/pitch/rhythm/accidental metrics match all original arms exactly. Strict measure/onset/pitch metrics, XML meter streams and physical measure counts also match exactly. No performance ceiling is inferred from this small diagnostic; page event edits are not the crop-token Lieder +0.25pp promotion gate.

The notation scorer fixes physical staff, native-relative measure, onset, written pitch/alter, duration and grace status. Only unique anchors shared by source and **all four** variants are scored; ambiguous unisons and missing/mistimed/wrong-pitch notes are not paired by their notation. Source stems must explicitly be up/down. Slur scoring requires one explicitly placed native start; absent/multiple predicted starts and unknown placement are reported separately from wrong side. This is conditional accuracy, not end-to-end notation recall, full slur-span pairing, or exhaustive printed-scan engraving adjudication.

| Page | Stem support | Accepted raw→policy errors | Joint raw→policy errors | Slur-start support | Accepted raw→policy errors | Joint raw→policy errors |
|---|---:|---:|---:|---:|---:|---:|
| Haydn |244|4→3|1→1|43|5→1|0→1|
| Glazunov |299|14→9|5→5|45|9→6|12→12|
| Chaminade |245|28→19|0→0|0|not scored|not scored|
| Franz |407|100→111|6→6|1|1→1|0→0|

OSSQ stem totals: accepted18→12/543, joint6→6/543. Lieder: accepted128→130/652, joint6→6/652. The legacy stem arbiter helps Chaminade but worsens accepted Franz; this does not authorize deploying the OSSQ0.8 policy to Lieder.

OSSQ slur-start totals: accepted14→7/88, joint12→13/88. The Glazunov joint12 comprise11 starts with no explicit side and one absent/multiple-start mismatch; they are **not12 wrong-side predictions**. Accepted Glazunov after policy has5 unknown-side starts and one start-count mismatch. Across OSSQ, known wrong-side errors fall8→1 for accepted, but rise0→1 for joint. Haydn's joint raw placement was already correct on this support; the opposite-stem rule introduces one error. There is no evidence here to apply the v7 slur rule indiscriminately to the joint model.

The exporter only borrows structured placement for a legacy/core endpoint when an event-compatible structured slot exists (`slur_placement`). The side rule acts on structured events; it does not invent compatible slots for preserved core starts. This code explains a possible route to unspecified XML placement, but a per-case event/slot audit is needed before attributing all11 Glazunov starts to it. Do not silently fill absent sides or weaken span pairing to make this metric improve.

Next: audit those unknown-side starts against saved core/structured event slots and final export, while keeping note/timing controls. Retain the joint candidate's raw slur placement as a comparison control and calibrate any joint policy separately. Finish complete-span and final provider identity/GPU/HTTP/OCR/attachment qualification before promotion; this local four-page result does not qualify the entire production path.

Canonical local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/page-notation-policy-20261005/`. Per page: `comparison.json`, `timing-context-audit.json`, `notation-audit-v2.json`, both XMLs, policies, confidence caches and model/runtime receipts. `executed-source.py` and `executed-notation-scorer.py` preserve executed code; a later harmless scope-string formatting change in the worker does not rewrite receipts. Earlier two-page `notation-audit.json` is superseded by v2's explicit error breakdown.

Tools: `training/onnx/qualify_page_notation_policy.py` and `audit_page_notation_policy.py`. Three new scorer tests protect fixed alignment, duplicate-unison refusal and absent-source-notation handling. Runtime/defaults, provider pins, original scores and user edits remain unchanged.
