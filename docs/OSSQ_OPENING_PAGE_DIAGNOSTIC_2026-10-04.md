# OSSQ opening-page certification and paired diagnostic — 2026-10-04

Two original scan pages now have reviewed edition/part identities and complete physical-measure boundaries. This resolves the input gap for a **small diagnostic**, not the corpus-wide absence of certified scanned-system metadata. Both scores were evaluated previously (RUNLOG27.81); neither is an untouched test.

| Score | Edition visible in scan | Reference physical measures | Four system counts |
|---|---|---:|---|
| sq8071278 — Beethoven Op.18 No.1 | Breitkopf & Härtel, plate B.37; source cites IMSLP04755 | 1–36 | 8,10,9,9 |
| sq9631717 — Grieg Op.27 | Edition Peters, plate7363; source cites IMSLP40157 | 1–28 | 7,9,6,6 |

Selection was frozen before recognition: SHA256 of `ossq-full-page-20261004:` plus score ID, from17 eligible published valid/scanned-test scores with full-score type1, nonempty CSV and local original pages, prioritizing no known exposure in the previously audited indexes. Selected first two identities and their first original pages, without model quality selection. Published scanned-test membership does not erase historical evaluation or incomplete historical training provenance.

The agent visually compared the original scans with MuseScore4.6.5 renderings of canonical-source windows, checking all four systems and fixed Violin1/Violin2/Viola/Violoncello order. All four source parts agree on printed page/system boundaries and physical measure identities. CSV and unaligned metadata remain hints/cross-checks only; synthetic coordinates were not substituted for scan evidence. The canonical MuseScore3.6.2 MusicXML is untouched;4.6.5 previews are not regenerated ground truth. This is boundary/edition review, not exhaustive annotation fidelity certification.

Local certification evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-page-certification-20261004/`: `eligible-candidates.json`, `selection.json`, `visual-review.json`, `certified/pages.json`, `certified/receipt.json`, and per-score MuseScore previews. Original pixels/source/PDF/render hashes are recorded. A fixed10MP pilot pixel budget is declared before inference (Beethoven9.25MP, Grieg6.83MP), independently of the earlier8MP reference inventory. No image resizing or relaxed source boundaries are used to fit the pilot.

## Matched runtime protocol

Accepted September18 core/September24 v7 heads versus public-joint-core-heads-20260928 epoch27, exact previously qualified float32 exports, corrected decoder continuation context, identical policies from `qualify_joint_pages.py`. Predicted advance off; no reference sidecars injected. Stem arbitration/slur-side rules remain off in this diagnostic, matching the earlier Lorelei comparison; it does not qualify final D1/provider policies. Text detectors/title OCR do not run. One CPU thread, sequential arms/pages, pinned to one CPU, local persistent tmux plus completion watcher. No GPU rental.

Outputs include graph/page/code hashes, frozen source snapshots (including pre-existing dirty runtime work), per-arm decoded symbols/XML/receipts and logs. Successful completed arms can resume; changed input/code identities are rejected. Failed arms remain explicit failures and carry no zero-error metrics. Missing result receipts stop the supervisor for investigation. This tooling commit alone does not reproduce dirty runtime policies from a clean checkout.

Scoring is per-physical-staff ordered MusicXML event Levenshtein, with pitch, fractional quarter-note duration/rhythm and accidental projections. Extra/missing staves incur edits. This is not crop token SER; it does not implement the Lieder+0.25pp gate and does not measure onset/barline/notation/lyrics accuracy or final production latency. Per-page outcomes and failure denominators must be read before drawing model conclusions.

Paired results/status: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-opening-page-pair-20261004/`: `progress.json`, `completed-arms.json`, `summary.json` when finished, per-score `comparison.json`, `EXIT`, and `completion-report.md` from the watcher. Sequential four-decode estimate20–40 minutes initially; revise after first completed arm. The watcher records completion/failure and updates dated status blocks in the release docs.

Ten focused reference/certification/scoring/resume/edit-audit tests pass; new-file Ruff/Black pass. Next read the paired OSSQ diagnostic beside Lorelei, audit concrete divergent cases, then expand independently certified OSSQ/Lieder page inputs before a complete-stack promotion gate. Text-fusion calibration remains independent. No model promotion, upload, push or deployment.

## Completed results

All4/4 arms succeeded,2/2 fully paired pages, EXIT0. Total sequential wall time405.9 seconds (6m46s); watcher completed2026-10-05 03:31UTC and updated release-doc status blocks. No running job remains, ETA zero.

| Page / metric | Accepted | Epoch27 |
|---|---:|---:|
| Beethoven pitch | 0.783% (3/383) | 0.261% (1/383) |
| Beethoven rhythm | 4.615% (21/455) | 3.956% (18/455) |
| Beethoven combined event | 4.615% (21/455) | 4.176% (19/455) |
| Beethoven accidental | 0.261% (1/383) | 0.261% (1/383) |
| Grieg pitch | 0.538% (2/372) | 0.538% (2/372) |
| Grieg rhythm / event | 12.105% (46/380) | 12.105% (46/380) |
| Grieg accidental | 0.538% (2/372) | 0.538% (2/372) |

**Beethoven:**18 rhythm substitutions are shared whole-bar rests exported as4 quarters rather than source3. Accepted additionally has one deletion/two insertions; epoch27 removes those coverage errors but introduces one B-natural/B-flat substitution (cello, measure32). Equal accidental totals therefore conceal different error types: accepted insertion versus candidate substitution. Do not infer improved accidental-head accuracy from the pitch reduction. All original scores remain unchanged.

**Grieg:**44 shared duration substitutions occur on Violin2 measures18–28, four notes per bar. Every source note is written half, has duration480 with divisions480, 2:1 time modification and alternating one-stroke tremolo start/stop; model XML uses two quarters rather than source one. This is the written/elapsed/tremolo representation boundary, not evidence that epoch27 learned a worse rhythm head. Both also collapse two source voices' simultaneous D4 whole notes into one note in each of measures9 and10. The ordered edit alignment assigns two deletions to measure10; that alignment location is not unique. Raw per-measure XML confirms the actual voice collapse across both bars. Neither discrepancy is waived or normalized out of the published diagnostic.

Per-score `edit-audit.json` separates substitutions/insertions/deletions and records quarter-note beat locations; Grieg `source-duration-audit.json` preserves all44 raw source notes and source/comparison hashes. `environment.json` records CPU Torch2.14.0, ONNXRuntime1.30.0 and other local package versions. Snapshot receipts bind the tested runtime; later edit-audit tooling is separately hashed.

Read beside the earlier Lieder Lorelei page: epoch27 improves pitch7→5 errors, but worsens rhythm14→17. The two OSSQ pages favor epoch27 modestly; these three diagnostic pages do not establish broad generalization or settle promotion.

**Next:** use the cached predictions for bounded full-bar-rest, paired-tremolo and same-pitch multi-voice export/label checks; distinguish recognized written glyphs from elapsed timing and avoid reference injection in inference. Extend paired page coverage with independently reviewed mappings and truthful training/selection provenance, then test final D1/text/provider policies. Keep the crop Lieder+0.25pp gate separate. Text-fusion calibration continues independently. No new training or unconditional notation redesign is required to release a stack that passes its declared gates.
