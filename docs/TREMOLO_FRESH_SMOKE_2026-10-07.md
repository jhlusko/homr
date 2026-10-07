# Fresh paired-tremolo pipeline smoke — 2026-10-07

Fresh image-first runs confirm the default-off runtime implementation integrates
with the accepted recognition pipeline. Both flag states reproduce their frozen
XML exactly on the positive Lieder and OSSQ pages. All four runs finish successfully,
and native ground-truth audits report no regressed measure.

| Source / page | Flag off → on rhythm errors | Pipeline seconds, off → on | Export seconds, off → on |
|---|---:|---:|---:|
| Lieder IMSLP634835 p004 | 30 → 10 / 357 | 61.52 → 62.22 | 0.053 → 1.686 |
| OSSQ sq9631717 | 46 → 2 / 380 | 124.87 → 126.19 | 0.016 → 1.348 |

Pitch/accidental errors remain 3/333 for each Lieder metric and 2/372 for each OSSQ
metric. These are two known positive pages, not an independent accuracy estimate.
Their flag-on and flag-off exports are identical to the separately frozen research
candidates/controls, confirming integration rather than adding new improvement claims.

## Measurement scope

Use the original certified scan images, accepted core dated2026-09-18 and v7 heads
dated2026-09-24, with the same graph SHA-256 pins and corpus policies as prior
qualification. Execute separate fresh processes for each flag state; run four
workers concurrently, each with one Torch/OpenCV/ORT thread. The flag is recorded
in per-arm policy identities and reaches the actual `main.process_image` export call.

Pipeline time starts at `process_image`; export time is measured at its actual
`generate_xml` call. All XML and inputs are saved before source references enter
scoring. Input image and graph hashes are checked; final frozen XML hashes are
checked before parity comparison. Code hashes, graph pins and CPU identity remain
in each arm's `identity.json`/`result.json`.

Title OCR and text detectors are disabled, matching the accepted comparators.
These local CPU timings therefore exclude text fusion and do not establish the
production scanner's 280-second timeout budget. Parallel contention, cold graph
setup and scheduling also affect the differences between pipeline times; do not
interpret those differences as a precise export overhead measurement. The directly
measured export timings and earlier17-page export-only measurement show modest
local overhead, but neither is a provider latency benchmark.

## Local candidate recommendation

The runtime rule is ready for **local opt-in release-candidate use**. Keep the
shipping default off; use `HOMR_PAIRED_TREMOLO_TIMING=1` only in the candidate's
explicit configuration. No new head model or model pin is needed. The tested
runtime implementation is commit `138af38`; all input/model/runtime source hashes
and source-specific flag policies are recorded in:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-fresh-smoke-20261007/local-candidate-policy.json`.

The current local worktree has other uncommitted work; per-arm identities retain
its full pipeline hashes. This receipt is not a claim that the entire tested
pipeline came from commit138af38. Corpus-specific OSSQ stem/slur settings must not
be inferred from a score's instrumentation for unknown inputs.

## Next release steps

1. Carry the explicit runtime flag/revision into the local scanner release-candidate
   configuration and verify the service/provider path. Preserve a disabled control.
   No production environment, provider revision, model pin or deployment changed here.
2. Resolve C1 lyric-versus-direction semantic routing before enabling automatic
   lyric fusion. Its qualification remains separate from notation timing.
3. Regenerate E3 side-by-side examples against the settled candidate policies;
   ordinary-duplet examples and timing-head promotion still need their own evidence.

Reproducible driver: `training/onnx/smoke_tremolo_pages.py`. Run with frozen pages,
accepted graph manifest, comparator directory and a new output directory. Separate
workers generate original-image exports; the parent freezes completion then scores
against the native source windows. The driver and runtime snapshots used for this
run are retained. Post-run script changes are lint-only string wrapping.

Local results: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tremolo-fresh-smoke-20261007/report.json`.
Per-score native audits and all four MusicXML exports are in the same directory.
All jobs finished; no remaining ETA applies. Script Ruff checks pass; runtime/export
qualification's71 tests remain valid because runtime code did not change.
