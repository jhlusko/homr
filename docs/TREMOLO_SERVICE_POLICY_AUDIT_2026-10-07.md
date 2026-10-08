# Paired-tremolo service policy audit — 2026-10-07

The unchanged Lieder export in the first isolated HTTP candidate was a configuration
mismatch, not evidence that its pinned source needs a weaker timing safety rule.
No tremolo rule or MusicXML safety checks were relaxed.

## Actual export-input trace

A diagnostic supervised worker captures the original symbol objects immediately
before export. Its resulting XML SHA-256 exactly matches the original enabled
Lieder core output, establishing that the diagnostic reproduced the observed case.
The captured symbols carry structured notation that the six-field review-token
replay omitted. The actual trace finds the same peer span9/2 and20 candidate notes,
but four proposed staff/measure edits fail the conservative timeline check:
multiple voice identifiers, and overflow in page measure11.

The local candidate left `HOMR_STRUCTURED_EXPORT` unset. That old runtime interprets
unset as emitting all implemented structured fields, including rhythmic advance.
The accepted benchmark instead specifies `beam,stem,tie,slur`, excluding advance.
Those predicted advances alter consumed onsets independently of written durations.
Replaying the same full captured objects with this explicit accepted family mask
makes all four timeline checks pass and the paired-tremolo XML change. This isolates
the cause without changing the rule or reference data.

Reproducible trace: `training/onnx/trace_tremolo_export.py --input <trusted-local-pickle>
--runtime-root <isolated-runtime> --output <receipt>`; repeat with explicit
`HOMR_STRUCTURED_EXPORT=beam,stem,tie,slur` for the controlled policy comparison.
The script is audit-only, accepts an agent-created local pickle, imports the chosen
runtime and never opens source references. Capture driver, original input, exact
parity receipt and both decision traces are retained under:
`/home/jhlusko/workspace/homr-artifacts/local-tremolo-service-20261007/lieder-trace/`.

## Separate head-routing bug

The old pinned runtime never reads `HOMR_STRUCTURED_HEADS_PATH`, even though the
service worker sets it. Both first-candidate workers therefore loaded the installed
default sidecar, so its labels core/additional-heads did not establish distinct
head configurations. Their original outputs remain useful paired timing-policy
checks, but must not be reported as a core-versus-head ablation.

An isolated source patch makes FilePaths honor an explicit sidecar path (including
empty to disable), preserving the original default when unset. Empty optional-model
paths skip downloads. Patch SHA-256:
`365346a3b81988c342264f68d618f3a64364deff0f0661d571a5bd3ba0aead3a`.
`HomrEngine` additionally accepts an explicit structured-family mask; its default
None preserves existing shipping behavior. Only the isolated candidate sets masks:
core=none, additional-heads=beam,stem,tie,slur. The shipping Modal builder does not
apply the source patch or adopt these candidate settings.

## Corrected candidate qualification

Fresh isolated directory:
`/home/jhlusko/workspace/homr-artifacts/local-tremolo-service-20261007-v2/`.
Same accepted experimental graph hashes, segmentation and source base, plus the
small head-routing patch and explicit masks. No training, replacement weights,
production configuration, deploy or push. 27 focused service tests and five runtime guard tests pass, with clean scoped lint.
Explicit empty/custom/default path and no-empty-path-download checks pass. All four
fresh HTTP scans finish successfully. Observed core structured-label counts are0;
heads counts385 for Lieder and514 for OSSQ. Both flag arms preserve decoded tokens,
pitch/written-duration inventory and measure counts. Native source audits show no
regressed measure for either inference candidate.

| Page | Rhythm errors off → on, core and heads | HTTP seconds off → on |
|---|---:|---:|
| Lieder IMSLP634835 | 30 → 10 / 357 | 50.72 → 47.17 |
| OSSQ sq9631717 | 46 → 2 / 380 | 94.23 → 106.95 |

Pitch and accidental errors each remain3/333 for Lieder and2/372 for OSSQ. These
are known positive pages, not independent corpus accuracy. Times include two real
worker candidates and HTTP handling but exclude OCR/text fusion and use CPU, not
production GPU latency. Single-pass scheduling differences prevent precise timing
rule overhead conclusions. All jobs/watchers/comparison tasks finished.

Receipts: `comparison.json`, `native-score-report.json`, per-arm HTTP responses and
XML, `candidate.json`, `routing-check.json` and `watch-status.json` in the v2 directory.
A supplementary negative-cache replay could not deserialize newer `VoiceClass`
metadata with the older pinned runtime; no new negative replay claim is made.
Failure scope is recorded in `negative-replay/status.json`. Fresh controls or an
explicitly qualified field bridge are needed, rather than patching in fake classes.
The five runtime guard tests remain valid; they are not eleven fresh page controls.

## Remaining release steps

1. Run fresh negative-control OSSQ/Lieder pages with the corrected local configuration
   and preserve a disabled control. Do not reuse newer object caches silently.
2. Carry explicit accepted family masks and distinct sidecar routing into a concrete
   provider runtime-source candidate. The shipping builder/settings remain unchanged;
   source inspection and the local reproduction indicate its head-path setting is
   ignored, but no live production behavior was measured here.
3. Qualify full OCR/text-fusion latency and production-compatible graph/source pins.
   The local graphs are accepted experimental fp32 exports, not the private pinned
   fp16 shipping graphs; their results are not interchangeable without qualification.
4. Resolve C1 lyric/direction semantic routing and regenerate E3 against final policies.

No tremolo rule/safety checks were changed. No training, new model weights, deployment,
push or backend environment change occurred. Implementation committed locally in OTS;
trace driver and this audit are retained in HOMR. Ordinary timing-head promotion
remains separate; rhythmic advance is explicitly excluded from this candidate.
