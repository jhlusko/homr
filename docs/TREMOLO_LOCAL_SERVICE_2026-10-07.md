# Local paired-tremolo service candidate — 2026-10-07

`local_tremolo_candidate.py` prepares a separate runtime directory from the pinned
b6eca7 archive plus runtime glue and the additive paired-tremolo candidate patch.
It verifies archive/patch hashes, copies the accepted experimental encoder,
decoder and v7 heads under a distinct model name, copies the local CPU segmentation
graph, and records graph/runtime/service hashes in `candidate.json`. It refuses
an existing output directory. Shipping sources, graph caches and pins are untouched.

The local adapter uses the actual supervised HomrEngine workers, separately for
plain core and heads, through the dedicated provider's authenticated HTTP scan-page
contract. It reports local CPU/experimental provenance rather than private shipping
identities. It rejects text-detector and title-OCR requests; there is no fusion or
production GPU latency claim. Both flag states are separate worker processes.
The source snapshot differs from the newer working-tree pipeline, so previous
corpus accuracy gains cannot be assumed to transfer without auditing these outputs.

26 focused service tests also pass against this isolated source candidate; scoped
lint passes. All four image-first HTTP scans finish successfully, returning both
actual inference candidates for each page and flag state. All four candidate
comparisons preserve decoded tokens, pitch/written-duration inventories and measure
counts. Native reference windows were opened only after outputs were frozen.

| Page | Rhythm errors off → on (both candidates) | HTTP seconds off → on |
|---|---:|---:|
| Lieder IMSLP634835 | 30 → 30 / 357 | 59.63 → 53.08 |
| OSSQ sq9631717 | 46 → 2 / 380 | 98.09 → 102.82 |

Pitch and accidental errors each remain3/333 for Lieder and2/372 for OSSQ; no
native-measure regressions. OSSQ transfers the known gain to this isolated source
candidate. Lieder does not: both XML files remain identical, whereas the newer
working-tree pipeline achieved30→10. Known positive pages, not independent corpus
accuracy. Timings include both worker candidates and HTTP handling but exclude
OCR/text fusion and use local CPU; scheduling/warm-up differs between arms, so
these times do not identify precise rule overhead or production timeout margin.

## Local files and persistence

Candidate and results:
`/home/jhlusko/workspace/homr-artifacts/local-tremolo-service-20261007/`.
Manifest: `candidate.json`; arms: `off/`, `on/`; logs: `off.log`, `on.log`;
completion markers: each arm's `complete.json` and final `all-complete`;
watcher: `watch-status.json`; final comparison: `comparison.json` when generated.

The job, watcher and automatic comparison ran in persistent tmux sessions
`tremolo-service-smoke`, `tremolo-service-watcher`, `tremolo-service-comparison`.
All finished; watcher reports success. No remaining ETA or running job.
A first startup attempt was stopped to correct the mandatory idempotency header,
and is preserved separately. The disabled launcher was already running when small
warm-up/provenance improvements were made for the enabled arm; runtime/graph files
and the tested policy remain unchanged. These are single-pass integration timings.

The six-field Lieder regeneration-token replay finds independent peer span9/2 and
20 candidate notes across bars11/13, so missing peer context alone does not explain
the unchanged output. That replay is supplementary, not a trace of the original
symbol objects. Investigate the conservative final timeline/protected-output guards
against the older exporter before porting more code; do not weaken those guards.
Diagnostic: `lieder-context-diagnostic.json` in the candidate directory.

Next: investigate Lieder's source-version-dependent abstention, preserve disabled
parity, and qualify any narrowly scoped source fix. Full OCR/text-fusion latency
and a production-compatible GPU candidate still require separate qualification.
C1 semantic lyric routing and E3 final examples remain open. No push/deploy/.env
change or new model training occurred.
