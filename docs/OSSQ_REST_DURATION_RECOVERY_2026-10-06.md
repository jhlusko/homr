# Canonical OSSQ rest-duration recovery — 2026-10-06

Recovered 295 omitted full-measure-rest durations in 102 crops from original
source notes. The unchanged Borodin training subset expands from 238 to **529
crops**, with **12,522 timed events**, **399 triplets** and **1,139 rests**
(817 ordinary, 322 full-measure). Every previous qualified crop is retained.
All public token and notation bytes remain unchanged. Restored timing can be
recorded beside a legacy whole-rest token without changing that token's text.

Final report, qualified local bundles and image/label training index:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-rest-duration-recovery-20261006-v2/`.
Control/log/exit receipt:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-rest-duration-recovery-20261006-control/`.
The unsuffixed packet stopped on a rest-only crop without a sidecar; only `-v2`
is complete. Persistent tmux job `ossq-rest-timing` completed with exit 0.

The opt-in `--restore-missing-rests` mode requires the existing unique,
accidental-aware source window shared by all four parts. Each paired visible
note retains its musical signature. An omitted duration is restored only for
an explicit full-measure rest with canonical timing. Explicit segment durations
still require semantic timing equality; their tick grid is not guessed.
When a segment lacks divisions but contains only durationless whole-measure
rests, the canonical rest's grid supplies both divisions and duration ticks.
The old writer emitted no timing sidecar for rest-only crops with no durations;
these sidecars are constructed from unchanged rest-token identities and filled
only through the verified source map. Every restoration retains physical
window, source note location and complete canonical timing in the receipt.

An ambiguous transfer (including a collision between different missing-rest
durations or between a rest and a grace note) is refused. Grace notes remain
without elapsed-duration supervision. The report's legacy `missing_timing=96`
field comprises exactly 96 `note_16G` grace tokens; there are no unlabelled timed
events in the accepted subset. Zero unseen classes or duration conflicts.
All 529 bundles reloaded; token/notation hashes, old-subset inclusion and tool
hash verified. Seventeen focused tests pass.

| OSSQ head | Eligible targets |
| --- | ---: |
| Elapsed | 12,522 |
| Written | 12,200 |
| Ratio | 12,200 |
| Full-measure rest | 1,139 |

Combining **only this replacement OSSQ packet** with the existing 3,409 unchanged
Lieder training crops gives 140,642 timed-event occurrences in 3,938 crops:
140,628 elapsed, 139,186 written, 139,172 ratio and 11,278 rest targets.
Class coverage remains 23/51, 15/21, 6/22 and 2/2 respectively. Previous OSSQ
packets are not added again; validation and changed Lieder labels remain excluded.

187 crops remain excluded: 107 lack a unique all-part musical window; 47 differ
in public/replayed token bytes; 20 have ambiguous timing transfer; 13 differ in
historical/public pixels. These are this audit's refusal criteria, not 187 human
quality rejections. The 13 pixel differences were already covered by the accepted
pixel review but remain outside the strict identical-pixel subset. Source-window
proof still does not independently certify historical scan boundaries.

Next: prepare a resumable frozen-core timing-head experiment using qualified
OSSQ/Lieder labels and stable frozen class dimensions. Benchmark feature-cache
extraction locally before estimating the full run; report unsupported classes
and keep evaluation separate. More archival repairs can expand coverage later;
they need not delay a common-class experiment. No GPU training, model promotion,
runtime activation, deployment, publication or corpus pin change here.
