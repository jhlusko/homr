# Fixed lyric recovery validation — 2026-10-05

Four previously unused calibration scores are frozen for a fixed-rule validation
pass. This is reserved calibration validation, not the formal test partition or
proof of detector/core training independence. No thresholds will be adjusted from
these results. Native references enter evaluation only.

| Scan | Layout | Native physical measures | Targets |
| --- | --- | --- | ---: |
| IMSLP154059 — Chaminade, Les rêves | Multi-verse opening | 1–5 | 30 |
| IMSLP24047 — Fauré, Après un rêve | Multi-verse continuation | 36–48 | 48 |
| IMSLP476183 — Warlock, Lusty Juventus | Single-verse opening | 1–8 | 29 |
| IMSLP556523 — Hardelot, In 1822 | Single-verse continuation | 37–46 | 26 |
| Total | | | 133 |

Selection excludes the ten already used diagnostic scores plus reserved
IMSLP154086/IMSLP89797. It uses native boundary-safe system windows, verse/layout
stratum and pixel hash order, never OCR/model quality. Original pages were viewed
to inspect edition/layout/vocal-over-piano structure. Literal scan transcription,
note pixels and full native verse identity are not independently certified.

Frozen gates are detector minimum area 200, OCR overlap 0.1, unique existing-row
overlap at least half the larger height and horizontal extension four median seed
heights. The candidate cannot create rows or modify existing row assignments.
Both baseline and candidate use the same core/export cache and literal native
note timing/pitch/alteration/duration evaluation with oracle page-global verse
mapping. Gains and lost exact targets are counted separately with multiplicity.

An explicit synthetic in-row `a tempo` control passes the geometric recovery rule.
Thus row geometry is not a semantic lyric/direction classifier; even a positive
validation result will not authorize runtime activation. This differs from the
earlier real distant `legato` control, which the horizontal bound rejects.

During input inventory, the first two rows of the formal test manifest, including
their positive annotations, were inadvertently printed. Neither score is selected
here, and no threshold/algorithm choice uses them. Do not claim the entire test
manifest remained unread; no formal test detections or scores were collected.

## Running status and local evidence

The job runs one thread at lowered priority in persistent tmux
`c1-fixed-validation`, with persistent watcher `c1-fixed-validation-watch`.
Full-page OCR (50.4s) and character positioning (30.5s) completed. Detector startup
initially failed because the provider import path was missing; the runner now
includes the provider path and resumed successful stages without rerunning OCR.
First-attempt failure/progress receipts and executed source snapshots are retained.
Remaining sequence: detector, fixed overlap selection, four core decodes, bounded
recovery, paired native scoring; watcher then writes exact-target gain/loss and
error decomposition. Initial remaining ETA is approximately 8–10 minutes.

Local root:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-fixed-rule-validation-20261005/`.
Monitor `progress.json`, stage logs and `decode/progress.json`; final outputs are
`comparison.json`, `completion.json`, `watcher-completion.json` and paired
`score-baseline/`, `score-candidate/`, `audit-baseline/`, `audit-candidate/`.
Inference inputs are separate from native evaluation metadata. Run identity and
selection receipts hash the inputs and executed code. No GPU/network/deployment.

Eight focused tests and scoped lint pass, including score-disjoint hash selection
and the admitted inline-direction limitation. Runtime fusion remains disabled.
