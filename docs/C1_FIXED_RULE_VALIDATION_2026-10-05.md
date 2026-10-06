# Fixed lyric recovery validation — 2026-10-05

**Completed:** frozen baseline 50/133 (37.6%), candidate 51/133 (38.3%): one exact
target gained, none lost. All baseline attachments survive. Attachments increase
61 to 64; outputs without exact native lyric-note identity stay at one. All four
cached exporter replays are exact. Jobs/watchers finished, ETA 0; fusion off.

| Score | Baseline exact | Candidate exact | Native targets |
| --- | ---: | ---: | ---: |
| Chaminade | 0 | 0 | 30 |
| Fauré | 5 | 6 | 48 |
| Warlock | 25 | 25 | 29 |
| Hardelot | 20 | 20 | 26 |

The broad calibration gain did not reproduce at similar magnitude here. More
significantly, this pass exposed upstream part/context assumptions. On Chaminade,
the first runtime voice is the piano grand staff; the vocal part is second. The
scorer's first-voice assumption therefore yields zero lyric attachments. The
second exported part omits the two-bar piano introduction from its measure count:
an evaluation-only +2-measure diagnostic matches all 30 frozen-window native
targets at exact note timing/pitch/alteration/duration. No voice reassignment,
offset repair or improved-score claim was made in the frozen comparison.

The original page visibly has a piano-only introduction and two vocal systems,
while the stored boundary metadata lists two systems covering physical measures
1–5. The scan's final vocal system appears to extend through measure 8 (42 native
lyric targets through that measure), so complete-page source windows need renewed
certification. The quoted 133 denominator is the frozen-window diagnostic, not
certified complete-page coverage. Do not blame its low total on recognition or
promote/reject lyric recovery using it as a full-page release gate.

**Next priority:** identify vocal staffs/parts from source-free page structure and
lyric-row evidence, preserve the piano introduction's measure context, and certify
complete native page windows. Evaluate these context changes separately on
calibration before another reserved validation pass. Keep inline-direction
semantic controls explicit; geometry alone still admits `a tempo`.

Results are `comparison.json`, paired scores/audits and
`chaminade-part-identity-audit.json` under the local root below. The comparison
watcher completed successfully. Eight selection/recovery/row tests plus one added
missing-provenance regression test pass; scoped lint passes.

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

## Execution history and local evidence

The job ran one thread at lowered priority in persistent tmux
`c1-fixed-validation`, with persistent watcher `c1-fixed-validation-watch`.
Full-page OCR (50.4s) and character positioning (30.5s) completed. Detector startup
initially failed because the provider import path was missing; the runner now
includes the provider path and resumed successful stages without rerunning OCR.
First-attempt failure/progress receipts and executed source snapshots are retained.
Detector completed in250.7s and four decodes in172.1s. Recovery then encountered
final notes without staff-call provenance; those notes cannot seed recovery.
The metadata guard was fixed without changing selection thresholds. Recovery and
paired scoring completed from the existing caches. First/second-attempt failure
receipts and the initial partial recovered directory remain preserved. Final
candidate selection is `recovered-v2/`, with executed finishing source preserved
in `final-source-snapshot/`. Final scoring/audits complete; current ETA 0.

Local root:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-fixed-rule-validation-20261005/`.
Monitor `progress.json`, stage logs and `decode/progress.json`; final outputs are
`comparison.json`, `completion.json`, `watcher-completion.json` and paired
`score-baseline/`, `score-candidate/`, `audit-baseline/`, `audit-candidate/`.
Inference inputs are separate from native evaluation metadata. Run identity and
selection receipts hash the inputs and executed code. No GPU/network/deployment.

Eight focused tests and scoped lint pass, including score-disjoint hash selection
and the admitted inline-direction limitation. Runtime fusion remains disabled.
