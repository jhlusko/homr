# Selective timing export probe — 2026-10-07

The source-free crop serialization probe completed all 299 cached validation crops.
It added seven correct full-measure-rest flags across five Lieder crops, with zero
flag regressions. No duplet proposals passed the conservative context gates;
written duration, elapsed duration, and existing triplet decisions remained unchanged.
This is **not an accepted full-stack accuracy qualification**: these crop caches
lack v7 notation predictions and complete page context.

## Policy and safeguards

Keep legacy written-duration and triplet decisions. A duplet requires a complete
adjacent pair, agreement between written/elapsed/ratio heads, and an exact repair
of a closed overfull bar to independently inferred meter. A full-rest proposal
can only add `<rest measure="yes">`; it cannot change duration. Missing or
conflicting meter causes abstention. Candidate XML must preserve pitch, notation,
written type/dots, measure count, and meter.

Neighboring decoded crops supply meter without references or candidate-bar duration
voting. Only 16 of 299 crops supplied trustworthy meter: 137 were refused for
mixed staff context, 131 for missing/conflicting declarations, eight for unsupported
OSSQ context identity, six for declaration conflict, and one for weak agreement.
Thus absence of duplet changes does not disprove their benefit; context coverage
is currently too small. References were opened only after both XML arms were frozen.

An initial run incorrectly rejected equivalent XML because of an empty
`<notations/>` element. The corrected comparison removes empty containers from
both arms. Both runs are retained; use v2 results below.

## Local artifacts and reproducibility

Final report and paired exports:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-selective-export-20261007-v2/`.

Scripts: `training/onnx/probe_selective_timing_export.py` and
`training/onnx/observe_page_timing_probe.py`. Three focused tests cover duplet
pairing, full-rest duration invariance, and independent neighboring meter context.
The research head bundle remains `probe-trained`; no inference promotion occurs.

Next: replay complete accepted pages with passive epoch-6 timing observations,
require an exact archived control XML match, audit how many final symbols retain
observations, then test the selective policy using complete predicted voice
context. Ground truth remains confined to evaluation after proposals are frozen.
Do not claim the seven crop flag gains as production improvements or request
visual review of identical-looking rest glyphs.

## Accepted-page replay result

The complete Haydn sq9146376 page replay finished successfully. Its unchanged
control MusicXML matches the archived accepted output byte for byte. The passive
probe observed 593 decoder symbols; 317 of 320 final note/rest symbols retain timing
observations through cleanup. Three missing observations must abstain, without
reference-based or ordinal reconstruction. No candidate export or page accuracy
claim has yet been made.

Local replay, receipt, final voice observations and missing-observation audit:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-page-observation-20261007/`.

Next implementation must retain complete symbol objects including v7 notation
when capturing the final voices; the current JSON observation is an audit format,
not a faithful replacement for the full exporter input. Then derive independent
bar context and freeze both full-stack XML arms before ground-truth scoring.

## Complete v7 export replay — four pages

Preserved complete final symbol objects (including v7 notation) in locally generated
trusted pickle artifacts, then re-exported control and selective arms. All four
controls reproduce archived accepted XML exactly. Ground truth was opened only
once proposals and both XML arms were frozen. Five focused tests pass.

| Page | Timing observations / final note-rest symbols | Control event error | Selective event error | Changes |
|---|---:|---:|---:|---:|
| OSSQ Haydn sq9146376 | 317/320 | 8/320 (2.50%) | 8/320 (2.50%) | 0 |
| OSSQ Glazunov sq7358579 | 364/364 | 2/364 (0.55%) | 2/364 (0.55%) | 0 |
| Lieder IMSLP154086 | 255/255 | 1/254 (0.39%) | 1/254 (0.39%) | 0 |
| Lieder IMSLP89797 | 479/479 | 8/486 (1.65%) | 8/486 (1.65%) | 0 |

Event error is the existing fixed-staff page MusicXML edit diagnostic, not crop
SER. Pitch, rhythm and accidental errors also remain identical. Full-rest flag
metadata is not part of that event metric. Here no flags changed either, because
there were no eligible proposals.

Leave-one-bar-out context supplied meter for all 84 Haydn voice-bars. Glazunov's
72 voice-bars conflict with predicted meter declarations; IMSLP154086's 33 have
no predicted declaration; IMSLP89797's 50 split between missing/changing meter
and mixed-staff refusal. No meter is borrowed from ground truth. Repeated
signatures are conservatively refused even when they might merely repeat a
header. This intentionally limits coverage.

Both OSSQ and Lieder full-page arms abstain. These pages establish safe replay and
abstention, **not a measured full-stack improvement**. Existing validation exposure
and policy development after inspecting crop results remain limitations.

The initial replay reader incorrectly treated plain `.musicxml` as compressed
`.mxl`. The frozen export arms were already written. Fixed reference reading and
reran only cached export/scoring, retaining both versions and their source snapshots.

Local root:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-page-selective-20261007/`.
Each score has `observed/` and final `selective-v2/`; use `summary.json` for receipts,
metrics, predicted meter tokens, bar-length distributions and head-positive cases.
The four jobs completed; no training or replay remains running.

Next priority: obtain complete accepted-stack examples for the known crop-level
duplet/full-rest positives and certify their native reference windows. Do not
weaken meter safeguards simply to force edits on this four-page cohort. If broader
page tests still abstain, report the coverage limitation and keep this timing policy
research-only. C1 semantic lyric routing remains a separate release blocker.
