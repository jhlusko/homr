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

## Positive-case complete pages — accepted-stack gains confirmed

Ran four additional complete pages, selected from previously inspected positive
crop cases. Each page received an unchanged accepted control run and a passive
head-observed replay; all four XML controls match exactly. Exported both policy
arms before reference scoring. A persistent watcher completed the strict-location
and explicit-rest-flag audits. All jobs finished; eight focused tests and lint pass.

| Page | Final timing coverage | Correct flag gains | Flag regressions | Reference rests matched / total |
|---|---:|---:|---:|---:|
| Chaminade Aubade, IMSLP154110 p001 | 250/250 | 1 | 0 | 6/6 |
| Schubert Der Zwerg, IMSLP16369 p004 | 478/478 | 0 | 0 | 10/12 |
| Grandval Chrysa, IMSLP578231 p002 | 393/396 | 3 | 0 | 22/33 |
| Faisst Jugend, IMSLP621832 p003 | 184/184 | 1 | 0 | 15/15 |

**Five correct full-rest metadata improvements, zero regressions.** Across these
pages, 53/66 source rests match uniquely at fixed physical staff, native measure,
onset and duration; 13 unmatched rests remain in the denominator. Correct matched
flags increase from 47 to 52. The five edits match explicit source
`<rest measure="yes">` flags. No written length or elapsed duration changes occur,
so pitch/rhythm/accidental metrics, strict onset metrics, meter and v7 notation
remain unchanged. This is a real accepted-stack **metadata** gain, not improved
recognition accuracy. Selection after crop results means it is diagnostic rather
than an independent promotion test.

Three pages already had complete certified alignment. Der Zwerg's leading system
was ambiguous, although its five native bars and boundaries were certified. The
immediately following aligned system begins at source index 69; five bars count
back to index 64. Existing candidate boundaries already agree with 64–69. Recorded
this adjacent-anchor derivation, preserved original alignment rows, and required
certified counts, contiguous systems and exact candidate agreement. No recognizer
prediction or reference-note matching was used to certify that boundary. Full page
window is source indices [64,87), zero-based. Tests reject uncertain counts,
uncertain anchors, gaps and contradictory endpoints. Other windows are [0,9),
[0,20), and [0,12) respectively.

The diagnostic pixel budget expands explicitly from 10 to 14 million original
pixels to include Aubade's 3048×4299 scan; no resizing. Models and runtime policies
remain pinned to the accepted stack. These are previously used validation scores,
not untouched test. Boundary certification does not establish lack of training
exposure.

Local results:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-positive-pages-20261007/`.
Use `summary.json`, `pages.json`, `receipt.json`, each score's `selective/report.json`
and `selective/timing-context-and-flags.json`. Sources executed, original alignment
rows, complete final v7 objects, frozen proposal/XML hashes and watcher completion
receipts are retained. Temporary jobs ran in `timing-positive-pages` and
`timing-positive-watcher`; both completed.

### Next priority

Full-rest flag-only handling has a useful diagnostic result. Keep the bundle
research-only until broader score-disjoint qualification; no review of nearly
identical glyphs is needed for metadata edits.

Duplets need a separate staff/voice-context investigation. Der Zwerg's piano voice
contains 89 event predictions of ratio 2:1, but crosses both physical staves; its
page also lacks usable meter declarations. The current policy abstains on every
edit. Do not remove these guards wholesale. Next retain and validate a per-staff,
per-voice timing timeline, using prediction-only meter/boundary evidence. A proposed
duplet must preserve chord simultaneity and untargeted durations, form a complete
run, and fit its own voice's independently established bar context. Freeze exports
before scoring against the certified native window. Report written type, consumed
time, triplet preservation and recognition errors separately.
