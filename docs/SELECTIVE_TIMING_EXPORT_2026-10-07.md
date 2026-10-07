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
