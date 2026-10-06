# Piano introduction and vocal measure context — 2026-10-05

The local runtime now retains Chaminade's piano-only introduction and final
piano grandstaff. The vocal part is first, with two silent introductory measures;
both parts reach physical measure 8. All 28 previously decoded vocal notes retain
their pitch, accidental, duration and within-measure onset. The piano retains
171 pitched notes versus 102 previously: 69 restored decoded notes, not a claim
that every restored note is correct.

The fallback requires an already connected piano-only row followed by at least
two complete connected solo/piano rows, in vertical order. It runs only after
confident geometry has declined, before periodic trimming. Flat alternation alone
does not establish an introduction. Existing uniform and geometry-supported
ensemble layouts retain their paths.

Silent bars require at least two nonempty, closed piano measures and a single
decoded meter. An explicit numerator supplies their duration; legacy streams
without it require equal observed measure durations. Changing meter, incomplete
bars and unrepresentable durations abstain. This is a narrow introduction fix,
not a policy for synthesizing rests wherever staff detection loses a part.

The first real replay retained all physical staffs but abstained on silent bars
because the piano's legacy rhythm durations disagree. Its explicit 4/4 token
provides the context in the final version. Both outputs are preserved. The
diagnostic prefix helper also recognizes already exported silent measures,
preventing a second +2 offset.

Using the original detector-selected OCR rows, Chaminade recovers **32/42** exact
native lyric targets directly from the corrected XML, with 34 attachments and
zero attachments without a native lyric-note identity. Exact cached exporter
replay passes. Neither reference text nor native note geometry enters inference.
The reference coverage was previously manually recertified after inspecting this
calibration page; this is not a new held-out result, semantic-role certification,
native verse-ID accuracy or independent pixel ownership.

One-thread paired runtime controls are recorded under:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-runtime-introduction-paired-v2-20261005/`

`progress.json` records each completed page; `completion.json` records the final
five-page comparison. Source snapshots and model/input hashes are retained.
`chaminade-event-comparison.json` checks vocal event preservation;
`score-chaminade/report.json` contains native lyric scoring. The preliminary,
abstaining replay is sibling `c1-runtime-introduction-20261005/`.

All five page runs succeeded in 217.4 seconds total. Fauré, Warlock and Hardelot
and the OSSQ Glazunov quartet (`sq7358579:0002`) produce XML identical to their
original candidate baselines. This establishes no regression on those controls,
not new accuracy on untested pages. All four lyric pages replay the exporter
exactly and score **82/145** native targets (32, 5, 25, 20), directly from runtime
XML with zero added diagnostic measure offset; `score-four-runtime/report.json`.
Seventy-five focused tests and lint pass. Persistent tmux job completed; ETA zero.

Reproduce with `python -m training.onnx.qualify_intro_context`; supply the four
calibration page manifest, original decode root, pinned models, an existing OSSQ
candidate directory and a new output directory. It runs sequential child
processes with one CPU thread and records failures without overwriting originals.

Next: qualify lyric semantic
role against in-row directions and broader untouched pages before runtime lyric
wiring. Detector fusion remains off; model/provider pins and production unchanged.
