# Lyric versus direction role controls — 2026-10-05

The current attachment pipeline has a real semantic failure. On the Bizet scan
`IMSLP83318`, the OCR row `a tempo.` sits between the same staff lines as the
lyrics. The row is routed to the vocal staff and its two OCR words are attached
to two decoded notes. The visually separate piano `tempo.` row is correctly
rejected as unsupported/non-vocal. Geometry therefore solves physical placement,
but it does not identify the role of text inside a lyric band.

The other real controls show the same distinction:

- `ben legato e cantando` is a written direction above the vocal staff, outside
  the established lyric row, and the bounded recovery rule rejects it.
- `Revenez au 1er mouv!` and `tremolando.` are written directions in the same
  physical gap/row family where a geometry-only rule could be tempted to retain
  them.
- `a tempo.` is the important in-row control: its placement is lyric-like even
  though its role is not.

Review page (live from the existing local loopback server):

`http://localhost:8765/c1-semantic-role-review-20261005/`

The source crops and inventory are under:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-controls-20261005/`

The review does not require individual labeling of every crop to establish the
engineering decision. The evidence is sufficient to keep automatic lyric
attachment disabled for rows whose semantic role is uncertain. A word whitelist
would be brittle, language-dependent and unnecessary for the safe gate.

Current measured results remain unchanged: corrected runtime context scores
82/145 exact native lyric targets across the four calibration pages; the Bizet
`a tempo.` control is one of the unresolved semantic cases, and the existing
score is not a semantic-role or production-promotion result.

The next useful experiment is a source-free semantic classifier evaluated on
separate pages, with an explicit ambiguous output. Its positive gate must measure
lyric recall, direction false attachment, and ambiguous abstention separately.
Until that exists, retain the current geometry and note-association diagnostics,
but do not wire them into automatic text fusion. Fusion remains disabled.
