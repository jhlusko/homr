# Write-ups

Local copies of the two published write-ups. Both cover the same work for different
readers, and both are condensed from `ENSEMBLE_TRANSCRIPTION_DESIGN.md` and
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` — those remain the authority when they disagree.

| file | title | audience |
|---|---|---|
| `omr-findings.html` | Extending homr for Ensemble Scores | academic/technical report; head-to-head against upstream homr, per-change attribution |
| `homr-devs.html` | What We Added to homr | blogpost for homr developers; same material, implementation-first |

`omr-findings.html` opens directly in a browser. `homr-devs.html` no longer does: its
"What the numbers look like" and "Cross-staff reranking, live" sections embed the real
OurTextScores score editor's checkpoint-compare view (`score-editor/`, a vendored static
build — see `docs/BUILD_EMBED.md` in `OTS_Web`) pointed at real MusicXML pairs
(`scores/*.musicxml`) via `compareLeft`/`compareRight`. Those iframes fetch same-origin,
which a plain `file://` open cannot do — serve this directory over HTTP first, e.g.
`python3 -m http.server` from `docs/writeups/`, then open `http://localhost:8000/homr-devs.html`.

**These are not the artifact files as published.** An artifact is published as page content
and the viewer wraps it in `<!doctype html><head>…</head><body>`, supplying a CSS reset.
Copied verbatim, both files rendered in **quirks mode** — no doctype — and picked up the
browser's default 8px body margin, which neither stylesheet overrides because it relied
on that reset. The local copies add the doctype, document structure, and an equivalent
reset so they render standalone as they do published. Content is otherwise unchanged.

If either is republished as an artifact, publish from the original unwrapped source
rather than these files, and reuse the existing artifact URL so the published page updates
in place instead of forking a second copy. **`homr-devs.html` can no longer be republished
this way as-is** — the artifact viewer's CSP and sandboxing don't support same-origin
iframes to sibling static files, which is what its live compare embeds now depend on. A
republish would need those two sections stripped or replaced with static renders first.

- Extending homr for Ensemble Scores — https://claude.ai/code/artifact/a46bacbb-9542-4248-950a-7989321a74c0
- What We Added to homr — https://claude.ai/code/artifact/7daaac8f-0b89-4c4a-a8ec-64a7a2851282

## review.html — label and beaming adjudication

`review.html` is a review tool, not a write-up. It presents one item at a time — the scan
crop, then a live compare of two readings — and records a verdict per item in
`localStorage`, exportable as JSON. Two sets:

| set | items | question |
|---|---|---|
| Truncation (labels) | 54 suspects + 12 controls | Does the reference label account for everything in the scan? |
| Beaming (model) | 20 | Does our predicted beaming beat the engraver's automatic beaming (= what upstream renders)? |

Data lives in `review-data/{truncation,beam}/`. It needs the same HTTP server as
`homr-devs.html`. Background and the measurements that motivated it are in
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md`, under the 2026-08-27 truncation entry.
