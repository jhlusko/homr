# Write-ups

Local copies of the two published write-ups. Both cover the same work for different
readers, and both are condensed from `ENSEMBLE_TRANSCRIPTION_DESIGN.md` and
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` — those remain the authority when they disagree.

| file | title | audience |
|---|---|---|
| `omr-findings.html` | Extending homr for Ensemble Scores | academic/technical report; head-to-head against upstream homr, per-change attribution |
| `homr-devs.html` | What We Added to homr | blogpost for homr developers; same material, implementation-first |

Open either directly in a browser.

**These are not the artifact files as published.** An artifact is published as page content
and the viewer wraps it in `<!doctype html><head>…</head><body>`, supplying a CSS reset.
Copied verbatim, both files rendered in **quirks mode** — no doctype — and picked up the
browser's default 8px body margin, which neither stylesheet overrides because it relied
on that reset. The local copies add the doctype, document structure, and an equivalent
reset so they render standalone as they do published. Content is otherwise unchanged.

If either is republished as an artifact, publish from the original unwrapped source
rather than these files, and reuse the existing artifact URL so the published page updates
in place instead of forking a second copy:

- Extending homr for Ensemble Scores — https://claude.ai/code/artifact/a46bacbb-9542-4248-950a-7989321a74c0
- What We Added to homr — https://claude.ai/code/artifact/7daaac8f-0b89-4c4a-a8ec-64a7a2851282
