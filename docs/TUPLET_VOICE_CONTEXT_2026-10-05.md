# Tuplet voice-context recovery — 2026-10-05

Recovered all 32 selected crop/model witnesses (23 correct raw predictions damaged by isolated cleanup and nine false-tuplet repair controls), across 21 page/voice groups. Both model arms use their own predictions; reference meter is never injected. Neighboring published systems are assembled in page order, and ownership survives chord reconstruction and tuplet copies.

| Selected witness group | Current cleanup exact | Guard exact | Tuplet cleanup disabled exact |
|---|---:|---:|---:|
| Correct raw predictions (23) | 22 | 23 | 23 |
| False-tuplet repair controls (9) | 8 | 7 | 0 |

These are note/rest recognition results after removing repeated key-signature headers from the scoring projection, not complete page scores. The original all-symbol metrics are retained separately. Selection was based on earlier crop behavior; this is a diagnostic, not an untouched test.

Every repair control has an earlier-page predicted meter declaration for the same part. Three have a same-page declaration before musical material in the target crop: two recover it from earlier systems, and one already declares 2/2 in the target. Earlier-page declarations may be superseded on the current page; HOMR does not currently carry meter between separate page requests.

Most isolated-crop damage disappears when the available same-page voice context is restored. The remaining current-cleanup failure on a correct prediction is accepted-stack `sq10307350_0016_0003_4`, with 24 rhythm errors; the guard avoids them. Current cleanup misses the false-tuplet control `sq8885571_0015_0004_3` (six rhythm errors), where the page contains different meters. The guard repairs it but misses two unequal-duration-chord controls, `sq7354505_0011_0002_2` and `sq7354505_0015_0002_2` (four errors each). Do not enable the guard globally on this evidence.

## Boundary provenance and limits

OSSQ page/system/part addresses, original scans and native source XML are available locally. The reconstruction covers the published neighboring systems; independent native scan-to-source measure-boundary certification remains open. Predicted barlines are not proof that boundary measures are complete. Lieder uses the independently sourced `alignment_v4_boundary_safe.json`, with model predictions excluded from alignment. No source meter or reference note is inserted into model context.

The nine-case review at http://localhost:8765/ now includes same-page and earlier-page declarations, context scores and links to the full original scan pages. The rendered panels retain their isolated-crop interpretation; the new context scores are explicitly separate.

## Actual page observation

An accepted-stack CPU run of Mendelssohn `sq8885571:0006.png` completed in 92.2 seconds: 20 staff decodes, five systems and four assembled voices. The observed pipeline uses the margin/candidate-reranking path. This is a separate production-segmentation observation, not the published-crop reconstruction, and does not certify native full-page accuracy. The stage-capturing repeat completed in 102.6 seconds with identical staff captures, voice-cleanup captures and final XML SHA-256. Two systems invoked candidate reranking. In the second system, part 2, it selected a candidate that replaces all 16 `note_24` triplets with ordinary durations before voice cleanup. The raw staff has 16 rhythm errors against the reviewed reference; its pre-cleanup segment has zero. This is candidate reranking, not evidence that the key/time-signature cross-staff repair performs rhythm correction. No meter declaration was emitted in any of the four page voices. Inference defaults are unchanged.

## Local artifacts and reproducibility

Artifact root: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `voice-context-20261005-v2/report.json`, `cases.json`, `page-voices.json`: canonical recovery, input/source hashes and retained metrics. `executed-source/` preserves the executed recovery script.
- `voice-context-page-observation-20261005/`: actual-page decoded staff/voice captures, XML and identity receipt.
- `voice-context-page-rerank-20261005/`: repeat with candidate-reranking captures.
- `guarded-tuplet-20261004/review/`: updated nine-case gallery, full scan assets and context provenance.
- `voice-context-20261005-launch/`: logs and launch receipts. The first observation rejected the page at the default 10-megapixel diagnostic budget; the successful run explicitly allowed 12 megapixels without altering the original image. Its launcher exit file contains extra printf arguments; the first line is the actual zero exit status.

Tools: `training/transformer/recover_tuplet_voice_context.py`, `training/transformer/extend_tuplet_context_review.py`, and `training/onnx/observe_page_voice_context.py`. All runs use one CPU thread. The recovery's earlier unqualified all-symbol report is preserved in `voice-context-20261005/`; use v2 for note/rest conclusions.

Next: independently certify the remaining native OSSQ boundaries, then compare full accepted and epoch-27 stacks under identical policies. A meter-local cleanup experiment must retain both correct-triplet and false-tuplet controls, honor meter changes and incomplete bars, and address unequal-duration chords. Broad cross-page meter inheritance is not justified by these selected cases.
