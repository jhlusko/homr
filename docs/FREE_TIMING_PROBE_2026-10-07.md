# Image-only timing probe qualification — 2026-10-07

> **2026-10-07 semantic correction:** the validation ratio-2:1 examples previously
> called “duplets” are tremolo-associated (all 135 targets), including unmarked
> members of tremolo-bearing chords. These results do not qualify ordinary duplets.
> The piano part does have a 4/4 declaration; its vocal peer supplies independent
> bar-length evidence. A legacy paired-tremolo rule outperforms the head policy on
> the inspected complete page. See [the staff-lane investigation](STAFF_LANE_TREMOLO_TIMING_2026-10-07.md).


The complete299-crop free-decoding benchmark finished06:09UTC:291 Lieder and eight
OSSQ validation crops. Epoch4 and epoch6 timing projections read the same frozen
September core predictions. This is a raw crop research probe, not the accepted
full-policy MusicXML export, and does not qualify a model promotion.

Live review: http://localhost:8765/timing-free-decode-20261007/ .
Local artifact root:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-free-decode-20261007/`.

## What the run establishes

Image-only inference manifests contain no token labels or references. The frozen
core decodes greedily with float32 CPU/math-SDPA, final-divider enforcement off,
no v7 notation projections or production arbitration. A read-only forward hook
captures the hidden state which predicts each emitted token. Separate timing heads
produce categorical labels; they do not modify recognition tokens or serialize
MusicXML. All21 pilot crops exactly match an independent unhooked decode. All299
full-crop receipts confirm reference-free inference and the executed source hash;
no decode reached the sequence limit.

Evaluation loads the separately frozen references only after inference. It pairs
measures only when counts agree and their pitch/rest/chord structure matches
exactly. Timing is deliberately excluded from that pairing rule. All crops had
equal measure counts, but some measures fail the stricter event-structure check.
Unmatched targets remain in the total denominator. Conditional accuracy is
reported separately; it must not hide recognition/alignment exclusions.

Recognition counters use raw autoregressive note/rest order rather than canonical
chord sorting, and exclude key-signature tokens. They therefore are not numerically
interchangeable with earlier complete-stack symbol-error reports. Both head epochs
share exactly one recognition sequence; this is not a joint-training comparison.

## Paired label results

Numbers below compare passive head labels with raw legacy kern-token interpretation
on the same matched events. A gain/regression in two heads can describe the same
note; do not sum rows into a number of corrected musical events.

| Source | Head | Epoch4 improvements / regressions | Epoch6 improvements / regressions | Unmatched reference targets |
|---|---|---:|---:|---:|
| Lieder | Elapsed duration | 39 /48 | 72 /29 | 1091 |
| Lieder | Written duration | 11 /7 | 7 /14 | 1091 |
| Lieder | Tuplet ratio | 64 /10 | 77 /8 | 1091 |
| Lieder | Full-measure-rest flag | 63 /1 | 63 /1 | 72 |
| OSSQ | Elapsed duration | 3 /0 | 3 /0 | 14 |
| OSSQ | Written duration | 0 /0 | 0 /0 | 14 |
| OSSQ | Tuplet ratio | 3 /0 | 3 /0 | 14 |
| OSSQ | Full-measure-rest flag | 0 /0 | 0 /0 | 2 |

Lieder conditional matched-event accuracy:

| Head | Raw legacy | Epoch4 | Epoch6 | Matched /reference events |
|---|---:|---:|---:|---:|
| Elapsed | 97.10% | 97.01% | 97.55% | 9516 /10607 |
| Written | 99.26% | 99.30% | 99.19% | 9453 /10544 |
| Ratio | 98.09% | 98.66% | 98.82% | 9453 /10544 |
| Full-rest flag | 89.77% | 99.84% | 99.84% | 616 /688 |

The strongest evidence concerns new representation capabilities:

- All63 matched Lieder full-measure-rest positives are identified by both head
  epochs, versus0 under raw token interpretation. One of553 matched negative rest
  flags is falsely positive. This is a flag-label result; the shipping exporter
  may already infer some full rests from meter, so it is not63 demonstrated
  production improvements.
- Epoch6 recognizes73/114 matched Lieder duplet `2:1` ratios (64.04%), versus0
  under raw token interpretation. Unmatched duplet targets21 remain visible.
- Lieder triplet `3:2` correctness slips from86/152 raw legacy to83/152 epoch6.
  Ordinary `1:1` correctness slips9186→9185 among9187 matched events.
- OSSQ elapsed/ratio correctness improves101→104 among107 matched events, but
  validation still has only two scores. Its rest flags contain no positives.

Consequently do not replace all timing outputs with epoch6. Written-duration
prediction is already strong in the legacy path and epoch6 regresses it. Epoch4
is not a universal replacement either: its elapsed-duration result is net worse.

## Reproduction and next gates

Executed scripts and hashes are archived in `control/source-snapshot/`. Prepared
manifests separately bind model weights, image identities and evaluation references.
Each crop writes atomically and is resumable; changed code/model/protocol identities
are refused. The persistent `timing-free-decode` supervisor ran four workers with
two CPU threads each, first the21-crop parity pilot and then all299 crops. Both
scoring stages completed successfully and all workers exited0. Completed reports:

- `full-report.json`: reference coverage, conditional timing counts, raw recognition,
  per-event results and input/prediction hashes.
- `full-paired/paired-report.json` and `.md`: class support and individual paired
  improvements/regressions. No subsequent inference required.

The same supervisor command resumes interrupted crop files after identity checks:

```bash
cd /home/jhlusko/workspace/homr
export PYTHONPATH=/home/jhlusko/workspace/homr-artifacts/tools/audit-python:/home/jhlusko/workspace/homr
.venv/bin/python -m training.onnx.run_free_timing_probe \
  --run /home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-free-decode-20261007 \
  --workers 4 --threads 2
```

For a fresh reproduction, first call `qualify_free_timing_probe prepare` with
`--dataset` pointing to `timing-probe-20261006/dataset.json` and `--run` pointing
to a new directory; then run that supervisor. Audit a completed report with
`audit_free_timing_probe --report <run>/full-report.json --out <new-audit-directory>`.

Next prepare a conservative research policy retaining legacy written duration and
triplet decisions, using epoch6 as a candidate for duplet/full-rest capabilities.
Measure internal written/ratio/elapsed consistency before allowing changes. Full
rests must respect predicted meter and measure boundaries. Compare against the
accepted full-policy export, not this raw baseline; demonstrate MusicXML duration,
ratio and full-rest semantics without pitch/rhythm/accidental regressions. Freeze
any policy before independent broader OSSQ evaluation, including positive full
rests and more tuplets. Keep probe status and runtime rejection until qualified.

Four focused tests cover conservative alignment, legacy duration interpretation,
unmatched-target accounting and paired gain/regression classification. Black/Ruff,
review JavaScript syntax, HTTP review/report/status checks pass. No deployment,
model pin, publication or runtime default changes.

## Parallel semantic training-pool audit

The calibration-only Lieder metadata audit excludes prior C1 scores, current eight
reviewed scores and reserved examples. Remaining451 pages from102 scores contain
4771 lyric positives and45 direction positives across37 scores.1198 Dynamic glyph
positives are separate; they are not text-role negatives. Partial annotations do
not label unannotated text as non-lyric, and there is no complete credit/title class.
No formal test file, OCR or classifier fitting was involved.

Reproducible audit: `training/onnx/audit_semantic_training_pool.py`.
Receipt and page IDs: artifact root `semantic-training-pool-audit-v2/report.json`.

Before fitting a semantic discriminator, verify these direction examples and
expand representative in-row directions, credits and titles using separate scores.
Keep the42-region evaluated gallery out of training. Use score-disjoint splits,
explicit abstention, and report lyric recall plus direction/credit false lyric
selection separately. Existing direction positives are too sparse to equate
successful training loss with reliable role separation. Automatic lyric fusion
remains disabled.
