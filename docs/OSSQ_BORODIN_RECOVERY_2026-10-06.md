# Borodin alignment recovery and OSSQ pixel differences

The missing historical mapping for Borodin's String Quartet No. 2
(`sq10675759`) is recoverable from the frozen public index and August symbolic
segments. Public-v2 data and model pins have not changed.

## Why the older build excluded it

August alignment log reports 179 symbolic segments versus 178 eligible scanned
systems. The public index contains 179 systems /716 training crops. The only
public system absent from August's system image inventory is `0014:0006`.

Scan page 14 (printed page 15) visibly has five quartet systems. Its archived
detector file includes a blank-top box at confidence 0.2997972071, just below
the 0.30 cutoff, then five real systems. The recorded ignore list removes
system 1. With the blank-top box excluded, numbering starts at the first real
system and that ignore removes actual music. Public-v2 numbers the five real
systems 2–6. **All 20 public page-14 crops are byte-identical to August crops
numbered 1–5**, proving the numbering shift and restoration of the first real
system. Threshold-sensitive blank-top detection explains this behavior, but
the rebuilt raw detector confidences were not recovered; the exact confidence
change is an inference, not a measured result.

## Checks completed

The needed public shard is `train-0000.tar.zst`, located by the packager's
lexicographic score grouping, not corpus index row order. Its SHA-256 matches
the pinned lockfile. All 716 Borodin images, tokens and sidecars were extracted
locally. A mistakenly selected `train-0001.tar.zst` was also hash-verified and
retained; it contains other scores, not Borodin.

The sorted historical reconstruction supplies 179 candidate source windows.
Page 14 corresponds to source measure labels 113–147, in five groups of seven.
Concatenated symbolic segments match the original score's visible-note
signatures across all four parts: 4,392 /4,324 /4,214 /3,886 notes. That signature
checks note type, pitch step/octave, chord membership and rest status; it is not
an exhaustive notation comparison or independent visual review.

Current label replay with source slur, dynamics, barline restoration and carried
clefs succeeds for all 716 crops. Timed-note pitch/rhythm/accidental columns
match all 716 published labels. Full token files match 664; the remaining 52
match after barline-type normalization. There is no timed-note regression in
this replay. This source/label consistency does not by itself certify scan
measure ownership.

After the page-14 numbering correction, 703/716 crop PNGs match the August
backup exactly; 13 differ. Separately, all 150 public-validation PNG differences
are decoded-pixel differences, not just PNG metadata/compression. Of those,
124 change dimensions and 26 retain their dimensions. Width changes range
from −10 to +8 pixels, height changes from −2 to +2. Visual inspection of the
largest size-change case shows the same passage at a slightly different scale;
this observation does not qualify the other crops.

## Local evidence and next step

All artifacts are under:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-mapping-recovery-20261005/`

- `borodin-candidate-system-mapping.json`: historical scan/source candidates.
- `borodin-page14-shift-receipt.json`: 20 exact shifted image comparisons.
- `borodin-all-pixel-receipt.json`: 703 exact /13 different image comparisons.
- `borodin-symbolic-source-signature-receipt.json`: four-part source check.
- `borodin-token-replay-classified-receipt.json`: exact versus barline-only changes.
- `validation-pixel-differences-hash-verified.json`: all 150 decoded differences.
- `borodin-public/` and `borodin-token-replay/`: published and replayed labels.

`python -m training.onnx.audit_ossq_pixel_differences --candidates ... --out ...`
reproduces validation pixel analysis, verifies input hashes and refuses to
replace existing evidence. Regression tests distinguish encoding-only changes
from pixel changes and reject changed evidence; both audit tests pass, as do
scoped lint/diff checks.

Next inspect the 13 Borodin and 150 validation differences with their source
measure/part context, then qualify physical scan ownership before replaying
timing labels. The 716-window gap is now a recovered provenance candidate,
not an unexplained missing score or permission to add it to a qualified replay.
No new training, publication or runtime change is needed for this investigation.
All jobs in this investigation have completed.
