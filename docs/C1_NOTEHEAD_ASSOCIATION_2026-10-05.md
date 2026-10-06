# C1 notehead association mismatch audit — 2026-10-05

Geometry-only mutual nearest association is useful but does not yet qualify a
runtime producer. At a 40-page-pixel distance gate and a 5-pixel ambiguity margin,
it correctly associates 65 of 70 vocal notes on the two existing calibration
pages; five remain unresolved. Missing decoded notes expose four incorrect
associations across the 70 controlled deletion cases. Do not enable this matcher.

The matcher uses only decoded attention x coordinates and segmented notehead x
coordinates in original-page units. It neither requires equal counts nor retries
after removing matched points. Near ties, duplicate heads and equal-x chords stay
unresolved. Native pitch, timing and text never enter matching. Evaluation uses
the existing count-exact, strictly monotonic vocal linkage, which is a calibration
proposal rather than independently certified note-pixel ownership.

| Control at distance 40 | Cases | Available pairs | Correct | Wrong | Unresolved decoded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 6 systems | 70 | 65 | 0 | 5 |
| One segmented head removed | 70 | 782 | 724 | 0 | 128 |
| One decoded note removed | 70 | 782 | 723 | 4 | 55 |
| One segmented head duplicated | 70 | 852 | 722 | 0 | 130 |

Counts aggregate repeated perturbations of the same six systems, not independent
held-out samples. The duplicated-head control deliberately creates an exact tie;
arbitrary false positives have not been qualified. Distance gates 20, 40 and 60
were swept without selecting a production default. All three produce four wrong
matches in the missing-decode control. Increasing distance alone does not resolve
that failure. The report records affected native measures/notes for inspection.

Local machine-readable report:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-notehead-association-20261005/report.json`.
It includes input/code hashes and explicit wrong matches. Reproduce with
`python -m training.onnx.audit_lyric_notehead_association --anchors <canonical-anchor-directory> --out <report-path>`.
Thirteen focused association, geometry, sequence and syllable tests pass.

Next investigate those wrong pairs and require additional source-independent
evidence (local staff-relative pitch and sequence consistency) before accepting
matches. A test with real count mismatches, chords and extra heads is still needed.
Automatic system/verse routing is also pending: physical row order cannot stand
in for native verse identity, because Chaminade's English verse 2 appears above
French verse 1. Existing horizontal staff-line heuristics fail on three expanded
pages; preserve those failures rather than silently dropping them. Ground truth
belongs in evaluation only. Fusion remains disabled; all audit jobs are complete.
