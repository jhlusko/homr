# C1 notehead association mismatch audit — 2026-10-05

Latest expansion: all eight scans have usable segmentation staff grids, with
255/290 selected OCR regions routed and 35 explicit unresolved. Repeated-pitch
stress controls expose 29 wrong pitch-guard matches; an opt-in ordered-sequence
guard reduces them to two but lowers original coverage to 59/70. Remaining errors
involve segmentation pitch uncertainty. See
`docs/C1_EXPANDED_ROUTING_LIMITS_2026-10-05.md`; these diagnostics remain unqualified.

## Follow-up: predicted pitch guard and physical row routing

The four wrong matches all shift a C5 to a neighboring D5 or B4 in Chaminade.
An exact staff-position guard rejects all four without rematching rejected notes.
It compares the decoded written pitch, interpreted using the decoded clef, with
the segmentation detector's existing staff-position integer. No native pitch,
source text or verse enters matching. Accidentals are irrelevant to this geometric
check. This diagnostic only supports a single decoded vocal clef and five-line
staffs; it refuses grandstaff mapping or changing clefs.

At distance 40 and ambiguity margin 5, results become:

| Control | Correct | Wrong | Unresolved decoded |
| --- | ---: | ---: | ---: |
| Original 70 vocal notes | 62 | 0 | 8 |
| One head removed, repeated controls | 698 | 0 | 154 |
| One decoded note removed, repeated controls | 697 | 0 | 85 |
| One head duplicated, repeated controls | 696 | 0 | 156 |

Compared with the original geometry matcher, the guard also rejects three correct
Franz pairs because segmented staff position differs from decoded/native position
by one diatonic step: physical measure 8 note 2 (A4), and measure 12 notes 4 (F4)
and 6 (A4). Allowing a one-step tolerance would also admit the four wrong matches.
Repeated-pitch neighbors can still pass this guard. Thus zero wrong matches in
these simulated controls is not a general ownership guarantee or release gate.

A one-thread replay in persistent tmux completed in 109.8 seconds. Capturing raw
staff grids and segmented staff positions preserved both exported XML files and
decoded-symbol caches exactly. All first-pass coordinate records also remain
identical. All 414 final coordinates and 70 vocal native/XML anchor comparisons
remain valid under the existing calibration scope.

The physical-row router uses those captured grids and the previous cached
source-independent lyric-selection diagnostic. It routes all 20 Chaminade and 55
Franz selected OCR regions to unique physical staff gaps. The six reviewed
Chaminade regions and one Franz separator witness all reach the expected staff.
The other 68 regions are not individually certified routing references; 75/75
routed is coverage, not accuracy. Input selection remains conditional and does
not newly qualify text-role classification or detector misses.

Physical row ordering is retained separately from native verse identity. The
reviewed Chaminade upper row is physical row 1 but native verse 2; the lower row
is physical row 2 but native verse 1. Franz's reviewed upper row is native verse 1.
Every prediction leaves `native_verse` null. Routing requires the entire OCR box
to fit a bounded gap, with half a staff-line spacing of margin, and rejects the
unbounded space below the last staff. It does not identify vocal instruments or
general system membership across multi-part layouts.

Local packet:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-staff-pitch-anchors-20261005/`:
`association-report.json`, `provenance-audit.json`, `row-routing-report.json`,
per-page grids/anchors/logs, source snapshot and completion receipt. Reports retain
input/code hashes. Eighteen focused tests and scoped lint pass.

Next qualify repeated pitches, real missing/extra heads and count mismatches,
investigate the three segmented-position disagreements, and expand staff/row
routing to the eight additional calibration scans. Native verse identity and
lyric-role/attachment quality remain open. Fusion remains disabled; no jobs run.

## Initial geometry-only probe (preserved)

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
