# Expanded lyric routing and repeated-pitch limits — 2026-10-05

Follow-up: interpolated whole-staff pitch geometry restores 65/70 correct original
associations and removes the two remaining stress-control errors. Same-row seed
recovery raises expanded routing to 265/290 but also routes `a tempo`, so semantic
role is still unqualified. See `docs/C1_GEOMETRY_FOLLOWUP_2026-10-05.md` and
http://localhost:8765/c1-geometry-followup/. Original failed probes are preserved below.

Segmentation staff geometry works on all eight additional calibration scans,
including the three that failed the earlier horizontal staff-line heuristic.
The physical-gap router assigns 255 of 290 preselected intact OCR regions; 35
remain explicit unresolved cases. This is routing coverage, not native ownership
or lyric precision. No native verse IDs are produced and no fusion is activated.

| Score | Physical staffs | Selected OCR regions | Routed | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| IMSLP89287 | 12 | 41 | 32 | 9 |
| IMSLP89290 | 12 | 48 | 39 | 9 |
| IMSLP617260 | 12 | 47 | 47 | 0 |
| IMSLP89796 | 12 | 49 | 48 | 1 |
| IMSLP435041 | 9 | 18 | 15 | 3 |
| IMSLP286659 | 12 | 35 | 32 | 3 |
| IMSLP83318 | 12 | 34 | 27 | 7 |
| IMSLP83314 | 12 | 18 | 15 | 3 |

Inputs are the existing area-200 / 10% detector-union overlap selection, not the
two-page pilot's row-recovery selection. These selection populations differ.
Staffs are captured before grandstaff merging using HOMR's segmentation path.
Identifiers are physical pre-merge indices, not native parts or decoded voice IDs.
Core/heads decoding is not run. Original image and selection hashes are checked;
nonidentity autocrops are refused because their offsets are not captured.

The one-thread persistent tmux pass completed all eight pages successfully in
98.7 seconds, with progress/completion receipts and source snapshot. Local packet:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-staff-routing-20261005/`.
Each `page-N/routing.json` retains routed and unresolved rows and cached text.

Unresolved regions include titles/credits but also clear lyric candidates:
`schwan- ken - de Wi - pfel schiesst gül` on IMSLP89287 and
`säu - selt das Gras auf den`, `Mat`, `ten,` on IMSLP89290.
These must remain in evaluation. A bounded staff gap alone does not guarantee
semantic role, system membership, row identity or syllable-to-note ownership.

## Real count differences

The saved two-page outputs have four genuine final note-count differences:

| Score / runtime staff call | Segmented heads | Final decoded notes | Staff lines |
| --- | ---: | ---: | ---: |
| Chaminade / 8 | 17 | 16 | 5 |
| Franz / 4 | 48 | 40 | 10 |
| Franz / 6 | 62 | 56 | 10 |
| Franz / 7 | 57 | 58 | 10 |

These are piano staves, not the count-exact vocal systems used by the attachment
calibration. They are an unscored inventory: the report deliberately leaves
`ownership_accuracy` null. Grandstaff/chord ownership remains unsupported by the
monophonic sequence guard. Different first-pass/final counts elsewhere also show
cleanup changing recognition output; neither array should be silently zipped to
segmentation when counts disagree.

## Repeated-pitch stress test

Added 336 explicit synthetic controls on the existing six vocal systems. Each
removes one decoded note, then shifts a neighboring same-pitch decoded attention
x coordinate by ±20, ±40, ±80 or ±120 original-page pixels. Native references are
used only to score; these are repeated calibration stress controls, not a measured
frequency of attention errors or independent held-out accuracy.

The geometry plus exact-pitch guard accepts 2,790 correct and 29 wrong pairs. All
29 wrong pairs have repeated pitch and occur at 80/120-pixel shifts; 20/40 controls
have no wrong pairs. The 40-pixel matching gate cannot detect this fault because
shifted attention is close to the wrong same-pitch head.

The new opt-in sequence guard keeps a geometric pair only if it appears in every
maximum-length ordered equal-staff-position alignment. It uses predicted pitches
and segmented staff positions, without native labels or distance-based alignment
tie breaking. Ambiguous insertion/deletion alignments stay unresolved. It does
not rematch rejected pairs and requires monophonic chronological ordering.

With this guard the stress test accepts 2,312 correct and two wrong pairs. Original
page coverage falls from 62/70 to 59/70 (zero wrong on the unchanged pages). The
remaining two errors are the same Franz note under shifts -80 and -120: physical
measure 12 note 4 (F4, `ten` / `sing`) is matched to note 3 (F4, `dor` / `tos`).
The detector reports note 4 one staff step too high. After deletion of note 3's
decode, this position error makes the wrong sequence pairing appear unique.
Ordered-pitch evidence is therefore insufficient when segmentation pitch is wrong.

Local reports:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-association-limits-20261005/`:
`report.json` (pitch guard), `sequence-report.json`, and
`original-sequence-report.json` (unchanged and single-fault controls).
They record source-note identities for evaluation, input/code hashes, per-offset
totals, explicit wrong matches and the real count-difference inventory.

Twenty-one focused tests pass, including exhaustive comparison of the sequence
algorithm with all maximum alignments for short two-pitch sequences. This validates
the algorithm, not the correctness of segmentation pitch or ownership in scans.

Next inspect/correct staff-position uncertainty using raw head/staff geometry,
diagnose the unresolved lyric rows without relaxing the router globally, and link
expanded predicted notes to native measure/timing references for real attachment
scoring. Preserve detector misses, ambiguous rows, chords and count mismatches.
All jobs complete, ETA 0. Runtime integration remains gated.
