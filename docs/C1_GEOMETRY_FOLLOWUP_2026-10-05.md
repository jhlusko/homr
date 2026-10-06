# Lyric geometry follow-up — 2026-10-05

Expanded native-output scoring is now complete: 300/480 exact joint lyric targets
under an evaluation-only page-global verse permutation. Eight scans decode in
433.9s and cached export replay reproduces all eight XML files exactly. Dense
Franz extra-head and Bizet chord ambiguities are the next priorities. See
`docs/C1_EXPANDED_NATIVE_ATTACHMENTS_2026-10-05.md`; fusion remains disabled.

The targeted geometry diagnostic restores 65/70 correct vocal note associations
(five unresolved) and eliminates the two remaining errors in the 336 synthetic
repeated-pitch controls. Same-row neighbor recovery routes ten additional expanded
OCR regions, increasing coverage from 255/290 to 265/290. One recovered region is
the scanned `a tempo` direction. Physical routing is therefore still conditional
on lyric-role classification; neither change qualifies production fusion.

Review 13 original-scan/overlay comparisons at
http://localhost:8765/c1-geometry-followup/. Nine recovered lyric-looking regions,
the `a tempo` control and three staff-position disagreements are shown. No
individual approval decisions are requested. All 26 image assets return HTTP 200;
the pitch and direction overlays were visually inspected.

## Staff-position diagnosis

Recomputing the existing nearest-line/average-spacing rule reproduces all three
Franz position errors. The five segmented lines are not evenly spaced locally.
Instead, the diagnostic interpolates the staff lines at the actual head x coordinate
and measures y against the full bottom-to-top span:
`1 + 8 * (bottom_line_y - head_y) / (bottom_line_y - top_line_y)`.
The bottom line is position 1; each diatonic step advances one unit. It rounds
only for this diagnostic and records the continuous value and rounding residual.
Stored detector positions, recognition and exported XML are not rewritten.

| Franz case | Stored position | Continuous diagnostic | Rounded | Decoded/native |
| --- | ---: | ---: | ---: | ---: |
| Measure 8 note 2, A4 | 5 | 4.099 | 4 | 4 |
| Measure 12 note 4, F4 | 3 | 2.097 | 2 | 2 |
| Measure 12 note 6, A4 | 3 | 4.215 | 4 | 4 |

Interpolated positions agree with the decoded written-pitch sequence on all 70
count-exact vocal calibration notes, compared with 67/70 stored positions.
Maximum rounding residual is 0.345. This does not establish confidence thresholds
or accuracy on new scans, ledger extremes, ambiguous head shapes or grandstaffs.
Only single five-line staffs are supported; unknown geometry fails explicitly.
Neither native text nor native pitch enters interpolation or matching.

With exact-pitch and ordered-sequence guards, the unchanged-page geometric matcher
now accepts 65 correct/0 wrong, versus the previous 59 correct/0 wrong. The 336
combined deletion/attention-shift stress controls accept 2,656 correct/0 wrong,
versus 2,312 correct/2 wrong. These are repeated synthetic controls on the same
six calibration systems, not independent accuracy or an attention-error rate.
Single-head deletion, single-decode deletion and exact duplicate-head controls
also have zero wrong accepted pairs. Ambiguous pairings remain unresolved.

## Same-row recovery

The nine lyric-looking unresolved regions examined lie below the staff's bottom
line but their OCR rectangles fail the extra half-spacing margin by a few pixels.
An opt-in recovery uses a strict-pass OCR neighbor on the same physical row and
staff gap. It still requires the entire rectangle to remain between the actual
staff lines, requires one unique candidate gap, and cannot use another recovered
row as a seed. The global strict margin and bottom-of-page rejection stay intact.

| Score | Before | After | Recovered | Still unresolved |
| --- | ---: | ---: | ---: | ---: |
| IMSLP89287 | 32 | 35 | 3 | 6 |
| IMSLP89290 | 39 | 45 | 6 | 3 |
| IMSLP617260 | 47 | 47 | 0 | 0 |
| IMSLP89796 | 48 | 48 | 0 | 1 |
| IMSLP435041 | 15 | 15 | 0 | 3 |
| IMSLP286659 | 32 | 32 | 0 | 3 |
| IMSLP83318 | 27 | 28 | 1 | 6 |
| IMSLP83314 | 15 | 15 | 0 | 3 |

The two original pages retain all 75 previously routed regions with no additional
recovery. Their selection population differs from the expanded eight-page
area-200/10%-overlap selection, so the populations are reported separately.
Recovered `tempo.` is seeded by `a`: the original scan confirms the musical
direction. This is a semantic-role failure control, not a new lyric true positive.
Native verse remains null; physical row order cannot identify the original verse
number on bilingual layouts. Native reference labels are evaluation-only.

Local packet:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-geometry-followup-20261005/`:
`geometry-report.json`, `original-report.json`, `stress-report.json`, eight per-score
routing records and `review-v2/` (current; original `review/` preserved). Pitch
comparison labels use native measure and notated beat, after checking the source
hash and meter. Reports include input/code hashes, continuous pitch
evidence, row recovery seed IDs, and explicit unresolved cases. All computations
reuse cached geometry/OCR; no new training, page decode or production change.

Twenty-four focused tests and scoped lint pass. Tests cover slope/scaling,
grandstaff refusal, staff-line intrusion, absence of a recovery seed, sequence
ambiguity and exhaustive sequence alignment. All jobs finished, ETA 0.

Next use the expanded scans' native measure/timing targets to score real note/verse
attachment, keeping text-role failures such as `a tempo`, detector misses, genuine
count mismatches and ambiguous rows explicit. Broader pitch geometry confidence
and grandstaff/chord ownership remain open. Runtime producer/fusion stays disabled.
