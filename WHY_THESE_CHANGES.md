# Why these changes: the case for the 5 PR branches

This document summarizes the measured evidence behind the five branches published
to https://github.com/jhlusko/homr for review before offering them upstream to
[liebharc/homr](https://github.com/liebharc/homr):

- `pr/zero-duration-rest-crash`
- `pr/cross-staff-consistency`
- `pr/cross-staff-rerank`
- `pr/score-profile-conditioning`
- `pr/barline-segnet-channel-split`

Every number below comes from a real run against real data, not a projection - see
each section for how it was produced and where to find the fuller writeup.

## A direct comparison: this fork vs. vanilla `upstream/main`

A 200-page real-corpus sample (`benchmark_sample.txt`, mixed scanned and synthetic
pages across dozens of pieces) was run once through a fresh clone of vanilla
`upstream/main` and once through this fork, same pinned base checkpoint, same
images, nothing else changed.

| | upstream/main | this fork |
|---|---|---|
| pages processed OK | 198/200 | 199/200 |
| crashes | 1 | 1 |
| timeouts (>90s) | 1 | 0 |

The one crash is identical on both sides and is not a bug: `sq7383977:0120.png` is
a blank/non-music page, and `homr.main` correctly raises `"No noteheads found"` on
it (0 staff line fragments, 0 detected noteheads). Neither codebase's changes are
about raw crash-avoidance on typical pages, so this comparison reading as
"basically a wash, with the fork slightly ahead" is expected, not disappointing -
the real case for these changes is in *decode quality*, not robustness on pages
that were never going to be a problem either way. That quality evidence is
substantial and is what the rest of this document covers.

## `pr/cross-staff-consistency`: real, frequent problems upstream has no way to see

Stage A (`homr/cross_staff_consistency.py`) is a deterministic pass over a
system's already-decoded staves - measure counts, key/time signatures, clef vs. an
optional score profile, barline positions, shared-motif articulation, and more.
None of this exists in vanilla `homr` at all; a system decoding four staves that
silently disagree with each other is invisible to it.

A systematic 200-page benchmark (same sample as above) found:

- **71.4% of pages (142/199) have at least one Stage A finding.**
- The single largest finding, `barline_position_mismatch`, fires 306 times -
  confirmed by direct inspection to be a genuine decoder rhythm/duration
  divergence between staves, not a vision/crop artifact (every case shows a clean
  constant additive offset or constant ratio between the agreeing staves and the
  diverging one - exactly the signature of one mis-decoded note or a systematic
  meter misread, never an occasional non-proportional jump a crop error would
  produce).
- Stage B's conservative, majority-corroborated repair proposals
  (`homr/cross_staff_repair.py`) cover 44.2% of pages (88/199) once
  `propose_majority_position_corrections` is included - real, useful coverage of
  that largest finding, though the majority of it remains *localization*
  (which staff, which measure, how large an offset) rather than a full content
  fix, by design: there is no defensible automatic guess for a genuine decode
  error.

Full detail, including the tier-1 key/time-signature majority-vote repair and the
shared-motif-articulation check: `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §4.

## `pr/cross-staff-rerank`: a measured, validated fix for that largest finding

Phase 1 (`homr/cross_staff_rerank.py`, `homr/transformer/decoder_inference.py`)
builds directly on Stage A's own finding above: at each staff's narrowest-margin
rhythm decisions, fork a full alternate decode, then keep whichever candidate best
matches the other staves' majority cumulative barline positions. Cost-gated: a
system only pays for the expensive forking pass once Stage A already flags a
finding on the cheap greedy decode.

Measured on the same 200-page sample:

- **428 → 339 combined `barline_position_mismatch`/`measure_duration_mismatch`
  findings - a 20.8% reduction, zero pages made worse** across all 198
  successfully-processed pages.
- **Ground-truth spot-check: 38 of 49 resolvable corrected measures now match real
  ground truth exactly, 2 neither-matches-truth, 0 regressions.** (9 more had no
  ground-truth mapping available and were correctly excluded rather than guessed
  at.) The two non-matching cases were checked by hand: the *greedy* decode was
  already wrong there too - reranking didn't break a right answer, it landed on a
  different wrong one for a genuinely messy passage.

This is a clean, decisive, real-data result: reranking never made a page worse,
and where ground truth is available to check against, it overwhelmingly turned a
wrong decode into a correct one.

## `pr/score-profile-conditioning`: a measured positive training signal

An optional, additive per-sample embedding (instrument family, clef, staff count,
transposition) conditions the decoder's input - zero-initialized gate, so
attaching it reproduces the exact baseline until trained. `phase20` (10 epochs,
frozen core, only 9 tensors trainable) measured a **mean validation-loss delta of
+0.0615 with vs. without profile context, all 10 epochs positive** (range
+0.04-0.07) - a real, repeatable signal from a mechanism that costs nothing when
absent.

A trained instance of this mechanism (9 tensors, 227KB, not the 287MB base
checkpoint) is published as a release asset:
https://github.com/jhlusko/homr/releases/tag/phase22-profile-context-weights

## `pr/zero-duration-rest-crash`: a real, reproducible crash, now a graceful degradation

`build_note_chord` had an unconditional `assert group_duration > Fraction(0)` that
fires whenever a decode inconsistency (the rhythm head says "note", the pitch head
still holds the "no pitch" sentinel) produces a zero-duration rest inside a chord
group. This didn't happen to appear in the 200-page benchmark sample above, but
it's directly reproducible (2 new tests, including a full `generate_xml` pipeline
repro) and was the confirmed mechanism behind a real production crash reported
against a separate project built on this code
(`inference_failed`, whole page lost for one malformed symbol). Now logs a warning
and drops the one malformed symbol instead of taking down the entire page.

## `pr/barline-segnet-channel-split`: prepared, not yet measured

Splits bar lines into their own segmentation channel, motivated directly by
`barline_position_mismatch` being the single largest Stage A finding above -
worth testing whether the current width/height heuristic for classifying bar
lines (rather than decoder rhythm accuracy) is a contributing bottleneck. This is
training-side preparation only: it doesn't touch the live inference pipeline, and
actually measuring its effect requires a retrain + re-exported segmentation model,
which has not been run. Included for completeness, not overclaimed as validated.

## What's shared and deferred

Three of these branches (`cross-staff-consistency`, `cross-staff-rerank`, and
`score-profile-conditioning`'s clef-check use) share a live-pipeline wiring step
(`homr/staff_parsing.py`'s `parse_staffs`, plus `main.py`'s `--score-profile` CLI
flag) that's deliberately not included in any single branch here, since it depends
on more than one of them landing first. That's the natural follow-up PR once
there's agreement on which of these to take.
