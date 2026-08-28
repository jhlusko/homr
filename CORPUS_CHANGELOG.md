# Lieder corpus: what changed, when, and what it cost

Written because a bisect was about to test the wrong variable. The plan was to strip the
model-derived half of the corpus and see whether OSSQ recovered toward 447's 94.03 —
but **447 was itself trained on a corpus that was 50% model-derived**, so that arm
tests a recipe 447 never had, and a null result would have proved nothing.

Every corpus version, what produced it, and what it scored. Independent = OSSQ, 792
staves, never relabelled by this project. Ours = the Lieder held-out split.

**The noise floor on OSSQ is 4.06pp** (same corpus, two seeds), which is larger than
every corpus effect below. Treat the score column as indicative, not as evidence, and
read the floor section of `BENCHMARKS.md` before drawing a conclusion from any gap here.

Scores quoted here are from `BENCHMARKS.md`, which holds every paired comparison
with its confidence interval and better/worse/tied split.

## Lineage

| index | built | pairs | composition | checkpoint | independent | ours |
|---|---|---|---|---|---|---|
| `imslp_train_index.txt` | 08-25 08:06 | 3,353 | 3,353 `stage2_pairs_out` | — (pre-rebuild) | — | — |
| `imslp_train_index_v6.txt` | 08-27 14:02 | 7,273 | 3,608 `clean_v2` + 3,665 `reverse_v2` | **447** | **94.03** | 91.81 |
| `imslp_train_index_v7.txt` | 08-27 21:24 | 7,016 | 3,327 `clean_v5` + 3,689 `reverse_v3` | **448** | 92.80 | 92.69 |
| `imslp_train_index_v8.txt` | 08-28 01:14 | 6,989 | 3,548 `clean_v5` + 3,441 `reverse_v3` | **449** | 92.43 | 94.53 |

Baseline `426` (five original corpora, no Lieder scans): independent 91.70, ours 83.25.

**The trend is monotonic and opposite on the two axes.** Independent 94.03 → 92.80 →
92.43, back toward the 91.70 baseline. Ours 91.81 → 92.69 → 94.53. Every corpus change
since 447 has been rewarded by our own metric and punished by the only corpus that can
see us.

## What landed between 447 and 448

Three commits, all on 08-27, between v6 (14:02) and v7 (21:24).

| commit | change | effect on the corpus |
|---|---|---|
| `7933f20` 15:10 | Represent time signature numerators; act on review findings | vocabulary 260 → 279 rhythm tokens; labels gain `timeSignatureBeats_*` |
| `43b9db2` 16:18 | Measure the content gap, **exclude implied tuplets** and `IMSLP405017` | **−417 pairs** quarantined as "overfull"; one score excluded entirely |
| `9407e64` 16:46 | Stop `MeasureCutter` dropping the numerator | 77 of 416 labels regained a stated numerator |

Plus the pair sets themselves were rebuilt: `clean_v2` → `clean_v5`, `reverse_v2` →
`reverse_v3`.

## The exclusion accounts for almost the whole difference

Measured after the table above was written, and it narrows the search sharply.

| corpus | overfull pairs present | independent (OSSQ) |
|---|---|---|
| v6 -> **447** | **334** of 417 | **94.03** |
| v7 -> 448 | 28 | 92.80 |
| v8 -> 449 | **0** | 92.43 |

Monotonic: as the implied-tuplet pairs left the corpus, independent performance fell.

More decisive than the correlation is the set arithmetic. **v6 holds 355 stems that v8
does not, and 334 of them - 94% - are the overfull pairs.** v8 holds only 71 stems v6
lacks. So the corpora behind our best and our worst independent scores are near
identical in *which systems they contain*; the exclusion is the only change that
removed data in quantity.

That also demotes the displacement finding as an explanation for 447 vs 449: it alters
labels *within* stems both corpora share, so it cannot separate two checkpoints whose
stem sets differ almost entirely by this one exclusion.

**This paragraph previously called the phantom-system displacement "real, and worth
fixing".** That was written before the structural checks below, which found no
mechanism and could not establish which corpus is the displaced one. Nothing has been
fixed, because nothing was shown to be broken - see "Consequence for the displacement
finding".

## Ranked suspects, and what is already ruled out

**Ruled out — do not spend a run on these:**

- *Model-derived (`reverse`) pairs.* 447, 448 and 449 are all ~50% model-derived. This
  cannot explain a difference between them.
- *The arbitration rule.* Directly ablated: 449 reverted it to bar-count labels and
  OSSQ got **worse** (92.80 → 92.43), not better.
- *Training schedule.* 451 reran v8 at 2 epochs instead of 12: OSSQ 91.83, below 449's
  92.43. Fewer epochs did not preserve general performance, it just trained less.
- *The metre vocabulary, at inference.* Stripping the numerator slots from predictions
  recovered 0.03pp of a 1.23pp gap.

**Live suspects, in order:**

1. **The overfull exclusion (`43b9db2`).** It removed 417 pairs that 447 *had*. We have
   since shown the rule is unsound for the pairs it removes: **371 of the 417 are grand
   staves**, where `group_into_chords` takes the *minimum* duration across a chord, so
   a bar where the hands differ is neither their sum nor either hand's length — the
   corpus audit refuses to run any duration check on a grand staff for exactly this
   reason, and the discard rule calls that arithmetic anyway. Grand staves are
   discarded at 16.5% against 2.0% for single staves, 8.4x the rate. This is a change
   we identified as wrong and never reverted.
2. **The `clean_v2` → `clean_v5` rebuild.** Of the systems both corpora contain, 24%
   changed: 12.3% got a different bar count, 11.8% different content at the same bar
   count. 100 of 364 changed labels are *exactly the old label of a neighbouring
   system*, with negative offsets outnumbering positive 3:1 — phantom system detections
   displacing everything after them. 10 scores show a constant-offset run to the end of
   the score; 247 of 3,548 clean pairs (7%) belong to them.
3. **The metre numerator tokens in training.** Exonerated at inference, not in
   training: the embedding and output matrices grew by 19 rows, and 6.0% of the labels
   that state a numerator contradict their own bars.
4. **`reverse_v2` → `reverse_v3`.** Never examined. Same size (3,665 → 3,689 in v7),
   but rebuilt with the rest fix and the widened `MAX_SPAN`.

## The structural problem behind suspect 2

Consensus is the rebuild's central safeguard: a label is evaluation-admissible only
where two independent methods agree. **They are not independent.** Bar-count alignment
and reverse fingerprinting both consume the same `detect_imslp_systems` output — they
cut the same crops from the same boxes. A phantom system is inherited by both, both
agree, and the pair is stamped `consensus`.

Consensus validates *which measures a crop gets*. It cannot validate *the crop set*.
Every guard in the pipeline sits downstream of the one thing that is wrong.

## Structural checks that came back clean

Run while looking for the cause of the displacement. All negative, and worth recording
so the same ground is not covered twice.

- **Tiling.** Consecutive aligned systems should satisfy `end[i] == start[i+1]`.
  2,232 of 2,237 consecutive pairs do; 5 scores carry a gap. Not a mechanism for
  displacement, which tiles perfectly and merely shifts.
- **Degenerate detections.** 55 detected "systems" across 38 scores show two or more
  independent signs of not being music (typically a title block: under 35% of the
  score's median width and zero barlines). **All 55 are already `skipped` by
  `align_lieder_systems`'s narrow-detection filter and consumed zero measures.** The
  guard works; this is not a live defect.
- **Span against the image.** Assigned measures equal detected barlines for all 2,595
  aligned systems - but this is **circular**, since the alignment is constructed to
  match those counts. It is not evidence of anything.
- **Phantom geometry at the displacement onsets.** If a spurious system caused the
  constant offsets, the system at or before each onset should look anomalous. It does
  not: widths and barline counts at the onsets are indistinguishable from their
  neighbours in 5 of the 6 scores checked.

**Direction, resolved by reading the crops.** 126 stems in the current corpus carry a
label identical to a neighbouring system's pre-rebuild label, offsets skewing negative
98 to 28.

Two metrics were tried and rejected before inspection:

* **Ink.** Calibrating dark-pixel count against symbol count on the 1,311 pairs both
  corpora label identically gives a median error of 31% and a p90 of 92% - wider than
  the difference between the candidate labels. Its 59-45 split is a coin flip.
* **Structural glyphs.** A mid-system clef change or time signature is unambiguous, but
  115 of the 126 disputed pairs carry identical glyphs and the remaining 11 split 4-5.

So the crops were read directly:

| stem | what the scan shows | verdict |
|---|---|---|
| `IMSLP624193-sys6-v1` | a bass-clef change and a 3/8 time signature; **only the rebuilt label has either** | rebuild correct |
| `IMSLP227283-sys8-v1` | 3 barlines; rebuilt says 3 bars, old says 2 | rebuild correct |
| `IMSLP83314-sys10-v0` | 5 bars `[5,1,5,1,8]`; rebuilt over-extends by one, old truncates by one | neither exact |

**The pre-rebuild corpus is the displaced one**, erring short - which is the truncation
defect the rebuild was built to fix. No inspected case had the old label right and the
rebuilt one displaced. The rebuild over-reaches occasionally, a different and milder
fault than the one suspected.

**Nothing required fixing on the rebuild side.** `displaced` is now a review set of all
126 in `review.html` if a larger sample than these three is wanted.

## Bisect results - the leading suspect was wrong

**Arm A (452): remove the model-derived half.** OSSQ 92.43 -> 93.12, Lieder
94.53 -> 91.67. Dropping 3,441 reverse pairs cost 2.86pp on our own labels
(significant) and returned 0.69pp on the independent corpus (not individually
significant). They were inflating our headline number by nearly three points while
contributing nothing demonstrable to real performance.

**Arm B (453): restore the 417 overfull pairs.** OSSQ 92.43 -> **91.12**, below the
426 baseline of 91.70. **This refutes the leading hypothesis.** Excluding those pairs
is not what cost 447 its advantage; putting them back makes the corpus worse.

The argument that the rule is unsound still holds - `group_into_chords` really does
take the minimum duration across a chord, and that really is invalid on a grand staff.
But the pairs it removes are bad in practice regardless, so the overfull flag is
apparently *correlated* with some other defect rather than producing false positives.
Being right about the mechanism did not make the conclusion right.

**What both arms agree on.** Two independent additions to the corpus, the same
direction each time: more pairs, worse on the corpus that can see us, better on the
one that cannot.

| pairs | OSSQ |
|---|---|
| 3,548 | **93.12** |
| 6,989 | 92.43 |
| 7,369 | 91.12 |

447 remains unexplained at 94.03, still 0.91 above the best variant. Its corpus, v6,
was built from `clean_v2` and `reverse_v2` - pair sets no later corpus uses - so the
remaining difference lives in those, not in any rule since changed.

## v6: the two fixes, applied

Built 2026-08-28 05:34 with the overfull rule guarded to single staves and stale
numerators dropped.

| | v5 | v6 |
|---|---|---|
| clean pairs | 4,172 | **4,543** |
| quarantined as overfull | 417 | **46** |
| pairs with a stale numerator token dropped | — | 41 |
| skipped, divider count disagreed with span | — | 6 |

The 371 restored are exactly the grand staves; the 46 remaining are the single-staff
cases where the duration arithmetic is valid and the bar really is long.

**The restored pairs are NOT being trained on.** Arm B measured that directly and OSSQ
fell to 91.12, below baseline. The guard is still correct - the arithmetic it removes
was never valid on a grand staff - but the pairs it was removing are bad for some other
reason, so v6 keeps them out of the training index while no longer pretending the rule
justified it. The fix that is actually carried into training is the numerator one.

## What the v6 training index actually is

Stated because I described it as "452's corpus plus the numerator fix", and it is not.

| | pairs |
|---|---|
| 452's index | 3,548 |
| v6 index | 3,880 |
| shared | 3,548 |
| only in 452 | **0** |
| only in v6 | **332** |

A strict superset. The 332 extra are clean v6 pairs the consensus build had dropped -
its `rejected`, `phantom` and `unarbitrated` verdicts never reached the v8 clean subset,
while the v6 index is built from the raw clean manifest minus validation scores and
minus the originally-overfull stems.

So the v6 runs vary **two** things against 452: 332 more pairs and the numerator fix.
With a 4.06pp noise floor neither is separable from the other or from chance in a single
run, which is why these are being run at three seeds and reported as a mean. A matched
one-variable comparison would need 452's index re-run at the same three seeds - about
another two hours of GPU, and worth it only if the means differ by more than the floor.

## Corpus profile: a structural mismatch with the benchmark

From `profile_corpus.py`, and visible without any training run.

| | v6 training | OSSQ benchmark | Lieder validation |
|---|---|---|---|
| pairs | 3,880 | 792 | 292 |
| scores | 220 | **792** | **14** |
| **grand staff** | **1,749 (45%)** | **0** | 124 (42%) |
| single staff | 2,131 (55%) | 792 (100%) | 168 |
| median bars/pair | 4 | ~6 | 4 |
| symbols p75 / max | 90 / 332 | 34 / 108 | 93 / 255 |
| distinct labels | 96.7% | 99.9% | 97.9% |

**45% of the training corpus is a staff type the benchmark never contains.** OSSQ is
string quartets, so every crop is one staff; Lieder is voice-and-piano, and the piano is
a grand staff. Nearly half the training signal targets a shape OSSQ cannot reward, which
explains far more about the flat transfer than any label-quality story does - and unlike
those stories it needs no training run to establish.

Two further mismatches: our crops are systematically **shorter** (median 4 bars against
~6), and the dense grand-staff tail (332 symbols) has no counterpart in OSSQ (max 108).

**The Lieder validation set is 292 pairs from 14 scores**, half of them from 5. OSSQ is
792 pairs from 792 distinct scores. Our held-out set has roughly 14 effective independent
units, which is a large part of why its intervals are wide and why in-corpus movements
have been so easy to over-read.

Nothing alarming in duplication or triviality: 96.7% distinct labels, 0.9% trivial, no
all-rest labels, none unreadable.

## Rule for choosing the next arm

Vary one thing that **447 actually had** and 448/449 do not, or vice versa. Check this
table first. The arm launched before this document existed — v8 with the model-derived
half removed — fails that test, and its result should be read as "what does clean-only
do", not as "did the pseudo-labels cost us 447".
