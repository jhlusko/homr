# OSSQ-OMR ground-truth errors: measures where parts disagree on length

## RETRACTED - do not forward anything in this document to anyone

**Every finding below is invalid.** The `<name>.musicxml` files this entire investigation
read as "ground truth" (sitting next to each training image at `images/*/original/
<page>.musicxml`) are not part of the OSSQ-OMR corpus at all - they are `homr.main`'s own
output, written to that exact path as a side effect of running it, from an earlier point
in this session. Confirmed directly: every one of these files' `<identification>` block
reads `<encoding><software>homr</software></encoding>`, and their `<work-title>` is
whatever HOMR's own (imperfect) title-detection produced, not the real title. Checked
against the pristine local backup taken before this session's runs: no `.musicxml` ever
existed at that path in the real corpus - only `.png`, `_teaser.png`, and the bbox `.txt`
files.

**The real ground truth** lives at each piece's own top level -
`scores/<composer>/<piece>/sq<id>.musicxml` - a whole-score file with genuine MuseScore/
IMSLP/OpenScore provenance. Confirmed for the Moeran page that prompted this discovery:
real composer credit, real transcriber credit, `<software>MuseScore 3.6.2</software>`,
CC0 license.

**What this means for every finding below**: the 999-measure corpus sweep, the
Beethoven/Borodin spot-checks, and the corpus-wide `deep_barline_audit.py`/
`deep_barline_audit_broad.py` runs (documented in `DECODER_RHYTHM_ACCURACY_DESIGN.md`
§7.1) all compared HOMR's decode against **HOMR's own earlier output**, not real ground
truth. None of it says anything about the OSSQ-OMR corpus's actual quality. Whether the
underlying pattern (a decoder reproducing specific "errors") reflects anything real about
the model, or is simply an artifact of near-deterministic decoding producing similar
output across two runs, is genuinely unknown and unaddressed by anything in this
document.

**Update: the redo happened, and it found the opposite of a corpus defect.**
`metadata/scanned/systemwise/sq<id>:<page>:<system>.yaml` gives the corpus's own
`measure_start`/`measure_end` per page/system - the exact mapping needed, already
provided, not something to reconstruct. Redone correctly for the Beethoven case
specifically (real whole-score ground truth + this mapping + a fresh `homr.main` run):
**the viola's measure is not a ground-truth error at all.** Real ground truth agrees
with the other three parts exactly (`quarter-eighth-quarter-eighth`); HOMR's own fresh
decode is what turns the eighth into a dotted quarter. This page contributes zero
entries to this document - it's a confirmed HOMR decode error instead, documented in
`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.1.

**Extended corpus-wide with `deep_barline_audit_v2.py`** (the corrected rerun of
`deep_barline_audit.py`, same 200-page sample, using `ossq_ground_truth.py` instead of
the broken path): of the 91 `majority_position_correction` proposals across the sample,
only **1** lands on a real ground-truth disagreement - and it's a rounding-scale blip
(three parts read `1440` duration units, one reads `1441` - a 1-part-in-1440
discrepancy, not a meaningful musical error). The other 90 either show ground truth
agreeing (87 - real evidence of decoder divergence, not a corpus defect) or have no
usable mapping (3). **This corpus, at least for this specific failure family, looks
much cleaner than the invalid 999-measure sweep suggested** - that number was itself
built on the same broken ground-truth file and should not be trusted at its prior
value. The 999-measure sweep and every entry above it in this document remain
genuinely unverified against real ground truth (not redone) - this new, much smaller
scope (91 measures, corrected) is the only part of this document with a trustworthy
answer as of this update.

The rest of this document is kept for the historical record of what was (incorrectly)
found and how the error was investigated - not as a source of real findings.

---

Originally drafted for possible forwarding to the
[ossq-omr](https://github.com/MALerLab/ossq-omr) authors. **Do not do this** - see the
retraction above.

## The invariant

In valid music notation, every part in a system spans the same duration within a given
measure - this is definitional, not a stylistic convention. A ground-truth file where
two parts disagree on a measure's total length is, by construction, a labeling defect:
either a wrong note/rest duration, a missing or extra note, a dropped tuplet marking, or
some other encoding mistake - never a legitimate reading of two different lengths.

## How this was found

Started from one instance discovered while investigating HOMR's own decode accuracy
(`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.1): Beethoven's Grosse Fuge, Op. 133, p. 13,
measure 8. The encoded ground truth gives the viola a dotted quarter the other three
parts don't have (a `1/4`-whole-note excess) - and HOMR's own decode reproduced that
exact excess, which first looked like "HOMR correctly read a real irregularity."
**Directly comparing the encoding against the scan overturned that**: the scan shows a
plain quarter-eighth-quarter-eighth in the viola, identical to the other three parts.
The dotted quarter exists only in the encoding.

Given the invariant above, this single confirmed case justified a systematic,
corpus-wide check rather than treating it as a one-off: `training/omr_datasets/
ossq_measure_length_audit.py` (committed alongside this doc) walks every part's
note/backup/forward stream in document order, tracks a cursor and the *peak* it
reaches within each measure (backup rewinds the cursor for a second voice; a measure's
true length is the peak reached, not where the cursor ends), and normalizes by each
part's own current `<divisions>` value (this can differ between parts, and can change
mid-piece - confirmed necessary by the Borodin case below). Two parts' peaks disagreeing
for the same measure number is the flag.

Before trusting the results, the method itself was checked against a case with real
multi-voice `<backup>` structure (Bartók Quartet No. 1, p.15, measure 24) by hand-tracing
the cursor through two voices and a chord - it produced the same peak the script
reported. It was also checked for a positional-alignment bug (if one part had an extra
or missing measure somewhere, everything after it would misalign and produce cascading
false positives): every one of the 999 findings below has all disagreeing parts citing
the *same* measure `number`, ruling that out.

## Results

**999 measures across 164 of 475 ground-truth files (~35%)** show at least one part
disagreeing with the others on total length. Full listing:
`training/omr_datasets/ossq_audit_findings/measure_length_findings.txt` (regenerate with
`python -m training.omr_datasets.ossq_measure_length_audit --dataset-root <path>`).

Split by direction (a part can either have *more* duration than the majority, as in the
Beethoven case, or *less*):

| direction | count |
|---|---|
| some part has more than the majority (excess) | 499 |
| some part has less than the majority (shortfall) | 459 |
| both directions in the same measure | 41 |

The roughly even split matters: a shortfall alone could plausibly be explained by a
systematic "omitted trailing rest" convention in this corpus's encoding pipeline (a part
whose written content just stops before the measure's notated end, with no explicit
closing rest) - relatively benign, and not necessarily a content error. Excess cases
cannot be explained that way at all; nothing about an omitted rest would ever produce
*more* duration than the majority. With excess essentially as common as shortfall, most
of these look like genuine content-level defects (wrong note values, dropped/extra
notes, missing tuplet markings), not one uniform, explainable convention.

Files with the most flagged measures (a first-pass ranking of where problems cluster,
not necessarily where they're worst):

| file | flagged measures |
|---|---|
| Kalliwoda, Op.90, p.22 (synthetic) | 24 |
| Cherubini Quartet No.1, p.14 (scanned) | 23 |
| Beethoven Op.18 No.1, p.5 (scanned) | 22 |
| Debussy Op.10, p.50 (scanned) | 21 |
| Beethoven Op.127, p.23 (scanned) | 19 |

## Confirmed against the actual scan (not just the encoding)

Two cases in this investigation have been checked against the primary source image
directly, and **both turned out to be ground-truth errors, of two different kinds** -
neither survived as "HOMR made a decode error" once the actual page was checked. The
remaining 997 measures below are flagged by the invariant alone (internal disagreement
between parts in the same encoded file), which is sufficient to say *something* is
wrong, but not yet checked against each page's scan to confirm exactly what - these two
are the only ones confirmed at that level of certainty so far, and both point the same
direction: don't trust the encoding at face value even when it looks internally
consistent.

- **Beethoven Grosse Fuge Op.133, p.13, measure 8** (detailed above): a wrong note
  value - a dotted quarter in the ground truth that doesn't exist on the page. This one
  *is* on the measure-length-disagreement list (the wrong duration shows up as a total
  that disagrees with the other three parts).

- **Borodin Quartet No. 2, p.24, measure 6 (internal numbering), violin II**: a
  different and arguably more concerning failure mode. HOMR's decode disagreed with the
  ground truth (originally described as "a constant `1/8` whole note" offset, which
  turned out itself to be wrong - see the correction below), and the ground truth's
  *total* duration for this measure agreed exactly with the other three parts (`3/4`
  each) once normalized for each part's own `<divisions>` value - which first read as
  "the encoding is internally consistent, so this must be a genuine HOMR decode error,"
  probably an implicit (unmarked) triplet HOMR's decoder had no context to resolve.
  **That reading didn't survive comparing the actual note content against the scan.**
  The encoded ground truth is a quarter rest, four 16th notes, and a closing quarter -
  five discrete events. The scan shows six plain eighth notes beamed 3+3 under one
  slur, no rest, no closing quarter. These are different rhythms that happen to sum to
  the same total, not two readings of the same passage - there is no triplet ambiguity
  to explain, the encoding simply doesn't match the page. **This measure is *not* on
  the 999-count list above**, because `ossq_measure_length_audit.py` only compares
  total duration per measure across parts - it is structurally blind to a content
  substitution that preserves the correct total. That is a real limitation of the
  method: the true error count in this corpus is likely higher than 999, and finding
  the rest would need a check that compares actual note content, not just totals.

  **Correction**: the "constant `1/8` offset" description was itself wrong - it came
  from checking only the first divergent measure, not the whole system.
  `deep_barline_audit.py` (built to avoid exactly this mistake at scale) confirmed
  `propose_majority_position_corrections` does not actually fire on this measure at
  all: the offset across the system's five barlines is `-3/8, -3/4, -9/8, -9/8, -9/8`,
  not constant. This doesn't change the ground-truth verdict above (the encoded content
  is still wrong, confirmed against the scan directly), but it does mean this measure
  was never a clean, isolated single-measure divergence to begin with.

Net effect from the two manually-checked pages: of the two cases checked against their
scans, zero are confirmed HOMR decode errors.

## Corpus-wide follow-up: `deep_barline_audit.py` (91/91, not just 2 pages)

The two-page result above prompted a proper corpus-wide version, not just more manual
spot-checks: `training/omr_datasets/deep_barline_audit.py` calls HOMR's own pipeline
in-process (rather than shelling out and scraping printed diagnostics) to get exact
per-system barline counts on every system, converts each `propose_majority_position_
correction` proposal's local measure index into the correct absolute ground-truth
measure number (summing barline counts across every earlier system on the page - the
exact step that made the manual Borodin check unreliable), and checks that measure's
ground truth directly.

**Run across the full 200-page benchmark sample: 91 `majority_position_correction`
proposals total. All 91 land on a measure where ground truth already disagrees by the
invariant. Zero land on a measure where ground truth agrees.** Full output:
`training/omr_datasets/ossq_audit_findings/majority_position_correction_ground_truth_check.json`.
One additional instance (Wolf String Quartet, p.22, system 1, measure 6, cello - a
`+1/2`-whole-note excess) was spot-checked visually for extra confidence: the flagged
part's measure is a clearly different, longer sustained figure than the same measure in
the other three parts, consistent with the ground truth's own numbers.

**This is stronger than "some corpus noise" - it's total, for this specific slice.**
Every clean, majority-corroborated position divergence found across 200 real pages
traces to an already-broken ground-truth measure. Read alongside the Beethoven case's
own speculation (HOMR's decode reproduces the *exact* wrong duration the mislabeled
ground truth encodes) and this looks like the general mechanism, not a one-off:
`propose_majority_position_corrections` firing appears to function as a ground-truth-
defect detector, riding on a decoder that has learned to faithfully reproduce those
specific defects - not an independent decoder-error detector. See
`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.1 for what this means for that document's
Phase 1/2 plans (short version: corpus cleanup now looks like the higher-priority lever
for this specific finding, ahead of building beam-search reranking to fix decoder
errors this sample found none of).

## Open questions before this goes anywhere

- Whether to sample more of the 999 against their scans before reporting upstream, to
  get a real (not just plausible) estimate of how many are genuine content errors vs.
  some other explainable pattern this investigation hasn't considered.
- Whether the excess/shortfall split holds up under a larger, targeted spot-check, or
  whether one direction turns out to concentrate in a way that suggests a single
  systematic cause (e.g., one specific transcriber, one specific edition, one specific
  pipeline stage).
- Whether this method (or `ossq_measure_length_audit.py` itself) is useful to the
  ossq-omr/omr-data-preprocessor authors as a validation check in their own pipeline,
  independent of whether these specific 999 measures get individually corrected.
- The Borodin case shows the duration-only method has a real blind spot: a content
  substitution that preserves the correct total goes undetected. Whether a
  content-level check (comparing note sequences, not just totals) is worth building -
  and how it would even establish ground truth to compare against, given the primary
  source is a scan, not another symbolic file - is an open design question, not just an
  extension of the existing script.

---

# Systemwise segments can omit the key signature (2026-08-29)

**This finding does not repeat the mistake retracted above.** It compares against the
corpus's own `scores/<composer>/<piece>/musicxml/scanned/systemwise/*.musicxml` - the
files `convert_ossq.py` actually reads - never against any file HOMR wrote, and it was
confirmed by opening the segment XML directly rather than by inference from tokens.

Found while triaging staves where a new checkpoint scored far below the pinned base on
OSSQ. Two checkpoints separated by a long stretch of commits agree on a key signature the
reference does not state at all. That agreement is evidence about the reference: they
share no training corpus revision, and a shared hallucination of the same accidental
count on the same staff is a far worse explanation than a segment that never carried it.

**The decisive case.** `sq8806881:0012:0001` is Haydn's String Quartet **in E-flat
major** - three flats. Both checkpoints open the staff `clef_F4 keySignature_-3`; the
segment's `<attributes>` block carries `<divisions>` and `<time>` and **no `<key>`
element whatsoever**, so the converter has nothing to emit and the staff's reference opens
straight into `timeSignature/4`. The models read the printed key signature off the page
correctly and are scored wrong for it.

**Scale.** Of the 792 staves in the OSSQ validation split, **7 open with no key signature
in the reference, and on 6 of those both checkpoints agree one is present**:
`keySignature_-3` x3, `keySignature_-1` x2, `keySignature_2` x1. Small, but it is a
systematic loss rather than noise, and it is concentrated exactly where it hurts most: an
omission in the opening attributes shifts every position in the staff.

This is consistent with an information loss the converter already documents for a
different element - `convert_ossq.py` notes 3,740 repeats present in the whole scores and
zero in the segments. The systemwise segments are lossy relative to the whole-score
MusicXML; the key signature is a second instance of the same shape, not a new surprise.

**What to do with it.** These 7 staves should not be read as recognition errors in any
per-staff analysis. Recovering the key from the whole score, the way the repeat gap was
handled, would fix them at the source; nothing here has done that. The remaining 20 of
the 25 collapsed staves are *not* explained by this - the two checkpoints disagree with
each other there, which is ordinary model error and not a corpus claim.
