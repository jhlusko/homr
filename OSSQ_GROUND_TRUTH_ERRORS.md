# OSSQ-OMR ground-truth errors: measures where parts disagree on length

Drafted for possible forwarding to the [ossq-omr](https://github.com/MALerLab/ossq-omr)
authors. Not filed anywhere - this is a local write-up of what was found and how, for
review before anyone decides whether/how to report it upstream.

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
  ground truth by a constant `1/8` whole note, and the ground truth's *total* duration
  for this measure agreed exactly with the other three parts (`3/4` each) once
  normalized for each part's own `<divisions>` value - which first read as "the
  encoding is internally consistent, so this must be a genuine HOMR decode error,"
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

Net effect: of the two cases checked against their scans, zero are confirmed HOMR
decode errors. Finding one - to actually validate the Phase 1/2 remedies in
`DECODER_RHYTHM_ACCURACY_DESIGN.md` are addressing a real model problem, not corpus
noise - is still open.

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
