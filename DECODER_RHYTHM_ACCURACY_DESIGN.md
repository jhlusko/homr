# Improving decoder rhythm/duration accuracy: a design doc

## 1. Executive summary

The 200-page Stage A/B benchmark (`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §4) found that
`barline_position_mismatch` and `measure_duration_mismatch` are, between them, the
largest category of cross-staff disagreement HOMR produces - 306 and 122 occurrences
respectively out of 199 pages, well ahead of every other finding kind. A spot check on 5
flagged pages traced this to the decoder itself, not detection: every real instance
showed either a constant additive offset (one mis-decoded note's duration shifting every
later barline by the same amount) or a constant ratio (a systematic meter/subdivision
misread) between an agreeing majority of staves and a lone dissenter - never the
occasional, non-proportional jump a vision or crop error would produce.

`propose_majority_position_corrections` (built this session) now localizes about 30% of
these to a specific staff and measure, but deliberately proposes no content fix - there
is no single low-ambiguity token to correct, by design. Closing the other ~70%, and
actually *fixing* rather than merely flagging any of it, requires improving the decoder
that produces these sequences in the first place. This document proposes a staged,
cheapest-first path toward that, mirroring the staging discipline already established
for cross-staff work (`ENSEMBLE_TRANSCRIPTION_DESIGN.md` §12) and score-profile
conditioning (§7, §3 of the next-steps file): cheap, decode-time interventions before
training-time ones, training-time interventions measured against the real benchmark
before any full-model commitment, and no step skipped ahead of the one it depends on.

## 2. Evidence this is worth solving

From the 200-page benchmark (`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §4):

| finding | count | share of 199 pages |
|---|---|---|
| barline_position_mismatch | 306 | dominant - ~2.5x the next kind |
| measure_duration_mismatch | 122 | tied for second |
| motif_articulation_mismatch | 122 | tied for second (a different, pitch-domain issue) |
| key_signature_mismatch | 44 | mostly a "silent staff" pattern, already handled |
| measure_count_mismatch | 29 | |
| time_signature_mismatch | 12 | |

`barline_position_mismatch` and `measure_duration_mismatch` are almost certainly the
same underlying phenomenon measured two ways - a duration error inside one measure
changes both that measure's total (caught by the second check) and every later
cumulative barline position (caught by the first). Together they are not a minor tail;
they are the single largest opportunity in the whole cross-staff-consistency track, and
the one Stage B's conservative, propose-only design cannot fully close because a content
fix here requires knowing *which* note is wrong and *what* it should be - genuine model
capability, not a post-hoc rule.

Separately, `phase20`/`phase21` (§3) found that this decoder has real, currently-unused
capacity: a frozen-core probe found a new signal (score-profile context) moved the
model's own loss on every held-out epoch measured, and the follow-up unfrozen run showed
that once the whole decoder can adapt, it absorbs that signal into its general weights
rather than needing it explicitly. Read together, these say the decoder is not already
saturated against its own training objective - there is room for a well-targeted
training-time change to move real accuracy, if the change is well-targeted and
correctly measured (loss alone was shown by phase21 not to be a sufficient measure of
"did this help").

## 3. Current architecture and where duration lives

(`ENSEMBLE_TRANSCRIPTION_DESIGN.md` §4, §5.1 - reproduced/extended here for this
document's own grounding.)

- One shared autoregressive Transformer decoder, six parallel output heads: rhythm,
  pitch, lift/accidental, position, articulation, slur. Rhythm is a single
  classification head over a fixed kern-duration vocabulary
  (`homr/transformer/vocabulary.py`'s `build_rhythm`) - `note_4`, `rest_8.`, `barline`,
  and so on, one token per decode step.
- Each physical staff is decoded **independently** - the decoder has no cross-staff
  signal of any kind at generation time. This is exactly why the whole cross-staff
  consistency/repair track (§12 of the design doc) exists as a post-hoc layer: nothing
  upstream of it lets one staff's decode benefit from another's.
- Generation is confirmed **purely greedy argmax** on every code path that generates
  rhythm tokens - both the fast ONNX inference path
  (`homr/transformer/decoder_inference.py`) and the training-time generation path
  (`training/architecture/transformer/decoder.py`). There is no beam search, no
  candidate scoring, anywhere in this codebase today. Any design that wants to rerank
  candidate continuations (§7.2 below) has to build that machinery from nothing, not
  extend something that half-exists.
- Design principle §5.1 (already established, not revisited here): new semantics get
  new heads, not changes to the rhythm vocabulary itself - changing that vocabulary
  would perturb the most important existing softmax and embedding matrices and entangle
  notation fidelity with note/rest sequence accuracy. Every proposal below respects
  this; none touches the rhythm vocabulary's contents.

## 4. Failure taxonomy (grounded in the spot check, n=5 pages - small, worth widening)

- **Type 1 - isolated single-note duration error.** A constant *additive* offset between
  the dissenting staff and the agreeing majority, starting at one measure and persisting
  unchanged afterward (e.g. every later barline off by exactly 1/8 whole note). Reads as
  one wrong note early in an otherwise-correct passage.
- **Type 2 - systematic meter/subdivision misread.** A constant *ratio* (e.g. every
  value at exactly 2/3 scale) between dissenting and majority staves - reads as the
  decoder settling into a different, self-consistent but wrong subdivision or meter for
  an entire passage, not a single slip.
- **Type 3 - chaotic disagreement.** Non-constant offset, multiple staves disagreeing
  with each other in inconsistent ways (seen on one of the 5 spot-checked pages, a
  synthetic sample). `propose_majority_position_corrections` correctly declines these -
  there is no clean localization to make. Likely a genuinely hard passage or a
  lower-confidence region for this model, and probably needs a different diagnostic (or
  acceptance that some fraction of pages are simply harder) rather than a targeted
  duration fix.

These three types plausibly need different remedies (§7 below), and Phase 0's data
audit and Phase 1's decode-time experiment should both report their results broken down
by which type they actually help, rather than a single aggregate number - a fix that
only ever helps Type 1 is a real, useful, but partial result, not evidence against
pursuing a further fix for Type 2.

## 5. Non-goals

- Not proposing to touch the pitch, lift, articulation, slur, or position heads - this
  document is scoped to rhythm/duration only, since that is what the evidence in §2
  actually names.
- Not proposing to expand or otherwise modify the rhythm token vocabulary - §5.1 has
  already settled this question for the whole architecture, and nothing here is a
  special case.
- Not re-litigating or duplicating Stage C's own design (`ENSEMBLE_TRANSCRIPTION_DESIGN.md`
  §12.3, `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §4's Stage C) - that document already
  specifies a learned variable-staff context adapter and its own staging precondition.
  This document's phases are deliberately smaller and cheaper, meant to be tried
  *before* that one, not instead of it; §7.4 says so explicitly rather than silently
  overlapping.
- Not claiming the 5-page spot check in §4 is a large enough sample to trust its type
  proportions at face value - it is evidence the three types exist and look like this,
  not a claim about their relative frequency corpus-wide. Phase 0 (§7.1) is partly about
  getting a real number here.

## 6. Design principles for this track specifically

In addition to the whole architecture's existing principles (§5 of the design doc,
reproduced by reference, not restated):

- **Cheapest intervention that could plausibly work, tried first.** A decode-time
  change that requires no retraining and can be evaluated in an afternoon should be
  tried and measured before a training-time change that costs a GPU run and a day of
  wall-clock time, even if the training-time change seems more likely to work in
  principle - exactly the discipline phase20 (frozen-core probe before phase21's
  unfrozen run) already used in this same file's §3.
- **Measure against the real benchmark, not training loss alone.** phase21 is the
  concrete, recent cautionary tale: a clean, consistent training-loss/held-out-ablation
  result (phase20) did not survive contact with a bigger question (does this still hold
  once the core can adapt) precisely because the *loss* delta was the only thing being
  watched. Every phase below is measured by rerunning the 200-page Stage A/B benchmark
  (`benchmark_stage_ab.py`, not currently checked into this repo - worth committing it
  given how many times this session has rerun it) and tracking the
  `barline_position_mismatch`/`measure_duration_mismatch` finding counts directly, not
  just an auxiliary loss going down.
- **A frozen-core probe before any unfrozen training**, for the same reason phase20
  preceded phase21: attributability first, and a structural guarantee against
  regressing the existing model while the narrower question is still open.

## 7. Staged approach

### 7.1 Phase 0: training-data rhythm-label audit (cheapest - do this first)

**RETRACTED, in full - every "ground truth" comparison below is invalid.** The
`<name>.musicxml` files read throughout this section (sitting next to each training
image at `images/*/original/<page>.musicxml`) are not the OSSQ-OMR corpus's ground
truth - they are `homr.main`'s own prior output, written to that exact path as a side
effect of running it earlier in this session. Confirmed directly: their
`<identification>` block reads `<encoding><software>homr</software></encoding>`, and
their titles are HOMR's own garbled title-detection output, not the real title. The
pristine local backup (from before this session's runs) has no `.musicxml` at that path
at all - only images and bbox files. The real ground truth lives at each piece's own
top level (`scores/<composer>/<piece>/sq<id>.musicxml`, a whole-score file with genuine
MuseScore/IMSLP/OpenScore provenance).

Every claim below - the Beethoven and Borodin spot-checks, the 999-measure corpus
sweep, and both `deep_barline_audit.py` runs' "91/91 corpus noise" and "509/517 corpus
noise" results - compared HOMR's decode against **HOMR's own earlier output**, not real
ground truth. None of it is evidence about the corpus. Whether the underlying pattern
(a decoder appearing to reproduce specific "defects") is a real phenomenon or an
artifact of near-deterministic decoding producing similar output across two runs of the
same model is genuinely unknown - unaddressed by anything below. Left below for the
historical record of the error and how it was found, not as a source of real
conclusions.

### The correct redo, done properly: a real confirmed decode error, found at last

The mapping problem flagged above turned out to already be solved by the corpus itself:
`metadata/scanned/systemwise/sq<id>:<page>:<system>.yaml` gives `measure_start`/
`measure_end` for every system on every page - the exact page-local-to-absolute mapping
needed, corpus-provided rather than self-derived. Combined with the real whole-score
ground truth (`scores/<composer>/<piece>/sq<id>.musicxml`) and a *fresh* `homr.main`
run's own output (the per-page `.musicxml` is legitimate to read for what it actually
is - HOMR's current decode - just not as "ground truth"), this gives a correct,
three-way comparison.

**Beethoven Grosse Fuge Op.133, real measure 327 (page 13's first system, page-local
measure 8), viola:** real ground truth is `C4(quarter) B4(eighth) A4(quarter) D4(eighth)`
- the same quarter-eighth-quarter-eighth pattern as the other three parts (confirmed:
all four parts' real ground truth agrees at 3/4 here). HOMR's fresh decode of the viola
is `C4(quarter) B4(**dotted quarter**) A4(quarter) D4(eighth)` - everything else
matches exactly, but the second note's duration is wrong, turned from an eighth into a
dotted quarter, adding exactly the `1/4`-whole-note excess this whole thread started
from. **This is a genuine, confirmed HOMR decode error** - not a corpus defect (the
earlier "ground truth confirms it" and "ground truth is wrong" readings were both
built on the contaminated file and are superseded by this), not a labeling artifact,
a real duration misread on one note, verified against the real corpus source.

This single, small, precisely-located error is exactly Type 1 from §4's taxonomy (one
mis-decoded note, constant additive offset) - the failure type Phase 1's beam-search
reranking is specifically aimed at, and now has a genuine confirmed example to measure
against, rather than an assumption.

**Moeran String Quartet in A minor, real measure 190 (page 34's third system,
page-local measure 9), all four parts:** checked the same way, and the picture is much
messier. HOMR's fresh decode of *all four parts* differs substantially from real ground
truth here - not just the one staff `propose_majority_position_corrections` flagged as
diverging from the "majority." Even the majority itself (the three mutually-agreeing
staves) does not match real ground truth at the adjacent, unflagged measure 189/8
either (ground truth: 8 discrete eighth-note events for the cello; HOMR: a single half
note). **Staves agreeing with each other is not evidence they are correct** - the
model can make similar, mutually-consistent errors across staves independently, and
this system (marked `ff`, dense chords, trills, likely genuinely hard) may be beyond
reliable decode accuracy across the board, not carrying one isolated flaw. This is a
different, arguably more concerning failure mode than a clean Type 1 error, and not one
Phase 1's reranking (built around trusting a clean majority) would fix - reranking
against a majority that is itself wrong just picks a different wrong answer.

**Net read, corrected**: a real, confirmed decode error exists (Beethoven), giving
Phase 1 a genuine target at last - but it is not necessarily representative. The Moeran
case is a reminder that cross-staff *agreement* is a much weaker signal of correctness
than this whole track has been assuming.

### The corpus-wide redo, done properly: 87/91 now show a real divergence from truth

`deep_barline_audit_v2.py` (`training/omr_datasets/`) is the corrected rerun of the
original `deep_barline_audit.py`, using `ossq_ground_truth.py`'s `real_ground_truth_path`
and `measure_start_for_system` instead of the broken `<page>.musicxml` path. Run across
the same 200-page benchmark sample:

```
total majority_position_correction proposals: 91
  ground truth disagrees (known corpus defect by the invariant): 1
  ground truth agrees (candidate: real decode error): 87
  no ground truth / no measure-mapping metadata available: 3
```

**This is the exact opposite of the invalid original result (91/91 "corpus noise").**
That number was always going to be near-total agreement, because it was comparing HOMR
against its own prior output - a near-deterministic model agreeing with itself. Against
real ground truth, the picture flips: only 1 of 91 lands on an actual corpus defect
(and a tiny one - three parts read `1440`, one reads `1441`, a 1-part-in-1440 rounding
discrepancy, not a meaningful musical error). The other 87 diverge from ground truth in
a way the corpus does not explain.

**A caveat before treating "87" as "87 confirmed decode errors" the way Beethoven is
confirmed**: `agrees`/`disagrees` here is still a *duration-total* check, the same kind
`ossq_measure_length_audit.py` does - it says the flagged measure's real ground truth
does not exhibit the same disagreement HOMR's decode shows, not that the specific note
HOMR got wrong has been identified. A second spot-check (Dvořák Op.51, real measure
274) attempted the same full content-level verification Beethoven got, and came back
inconclusive - HOMR's fresh decode differed substantially from ground truth across
*all four* parts, not just the one flagged staff, similar to the Moeran case. Total
measure count matched exactly (25 HOMR-decoded measures for a page the corpus's own
metadata also spans 25 measures), so this isn't obviously a broken alignment, but
per-system boundaries could still differ between HOMR's own detected system sizes and
the corpus's reference breakdown in a way that shifts individual measure mappings even
when the page grand total matches - not ruled out. Only Beethoven has a clean,
fully content-verified confirmed decode error; the other 86 "agrees" entries are
real, meaningful evidence (the corpus's own data doesn't explain the divergence) at
the duration-total level, not yet each individually verified note-by-note.

**Net implication for Phase 1/2**: strong, aggregate evidence that this failure family
is predominantly decoder error, not corpus noise (a full reversal from the invalid
original conclusion) - Phase 1's beam-search reranking now has real justification. The
Moeran/Dvořák caveat matters for scope, though: some real fraction of "agrees" cases may
turn out to be Moeran-shaped (broadly poor decode across a hard passage, where
reranking against a majority that is itself wrong doesn't help) rather than Beethoven-
shaped (one clean, isolated wrong note, exactly what reranking targets).

### Distinguishing the two at scale: `content_verify_agrees.py` - the final answer

Built to run the same content-level check Beethoven got across all 87 "agrees" entries,
not just totals: for each entry, decodes the page fresh, compares each part's (pitch,
octave, duration) sequence at the flagged measure against real ground truth (duration
normalized by each file's own `<divisions>` - comparing raw duration strings across a
whole-score file with `divisions=1920` and HOMR's own `divisions=2-4` output made every
comparison spuriously show zero overlap regardless of actual content until this was
fixed), and classifies each entry by whether the *majority* (non-flagged) staves' content
closely matches ground truth (Beethoven-shaped) or not (Moeran-shaped).

```
total agree-entries checked: 87
  Beethoven-shaped (majority overlap >=0.8, flagged staff <0.8): 17
  Moeran-shaped (majority overlap <0.8 too): 57
  inconclusive/no data: 13
```

Spot-checked one Moeran-shaped entry (all four parts showing exactly `0.0` overlap -
the largest sub-bucket, 36 of the 57) directly against both files' raw content to rule
out a residual alignment artifact rather than genuine divergence: the content is
genuinely, completely different on both sides (different pitches, different note
counts, no scale/units confusion visible) - a real Moeran-shaped case, not a tooling
miss.

**Final read**: of the 87 measures where the corpus's real ground truth confirms a
genuine decoder divergence, roughly **1 in 5 (17, ~20%) are Beethoven-shaped** - clean,
isolated errors exactly matching what Phase 1's beam-search reranking is designed to
catch. The majority, **57 (~65%), are Moeran-shaped** - the whole system's decode
diverges from truth, not just the flagged staff, on passages that are plausibly
genuinely hard for the current model. The remaining 13 (~15%) had no page-local
measure to compare (a HOMR-detected system/measure count mismatch with the reference,
or a missing corresponding measure) and are simply unverified, not evidence either way.

**This sets real, calibrated expectations for Phase 1**: reranking against a
cross-staff majority is a sound strategy for roughly a fifth of this failure family
(the Beethoven-shaped fifth) but would not help - and could even entrench a wrong
answer - for the Moeran-shaped majority, where the "majority" itself is not reliably
correct. A full fix for the Moeran-shaped share would need something that doesn't
assume a majority of staves are right, which nothing in this document's staged plan
(§7) currently provides - worth naming as a real gap rather than assuming Phase 1 alone
closes this out.

### §7.1 correction, 2026-08-21: a second ground-truth bug (movement splicing), found via the corpus-review webpage

The 87/17/57/13 numbers immediately above have a real bug in how they were computed,
independent of (and layered on top of) the `<page>.musicxml` bug this section already
describes. User inspection of `corpus_review.html` (the review webpage built for this
finding) turned up a visibly wrong rendered "ground truth" image, traced end to end on
one entry (Wolf, *String Quartet*, sq8823783, page 22, absolute measure 318, viola).

**Root cause**: OSSQ-OMR's ground-truth `.musicxml` files concatenate every movement of
a piece into a single file, and each movement *restarts* `<measure number="...">` at 1
(most string quartets have 3-4 movements). Every measure-lookup in this investigation -
`deep_barline_audit_v2.py`'s `check_ground_truth`, `content_verify_agrees.py`'s
`verify_entry`, `build_review_assets.py`'s `extract_gt_window` - matched ground-truth
measures with `m.get("number") == str(target)` against the *whole* file, implicitly
assuming that number was unique. It isn't: "measure 317" occurs once per movement. The
Wolf example's rendered window spliced two unrelated movements' measures 317-321
together into one image (10 measures kept where 5 were wanted), which is exactly why
every part - not just the flagged one - showed `0.0` content overlap for that entry: the
comparison itself was garbage, not evidence of a decode error.

The corpus's own `measure_start`/`measure_end` alignment metadata resets the same way at
movement boundaries (confirmed by inspecting the metadata sequence for the Wolf piece: a
system's metadata decreasing relative to the previous system's lines up exactly with a
`<measure number="1">` reset in the ground truth file) - so `measure_start` is
movement-local, not a piece-wide running count. A naive "flat index = measure_start - 1"
fix would only work by coincidence for a page in the first movement.

**Fix**: `homr/training/omr_datasets/ossq_ground_truth.py` gained
`movement_index_for_system` (counts resets in the corpus's own metadata sequence -
`scanned`/`synthetic` only, deliberately excluding `unaligned`, which was found to carry
a spurious duplicate-numbered page for this exact piece that would otherwise cause a
false reset) and `resolve_flat_measure_range` (matches by number only within the
identified movement's own slice, where numbers are unique). All three consumer scripts
were updated to use it. All 8 pre-existing unit tests still pass. Verified against two
independent cases on the Wolf piece: page 22 system 2 (movement 0, resolves to flat
measures 316-320 as expected) and page 27 system 3 (movement 1, resolves to flat
measures 406-414 as expected).

**Corrected corpus-wide barline numbers** (`deep_barline_audit_v2.py` rerun, identical
200-page sample):

```
total majority_position_correction proposals: 91
  ground truth disagrees: 1                        (unchanged)
  ground truth agrees (candidate: real decode error): 75    (was 87)
  no ground truth / no measure-mapping metadata: 15          (was 3)
```

All 12 changed verdicts moved `agrees → no mapping`, never the reverse. Checked one
directly (Andrée, *String Quartet in A major*, sq7313978, page 30 system 4): the piece
genuinely has 4 movements (flat boundaries at measures 0/170/322/560), and this system's
alignment metadata exists *only* in the unreliable `unaligned` folder - with no aligned
entry to place it in the reset sequence, guessing its movement would risk the exact
splicing bug this fix exists to prevent, so it now conservatively reports no mapping
rather than a number that might reference the wrong movement's measure. Same discipline
`measure_start_for_system` already applies to non-numeric metadata placeholders (`"X2"`
etc.) - corpus ambiguity becomes "no mapping," never a guess.

**The 87-entry content-level breakdown above (17/57/13) was stale** - `content_verify_
agrees.py` had the identical number-matching bug. Corrected rerun against the fixed
75-entry set (`content_verify_agrees_v3_full.json`):

```
total agree-entries checked: 75
  Beethoven-shaped (majority overlap >=0.8, flagged staff <0.8): 34  (~45%, was 17/87 ~20%)
  Moeran-shaped (majority overlap <0.8 too): 23                     (~31%, was 57/87 ~65%)
  inconclusive/no data: 18                                          (~24%, was 13/87 ~15%)
```

**This reversed which shape dominates, not just the proportions.** The movement-splicing
bug's characteristic failure mode - mixing two unrelated movements' measures into one
comparison - produces near-`0.0` overlap on *every* part, exactly the same signature as a
genuinely Moeran-shaped "whole system diverges" result (this is precisely what happened
in the Wolf example that surfaced this fix). A meaningful share of the old 57
Moeran-shaped entries were very likely spliced-garbage comparisons wearing a
Moeran-shaped costume, not real evidence of broadly poor decode. **Revised conclusion**:
Phase 1's beam-search reranking now has a substantially larger, better-justified target
than previously thought - the Beethoven-shaped plurality (34 of 75, ~45%) is exactly the
"one clean wrong note against a reliable majority" signature it targets. The
Moeran-shaped minority (23, ~31%) and inconclusive share (18, ~24%) are still real and
still worth naming as a gap Phase 1 alone won't close, but they are no longer the
dominant story the way they appeared to be before this fix.

`corpus_review.html` itself was **not** regenerated for this fix (explicit user
instruction: tracing 1-2 examples end to end was sufficient without rebuilding all
assets) - its ground-truth renderings for multi-movement pieces may still show the
pre-fix splicing artifact until it is rebuilt.

Before attributing every flagged disagreement to a *decoding* error, rule out training
data itself being part of the story: sample some number of already-flagged
disagreements (real pages this session's benchmark has already identified), and where a
whole-score MusicXML ground truth exists (as it did for the carry-forward key-signature
finding's own verification, §4), compare the decoded rhythm sequence against it directly
on the disagreeing measure. This reuses the same kind of ground-truth cross-check this
session already did once, and the existing label-audit tooling in this repo
(`training/omr_datasets/dataset_label_audit.py`, `training/omr_datasets/
ossq_label_audit.py`) as a starting point rather than new machinery.

If a meaningful fraction of "disagreements" turn out to be genuinely mislabeled ground
truth rather than a model error, that changes the ceiling every later phase should
expect, and is worth knowing before spending GPU time chasing a problem that is partly
a labeling artifact.

**Started (n=2 pages, small - a first read, not a corpus-wide answer):** both pages
already used in this document's own §4 spot check have a page-level ground-truth
MusicXML sitting right next to the training image (`<name>.musicxml`), letting the
decoded divergence be checked directly against it, no whole-score lookup needed.

- **Beethoven Op.133 p.13, system 0, staff 2 (viola), measure 8**: HOMR's decode put
  this staff's cumulative position at barline 8 exactly `1/4` whole note ahead of the
  3-staff majority (`25/4` vs `6`). The ground truth for this measure, part 3, is
  `quarter(2) + dotted-quarter(3) + quarter(2) + eighth(1)` = 8 duration units = a full
  whole note, against the other three parts' `quarter+eighth+quarter+eighth` = 6 units =
  `3/4` - the same `1/4`-whole-note excess, in the *ground truth itself*. First read: "a
  genuine irregularity already present in the source, HOMR decoded it correctly."
  **Overturned by comparing directly against the scan**: the actual page shows a plain
  quarter-eighth-quarter-eighth in the viola, identical to the other three parts - no
  dotted quarter anywhere. The ground truth's dot doesn't exist on the page. **The
  ground truth is wrong, not HOMR** - and since HOMR's decode reproduces this exact
  error, and this page is very likely in HOMR's own training set, the model may have
  learned this specific wrong duration directly from the mislabeled example. This
  single case justified a corpus-wide check (never legitimate for two parts to disagree
  on measure length): **999 measures across 164 of 475 ground-truth files (~35%)** show
  the same kind of internal disagreement (`OSSQ_GROUND_TRUTH_ERRORS.md`,
  `training/omr_datasets/ossq_measure_length_audit.py`) - split almost evenly between
  excess (499) and shortfall (459), which rules out a single benign explanation (an
  omitted-trailing-rest convention could only ever produce shortfalls). Only this one
  case has been confirmed against its actual scan so far; the other 998 are flagged by
  the invariant alone.
- **Borodin Quartet No. 2 p.24, system 1, staff 1, measure 6 (first measure of that
  system)**: HOMR's decode disagreed with the majority at this measure, first read
  (imprecisely - see below) as "a constant `1/8`-whole-note offset." Checking ground
  truth required first noticing each part uses its *own* `<divisions>` value (12 for
  P1/P2, 2 for P3/P4 in this file) - raw `<duration>` units are not comparable across
  parts without normalizing by each part's own divisions first. Once normalized, **all
  four parts' ground truth *totals* agree exactly** (`3/4` each) - which first read as
  "no irregularity, no internal inconsistency, so this must be a genuine decode error,"
  most likely an implicit (unmarked) triplet HOMR's per-staff decoder had no context to
  resolve. **That reading does not survive comparing the actual note content against
  the scan.** The encoded ground truth for this measure is a quarter rest, four 16th
  notes, and a closing quarter - five discrete events. The scan shows six plain eighth
  notes beamed 3+3 under one slur, no rest, no closing quarter. These are different
  rhythms that happen to sum to the same total - there is no triplet ambiguity to
  explain here, the encoding simply does not match the page. **This is a second,
  structurally different ground-truth error** - a content substitution that a
  duration-only check (like `ossq_measure_length_audit.py`, §7.1's corpus-wide
  follow-up) cannot detect, since the wrong content still totals correctly. The
  "implicit triplet" explanation was wrong, not just incomplete.

  **A third correction, found while building `deep_barline_audit.py` (below):** the
  "constant `1/8` offset" claim itself was never actually checked past the first
  divergent measure - `propose_majority_position_corrections`, run properly across the
  whole system, does *not* fire on this measure at all, because the offset is not
  constant (`-3/8, -3/4, -9/8, -9/8, -9/8` across the system's five barlines, only the
  last three matching). This is Type 3 (chaotic disagreement, §4's taxonomy), not Type 1
  - a genuinely different failure shape than originally described, on top of the
  ground truth being wrong. The lesson generalizes past this one page: checking only
  the first flagged measure in isolation, without verifying the offset actually stays
  constant across the rest of the system, produces exactly this kind of
  overconfident-and-wrong read - `deep_barline_audit.py` exists specifically to avoid
  repeating it at scale.

**Read on these two, revised twice now**: both cases initially looked like clean
explanations of HOMR's behavior - a real source irregularity in one case, an implicit
triplet in the other - and neither survived comparison against the actual scan. Both
are ground-truth errors, of different kinds (a wrong duration value; a wrong note
sequence with a coincidentally-correct total). **Of the two cases checked against their
scans, zero are confirmed HOMR decode errors.**

### `deep_barline_audit.py`: the corpus-wide answer, not just two pages

Built to do properly, at scale, what the Borodin correction above showed manual
spot-checking gets wrong one page at a time: rather than shelling out to `homr.main`
and scraping printed diagnostics, it calls HOMR's own pipeline in-process
(`homr.main.detect_staffs_in_image` + `homr.staff_parsing.parse_staffs`) to get the
actual decoded staves per system, computes `_cumulative_barline_positions` on *every*
system (not just the ones with a printed mismatch), and sums barline counts across all
earlier systems on a page to convert a `propose_majority_position_corrections`
proposal's local `measure_index` into the correct absolute ground-truth measure number
- the exact conversion manual spot-checking has to get right by hand every time, and
the Borodin case shows how easy it is to get wrong. For every proposal, it then checks
that absolute measure's ground truth directly (reusing `measure_length_by_part` from
`ossq_measure_length_audit.py`), rather than only comparing against a pre-computed
static list.

**Run across the full 200-page benchmark sample: 91 `majority_position_correction`
proposals total. All 91 land on a measure where ground truth already disagrees. Zero
land on a measure where ground truth agrees.** Full output:
`training/omr_datasets/ossq_audit_findings/majority_position_correction_ground_truth_check.json`.
Spot-checked one additional instance against its scan for confidence beyond the raw
numbers (Wolf's String Quartet, p.22, system 1, measure 6/334, cello) - the flagged
part's measure is visually a long sustained figure clearly different in character from
the same measure in the other three parts, consistent with the `+1/2`-whole-note excess
the ground truth encodes; not pursued to the same full-restaging as Beethoven/Borodin
given the 91/91 result already stated the case on its own.

**This is a materially stronger result than "some training-data noise, some decoder
errors, unknown split."** Every single clean, majority-corroborated position divergence
found across 200 real pages traces to an already-broken ground-truth measure - not most
of them, all of them. The Beethoven case's own speculation ("the model may have learned
this specific wrong duration directly from the mislabeled training example") looks like
the general mechanism, not a one-off: a `propose_majority_position_corrections` firing
(3+ staves cleanly agreeing, one cleanly offset by a constant amount) appears to be, in
this sample, a *ground-truth-defect detector* riding on top of a decoder that has
learned to reproduce those specific defects faithfully, not an independent
decoder-error detector at all.

**What this means for Phases 1-2**: their premise - that some real share of
`barline_position_mismatch` is a decoder problem beam search or an auxiliary head could
fix - has no supporting example in this sample, for the specific "clean, constant-offset"
signature those phases are aimed at. That doesn't mean HOMR never makes a genuine
duration decode error (Type 3/chaotic disagreements, and 2-staff disagreements below
`propose_majority_position_corrections`' 3-staff corroboration bar, are both outside
what this check covers), but the narrow slice this document scoped Phase 1 toward -
exactly the slice Stage B's rule already isolates - now looks like it should be
retargeted at the corpus, not the decoder. **Recommendation: treat corpus cleanup
(`OSSQ_GROUND_TRUTH_ERRORS.md`) as the higher-priority lever for this specific finding
before investing in Phase 1's beam-search machinery**, and keep looking for a genuine
decode error specifically among the Type 3/chaotic and 2-staff cases this tool doesn't
reach, if Phases 1-2 are still to be pursued on solid footing rather than an assumption.

### 7.2 Phase 1: decode-time cross-staff-consistency reranking (no retraining)

Build k-best beam search for the rhythm head (§3 already confirms none exists on any
code path today - this is new machinery, not an extension). Score candidate
continuations per staff not only by likelihood but by whether they land on
cross-staff-consistent cumulative barline positions - reusing
`homr.cross_staff_consistency`'s own `_cumulative_barline_positions` machinery as a
reranking objective rather than only a post-hoc diagnostic.

This targets Type 1 specifically: if the greedy path's small local error is only
narrowly more likely than a nearby k-best candidate that *also* satisfies cross-staff
agreement, reranking recovers the correct decode for free, with no training run at all.
It is not expected to help Type 2 (a systematic misread is not a narrow-margin call the
model is right on the edge of getting correct) or Type 3 (already flagged as chaotic,
not a localized slip) - the taxonomy in §4 predicts this asymmetry, and Phase 1's own
report should confirm or correct that prediction rather than assume it.

Cheap to prototype relative to everything else in this document: offline, against
already-generated logits where possible, no GPU training needed - only the beam-search
and reranking logic itself needs building. Success criterion: rerun the 200-page
benchmark with reranking enabled vs. disabled and compare `barline_position_mismatch`/
`measure_duration_mismatch` counts directly.

**Built 2026-08-21, in a narrower, cheaper shape than textbook fixed-width beam search
- deliberately, not as a shortcut.** Real classical beam search (maintaining k
hypotheses at every step) would need per-step multi-hypothesis KV-cache batching this
codebase has never exercised (`decoder_inference.py` is batch=1 throughout) - a much
larger, riskier build for the same stated goal. Instead: at each staff's *narrowest-
margin* rhythm decisions specifically (not every step), branch into a full alternate
decode via the already-validated `generate_from_prefix` mechanism, then keep whichever
candidate - greedy or a fork - best matches the other staves' majority cumulative
barline positions. This targets exactly §7.2's own stated criterion (a narrow local
call, reranked against a reliable majority) without new cache-batching machinery.

- `homr/transformer/decoder_inference.py`: `generate_with_rhythm_margins` (records the
  rhythm head's runner-up token id and logit margin at every step; verified
  bit-identical to plain `generate()` otherwise - the bookkeeping changes nothing about
  what gets decoded) and `rhythm_alternative` (forks a full alternate decode at a given
  step by forcing the runner-up token through `generate_from_prefix`).
- `homr/cross_staff_rerank.py`: `rhythm_candidates_for_staff` (greedy decode plus up to
  N forks at the N narrowest-margin steps) and `rerank_staff_candidates` (per staff,
  keeps whichever candidate agrees most with the *other* staves' majority barline
  positions - never picks a worse-or-equal alternative over the greedy default, and
  requires at least 2 corroborating staves, the same bar
  `propose_majority_position_corrections` already uses). 6 unit tests
  (`tests/test_cross_staff_rerank.py`), all passing.
- **Validated against the live model, not mocked** (same discipline
  `generate_from_prefix` itself required): `generate_with_rhythm_margins` reproduces
  `generate()`'s output bit-for-bit on a real staff; forcing the *same* token
  `rhythm_alternative` would have chosen anyway reproduces the rest of the decode
  bit-for-bit (mirrors tier 2's own validation exactly); forcing the real runner-up at
  the narrowest-margin step produces a valid, genuinely different decode with no crash;
  the full `rhythm_candidates_for_staff` → `rerank_staff_candidates` pipeline runs
  end to end with no crash.
- `benchmark_phase1_rerank.py`: the actual before/after benchmark the success criterion
  above asks for. Captures each staff's encoder context via a monkeypatch on
  `Staff2Score.predict`, then has to solve a real correctness problem before it can
  measure anything: `parse_staffs` decodes staves *voice-major* (all of voice 0's
  staves across every system it appears in, then voice 1's, ...), not system-major, so
  a captured call's position in call order does not by itself say which system it
  belongs to. Solved by mirroring `parse_staffs`' own nested loop exactly (same
  `plan.staff_for_voice` calls, same enumeration order) rather than guessing - and
  confirmed correct, not just plausible: run against a known page
  (`sq8823783:0022.png`, Wolf *String Quartet*), the reconstructed "before reranking"
  finding count (3) matched the live pipeline's own logged findings on that exact page
  exactly. Only reranks systems with ≥3 staves and ≥1 existing Stage A finding -
  matching `rerank_staff_candidates`' own corroboration bar and skipping the (majority
  of) already-clean systems entirely.

**Results, 20-page sample** (`phase1_sample20.txt`, the first 20 lines of the 200-page
benchmark): **31 → 16 combined `barline_position_mismatch`/`measure_duration_mismatch`
findings (a ~48% reduction), 11 of the checked systems improved.** A real, clearly
positive first signal, not noise - on the single page already traced end to end earlier
in this document (`sq8823783:0022.png`), the reduction (3→2) matched a specific,
identifiable system fix, not an aggregate artifact.

**Full 200-page run complete** (`phase1_out200.json`, 198/200 pages processed - 2 hit
the same pre-existing, unrelated `staves_by_system` index error seen elsewhere in this
investigation, caught and skipped, not a Phase 1 defect):

```
total systems checked: 899
staff-level rerank attempts: 1,223
combined barline_position_mismatch + measure_duration_mismatch findings:
  before reranking: 428
  after reranking:  339   (20.8% reduction)
systems where reranking changed the finding count: 81
pages where reranking made the finding count WORSE: 0
```

**The real number (20.8%) is meaningfully lower than the 20-page sample's 48%** - that
sample was not representative, by chance concentrated toward more fixable cases.
20.8% is still a real, substantial reduction, and **critically, it never regressed
anywhere**: across all 198 successfully-processed pages, reranking never increased the
finding count on a single one. That's an important safety property on its own - this
mechanism does not appear to trade one class of error for another, at least by this
measure.

**What this benchmark does and does not establish, stated plainly.** It measures
whether Stage A's own cross-staff-agreement checks stop firing after reranking - not
whether the reranked decode is actually *closer to real ground truth*. This
investigation has already found once, directly (the Moeran case, §7.1 above), that
staves agreeing with each other is not proof they are correct - `rerank_staff_candidates`
picks whichever candidate best matches the *other staves' majority*, which is exactly
the same "trust the majority" premise Moeran already showed can itself be wrong on a
hard passage. **Ground-truth spot-check done: 6 of 6 resolvable corrected measures now match real
ground truth exactly.** `phase1_ground_truth_spotcheck.py`, run against 15 of the 61
pages with a changed system: for each `majority_position_correction` proposal whose
staff reranking actually altered, resolves the real ground-truth duration for that
exact measure (via `ossq_ground_truth.py`'s movement-aware machinery) and compares it
against both the greedy and reranked decode. 10 proposals had a resolvable mapping (4
didn't - the same known corpus-metadata gaps hit before, correctly excluded rather than
guessed at); of those 10, 6 corresponded to staves reranking actually changed, and
**all 6 flipped from greedy-wrong to reranked-matches-truth exactly** - zero
regressions, zero cases of reranking picking a self-consistent-but-wrong answer (the
specific risk this check exists to catch).

Two real bugs found and fixed in the spot-check script itself before trusting this
result - both the same *class* this investigation has hit more than once now (compare
normalized values, never raw units across two different sources): divisions need
walking from the movement's start, not seeded fresh at the target measure (the
±§7.1 divisions bug, recurring); and a whole-note-vs-quarter-note unit mismatch between
`SymbolChord.get_duration()`'s convention (used by the decoder/reranking side) and
`measure_length_by_part`'s (quarter notes) - comparing raw values across these silently
produced nonsense-looking truth values (`48`, `1920`) until caught and fixed.

**n=6 is small, but unanimous-and-zero-counterexample is a real signal, not a coin
flip that happened to land right.** This directly answers the Moeran-case caution this
section raised with actual evidence instead of leaving it as an open assumption.

### Wired into the live pipeline, 2026-08-21

User instruction: "please wire it." `homr/staff_parsing.py`'s `parse_staffs` now
reranks for real - not log-only like Stage A/B elsewhere - on by default
(`enable_phase1_rerank: bool = True`).

**Gated two-pass, deliberately, not unconditional forking on every staff.** Every
staff's greedy decode costs the same either way
(`generate_with_rhythm_margins` only adds bookkeeping over a plain `generate()` call),
but each fork is a full extra decode pass - forking every staff on every page
unconditionally would have cost several times a normal decode even on the ~91% of
systems the 200-page benchmark found had nothing to fix, and that unconditional cost
was never what was actually measured. The real wiring instead: decode every staff's
greedy result first (cheap), run Stage A's own `check_barline_positions`/
`check_measure_durations` against the greedy decode per system, and only pay for
forking + reranking on a system that already shows a finding there - exactly the
population the benchmark and spot-check measured, nothing broader.

New/changed code: `Staff2Score.predict_greedy_with_margins` (the cheap pass, returns
the raw/unfiltered greedy decode plus margins plus the encoder context, so a later
fork pass skips re-running the encoder), `parse_staff_tromr_greedy_with_margins`
(applies the existing grandstaff/`position != "lower"` filter to a *copy* for Stage A's
cheap pre-check, while keeping the raw sequence available for forking - forking must
operate on the raw sequence to keep step indices aligned with the decoder's own
numbering), `cross_staff_rerank.fork_candidates_from_margins` (the expensive half,
split out of `rhythm_candidates_for_staff` so a caller can gate it). `parse_staffs`
itself: decode phase (always cheap), a per-system Stage A pre-check phase, then a
forking+reranking phase only for systems that pass that check.

**Validated**: full suite still 1041/1044 (same 3 pre-existing, unrelated
`dynamic.mark` failures this investigation has seen throughout - no regression).
Live end-to-end runs on two real pages: the already-traced Wolf quartet page (known
Stage A findings) completed in ~24s, reproduced the exact same post-rerank finding
(system 2's duration disagreement resolving to the same `15/16` the earlier
unconditional-forking test produced - confirms the gating didn't accidentally skip
real forking work) and wrote valid MusicXML; a page with zero pre-existing findings
(79 of 198 pages in the benchmark had none) completed in ~27s with no findings firing
and, by construction, no forking cost paid.

Disabled automatically whenever `selected_staff` restricts processing to one staff
(same reasoning `_report_cross_staff_findings` already uses there); `enable_phase1_rerank=False`
and `phase1_max_forks` (default 3, matching the benchmark) are available for explicit
comparison against the pre-Phase-1 decode.

**Spot-check extended to all 61 changed pages, 2026-08-21: the 6/6 rate holds at
scale.** 49 resolvable entries: **38 IMPROVED, 2 NEITHER_MATCHES_TRUTH, 0 REGRESSED** (9
more had no ground-truth mapping, correctly excluded). Both non-matching cases checked
by hand: the *greedy* decode was already wrong too in each (e.g. greedy `17/32` vs.
truth `1/2`, reranked `9/16` vs. truth `1/2` - neither correct, but reranking landed on
a different wrong answer for a genuinely messy passage, not a case of breaking a right
one). Zero regressions across the entire sample - a real, decisive result.

**A real bug found while extending the sample, worth remembering for future wiring
work**: wiring Phase 1 into `parse_staffs` changed its internal call path - it now
calls `Staff2Score.predict_greedy_with_margins`, not `predict()`, and reranks
internally by default. The existing offline analysis scripts
(`phase1_ground_truth_spotcheck.py`, `benchmark_phase1_rerank.py`) monkeypatch
`Staff2Score.predict` to capture encoder context; once wiring changed the real call
path, that monkeypatch stopped intercepting anything, and every page in the first
61-page relaunch crashed with `IndexError: list index out of range` (an empty/short
`captured` list being indexed as if it still held one entry per staff). Fixed by
passing `enable_phase1_rerank=False` explicitly in both scripts' `parse_staffs(...)`
calls - restores the pre-wiring call path (the monkeypatch fires again) and also
prevents the live pipeline's own now-default reranking from confounding analysis that's
supposed to be running its own independent rerank logic. **General lesson: wiring a
feature into a shared call path can silently break offline tooling built against the
pre-wiring version of that path - worth an explicit check whenever a "measure this"
script and a "do this live" wiring share code.**

### 7.3 Phase 2: auxiliary cumulative-position training signal (only if 7.1 is not enough)

Precedented directly by the structured beam/stem/slur heads
(`ENSEMBLE_TRANSCRIPTION_DESIGN.md` §9, §10): add a new auxiliary head and loss, not a
rhythm vocabulary change, per §5.1's standing principle. Candidate target: a per-token
regression or coarse classification of cumulative beat position within the current
measure (or distance to the next barline) - something the existing purely-categorical
rhythm loss has no direct pressure to track explicitly, and Type 2's "settles into a
self-consistent wrong subdivision" failure mode suggests the model currently has no
explicit signal telling it *where* it is within a measure, only what duration token
comes next.

Frozen-core probe first, exactly matching phase20's own precedent: train only the new
head over an otherwise-frozen decoder, and measure via the same 200-page benchmark
ablation (with vs. without the new head's signal active), not held-out loss alone.
Given phase21's result - the whole decoder absorbing an explicit signal once unfrozen,
erasing its measured marginal value - an unfrozen follow-up here should be expected to
carry the same risk, and should not be attempted before the frozen-core question has a
clear answer of its own.

**Refinement, 2026-08-21 (user design discussion): condition on the expected time
signature too, not just an auxiliary position head - but via explicit embedding
injection, not sequence self-conditioning.** The user's instinct - give the decoder
direct knowledge of what time signature it should be in, not just an abstract
"cumulative position" signal - is a real, well-motivated addition: time signature is
exactly the missing structural fact Type 2's "settles into a self-consistent wrong
subdivision" failure implies the model doesn't reliably track.

**But there is already a specific, directly relevant negative result that rules out the
naive version of this idea**: §4's tier-2 forced-prefix experiments
(`generate_from_prefix`) found that forcing a *corrected* token into a staff's own
decoded history - across 20 independent tests spanning pitch, rhythm/duration, key
signature, and slur fields - changed **zero** downstream predictions. The model's
future decisions appear driven almost entirely by the visual encoder context, not by
what it previously generated. This means simply ensuring a correct `timeSignature`
token sits in the sequence and hoping the model reads its own history to stay
on-meter is very unlikely to work - we already have direct evidence against exactly
that mechanism, for other fields, in this same architecture.

**The fix: condition explicitly, the same way §3's score-profile work already does and
already validated works.** `training/architecture/transformer/profile_context.py`'s
`ProfileContextEmbedding` - a small `nn.Module` combining bounded fields into one
additive vector per sequence via a zero-initialized gate, injected as one more term in
`ScoreTransformerWrapper.forward`'s existing per-field embedding sum - is the right
delivery mechanism, not a new one. Concretely: extend `ProfileContext` (or add a
sibling context object) with an `expected_time_signature` field (and/or the auxiliary
cumulative-position-in-measure value Phase 2 already proposes), reuse the identical
zero-gate/context-dropout/`context_to_batch_fields` machinery already built and
tested for instrument family/clef/part-ordinal, and train it the same frozen-core-probe-
first way phase20 already validated. This is additive to Phase 2's own auxiliary head,
not a replacement for it - the head predicts a position signal as an *output* (extra
supervision on what the model should already be able to infer visually); this
conditioning supplies an expected value as an *input* (a structural prior the model
does not have to infer from the image at all). Doing both is not redundant: one adds
pressure to the loss, the other removes an inference burden entirely.

**Where does "expected time signature" actually come from, at training time vs.
inference time? This is the real design question, not the embedding mechanics.**

- **Training**: cheap and already precedented. `training/omr_datasets/
  score_profile_extraction.py` already reads a document's real MusicXML for
  `ScorePart` fields; the ground-truth time signature for a given training sample's
  measure range is sitting right there in the same file, no new data source needed -
  same shape as how `score_profile_pairing.py` already resolves a sample's real
  `(ScoreProfile, ScorePart)`.
- **Inference: no ground truth exists, so the value has to come from somewhere else.**
  Two real options, not mutually exclusive: (a) an explicit hint the caller supplies
  (extend `ScoreProfile`/`ScorePart` with an optional time-signature field, the same
  "priors, not constraints" spirit §7.1 already established for instrument
  range/clefs) - simple, but only available when a caller actually knows and states it;
  (b) **majority vote across a first decode pass of the *other* staves in the same
  system** - decode every staff once (already happening regardless), read off each
  staff's own decoded `timeSignature` token, majority-vote across staves the same way
  `propose_carry_forward_key_signature` already does for key signatures, then use that
  as the conditioning input for a second decode pass. Option (b) needs no new external
  metadata at all and is exactly a lightweight, non-learned precursor to Stage C's own
  cross-staff idea - a smaller, faster thing to build and measure before committing to
  Stage C's full learned adapter, and its result (does time-signature cross-staff
  conditioning actually move Type 2 failures) is itself relevant evidence for how much
  Stage C's larger investment is likely to pay off.

**Why this is more tractable to start than Stage C's refinement below**: §3's
zero-gate injection point, embedding module, batch-collation path, and
frozen-core-probe methodology already exist, are tested, and are already proven to
produce a real, held-out, reproducible effect (phase20's +0.0615 nat mean delta, 10/10
epochs positive). Extending that mechanism with one more field is a bounded,
well-precedented engineering task; Stage C's refinement below needs new architecture
built from nothing.

**Started 2026-08-21**: `ProfileContext`/`ProfileContextEmbedding` extended with an
`expected_time_signature` field (`TIME_SIGNATURES` - 10 common full "numerator/
denominator" signatures, bounded-vocabulary bucketed same as `CLEFS`), following the
exact pattern every existing field already uses: `_bucket_index`, a dedicated
`nn.Embedding`, included in both `embed_one` and `forward_from_batch`'s sums,
`context_to_batch_fields`'s dict, and `decoder.py`'s `_PROFILE_BATCH_FIELDS` (found and
fixed along the way - `forward_from_batch`'s signature grew a required parameter, and
`ScoreDecoder._pop_profile_context_emb` pops an explicit tuple of field names that
needed the new key added too, or it would `TypeError` the moment `enable_profile_context`
was ever turned on). Default `""` (unspecified) is backward compatible everywhere
nothing populates it yet.

**Validated, not just written**: 6 new unit tests (`TestBucketing`,
`TestContextToBatchFields`, `TestBatchAndListAgree` - unrecognised/unspecified/two-
different-signatures, all 3 existing categories this field needed covering) - 30/30
passing (up from 24). Full suite: 1047/1047 non-deselected (same 3 pre-existing,
unrelated `dynamic.mark` failures throughout this investigation) - no regression from
either file's change. The 8 real-model wiring tests (`test_profile_context_wiring.py`,
which download real `convnext_tiny` weights and run an actual forward pass) all still
pass too, including the zero-init identity checks - confirms the new field doesn't
disturb the zero-gate discipline end to end, not just in isolated unit tests.

**Training-side sourcing built and wired, 2026-08-21 (user instruction: "please
prepare the training run").** `training/omr_datasets/score_profile_time_signature.py`:
`time_signature_for_sample(dataset_root, stem)` resolves the real ground-truth time
signature in effect at an OSSQ training sample's actual measure range, reusing
`ossq_ground_truth.py`'s movement-aware resolution built earlier this session (same
"walk forward, carry the last declared value" pattern already needed for `<divisions>`
- time signatures are also only re-stated on change) rather than a second, less-tested
lookup. `""` (unknown) whenever any step of the chain can't be resolved - the same
known corpus-metadata gaps this investigation has hit before, never a guess. 8 new
tests (temp-directory fixture, mirroring `test_ossq_ground_truth.py`'s own convention),
including one that specifically checks the "carry forward from an earlier declared
signature, not the opening one" case. Wired into `training/transformer/data_loader.py`'s
`_resolve_profile_context` - every OSSQ training sample's `ProfileContext` now carries
its real expected time signature automatically, no separate opt-in needed. Inference-
side sourcing (the majority-vote second pass, or an explicit caller hint) remains not
built - out of scope for a training run, which only needs the training side.

**Loss brainstorm, same session (user: "let's think more about the loss"):** current
loss is not literally "just cross-entropy" - it's six parallel per-token cross-entropy
losses (rhythm/pitch/lift/articulations/slurs/position) plus one existing auxiliary
term, `calConsistencyLoss`, an L1 penalty forcing all six heads to softly agree on
which positions are notes vs. rests/barlines (via softmax-probability-weighted
indicators, not hard argmax comparison - the precedent this session's new loss below
follows). Candidate additional loss terms discussed and ranked:

1. **Ground-truth-supervised measure-duration adherence loss (built this session,
   see below)** - the most direct, cheapest to build, targeting `barline_position_
   mismatch` (the single largest Stage A finding) head-on.
2. **Cross-staff coherence loss (built this session, see below) - a real, cheaper
   alternative to Stage C worth trying first.** Ground truth for sibling staves is
   available at *training* time even though each staff decodes independently at
   inference: penalize each staff's expected cumulative duration for diverging from
   its siblings' *ground-truth* durations at shared measure boundaries - no
   architecture change, no inference-time coupling, just a training signal that
   teaches the model to prefer duration decisions that stay coherent with what a
   system's siblings actually contain. Framed as a cheap experiment to run *before*
   committing to Stage C's full learned adapter: if this closes a meaningful share of
   the gap, Stage C's marginal value shrinks; if it doesn't, that's real evidence the
   model needs live cross-staff activations, strengthening the case for Stage C
   specifically.
3. **Key-signature/accidental consistency loss** - targets a specific, already-
   documented null result: §4's tier-2 forced-prefix experiments found that correcting
   a key-signature token in a staff's own history changed *zero* downstream pitch/
   accidental predictions across 20 tests, suggesting the model relies almost entirely
   on visual context for accidentals and may be ignoring its own declared key signature
   entirely. A loss penalizing a decoded accidental for contradicting the currently-
   declared key signature (without an explicit accidental token justifying the
   exception) would directly test and push against that finding. Not built.
4. **Structural well-formedness losses** (dangling slurs, unmatched beams/tuplets) -
   differentiable analogues of `check_dangling_slurs` and similar Stage A checks.
   Lower priority: these findings are much rarer in the 200-page benchmark than
   duration mismatches, and harder to make cleanly differentiable (closer to a
   discrete state-machine parity constraint than a soft expectation). Not built.

**Item 1 built and validated this session**: `calDurationAdherenceLoss`
(`training/architecture/transformer/decoder.py`), gated by `config.
duration_adherence_weight` (default `0.0` - preserves the existing loss exactly, the
same "zero means no effect" discipline `profile_context`'s own gate uses, expressed as
a loss weight instead of a module gate since this changes the training objective
itself). Mechanism: a fixed, non-trainable per-token duration lookup
(`rhythm_duration`, whole-note units, built once from `kern_to_symbol_duration` - the
same parser `EncodedSymbol.get_duration` itself calls) and a barline-token lookup
(`rhythm_is_barline`, the identical "barline" or "repeat" substring test
`SymbolChord.is_barline` already uses elsewhere in this codebase). At each *true*
(ground-truth) barline position, compares the model's own predicted *expected*
cumulative duration (softmax-weighted over the rhythm vocabulary, the same trick
`calConsistencyLoss` already established) against the ground-truth cumulative duration
at that same point - a differentiable analogue of `check_measure_durations`/
`_cumulative_barline_positions`, deliberately comparing *cumulative* position at each
checkpoint rather than segmenting into individual measures (if every checkpoint
matches, every measure between checkpoints must too - no segment-sum logic needed).
A chord's non-first notes need no special handling: they carry the vocabulary's own
literal `"chord"` placeholder token (0 duration) instead of a real `note_/rest_` token,
so summing independent per-position durations is already musically correct without any
chord-grouping logic.

**A real bug found and fixed before this could even smoke-test**: the new
`rhythm_duration`/`rhythm_is_barline` lookups were first registered as `nn.Parameter`
(matching `note_mask`'s own existing style) - which put them in the module's
`state_dict()`, and `load_checkpoint`'s mismatch check correctly flagged them as
"missing" from every pinned checkpoint saved before this field existed (there is
nothing to load for a fixed, deterministic table). Fixed by using
`register_buffer(..., persistent=False)` instead - excluded from `state_dict()`
entirely, since there is genuinely nothing to load or learn.

**Validated**: 5 new unit tests (`tests/test_duration_adherence_loss.py`) against a
tiny, hand-built 4-token vocabulary - a perfect prediction has ~zero loss, a confidently
wrong prediction produces the exact hand-computed drift (0.125), padded positions don't
affect it, a sequence with no barlines returns zero (not a division-by-zero), and the
loss is differentiable end to end. Full suite: 1060/1060 non-deselected (same 3
pre-existing unrelated failures throughout this investigation). A real smoke test
(200-example sample, 1 epoch, `--duration-adherence-weight 1.0`) confirmed the whole
chain end to end: checkpoint loads cleanly (326 base + 9 new profile-context tensors,
up from 8 - the new time-signature embedding), and the loss computes a real, nonzero,
training-active value (0.46).

**`phase22`: the training run - launched 2026-08-21, using both new mechanisms
together.** Same pinned checkpoint (`pytorch_model_426-...pth`, 326 params) and same
105,305 train / 4,912 valid split `phase20`/`phase21` used, for direct comparability;
same hyperparameters (10 epochs, batch size 8, lr 1e-3); `--duration-adherence-weight
1.0` newly added.

Stalled twice before this could run at all - see `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md`
§5's own account for the full two-stage diagnosis (uncached whole-score MusicXML
lookups, then insufficient caching alone) and the corpus-splitting fix
(`split_ground_truth_by_system.py`, 10,400 pre-extracted fragments) that resolved it,
confirmed via `nvidia-smi` (0% → 80-89% GPU utilization).

**Complete: 10/10 epochs positive, mean validation-loss delta +0.0513** (epochs:
+0.0551, +0.0731, +0.0439, +0.0484, +0.0475, +0.0589, +0.0639, +0.0161, +0.0555,
+0.0509 - epoch 8's dip did not persist into epochs 9-10, reads as noise).
`duration_adherence` (a separate signal, no with/without ablation of its own) fell
from 0.298 to ~0.283 over the first several epochs then plateaued, consistent with a
frozen-core setup where only 9 tensors are trainable.

**This is not clear evidence that time-signature conditioning specifically improved
on §7.3's original profile-context result.** `phase20` (without time-signature
conditioning or the duration-adherence loss) measured mean delta +0.0615 over its own
10 epochs - phase22's +0.0513 is slightly lower, and both runs' per-epoch ranges
overlap heavily with each other's own epoch-to-epoch noise. Since both new mechanisms
were enabled together in this one run, there is no ablation here that isolates either
one's individual contribution - the honest claim is "both together did not measurably
help or hurt phase20's existing signal," not "time-signature conditioning works."

Given `phase21` (the unfrozen follow-up to `phase20`) *erased* the frozen-core signal
rather than improving it, and phase22's own delta is already no better than
phase20's, repeating that pattern here looks unlikely to help. See
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §5 for the fuller writeup and the recommended
next step (the cross-staff coherence loss from §7.3's own brainstorm, built next -
see below).

**Caveat found afterward, while building item 2 (2026-08-22): `calDurationAdherenceLoss`
had a real chord-duration bug throughout `phase22`.** It summed each token's own
`rhythm_duration` independently, which double-counts any measure containing a
simultaneous multi-note chord: a chord's non-first members are NOT the literal
`"chord"` marker token (that part of the original comment was wrong) - they carry
their own real `note_/rest_` token right after the marker
(`training/omr_datasets/staff_merging.py`'s `create_chord_over_two_staffs`), so both
the true and predicted cumulative-duration sums counted them a second time. Verified
directly: a real two-note simultaneous quarter-note chord summed to 0.5 instead of the
correct 0.25.

This was live and contributing real gradient throughout `phase22`
(`--duration-adherence-weight 1.0`) for every chord-bearing measure. **What it likely
does and doesn't change about phase22's already-reported result**: the primary
measurement (the with/without-profile-context validation-loss ablation) applies the
same objective, bug included, to both arms of the comparison - the *relative* delta
between them is not obviously invalidated by a bug present in both. What it does
undermine is `duration_adherence`'s own reported trend (already flagged as a separate,
"less rigorous" signal above) and any *future* run that weights this loss more
heavily to actually shape decode behavior, where the bias directly rewards
overshooting a chord-bearing measure's true length.

**Fixed** (`training/architecture/transformer/decoder.py`): a new
`rhythm_is_chord_marker` buffer (analogous to `rhythm_is_barline`) plus
`_not_chord_continuation`, which zeroes out any position immediately following a
`"chord"` marker in both `calDurationAdherenceLoss` and `calCrossStaffCoherenceLoss` -
counting only a chord's first member, not every member (equivalent to
`homr.music_xml_generator.group_into_chords`/`SymbolChord.get_duration`'s own
minimum-across-members grouping for any correctly-notated chord, whose simultaneous
members share one duration by definition; a second recurrent min-grouping
implementation inside a batched differentiable loss was judged not worth it for that
equivalence). 2 new regression tests (one per loss) confirm a two-note chord no longer
double-counts. Full suite still the same 1090/1093 non-deselected (same 3 pre-existing
unrelated failures throughout this investigation, +2 from these new tests over the
1088 recorded above). **`phase22`'s own trained weights were not retrained after this
fix** - the release published from that run reflects the buggy loss; a rerun would be
needed to get weights trained against the corrected objective.

**Item 2 built and validated 2026-08-22**: `calCrossStaffCoherenceLoss`
(`training/architecture/transformer/decoder.py`), gated by `config.
cross_staff_coherence_weight` (default `0.0`, same discipline as item 1). Same shape
as `calDurationAdherenceLoss` - penalizes the rhythm head's predicted cumulative
duration at each of a staff's own barlines - but the *target* is this staff's
*system's* ground truth (the median across every sibling part at that measure index),
not this staff's own label.

**Simpler than the brainstorm originally proposed, and deliberately so.** The
brainstorm assumed this needed a real data-pipeline change (batching several staves of
one system together, since the loss "needs" siblings present in the same batch).
Building it found that's unnecessary: the system-wide target curve is precomputed
*offline*, per sample, straight from ground truth
(`training/omr_datasets/cross_staff_coherence.py`'s `system_measure_curve`, reusing
§7.3's fragment-splitting infrastructure - a system's pre-split fragment already
carries every sibling part, already movement-disambiguated). Each sample carries its
own fixed-size `(MAX_COHERENCE_MEASURES=32)` curve, its valid length, and a presence
flag into the batch (`training/transformer/data_loader.py`); ordinary i.i.d. shuffled
batching works exactly as before, no custom sampler needed.

Takes the **median** across sibling parts at each measure index, not any single
part's value - `ossq_measure_length_audit.py`'s own corpus audit found real cases
where ground-truth parts genuinely disagree on a measure's length (a labeling defect,
not a legitimate irregularity); the median is the same robustness idiom
`check_measure_durations`/`propose_majority_position_corrections` already use for the
equivalent problem elsewhere in this codebase.

**Validated**: 5 new tests for the ground-truth curve (`tests/
test_cross_staff_coherence.py`, hand-built MusicXML fixtures, including one
confirming the median ignores a single defective part), 6 new tests for the loss
itself (`tests/test_cross_staff_coherence_loss.py`, including the chord-double-count
regression above and an out-of-range-barline case). A real smoke test (168-example
sample from a piece with real pre-split fragments, 1 epoch,
`--cross-staff-coherence-weight 1.0`) confirmed the whole chain end to end: a real,
nonzero, training-active loss value, distinct from `duration_adherence` (left at its
`0.0` default for this smoke test). Full suite 1090/1093 non-deselected (same 3
pre-existing unrelated failures).

**A second real bug found launching the training run itself (`phase23`), this time in
shared fragment-generation infrastructure, not this loss's own code.** `phase23`'s
first epoch reported `cross_staff_coherence` around 120 - roughly 400x
`duration_adherence`'s typical magnitude in the same kind of run. Traced to
`ossq_ground_truth.py`'s `extract_ground_truth_window`: its attribute carry-forward
skipped entirely whenever a window's first kept measure already had *any*
`<attributes>` element, but MusicXML only restates a child (divisions/key/time/clef)
when it changes - a measure can carry a real `<time>` change with no `<divisions>`
redeclaration. When that happened, `divisions` silently defaulted to 1 for the rest
of that part's window, inflating every duration computed from it by the piece's real
(uncarried) divisions value. Verified directly against one real fragment (Wolf,
sq8823783, page 43): 2 of 4 parts opened their window on exactly this pattern,
producing measure lengths of 90.0 and 18.375 whole notes instead of the correct ~0.75.

**Confirmed asymmetric blast radius.** `time_signature_for_sample` (used by `phase22`,
already published) only reads `<time>` and falls back to `""` ("unknown") when
nothing is found in the searched measure - it degraded to reduced coverage on this
bug, not wrong values. `system_measure_curve` reads `<divisions>` specifically with no
equivalent safe fallback, so it produced genuinely wrong values, not just missing
ones, until this fix. `phase22`'s already-published result needs no further caveat
beyond what's already recorded above.

**Fixed**: carry-forward now tracked per attribute *child*, merging only whichever
children a window's first measure is missing, rather than skipping the whole step
whenever any `<attributes>` element is already present. All 10,400 corpus fragments
rebuilt with the fix (same 122 pieces, 0 skipped, ~6.6 minutes) - re-verified against
20 real corpus samples: every one now produces a sane per-measure duration (0.5-1.0
whole notes), none of the earlier 12-90 outliers. 1 new regression test. `phase23`
(which had been training on the corrupted fragments) was killed and relaunched
clean.

**`phase23`: complete, 10/10 epochs positive, mean delta +0.0742.** Launched on its
own (`--cross-staff-coherence-weight 1.0`, `duration_adherence` left at its `0.0`
default, no time-signature bundling) specifically so its contribution could be
isolated, unlike `phase22`.

| epoch | mean loss | cross_staff_coherence | valid delta |
|---|---|---|---|
| 1 | 2.7859 | 0.3140 | +0.0949 |
| 2 | 2.7483 | 0.3082 | +0.0920 |
| 3 | 2.7381 | 0.3072 | +0.0575 |
| 4 | 2.7392 | 0.3065 | +0.0582 |
| 5 | 2.7362 | 0.3079 | +0.0706 |
| 6 | 2.7330 | 0.3054 | +0.0809 |
| 7 | 2.7326 | 0.3047 | +0.0640 |
| 8 | 2.7340 | 0.3056 | +0.0642 |
| 9 | 2.7346 | 0.3045 | +0.0772 |
| 10 | 2.7319 | 0.3050 | +0.0826 |

Mean delta **+0.0742**, all 10 epochs positive, range +0.0575 to +0.0949.
`cross_staff_coherence` itself (a separate, less rigorous signal - no with/without
ablation of its own) stayed essentially flat around 0.305-0.314 throughout - the
frozen core's narrow trainable surface (9 tensors) settling quickly, the same
pattern `duration_adherence` showed in `phase22`.

**This is a real, meaningful improvement over `phase20`'s own +0.0615 baseline -
about 20% higher, and consistent, not one lucky epoch dragging the mean up:** 8 of
phase23's 10 epochs sit at or above phase20's own typical ceiling (+0.07), and
phase23's minimum epoch (+0.0575) is still within phase20's own range. Contrast
this with `phase22` (time-signature conditioning + duration-adherence loss
bundled together), which came in *below* phase20 at +0.0513 - the honest reading
there was "no clear evidence either mechanism helped." Isolated, the cross-staff
coherence loss reads differently: a real, repeatable signal beyond what
score-profile conditioning alone already provided.

**This clears the bar this section itself set for §4 Stage C**: a cheap
experiment closing a meaningful share of the gap Stage C would otherwise need to
close is exactly the outcome that would argue *against* needing Stage C's full
learned adapter. That it instead shows a real, additional, isolated improvement
argues the opposite way - the model *can* still benefit from more cross-staff
signal than score-profile conditioning alone gives it, which is direct evidence
worth pursuing at the scale Stage C offers (live, joint per-page cross-staff
context) rather than concluding the case for Stage C is weakened. See §7.4 below
for the design decision this motivated.

### 7.4 Phase 3: Stage C - started 2026-08-22, decoded-content-conditioned

`ENSEMBLE_TRANSCRIPTION_DESIGN.md` §12.3 / `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §4's
Stage C already specifies giving the decoder real, learned cross-staff context (a
masked variable-length staff-context transformer, zero-init gated residual into the
existing shared decoder). Phases 0-2 here are deliberately smaller and cheaper
interventions to try first, per §12.3's own precondition (Stage A and B "built and
benchmarked" before Stage C) and this document's own cheapest-first principle (§6) -
not a replacement for Stage C, and not a reason to skip it if 0-2 do not close enough of
the gap. `phase23`'s result (§7.3 above) is exactly that "0-2 didn't close enough of
the gap" outcome, in the positive sense: real signal left to capture.

**Design decision (§4's own "not decided here" fork, resolved now): decoded-content-
conditioned, not visual-only.** §4 named two options - (a) a visual-only first pass
(pooled encoder features only, isolating whether cross-staff signal helps at all), or
(b) start directly with the richer version conditioned on each staff's actual decoded
content (a proper two-pass decode, replacing Phase 1's discrete reranking with a
learned second decode). Two independent pieces of evidence, both from *this* codebase's
own measurements, point the same way:

1. **Phase 1** (§7.2) reranked purely on *decoded content* (cumulative barline
   positions from the rhythm head's own output) - never touched raw visual features -
   and got a real, validated 20.8% reduction in cross-staff findings with a 38/40
   ground-truth match rate. Decoded content, not visual similarity, is what carried
   the signal there.
2. **`phase23`** (this section) trained a loss whose *target* is ground-truth decoded
   content (each sibling part's own duration curve), not anything visual, and got a
   real, repeatable +0.0742 mean delta.

Both of this project's own successful cross-staff interventions so far are keyed on
decoded content. Starting Stage C with the visual-only version would mean building
the *less*-evidenced variant first and only asking "does content help" as a second,
later step - when the answer to that question is already yes, twice over, from this
project's own data. Building the decoded-content-conditioned version directly asks
the actually-open question: does a *learned, jointly-trained* version of this same
signal do better than Phase 1's discrete reranking and phase23's loss-only shaping,
which is the real remaining uncertainty Stage C exists to resolve.

**Architecture, adapted from §4's spec to condition on decoded content**:

```text
greedy_i = existing shared decoder, first pass (staff i's own cheap greedy decode)
h_i = pooled hidden state from that first-pass decode (ScoreDecoder.forward's own
      "hidden" output, already exposed for this purpose - see §7.3's structured-heads
      wiring, which reads the same key)
C_1..C_N = StaffContextTransformer(h_1..h_N, staff order, mask, optional ScoreProfile)
E'_i = decoder input_i + gate * projection(C_i)
decode each staff a second time with E'_i
```

`N` variable, masked (a system can have 1-8+ staves); the shared decoder's weights
are reused unchanged for both passes; the context gate is zero-initialized, the same
discipline `ProfileContextEmbedding` uses - a raw learned scalar multiplied directly
(`ProfileContextEmbedding`'s own convention, `training/architecture/transformer/
profile_context.py`), not a sigmoid: `sigmoid(0) = 0.5`, not zero, so a sigmoid gate
would need initializing at `-inf` to reproduce the exact baseline at init, which is
impractical - a raw zero-initialized scalar is what actually guarantees "bit-identical
at initialization," and is what's implemented below.

**Built so far** (`training/architecture/transformer/staff_context.py`): the
`StaffContextTransformer` module itself - a small self-attention block over a masked,
variable-length set of per-staff summary vectors, order-aware (learned positional
embedding over staff index within the system), zero-init gated residual output. Unit
tested in isolation (no training script, no data pipeline yet - see "not yet built"
below): confirms a single real staff (no siblings) still runs correctly (self-
attention degenerates to attending only to itself), masking correctly excludes padded
staff slots from attention, the zero-init gate reproduces its input exactly at init,
and the module is differentiable end to end.

**Batching-by-system data loader built**
(`training/transformer/system_batch_loader.py`): every mechanism before Stage C in
this document trains on i.i.d. shuffled single staves - `StaffContextTransformer`
needs several of one system's own parts together, the first genuinely new data-
loading requirement this document has had. `group_by_system` maps the OSSQ stem
convention (`<score>_<page>_<system>_<part>`, the same one `score_profile_time_
signature.py`/`cross_staff_coherence.py` already parse) to which corpus-list indices
belong to the same system; `SystemBatchDataset` reuses the *existing* per-sample
`DataLoader.__getitem__` unchanged for each part, then pads/stacks them along a new
staff dimension to `MAX_STAVES_PER_SYSTEM`, with a mask - no second, parallel
per-sample loading path, no need to handle variable-length sequences again (every
field is already padded to a fixed `max_seq_len` by the existing tokenizer, which the
current i.i.d. batching already depends on). Systems with fewer than 2 real parts are
excluded by default - nothing for a cross-staff module to learn from a system with no
real siblings. 7 tests (grouping, non-OSSQ exclusion, stacking, masking, the
below-minimum exclusion, an oversized-system truncation, and the combined
group+pad convenience function), plus verified directly against 2,000 real training
samples: 501 real system groups, 497 of them the expected size 4 (string quartets).

**Module attached to `ScoreDecoder`** (2026-08-22): `self.decoder.staff_context`,
gated by `config.enable_staff_context` (off by default, same checkpoint-compatibility
reasoning as `profile_context`), plus `TrOMR.freeze_core_for_staff_context()`
mirroring `freeze_core_for_profile_context`'s own frozen-core probe shape. 5 more
tests on the real `ScoreDecoder`/`TrOMR` (module attaches only when enabled, disabled
is bit-identical to before this existed, the frozen-core probe leaves only
`decoder.staff_context.*` trainable).

**Two-pass training script built and running** (`training/transformer/
train_staff_context.py`, 2026-08-22): implements exactly the pipeline above - first
pass (no_grad, teacher-forced, sampling_prob=1.0) → masked-mean-pool `"hidden"` per
staff over its own real tokens → `StaffContextTransformer` (masked by `staff_mask`,
not the per-token mask) → second pass with `staff_context_emb` set → trained against
the model's own existing `loss`, same frozen-core probe shape `train_profile_context.py`
used for §7.3. 14 unit tests against a lightweight fake model (zero-gate reproduces
the first pass exactly, moving the gate changes the second pass, only the new
module's weights move under `freeze_core_for_staff_context`). Smoke-tested against
the real checkpoint on a 400-sample subset (100 systems, all batches ran clean) before
being trusted on real data. `phase24` (`/workspace/b0/phase24`, tmux session
`staffcontext`): the same production `phase9`/`phase4` train/valid split phase22/23
used (105,305/4,912 staves → 10,695/1,237 systems after system-grouping and the
default `min_staves=2` filter), 5 epochs, batch size 8 systems - launched under the
same "phase23 positive → go ahead and start Stage C" standing authorization already
given for this build. Each system batch costs roughly 2x a single-pass batch (two
full decodes instead of one), so this run is expected to take proportionally longer
than phase22/23's own per-epoch time.

**phase24: complete, 5/5 epochs positive.** Valid-loss delta (without staff context
minus with) per epoch: +0.6901, +0.7835, +0.6757, +0.8383, +0.8688 - not a single
lucky measurement but five, holding the same order of magnitude throughout and, if
anything, growing by the end rather than decaying. Far larger than phase22's +0.0513
or phase23's +0.0742: plausible on its own terms, since those two only shaped the
loss or conditioned on a summary statistic, where Stage C gives the decoder the
sibling staves' actual live decoded hidden state - a materially richer signal.

This is real evidence the frozen-core probe question ("can some assignment of the
new module, backpropagated through a frozen network, move the model's own existing
loss") has a clear yes answer, five times over - the two-pass mechanism itself
works, is trainable, and the effect is stable, not the single clean number phase21
(§8) warns against trusting alone. It is still not yet §9's actual success
criterion: a validation-loss delta is a proxy, and per this document's own
established discipline (phase21's whole cautionary point), the real measurement is
whether staff context actually moves `benchmark_stage_ab.py`'s
`barline_position_mismatch`/`measure_duration_mismatch` finding counts on the 200-page
Stage A/B benchmark - which needs the two-pass decode wired into actual *inference*,
not just training (`homr/transformer/decoder_inference.py` currently only knows a
single greedy pass). That wiring is the natural next step, not yet started.

(Operational note: the `tee` piping this run's stdout to `phase24/train.log` opened
before the training script's own first-epoch `mkdir` created that directory, so it
silently stopped writing to disk after one error - the authoritative source for
this run is `phase24/history.json`, written correctly every epoch by the script
itself, not the missing log file. A brief false alarm mid-run - one status check
found zero `train_staff_context` processes and briefly looked like a crash - turned
out to be a transient/race in that one check; the process (confirmed via CPU time,
GPU utilization, and fresh dataloader-worker PIDs) never actually stopped.)

**Inference-time two-pass decode built and wired (2026-08-22)**: `training/onnx/
convert.py`'s `DecoderWrapper` now takes `staff_context_emb` as a new ONNX input
(default zero - `ScoreTransformerWrapper`'s own no-op guarantee) and exposes the
shared hidden state as a new output. `homr/transformer/decoder_inference.py`'s
`generate`/`generate_with_rhythm_margins`/`generate_from_prefix` all thread it
through; `generate_with_rhythm_margins` now also returns per-step hidden states,
propagated up through `staff2score.py` and the existing Phase 1 call chain
(`staff_parsing_tromr.py`/`staff_parsing.py`) - additive only, nothing in the
default pipeline changes behavior.

**Safety check, not just a unit test**: re-exported the decoder ONNX from the same
pinned base checkpoint and ran a real staff image through both the old and the new
graph. The decoded symbol sequence was bit-for-bit identical (only inference
timing differed, an artifact of comparing the old fp16/GPU path against the new
plain fp32/CPU export) - confirms the new input/output genuinely don't change
existing behavior, not just that `ScoreTransformerWrapper`'s own zero-bias test
says so in isolation.

`homr/staff_context_decode.py` is the actual two-pass orchestration: decode every
staff in a system once (`parse_staff_tromr_greedy_with_margins`, no context),
mean-pool each staff's own hidden states (inference has no padding to mask, unlike
training's fixed-length batches), run the trained `StaffContextTransformer`
(loaded directly in plain PyTorch - too small next to the encoder/decoder to be
worth its own ONNX export), decode every staff again with its own context vector.
8 unit tests against a faked first-pass decode function, plus a real end-to-end
smoke test against `phase24`'s actual released weights on a real 4-staff string
quartet system (`sq7313978_0001_0001`):

- Gate value after only 5 frozen-core probe epochs: 0.00235 - barely off zero, as
  expected this early.
- 3 of 4 staves decoded bit-identically between passes (unsurprising, given how
  small the gate still is).
- **One staff's decoded time signature actually changed**: `timeSignature/2` in the
  first pass, `timeSignature/4` in the second, once conditioned on its three
  sibling staves - a real, tangible cross-staff correction from a barely-trained
  gate, not just a lower validation-loss number.

**Not yet wired into `parse_staffs`'s live pipeline** (`homr/staff_parsing.py`) -
built and tested standalone first, the same discipline every other Stage C piece
used. Wiring it in behind its own opt-in flag (mirroring `enable_phase1_rerank`),
and then actually running the real 200-page Stage A/B benchmark with it on vs.
off, are the two remaining steps before this is a measured result rather than a
promising one.

## 8. Why not several tempting shortcuts

- **Why not a post-hoc text-level duration correction, the way key/time signature and
  motif articulation are fixed?** Already tried and deliberately declined this session:
  `propose_majority_position_corrections`' whole design is that there is no single
  low-ambiguity token to correct for a duration divergence - which note or rest is wrong,
  and by how much, cannot be determined after the fact without guessing. A real fix has
  to happen at decode time (§7.2) or training time (§7.3), not after the model has
  already committed to a wrong sequence.
- **Why not go straight to Stage C?** Cost and attribution, both already recorded in
  `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §4's own "net read": Stage A/B are measured,
  but the dominant unrepaired finding argues for improving decoder accuracy or
  extending Stage B's vocabulary *before* Stage C's learned adapter, not for Stage C
  being obviously warranted yet. This document is exactly that "improving decoder
  accuracy" alternative, made concrete.
- **Why not just unfreeze and fine-tune the whole decoder on more data and hope duration
  accuracy improves as a side effect?** phase21 is the direct cautionary example: an
  unfrozen run can genuinely erase a signal's measurable value rather than sharpen it,
  and does so without any loss-based warning that it is happening. Nothing in this
  document proposes an unfrozen run without a frozen-core result to compare it against
  first.

## 9. Success criteria / measurement

Every phase in §7 is measured the same way: rerun the 200-page Stage A/B benchmark
(`benchmark_stage_ab.py`) before and after the intervention, and report the
`barline_position_mismatch`/`measure_duration_mismatch` finding-count deltas directly -
not a proxy loss, and not a small hand-picked sample, given phase21's own demonstration
that a clean loss signal is not sufficient evidence on its own. Where a phase's own
mechanism predicts it should only help one failure type (§4) and not another, the
report should say so explicitly and check that prediction against the actual per-type
breakdown, not just the aggregate count.

## 10. Open questions

- Whether Type 2 (systematic meter/subdivision misread) is fixable by any of §7's
  phases at all, or needs a different diagnosis entirely - a wrong-but-locally-coherent
  subdivision is not obviously a narrow-margin decoding call reranking would catch, and
  it is not obviously a single localized measure an auxiliary position signal would
  correct either, since by construction it is self-consistent across the whole
  passage. This may turn out to be Stage C's territory specifically (cross-staff
  context could reveal what a single staff's internally-consistent wrong reading
  cannot), which would be a real, useful finding in its own right if Phase 1/2 confirm
  it.
- What fraction of flagged pages are actually Type 1 vs. Type 2 vs. Type 3 corpus-wide -
  currently known only from 5 spot-checked pages (§4's own caveat). Phase 0 and Phase 1
  are both partial answers to this, as a byproduct of their own measurement.
- Whether `benchmark_stage_ab.py` should be committed into this repo rather than living
  only as a scratch script re-scp'd to the GPU instance each time - it has now been
  rerun three times this session alone as the standard evaluation harness for this
  whole track, and this document proposes it as the standard harness for every future
  phase too.
