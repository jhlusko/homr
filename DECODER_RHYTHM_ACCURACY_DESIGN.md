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
shaped (one clean, isolated wrong note, exactly what reranking targets). Distinguishing
the two at scale - not attempted here - would need the same content-level check
Beethoven got, run across all 87, not just totals.

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

### 7.4 Phase 3 (already designed elsewhere, referenced not duplicated): Stage C

`ENSEMBLE_TRANSCRIPTION_DESIGN.md` §12.3 / `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §4's
Stage C already specifies giving the decoder real, learned cross-staff context (a
masked variable-length staff-context transformer, zero-init gated residual into the
existing shared decoder). Phases 0-2 here are deliberately smaller and cheaper
interventions to try first, per §12.3's own precondition (Stage A and B "built and
benchmarked" before Stage C) and this document's own cheapest-first principle (§6) -
not a replacement for Stage C, and not a reason to skip it if 0-2 do not close enough of
the gap.

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
