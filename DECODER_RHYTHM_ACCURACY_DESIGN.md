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
