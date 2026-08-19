# Ensemble transcription: open threads to work from

**Status:** active reference, companion to `ENSEMBLE_TRANSCRIPTION_DESIGN.md`

**Date:** 2026-08-19

This is the working document. `ENSEMBLE_TRANSCRIPTION_DESIGN.md` (§27, "Reproduction
record") is the full session-by-session history — good for reconstructing *why* a
decision was made or re-deriving a number, too long to read before starting new work.
This file pulls forward only what is still open, with enough of the "why" attached that
a fresh session does not repeat an already-failed attempt. When a thread here gets
resolved, record the result in the history doc (continue its `27.N`/`28.N` numbering) and
either delete the corresponding section here or replace it with a one-line pointer to
where it landed.

Of the five parts in the original design's executive summary:

| Part | Status |
|---|---|
| 1. Structured output heads (beam/stem/slur/tie/dynamics) | Done |
| 2. Optional score-profile conditioning | **In progress** — §3 below |
| 3. System grouping after segmentation | Done (`homr/system_grouping.py`) |
| 4. Cross-staff consistency checks and repair | **In progress** — §4 below |
| 5. Page-local inference with review-system evidence | Partially — the per-head structured evidence exists; no page-assembly/review surface has been built |

The four threads below are the ones with enough existing work (or existing spec) to
pick back up directly, in the priority order the evidence supports.

---

## 1. The text detector's page-level precision has collapsed for five of seven classes

**This is the most immediately actionable thread** — the detector, training scripts, and
evaluation tooling all already exist and work; this is a training-recipe/architecture
question, not a new subsystem.

### Where it stands

`training/ocr/detector_masks.py`'s `CLASS_ORDER` is 7 classes: `Dynamic, Fingering,
Expression, Tempo, MeasureNumber, StaffText, Lyrics` (SystemText was folded into
StaffText via `CLASS_ALIASES` — see §1.3). Whole-page box evaluation
(`training/ocr/detector_box_eval.py`, tiled inference with 50%-overlap stitching,
greedy one-to-one IoU matching), most recent run (`detector4`, adding Dynamic, 27.95),
against 307 held-out pages:

```
class           precision  recall     f1   gt boxes
Dynamic            75.9%   93.9%   84.0%       429
MeasureNumber      75.5%   94.9%   84.1%        78
Lyrics             66.7%   94.1%   78.1%     3,555
Tempo              16.2%   67.9%   26.2%        53
StaffText           9.9%   73.6%   17.4%       106
Expression          4.4%   72.7%    8.3%        44
Fingering             0%      0%      0%        12
```

**Only Lyrics and MeasureNumber (and now Dynamic) are usable.** Tempo, StaffText, and
Expression have high recall (the model finds real boxes) but precision in the
single-digit-to-teens range (it also invents many more) — 50 predicted Tempo boxes for
every real one. Fingering is at exact zero on both axes.

### Why this happened (§27.86–27.87, `ENSEMBLE_TRANSCRIPTION_DESIGN.md`)

The original per-pixel training/valid IoU numbers (27.86: 0.81–0.997 across every class)
looked strong enough to skip an architecture search. That measurement only ever showed
the model `DetectorPatches`' 70%-positive-biased training crops. A full page is >99%
background outside boxes; the model was never evaluated — or, implicitly, ever trained —
against that ratio. The whole-page eval (27.87) exposed the real failure: a
training-patch-distribution vs. inference-distribution domain gap, not a per-pixel
segmentation problem. **`DetectorPatches.POSITIVE_RATIO = 0.7` is still the leading
suspect and has not been changed.**

### What has already been tried

- **Class-balanced positive sampling** (27.89, `training/ocr/detector_patches.py`'s
  `box_centres_by_class` + two-step draw): fixed a *within-page* imbalance (a page with
  many Lyrics boxes and one Tempo mark almost never centred a patch on the Tempo mark).
  Real, broad win — overall F1 58.6%→70.5%, Tempo precision roughly quintupled. **This is
  in production** (current `detector_patches.py`). Did not move Fingering or SystemText at
  all, because the fix only redistributes attention *within* a page that contains the
  class — it cannot manufacture positive examples on pages that never contain the class.
- **Synthetic rendered data for the rarest classes** (27.90–27.93,
  `training/ocr/rare_class_synthesis.py`): rendered real MuseScore engravings with
  injected `<fingering>`/`<direction>` elements, verified against actual SVG output before
  trusting it at scale. Moved Fingering off zero (18.8% F1) — **the technique works** — but
  SystemText stayed at exactly 0% despite a comparable injection with confirmed-present
  mask pixels (not a pipeline bug), and **every other class got measurably worse**
  (overall F1 70.5%→57.9%, MeasureNumber 95.7%→78.7%). Diagnosis: the class-balanced
  sampler draws a positive centre uniformly among classes *present on a page*; for a class
  whose only positive pages are the 79 synthetic ones, those 79 pages get selected far more
  often than their 79/2,621 share of the corpus would suggest, and the model overfit to
  something idiosyncratic about them — the same mechanism 27.72 found for global focal
  loss (a correction aimed at one starved class bleeding into classes it was never meant to
  touch). **Not currently used** — `detector4`'s weights come from the class-balanced-only
  recipe, not this run.
- **SystemText folded into StaffText** (27.93, user decision): given up as its own class.
  `detector_masks.py: CLASS_ALIASES = {"SystemText": "StaffText"}`. Fingering's synthesis
  path is unaffected and kept; SystemText's injection code was removed (not left dormant).

### Not yet tried

1. **Cap a class's positive-draw frequency by distinct page count, not just uniform
   per-page draw.** 27.92's own diagnosis names this directly: weight the sampler so 79
   synthetic pages cannot out-compete 2,542 real ones for attention, even though the
   current two-step draw treats "present on this page" as page-level not corpus-level
   evidence. This is a `DetectorPatches`/`box_centres_by_class` change, not a new script.
2. **Generate synthetic data on substantially more distinct source scores at lower
   density per page**, rather than concentrating heavy injection on the same 79 — the
   other half of 27.92's proposed fix, untested against #1.
3. **Raise `POSITIVE_RATIO` toward true page frequency** (or make it class-dependent) —
   27.87's leading, still-untested hypothesis for the *general* five-class precision
   collapse, independent of the Fingering/SystemText-specific rarity problem above. This
   is the fix that would move Tempo/StaffText/Expression, which the class-balancing and
   synthesis work never targeted (Tempo already existed in every page category; its
   problem is background-vs-foreground ratio, not within-page class competition).
4. ~~A confusion matrix for SystemText specifically~~ — **closed, not to be revisited.**
   User decision: the staff/system distinction does not matter for this project's
   purposes, so SystemText-vs-StaffText is not worth further detector capacity or
   analysis regardless of what a confusion matrix would show. The fold (27.93) is final,
   not conditional on a future measurement.
5. **`train_detector.py` has no per-epoch checkpointing** (27.91) — a real gap independent
   of the above: a long run that plateaus early cannot be stopped without losing all
   progress. Worth fixing before the next multi-hour detector run, the same problem 27.88
   hit for the recognizer.

---

## 2. Fingering's corpus-level rarity (sharpest edge of §1; SystemText is closed, see below)

Kept separate from §1 because the fix shape is different: this is not a training-recipe
problem on the existing corpus, it is that the *source* corpus contains 12 (Fingering)
ground-truth boxes total across the entire 2,542-page training set (27.68) — no amount
of resampling within that data can manufacture signal the corpus does not contain.

- SystemText: folded into StaffText (§1.3 above) — final, per user decision. Not a
  measurement question; the staff/system distinction does not matter for this project's
  purposes, so there is nothing left to check here.
- Fingering: synthesis (§1) is the only lever that has moved it off zero, and items 1–2
  in §1's "not yet tried" list are exactly what would make that lever usable without
  collateral damage. There is no separate Fingering-only next step beyond what §1 already
  lists — resolving §1 resolves this.

---

## 3. Score-profile conditioning — schema and layout use built; conditioning not started

Full contract already specified in `ENSEMBLE_TRANSCRIPTION_DESIGN.md` §7; reproduced
here so this file is self-contained.

### Built so far

- `homr/score_profile.py`: `ScorePart`/`ScoreProfile` dataclasses implementing §7.1's
  contract exactly (schema `homr.score-profile.v1`, `to_dict`/`from_dict`, a missing or
  mismatched `schemaVersion` refuses rather than defaulting — the same discipline
  `homr.transformer.capability_manifest` already uses). `expected_staff_pattern` expands
  each part to one `stableId` per physical staff it occupies (a `expectedStaffCount=2`
  piano part contributes its id twice, adjacently), which is what makes a profile
  directly comparable to a detected staff sequence regardless of how many staves each
  part spans. A `STRING_QUARTET` example profile is included as the running example this
  design and the OSSQ corpus both use.
- `homr/score_profile_layout.py`: `propose_part_assignment`, the §7.2 layout use. Built
  directly on `homr.system_grouping.assign_voice_slots` rather than re-solving staff
  identity — that function already resolves which voice slot (0..N-1) a detected staff
  belongs to, including the missing-staff case a real page's bracket detector gets
  wrong, using pure geometry. This module only lays a profile's expected pattern against
  that resolved slot sequence. Per system: an exact staff-count match produces a full
  `staff_to_part` mapping at `evidence_score=1.0`; any mismatch (wrong staff count, or
  voice slots `system_grouping` itself could not resolve) produces an empty mapping with
  a stated deviation reason rather than a guess — per §7.1's "never a hard constraint"
  and the concrete risk that a wrong mapping would attach the wrong instrument context to
  real music, which is worse than no mapping at all.
- 23 tests (`tests/test_score_profile.py`, `tests/test_score_profile_layout.py`), all
  passing; `tests/test_system_grouping.py`'s existing 30 unaffected.
- **Not wired into `homr/main.py` or any live pipeline yet.** Both modules are pure
  functions over plain data (a `ScoreProfile`, a `SystemPartition`, a voice-slot list) —
  deliberately decoupled from image/`Staff`-object handling so they could be built and
  tested without touching the live pipeline, the same way `system_grouping.py` itself
  stays decoupled from pixels. Wiring a profile *in* (accepting one as a job input) and
  wiring the assignment *out* (into whatever consumes per-staff decoding today, and into
  a review surface per §7.2's "exact source-image regions" requirement) is the next step
  before this thread is usable end to end.
- `evidence_score` is binary (1.0 exact match, 0.0 anything else) — §7.2 asks for "an
  evidence score with competing hypotheses," which this does not yet provide. A partial
  match (right total count, ambiguous per-part split) currently reports the same 0.0 as a
  completely wrong count; distinguishing those is unimplemented, not yet needed by
  anything downstream.

### Contract

An optional, document-scoped JSON profile — not an "ensemble type" flag — describing
expected parts:

```jsonc
{
  "schemaVersion": "homr.score-profile.v1",
  "parts": [
    {
      "stableId": "violin-1",
      "displayName": "Violin I",
      "instrumentFamily": "strings.violin",
      "expectedStaffCount": 1,
      "likelyClefs": ["G2"],
      "transpositionSemitones": 0,
      "lyricsExpected": false
    }
    // ... one entry per expected part
  ]
}
```

`stableId` is scoped to the submitted job, not a universal instrument registry. Unknown
names/families/clefs/counts/transpositions are all valid — this is a soft hint, never a
hard constraint (explicitly: "Do not make instrument range a hard pitch constraint.
Ranges and likely clefs are priors; real music legitimately violates them.").

### Use in layout

Supplies an expected ordered physical-staff pattern (`[1,1,1,1]` for a quartet, `[1,2]`
for voice+piano) used as a *scored hypothesis*, not an assertion. Layout must report:
detected physical staff count, proposed systems/staff rows, proposed row→profile-part
mapping, an evidence score with competing hypotheses, deviations from the supplied
profile, and exact source-image regions.

### Use in staff recognition

For a recognized staff, encode only the applicable context: instrument-family embedding,
part-ordinal embedding, staff-within-part-ordinal embedding, expected-staff-count
embedding, likely-clef-set embedding, transposition embedding, and an
unknown/context-missing indicator. First implementation: inject as prefix/context tokens
to the decoder, or a gated additive vector to encoder context — **the gate must be
zero-initialized so the unconditioned path is bit-identical at initialization**, the
same zero-gate discipline the structured heads already use for backward compatibility.

### Training: context dropout

Randomly remove the whole profile and independently mask fields during training, so the
model does not become dependent on it. Starting hypothesis (not fixed): 30% no profile,
30% partially masked, 40% complete. Evaluate both conditioned and unconditioned
inference.

### Why this is next after §1, not before

§24's original implementation slice lists this as item 9, after the structured heads
(item 1, done) and system grouping (also done). Nothing about it depends on §1's detector
work resolving first — it is independent, parallel work. It is ranked below §1 here only
because §1 has existing infrastructure and a clear next experiment already named, where
this thread needs a new module (`homr/score_profile.py` or similar does not exist yet)
built from a written spec with no empirical validation yet.

### Next implementation step

The schema, the deterministic layout use (§7.2), `staff_to_part_by_system`, and the
`--score-profile` CLI wiring are all built and validated end to end (above) - a supplied
profile now reaches `parse_staffs` and activates #4's clef-vs-profile check. What
remains is genuinely new work, not plumbing:

- **§7.2's layout use (`propose_part_assignment`) is still not called from
  `parse_staffs` or surfaced anywhere** - only its `staff_to_part_by_system` sibling
  (built for #4's clef check specifically) is wired in. The layout mapping itself -
  proposed systems/staff rows, evidence score, deviations, source-image regions - has no
  consumer yet. Worth revisiting whether `propose_part_assignment` and
  `staff_to_part_by_system` should collapse into one function now that both exist and
  overlap in purpose, or stay separate because they answer different questions
  (`SystemPartition`-indexed staff mapping vs. voice-number-indexed).
- **§7.3 decoder conditioning** — genuinely new work: instrument-family/part-ordinal/
  staff-within-part/expected-staff-count/likely-clef/transposition embeddings injected
  as prefix tokens or a zero-initialized gated additive vector, plus §7.4's training-time
  context dropout. Needs a training run to validate. Do this after the layout use has a
  real consumer — a profile with no way to reach a human reviewer or the decoder is not
  yet worth conditioning training on.

---

## 4. Cross-staff consistency checks and repair — Stage A partially built

Full spec in `ENSEMBLE_TRANSCRIPTION_DESIGN.md` §12; reproduced here. Staged deliberately
from least to most invasive — **do not start Stage C before Stages A and B are built and
benchmarked**, per §12.3's own explicit precondition, which still has not been met (only
part of Stage A exists).

### Stage A: deterministic consistency analysis — 4 of 8 checks built

`homr/cross_staff_consistency.py`: `analyze_system(staves, staff_to_part=None)` takes one
`list[EncodedSymbol]` per staff of an already-decoded system (plus, optionally,
`score_profile_layout.py`'s `staff_to_part` mapping) and returns structured `Finding`s -
no MusicXML is altered. Decoupled from images/`Staff` geometry, same as
`system_grouping.py`/`score_profile_layout.py`, so it is tested against hand-built token
sequences (21 tests, `tests/test_cross_staff_consistency.py`).

Built:

- different decoded measure counts across parts (`check_measure_counts`);
- conflicting key/time signatures (`check_key_signatures`/`check_time_signatures`) - the
  **full sequence** of key/time-signature tokens is compared, not just the opening value,
  so two staves that agree at the start but diverge on a later change still get flagged;
- a clef inconsistent with a supplied score profile (`check_clefs_against_profile`) - ties
  directly into `homr/score_profile.py`'s `ScorePart.likely_clefs`; silent when no part
  was proposed for a staff or the part states no expected clefs, per §7.1's "unknown is
  valid, not an error";
- a beam or slur endpoint made dangling (`check_dangling_slurs`) - within one staff's own
  decode: a slur slot still open at the end, or a STOP/START_AND_STOP with nothing open
  in that slot.

Not built, named honestly in the module docstring rather than left silently uncovered:

- conflicting barline **locations** (needs relative position within the measure, not just
  a count - two staves can agree on measure count and still have barlines drift);
- one voice's measure duration disagreeing with the time signature (needs duration
  arithmetic across a whole measure, not just token-sequence comparison);
- part order changing between systems, and missing/extra staff output - both need state
  carried **across** systems on a page, which nothing built so far tracks (everything
  above operates on one system's staves in isolation).

### Wired into the live pipeline, log-only, validated on real pages

`homr/staff_parsing.py`'s `parse_staffs` now calls `findings_by_page` (via
`_report_cross_staff_findings`) right after a page's voices are fully decoded, and logs
whatever it finds with `eprint`. `parse_staffs`' return value is untouched - nothing
downstream can be affected by what this reports - and the whole call is wrapped in a
broad `try/except` that logs and swallows, since a diagnostic must never be the reason a
page fails to transcribe. Skipped when `selected_staff >= 0` (a debug mode where most
voices are deliberately absent from most systems for reasons unrelated to real music).

Run end to end against two real OSSQ pages on a GPU instance (`sq7313978:0001`,
`sq8823783:0061`): both completed normally, `homr.main` wrote MusicXML in both cases, the
exception guard never fired, and it surfaced real findings - a key-signature restatement
only one of four parts carried into its second system on the first page; a
time-signature mismatch and two key-signature mismatches on the second. These read as
genuine per-voice decode disagreements (each staff decodes independently, so nothing
stops one voice's transformer output from restating or dropping a signature differently
than its neighbours), not this module misreading its own input.

**Not done yet:**

- ~~No score profile reaches this call site~~ - fixed: `--score-profile` (§3) now
  threads a `ScoreProfile` through `main.py` into `_report_cross_staff_findings`, which
  calls `staff_to_part_by_system` and activates the clef check. Validated on a real page
  with a `STRING_QUARTET` profile: no crash, and correctly found no mismatch (this
  page's actual clefs match the profile).
- Findings only go to `eprint`. Logged, surfaced to a review UI, or asserted in a
  benchmark are all still open - nothing beyond a log line consumes them yet.
- Only 4 of 8 §12.1 checks exist (above) - the other four would surface through the same
  wiring once built, no further pipeline changes needed.

### Stage B: targeted repair proposals from existing alternatives (tier 1 built for key/time signature)

Use the decoder's already-computed greedy logits and top-k alternatives to propose a
*bounded local* repair when a specific low-confidence head explains a Stage-A
inconsistency — e.g. a viola's measure reads 7/8 against every other part's 4/4, and
token 31's second-choice alternative (0.41 confidence, one flag shorter) restores 4/4.
This is a review question surfaced with the aligned source crop and every affected
staff's reading, not an automatic correction — applying it follows the same
token-regeneration and content-signature rules as an ordinary confidence correction
already documented elsewhere in the design. Explicitly **not** beam-search sequence
decoding: targeted use of alternatives already computed, after a deterministic check has
narrowed the search neighborhood to almost nothing.

**Refinement (from a design discussion, not yet acted on): Stage B as specified above
has a real limitation for exactly the case that motivates it.** Swapping one token from
its own top-k alternative changes *only* that token — everything decoded after it in
that staff was chosen under the *original* (wrong) context and is not regenerated. That
is fine for an isolated rhythm token (the 7/8-vs-4/4 example), but wrong for something
like a key signature: a wrong key can plausibly have shaped every accidental spelling
decoded after it in that staff, and a bare token swap does not fix those. Three tiers,
increasing in power and cost, only the first of which is what Stage B's text above
actually describes:

1. **Deterministic repair, no model call.** Edit the outlier's token stream directly —
   e.g. replace a key/time-signature token with the system's majority value. Cheapest,
   and reasonable specifically near the start of a staff (a key/time signature) where
   little has been decoded yet under the wrong assumption.
2. **Forced-prefix re-decode using the existing frozen decoder — real conditioning, no
   training.** `homr/transformer/decoder_inference.py`'s `ScoreDecoder.generate()` is a
   causal transformer decoder with a KV cache keyed by absolute step (`cache_len` is a
   literal input to the ONNX graph) - architecturally capable of taking a corrected
   prefix and regenerating everything after it consistently. **Not built**: `generate()`
   as written assumes a length-1 `BOS` seed - each step feeds only `start_tokens[:, -1:]`
   into the loop, so handing it a genuine multi-token forced prefix today would silently
   drop everything in it but the last token, never populating the cache for the earlier
   ones. Supporting this needs a "prefill" phase (run the corrected prefix through the
   model first, one step per token, to build the cache properly - all of
   rhythm/pitch/lift/articulation/slur/position, not just rhythm, since the model takes
   them as parallel inputs) before the existing greedy loop takes over. Real, scoped,
   buildable as new, additive capability (a new method, not a change to the existing
   `generate()` call path) - not attempted yet because it touches actual inference
   correctness, unlike everything built for #3/#4 so far, which was either pure logic or
   log-only and additive to the live pipeline.
3. **Learned cross-staff conditioning (§12.3 Stage C, or extending §7.3's own
   profile-injection mechanism to also carry Stage A's computed majority signature as a
   signal, not just a user-supplied profile).** The "real" answer, needs training, and
   §12.3 already gates it behind measuring the simpler stages first.

**Tier 2 built and validated on the real model - and it surfaced a finding that changes
how promising this whole tier is for the motivating use case.**
`ScoreDecoder.generate_from_prefix()` (`homr/transformer/decoder_inference.py`)
teacher-forces a caller-supplied prefix through the model one token at a time, then
falls through to ordinary greedy decoding - additive, `generate()` itself untouched, no
retraining. Two things measured against the real ONNX model on the instance (this
module has no mocked test coverage - the IO-binding/cache contract is exactly what a
mock risks getting subtly wrong, so this was validated the same way #4's live wiring
was, against real inference rather than a substitute):

1. **Mechanism check: replaying a staff's own first 5 decoded symbols as a forced
   prefix and continuing reproduces the rest of that staff's decode bit-for-bit
   identical to an unforced `generate()` call.** Confirms the cache/step/position
   handling is correct, not merely plausible.
2. **The motivating case itself: forcing a genuinely different key signature
   (`keySignature_-1` -> `keySignature_4`, mid-staff) left every downstream pitch and
   accidental (`lift`) prediction unchanged**, on the one staff tested. Not a defect in
   the mechanism - (1) already proves that works - but a real, measured property of
   this model: its pitch/accidental decisions appear driven by the visual encoder
   context (attention over the image) more than by the key-signature token in the
   autoregressive history. Plausible for an OMR model specifically - the accidentals are
   directly visible in the image, so the decoder may not need to "remember" a key
   signature to read them - but it means **tier 2's practical value for exactly the
   key/time-signature repair case that motivated it is not yet established**, on n=1.

**Measured across 12 more tests (6 key-signature, 6 rhythm/duration corrections, 6
different scores): 0/12 showed any downstream difference at all.** Every corrected
continuation was length-for-length, token-for-token identical to the original
uncorrected tail - including the rhythm/duration case this section originally expected
*might* differ (a mid-measure duration change has a plausible reason to shift subsequent
barline-relative decoding that a key signature's effect on accidentals does not; it
didn't). This is no longer an n=1 curiosity - it is a consistent, decisive result:

**Decided: tier 1 (the plain deterministic token swap) is the recommended Stage B
implementation, not tier 2, until a case is found where they diverge.** Given the model's
future predictions did not depend on the corrected token's identity in any of 12
independent tests, tier 2's extra decoder pass buys nothing over tier 1's free edit for
every case measured so far - both tiers necessarily produce the identical final result
when the model ignores the correction either way. Tier 2's code stays
(`generate_from_prefix`, already built, harmless, cheap to call) for the case that does
turn out to be token-history-sensitive, rather than deleted on a 12-sample null result -
but the working assumption going forward is tier 1 first, tier 2 only if a specific
correction type is shown to need it.

**Slur hypothesis tested and closed too: 0/8, even for a stronger test than planned.**
Forced a `slurStop` onto a position whose real value was "no slur" (these 8 staves
happened to start on a real note, not a restated clef, so the forced correction was
introducing a *phantom slur stop with no matching start anywhere in the decode* - a
musically invalid, maximally provocative case for testing whether the model's own
sense of open/closed slur spans depends on token history). Zero effect downstream on
all 8. Combined total: **0/20 across pitch, rhythm/duration, key signature, and slur,
on 20 independent corrections across the corpus.** No longer leaving this as an open
hypothesis - the model's future predictions do not appear to depend on any field's
token-history identity, only on cache position and the visual context. Tier 1 (§4's
plain deterministic swap) is the settled recommendation for Stage B; tier 2's code
stays available rather than deleted, in case some as-yet-untested correction type
(a different field, a much longer prefix, a correction much later in a staff) turns out
to behave differently.

**Tier 1's own proposal logic built.** Deciding tier 1 over tier 2 answered *how* to
apply a correction, not *what* to propose - Stage A only logged findings, nothing turned
one into a concrete edit. `homr/cross_staff_repair.py`: `propose_majority_correction(staves,
prefix)` finds each staff's *opening* `"keySignature"`/`"timeSignature"`-prefixed token
and proposes correcting every minority staff toward the majority value, refusing (empty
result) on a genuine tie or when fewer than two staves state a value at all - the same
"report nothing rather than guess" discipline `score_profile_layout.py` already uses.
`apply_proposal` returns a new symbol list (never mutates in place) and refuses a stale
proposal whose target position no longer matches what it was built against. Deliberately
scoped to the opening value only, matching §12.2's own worked example - not the full
sequence-alignment problem a later key/time change would need. `propose_repairs` pools
both checks (key + time signature) into one call. 14 tests, all passing.

**Wired into the live pipeline, log-only, alongside Stage A.**
`staff_parsing._report_cross_staff_findings` now logs both Stage A findings and tier-1
proposals for the same system's staves in one pass - `staves_by_system` (the
voice-to-system reshaping, extracted out of `findings_by_page` rather than duplicated a
second time, since the presence-cursor logic is exactly the kind of subtle
correctness-critical thing that should not be reimplemented independently) is computed
once and both `analyze_system` and `propose_repairs` read from it. Proposals are logged,
never applied - same discipline as Stage A. Same broad try/except guard as before: a
diagnostic must never be the reason a page fails to transcribe.

**Not yet validated against a real page** - unlike Stage A itself and tier 2, both
already run end to end on the GPU instance. The GPU was occupied with the §1 detector
retrain when this was built; next step once it frees up.

### A second motivating case for cross-staff repair, from a design discussion: shared
motifs, not just shared attributes

Not yet built, and a different kind of check from anything in §12.1's original list.
Key/time signature/clef are page-wide *attributes* that should trivially agree across
parts - Stage A's existing checks are all this shape: compare one value per staff,
flag disagreement. A shared *motif* is richer: multiple voices playing the same
rhythmic/melodic idea (a fugal subject, an imitative entry, doubled parts) is content-level
agreement, not attribute-level, and one voice mistaking an accent for a marcato (or any
single-note articulation/ornament misread) would not show up in any check built so far -
nothing currently compares *note-level* content across staves at all, only page-wide
attributes and each staff's own internal slur consistency.

What this would need, not yet designed in detail: a similarity/alignment pass between
two decoded voices' rhythm+articulation (and plausibly pitch-interval-normalized, so a
transposed imitative entry still matches) token sequences - a windowed alignment or
edit-distance search, not exact equality, since the "same" motif can appear at different
metric positions or transposed. Once a matching span is found with one voice disagreeing
on a single field (articulation being the concrete example), the repair question is the
same Stage B question already answered above: apply the majority reading, and per the
tier-1/tier-2 finding just measured, a deterministic swap is the reasonable first attempt
rather than assuming forced-prefix conditioning is needed.

This is a real, valuable addition to what Stage A checks for, but it is new detection
work (motif alignment across staves), not a repair-tier question - left here as a named
idea rather than folded into the existing 4 built checks, since it needs its own design
pass on the alignment method before implementation.

### Stage C: learned variable-staff context adapter (not started, blocked on A+B being measured)

```text
E_i = shared visual encoder(staff image i)
h_i = masked pool(E_i)
C_1..C_N = StaffContextTransformer(h_1..h_N, staff order, ScoreProfile)
E'_i = E_i + sigmoid(gate_i) * projection(C_i)
decode each E'_i with the existing shared decoder
```

`N` is variable, padded with an explicit mask; the visual encoder and decoder weights are
reused unchanged; the context gate is zero-initialized (reproduces the baseline exactly
at init, the same discipline as §3's profile-conditioning gate); missing profile
information is supported; no token alignment across parts is required. First version
uses one summary per physical staff — do not build a richer version (exchanging
system-position features or decoded measure summaries) before this one is measured.

**Why not a fixed four-staff decoder** (§12.4, settled): would make the first benchmark
easier but blocks or complicates piano, Lieder, trios, orchestral reductions, missing
staves, divisi, and partial crops — a masked variable-length set of staff summaries gives
quartet context without hard-coding quartet as the architecture.

### Why this is last

§12.3 states its own precondition in the imperative: only build the learned adapter
*after* Stages A and B are benchmarked. Stage A alone may resolve enough — the design
never assumes it will need Stage C. Starting here means starting at the beginning
(Stage A), which has no dependency on §1 or §3 resolving first, but is scoped last in
this document because it is the newest, least-derisked of the four threads: no code, no
measurement, and its own success criterion (Stage C being *unnecessary*) is still
completely open.
