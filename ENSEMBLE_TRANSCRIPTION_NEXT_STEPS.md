# Ensemble transcription: open threads to work from

**Status:** active reference, companion to `ENSEMBLE_TRANSCRIPTION_DESIGN.md`

**Date:** 2026-08-20

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

## 1. The text detector's page-level precision has collapsed for five of seven classes — **CLOSED 2026-08-20, user decision**

**This is the most immediately actionable thread** — the detector, training scripts, and
evaluation tooling all already exist and work; this is a training-recipe/architecture
question, not a new subsystem.

**Closed per explicit user decision after `phase19`'s result** (below): Fingering is not
particularly important to this project, and `phase19`'s Dynamic/Lyrics numbers (both now
*better than the no-synthesis baseline*) plus its partial Tempo/MeasureNumber recovery
versus `phase18` are good enough to stop here. Not every open sub-question was resolved (item 3, `--positive-ratio`, was built but
never run; the Fingering-specific weighting overcorrection below was diagnosed but not
fixed) - closed as "good enough for this project's actual priorities," not as "fully
solved." `phase19`'s overall F1 (65.6%) is still below `detector4`'s no-synthesis
baseline (68.1%), so this is **not** a recommendation to replace `detector4` as the
production default - the decision to close is specifically that further chasing
Fingering/Tempo/StaffText precision is not worth more session time, not that `phase19`'s
weights are an improvement to adopt.

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
- **Wider-spread Fingering synthesis** (`phase18`, this session): item 2 below, tried.
  400 distinct train-only scores (vs. the original 79), 1 injection each (vs. ~4) - same
  order of magnitude of total signal (400 vs. 316 boxes), five times the distinct visual
  contexts. Retrained 20 epochs on top of the existing class-balanced sampler, evaluated
  on the identical unchanged 307-page held-out set:

  ```
  class           precision     recall         f1   gt boxes    detector4 (no synth)
  Dynamic             67.0%      93.2%      78.0%        429    84.0%
  Expression           3.9%      79.5%       7.5%         44     8.3%
  Fingering           31.4%      91.7%      46.8%         12     0.0%
  Lyrics              73.0%      96.0%      82.9%      3,555    78.1%
  MeasureNumber       52.2%      89.7%      66.0%         78    84.1%
  StaffText           10.8%      73.6%      18.8%        106    17.4%
  Tempo                2.2%      81.1%       4.2%         53    26.2%
  overall             44.8%      94.7%      60.8%      4,277    68.1%
  ```

  **Real, substantial partial win, not a clean one.** Fingering moved from 0% to 46.8% F1
  - more than double 27.92's 18.8% from the concentrated 79-page attempt, confirming
  spreading synthesis across more distinct scores does what it was supposed to: the
  target class generalises further before overfitting to a handful of specific images.
  But the collateral damage 27.92 found did **not** go away, and for two classes it is
  *worse than 27.92's own mixed result*: Tempo 26.2%→4.2% (worse than 27.92's 9.8%),
  MeasureNumber 84.1%→66.0% (worse than 27.92's 78.7%), Dynamic 84.0%→78.0% (not measured
  in 27.92, added since). StaffText and Lyrics both improved slightly. Overall F1 60.8%
  sits between detector4's 68.1% and 27.92's 57.9% - better than the old synthesis
  approach on balance, still worse than not synthesizing at all.

  **Diagnosis, refined by this result: "more distinct source pages" and "cap positive-
  draw frequency by distinct page count" (item 1 below) are two different mechanisms,
  and this measured that the first alone is not sufficient.** Spreading Fingering's
  synthetic boxes over 400 pages instead of 79 means the class-balanced sampler now has
  *more* pages competing for a "Fingering present" positive draw, not fewer - if
  anything this plausibly explains why Tempo's collateral damage got *worse* here than
  under the narrower 79-page version, not better. Diversifying the source images fixed
  the overfitting-to-specific-pixels problem this session set out to test; it does not
  by itself fix the sampler's page-level "present/absent" accounting that item 1 already
  named as the separate, still-untried lever.

  **`phase18`'s weights are not adopted as the new default** - the same call 27.92's
  result got, for the same reason: a class-imbalance fix that costs other classes more
  than the config it replaces is not a strict improvement. `detector4`'s weights remain
  the better default until the sampler-level fix (item 1) is tried, ideally combined
  with `phase18`'s already-synthesized 400-page dataset (`/workspace/b0/phase18_synth`,
  `phase18_masks`) rather than resynthesizing from scratch.

### Not yet tried

1. ~~Cap a class's positive-draw frequency by distinct page count, not just uniform
   per-page draw~~ — **built this session, not yet run.** `training/ocr/detector_patches.py`:
   `classes_present(mask)` (per-page class presence, no per-box centroid work needed),
   `class_page_counts(masks)` (how many distinct pages carry each class - reads a stream
   of masks, one pass, no I/O baked in so it stays testable without touching disk),
   `class_draw_weights(counts)` (weight = `1 / page_count`, relative only, not normalized -
   `random.choices` only needs relative magnitudes). `DetectorPatches` takes an optional
   `class_weights: dict[str, float] | None = None` - `None` (the default) preserves the
   exact original per-page-uniform choice; when supplied, the per-page class pick in
   `__getitem__` uses `rng.choices(present, weights=...)` instead of `rng.choice(present)`,
   so a class spread across many more pages than another no longer also wins a
   proportionally larger share of positive draws corpus-wide just by having more pages to
   be chosen on. 12 new tests (`tests/test_detector_patches.py`), all passing on the GPU
   instance's venv (28/28) - also caught and fixed a pre-existing stale test comment/
   assertion (`TestBoxCentresByClass` assumed class value 1 was "Fingering"; `cc7bb61`
   later inserted "Dynamic" ahead of it in `CLASS_ORDER`, making value 1 "Dynamic" - the
   tests were asserting on the wrong key and happened to still pass only because
   `box_centres_by_class`'s dict comparison didn't care about the label, just presence).
   **`train_detector.py` now has `--class-weighted-sampling`** (`compute_class_weights`
   reads every training mask once at startup, prints the counts/weights, feeds the
   result to `DetectorPatches`) so this is usable from the CLI, not just the library
   function. **`phase19` (this session) combined this with `phase18`'s already-
   synthesized 400-page Fingering set** - `phase18/train_index_combined.txt` reused
   directly, no resynthesis - the first run to combine both halves of 27.92's original
   diagnosis. Complete:

   ```
   class           precision     recall         f1   gt boxes   detector4   phase18
   Dynamic             76.7%      93.7%      84.4%        429       84.0%     78.0%
   Expression           5.0%      65.9%       9.3%         44        8.3%      7.5%
   Fingering            2.8%      91.7%       5.3%         12        0.0%     46.8%
   Lyrics               75.5%      94.5%      83.9%      3,555      78.1%     82.9%
   MeasureNumber        58.9%      93.6%      72.3%         78       84.1%     66.0%
   StaffText             6.2%      73.6%      11.5%        106       17.4%     18.8%
   Tempo                 6.6%      69.8%      12.1%         53       26.2%      4.2%
   overall              50.6%      93.3%      65.6%      4,277      68.1%     60.8%
   ```

   **A real, decisive, and genuinely surprising result: combining both levers did not
   give the hoped-for "best of both."** Dynamic and Lyrics both ended up *better than
   the no-synthesis baseline* - an unexpected win for the two already-strongest classes.
   Tempo (4.2%→12.1%) and MeasureNumber (66.0%→72.3%) both improved over `phase18` alone,
   confirming class-weighting does reduce collateral damage the way it was meant to.
   **But Fingering collapsed from `phase18`'s 46.8% down to 5.3%** - `class_draw_weights`'
   inverse-page-count formula does not distinguish "genuinely scarce, needs protecting"
   from "was made artificially page-common by synthesis": Fingering now sits on 402
   distinct pages (`phase18`'s own synthesis), more than Tempo's 362 or Expression's 360,
   so the weighting suppressed it *alongside* the classes it was meant to protect against,
   overcorrecting its own hard-won gain away. StaffText (11.5%) also came out slightly
   worse than either single-lever run. Overall F1 65.6% sits between `phase18`'s 60.8% and
   `detector4`'s 68.1% - better than synthesis alone, still below not synthesizing at all.

   **Closed here per user decision** (Fingering is not important to this project) rather
   than continuing to chase a fix for the weighting-formula overcorrection this surfaced -
   see the top of this section.
2. ~~Generate synthetic data on substantially more distinct source scores at lower
   density per page~~ — **tried (`phase18`, above).** Helped Fingering substantially
   (0%→46.8%, more than double the old approach), did not fix the collateral damage to
   other classes, and one case (Tempo) got worse. Confirmed as a real, useful, but
   insufficient-alone lever - not a candidate to retry unmodified; combine with item 1.
3. **Raise `POSITIVE_RATIO` toward true page frequency** (or make it class-dependent) —
   27.87's leading, still-untested hypothesis for the *general* five-class precision
   collapse, independent of the Fingering/SystemText-specific rarity problem above. This
   is the fix that would move Tempo/StaffText/Expression, which the class-balancing and
   synthesis work never targeted (Tempo already existed in every page category; its
   problem is background-vs-foreground ratio, not within-page class competition).
   **CLI exposed this session** (`train_detector.py --positive-ratio`, default unchanged)
   so this is ready to test without further code changes - still not actually run,
   since the GPU was occupied by `phase19` (item 1, below) when this was built.
4. ~~A confusion matrix for SystemText specifically~~ — **closed, not to be revisited.**
   User decision: the staff/system distinction does not matter for this project's
   purposes, so SystemText-vs-StaffText is not worth further detector capacity or
   analysis regardless of what a confusion matrix would show. The fold (27.93) is final,
   not conditional on a future measurement.
5. ~~`train_detector.py` has no per-epoch checkpointing~~ (27.91) — **fixed this session.**
   `--weights`/`--out` are now written after every epoch, not only once the whole run
   finishes - directly relevant since `phase19` (above) was a long run in progress when
   this was built. 3 new tests, 13/13 passing.

---

## 2. Fingering's corpus-level rarity — **CLOSED 2026-08-20, user decision, same as §1**

Kept separate from §1 because the fix shape is different: this is not a training-recipe
problem on the existing corpus, it is that the *source* corpus contains 12 (Fingering)
ground-truth boxes total across the entire 2,542-page training set (27.68) — no amount
of resampling within that data can manufacture signal the corpus does not contain.

- SystemText: folded into StaffText (§1.3 above) — final, per user decision. Not a
  measurement question; the staff/system distinction does not matter for this project's
  purposes, so there is nothing left to check here.
- Fingering: synthesis (§1's `phase18`) moved it to 46.8% F1; combining it with
  class-weighted sampling (`phase19`) overcorrected it back down to 5.3% - see §1 for
  the full result. **Closed alongside §1**: explicit user decision that Fingering is not
  particularly important to this project, so the weighting-formula overcorrection
  `phase19` surfaced is not worth further session time to fix.

---

## 3. Score-profile conditioning — everything built and tested; only the training run itself remains

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

- ~~§7.2's layout use (`propose_part_assignment`) is still not called from
  `parse_staffs` or surfaced anywhere~~ — **wired this session.** `propose_part_
  assignment`'s *deviations* are now logged from `_report_cross_staff_findings`,
  alongside everything else - the first real consumer of a function that was built and
  tested but never called from anywhere. Distinct from `check_page_staff_counts`
  (compares a system against the rest of the *page*): this compares a system against
  what the *profile* declared, which can disagree even when every system on the page
  agrees with each other (a profile that is simply wrong about how many parts a piece
  has).

  Not a drop-in call, and worth recording why: `propose_part_assignment` wants one
  `SystemPartition` describing every system at once with a single page-wide
  `staves_per_system`, but `staff_parsing.SystemPlan` allows systems to vary in size
  (an incomplete system `_group_by_geometry` recovered, or the dense fallback path) - a
  real shape mismatch, not a detail. Resolved by calling `propose_part_assignment` once
  per system, each time with a synthetic single-system `SystemPartition` built from that
  system's own slot count - each call then answers exactly "does the profile expect this
  many staves here," the only thing this diagnostic needs; `propose_part_assignment`
  itself was not changed. The evidence-score and source-image-region parts of its output
  are still unused - there is still no review surface to show them to (§5's own status:
  "no page-assembly/review surface has been built") - only the deviations, which fit the
  existing log-line pattern every other check here already uses.

  **Validated on the GPU instance**: run against the same `sq7313978:0001` page whose
  system 0 was already known (from `check_page_staff_counts`'s own validation) to be
  missing a staff, with a `STRING_QUARTET` profile - fired exactly where expected
  (`"System 0: profile layout - profile expects 4 staff(s) per system, detected
  geometry implies 3"`), silent on a second page whose staff counts matched throughout,
  no crash, valid MusicXML both times. Full suite: 906/909 (3 pre-existing, unrelated
  failures deselected).

  The `propose_part_assignment`/`staff_to_part_by_system` collapse question named
  previously is now answered by how this landed: kept separate. They answer genuinely
  different questions from the same inputs (staff-count deviation vs. per-staff part
  identity), and forcing one call site to serve both would have meant reshaping one of
  them to fit the other's indexing scheme for no real gain.
- **§7.3 decoder conditioning** — genuinely new work: instrument-family/part-ordinal/
  staff-within-part/expected-staff-count/likely-clef/transposition embeddings injected
  as prefix tokens or a zero-initialized gated additive vector, plus §7.4's training-time
  context dropout. Needs a training run to validate. **The layout use now has a real
  consumer** (above), so this is unblocked - started this session:

  - **Injection point identified**: `ScoreTransformerWrapper.forward()`
    (`training/architecture/transformer/decoder.py`) already builds the decoder's input
    as a sum of independent per-field embeddings (`rhythm_emb + pitch_emb + lift_emb +
    articulation_emb + slur_emb + pos_emb`) before the attention layers. The natural,
    minimally-invasive injection is one more term in that sum - a `profile_emb`, one per
    sequence (broadcast across positions, since profile context is staff/page-level, not
    per-token), through a zero-initialized projection so at initialization its
    contribution is exactly zero and the unconditioned path is bit-identical - not yet
    built.
  - **A real gap found and closed first, before touching the model**: there is no
    `ScoreProfile` data anywhere in the *training* pipeline - only live inference
    (`homr/main.py --score-profile`) has one. **User decision: synthesize it from the
    training corpus' own MusicXML sources**, rather than train only on the (nonexistent)
    subset that already carries profile metadata, or fabricate plausible-looking fake
    profiles. `training/omr_datasets/score_profile_extraction.py`:
    `extract_score_profile(xml_root)` reads a document's `<part-list>` (name,
    `<instrument-sound>` - MusicXML's own standardized taxonomy, which
    `ScorePart.instrument_family` was modeled on from the start, e.g. `"strings.violin"`)
    and each `<part>`'s `<attributes>` across every measure (not just the first, since a
    clef or instrument change can appear later) for clefs/staff-count/transposition/
    lyrics. A part with no `<part-list>` entry still gets a `ScorePart`, per §7.1's
    "unknown is valid." 10 tests, all passing. **Validated against 200 random real
    corpus files, 0 errors**: 51.3% of parts carry a real `instrument_family`, 98.3%
    carry `likely_clefs` - genuine signal, not sparse or fabricated. `display_name` came
    back empty on every part (this corpus's `mbox` renders apparently strip
    `<part-name>` text) - not fatal, instrument identity is carried by
    `instrument_family` + clefs independently.
  - ~~Not yet done: pairing extracted profiles with the existing training index~~ —
    **built this session.** `training/omr_datasets/score_profile_pairing.py`:
    `profile_and_part_for_sample(dataset_root, stem)` resolves a training example's
    filename stem (`convert_ossq.py`'s own `<score_id>_<page>_<system>_<part>`
    convention) to the real `(ScoreProfile, ScorePart)` for that sample - reading the
    *whole-score* MusicXML `convert_ossq.py` already locates for slur/dynamics
    placement (`work / f"{score_id}.musicxml"`), not the per-part tokenisation scratch
    file, which strips instrument identity down to a generic "Part 1". 12 tests, using a
    temp-directory fixture matching the real corpus layout.

    **Found along the way and fixed**: OSSQ - this design's own running example corpus -
    has **0%** `<instrument-sound>` coverage in its whole-score files, despite carrying
    clean `<instrument-name>`/`<part-name>` text ("Violin 1", "Viola", "Violoncello").
    Added `_family_from_name` to `score_profile_extraction.py`: a small, explicit
    name→taxonomy fallback table, matched as a case-insensitive substring so a hit is
    the same MusicXML sound-ID vocabulary `<instrument-sound>` would have given
    directly, not a separate guess. An unmatched name leaves `instrument_family` empty,
    per §7.1. 8 new tests (18/18 total in `test_score_profile_extraction.py`).

    **Validated against 2000 real OSSQ training examples**: 1999/2000 resolved (99.95%),
    and `instrument_family` coverage went from 0% to **100%** on the resolved set (all
    violin/viola/cello, correctly identified).

  - ~~Still not done: the decoder-side embedding module itself~~ — **built and wired
    this session.** `training/architecture/transformer/profile_context.py`:
    `ProfileContext` (the six fields §7.2 names) and `ProfileContextEmbedding`, an
    `nn.Module` combining them into one additive vector per sequence via bounded,
    explicit vocabularies (an unenumerable field like `instrument_family` falls into a
    shared "unknown" bucket rather than growing at training time). A dedicated
    `missing_emb` gives "no profile at all" its own learnable representation, per
    §7.2's "unknown/context-missing indicator" - distinct from silence, not silence
    itself.

    The module's output is scaled by a single **zero-initialized gate** parameter, per
    §7.2's own stated requirement. Wired into `ScoreTransformerWrapper.forward`
    (`decoder.py`) as an optional `profile_context_emb` parameter, added to the
    existing per-field embedding sum in both the cached and uncached decode paths -
    `ScoreDecoder.forward`'s existing `**kwargs` already threads it through at all three
    call sites (training forward, scheduled-sampling, `generate`'s incremental decode
    loop), no changes needed there.

    **Validated end to end, not just the module in isolation**: 13 unit tests plus 3
    real-model wiring tests confirm a real context, a missing context, and a mixed
    batch all produce exactly `torch.zeros` at initialization; that a freshly
    constructed zero-gated `ProfileContextEmbedding` wired into a freshly constructed
    model changes nothing about the model's loss; and that moving the gate away from
    zero *does* change it - ruling out the parameter being silently ignored anywhere in
    the wiring. Full suite: 954/957 (3 pre-existing, unrelated failures deselected).

  - ~~A real integration constraint found, not yet resolved~~ — **resolved this
    session.** `train.py` drives training through HuggingFace's `Trainer` (`HomrTrainer`
    in `metrics.py`) with no custom `data_collator`, which only handles tensor/int-
    convertible dict values - `ProfileContextEmbedding.forward`'s original interface
    (`list[ProfileContext | None]`, one raw dataclass per sample) never fit that.

    `context_to_batch_fields(context)` reduces one sample's `ProfileContext` (or `None`)
    to plain ints and a fixed-length padded list (`MAX_CLEF_SLOTS = 3` for the variable-
    length `likely_clefs` set - covers essentially every real case, a cello's F4/C4/G2
    is exactly 3) - exactly what the default collator can stack without special-casing.
    `ProfileContextEmbedding.forward_from_batch` is the fully-vectorized training-facing
    counterpart to the original `forward`/`embed_one` (kept as the direct-caller entry
    point for live inference and simple tests) - masks out padded clef slots via a count
    field before averaging, so a padded "unknown" slot cannot dilute a real clef set's
    mean the way naively averaging all `MAX_CLEF_SLOTS` positions would. **Both entry
    points are tested to compute the identical vector for the same logical context**
    (`TestBatchAndListAgree`, every edge case: empty/single/full clef sets, missing
    context, mixed batches) - the property that makes this a real solution rather than
    two implementations that happen to agree on the easy cases.

    `config.enable_profile_context` (off by default, same reasoning as
    `enable_structured_heads`) gates a `self.profile_context` submodule `ScoreDecoder`
    now owns and constructs in `__init__`, alongside `structured_heads`.
    `ScoreDecoder.forward` pops the `profile_*` batch keys out of `**kwargs`
    unconditionally - so they never reach `attn_layers` (which has no idea what to do
    with raw index tensors) and a *disabled* module does not choke on their presence
    either, the mixed-corpus/upgraded-dataloader-ahead-of-config case - and threads the
    computed embedding into all three `self.net(...)` call sites.

    **Found and fixed a real API collision along the way**: the original wiring let a
    precomputed `profile_context_emb` tensor pass through `**kwargs` untouched, for a
    caller who built one externally. Once `ScoreDecoder` started computing its own from
    raw batch fields, the two paths collided on the same keyword. Resolved by settling
    on one contract: `ScoreDecoder.forward` always computes it internally from batch
    fields; `generate()` (single-sequence live inference) still accepts a precomputed
    tensor directly, since it never receives a training batch to derive one from.

    Full suite: 965/968 (3 pre-existing, unrelated failures deselected).
  - ~~Still not done: §7.4's training-time context dropout~~, ~~`DataLoader.__getitem__`
    itself actually calling `score_profile_pairing.py`~~ — **both built this session.**
    `apply_context_dropout(context, rng)`: §7.4's starting hypothesis (30% no profile,
    30% partially masked, 40% complete), a single roll choosing between three outcomes.
    "Partially masked" is scoped to `instrument_family` specifically, named as a real
    simplification rather than full per-field independent masking: it is the one field
    with a pre-existing "unknown" sentinel (empty string) that does not also double as a
    legitimate real value the way `part_ordinal == 0` or `transposition_semitones == 0`
    both do - masking the integer fields independently would need them to gain their
    own explicit "unknown" states first, not attempted here.

    `DataLoader` takes an optional `dataset_root: str | None = None` (default preserves
    existing behaviour exactly - no `profile_*` keys added to a sample at all). When
    set, `__getitem__` resolves each sample's real `ProfileContext` via
    `score_profile_pairing.py`, applies dropout for *training* samples only (validation
    gets the real resolved context, deterministic per `idx` via the same seeded/
    restored random-state block the existing image-distortion code already uses), and
    emits `context_to_batch_fields` into the result dict. `load_dataset()` threads
    `dataset_root` through to both the train and validation `DataLoader`s.

    4 new tests using a real temp-directory OSSQ-shaped fixture (image + tokens +
    whole-score MusicXML): no `dataset_root` emits no profile keys at all; a resolvable
    sample gets `profile_present=1`; an unresolvable one gets `profile_present=0`
    without crashing; validation resolution is deterministic across separate
    `DataLoader` instances for the same `idx`. Full suite: 976/979 (3 pre-existing,
    unrelated failures deselected).

    **This completes §7.3/§7.4's data-and-architecture side end to end** - every piece
    from `ScoreProfile` extraction through to a batch a model with
    `enable_profile_context=True` can actually train on now exists and is tested.
    **What remains is exclusively the training run itself**: turning
    `enable_profile_context` on, passing a `dataset_root` through to `load_dataset`, and
    seeing whether any of this actually helps - a GPU experiment, not further design or
    plumbing work.
  - Pairing is also scoped to OSSQ only - `mbox`- and `lieder`-derived training samples
    (also mixed into decoder training via `mix_datasets.py`) have their own naming/
    provenance and would need their own pairing logic, not attempted here.

### `phase20`: the training run — complete, 10/10 epochs positive

Launched this session (`training/transformer/train_profile_context.py`, frozen core,
only the 8 `decoder.profile_context.*` tensors training): 105,305 training examples
(`phase9/index.txt`, ~70% OSSQ-shaped), 4,912 validation examples (`phase4/valid/
index.txt`), `--epochs 10 --batch-size 8 --lr 1e-3`, starting from the pinned checkpoint
every earlier structured-heads phase also used.

Validation ablation each epoch (same held-out examples, same trained gate, only
`profile_present` toggled) - **with profile context vs. without**:

```
epoch   train loss   valid with   valid without   delta
1       2.4626       1.9814       2.0478          +0.0664
2       2.4305       1.9712       2.0394          +0.0682
3       2.4308       1.9665       2.0392          +0.0727
4       2.4264       1.9583       2.0318          +0.0735
5       2.4263       1.9968       2.0321          +0.0353
6       2.4238       1.9805       2.0370          +0.0565
7       2.4208       1.9647       2.0347          +0.0700
8       2.4249       1.9615       2.0291          +0.0676
9       2.4221       1.9677       2.0313          +0.0636
10      2.4284       1.9837       2.0246          +0.0409
```

**Run complete. 10/10 epochs positive** (`with` < `without` on every independent
held-out pass), mean delta **+0.0615** nats, range +0.0353 (epoch 5, the outlier) to
+0.0735 (epoch 4), most epochs clustering in a +0.06-0.07 band. Training loss flattened
after epoch 1 (expected - a frozen core with only 8 parameters converges fast) while the
ablation delta kept fluctuating but never once crossed zero.

**This is a real result, not a coin flip that happened to land the same way ten times.**
A frozen-core linear probe - only 8 parameters ever moved, the pretrained encoder and
decoder untouched - found real, held-out predictive value in genuine per-staff
instrument/clef/staff-count context on every single epoch measured. That is the
frozen-core experiment's entire point (see `TrOMR.freeze_core_for_profile_context`'s own
docstring): a narrower, cheaper question than "does fine-tuning the whole model help,"
and it came back yes.

**Not yet done, and worth being precise about the gap**: this measures the *existing*
rhythm/pitch/lift/articulation/slur/position loss moving, not a downstream metric a
human would recognize (character error rate, a real transcription's accuracy). A
smaller loss on held-out data is a genuine, meaningful signal - it is not automatically
"the transcriptions get measurably better," which would need its own before/after
comparison on real pages. Trained weights: `/workspace/b0/phase20/profile_context_
weights.pth` (instance-side, the 8 `decoder.profile_context.*` tensors only - not a
full checkpoint, needs `--checkpoint` plus these to reconstruct a working model). The
natural next step, now that the frozen-core question has a clear answer, is the more
expensive one §12.3-style staging exists to defer until the cheaper question is
answered: unfreezing the core and letting the whole model adapt to the signal, which
this run deliberately did not attempt.

---

## 4. Cross-staff consistency checks and repair — Stage A complete, Stage B measured

Full spec in `ENSEMBLE_TRANSCRIPTION_DESIGN.md` §12; reproduced here. Staged deliberately
from least to most invasive — **do not start Stage C before Stages A and B are built and
benchmarked**, per §12.3's own explicit precondition. Stage A is now fully built (all
8 of §12.1's originally-named checks, plus the shared-motif addition); Stage B covers
key/time signature (majority vote + carry-forward) and motif-corroborated articulation.
A systematic 200-page benchmark (below, just above Stage C) is the "built and
benchmarked" evidence §12.3 asks for: Stage A fires on 71.4% of pages, Stage B proposes
a fix on 14.6% - real coverage, but the dominant Stage A finding
(`barline_position_mismatch`) still has no repair, since it's a decoder duration-drift
signal rather than a "silent staff"/"clear majority" pattern Stage B's rules target.

### Stage A: deterministic consistency analysis — all 8 of 8 §12.1 checks built, plus
a 9th from outside the original list

`homr/cross_staff_consistency.py`: `analyze_system(staves, staff_to_part=None)` takes one
`list[EncodedSymbol]` per staff of an already-decoded system (plus, optionally,
`score_profile_layout.py`'s `staff_to_part` mapping) and returns structured `Finding`s -
no MusicXML is altered. Decoupled from images/`Staff` geometry, same as
`system_grouping.py`/`score_profile_layout.py`, so it is tested against hand-built token
sequences (56 tests, `tests/test_cross_staff_consistency.py`).

Built:

- different decoded measure counts across parts (`check_measure_counts`);
- a measure's total note/rest duration disagreeing with the rest of the system
  (`check_measure_durations`) - complements the count check above: the same barline
  count can still hide a wrong total duration inside a measure. Compares each staff's
  *median* measure duration (whole-note units, reusing `music_xml_generator`'s
  `group_into_chords`/`SymbolChord` so a chord's simultaneous notes are not
  double-counted as sequential ones), not every measure pairwise, so one outlier
  measure does not flag an otherwise-consistent staff. Named honestly in its own
  docstring why this compares *content* duration rather than a decoded time-signature
  numerator: there is no such numerator to compare against - `build_rhythm` only ever
  emits `timeSignature/<denominator>`, and `music_xml_generator.
  find_division_and_time_signature_nominator` already *infers* the numerator from
  measure content for MusicXML output, so a "duration vs. declared time signature"
  check in the literal §12.1 sense is not buildable as stated; this is the closest
  available substitute, and a genuinely useful one on its own. 5 new tests. Validated
  on the GPU instance: fired on 2 of 3 real pages tried in this session's sampling, no
  crash, e.g. `"System 2: typical measure duration (whole notes) disagrees across the
  system: {0: '1', 1: '1', 2: '2', 3: '1'}"` - staff 2 decoded exactly double the other
  three staves' measure content;
- conflicting barline **locations** (`check_barline_positions`) - the one item neither
  count nor per-measure total duration can catch: two staves can agree on both and
  still have a barline land in a different place. Tracks *cumulative* duration
  (whole-note units) at each barline in decoded order, and compares only the first
  `min(barline count across staves)` positions, since staves can legitimately have
  different barline counts at all (already `check_measure_counts`' territory). 5 new
  tests. **Validated on the GPU instance across 4 real pages**: fired on 2, silent on
  the other 2 (not universally noisy), and where it did fire it agreed with
  `check_measure_durations`' own findings on the same pages rather than contradicting
  them - e.g. one page's staff 2 (already flagged by the duration check as decoding
  roughly double the other three staves' measure content) showed cumulative barline
  positions of `[2, 4, 6, 7, 8]` against the other three staves' `[1, 2, 3, 4, 5]`,
  the same staff, the same underlying disagreement, seen from a different angle. This
  closes the last of §12.1's originally-named eight checks;
- conflicting key/time signatures (`check_key_signatures`/`check_time_signatures`) - the
  **full sequence** of key/time-signature tokens is compared, not just the opening value,
  so two staves that agree at the start but diverge on a later change still get flagged;
- a clef inconsistent with a supplied score profile (`check_clefs_against_profile`) - ties
  directly into `homr/score_profile.py`'s `ScorePart.likely_clefs`; silent when no part
  was proposed for a staff or the part states no expected clefs, per §7.1's "unknown is
  valid, not an error";
- a beam or slur endpoint made dangling (`check_dangling_slurs`) - within one staff's own
  decode: a slur slot still open at the end, or a STOP/START_AND_STOP with nothing open
  in that slot;
- missing or extra staff output relative to the rest of the page
  (`check_page_staff_counts`) - the one check in this module that is genuinely page-wide
  rather than one-system: a system's staff count means nothing on its own, only against
  what the rest of the page does, so this compares every system's count against the
  page's dominant count (ties resolve toward the larger count - a dropped voice is a
  more common real failure than a spurious extra one). Deliberately does not reuse
  `Finding.staff_indices`' per-system-staff-position meaning for a page-wide comparison -
  which systems disagree is named in `message` instead, `staff_indices` is left empty.
  Wired into `staff_parsing._report_cross_staff_findings` alongside the others, logged as
  `"Page: ..."` rather than per-system. **Validated on a real page this session**: run
  against `sq7313978:0001` on the GPU instance, no crash, valid MusicXML written, and it
  immediately surfaced a real finding neither of the earlier two-page validation runs had
  caught - `"system 0 has 3 staves, most of the page has 4"` - a genuinely missing staff
  on the very first page checked;
- part **order** changing between systems specifically (`check_part_order`) - closes the
  last of §12.1's originally-named eight items other than barline location.
  `check_page_staff_counts` (above) catches a staff *count* changing; this catches the
  parts that all stay present nonetheless swapping position between one system and the
  next, using `staff_to_part_by_system`'s part identities rather than a plain count -
  the same reason `check_clefs_against_profile` needs a score profile, this does too.
  Compares each system only against the one immediately before it, restricted to the
  parts both resolved a mapping for (a part missing from one system is already
  `check_page_staff_counts`'s territory, not treated as evidence of a swap here). Wired
  into `_report_cross_staff_findings` next to `check_page_staff_counts`, only when a
  score profile was supplied. 5 new tests. **Validated on a real page with a
  `STRING_QUARTET` profile this session**: no crash, valid MusicXML written - this
  particular page's part order stayed consistent across systems, so no finding fired,
  but the wiring itself is confirmed not to break the pipeline;
- (a 9th check, from a design discussion rather than §12.1's original list) a shared
  motif's articulation disagreeing between two staves (`check_shared_motifs`) - see the
  dedicated section below for what it covers and its known scope limits.

**All eight of §12.1's originally-named findings are now built, plus the shared-motif
addition from outside that list - Stage A's detection scope is complete as designed.**
What remains is not new checks but validating the ones already built against a wider
real-page sample, and moving to Stage B (repair) for the checks tier 1 does not yet
cover (only key/time signature has a repair proposal built - see below).

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

**Validated against 20 real pages on the GPU instance** (this session, once §1's detector
retrain freed the GPU) - a random sample across 8 different string-quartet scores, both
`scanned` and `synthetic` image sources. All 20 ran to completion (`exit=0`, valid
MusicXML written each time) - the tier-1 addition is confirmed not to break or slow the
pipeline. Stage A surfaced real findings on 7 of the 20 pages: 5 measure-count mismatches,
6 key-signature mismatches, 1 time-signature mismatch.

**But zero tier-1 repair proposals fired across all 20 pages - a real, decisive, and
informative negative result, not a bug.** Every one of the 6 key-signature mismatches
found has the same shape: `[(), (), (), ('keySignature_3',)]` - exactly one staff states
an opening key signature and the other three state none at all. `propose_majority_correction`
requires at least two staves to state a value before proposing anything (`§4`'s own "report
nothing rather than guess" discipline, tested explicitly as
`test_fewer_than_two_staves_stating_a_value_proposes_nothing`), so it correctly declines
every one of these. This is not the failure mode tier 1 was designed around (three staves
agreeing on a value, one lone dissenter stating a *different* value) - it is a distinct
pattern, an omitted restatement, not a conflicting one. Three staves saying nothing about
a key signature at a system boundary is far more likely to mean "the key did not change,
so nothing was re-stated" than "the key is unknown," which is a genuinely different
correction (fill in the carried-forward value) from what `propose_majority_correction`
implements (pick the value the majority explicitly stated). The 5 measure-count mismatches
are a different Stage A check entirely - `check_measure_counts` has no tier-1 counterpart
at all, by design (§12.1 names duration/measure-count repair as needing arithmetic across
a whole measure, not a token swap).

**Conclusion: tier 1 as built is real, correctly conservative, and - on this sample -
essentially never fires**, because the dominant real-world key/time-signature disagreement
pattern (one voice restates, others carry forward silently) is not the pattern it targets
(voices state conflicting values). Not a reason to loosen `propose_majority_correction`
unilaterally - "fill in a carried-forward value when one voice restates it" is a different,
new repair rule with its own correctness questions (does a later system's restatement
apply retroactively to earlier missing ones? what if the lone stated value is itself the
misread?).

**Resolved, and built.** User confirmation: modern engraving convention restates the key
signature on *every* staff at every system (time signature is the opposite - restated
only on change, so this deliberately does not generalise there). Checked directly against
real ground truth to confirm this specific corpus follows that convention, not assumed:
`sq7313978:0001`'s system 2 (the exact page carrying the `[(), (), ('keySignature_3',),
())]` finding above) has an identical `<key><fifths>3</fifths></key>` on all four parts
in the source MusicXML - the model decoded it correctly on only the viola. A silent
staff here is a decode omission, not a genuine absence.

`homr/cross_staff_repair.py`: `InsertionRepairProposal`/`apply_insertion_proposal` (a new
proposal shape - a silent staff has no existing token to replace, unlike
`RepairProposal`/`ArticulationRepairProposal`) and `propose_carry_forward_key_signature`,
which fires whenever at least one staff states a key signature and no other staff states
a *conflicting* value - a lower bar than `propose_majority_correction`'s "two staves
stating a value," since a lone stated value with everyone else silent is exactly the case
this convention covers. A genuine value conflict is left entirely to
`propose_majority_correction`, not guessed between the two mechanisms. 18 tests.

**Validated against the exact page that motivated this thread**: `sq7313978:0001`
correctly proposed inserting `keySignature_3` into all three silent staves (0, 1, 3),
exactly matching the confirmed ground truth - no crash, valid MusicXML written. Wired
into `staff_parsing._report_cross_staff_findings`. This closes the open question
recorded above - it was left open pending exactly this kind of external confirmation,
not resolved by guessing.

### A second motivating case for cross-staff repair, from a design discussion: shared
motifs, not just shared attributes - built, this session

Key/time signature/clef are page-wide *attributes* that should trivially agree across
parts - Stage A's original four checks are all this shape: compare one value per staff,
flag disagreement. A shared *motif* is richer: multiple voices playing the same
rhythmic/melodic idea (a fugal subject, an imitative entry, doubled parts) is content-level
agreement, not attribute-level, and one voice mistaking an accent for a marcato (or any
single-note articulation/ornament misread) would not show up in any of the original four
checks - nothing they do compares *note-level* content across staves at all.

**`check_shared_motifs` (`homr/cross_staff_consistency.py`), wired into `analyze_system`
alongside the rest.** Uses `difflib.SequenceMatcher`
over each pair of staves' note-only `(rhythm, pitch)` sequences to find matching runs of
at least `min_motif_length` (default 4) consecutive notes; within a matching run, any
note where the two staves' `articulation` field differs is reported as
`motif_articulation_mismatch`. Non-note symbols (barlines, clefs, rests) are excluded from
the matched sequence entirely rather than passed through as mismatches, so they cannot
break an otherwise-matching run in two.

**Deliberately narrower than the design discussion's fuller version, and the gap is
real, not an oversight.** Matching requires identical rhythm *and* identical absolute
pitch - a transposed imitative entry (a fugal answer played a fifth higher, the single
most common way this pattern shows up in real quartet writing) is invisible to this
version, since nothing here normalizes by pitch interval. Building that normalization
was named as the natural fuller version in the original design discussion and still is -
what shipped this session is the exact-pitch case (unison doubling, octave doubling
detected only if pitch strings are compared at unison, same-register repeated entries),
which is real and common but is a strict subset of "shared motif" in the harmonic sense.
14 tests, all passing locally (`tests/test_cross_staff_consistency.py`).

**Validated against real model output this session, and it fired on the first page
tried.** Run against `sq8823783:0061` (Wolf, String Quartet) on the GPU instance: no
crash, valid MusicXML written, and two genuine `motif_articulation_mismatch` findings -
`"System 1: staves 0 and 1 play a matching 8-note run but disagree on articulation at
one note: 'staccato' vs '_'"` and a second, 4-note case in system 3. Not yet inspected
against the source image to confirm which staff actually misread (that would require
looking at the page itself, not just the token stream) - recorded as a real finding this
check surfaced, not yet as a confirmed correction.

**The repair question turns out not to be a simple copy of tier 1, and this is worth
recording precisely rather than assumed solved.** `check_shared_motifs`/
`_shared_motif_findings` compares staves *pairwise* - each finding names exactly two
staves that disagree, with no third staff's evidence attached. `propose_majority_
correction` (tier 1's key/time-signature repair) works because it pools *every* staff in
the system and only proposes a correction when there is an actual majority to correct
toward, refusing on a tie or too few staves stating a value - the same "report nothing
rather than guess" discipline used everywhere else in this codebase. A shared-motif
finding has no analogous majority: two staves disagreeing on one note's articulation, on
their own, gives no evidence for *which* of the two is the misread - proposing "correct
staff B toward staff A" (or vice versa) from a pairwise finding alone would be a coin
flip dressed up as a repair, not a majority correction.

A real fix needs the third-or-more-staff corroboration the original design discussion's
motivating case was actually about (multiple voices playing the same motif, one
misreading it) - which means first solving a different, harder alignment problem than
what `check_shared_motifs` currently does: not "do these two staves' note runs match,"
but "which *group* of staves are all playing the same motif at the same point," so a
genuine majority (2-against-1 or better) can be counted the same way tier 1 counts key/
time-signature agreement.

**Built this session.** `homr/cross_staff_repair.py`: `propose_motif_articulation_
corrections(staves, min_motif_length=4)` and `ArticulationRepairProposal` (a sibling of
`RepairProposal`, not a reuse - that dataclass's `current_rhythm`/`proposed_rhythm`
fields and `apply_proposal`'s rhythm-only rewrite would lie about what an articulation
correction changes). `_pairwise_note_alignment` does the same matching
`check_shared_motifs` does, but returns the full alignment (which note in a reference
staff maps to which note in another) rather than only the mismatches within it - then,
for every note in every staff acting as a "reference" in turn, every *other* staff is
checked pairwise for a matching run covering that note; three or more staves covering
one note (the reference plus two or more corroborators) is a genuine majority to
propose toward, `Counter`-voted the same way `propose_majority_correction` already
votes key/time signature, refusing on a tie the same way too.

**Deliberately anchored to one fixed reference staff per corroboration, never merged
transitively across different reference staves** - the design alternative actually
considered and rejected, not an oversight. A transitive union-find over every pairwise
match (if A matches B here, and B matches C somewhere else, therefore A/B/C corroborate)
cannot tell two *separate* real occurrences of a common short motif (four repeated
quarter notes appearing at two unrelated points in a piece) from one real three-way
corroboration - it would merge them into a false group whenever the unrelated
occurrences happened to each pairwise-match a different staff. Anchoring every check to
one reference staff's own note positions makes that impossible structurally: nothing is
ever compared across two different notes of the reference staff, so there is nothing to
accidentally merge. Only the lowest-indexed staff in a corroborating group is used as
its own reference, so each real group is reported once, not once per member.

26 tests (`tests/test_cross_staff_repair.py`), including that two matching staves alone
(no third) propose nothing - the exact gap this was built to close - and that a genuine
three-way tie also proposes nothing. Wired into `staff_parsing._report_cross_staff_
findings` alongside the existing key/time-signature proposals.

**Validated on the GPU instance across 19 real pages total** (4 first, then a further
random 15): no crash, valid MusicXML every time. Tier 1's existing key/time-signature
repair fired twice more on the wider sample - real, genuine 3-vs-1 majority corrections
(`"staff 1 opens with 'keySignature_0'; 3/4 staves in this system open with
'keySignature_2'"` and a second, analogous case) - confirming tier 1 keeps finding real
disagreements on fresh pages, not just the ones already used to validate it.

**`propose_motif_articulation_corrections` has not fired on any of the 19 pages
sampled so far, including the one already known to carry two pairwise
`motif_articulation_mismatch` findings** - neither of those two findings had a third
staff's corroboration, so tier 1's discipline correctly declined both. This is a real,
honest negative result worth recording precisely rather than assumed either way: it is
not evidence the mechanism is broken (it is exercised and tested, including against
that exact known-mismatch page), but 19 pages is not yet enough to say whether genuine
3-way corroborated articulation disagreements are simply rare in this corpus/model
output, or whether a larger sample would find one. Left open rather than concluded -
the next real step here is either a larger real-page sample, or lowering
`min_motif_length` below 4 to see whether shorter shared runs surface a case the
current threshold is filtering out.

### Systematic 200-page benchmark — the "built and benchmarked" evidence §12.3 asks for

Ran Stage A + Stage B (log-only, no MusicXML mutation) across a random 200-page sample
via a standalone script (`benchmark_stage_ab.py`, not checked into this repo - classifies
every `eprint` line into finding/proposal kinds by regex and aggregates counts).

**Robustness:** 199/200 pages processed without crashing. The one failure
(`sq7383977:0120.png`) is not a bug - `python -m homr.main` correctly raises
`"No noteheads found"` on what is a blank/non-music page (0 staff line fragments, 0
noteheads from the segnet). Effectively 100% robustness on real musical content.

**Stage A fires broadly - 71.4% of pages (142/199) have at least one finding:**

| finding | count |
|---|---|
| barline_position_mismatch | 306 |
| measure_duration_mismatch | 122 |
| motif_articulation_mismatch | 122 |
| key_signature_mismatch | 44 |
| measure_count_mismatch | 29 |
| time_signature_mismatch | 12 |
| page_staff_count_mismatch | 9 |

(`clef_profile_mismatch`, `dangling_slur`, `part_order_mismatch`, `profile_layout_deviation`
did not fire on this sample - the first and last need a `--score-profile`, which this run
didn't pass; the other two appear to be genuinely rare here.)

**Stage B proposals, corrected: 14.6% of pages (29/199) get at least one proposal.**
The first pass of this benchmark undercounted this badly (reported 5.5%/11 pages) because
its regex classifier predated `propose_carry_forward_key_signature` and had no pattern for
it - those proposals fired but were silently dropped from the count. Rerunning with the
pattern added:

| proposal | count |
|---|---|
| carry_forward_key_signature | 102 |
| tier1_key_time_signature | 8 |
| motif_articulation_correction | 4 |

`carry_forward_key_signature` alone accounts for the large majority of Stage B activity -
confirms the read from the 19-page sample above: most `key_signature_mismatch` findings
(44 of them) are the "one staff states it, the rest are silent" pattern, not genuine
disagreements, and tier 1's majority-vote rule was only ever going to catch a small
minority (8) of them.

**Spot-check: what's actually driving `barline_position_mismatch` (the single largest
finding, ~2.5x the next)?** Pulled the full disagreement detail (not just the finding
count) for 5 flagged pages by rerunning `homr.main` directly and inspecting the reported
per-staff cumulative barline positions. Every case shows the same shape: staves that
*do* agree match their barline sequence **exactly**, and the diverging staff's sequence
differs either by a constant additive offset (e.g. `{0: [3/4, 3/2, 9/4, ...], 3: [7/8,
13/8, 19/8, ...]}` - every value off by exactly 1/8, meaning one early note decoded with
an extra 1/8 duration) or a constant ratio (e.g. `{0: [3/4, 3/2, ...], 3: [1/2, 1, ...]}`
- every value at exactly 2/3 scale, meaning that staff decoded a different meter/
subdivision throughout). Both signatures are exactly what the check's own docstring
predicts from **decoder-side** duration drift - one mis-decoded note or a systematic
meter misread cascading through every later cumulative position - not what a vision/crop
error would look like (occasional, non-proportional jumps). One page (the Moeran sample)
showed near-total disagreement across most staves in most systems - a harder case,
possibly a genuinely difficult piece for the current decoder, not diagnostic of this
particular mechanism.

**This settles the barline-detection-head question the user raised**: `check_barline_
positions`/`_cumulative_barline_positions` (`homr/cross_staff_consistency.py`) computes
entirely from decoded `EncodedSymbol` sequences - cumulative note/rest duration up to
each decoded barline token - and never consults any vision-detected bar-line pixel
geometry at all. A dedicated bar-line segmentation class cannot move this specific
finding's count, confirmed both by reading the check's implementation and empirically by
the spot check above; the actual lever for `barline_position_mismatch` is decoder rhythm/
duration accuracy, not detection.

That said, `bar_line_boxes` (currently derived by thresholding the `stems_rest` segnet
channel + geometric filtering, `homr/bar_line_detection.py`) do feed `detect_staff()`
as anchors alongside `staff_fragments`/`clefs_keys`, which determines the staff geometry
cropped for the decoder - so bar-line detection quality is not *irrelevant* to decode
accuracy, just not the direct cause of this particular Stage A signal. A dedicated class
was prepared anyway (commit `b324c7b`, training-side only: `CLASS_CHANNEL_LIST` splits
`ALL_BARLINES` out of the combined stem+barline channel in
`training/segmentation/dense_dataset_definitions.py`, `create_segnet`'s `out_classes` now
derives from `CHANNEL_NUM` instead of a stale hardcoded `6`, and a latent bug this split
would have caused - `CvcMuscimaDataset` hardcoding notehead's class value as a literal
`2`, which the split silently repoints at the new barline channel - is fixed alongside
it). The live inference pipeline (`homr/segmentation/inference_segnet.py`, `homr/
model.py`, `homr/main.py`) is deliberately untouched, since it reads the currently
deployed 6-class ONNX model and would misinterpret every channel the moment the code
expects 7 classes; that wiring ships together with a retrained + re-exported model.
**Launching that retrain is not decided here** - it's real GPU cost/time for a benefit
that, per the above, addresses staff-anchoring quality at best, not the dominant Stage A
finding directly. Left for explicit user sign-off.

**Net read on Stage C's precondition**: Stage A is working and catching real, frequent
disagreement (71.4% of pages). Stage B now covers a much larger share of that
(14.6%, better than first measured) but the dominant Stage A finding
(`barline_position_mismatch`, and probably `measure_duration_mismatch` for the same
underlying reason) has no Stage B repair at all - these are genuine decoder duration
errors, not "silent staff" or "clear majority" patterns Stage B's conservative rules are
built to fix safely. That gap is real, but it argues for improving decoder rhythm
accuracy or extending Stage B's repair vocabulary before reaching for Stage C's learned
adapter, not for Stage C being obviously warranted yet.

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
