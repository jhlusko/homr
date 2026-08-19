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
| 2. Optional score-profile conditioning | **Not started** — §2 below |
| 3. System grouping after segmentation | Done (`homr/system_grouping.py`) |
| 4. Cross-staff consistency checks and repair | **Not started** — §3 below |
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
4. **A confusion matrix for SystemText specifically** — 27.92 left "SystemText may be
   harder to separate from the visually similar StaffText class than Fingering is from
   anything else" as an unconfirmed hypothesis. Worth checking before deciding whether the
   fold-into-StaffText decision (27.93) should ever be revisited.
5. **`train_detector.py` has no per-epoch checkpointing** (27.91) — a real gap independent
   of the above: a long run that plateaus early cannot be stopped without losing all
   progress. Worth fixing before the next multi-hour detector run, the same problem 27.88
   hit for the recognizer.

---

## 2. Fingering/SystemText's corpus-level rarity (sharpest edge of §1)

Kept separate from §1 because the fix shape is different: this is not a training-recipe
problem on the existing corpus, it is that the *source* corpus contains 12 (Fingering)
and 3 (SystemText, pre-fold) ground-truth boxes total across the entire 2,542-page
training set (27.68) — no amount of resampling within that data can manufacture signal
the corpus does not contain.

- SystemText: folded into StaffText (§1.3 above) — settled, not reopened without new
  evidence from the confusion-matrix check in §1's item 4.
- Fingering: synthesis (§1) is the only lever that has moved it off zero, and items 1–2
  in §1's "not yet tried" list are exactly what would make that lever usable without
  collateral damage. There is no separate Fingering-only next step beyond what §1 already
  lists — resolving §1 resolves this.

---

## 3. Score-profile conditioning — not started

Full contract already specified in `ENSEMBLE_TRANSCRIPTION_DESIGN.md` §7; reproduced
here so this file is self-contained.

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

### First implementation step (not yet started)

None of §7's contract has an implementation. The smallest defensible first slice, per
§24's own "recommended first implementation slice" ordering: the schema and a
deterministic layout-scoring use (§7.2) — reading a supplied profile and reporting the
evidence/deviation fields — before touching the decoder conditioning (§7.3), which needs
the zero-gate discipline and a training-time dropout schedule (§7.4) to be safe to turn
on.

---

## 4. Cross-staff consistency checks and repair — not started

Full spec in `ENSEMBLE_TRANSCRIPTION_DESIGN.md` §12; reproduced here. Staged deliberately
from least to most invasive — **do not start Stage C before Stages A and B are built and
benchmarked**, per §12.3's own explicit precondition, which has not yet been met.

### Stage A: deterministic consistency analysis (not started)

After independent per-staff decoding, align measures within a proposed system and emit
structured findings — no MusicXML is altered:

- different decoded measure counts across parts;
- conflicting barline locations;
- conflicting key/time signatures;
- one voice's measure duration disagreeing with the meter and the other voices;
- a clef inconsistent with both the image and a supplied score profile;
- part order changing between systems;
- missing/extra staff output;
- a beam or slur endpoint made dangling by a structural edit.

### Stage B: targeted repair proposals from existing alternatives (not started, depends on A)

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
