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

## Contents

- [1. The text detector's page-level precision has collapsed for five of seven classes — **CLOSED 2026-08-20, user decision**](#1-the-text-detectors-page-level-precision-has-collapsed-for-five-of-seven-classes-closed-2026-08-20-user-decision)
  - [Where it stands](#where-it-stands)
  - [Why this happened (§27.86–27.87, `ENSEMBLE_TRANSCRIPTION_DESIGN.md`)](#why-this-happened-27862787-ensembletranscriptiondesignmd)
  - [What has already been tried](#what-has-already-been-tried)
  - [Not yet tried](#not-yet-tried)
- [2. Fingering's corpus-level rarity — **CLOSED 2026-08-20, user decision, same as §1**](#2-fingerings-corpus-level-rarity-closed-2026-08-20-user-decision-same-as-1)
- [3. Score-profile conditioning — both training runs complete; frozen-core result stands, unfrozen follow-up did not improve on it](#3-score-profile-conditioning-both-training-runs-complete-frozen-core-result-stands-unfrozen-follow-up-did-not-improve-on-it)
  - [Built so far](#built-so-far)
  - [Contract](#contract)
  - [Use in layout](#use-in-layout)
  - [Use in staff recognition](#use-in-staff-recognition)
  - [Training: context dropout](#training-context-dropout)
  - [Why this is next after §1, not before](#why-this-is-next-after-1-not-before)
  - [Next implementation step](#next-implementation-step)
  - [`phase20`: the training run — complete, 10/10 epochs positive](#phase20-the-training-run-complete-1010-epochs-positive)
  - [`phase21`: the unfrozen follow-up — complete, and it reverses phase20's read](#phase21-the-unfrozen-follow-up-complete-and-it-reverses-phase20s-read)
- [4. Cross-staff consistency checks and repair — Stage A complete, Stage B measured](#4-cross-staff-consistency-checks-and-repair-stage-a-complete-stage-b-measured)
  - [Stage A: deterministic consistency analysis — all 8 of 8 §12.1 checks built, plus](#stage-a-deterministic-consistency-analysis-all-8-of-8-121-checks-built-plus)
  - [Wired into the live pipeline, log-only, validated on real pages](#wired-into-the-live-pipeline-log-only-validated-on-real-pages)
  - [Stage B: targeted repair proposals from existing alternatives (tier 1 built for key/time signature)](#stage-b-targeted-repair-proposals-from-existing-alternatives-tier-1-built-for-keytime-signature)
  - [A second motivating case for cross-staff repair, from a design discussion: shared](#a-second-motivating-case-for-cross-staff-repair-from-a-design-discussion-shared)
  - [Systematic 200-page benchmark — the "built and benchmarked" evidence §12.3 asks for](#systematic-200-page-benchmark-the-built-and-benchmarked-evidence-123-asks-for)
  - [Stage C: learned variable-staff context adapter (not started, blocked on A+B being measured)](#stage-c-learned-variable-staff-context-adapter-not-started-blocked-on-ab-being-measured)
  - [Why this is last](#why-this-is-last)
- [5. Decoder rhythm/duration accuracy — corrected 2026-08-21 for a second ground-truth bug (movement splicing); Beethoven-shaped is now the plurality (34/75, ~45%), reversing the prior ~20%/~65% read](#5-decoder-rhythmduration-accuracy-corrected-2026-08-21-for-a-second-ground-truth-bug-movement-splicing-beethoven-shaped-is-now-the-plurality-3475-45-reversing-the-prior-2065-read)
  - [Phase 1 started 2026-08-21: decode-time cross-staff-consistency reranking, built and validated, 20-page result already positive (31→16 findings)](#phase-1-started-2026-08-21-decode-time-cross-staff-consistency-reranking-built-and-validated-20-page-result-already-positive-3116-findings)
  - [Phase 2 started 2026-08-21: time-signature conditioning + a ground-truth-supervised duration loss, training run launched](#phase-2-started-2026-08-21-time-signature-conditioning-a-ground-truth-supervised-duration-loss-training-run-launched)
- [5. (prior) Decoder rhythm/duration accuracy — final: real decoder divergence confirmed, ~20% Beethoven-shaped (Phase 1's target), ~65% Moeran-shaped (broadly poor decode, Phase 1 alone won't fix)](#5-prior-decoder-rhythmduration-accuracy-final-real-decoder-divergence-confirmed-20-beethoven-shaped-phase-1s-target-65-moeran-shaped-broadly-poor-decode-phase-1-alone-wont-fix)
- [6. IMSLP corpus expansion beyond OLiMPiC's own 200 manually-annotated scores — download complete, automated detection built, review tooling built](#6-imslp-corpus-expansion-beyond-olimpics-own-200-manually-annotated-scores-download-complete-automated-detection-built-review-tooling-built)
- [7. Stage 2 & Stage 3 training-data extraction from the expanded IMSLP corpus — scoped, not started](#7-stage-2-stage-3-training-data-extraction-from-the-expanded-imslp-corpus-scoped-not-started)
  - [Stage 2: real-scan training pairs from the bar-count-confirmed matches](#stage-2-real-scan-training-pairs-from-the-bar-count-confirmed-matches)
  - [Stage 3: real lyrics/dynamics text-region ground truth from the scans](#stage-3-real-lyricsdynamics-text-region-ground-truth-from-the-scans)
  - [Update, 2026-08-24 overnight: real pairing fix, OCR-first built and validated, scope corrected to 472 scores](#update-2026-08-24-overnight-real-pairing-fix-ocr-first-built-and-validated-scope-corrected-to-472-scores)
  - [Update, 2026-08-24 (later still): Stage 2's pair-extraction script - built, tested, and validated on real data](#update-2026-08-24-later-still-stage-2s-pair-extraction-script---built-tested-and-validated-on-real-data)
  - [Update, 2026-08-25: full-scale Stage 2 extraction run complete - 2,535 real training pairs](#update-2026-08-25-full-scale-stage-2-extraction-run-complete---2535-real-training-pairs)
  - [Update, 2026-08-25: a real user-found bug in the extracted pairs, root-caused and fixed](#update-2026-08-25-a-real-user-found-bug-in-the-extracted-pairs-root-caused-and-fixed)
  - [Update, 2026-08-25: review sites built - a Stage 2 pair reviewer and a Stage 3 text reviewer, merged into one server](#update-2026-08-25-review-sites-built---a-stage-2-pair-reviewer-and-a-stage-3-text-reviewer-merged-into-one-server)
  - [Update, 2026-08-25: a real multi-verse lyrics bug found via the review site, root-caused and fixed](#update-2026-08-25-a-real-multi-verse-lyrics-bug-found-via-the-review-site-root-caused-and-fixed)
  - [Update, 2026-08-25: THE root cause - a one-measure off-by-one in every system range in the whole corpus](#update-2026-08-25-the-root-cause---a-one-measure-off-by-one-in-every-system-range-in-the-whole-corpus)
  - [Update, 2026-08-25: what "Stage 2 training" actually requires on this box](#update-2026-08-25-what-stage-2-training-actually-requires-on-this-box)
  - [Update, 2026-08-25: post-fix bar-count result, and what ENSEMBLE_TRANSCRIPTION_DESIGN.md requires of the training run](#update-2026-08-25-post-fix-bar-count-result-and-what-ensembletranscriptiondesignmd-requires-of-the-training-run)
  - [Update, 2026-08-25: recovering the excluded systems by content, and the replay decision](#update-2026-08-25-recovering-the-excluded-systems-by-content-and-the-replay-decision)
  - [Update, 2026-08-25: recovery complete, and the Stage 2 scans fine-tune is running](#update-2026-08-25-recovery-complete-and-the-stage-2-scans-fine-tune-is-running)
  - [Update, 2026-08-25: Stage 2 scans fine-tune - result, and stopped early on a plateau](#update-2026-08-25-stage-2-scans-fine-tune---result-and-stopped-early-on-a-plateau)
  - [Update, 2026-08-25: Stage 3 (text detector) - tooling built, experiment matrix started](#update-2026-08-25-stage-3-text-detector---tooling-built-experiment-matrix-started)
  - [The box has a pid limit, and onnxruntime does not respect thread hints](#the-box-has-a-pid-limit-and-onnxruntime-does-not-respect-thread-hints)
  - [E0 baseline — complete (2026-08-25)](#e0-baseline-complete-2026-08-25)
  - [E1-E3 results — real-scan data helps substantially, and E3 is the configuration to keep](#e1-e3-results-real-scan-data-helps-substantially-and-e3-is-the-configuration-to-keep)
  - [The page-level measurement contradicts the patch measurement — E3 halves precision](#the-page-level-measurement-contradicts-the-patch-measurement-e3-halves-precision)
  - [E4/E5 — the middle masking policy (running)](#e4e5-the-middle-masking-policy-running)
  - [The OSSQ scanned track is systematically mislabeled — root cause found (2026-08-25)](#the-ossq-scanned-track-is-systematically-mislabeled-root-cause-found-2026-08-25)
  - [Beams reach the output, and repeats are recovered — two gaps closed from one review session](#beams-reach-the-output-and-repeats-are-recovered-two-gaps-closed-from-one-review-session)
  - [Structured heads on the final base — beaming recovered at 95%, dynamics did not train](#structured-heads-on-the-final-base-beaming-recovered-at-95-dynamics-did-not-train)
  - [Adding the OSSQ synthetic track costs scan accuracy — measured, not assumed](#adding-the-ossq-synthetic-track-costs-scan-accuracy-measured-not-assumed)
  - [Shipping decision: two detectors, E2 for vocal and the instrumental model for the rest](#shipping-decision-two-detectors-e2-for-vocal-and-the-instrumental-model-for-the-rest)
  - [Head-to-head against upstream homr — the number the whole effort is judged on](#head-to-head-against-upstream-homr-the-number-the-whole-effort-is-judged-on)
  - [The instrumental detector — the "without lyrics" half of the two-model split](#the-instrumental-detector-the-without-lyrics-half-of-the-two-model-split)
  - [The clef-corrected continuation — plateaued, and the sequence of runs so far](#the-clef-corrected-continuation-plateaued-and-the-sequence-of-runs-so-far)
  - [A second, independent data bug: 2.4% of staves had no clef — found by eye, invisible to every metric](#a-second-independent-data-bug-24-of-staves-had-no-clef-found-by-eye-invisible-to-every-metric)
  - [Training on the corrected corpus — first numbers, and what they do not yet prove](#training-on-the-corrected-corpus-first-numbers-and-what-they-do-not-yet-prove)
  - [The OSSQ fix, and its verification (2026-08-25)](#the-ossq-fix-and-its-verification-2026-08-25)
  - [E4/E5: the middle masking policy did not work, and the prediction it was built on failed](#e4e5-the-middle-masking-policy-did-not-work-and-the-prediction-it-was-built-on-failed)
  - [OSSQ instrumental text extraction — complete (2026-08-25)](#ossq-instrumental-text-extraction-complete-2026-08-25)
  - [Building the best Stage 2 model: what the artifacts actually are, and the order to do it in](#building-the-best-stage-2-model-what-the-artifacts-actually-are-and-the-order-to-do-it-in)
  - [Stage 2 renders and the review site (2026-08-25)](#stage-2-renders-and-the-review-site-2026-08-25)
  - [Packaging the corpora and models for distribution](#packaging-the-corpora-and-models-for-distribution)
- [Structured heads in production, and the refinement UI](#structured-heads-in-production-and-the-refinement-ui)
  - [The chain is built at both ends and missing in the middle](#the-chain-is-built-at-both-ends-and-missing-in-the-middle)
  - [What ships as a choice, and what only ships](#what-ships-as-a-choice-and-what-only-ships)
  - [Surfacing: threshold-gated, not always-on](#surfacing-threshold-gated-not-always-on)
  - [Why `/v1/regenerate` is the right seam](#why-v1regenerate-is-the-right-seam)
  - [~~Still unproven~~ Resolved 2026-08-25: the heads export cleanly](#still-unproven-resolved-2026-08-25-the-heads-export-cleanly)
  - [What is now the real blocker](#what-is-now-the-real-blocker)

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

## 3. Score-profile conditioning — both training runs complete; frozen-core result stands, unfrozen follow-up did not improve on it

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

### `phase21`: the unfrozen follow-up — complete, and it reverses phase20's read

Same corpus, same held-out validation split, same with/without-profile ablation
methodology as phase20 - only the freeze policy differs
(`TrOMR.unfreeze_decoder_for_profile_context`: encoder frozen, whole decoder trainable,
150 tensors instead of phase20's 8), so unfreezing is the one variable that changed.
Conservative LR (1e-5, two orders of magnitude below phase20's 1e-3 probe), full model
checkpointed every epoch (`/workspace/b0/phase21/checkpoints/epoch_{1..5}.pth`).

```
epoch   train loss   valid with   valid without   delta
1       1.7829       1.0848       1.0840          -0.0008
2       1.6330       1.0628       1.0634          +0.0006
3       1.5849       1.0532       1.0539          +0.0007
4       1.5542       1.0509       1.0490          -0.0018
5       1.5344       1.0429       1.0416          -0.0013
```

For comparison, phase20's ablation delta by epoch: +0.0664, +0.0682, +0.0727, +0.0735,
+0.0353, +0.0565, +0.0700, +0.0676, +0.0636, +0.0409 - positive on all 10 epochs, never
closer to zero than +0.0353.

**Absolute loss dropped sharply and steadily** (1.78 → 1.53 over 5 epochs, versus
phase20's ~1.98-2.05 range) - expected and healthy for an unfrozen fine-tune actually
adapting to the training distribution, not a regression signal. **But the
profile-context delta itself collapsed to noise**: all 5 epochs land within ±0.002 nats
of zero, oscillating sign, an order of magnitude below phase20's smallest value
(+0.0353) and never once matching its direction consistently the way phase20 did on
every single epoch.

**Verdict: unfreezing did not improve on phase20's result - it erased the measurable
part of it.** This is not the core-competence regression this run was watching for
specifically (that would show up as `without`-profile loss getting *worse* than
phase20's own numbers at the same epoch; instead it dropped by almost half, a genuine
improvement in the model's general fit). What collapsed is narrower and more specific:
the *marginal* value of telling the decoder about a staff's instrument/clef/part
explicitly, over and above what the decoder can already infer once every one of its
150 tensors is allowed to adapt to the same training data profile context was built
from. Once the core can adapt, it appears to absorb whatever profile context was
supplying into its general weights, making the explicit signal redundant rather than
additive - a real, coherent finding, just not the one this experiment was hoping for.

**Recommendation**: prefer phase20's frozen-core checkpoint
(`/workspace/b0/phase20/profile_context_weights.pth`) over phase21's for any decision
that follows from this track - phase20 is the run with a clear, positive, held-out
result; phase21's checkpoints remain on disk
(`/workspace/b0/phase21/checkpoints/epoch_{1..5}.pth`) if there is reason to inspect
them further, but nothing in this run's own numbers argues for preferring it. Whether
score-profile conditioning is worth deploying at all now rests on phase20's frozen-core
result alone, with the understanding that its ceiling (a training-loss-only, no
downstream-metric signal - see this section's own earlier caveat) has not moved.

---

## 4. Cross-staff consistency checks and repair — Stage A complete, Stage B measured

Full spec in `ENSEMBLE_TRANSCRIPTION_DESIGN.md` §12; reproduced here. Staged deliberately
from least to most invasive — **do not start Stage C before Stages A and B are built and
benchmarked**, per §12.3's own explicit precondition. Stage A is now fully built (all
8 of §12.1's originally-named checks, plus the shared-motif addition); Stage B covers
key/time signature (majority vote + carry-forward), motif-corroborated articulation,
and a majority-corroborated localization (not a content repair) for barline-position
divergence. A systematic 200-page benchmark (below, just above Stage C) is the "built
and benchmarked" evidence §12.3 asks for: Stage A fires on 71.4% of pages, Stage B
proposes something on 44.2% - real, substantial coverage, but the dominant Stage A
finding (`barline_position_mismatch`) still has no *content* repair for most
occurrences, since it's a decoder duration-drift signal rather than a "silent
staff"/"clear majority" pattern Stage B's rules can safely fix outright.

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
only on change, so this deliberately does not generalise there).

**Correction (found later, see §5): the "checked directly against real ground truth"
claim below was wrong.** `sq7313978:0001.musicxml` (read at the time as this page's
ground truth) is actually `homr.main`'s own prior output, not the real corpus source -
confirmed via its own `<encoding><software>homr</software></encoding>` metadata; the
real ground truth lives at the piece's top level instead (`sq7313978.musicxml`), not
checked here. The rule itself doesn't depend on this one page's confirmation to be
sound - "restate the key signature on every staff" is a well-established, near-universal
engraving convention independent of any one corpus check - but the specific claim that
this corpus's own data was verified to follow it is unsupported by what was actually
read. ~~Checked directly against real ground truth to confirm this specific corpus
follows that convention, not assumed: `sq7313978:0001`'s system 2 (the exact page
carrying the `[(), (), ('keySignature_3',), ())]` finding above) has an identical
`<key><fifths>3</fifths></key>` on all four parts in the source MusicXML - the model
decoded it correctly on only the viola. A silent staff here is a decode omission, not a
genuine absence.~~

`homr/cross_staff_repair.py`: `InsertionRepairProposal`/`apply_insertion_proposal` (a new
proposal shape - a silent staff has no existing token to replace, unlike
`RepairProposal`/`ArticulationRepairProposal`) and `propose_carry_forward_key_signature`,
which fires whenever at least one staff states a key signature and no other staff states
a *conflicting* value - a lower bar than `propose_majority_correction`'s "two staves
stating a value," since a lone stated value with everyone else silent is exactly the case
this convention covers. A genuine value conflict is left entirely to
`propose_majority_correction`, not guessed between the two mechanisms. 18 tests.

**Validated against the exact page that motivated this thread**: `sq7313978:0001`
correctly proposed inserting `keySignature_3` into all three silent staves (0, 1, 3) -
no crash, valid MusicXML written (this part is still true regardless of the ground-truth
mixup above - it's a direct observation of the proposal firing correctly, not a
ground-truth comparison). ~~exactly matching the confirmed ground truth~~ - see the
correction above; "matching ground truth" specifically was not actually established.
Wired into `staff_parsing._report_cross_staff_findings`. The convention-following
behavior itself is still a reasonable, low-risk rule on the strength of the convention
being near-universal, just not corpus-verified the way this section originally claimed.

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

**`propose_majority_position_corrections` (built after this benchmark, closing part of
the gap below):** the barline spot check's finding - divergences show either a constant
additive offset or constant ratio between staves, never occasional non-proportional
jumps - is exactly the kind of clean, majority-corroborated signal the rest of this
module already knows how to act on, just with one difference: there is no single
low-ambiguity token to correct here the way a key/time-signature or articulation value
is. So this proposal *localizes* (which staff, which measure, how large an offset) and
deliberately stops there - no `apply_*` counterpart, the same "propose, never guess"
discipline extended to a case where guessing the actual content edit is not defensible.
Requires a genuine 3+ staff majority (no tie) and a constant offset from the first
divergent barline onward; a non-constant offset (a messier disagreement) or a
constant-*ratio* case (a different meter reading entirely, seen on one of the 5
spot-checked pages) declines to propose, for the same reason. Validated against a real
flagged page (Beethoven Op.133 p.13): correctly fired on two genuine constant-offset
divergences and declined the ratio case. 5 new tests, `tests/test_cross_staff_repair.py`.

**Re-benchmarked with `propose_majority_position_corrections` included: Stage B proposal
rate jumps from 14.6% to 44.2% of pages (88/199).** Full corrected proposal counts on
the same 200-page sample:

| proposal | count |
|---|---|
| carry_forward_key_signature | 102 |
| majority_position_correction | 91 |
| tier1_key_time_signature | 8 |
| motif_articulation_correction | 4 |

91 occurrences against 306 `barline_position_mismatch` findings means roughly **30% of
that single largest Stage A finding now gets at least a diagnostic localization** (which
staff, which measure, how large an offset) - real, useful progress, but still not a
*content* repair: these remain genuine decoder duration errors with no low-ambiguity
edit to propose, by design (see `propose_majority_position_corrections`' own docstring
above for why). The remaining ~70% either lack a clean 3+ staff majority, show a
non-constant offset, or show a constant-ratio (different-meter) pattern - all cases this
rule correctly declines rather than guesses at.

**Net read on Stage C's precondition, updated 2026-08-21 - the gap below is now
substantially closed, by the exact route this section itself pointed to.** Stage A is
working and catching real, frequent disagreement (71.4% of pages). Stage B now covers
a much larger share of that (44.2%, up from the first benchmark's undercounted 5.5%) -
genuine, measured progress - but the paragraph below originally said the dominant
Stage A finding "still only gets localization, not a content fix," arguing for
"improving decoder rhythm accuracy... before reaching for Stage C's learned adapter."
**That is exactly what §5's Phase 1 (decode-time cross-staff-consistency reranking) now
does.** Built, live-wired (`homr/staff_parsing.py`'s `parse_staffs`, on by default), and
measured: a 200-page benchmark showed a 20.8% reduction in combined
`barline_position_mismatch`/`measure_duration_mismatch` findings with zero regressions,
and a ground-truth spot-check across 61 changed pages found 38 of 40 resolvable
corrections now match real ground truth exactly (0 regressions there either). Phase 1
does not close the whole gap - it only fires where Stage A already flags a
≥3-staff-corroborated disagreement, and §5's own content-level breakdown still finds a
genuine Moeran-shaped minority (~31%) where the majority itself isn't reliable enough to
rerank against - but it is real, validated, deployed content correction for the
majority-corroborated share of exactly the finding this paragraph used to say had none.
**Original paragraph, kept for the history but superseded by the above:** "the dominant
Stage A finding still only gets localization, not a content fix, for the same reason it
never will without guessing: these are genuine decoder duration errors, not 'silent
staff' or 'clear majority' patterns Stage B's conservative rules can safely fix
outright. That gap is real, but it argues for improving decoder rhythm accuracy or
extending Stage B's repair vocabulary further before reaching for Stage C's learned
adapter, not for Stage C being obviously warranted yet." The conclusion (Stage C not
obviously warranted yet) still stands, on firmer footing than before - the smaller,
cheaper intervention this section named as the alternative has now actually been tried
and shown to work, not just proposed.

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

**Refinement, 2026-08-21 (user design discussion): should this run as a second pass,
conditioned on each staff's actual *decoded* content, rather than only pooled visual
encoder output?** As written above, `h_i` comes from the shared visual encoder alone -
computed before anything has been decoded, so the cross-staff context every staff
receives is purely "what the neighboring staves' images look like," never "what the
neighboring staves actually contain musically" (rhythm/pitch/duration). The user's
proposal: decode every staff once (a first pass), build cross-staff context from that
*decoded* content instead of (or in addition to) pooled image features, then redecode
each staff with that richer signal - the same "decode, then use cross-staff
information to improve the result" shape Phase 1's reranking already uses, but with a
proper learned second decode instead of discrete reranking among a handful of forked
candidates.

**This is a real, well-motivated refinement, and one the original design already
anticipated and deliberately deferred, not overlooked**: the line directly above -
"do not build a richer version (exchanging system-position features or decoded
measure summaries) before this one is measured" - names almost exactly this second-pass
idea as a *later* elaboration, on the reasoning that the simpler visual-only version
should be measured first before adding the complexity of conditioning on decoded
content. That ordering argument still has real weight (isolate whether cross-staff
context helps at all before asking whether richer content-aware context helps more) -
but it was written before Phase 1 existed. Phase 1's own result (a 200-page benchmark
showing a 20.8% reduction in cross-staff findings from reranking on *decoded content
alone*, then a 61-page ground-truth spot-check confirming 38 of 40 resolvable
corrections exactly matched truth) is now direct, measured evidence that decoded
content - not just visual similarity - is a strong cross-staff signal in this specific
domain. That evidence didn't exist when Stage C's "measure the simple version first"
ordering was decided, and arguably weakens the case for starting with the visual-only
version: we already know a decoded-content signal works well post-hoc (Phase 1); Stage
C's open question is really whether a *learned, jointly-trained* version of roughly
that same signal (rather than discrete post-hoc reranking) works better still - which
argues for starting the architecture closer to what's already validated, not further
from it.

**Not decided here - a real design choice with an argument on each side, worth
resolving explicitly before Stage C's first line of code, not silently defaulting to
the older ordering:** (a) build the visual-only first pass as originally specified,
to isolate cross-staff context's value in the cleanest possible ablation, accepting
that Phase 1 has already somewhat pre-answered "does cross-staff signal help" and this
version mostly asks "does image-level context specifically help"; or (b) start directly
with the second-pass, decoded-content-conditioned version, on the reasoning that it is
now the better-motivated starting point given Phase 1's result, at the cost of a
messier first ablation (a richer architecture and a two-pass training/inference
procedure, harder to isolate which part of any improvement is doing the work).

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

---

## 5. Decoder rhythm/duration accuracy — corrected 2026-08-21 for a second ground-truth bug (movement splicing); Beethoven-shaped is now the plurality (34/75, ~45%), reversing the prior ~20%/~65% read

**Second correction, layered on top of the `<page>.musicxml` bug below**: user inspection
of the corpus-review webpage (§5's own review tool, `corpus_review.html`) found the
rendered "ground truth" images were visibly wrong - traced end to end on one example
(Wolf, *String Quartet*, sq8823783, page 22, absolute measure 318). Root cause: OSSQ-OMR's
ground-truth `.musicxml` files concatenate every movement of a piece into one file, and
each movement **restarts** `<measure number="...">` at 1 (string quartets routinely have
3-4 movements). Every script in this investigation that matched ground-truth measures by
`m.get("number") == str(target)` - `deep_barline_audit_v2.py`, `content_verify_agrees.py`,
`build_review_assets.py` - treated that number as unique across the whole file. It isn't:
the same number string recurs once per movement, so a naive match either silently picks
whichever movement's measure happens to come first, or (if keeping every match, as
`build_review_assets.py` did) splices two unrelated movements' measures together into one
rendered image - exactly what produced the visibly-wrong ground truth image and the
uniform `0.0` content-overlap score on *every* part (not just the flagged one) for that
entry.

The corpus's own `measure_start`/`measure_end` alignment metadata resets the same way at
movement boundaries (confirmed by inspecting the full page sequence for the Wolf piece:
a system's metadata decreasing relative to the previous system's lines up exactly with a
`<measure number="1">` reset in the ground truth file) - so `measure_start` is
movement-local, not a piece-wide running count; a naive "flat index = measure_start - 1"
fix would only have worked by coincidence for movement 1.

**Fix** (`homr/training/omr_datasets/ossq_ground_truth.py`): `movement_index_for_system`
counts resets in the corpus's own metadata sequence (scanned/synthetic only - `unaligned`
excluded here, since one piece was found to carry a spurious duplicate-numbered page in
`unaligned` that would otherwise cause a false reset) to find which movement a page/system
belongs to; `resolve_flat_measure_range` then matches by number *only within that
movement's own slice*, where numbers are actually unique. All 8 pre-existing unit tests
still pass. Verified against two independent cases: the Wolf page 22 system 2 case
(movement 0, resolves to the expected flat measures 316-320) and a second system on the
same piece landing in movement 1 (resolves to the expected flat measures 406-414).

**Effect on the corpus-wide numbers** (rerun of `deep_barline_audit_v2.py`, identical
200-page `benchmark_sample.txt`):

```
total majority_position_correction proposals: 91
  ground truth disagrees (known corpus defect by the invariant): 1   (unchanged)
  ground truth agrees (candidate: real decode error): 75             (was 87)
  no ground truth / no measure-mapping metadata available: 15        (was 3)
```

All 12 changed verdicts flipped `agrees → no mapping available`, never the other
direction. Checked one flip directly (Andrée, *String Quartet in A major*, sq7313978,
page 30 system 4): the piece genuinely has 4 movements (measure-flat boundaries at
0/170/322/560), and this specific system's alignment metadata exists *only* in the
unreliable `unaligned` folder - with no aligned entry to place it in the reset sequence,
guessing its movement risked exactly the Wolf-style splicing bug this fix exists to
prevent, so it now conservatively reports "no mapping" instead of a number that might be
comparing the wrong movement. This is the same discipline `measure_start_for_system`
already applies to non-numeric placeholders (`"X2"` etc.) - treat corpus ambiguity as "no
mapping," never guess.

**The 87-entry content-level breakdown below (17 Beethoven-shaped / 57 Moeran-shaped / 13
inconclusive) was stale** - built from `content_verify_agrees.py`, which had the exact
same number-matching bug. Rerun against the corrected 75-entry set
(`content_verify_agrees_v3_full.json`):

```
total agree-entries checked: 75
  Beethoven-shaped (majority overlap >=0.8, flagged staff <0.8): 34  (~45%, was 17/87 ~20%)
  Moeran-shaped (majority overlap <0.8 too): 23                     (~31%, was 57/87 ~65%)
  inconclusive/no data: 18                                          (~24%, was 13/87 ~15%)
```

**This did not just shrink the same proportions - it reversed which shape dominates.**
Beethoven-shaped went from roughly a fifth of the failure family to nearly half, and
Moeran-shaped dropped from roughly two-thirds to under a third. The mechanism makes
sense in hindsight: the movement-splicing bug's characteristic failure mode was
mixing two unrelated movements' measures into one comparison, which - like the Wolf
example that surfaced this whole fix - produces near-`0.0` overlap on *every* part,
indistinguishable from a genuinely Moeran-shaped "whole system diverges" result. A
meaningful share of the old 57 Moeran-shaped entries were very likely spliced-garbage
comparisons wearing a Moeran-shaped costume, not real evidence the model decodes badly
there. **Revised conclusion: Phase 1's beam-search reranking now has a substantially
larger, better-justified target than previously thought** - the Beethoven-shaped
plurality (34 of 75) is exactly the "one clean wrong note against a reliable majority"
signature it's designed to fix. The Moeran-shaped minority (23, ~31%) and inconclusive
share (18, ~24%) are real and still worth naming as a gap Phase 1 alone won't close, but
they are no longer the dominant story.

The `corpus_review.html` review webpage was **not** regenerated for this fix (explicit
user instruction - tracing 1-2 examples end to end was judged sufficient rather than
rebuilding all assets); its ground-truth renderings for multi-movement pieces may still
show the pre-fix splicing artifact until it is rebuilt.

### Phase 1 started 2026-08-21: decode-time cross-staff-consistency reranking, built and validated, 20-page result already positive (31→16 findings)

User instruction: "go ahead with Phase 1" - the natural next step once the corrected
numbers above gave it a substantially better-justified target (~45% Beethoven-shaped,
up from ~20%). Full build detail and live-model validation:
`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2.

Built in a deliberately narrower, cheaper shape than textbook fixed-width beam search -
real multi-hypothesis beam search would need per-step KV-cache batching this codebase
has never used (`decoder_inference.py` is batch=1 throughout), a much larger and
riskier build for the same stated goal. Instead: at each staff's *narrowest-margin*
rhythm decisions, fork a full alternate decode via the already-validated
`generate_from_prefix` mechanism, then keep whichever candidate - greedy or a fork -
best matches the other staves' majority cumulative barline positions.

New code: `homr/transformer/decoder_inference.py` (`generate_with_rhythm_margins`,
`rhythm_alternative`), `homr/cross_staff_rerank.py` (`rhythm_candidates_for_staff`,
`rerank_staff_candidates`), 6 passing unit tests, and `benchmark_phase1_rerank.py` (the
before/after benchmark itself). All validated against the live model, not mocked - see
the design doc for the four specific checks run.

**Result, 20-page sample: 31 → 16 combined `barline_position_mismatch`/
`measure_duration_mismatch` findings (~48% reduction), 11 systems improved.** A real,
clearly positive signal, confirmed on a page already traced end to end earlier in this
investigation (the reduction there, 3→2, matched a specific identifiable fix, not an
aggregate artifact).

**Full 200-page run complete: 428 → 339 findings (20.8% reduction), 81 of 899 systems
improved, zero pages made worse.** The 20.8% real number is meaningfully lower than the
20-page sample's 48% (that sample happened to concentrate more fixable cases by
chance), but it's real, substantial, and - importantly - reranking never once increased
a page's finding count across all 198 successfully-processed pages (2 of 200 hit an
unrelated, pre-existing crash already seen elsewhere in this investigation, caught and
skipped).

**Ground-truth spot-check done, and it's a clean, decisive result: 6 of 6 resolvable
corrected measures now match real ground truth exactly - zero regressions, zero cases
where reranking picked a wrong-but-self-consistent answer.** Sample: 15 pages drawn
from the 61 that had a changed system, checked with a new script
(`phase1_ground_truth_spotcheck.py`) that resolves each corrected measure's real
ground-truth duration via the fixed, movement-aware `ossq_ground_truth.py` machinery
and compares it against both the greedy and the reranked decode for that exact measure.
10 proposals had a resolvable ground-truth mapping; 4 didn't (the same known corpus-
metadata gaps this investigation has hit before) and are correctly excluded rather than
guessed at. Of the 10: **6 flipped from greedy-wrong to reranked-matches-truth exactly;
the other 4 weren't among the "changed" staves that specific run of reranking actually
touched** (i.e. not counterexamples - just proposals whose staff wasn't the one
reranking altered on that pass). Found and fixed two real bugs in the spot-check script
itself before trusting this result: divisions needing to be walked from the movement's
start (not seeded fresh at the target measure), and a whole-note-vs-quarter-note unit
mismatch between the decoder's own duration convention and `measure_length_by_part`'s -
both of the same *class* of bug this investigation has hit more than once now
(normalize before comparing, never compare raw units across two different sources).

**n=6 is small - this is a real, clean, unanimous positive signal, not yet a large-
sample statistical claim.** But unanimous-and-zero-counterexample at any sample size is
meaningfully different from "coin flip," and it directly answers this section's own
open caution (a self-consistent majority is not proof of correctness, per the Moeran
case) with actual evidence rather than leaving it as an assumption.

**Wired into the live pipeline, 2026-08-21** (user instruction: "please wire it"):
`homr/staff_parsing.py`'s `parse_staffs` now reranks for real, on by default. Gated
two-pass, not unconditional - every staff's greedy decode costs the same as before,
but a system only pays for the expensive forking+reranking pass once Stage A's own
checks already show a finding on its greedy decode (the exact population the benchmark
and spot-check measured, never a broader unvalidated "always fork everything"
behavior). Full detail, including the live end-to-end validation on both a flagged and
a clean page: `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2's own wiring section. Full test
suite still 1041/1044 (same 3 pre-existing unrelated failures, no regression).

**Spot-check extended to all 61 changed pages, 2026-08-21: the 6/6 rate holds at
scale.** 49 resolvable entries: **38 IMPROVED, 2 NEITHER_MATCHES_TRUTH, 0 REGRESSED**
(9 more had no ground-truth mapping, correctly excluded rather than guessed at). Checked
both non-matching cases by hand: in each, the *greedy* decode was already wrong too
(e.g. greedy `17/32` vs. truth `1/2`, reranked `9/16` vs. truth `1/2` - neither correct,
but reranking didn't break a right answer, it landed on a different wrong one for what
looks like a genuinely messy passage). Zero regressions across the entire sample - this
is a strong, real result, not a small-n coincidence.

Found and fixed a real bug while extending the sample, worth remembering: wiring Phase 1
into `parse_staffs` changed its internal call path (it now calls
`Staff2Score.predict_greedy_with_margins`, not `predict()`, and reranks by default) -
which broke the existing offline analysis scripts, since they monkeypatch
`Staff2Score.predict` to capture context. Every one of the first 61-page relaunch's
pages crashed with `IndexError` because the monkeypatch was no longer intercepting
anything. Fixed by passing `enable_phase1_rerank=False` explicitly in both analysis
scripts' `parse_staffs(...)` calls - restores the old call path (so the monkeypatch
fires again) and prevents the live pipeline's own default reranking from confounding
external analysis that's supposed to be doing its own independent reranking.
**General lesson: wiring a feature into a live call path can silently break offline
tooling built against the pre-wiring version of that same path - worth an explicit
check whenever a "measure this" script and a "do this live" wiring share a code path.**

### Phase 2 started 2026-08-21: time-signature conditioning + a ground-truth-supervised duration loss, training run launched

User instructions: "please prepare the training run," then "let's think more about the
loss... cross staff coherence? what other ideas..." then "yes, do both." Full detail:
`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.3.

Two mechanisms built and validated, now training together in one run:

1. **Explicit time-signature conditioning** (the §7.3 refinement from the earlier
   design discussion) - `ProfileContext`/`ProfileContextEmbedding` extended with an
   `expected_time_signature` field; real values sourced from ground-truth MusicXML per
   training sample (`score_profile_time_signature.py`, reusing today's movement-aware
   `ossq_ground_truth.py` machinery) and wired into the DataLoader automatically.
2. **Ground-truth-supervised measure-duration adherence loss** - a new, differentiable
   analogue of Stage A's own `check_measure_durations` check, penalizing the rhythm
   head's predicted expected cumulative duration for diverging from ground truth at
   each true barline. Gated by a weight defaulting to `0.0` (no effect until turned on).
   A real bug (new fixed lookup tables registered the wrong way, tripping the
   checkpoint-loader's mismatch check) found and fixed before it would even smoke-test.

**Also documented, not built**: a loss-design brainstorm ranking further candidates -
most notably a **cross-staff coherence loss** (batch sibling staves together at
training time and penalize divergence from their *ground-truth* durations - cheaper
than Stage C's full learned adapter, and a useful signal for whether Stage C's premise
is worth its cost before building it) - plus a key-signature/accidental consistency
loss and structural well-formedness losses, ranked lowest priority.

Validated: 13 new tests (8 for the time-signature sourcing, 5 for the duration loss),
full suite 1060/1060 non-deselected (same 3 pre-existing unrelated failures throughout
this investigation), and a real smoke test confirming the whole chain works end to end.

**`phase22` training run launched**: same pinned checkpoint and same 105,305/4,912
train/valid split `phase20`/`phase21` used, same hyperparameters, both new mechanisms
active together.

**Two-stage performance stall, found and fixed before the result below could be
trusted.** The first launch attempt sat at 0% GPU utilization with 4 DataLoader
workers pinned near 100% CPU for 13+ minutes with no epoch-1 output. Root cause:
`time_signature_for_sample`'s original implementation re-parsed the entire
whole-score ground-truth MusicXML (up to 6.5MB observed) via a bare `ET.parse` and
rescanned every systemwise metadata file in the piece, on *every training sample*,
with zero caching - measured at ~2000ms per successful lookup. A first fix
(`@lru_cache` on the parse and the systemwise-entries lookup) improved warm-cache
lookups to ~600ms, but a relaunch still stalled at 0% GPU with per-worker memory
climbing into multiple GB: a shuffled dataset means each worker accumulates many
large parsed trees before the cache helps, and the underlying whole-score-file
lookup is too expensive per sample regardless of caching.

The real fix was corpus-wide preprocessing, prompted by the user's own question
("should we split the examples at the page level for smaller xml?"):
`split_ground_truth_by_system.py` pre-extracts tiny, already
movement-disambiguated per-(page, system) MusicXML fragments once, corpus-wide -
10,400 fragments across 122 real pieces, in 6.8 minutes with 48-way parallelism
(vs. a ~70-minute serial estimate), made possible by discovering the GPU rental
instance actually has 128 vCPUs / 755GB RAM (`nproc`/`free -h`) against only 4
DataLoader workers previously in use - no machine change needed, just
underutilized capacity. `time_signature_for_sample` was rewritten to try this
fast fragment path first, dropping the per-sample lookup to ~2-7ms. Relaunched
with `--workers 32`: GPU utilization confirmed directly via `nvidia-smi` at
80-89% throughout the run (one transient 0% reading between epochs 3-4 was
checked and confirmed to be a checkpoint-write/worker-pool-recreation pause, not
a stall - GPU was back to 84%+ on the next check).

**Result: complete, 10/10 epochs positive.**

| epoch | mean loss | duration_adherence | valid delta (with − without profile) |
|---|---|---|---|
| 1 | 2.7787 | 0.2980 | +0.0551 |
| 2 | 2.7391 | 0.2871 | +0.0731 |
| 3 | 2.7298 | 0.2859 | +0.0439 |
| 4 | 2.7306 | 0.2854 | +0.0484 |
| 5 | 2.7328 | 0.2846 | +0.0475 |
| 6 | 2.7309 | 0.2832 | +0.0589 |
| 7 | 2.7222 | 0.2834 | +0.0639 |
| 8 | 2.7258 | 0.2837 | +0.0161 (outlier) |
| 9 | 2.7201 | 0.2828 | +0.0555 |
| 10 | 2.7218 | 0.2834 | +0.0509 |

Mean delta **+0.0513**, all 10 epochs positive. Epoch 8's dip (+0.0161, driven by
"with profile" validation loss ticking up rather than "without" moving) did not
continue into epochs 9-10 - both recovered to the same +0.04-0.07 band every other
epoch sat in, so it reads as training noise, not a real degradation.

`duration_adherence`'s own trend (a separate, less rigorous signal - it has no
with/without ablation of its own) fell from 0.2980 to ~0.283 over the first 5-6
epochs, then plateaued (0.2832-0.2837 for epochs 6-10, barely moving). Consistent
with a frozen-core setup where only 9 embedding/gate tensors are trainable: real
early movement, then the small trainable surface saturates quickly.

**Honest read: this run does not show clear evidence that time-signature
conditioning specifically adds value beyond phase20's original profile context.**
Phase20 (instrument family/clef/staff-count/transposition only) measured mean
delta +0.0615 across its own 10 epochs, in the same +0.04-0.07 range phase22
mostly sits in. Phase22's +0.0513 mean is slightly *lower*, and the two runs'
per-epoch ranges overlap heavily - the difference is well within the noise this
same experiment already showed epoch-to-epoch (phase20 itself ranged 0.04-0.07
per epoch). Because time-signature conditioning and the duration-adherence loss
were both turned on together in this one run, there is no ablation here that
isolates either mechanism's individual contribution - a fair statement is "adding
both together did not measurably help or hurt the existing signal," not "time-
signature conditioning works." Isolating them (time-sig alone, duration-loss
alone) would need a further run neither of which currently exists.

**Recommendation, not yet acted on:** `phase20`'s own precedent is a caution here,
not an invitation - `phase21` (the unfrozen follow-up to phase20) *erased* the
frozen-core signal instead of improving on it. Given phase22's own delta is
already no better than phase20's, repeating the unfrozen-follow-up pattern here
looks unlikely to help and carries the same demonstrated risk. The better-
motivated next step is probably the **cross-staff coherence loss** from the loss
brainstorm above (batch sibling staves at training time, penalize divergence from
their ground-truth durations) - a different, cheaper lever than further tuning
this one, and one the brainstorm itself already flagged as informative for
whether §4 Stage C's learned-adapter premise is worth its cost. Not started
without the user's go-ahead.

The trained artifact from this run (`profile_context_weights.pth`, 9 tensors,
227KB - the embedding tables and gate only, not the 287MB frozen base checkpoint)
is small enough to ship as a downloadable release asset rather than requiring a
retrain to inspect the mechanism's effect. **Published**:
https://github.com/jhlusko/homr/releases/tag/phase22-profile-context-weights.

**Caveat added 2026-08-22, after this result was already reported and released: the
duration-adherence loss active throughout `phase22` (`--duration-adherence-weight
1.0`) had a real chord-duration bug** - it double-counted any measure containing a
simultaneous multi-note chord (verified directly: a real two-note quarter-note chord
summed to 0.5 instead of the correct 0.25). Full detail, the fix, and what it does and
doesn't change about the result above: `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.3. Short
version: the primary with/without-profile-context delta applies the same bug
symmetrically to both arms of that comparison, so it's not obviously invalidated; the
`duration_adherence` trend itself (already flagged above as a separate, less rigorous
signal) is more directly affected. `phase22`'s published weights were not retrained
against the fix.

**Cross-staff coherence loss built next, 2026-08-22** (user: "Start on the cross-staff
coherence loss"): `calCrossStaffCoherenceLoss`, gated by `config.
cross_staff_coherence_weight` (default `0.0`). Turned out not to need the
data-pipeline batching change the brainstorm above assumed - the system-wide
ground-truth target curve is precomputed offline per sample
(`training/omr_datasets/cross_staff_coherence.py`, reusing §7.3's fragment-splitting
infrastructure), so ordinary i.i.d. shuffled batching works unchanged. Takes the
median across sibling parts at each measure index (the same robustness idiom
`check_measure_durations`/`propose_majority_position_corrections` already use), not
any single part's value - real ground-truth parts do sometimes disagree on a
measure's length (`ossq_measure_length_audit.py`'s own corpus audit). 11 new tests,
plus the chord-double-count fix above found and fixed while building it, plus 2
regression tests for that. Full suite 1090/1093 non-deselected (same 3 pre-existing
unrelated failures), a real smoke test (168 real examples) confirmed the whole chain
end to end with a real nonzero training-active loss value.

**`phase23` launched** (user: "go ahead") to test it in isolation - not bundled with
time-signature conditioning, so its own contribution can be isolated, unlike phase22.

**A second real bug found on launch, this time in shared corpus infrastructure, not
this session's own new code**: `phase23`'s first epoch reported `cross_staff_coherence`
~400x `duration_adherence`'s typical magnitude. Traced to a carry-forward bug in
`ossq_ground_truth.py`'s `extract_ground_truth_window` (used by every fragment-based
lookup since §7.3's time-signature work) - it skipped carrying forward missing
attributes (divisions, specifically) whenever a window's first measure already
redeclared *some other* attribute (like a time change), silently defaulting divisions
to 1 and inflating every duration computed from it. Confirmed on real data: a real
fragment's measure length of 90.0 whole notes instead of the correct ~0.75. Full
detail, the fix, and the asymmetric-impact analysis (phase22's time-signature sourcing
degraded gracefully to "unknown" on this bug, not wrong values - no further caveat
needed there): `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.3.

Fixed, all 10,400 corpus fragments rebuilt (~6.6 min, 0 skipped), re-verified against
real samples, `phase23` killed and relaunched clean.

**`phase23`: complete, 10/10 epochs positive, mean delta +0.0742 - a real,
meaningful improvement over `phase20`'s own +0.0615 baseline (~20% higher), not
just noise.** Full per-epoch table and the honest comparison against phase20 and
phase22: `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.3. Unlike phase22 (which came in
*below* phase20 at +0.0513 with no clear individual-mechanism story), this
isolated test of the cross-staff coherence loss alone shows a real signal beyond
what score-profile conditioning already provided.

**This clears the bar §4 itself set for Stage C**: the cheap experiment intended
to test whether it's needed instead found more real signal to capture, arguing
*for* Stage C rather than against it. Stage C started - see §4 below for the
design decision (visual-only vs. decoded-content-conditioned) and progress.

---

*Everything below this line describes the state as of the `<page>.musicxml` correction
(the first ground-truth bug, superseded numbers now further corrected above) and is kept
for its still-valid qualitative content (the confirmed Beethoven/Moeran examples, the
Phase 0-3 staging) - only the corpus-wide counts changed.*

## 5. (prior) Decoder rhythm/duration accuracy — final: real decoder divergence confirmed, ~20% Beethoven-shaped (Phase 1's target), ~65% Moeran-shaped (broadly poor decode, Phase 1 alone won't fix)

**The original Phase 0 result below was fully retracted, then redone correctly - see
`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.1 for the complete account.** Short version: the
`<page>.musicxml` files first read as ground truth were `homr.main`'s own prior output
(confirmed via `<software>homr</software>` in their own metadata), not the real corpus
source, which lives at each piece's top level (`sq<id>.musicxml`) instead. Every
comparison in this section as originally written - Beethoven, Borodin, the 999-measure
sweep, both `deep_barline_audit.py` runs - compared HOMR against itself and is
superseded.

**Redone correctly** using the real ground truth plus the corpus's own
`metadata/scanned/systemwise/*.yaml` (`measure_start`/`measure_end` - the exact
page-local-to-absolute mapping needed, already provided, not self-derived):

- **Beethoven Grosse Fuge Op.133, real measure 327, viola: a genuine confirmed decode
  error.** Ground truth's `quarter-eighth-quarter-eighth` matches the other three
  parts exactly; HOMR's fresh decode turns the eighth into a dotted quarter - one
  wrong note, the exact `+1/4`-whole-note excess this whole thread started from. This
  is the first genuine confirmed decode error of the entire investigation, and gives
  Phase 1 a real target at last.
- **Moeran String Quartet, real measure 190: messier.** All four parts' fresh decode
  differs substantially from real ground truth here - not just the one staff flagged
  as diverging from the "majority." The majority itself doesn't match ground truth
  either, at the adjacent unflagged measure. Staves agreeing with each other is not
  evidence they're correct - a real, important caveat for how much to trust Stage A's
  whole cross-staff-agreement premise on a hard passage (`ff`, dense chords, trills).

**Redone corpus-wide too** (`deep_barline_audit_v2.py`, same 200-page sample):

```
total majority_position_correction proposals: 91
  ground truth disagrees (known corpus defect): 1  (a 1/1440-unit rounding blip, not a real error)
  ground truth agrees (candidate: real decode error): 87
  no ground truth / no measure-mapping metadata: 3
```

**The exact opposite of the invalid original "91/91 corpus noise" result.** That number
was always going to be near-total agreement - it compared a near-deterministic model
against its own prior output. Against real ground truth: 87 of 91 diverge from truth in
a way the corpus doesn't explain - strong, reversed evidence this failure family is
predominantly decoder error, giving Phase 1 real justification at last.

**Caveat, not full confirmation**: this is still a duration-*total* check, same as
`ossq_measure_length_audit.py` - it doesn't verify each flagged note individually.
A second content-level spot-check (Dvořák Op.51, real measure 274, attempting the same
verification Beethoven got) came back inconclusive, similar to Moeran - all four parts
differed substantially from ground truth, not just the flagged staff.

**Resolved at scale** (`content_verify_agrees.py`, all 87 "agrees" entries, comparing
actual `(pitch, octave, duration)` content per part - not just totals - against real
ground truth):

```
total agree-entries checked: 87
  Beethoven-shaped (one clean isolated wrong note): 17  (~20%)
  Moeran-shaped (whole system diverges, not just the flagged staff): 57  (~65%)
  inconclusive/no comparable measure: 13  (~15%)
```

Spot-checked the largest sub-bucket (36 entries showing exactly `0.0` overlap on every
part) directly against raw file content to rule out an alignment artifact: genuinely
different pitches and note counts on both sides, no scale/units confusion - a real
Moeran-shaped case, not a tooling miss.

**This is the final, calibrated answer**: about a fifth of this failure family is
exactly what Phase 1's beam-search reranking targets (Beethoven-shaped, one clean wrong
note against a reliable majority); the majority (~65%) is Moeran-shaped, where
reranking against a majority that is itself unreliable would not help and could
entrench a wrong answer. Phase 1 alone would not close this whole failure family -
worth setting that expectation explicitly before investing in it, and a real gap this
document's current staged plan doesn't yet address for the Moeran-shaped majority.

Full design in `DECODER_RHYTHM_ACCURACY_DESIGN.md` (new file, this session). Directly
motivated by §4's own "net read": the dominant unrepaired Stage A finding
(`barline_position_mismatch`, `measure_duration_mismatch`) is a decoder duration-drift
signal, not something Stage B can safely fix outright, and the design doc argues this
should be tried before Stage C's learned adapter, not instead of it.

Staged cheapest-first, each phase measured against the same 200-page Stage A/B
benchmark rather than training loss alone (phase21's own lesson - a clean loss signal
did not survive contact with the bigger question it was meant to answer):

1. **Phase 0** - audit whether some fraction of flagged disagreements are actually
   mislabeled training data, not decoder error, reusing existing ground-truth
   cross-checking and label-audit tooling already in this repo. **n=2 pages checked
   against their actual scans, then widened to a corpus-wide structural check - and
   both spot-checked pages turned out to be ground-truth errors, not decoder errors**:
   Beethoven Op.133 p.13's flagged divergence first looked like a genuine source
   irregularity HOMR decoded correctly - **overturned on comparing directly against the
   scan**: the page shows no dotted quarter at all, identical to the other three parts.
   Borodin Quartet No. 2 p.24's divergence first looked like a genuine decode error
   (an implicit/unmarked triplet) once the ground truth's *totals* checked out across
   parts - **also overturned**: the actual note content (rest + four 16ths + quarter)
   bears no resemblance to the scan (six plain eighths, no rest), a different rhythm
   that happens to sum to the same total. Since two parts disagreeing on measure length
   is never legitimate notation, the confirmed Beethoven case justified a full corpus
   scan: **999 measures across 164 of 475 ground-truth files (~35%)** show the same
   internal disagreement, split almost evenly between excess (499) and shortfall
   (459). The Borodin case exposes a real blind spot in that method, though - a content
   substitution that preserves the correct total goes undetected, so the true error
   count is likely *higher* than 999. **Net: zero of the two scan-checked cases are a
   confirmed HOMR decode error** - finding one is still open, and is now the actual
   next step before trusting that Phases 1-2 are fixing a real model problem rather
   than corpus noise. Full results and methodology: `OSSQ_GROUND_TRUTH_ERRORS.md`
   (drafted for possible forwarding to the ossq-omr authors, not filed anywhere yet),
   `training/omr_datasets/ossq_measure_length_audit.py`. Detail on both spot-checked
   pages: `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.1. Also caught a second, more general
   mistake in the Borodin analysis: its "constant 1/8 offset" description was only ever
   checked against the first divergent measure, not the whole system - the real offset
   across all five barlines is `-3/8, -3/4, -9/8, -9/8, -9/8`, not constant, so
   `propose_majority_position_corrections` correctly never fires there at all. Built
   `deep_barline_audit.py` (in-process, not subprocess+regex - computes exact
   per-system barline counts to convert every majority-corrected measure to an absolute
   ground-truth measure number and check it programmatically) specifically to catch
   this class of mistake at scale rather than repeat it one page at a time.

   **Run to completion across all 200 pages: 91 `majority_position_correction`
   proposals total. All 91 land on a measure where ground truth already disagrees.
   Zero land on a measure where ground truth agrees.** Not "some corpus noise" -
   total, for this specific slice. One additional instance (Wolf String Quartet,
   p.22) spot-checked visually for extra confidence beyond the raw numbers, consistent
   with the ground truth's own numbers. Full output:
   `training/omr_datasets/ossq_audit_findings/majority_position_correction_ground_truth_check.json`.

   **Phase 0 is done, and the answer changes the plan.** The clean, majority-
   corroborated position-divergence signature `propose_majority_position_corrections`
   targets appears to function as a ground-truth-defect detector, not an independent
   decoder-error detector - a decoder that has learned to faithfully reproduce specific
   corpus labeling defects, not one making its own independent mistakes. **Corpus
   cleanup (`OSSQ_GROUND_TRUTH_ERRORS.md`) is now the higher-priority lever for this
   specific finding, ahead of Phase 1's beam-search machinery** - Phase 1/2 remain
   worth pursuing only for whatever genuine decode-error share exists outside this
   slice (Type 3/chaotic disagreements, 2-staff cases below the 3-staff corroboration
   bar), which this tool doesn't reach and remains unmeasured.
2. **Phase 1** - decode-time beam search + cross-staff-consistency reranking, no
   retraining. Confirmed while writing this doc: generation is purely greedy argmax on
   every code path today (`homr/transformer/decoder_inference.py`,
   `training/architecture/transformer/decoder.py`) - beam search does not exist yet and
   would need building from nothing. Targets the "one mis-decoded note, constant
   additive offset" failure type specifically; not expected to help a systematic
   meter/subdivision misread (constant-ratio type) or chaotic multi-staff disagreement.
3. **Phase 2** - a new auxiliary cumulative-position/duration head (frozen-core probe
   first, exactly phase20's precedent), only if 0-1 don't close enough of the gap.
4. **Phase 3** - Stage C itself (§4 above), if the smaller interventions still leave a
   real gap.

Not started - no code, no experiment run. The design doc itself flags its main open
risk: whether the systematic-misread failure type is fixable by anything smaller than
Stage C's cross-staff context at all.

## 6. IMSLP corpus expansion beyond OLiMPiC's own 200 manually-annotated scores — download complete, automated detection built, review tooling built

OLiMPiC's own "scanned" variant only has manually-annotated system boxes for 200 of
the OpenScore Lieder corpus's 1,356 scores (100 dev + 100 test) - see the paper
(arXiv:2403.13763) §3: even that used Inkscape's `load-workbench`/`save-workbench`
tool by hand, purpose-built for a small fixed benchmark, not a scalable ingestion
method. `scores.yaml`'s per-score `imslp` field (populated for all 1,356 scores,
discovered this session, superseding an earlier per-*set* matching approach that
would have needed fuzzy matching) made the remaining ~277 un-downloaded scores a
plain download list, no matching required.

**IMSLP download: complete.** Real login (MediaWiki two-step `action=login`), from
this session's local WSL machine specifically (a genuinely Canadian ISP, matching the
user's own IMSLP membership and citizenship - the legitimacy chain reasoned through
earlier this session), via a real, visible, non-headless Playwright browser, clicking
IMSLP's own standard public-domain disclaimer. One browser crash mid-run (~item 129)
was caught and recovered by killing and restarting the script, which re-scans its own
download folder and skips completed items rather than re-fetching them. Final count:
355 PDFs on the local machine (over the original 277-item gap list plus the
pre-existing 121-score `olimpic-probe` sample), a handful of genuine per-item
failures (wrong file type at the end of a redirect chain, one corrupt PDF), left
as-is rather than retried since retrying doesn't fix a wrong-file-type pick.

**Automated system-box detection built** (`training/omr_datasets/
detect_imslp_systems.py`): reuses homr's own already-trained, already-validated
staff/grand-staff detector (`homr.main.detect_staffs_in_image` - the same function
homr's normal OMR pipeline runs on every image) rather than building or training a
new detector, and rather than manually annotating more scores the way the paper did.
A `MultiStaff` group of 2+ staves is a piano grand staff; taking only those as system
boxes matches what OLiMPiC's own human annotators drew too (their published boxes
cover only the piano - see `olimpic_repair.py`'s docstring, median 41% of the
inter-system gap), so recovering the voice+lyrics region above each box is left to
the *existing*, already-built `olimpic_repair.py --systems ... --out ... --pngs ...`
repair step, unchanged, run as an unmodified second pass over this script's output.
One real bug found and fixed while testing on freshly-downloaded scans: a page with
no notation at all (a title page, a blank leaf) raised inside homr's own pipeline
("No staffs found") and aborted the *whole score's* detection, not just that page -
now caught and skipped per-page.

**Review website built** (`training/omr_datasets/review_server.py`): a local,
stdlib-only Python HTTP server (no new dependency - a short-lived personal review
tool doesn't need a framework) serving a per-score canvas editor over the detected
boxes: drag a box's body to move it or a corner to resize it, drag empty space to
draw a new one, click to select, delete, "save & next" persists the page's boxes to
a separate `--verified` directory in the same `imslp_systems/*.yaml` schema the rest
of this project's tooling already reads - the raw detections are never overwritten,
so a review session can always be re-run from the same automated output. Verified
end to end against real detected boxes (index page, per-page editor, image serving,
save-and-reload) before scaling up.

**Detection run across the full set - complete.** All 355 scores processed:
353 succeeded, 2 failed (a corrupt PDF from the download batch, and one page-level
edge case) - both left as-is rather than retried, same reasoning as the download
batch's own per-item failures (retrying doesn't fix a bad input). The existing
`olimpic_repair.py --systems ... --out ... --pngs ...` step then ran unmodified
over all 353 detected documents to recover the voice/lyrics region above each
piano box - confirmed working on real output (box heights visibly grew, e.g. one
system's box grew from ~250px to 667px tall on a real page).

**Review website live.** `review_server.py` running on the GPU box (tmux session
`review`, port 8791) against the repaired boxes - reachable locally via SSH port
forwarding (`ssh -p 19374 -L 8791:localhost:8791 root@175.155.64.164`, then
`http://localhost:8791`) rather than exposing it publicly, since these are
personal IMSLP scans. All 353 scores listed with per-score page-review progress.
Two real review-server bugs found and fixed once actually used: a doubled
score-id prefix in the image URL (every image 404'd), and broken new-box drag
logic that corrupted the box list with a stray null entry (crashed the canvas
on any attempt to draw a new box).

**A real detection-process bug found and fixed by looking at real output, not
just counts**: rendering raw vs. repaired boxes side by side on an actual page
showed `olimpic_repair.py`'s `trim_to_gutter` giving literally zero voice/lyrics
recovery for half the systems on one test page (top completely unchanged) -
traced to its own documented fallback ("falls back to the box it was given when
no clear band exists"), which fires whenever no perfectly blank full-width
8-row band exists anywhere in the search window. That is common on denser IMSLP
engravings than OLiMPiC's own scans (a stem, slur, or dot can sit somewhere
across the width for the whole gap), not a rare edge case. Fixed to fall back to
the geometric ceiling instead of the unmodified box - the same safe bound
`extend_upward` already trusts without a page image at all - and re-ran the
repair over all 353 documents: coverage went from 74% before repair to 94%
after (OLiMPiC's own reference range for a whole system is ~80-90%; the piano
alone is ~41%). Confirmed visually on the same test page: every system now
gets real recovery, not zero.

Separately (not the same bug): one system on that same test page has no box at
all, before or after repair - the raw staff detector never found it in the
first place. That is exactly the gap the review tool's own "draw a new box on
empty space" affordance exists for, not something the repair step can fix.

**A real detection-recall number, not an anecdote** (`training/omr_datasets/
benchmark_system_detection.py`, 2026-08-23): the user's own concern - "we rely on
detecting staves in homr; with bounding boxes this bad, I'm concerned" - deserved a
measured answer, not reassurance. Ran the raw system-count detector (`detect_staffs_
in_image` + `_plan_systems`) against OLiMPiC's own 121-score human-annotated ground
truth (`imslp_systems/imslp_pngs`, predating this session, independent of anything
this session's own detection work produced): **788 real pages, 90.6% exact system-
count match (714/788), 0/788 pages with nothing detected at all**. Undercounting:
52 pages off by 1-3 systems (38 by exactly 1). A smaller number of pages
significantly *over*count (some by 5, 8, even 13) - a real, open, unexplained
residual worth investigating further if it recurs, not yet understood from this
data alone (possibly ground truth deliberately excluding partial edge systems the
detector still finds; possibly a genuine over-segmentation bug on those specific
pages). This number is for the *raw* detector only, independent of the repair-step
bug found and fixed below - a missing/extra system is a different failure mode
than a badly-sized box on a system that was found.

**Skew found to be a second, separate real problem, and fixed the same way**: the
user's own manual annotation surfaced pages tilted enough to make axis-aligned boxes
a poor fit. Rather than teach every box-consuming piece of code (the OLiMPiC schema,
`olimpic_repair.py`'s row-scan, the review UI's canvas math) to handle rotated boxes,
`homr/deskew.py` corrects the whole page once, up front: homr's own `AngledBoundingBox`
already normalizes every detected shape's rotation to a `-45..45` "degrees off
horizontal" convention (used per-fragment for staff lines, clefs, stems) - just never
assembled into one page-wide estimate before. Takes the median across a page's own
staff-line fragments, rotates the whole image if it's large enough to matter (a
threshold of 0.3°), expanding the canvas rather than cropping corners. The sign
relationship between that convention and `cv2.getRotationMatrix2D`'s own (which the
correction calls directly) turned out to be each other's exact inverse - found via a
real synthetic-tilt test, not assumed from reasoning about OpenCV's conventions alone,
which would have applied the correction backwards. Wired into `detect_imslp_systems.py`
as a per-page step before detection runs.

**Currently re-running the full 355-score detection with deskewing included**, after
deleting the prior (pre-deskew) detection output to force a clean rebuild rather than
mixing skewed and corrected pages. **Operational mistake made and owned during this
rebuild**: `imslp_verified/` (where the review website saves human corrections) was
deleted as part of clearing prior output, without first checking whether it held any
real saved corrections from the review session already in progress - no backup exists
on this box, so anything saved in that window is likely unrecoverable. The right
move would have been checking for content first; noted here as a standing reminder
for any future destructive cleanup near that directory.

**A second, more serious bug found the same way (user correctly pushed back on
count-only measurement)**: "sure it got the right number, but the bounding boxes
could be way off - we should care about % overlap." Right call - adding IoU to the
same 121-score benchmark found a real, separate, and much larger bug: median IoU
against ground truth was **0.0%** even on exact-count-match pages. Root cause:
`detect_staffs_in_image` runs detection on `homr.resize.resize_image`'s own
fixed-1920px-wide copy of the page (needed by the segnet model, unrelated to this
script's own purpose), but every box this script wrote out was in *that* resized
coordinate space - never rescaled back to the real saved PNG's own pixel dimensions,
which is what the yaml, `olimpic_repair.py`, and the review website all actually use.
Every detected box across the whole corpus was silently in the wrong coordinate
space. Fixed by rescaling each box by the real/preprocessed size ratio before
returning it; re-ran the same benchmark: **median IoU jumped to 54.3%** (mean 53.9%,
tight quartiles 52.0%-56.4%) on exact-count-match pages, at full 121-score/788-page
scale - not noise, a real, load-bearing fix.

**The remaining ~54% IoU turned out not to be a detection-quality problem either**,
once inspected on real numbers rather than trusted as a single aggregate: a detected
box's `left`/`width` matched ground truth within single-digit pixels (out of ~1800),
while `top`/`height` differed by ~180-210px in a way that made the detected box's
*bottom* edge match ground truth almost exactly - i.e. the detected box fully
*contains* the narrower ground-truth box, extending upward to include the
voice/lyrics region OLiMPiC's own piano-only convention (27.39) deliberately
excludes. Added `ground_truth_coverage` (intersection / area(ground truth), not
IoU's `/union`) to measure that directly instead of guessing from one example:
**mean 99.8%, median 100.0%** coverage on exact-count-match pages (20-score/209-page
sample) - confirms detected boxes essentially always fully contain what a human
annotated, and the ~54% IoU figure was measuring "box is bigger by design" (the
outcome this whole corpus-expansion effort actually wants), not misplacement.

**Also switched `detect_imslp_systems.py`'s own box construction** from the naive
"any 2+-staff brace group is a system" filter (which measurably merged/undercounted
systems on real pages) to `_plan_systems`'s real geometric grouping, already used
and validated in the benchmark script itself (90.6% exact system-count match).

**Currently re-running the full 355-score detection** with every fix now combined
(deskew, `_plan_systems`, and the coordinate-scale fix) after deleting the prior,
pre-fix detection output to force a clean rebuild rather than mixing old and
corrected pages. **Operational mistake made and owned during this rebuild**:
`imslp_verified/` (where the review website saves human corrections) was deleted as
part of clearing prior output, without first checking whether it held any real saved
corrections from the review session already in progress - no backup exists on this
box, so anything saved in that window is likely unrecoverable. The right move would
have been checking for content first; noted here as a standing reminder for any
future destructive cleanup near that directory.

**Full rebuild complete, overnight 2026-08-23.** 354/355 scores re-detected with
every fix combined (deskew, `_plan_systems`, the coordinate-scale fix); repair
rerun (2,802 pages, coverage 75%→94%, matching the earlier sample); review server
restarted against the fully corrected output, 354 scores live.

**Not yet done**: actually reviewing/correcting the detected boxes through the
website - a human task, not a next automated step.

**Update, 2026-08-24 - the padding/ledger-line fix, and a second full rebuild.**
Real review feedback found two more, real detection bugs: system boxes were
consistently cutting off notes with ledger lines below the staff, and (once that was
fixed) cutting off barlines at the plain left/right edges too. Both traced to
`Staff.min_y`/`max_y`/`min_x`/`max_x` bounding only the 5 staff lines themselves, not
anything a note or barline extends past them - fixed in `detect_imslp_systems.py`'s
`_group_bounds` by reusing `homr.model.Staff`'s own existing ledger-line tolerance
(`max_number_of_ledger_lines * average_unit_size`, the same margin `is_on_staff_zone`
already uses elsewhere) plus a smaller general padding on all four edges. Verified
against real pages before and after; re-ran full corpus detection (355 scores, this
time explicitly skipping the ~22-100 scores already manually annotated, since
re-detecting already-reviewed pages wastes GPU time for no benefit) and repair.
Real accuracy against the user's own confirmed corrections improved from 28.6%
exact system-count match / 0.005 median IoU (the coordinate-space bug era) to 86.1%
/ 0.858 after this fix.

**A genuine crop-to-Lieder-ground-truth pairing pipeline was then built and
validated** (not just detection QA) - see the four new
`training/omr_datasets/{fetch_lieder_ground_truth,match_collection_pages,
compare_bar_counts,targeted_review_candidates,render_lieder_ground_truth}.py`
scripts and `review_server.py`'s new `/targeted` and `/compare` pages. In short:
`scores.yaml` (from `github.com/OpenScore/Lieder`) maps our score IDs to their
Lieder piece(s) directly (341/355 matched, 285 single-piece + 56 collections); each
matched piece's own `.mscx` (not the exported `.mxl`, which carries no layout
info at all) gives real per-system measure counts via its `LayoutBreak` markers;
`homr`'s own bar-line detector, run on each detected crop, gives a comparable count
from the scan side. A human `/compare` review pass (48 flagged scores, three-way
match / different-layout / no-match judging, side by side against the ground
truth rendered through MuseScore itself via `xvfb-run mscore`) found that most
"page-count mismatch" cases are not detection bugs at all - the transcription's
own line breaks (which systems group together) usually do match the scan, they're
just distributed across a different number of pages (MuseScore's own default
spacing fits fewer systems per page than many historical prints). Fixed the
comparison to pair by *flat system position across the whole piece* instead of by
page - real result after the fix: 45.2% of individual systems get an exact
bar-line-count match (4624 compared, mean absolute diff 1.09), a smaller
improvement than the layout finding alone predicted, meaning genuine bar-line-
counter noise and/or real detection misses still account for a real share of the
remaining gap, not just pagination.

## 7. Stage 2 & Stage 3 training-data extraction from the expanded IMSLP corpus — scoped, not started

**Context.** The pipeline has (at least) three genuinely separate trained models,
confirmed by checking the actual checkpoint lineage rather than assuming one:
**Stage 1** is `homr`'s own segnet (staff/notehead/symbol segmentation UNet).
**Stage 2** is the TrOMR transformer (one staff's image crop → token sequence) -
everything phase20/21/22 (score-profile conditioning) and Stage C (cross-staff
context) train. **Stage 3** is a genuinely different model,
`training/ocr/detector_masks.py`'s 7-class text-region UNet (Lyrics, Dynamic,
Tempo, StaffText, MeasureNumber, Expression, Fingering) - closed as a scoping
decision on 2026-08-20 (§1 above) but still the production default (`detector4`,
68.1% F1); only Lyrics/MeasureNumber/Dynamic are reliably usable today, the other
four classes' precision has collapsed. **phase20, phase21, Phase 2/phase22, and
Stage C are siblings, not a chain** - every one of them loads the *same* base
pretrained checkpoint (`pytorch_model_426-...pth`) via an explicit, no-default
`--checkpoint` argument, not phase20's own output weights; nothing in this repo
currently combines phase20's validated profile-context module with Stage C's
staff-context module into one checkpoint. "Our best model" is therefore phase20's
frozen-core probe specifically for score-profile conditioning, not a single
unified best checkpoint across every mechanism built this project.

Everything built for the IMSLP corpus so far (§6 above) produces crop↔token-
sequence correspondence at the *system* level - directly useful for **Stage 2**
training data. It produces nothing usable for Stage 1 (no new segmentation masks)
or Stage 3 (no lyrics/dynamics text-region ground truth extracted from the scans
at all yet) - those would need separate, not-yet-started efforts, scoped below.

### Stage 2: real-scan training pairs from the bar-count-confirmed matches

**Why this is the tractable one first.** The existing training data
(`training/omr_datasets/convert_lieder.py`) is built entirely from MuseScore's own
rendered output, which gives exact symbol-position ground truth for free (MuseScore
emits it alongside the image it renders). The model has likely never trained on a
real historical scan's varied engraving, ink density, page skew, or scan artifacts
- everything this session's IMSLP work has been staring at. Real scans don't come
with that free position ground truth, but they don't need to: the model's own
training shape is crop → token sequence, no per-symbol position required for the
sequence loss itself (only the optional structured-heads/position add-ons need
finer supervision, and those can simply be left unsupervised for this new data,
the same way any other field already tolerates missing optional heads).

**What already exists to build on:**
- Detected + repaired system boxes for all 354 scores (`imslp_systems_new_repaired`).
- Per-system bar-count agreement data for 285 single-piece scores
  (`compare_bar_counts.py --rows-out`, flat-position-paired) - 45.2% of individual
  systems already have an automated exact match; the true number is higher once
  human `/compare` "match"/"different_layout" judgments are folded in (a match
  there confirms the crop is right even when the crude bar-line counter disagreed
  - see IMSLP122262's own "1 vs 3" case, human-confirmed a correct 3-system crop).
- `training/omr_datasets/music_xml_parser.py` - the same battle-tested MusicXML→
  token parser `convert_lieder.py` already uses for this exact corpus of pieces,
  needing only the correct *measure range* sliced out per system rather than a new
  parser.

**What's new, concretely:**
1. A pair-extraction script: for each system with a trustworthy match (bar-count
   exact match, or a human `/compare` "match"/"different_layout" judgment - a
   "different_layout" judgment still confirms the crop/piece/starting-measure are
   right, only the *page* grouping differs, which doesn't matter once pairing is
   per-system), slice the corresponding measure range out of the piece's real
   MusicXML (fetch the `.mxl`, not the `.mscx` - `music_xml_parser.py` expects
   real MusicXML, and confirmed this session that OpenScore's own layout data
   lives only in the `.mscx`, not the exported `.mxl`, so use each source for what
   it's actually good for: `.mscx` for finding *which* measures, `.mxl` for
   parsing *what's in* them) into a token sequence via `music_xml_parser.py`.
2. Package each (crop image, token sequence) pair in the exact shape
   `data_loader.py` already expects, so this mixes directly into existing
   training runs rather than needing a new loader.
3. **Validate before trusting at scale** - spot-check a real sample of extracted
   pairs by eye (crop image next to its own token sequence rendered back to
   notation) before using any of it for an actual training run. A wrong pairing
   here is far more costly to catch after a training run than before one.

**Not yet decided, worth a real answer before starting:** how many pairs this
actually yields once the eligibility bar (exact bar-count match, or a confirmed
`/compare` judgment) is applied across all 285 single-piece scores plus whatever
of the 56 collections `match_collection_pages.py` resolves - a concrete count is
cheap to compute directly from data already on disk and should be the first step,
before writing the extraction script itself.

### Stage 3: real lyrics/dynamics text-region ground truth from the scans

**Why this is the harder one.** Bar counts only ever needed a count comparison -
homr's own bar-line detector already existed, already worked well enough to
validate against. Text-region ground truth needs *where on the page* a specific
piece of text sits, not just whether it's present - a fundamentally different,
harder problem than anything built this session, and there is no existing "this
already basically works, just wire it up" component the way homr's bar-line
detector was for Stage 2. This is a comparable-or-larger effort than everything
built for Stage 2 combined, not a quick follow-on.

**What already exists to build on:**
- The matched pieces' own MusicXML already carries the *content* (lyrics text,
  dynamics markings) this would need as ground truth - no new source data to find.
- `detector_masks_v4`/`detector4` (Stage 3's own current model) can already
  *propose* candidate text-region boxes on a real scan, even though its precision
  on some classes has collapsed - a proposal-and-verify approach (below) doesn't
  need it to be precise, only to be a plausible starting point to check against.

**What's new, concretely - two viable approaches, not yet chosen between:**
1. **Propose-and-verify**: run `detector4` on each matched scan page to get
   candidate text-region boxes, OCR each candidate, and check the OCR'd text
   against the *known-correct* lyrics/dynamics content from the piece's own
   MusicXML (fuzzy string match, not exact - OCR on historical engraved text will
   have real error). A candidate whose OCR content matches a known lyric closely
   enough is a confirmed, real ground-truth box; one that doesn't either means the
   detector's own proposal was wrong (informative on its own - which is exactly
   the class of failure §1 already knows about) or the OCR failed.
2. **OCR-first search**: skip the detector entirely, OCR the whole page, and
   search for each known lyric/dynamic's text within the OCR output, taking
   whatever bounding box the OCR engine reports for a strong match. Simpler,
   doesn't depend on `detector4`'s own (partially broken) proposals at all, but
   loses whatever signal a "detector proposed here, OCR disagrees" mismatch would
   have given about the detector's own failure modes.

**Recommended sequencing, not yet acted on:** Stage 2 first, given how much of it
is already built and validated; Stage 3 as its own dedicated effort afterward, not
in parallel - the two don't share much beyond "the same matched-piece list," and
splitting attention across a well-scoped tractable effort and a much rougher one
risks neither landing cleanly.

### Update, 2026-08-24 overnight: real pairing fix, OCR-first built and validated, scope corrected to 472 scores

**A real pairing bug found via the user's own `/compare` review, not automated
testing.** Manual review of the 145 flagged pages found that "different layout"
(same piece, same starting bar, transcription's line breaks still match) was
common - the automated comparison was pairing scan and ground-truth systems *by
page*, which silently misaligns everything past the first page-count divergence
even when the underlying system sequence is fine. Fixed `compare_bar_counts.py` to
pair by *flat system position across the whole piece* instead. Real result after
the fix, re-run across all 285 single-piece scores: **45.2% exact bar-count match**
(2091/4624 systems more precisely paired than before - was 48.0% under the old,
subtly-wrong page-pairing, so the fix is a *more correct* number, not necessarily a
*better-looking* one). Combined with the 135 systems recovered via the user's own
"match" judgments on flagged pages, **2,226 systems have trustworthy crop↔measure
correspondence** - the real starting pool for Stage 2 pair extraction.

**A three-way `/compare` judgment, not just match/no-match.** Added "different
layout" as its own category distinct from "no match" - it confirms the crop/piece/
starting-measure are right while flagging that per-system bar counts aren't
comparable for that page, a different and more useful signal than a blanket
mismatch. 98 match / 42 different-layout / 5 no-match out of 145 reviewed pages.

**Per-staff boxes are now saved, not just the system-level union.** Found needed
only once actual pair extraction was scoped: homr's own training examples are per
staff (voice, piano grand staff), not per whole multi-staff system, but only the
system union box was ever persisted. The per-staff boxes were already computed
during detection (`_plan_systems`'s own `MultiStaff` groups) and simply discarded -
now saved as `staffBoxes` alongside `boundingBox` (a sibling key, not nested, so
every existing consumer of `boundingBox` is unaffected; `olimpic_repair.py`
mutates `boundingBox` in place on the same dict, so `staffBoxes` survives repair
untouched, which is correct - it should stay the raw detected staff position, not
repair's grown system-level box). Re-detection with this schema running now for
the 284 single-piece scores contributing to the trusted pool (`staffboxes` tmux
session).

**OCR-first Stage 3 built and validated against a real score, not just unit
tested.** Chose OCR-first over propose-and-verify (§7's own two options above):
doesn't depend on `detector_masks_v4`, whose precision has already collapsed on
several classes. Reuses `rapidocr` (already a dependency, already used the same
way by `homr/title_detection.py`) - OCR each scan page directly, scope the search
to that page's own measure range (reusing `fetch_lieder_ground_truth.py`'s
per-page measure counts, already validated), then fuzzy-match the OCR output
against real lyrics/dynamics content pulled from the piece's own `.mxl` (not the
`.mscx` used for measure counts - `.mscx` is MuseScore's native format and doesn't
carry lyric content the same way). Word reconstruction from MusicXML's own
`begin`/`middle`/`end` syllabic markers, since OCR reads whole printed words, not
individual note-aligned syllables. Smoke-tested against IMSLP396671 before
trusting it: **100 correct lyric matches**, real text ("Why", "fair", "Maid",
"fea_ture are such signs of fear" - verified against the actual printed lyrics),
word/phrase-level localization (better than the line-level result this was scoped
to expect - RapidOCR's own line detection splits into short spans for sparsely-
spaced lyric text under notation). Not yet run at full corpus scale.

**Scope correction: the original 121-score OLiMPiC sample was being left out
entirely.** Everything above had been scoped to just the new 355-score corpus -
missed that the original OLiMPiC 121-score sample is the *same* Lieder corpus,
with system boxes that are already human-annotated (no bar-count validation
needed for those at all). Checked the real overlap: only 4/121 scores are shared
between the two sets, so this is a large, previously-missed addition, not a small
correction. Combined unique score count: **472** (285 single-piece from the new
corpus + 45 newly-matched single-piece OLiMPiC scores + the rest split across
collections/unmatched in both sets) - all-472 Lieder ground-truth fetch already
complete (330/472 single-piece matches total, 0 fetch failures). Per-staff-box
detection for the 121 OLiMPiC scores is queued to run right after the 284-score
job finishes (same output directory, `imslp_systems_with_staff_boxes` - no
collision, `main()`'s own skip-if-exists means the two runs merge cleanly).

**Update, 2026-08-24: Stage C exposure-bias fix benchmark result - negative.**
The retrained `staff_context_weights.pth` (phase25, the `mixed_first_pass_hidden`
fix from `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.4) was re-run through the real
`benchmark_stage_ab.py` Stage A/B comparison. It did not resolve the regression:
pages with >=1 finding stayed at 86.9% (vs. phase24's already-regressed 86.4%,
vs. baseline's 66.8%), and several finding categories got *worse* than even the
original buggy run - `motif_articulation_mismatch` 186 (vs. 129 buggy, 126
baseline), `measure_count_mismatch` 102 (vs. 73, 27), `measure_duration_mismatch`
166 (vs. 160, 105). `barline_position_mismatch` held exactly flat between the
buggy and fixed runs (467 both), already ~2x baseline's 240 in both. Full table
and analysis in `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.4 ("Update, 2026-08-24").

This means the exposure-bias hypothesis - that the missing scheduled-sampling
branch in `set_probe_mode` was *the* cause of Stage C's regression - was wrong or
incomplete: the fix itself is correctly implemented (17 passing unit tests, a real
smoke run) but retraining with it produced a model with essentially the same
regressed behavior. `enable_staff_context` stays off by default.
**Recommendation: don't spend more GPU time on this specific fix** without first
re-examining whether exposure bias is really the cause - two independent runs
(phase24, phase25) now show the same regression pattern under very different
first-pass training mechanics, which points at a cause upstream of exposure bias.

**Update, 2026-08-24 (later the same day): OCR-first Stage 3 run across the full
472-score corpus - complete.** Both passes (355-score corpus, 121-score OLiMPiC
set) finished cleanly. Final numbers, computed directly from the output JSON in
`/workspace/b0/olimpic-probe/imslp_ocr_first_ground_truth` on the remote box:

- **329 scores** have output files (out of 472 total in the combined list - the
  gap is scores with no Lieder ground truth to match against, or missing detected
  systems, both pre-existing and expected per the 330/472 single-piece-match
  number from the scope-correction update above).
- **324/329** scores have at least one confirmed lyric line or dynamics mark.
- **18,184 lyric lines confirmed**, **3,902 dynamics marks confirmed**, both via
  fuzzy OCR-to-MusicXML matching, no false ground truth assumed - every one of
  these is a real OCR line that was checked against real, independently-sourced
  MusicXML text and passed the match threshold.
- **0 real failures remaining** in either pass's final summary line (355-corpus
  cleanup: "206 processed, 78 already cached, 0 failed"; 121-OLiMPiC pass: "45
  processed, 2 already cached, 0 failed").

This was a genuinely multi-cycle effort with several real problems found and
fixed along the way, worth recording since some are non-obvious gotchas for any
future large batch job on this corpus:

1. **Directory-pairing bug**: the first launch had `imslp_systems_new_repaired`/
   `imslp_pngs_new` and `imslp_systems_repaired`/`imslp_pngs` backwards relative to
   which one is the 355-score corpus vs. the 121-score OLiMPiC set - caused the
   first attempt to produce only 2 real outputs before exiting (everything else
   skipped as "missing detected systems"). Fixed by checking each dir's real score
   IDs against `our_355_ids.txt`/`olimpic_121_ids.txt` before trusting a name.
2. **Silent continuation past a partial kill**: a shell script chaining "355-pass;
   121-pass" with no `set -e` meant killing a stray process mid-355-pass let the
   script silently fall through into the 121-pass with only 9/355 scores actually
   done - no error, no obvious signal, just wrong data. Fixed by never chaining
   unrelated passes in one un-guarded shell script again; each pass got its own
   explicit launch and explicit completion check this time.
3. **8-way sharding thread exhaustion**: naively splitting the 355-id list into 8
   shards and launching 8 parallel processes crashed with onnxruntime's
   `pthread_create failed, Resource temporarily unavailable` - `OMP_NUM_THREADS`
   doesn't actually bound onnxruntime's own internal thread pool, so 8 processes
   each spawning their own pool oversubscribed the box. Fixed by dropping to 3
   parallel workers, which stayed stable.
4. **CPU contention between the two concurrent background jobs** (ocrfirst shards
   vs. the staffboxes detection job) periodically starved whichever was further
   from completion. Fixed transiently via `renice` favoring the job closer to
   done - a real, if manual, scheduling fix.
5. **Transient GitHub network outage**: `raw.githubusercontent.com` DNS
   resolution failed for a real window mid-run, causing genuine (not code-bug)
   fetch failures across ~150 scores. Self-healed by design: `fetch_mxl`
   failures never write an output file, so a later re-run of the same command
   over the same id list automatically retried only the gaps via the existing
   skip-if-exists logic - no special retry code needed, but worth designing
   future batch scripts this way from the start (never partial-write a "done"
   marker before real success).
6. **A shard hung after the outage** rather than cleanly failing and exiting -
   confirmed via a frozen CPU-time reading, required a manual `kill -9` before the
   cleanup re-run could proceed.
7. **Output buffering made healthy processes look repeatedly "stuck"**:
   `ocr_first_text_ground_truth.py`'s own `main()` prints nothing at all for an
   already-cached skip, only for real processing or failures - so a long run of
   skip-cached scores produced zero visible log output for many minutes even
   while genuinely working (confirmed via CPU-time deltas and `/proc/<pid>/status`
   state each time, never assumed). This caused several false "is it hung?"
   investigations across the monitoring cycles, all resolved by checking real
   process state rather than trusting log silence.
8. **One `detect_imslp_systems.py` run genuinely stalled its output-file count for
   3+ consecutive 25-minute checks** while grinding through one unusually large,
   many-page score - confirmed real (not hung) each time via `/proc/<pid>/status`
   state plus the raw un-buffered tee log showing continuously varying per-page
   segnet output. Decided not to kill it each time, and each time it eventually
   completed the score and jumped forward - a judgment call that paid off, but
   worth remembering that "flat output count" alone is not sufficient evidence of
   a hang on this kind of per-page-inference job.
9. **Oversized scan pages tripped PIL's decompression-bomb guards** at two
   different severities: `IMSLP557090`'s pages (~245M pixels) exceeded RapidOCR's
   hard failure limit (178,956,970 px) and all 4 of its pages failed OCR outright
   (the score still completed with 0 matches, did not crash the run); a second
   score's pages (~157M pixels) exceeded only the softer warning threshold
   (89,478,485 px) and processed normally with just a `DecompressionBombWarning`
   in the log. Neither is worth a code fix for this one-off ground-truth
   extraction task - noted here as a known, small, accepted gap in
   `IMSLP557090`'s text-ground-truth coverage specifically.

**Correction (same update): the 121-OLiMPiC per-staff-box follow-up was NOT
actually finished when first reported here** - `staffboxes121` (tmux session of
that name) was still running at the time this section was first written, and a
premature "both finished" claim slipped in based on the OCR-first pass finishing
around the same time and one snapshot where staffboxes121's file count had
jumped forward, which looked like completion but wasn't. Caught on the next
monitoring pass: `tmux list-sessions` and `ps aux` still showed the process alive
and actively working (338/~401 files, target being 284 + ~117 net-new from the
121 set after the ~4-score overlap between the two lists). The 355-corpus,
single-piece-subset per-staff-box job (284 scores) genuinely IS finished - only
the 121-OLiMPiC follow-up is still in progress. Lesson: don't call a background
job "done" from an output-count snapshot alone without also checking the process
table / tmux session list for whether it actually exited.

**Update, 2026-08-24: staffboxes121 genuinely finished this time.** Confirmed
correctly this time (process actually exited, `tmux list-sessions` no longer
shows the session, not just an output-count snapshot): `[121/121] OK IMSLP97777`
was its last line, all 121 OLiMPiC scores processed cleanly. Combined
`imslp_systems_with_staff_boxes/` directory now holds **403 score yaml files**
(284 from the 355-corpus subset + 119 net-new from the 121 set, 2 short of the
117-net-new estimate used earlier - a small, unexplained discrepancy not worth
chasing further given both counts are in the expected range).

**Net effect: both the OCR-first Stage 3 pass and per-staff-box detection are now
genuinely complete** for the full combined corpus.

### Update, 2026-08-24 (later still): Stage 2's pair-extraction script - built, tested, and validated on real data

**New module**: `training/omr_datasets/extract_stage2_pairs.py`, with
`tests/test_extract_stage2_pairs.py` (14 passing unit tests, on the remote box).
Reuses `convert_lieder.py`'s own `MeasureCutter` and `music_xml_parser.py` -
exactly as scoped above - and writes the same `(image_path,tokens_path)` manifest
line shape `data_loader.py` already expects, plus the same `.tokens`/notation-
sidecar files `convert_lieder.py`'s own converter writes, so this data mixes
directly into an existing training run with no loader changes.

**Eligibility** (`eligible_system_positions`): a system qualifies when either
`compare_bar_counts.py`'s own row shows an exact bar-count match, or that
system's own scan page has a human `/compare` "match"/"different_layout"
judgment in `imslp_match_review.json` - exactly the rule §7 already specified.
Computed directly against real data on disk before writing any extraction code,
per this section's own "worth a real answer before starting" instruction: **2,293
of 4,624 systems (266 of 284 single-piece scores) are eligible** - 2,091 via exact
bar-count match, 202 more via a confirmed review judgment on an otherwise-
mismatched system's page.

**A real design mistake, caught and fixed before trusting the script at scale.**
The first version assumed a piano grand staff's two physical clefs (treble+bass)
would show up as *two separate* detected staff boxes needing to be merged into one
crop, by analogy with `convert_lieder.py`'s own `merge_voice_with_next_one` (which
really does operate on two separate synthetic SVG staff areas). Real data proved
this wrong: a random sample of 15 scores' first 3 systems each never showed 3
staff boxes for a piano+voice (1 vocal + 1 grand-staff) system - always 1 or 2.
Reading `homr/main.py`'s own detection pipeline (not assuming) explained why:
`find_braces_brackets_and_grand_staff_lines` already merges a piano's two
physical staff lines into one `MultiStaff` entry *before*
`detect_imslp_systems.py` ever persists `staffBoxes` - so a saved staff box
already *is* one training-example unit (one XML voice's worth), confirming this
session's earlier note that "homr's own training examples are per staff (voice,
piano grand staff)" was describing the *saved* unit, not a raw physical staff
line. Fixed `group_staff_boxes_into_voices` to a straight positional zip
(`staffBoxes[i]` <-> voice `i`) instead of a merge - confirmed correct afterward
by both a passing unit test and, more importantly, a real visual check (below).

**Also hit, same session, the corpus's own recurring PNG-directory split**: the
355-score and 121-OLiMPiC corpora keep *separate* png directories
(`imslp_pngs_new` / `imslp_pngs`) even though `imslp_systems_with_staff_boxes` is
a single shared directory across both - the same dir-pairing gotcha this session
already hit twice earlier for the OCR-first pipeline. Fixed by accepting multiple
`--pngs` directories and resolving the right one per score (whichever directory
actually has that score's own subdirectory), rather than hardcoding one.

**Real visual validation, not just unit tests.** Ran the script against
IMSLP396671 (18 eligible systems, all previously spot-checked this session for
OCR-first) and inspected two of the resulting crop images directly against their
own written token sequences: `IMSLP396671-sys0-v0.png` (the vocal staff, 3/4 time,
"Why fair Maid...") shows F4-eighth, Bb4-eighth as its first two notes, exactly
matching the token file's `note_8 F4` / `note_8 B4 b` opening; `IMSLP396671-sys0-
v1.png` (the piano grand staff) shows both treble and bass clef in one crop,
matching the token file's own two-clef (`clef_G2`/`clef_F4`) grandstaff preamble.
Both real, both correct.

**A 30-score batch validation** (`extract_stage2_pairs.py` against the first 30
scores of `trusted_pool_scores.txt`): **23/30 scores succeeded, 437 pairs
extracted** (~19 pairs/score on average). The 7 failures are all real, legible,
*pre-existing* limitations in the shared `music_xml_parser.py` (not new bugs from
this module) - "Octave shift isn't supported" (6 of 7) and "Backup duration is too
long" (1 of 7) - the same parser `convert_lieder.py`'s own synthetic pipeline
already depends on and already has this same limitation with. Not fixed here;
out of scope for this pair-extraction task, and pre-existing across every corpus
this parser already serves.

### Update, 2026-08-25: full-scale Stage 2 extraction run complete - 2,535 real training pairs

Ran `extract_stage2_pairs.py` across all 266 eligible single-piece scores once
`staffboxes121` genuinely finished and freed the box's background capacity.
**Result: 169/266 scores succeeded, 97 failed, 2,535 total (crop, token-sequence)
pairs written** to `/workspace/b0/olimpic-probe/stage2_pairs_out` with a manifest
at `/workspace/b0/olimpic-probe/stage2_pairs_manifest.txt` in exactly the
`data_loader.py`-ready `"image_path,tokens_path"` shape.

**Failure breakdown** (all real, legible, all pre-existing limitations in the
shared `music_xml_parser.py` this module reuses - none are new bugs in this
module's own code, confirmed by reading the traceback for the one previously-
unseen failure kind below):

| reason | count |
|---|---|
| Octave shift isn't supported | 72 |
| Backup duration is too long | 12 |
| Octave change isn't supported | 10 |
| Image size exceeds decompression-bomb limit | 2 |
| `list index out of range` | 1 |

The 36.5% full-run failure rate is noticeably higher than the 30-score sample's
23% - expected, since a small early sample skews toward whatever engraving
conventions happen to appear first; the two new-to-the-full-run failure kinds
("Octave change isn't supported", the decompression-bomb image-size limit
already documented for the OCR-first pass) are the same *kind* of pre-existing
limitation, not a new failure mode. The one `list index out of range` case
(`IMSLP441122`) was traced directly (not assumed) to a real parser edge case
inside `music_xml_parser.py`'s own `TokensPart.append_note`, several calls deep
from this module's own `extract_score_pairs` - confirms this is a pre-existing
parser bug, not a bug in the new extraction code. None of these failure kinds
were fixed here - out of scope for this pair-extraction task, and shared,
pre-existing limitations `convert_lieder.py`'s own synthetic pipeline already
has too.

**A second, independent visual spot-check**, beyond the earlier IMSLP396671
check: `IMSLP89026-sys5-v0.png` (vocal) and `IMSLP89026-sys5-v1.png` (piano grand
staff) both show the real printed lyrics "plough-song, my father will sing the
plough-song That" - the same text appearing in both crops confirms they're the
same system's two voices, correctly paired; the piano crop shows a real 1-sharp
key signature and proper treble+bass grand-staff structure, matching the token
file's own `keySignature_1` and two-clef preamble. Two different pieces now
independently confirmed correct by eye, not just by test coverage.

**Status: Stage 2 pair extraction is now genuinely complete for the 266-score
single-piece eligible pool** - 2,535 real (crop, MusicXML token-sequence) pairs
from actual historical IMSLP scans, ready to mix into an existing transformer
training run via `data_loader.py`'s own manifest format. Not yet done: the 56
collection scores this section always deferred (`match_collection_pages.py`
would need to resolve them first), and a larger systematic spot-check sample
before actually launching a training run with this data (two manually-verified
examples is a real, positive signal, not a substitute for a proper sample review
the way `compare_verified.py`/`/compare` gave the box-detection and bar-count
work earlier this session).

### Update, 2026-08-25: a real user-found bug in the extracted pairs, root-caused and fixed

Building the review website (below) immediately paid for itself: the user found
a real correctness bug by eye within minutes of browsing it - a rendered pair
(`IMSLP10416-sys0-v1`) was visibly missing the last bar(s) compared to its own
scan crop, and a second pair (`IMSLP10416-sys4-v1`) was "1 bar off." Both traced
to the same root cause, not two separate bugs.

**Root cause**: `bar_count_rows.json` (which `eligible_system_positions` reads to
decide which systems are safe to extract) had been computed against the *older*
`imslp_systems_new_repaired` detection - but `extract_stage2_pairs.py`'s actual
crops come from the *newer* `imslp_systems_with_staff_boxes` detection (built
later this session, after the ledger-line/padding fix). The two detections'
system box boundaries disagree for a meaningful fraction of scores, so
eligibility was being judged against different content than what actually got
cropped and sliced. Confirmed directly, not assumed: `IMSLP10416` system 0's box
differs between the two runs (old: left=439/width=1950; new: left=409/width=2010,
i.e. wider), and a fresh bar-line count against the current detection gives 4,
not the stale row's recorded 3.

**Fix**: re-ran `compare_bar_counts.py` against the current
`imslp_systems_with_staff_boxes` detection (284 scores, ~2 hours - slower than
expected due to CPU contention with other work this session, plus one real
transient GitHub connectivity blip mid-run that self-resolved within ~15
seconds), replaced `bar_count_rows.json` with the corrected result (old backed up
as `.stale-2026-08-25`), recomputed eligibility, and re-ran the full extraction.
**Score-level eligibility was unchanged** (still 266/266 eligible scores, 169/266
succeeding - the same scores that could extract before still can, matching the
same known pre-existing MusicXML-parser limitations from before), but
**system-level** eligibility shifted: 2,333 eligible systems now vs. 2,293 before
(individual systems flipped in and out, not whole scores), yielding **2,603 total
pairs** (up from 2,535).

**Verified directly on both reported cases**, not just re-run and assumed
correct: `IMSLP10416-sys0-v1` no longer exists as an output pair at all - the
corrected detection shows this system as a genuine 4-vs-3 bar-count mismatch, so
it's now honestly *excluded* rather than silently extracted with the wrong
measure range as before (the right behavior: no data beats wrong data).
`IMSLP10416-sys4-v1` now shows an exact 4-vs-4 match and its token file has
exactly 4 `barline` tokens, matching what the user found by counting bars in
their own crop.

**A real, if minor, operational hiccup along the way**: the corrected extraction
run hit a genuine mid-run stall on one score's MusicXML fetch (confirmed via
`/proc/<pid>/status` showing it blocked in `do_poll` network wait, plus a
directly-observed GitHub connectivity timeout at the same time) - killed it,
computed the remaining unprocessed score list, and relaunched a continuation run
that finished cleanly. This left ~750 duplicate manifest lines (the same score
re-processed by both the original and continuation runs, since
`extract_stage2_pairs.py` has no skip-if-already-done check and the manifest is
opened in simple append mode) - harmless (same real files, not corrupted data)
but wasteful; deduplicated by exact line before finalizing. The one score that
had stalled (`IMSLP588848`) was re-run cleanly afterward on its own and turned
out to fail on the same pre-existing "Octave shift isn't supported" parser
limitation several dozen other scores already hit - the stall was a genuine
transient network issue, not a sign anything was wrong with that score's data.

`render_stage2_tokens.py` was re-run against the corrected `stage2_pairs_out`,
and `stage2_pair_review_server.py` (tmux session `stage2review`, port 8792) was
restarted with the corrected manifest - the review site now reflects genuinely
fixed data.

### Update, 2026-08-25: review sites built - a Stage 2 pair reviewer and a Stage 3 text reviewer, merged into one server

Built `training/omr_datasets/stage2_pair_review_server.py` (stdlib `http.server`,
matching `review_server.py`'s own house style) so the 2,535+ extracted pairs
could actually be spot-checked systematically instead of one-off - each pair
shows its own scan crop, a notation image **rendered directly from its own token
file** via `homr/music_xml_generator.py`'s existing `generate_xml` + MuseScore
(`render_stage2_tokens.py`, new - batches ~200 files per `mscore -j` conversion-
job invocation rather than one process launch per file, which measured ~3x
faster), a compact pitch-sequence summary, the raw token table, and Good/Bad/
Unclear buttons persisting to a judgments file.

Also built `stage3_text_review_server.py` for the OCR-first lyrics/dynamics
matches (22,086 of them) - same pattern, on-the-fly crops around each match's own
detected box rather than a pre-render step (cheap enough not to need one).

**Merged into one process/port on explicit user request** (they didn't want to
forward a second port): refactored `stage3_text_review_server.py` to expose pure
`render_index`/`render_score_page`/`crop_bytes` functions parameterized by a
`base_path`, and `stage2_pair_review_server.py` now optionally mounts them under
`/text` via `--text-matches`/`--text-judgments`/`--text-pngs`. Both served from
port 8792 now - `/` for pairs, `/text` for lyrics/dynamics. 40 tests passing
across the three new/touched modules.

### Update, 2026-08-25: a real multi-verse lyrics bug found via the review site, root-caused and fixed

Browsing the new `/text` section, the user noticed strophic Lieder (2+ printed
verses under one staff) were only ever getting *one* verse confirmed, and it
wasn't consistently verse 1. Root cause: `extract_expected_texts` discarded
MusicXML's own per-note `<lyric number="N">` verse tag, so different verses'
syllables got flattened into one sequence and `words_from_syllables` interleaved
them into corrupted words; the matcher's fuzzy *set* comparison (word order
doesn't matter) would then non-deterministically confirm whichever verse's
garbled fragments happened to resemble real OCR text. Checking a real sample (20
scores) confirmed this is common in this corpus, not an edge case - most matched
pieces carry 2 verses, at least one carries 4.

**Fix**: kept the verse number on each extracted lyric entry
(`musicxml_text_ground_truth.py`), added `words_by_verse()` to reconstruct each
verse's words independently rather than pooling them, and added
`match_verses_to_ocr()` (`ocr_first_text_ground_truth.py`) to match each verse
against a page's OCR lines separately, excluding lines already claimed by an
earlier verse so one physical printed line can't be double-counted as two
different verses. 13 new tests.

**Validated on a real multi-verse score** (`IMSLP10602`, Schubert's *Die
Forelle*, verses 1+2): before, 79 pooled/undifferentiated matches; after, 71
matches cleanly split verse 1 (52) / verse 2 (19). The ~9% count drop was traced
directly (not assumed) to OCR-noisy fragments (e.g. a stray-underscore artifact
`"Bäch _ lein"`) that only cleared the old pooled-set threshold by accident -
fewer but more trustworthy matches, and multi-verse content is now visible and
correctly attributed by verse.

**Status: built and validated, full corpus re-run deliberately held back** per
the user's own explicit choice ("build and validate now, hold the re-run") -
the `/text` review site still shows pre-fix data until the user says go ahead
on the multi-hour re-extraction.

### Update, 2026-08-25: THE root cause - a one-measure off-by-one in every system range in the whole corpus

Everything in this section that depended on ground-truth measure ranges was built
on a single, systematic, corpus-wide bug. Found by building the content-level
verification the user proposed ("scan with homr and compute measure similarity";
they had already noticed "for many of the pairs, they were only shifted, but
contained the correct passage" - exactly right).

**The bug** (`fetch_lieder_ground_truth.py`'s `measures_per_system`): a MuseScore
`LayoutBreak` attached to a measure breaks *after* that measure, so the measure
carrying it is the **last** of its own system. The code closed the current system
*before* counting that measure, which pushed every break-carrying measure into the
**following** system. Net effect: **every system in every piece was assigned a
measure range exactly one measure late**, for the entire corpus. The old unit
tests encoded the wrong behaviour too (`[[2, 2]]` where the right answer is
`[[3, 1]]`), so nothing caught it.

**How it was found - content fingerprinting, not bar counting.** New modules
`fingerprint_measures.py` + `build_ground_truth_tokens.py`: run homr's own Stage 2
transformer (`Staff2Score`, the same one production uses) over each real crop,
flatten its reading to a pitch-token sequence, and align that against the same
piece's real MusicXML note stream (`difflib.SequenceMatcher` matching blocks).
Whatever ground-truth measure range the aligned region lands in *is* the crop's
true range, regardless of what bar counting believed. Alignment is deliberately on
the **flat note sequence, not per measure** - homr's own bar-line reading can be
wrong too, so anything assuming the crop's measure boundaries agree with ground
truth's would inherit the very error being corrected.

**The evidence was unambiguous.** Across every piece sampled (IMSLP148200,
IMSLP10416, IMSLP10602, IMSLP154060) the *dominant* recovered offset was a
consistent **+1** (3, 6 and 4 systems respectively in the three multi-system
samples). And after the fix, the corrected ranges match the independently
recovered alignments exactly - for IMSLP148200: sys1 recovered (3,6) vs assigned
(3,6); sys2 (6,9) vs (6,9); sys4 (12,15) vs (12,15); sys5 (15,17) vs (15,17);
sys3 recovered (11,12) inside its assigned (9,12), the difference being that
measures 9-10 are rest-only and so contribute no fingerprint. `measures_per_system`
now yields `[[3,3,3],[3,3,2,2],[2,2,2,3]]` = 28 measures for that piece, matching
its `.mxl`'s own 28 exactly.

**A correction to this section's own earlier claim.** The 2026-08-25 entry above
reasoned that a stale-vs-fresh *detection* mismatch was the cause of the
user-reported missing/off-by-one bars, and separately speculated about
"cutter drift" downstream of a locally-mismatched system. The stale-detection
mismatch was real and worth fixing on its own, but it was **not** the cause of the
shifts the user saw - this off-by-one was. An intermediate diagnosis in that same
investigation (that IMSLP148200 sys3's crop "contains sys4's content") was also
wrong: the crop is rest, rest, content, and its content is measure 11, which sits
correctly inside the *corrected* (9,12) range. Recorded here because the wrong
intermediate conclusions were stated with more confidence than the evidence then
supported, and the corpus-wide lesson is the opposite one: **bar counts cannot
validate content alignment; only content can.**

**The known limitation of the fingerprint method**, visible in the same data: large
negative offsets (-15, -20, -22, -29, -45) appear for a minority of systems. These
are repeated passages - a strophic song's later verse aligning to its first,
identical occurrence. The method finds *a* true match, not necessarily *the* one,
wherever music genuinely repeats. Any use of these recoveries to *correct* ranges
must handle that (e.g. prefer the candidate nearest the expected position rather
than the globally best block); the `trusted` flag alone does not cover it.

**What this invalidated, and the rebuild order.** Everything downstream of the
ranges: the cached `imslp_lieder_ground_truth/*.json` themselves (moved aside as
`.buggy-layoutbreak-2026-08-25`), `bar_count_rows.json` and therefore all
eligibility, all 2,603 extracted pairs, their renders, and the Stage 3 lyric
page-ranges (the in-flight verse-aware re-extraction was killed at 27/355 rather
than finish onto a wrong foundation; the re-render was killed at 1,559/2,603 for
the same reason). Rebuild order is bottom-up: ground truth -> bar counts ->
eligibility -> pairs -> renders + Stage 3. The low 46.0% bar-count exact-match
rate reported earlier is itself suspect under this bug and should be re-measured,
not carried forward.

**Rebuild verification (first 40 regenerated scores).** Every one of the 40 differs
from its buggy predecessor - the bug really was corpus-wide, not a subset - while
**total measure count per piece is preserved exactly** (2238 = 2238 across the 40),
which is the correct invariant: the fix redistributes measures across system
boundaries, it does not lose or invent any. The per-piece signature matches the
theory precisely: the first system gains one measure, interior systems keep their
size, the final system loses one (e.g. IMSLP10416 page 1 `[3,5,5,5]` -> `[4,5,5,5]`).

This also closes the loop on the user's original bug report. The first pair they
flagged, `IMSLP10416-sys0-v1` ("missing the last bar(s)"), had a bar-count row of
`detected=4, ground_truth=3` - so its token file really was built one bar short of
its own crop. Corrected ground truth now says 4 for that system, matching the
detector exactly. The visible symptom, the "stale detection" red herring, and this
off-by-one were all one bug.

### Update, 2026-08-25: what "Stage 2 training" actually requires on this box

Groundwork done while the rebuild ran, prompted by the user asking for a Stage 2
training run to follow the corpus rebuild. Two hard constraints surfaced, both of
which would have made a naive launch either destructive or impossible:

**1. The default training command auto-downloads and converts missing corpora.**
`train_transformer`'s `_check_datasets_are_present` calls `convert_lieder()` /
`convert_grandstaff()` / `convert_primus()` / `convert_musetrainer()` for any index
that is absent. On this box only **pdmx** (35,800 files) is built; the other four
are missing, and `convert_lieder`'s own docstring warns it "can take up to several
hours" (it installs MuseScore and re-renders every piece to SVG). Running
`python -m training.train transformer` unattended would therefore have started
many hours of downloading and rendering rather than any training.

Fixed by making the dataset choice explicit and safe: `train_transformer` now
takes `dataset_index`/`dataset_weights`, and **naming indexes explicitly suppresses
the auto-convert path entirely** - the named indexes are verified to exist and the
run refuses with a clear error otherwise, rather than silently starting a download.
The default five-corpus interactive path is unchanged. Three tests cover exactly
this (`tests/test_train_transformer_datasets.py`), including that the default path
still reaches `_check_datasets_are_present`.

**2. A warm start had no mode.** `fine_tune=True` loads the pretrained checkpoint
but freezes the encoder *and* decoder, unfreezing only the lift head - far too
narrow for adapting to a genuinely different data distribution (real historical
scans vs. the synthetic renders every existing index is built from). The only
alternative was `TrOMR(config)`, i.e. from scratch, discarding the pretrained
weights. Added `warm_start=True`: loads the same checkpoint with **every parameter
trainable**, at `3e-5` - deliberately between the `1e-4` from-scratch rate (which
would wash out the pretrained weights) and the `1e-5` frozen-backbone rate (slower
than a full-model adaptation needs).

**3. `transformers` is not installed anywhere on the box.** The main Stage 2
trainer uses HuggingFace `Trainer`/`TrainingArguments`, but neither
`/workspace/b0/homr/.venv` nor `/venv/main` has `transformers` - it is declared
only in pyproject's **dev** dependency group, and this box has a runtime-only
install. This is why every prior phase (20/24/25) used `train_staff_context.py`
instead, which is a **pure PyTorch loop with no HuggingFace dependency**. So Stage
2 training here needs a dependency install first, and that install must be
surgical: `torch 2.13.0+cu129` and `onnxruntime-gpu` are working and are what every
inference/segnet job on this box depends on, so `transformers` must go in with
`--no-deps` (plus only its genuinely missing pure-python deps) rather than being
allowed to resolve and upgrade torch underneath a working environment. It must
also wait until the bar-count job finishes, since that job is running inside the
same venv right now.

**Mix proportion, for the record.** `mix_training_sets` ignores `weights` entirely
when `number_of_files < 0` (it takes every file from every source), so a
pdmx + IMSLP run at the default `-1` puts the new real-scan data at roughly
**2,600 / 38,400 = 6.8%** of the mix. That is an honest first experiment but a
dilute one; oversampling the IMSLP index is the obvious follow-up lever and is a
modelling decision worth taking deliberately rather than by default.

**What real-scan training data actually exists here (checked properly, after
getting it wrong once).** The user asked to fine-tune on "only the new scans
(lieder and omr-ossq)". Both halves are available:

| source | examples | real scan? | usable for training? |
|---|---|---|---|
| OSSQ **scanned** (`phase7`) | 32,982 train + 3,571 valid | yes | yes - already built |
| new IMSLP Lieder pairs | ~2,600 | yes | yes - the genuinely new data |
| OSSQ synthetic (phase14/15) | 42,088 | no | excluded - wrong distribution |
| pdmx | 35,800 | no | excluded per user |
| OLiMPiC scanned dev / test | 1,350 / 1,381 | yes | eval splits - do not train on |

**A wrong claim, corrected.** An earlier version of this section asserted that the
OSSQ scanned crops "were never built here", that `data/images` was empty, and that
`omr-data-preprocessor` "is not set up" - and concluded the run would have to fall
back to the ~2,600 Lieder pairs alone. Every part of that was wrong, and the user
caught it by simply asking whether both machines had actually been checked:

- `omr-data-preprocessor` **is** present at `/workspace/b0/omr-data-preprocessor`,
  with its `ossq_step_001..005.sh` pipeline scripts.
- The OSSQ corpus is **12G**, essentially all of it under `ossq-omr/scores/`. The
  original check listed only `ossq-omr/data/` (404K of tsv/yaml) and generalised
  from that one directory to the whole corpus.
- **44,682 scanned partwise staff crops** exist, across 93 scores
  (`scores/*/*/images/scanned/partwise/`), alongside 52,973 synthetic ones.
- `phase7` is **already the converted OSSQ scanned training index** - not a second
  synthetic one. Proven by `md5sum`: `phase7/train/sq7383977_0003_0001_1.png` and
  `images/scanned/partwise/sq7383977:0003:0001:1.png` are the same file.

The original mistake was calling the track "synthetic, confirmed by eye" from a
thumbnail. OSSQ scanned crops *look* clean because they are tight single-staff
crops of well-printed historical string-quartet parts - nothing like the obvious
page texture of an OLiMPiC piano-vocal scan. Appearance was the wrong instrument
for that question; the checksum was the right one and was available the whole time.

So the fine-tune set is **OSSQ scanned (phase7) + the new IMSLP Lieder pairs**,
roughly **35,600 real-scan examples** - not the 2,600 the wrong reading would have
settled for. Synthetic OSSQ and pdmx stay excluded per the user's instruction, and
both OLiMPiC scanned splits stay untouched as evaluation data.

### Update, 2026-08-25: post-fix bar-count result, and what ENSEMBLE_TRANSCRIPTION_DESIGN.md requires of the training run

**The re-measured bar-count rate: 49.5%** (2,283 exact of 4,612 systems compared,
330 scores), up from **46.0%** under the bug. Total-system-count mismatches fell
from 163 to 149. Eligible systems rose from 2,333 to **2,443** across the same 266
scores.

**That is a smaller gain than predicted, and the prediction was the problem, not
the fix.** A stated bar of ">=55% or don't train" was set earlier; it was the wrong
instrument and is withdrawn as a gate. The off-by-one changes *which* measures a
system maps to, but it only changes the *count* for the **first and last** system
of each piece - every interior system keeps the same measure count at a shifted
position. Bar counting is therefore nearly blind to this bug by construction, and
a ~3.5pt gain concentrated in first/last systems is exactly what the fix predicts
rather than evidence it underperformed. The real validation remains the content
fingerprinting, where recovered ranges matched corrected assignments exactly and
unambiguously. The user's own criterion was "high pair eligibility", which is the
direct measure and is what should decide the training run.

**What `ENSEMBLE_TRANSCRIPTION_DESIGN.md` requires that this plan must respect.**
Pulled in on the user's prompt; it materially constrains the fine-tune:

- **§14.4 independently specifies the mode built here.** "The current fine-tuning
  path that freezes most of the model and trains only lift is not suitable for this
  domain adaptation and needs a separate explicit mode." That is exactly the new
  `warm_start`. §14.4 also asks for **staged** unfreezing (last decoder layers
  first, visual encoder last, at an even lower rate) and **separate learning rates**
  for pretrained vs. new parameters - `warm_start` currently unfreezes everything
  at once at a single 3e-5, so it satisfies the requirement's intent but not yet
  its detail.
- **§13.6 already predicted the sampler bug found independently here.** "The
  sampler must implement its declared dataset weights even when all files are
  loaded. The current mixing behavior should be audited because concatenating all
  datasets before shuffling can make configured weights ineffective." That is
  precisely `mix_training_sets`: `number_of_files < 0` routes to
  `_take_all_training_sets`, which concatenates and **ignores weights entirely**.
  The audit §13.6 asked for is now done and the answer is yes, it is broken.
- **§13.5 forbids what `load_dataset(val_split=0.1)` does.** "All crops, systems,
  pages, movements, and source variants from one score belong to one split. Never
  randomly split staff strips... No data-loader fallback may silently create a
  sample-level random split." The default val split is exactly a sample-level
  random split, so running it over the IMSLP pairs would put systems from the same
  score in both train and validation. OSSQ `phase7` already ships score-disjoint
  `train/` and `valid/` directories; the IMSLP pairs have no split at all and need
  a score-disjoint one built before training.
- **§13.4 and §23 warn about scanned labels and forgetting.** Scanned editions can
  disagree with the symbolic source on beams and slurs even where pitch and rhythm
  are right, so scans are for "legacy pitch/rhythm adaptation... while masking
  uncertain beam/slur losses". And "Adding OSSQ only can specialize the model at
  the expense of other scores. Retain general-data replay". The user's instruction
  ("only the new scans, not pdmx") is in direct tension with that replay
  requirement; warm-starting from the pretrained checkpoint is a partial mitigation
  but not the one the doc names. Flagged rather than silently resolved either way.

### Update, 2026-08-25: recovering the excluded systems by content, and the replay decision

**Pair re-extraction after the off-by-one fix: 2,761 pairs** (168 scores, 98 failed
on the pre-existing MusicXML-parser limitations), up from 2,603 and originally
2,535.

**Recovering what eligibility throws away.** Eligibility rejects ~47% of detected
systems - 2,169 of 4,612 - on a *proxy*: a bar-line miscount means the counter and
the layout disagree, not that the crop is unusable. `recover_excluded_pairs.py`
runs homr's own Stage 2 transformer over each excluded crop, aligns what it reads
against the piece's real MusicXML note stream, and takes the measure range the
content actually lands in. `slice_voice_measures` walks a fresh `MeasureCutter`
forward from the start of the voice so a mid-piece slice still carries the
clef/key/time in effect, rather than starting bare.

**A real bug in that tool, found by an independent check rather than by the tool's
own confidence.** First real run recovered `IMSLP10602-sys11-v0` as measures
(41,44) with `coverage=1.00, similarity=1.00` - a perfect score. It was wrong.
Checking the *lyrics* of the recovered range (independent evidence: the alignment
uses only pitches) showed (41,44) reads "...Be-trog'ne an**,** und" while the crop
plainly shows "...Be-trog'ne an**.**" - the text of (45,48), the expected position.
Die Forelle is strophic: both candidates sat inside the same window, both matched
perfectly, and `difflib` prefers the *earliest* matching block, so recovery
silently biased backwards by four measures. A perfect similarity score says the
passage matches, not that it is the right instance of a passage that repeats.

**Fix: widen the window in stages instead of opening it fully at once**, taking the
first trusted alignment. The nearest candidate then becomes the only candidate, and
a distant repeat is reachable only when nothing closer explains the crop at all.
Verified on the same real data: `IMSLP10602-sys11-v0` moved (41,44) -> **(45,48)**,
the correct range; recoveries that were already stable were unchanged; and yield
*rose* (10 -> 14 recovered, 8 rather than 12 refused) because narrow windows also
let four piano-grandstaff voices align that a single wide window had drowned. Two
tests pin both directions - a near repeat must not pull the match backwards, and a
genuinely distant match must still be reachable.

**Replay decision.** The user resolved the §23 / §14.4 tension directly: include
"only enough pdmx to prevent catastrophic forgetting". Because `mix_training_sets`
ignores `weights` entirely when `number_of_files < 0` (§13.6's predicted bug,
confirmed here), a replay *fraction* is only expressible with a **positive**
`number_of_files`. Working through `_calc_number_of_files_to_take`: with weights
set proportional to the desired per-source counts and `number_of_files` equal to
their sum, every `max_available_ratio` is 1, the limiting ratio is 1, and each
source contributes exactly its target - so the ratio is honoured exactly rather
than approximately. Planned mix: OSSQ scanned (phase7 train, 32,982) + IMSLP pairs
(~2,761 plus recovered) + **~15% pdmx replay** (~6,400 of 35,800). 15% is a
deliberate, adjustable choice - minimal, but in the range normally sufficient to
mitigate forgetting - not a value derived from anything measured here.

**Recovery run 1 died at 105 of 266 scores** - no traceback, no summary line, the
log simply stops after a successful score, which is the signature of an external
kill rather than a failure the program saw. 455 pairs had been written by then
(2,761 + 455 = 3,216 token files on disk, consistent).

Two robustness gaps that the kill exposed, both now fixed rather than worked
around by hand:

- **The audit report was a single write after the loop**, so a killed run loses its
  entire trail. It is now rewritten after every score.
- **Nothing distinguished "processed, recovered nothing" from "never reached"** -
  the per-score line only printed when the count was non-zero, so the log could not
  be used to resume. It now prints unconditionally, and `--skip-logged <previous
  log>` reads back the scores a prior run reported on (recovered *or* failed) and
  skips them. That matters specifically because this tool appends to its manifest
  and has no skip-if-exists: a naive re-run would have duplicated hundreds of lines
  rather than resuming. Two tests cover the log parsing, including that summary and
  noise lines are not mistaken for score records.

Resumed cleanly: "resuming: skipping 105 score(s) already logged", now working
through the remaining 161.

**The cause of the kills, found on the second one: an uncaught
`PIL.Image.DecompressionBombError`.** Run 2 died the same way after 16 scores and
this time left the traceback - "Image size (240731800 pixels) exceeds limit of
178956970 pixels". PIL refuses images past ~179M pixels as a suspected
decompression bomb, and it *raises* rather than skipping. In this tool the page
load sat outside any `try`, so a single oversized page killed the whole run
mid-corpus. It also explains run 1's silent death at 105 scores.

Notably this is the *same* oversized-page problem the OCR-first pipeline hit
earlier with `IMSLP557090`; that pipeline survived it only because its image work
happened to sit inside a `try`, so the page was lost instead of the run. The
difference was luck, not design.

Fixed on both axes, and in both tools:

- `Image.MAX_IMAGE_PIXELS = None` in `recover_excluded_pairs.py` **and**
  `extract_stage2_pairs.py`. The guard exists to protect against hostile uploads;
  these are our own full-resolution IMSLP scans, and some are legitimately that
  large. `extract_stage2_pairs.py` had the same latent bug - its page load is
  inside the per-score `try`, so an oversized page there silently cost the entire
  *score* rather than the page.
- A `try/except` around page loading in the recovery loop, so any unreadable page
  costs that page and nothing else.

The lesson worth keeping: a per-score `try` that "handles" errors can mask a bug
by converting a crash into silent data loss. Run 1 looked like an infrastructure
kill precisely because nothing in the program reported anything.

### Update, 2026-08-25: recovery complete, and the Stage 2 scans fine-tune is running

**Recovery finished: 499 recovered / 510 not trusted / 198 skipped** on the final
run, **954 recovered pairs in total** after deduplication (runs 1+2 contributed 455
before the DecompressionBomb crashes). Combined corpus:

| source | pairs |
|---|---|
| extracted (eligible systems) | 2,761 |
| recovered (previously excluded) | 954 |
| **total** | **3,715** |

Recovery added **+35%** on top of extraction, from systems bar-counting had
rejected. The 510 "not trusted" are refusals, not failures - the tool declines
rather than guessing, which is the whole point given a wrong pair is worse than a
missing one.

**Score-disjoint split** (§13.5): 3,353 train / 362 validation, **0 score overlap**,
150 vs 11 scores.

**Environment work needed before training could run at all.** `transformers` was
absent from every environment. Three real traps on the way in, each caught by
verifying rather than assuming:

- `which pip` resolved to `/usr/bin/pip`, the *system* interpreter - the venv ships
  `pip3`, not `pip`. The first install attempt was therefore aimed at system Python
  and was (correctly) refused by PEP 668. Used `./.venv/bin/python -m pip` instead.
- Installing latest pulled **transformers 5.15.1**, outside pyproject's own
  `>= 4.53.2, < 5` pin. v5 has breaking Trainer API changes and `train.py` imports
  those APIs directly, so the pin was honoured rather than hoped past: pinned back
  to **4.57.6**, with `huggingface-hub` and `tokenizers` pinned to the ranges it
  actually requires (`0.23.0` turned out never to have been released - only
  `0.23.0rc0` then `0.23.1` - so `tokenizers==0.22.2`).
- The first launch died on `TensorBoardCallback requires tensorboard`. An earlier
  reading of "tensorboard 2.20.0 present" had come from the *system* pip list - the
  same `which pip` trap, believed once and wrong twice.

Everything installed `--no-deps` throughout, and verified after every step that
`torch 2.13.0+cu129 / cuda True`, `onnxruntime 1.22.0` and `numpy 2.5.2` were
untouched, since the whole inference and segnet stack on this box depends on them.

**Run manifest** (§14.7 asks for this explicitly):

| setting | value |
|---|---|
| launcher | `training/transformer/train_scans.py` |
| mode | `warm_start` (pretrained ckpt, all params trainable) |
| checkpoint | `pytorch_model_426-b6fd2080...pth` |
| mix | OSSQ scanned 32,982 + IMSLP scans 3,353 + pdmx replay 6,400 (15.0%) = 42,735 |
| after `_filter_valid_samples` | 42,137 train / 362 val |
| validation | explicit score-disjoint index (not `val_split`) |
| learning rate | 3e-5, cosine, warmup_ratio 0.1 |
| epochs | 15 |
| per-device batch / grad accum | 8 / 4 (effective 32) |
| precision | bf16 |
| dataloader workers | 12 |
| seeds | HF default 42; `mix_training_sets` fixes its own np seed 1720697007 |
| total steps | 19,755 |

**On whether the instance is wrong for this** (the user asked, offering to rent
something else): no. Steady state is ~2.8-3.5 it/s for an ETA around 1h40m - the
288-hour figure visible at step 1 was `torch_compile` warmup, not the real rate.
More importantly **GPU utilisation is ~41% and VRAM use is 3.8 GB of 49 GB (7.7%)**,
so the run is not GPU-bound and a larger card would idle harder rather than finish
sooner. §14.7 anticipated exactly this: "The data loader is CPU-sensitive. Measure
GPU utilization before buying more GPU capacity." The real levers are on this same
box - batch size (8, on a 48 GB card) and worker count (12 of 128 cores) - and both
are better tuned deliberately for a *next* run, with this one as the baseline,
since batch size also changes optimisation dynamics rather than being a free speed
knob.

For the same reason the renders and Stage 3 re-extraction are deliberately **not**
being run concurrently: they are CPU-heavy, and this run's bottleneck is CPU.

**Stage 3 training: investigated, and deliberately NOT launched unattended.** The
user asked to begin it once Stage 2 finished. Reading the code first (rather than
launching and finding out) shows two blockers that make an unattended launch the
wrong move:

- `detector_masks.py` is a *data-prep* script; the trainer is
  `training/ocr/train_detector.py` (`--index` a detector_masks index, `--valid-index`
  a `detector_split.py` split, `--weights`, `--epochs`, ...). Both the data and a
  valid split already exist (`detector_masks_v4/index.txt`, 2,889 masks, plus
  `detector_masks_v4/split/valid_index.txt`), so it *would* run.
- But that data's source images live under `/workspace/b0/mbox/` - **synthetic
  MuseScore renders**. Retraining on it reproduces the existing `detector4` on the
  existing distribution and uses none of this session's work.

The valuable run would use the new **real-scan** lyrics/dynamics ground truth from
`ocr_first_text_ground_truth.py`. That cannot be converted naively, because it
covers **2 of the detector's 7 classes** (Lyrics and Dynamic). Rasterising those
boxes into masks and training would mark every Tempo, StaffText, Expression,
Fingering and MeasureNumber region on those pages as **background** - teaching the
model those classes do not occur on real scans. On a detector whose precision has
already collapsed for exactly those weaker classes (§1), that is not a neutral
mistake; it would make the known problem worse.

§13.4 names the requirement directly: "attach per-head label confidence/masks, not
only one sample confidence... use scanned staff images for legacy pitch/rhythm
adaptation when those labels are valid while masking uncertain beam/slur losses."
The equivalent here is a **per-class loss mask** - supervise Lyrics and Dynamic on
these pages, and leave the other five classes unsupervised rather than implicitly
negative. That is a real change to the training objective, not a conversion
script, and it is a modelling decision worth making deliberately rather than at
04:00 with nobody to review it.

Recommendation, for the morning: either (a) build the per-class-mask path and
train on synthetic + real-scan mixed, which is the version that uses this
session's Stage 3 data, or (b) retrain on synthetic only as a pure baseline
refresh, knowing it adds no new information. Not doing (c) - converting the new
data without masking - is the actual finding here.

**Can we fine-tune only the Lyrics/Dynamic heads? Checked the architecture; the
answer shapes the whole approach.**

*There are no separate heads.* The detector is one `smp.Unet` (resnet18 encoder)
with a single output convolution producing 8 channels - background plus
`Dynamic, Fingering, Expression, Tempo, MeasureNumber, StaffText, Lyrics`
(`detector_masks.CLASS_ORDER`, background 0, `CLASS_INDEX` 1-based). Nothing
per-class exists to freeze.

*The loss does support it, though.* `smp.losses.DiceLoss` takes
`classes: Optional[List[int]]` ("classes that contribute in loss computation") and
`ignore_index` - verified against the installed signature, not assumed. So
`DiceLoss(MULTICLASS_MODE, classes=[Dynamic, Lyrics])` is a real, one-line way to
train only those two.

Two caveats decide the design:

1. **Softmax couples the channels.** Even with only Dynamic and Lyrics in the loss,
   raising their logits mechanically lowers the other six at the same pixels. There
   is no explicit penalty on Tempo, but there is implicit pressure, and freezing
   cannot prevent it because the coupling lives in the shared output conv.
2. **Our boxes are high-precision but low-recall** - and this matters more. The
   OCR-first data only contains lyrics whose OCR text *matched* known MusicXML
   text. Every lyric the OCR missed would be rasterised as background, training the
   model to miss lyrics. That hazard is present even in a clean 2-class model, so
   "just use fewer classes" does not solve it.

**Suggested scheme - one mechanism fixes both:** *supervise positives only on real
scans, take negatives from synthetic replay.*

- Real-scan pages: pixels inside matched Lyrics/Dynamic boxes get that class;
  **everything else is `ignore_index`**, not background.
- Synthetic pages (`detector_masks_v4`): unchanged, full 8-class labels.
- Mix both in each epoch.

This is structurally the same move as the pdmx replay in Stage 2: the new data
teaches what a real scanned lyric looks like, the synthetic data preserves what
background and the other five classes look like. It removes the false-negative
pressure from unmatched lyrics *and* the implicit pressure on unlabelled classes,
without needing to know where the unlabelled regions are - which we do not.

Class count is then a secondary choice. Keeping the 8-class model (unchanged
inference path, `classes=[Dynamic, Lyrics]`) is preferred; a dedicated 3-class
model fully decouples from the collapsed classes but forks the pipeline, and is
only worth it if measurement shows the coupling actually hurts.

**Missing piece:** a mask builder that converts `imslp_ocr_first_ground_truth`
JSON into mask PNGs writing `ignore_index` outside matched boxes. `detector_masks.py`
does not do this - it rasterises full-label MuseScore boxes with background
everywhere else, which is exactly the behaviour that would be wrong here.

### Update, 2026-08-25: Stage 2 scans fine-tune - result, and stopped early on a plateau

**Best: `eval_accuracy` 0.93004 at epoch 11**, checkpoint saved to
`/workspace/b0/stage2_scans_best/` (copied from
`current_training/checkpoint-14487`, confirmed as `best_model_checkpoint` in the
trainer state). Full series on the score-disjoint held-out set:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval_accuracy | .9161 | .9233 | .9267 | .9274 | .9293 | .9288 | .9294 | .9292 | .9293 | .9297 | **.9300** | .9295 | .9296 |

**The headline is the plateau, not the peak.** Essentially all of the gain
(+1.3 points) arrived by epoch 5; epochs 5-13 moved within ±0.0006 and bought
about +0.001 total. The run was configured for 15 epochs and stopped at 13 on the
user's instruction to abandon it and move to Stage 3. **For any repeat: ~6 epochs
reaches the same place in under half the time.** Worth noting the validation set
is *held-out scores*, not held-out systems, so this is a genuine generalisation
number rather than a memorisation artifact.

**A second, cheaper lesson: the CPU contention was real and large.** When the
Stage 3 OCR re-extraction was started alongside it, training fell from ~2.5 it/s
to **6.35 s/it - roughly 16x slower** - and the ETA for the remaining 2,000 steps
jumped from ~13 minutes to 3.5 hours. That is the concrete cost of running
CPU-heavy work next to a dataloader-bound trainer, and it confirms the earlier
diagnosis (~41% GPU utilisation, 3.8 GB of 49 GB VRAM) that this run was never
GPU-bound. It is also why renting a larger GPU would not have helped.

### Update, 2026-08-25: Stage 3 (text detector) - tooling built, experiment matrix started

**`training/ocr/scan_text_masks.py`** (new, 12 tests): converts
`ocr_first_text_ground_truth.py`'s real-scan JSON into detector masks, writing
`IGNORE = 255` outside matched boxes rather than background - the point argued in
the previous section. `--background-outside` reproduces the naive behaviour
deliberately, so ignore-vs-background becomes a measured ablation rather than an
assumption, in keeping with `train_detector.py`'s own standing rule ("measure the
unweighted baseline before reaching for class weighting or focal loss").

**`train_detector.py` had no way to consume those masks**, which would have made
the mask builder useless: its loss is `CamVidModel`'s plain
`DiceLoss(MULTICLASS_MODE)` with no `ignore_index`, so a 255 would have been read
as a nonexistent class. Added `--ignore-index` and `--loss-classes` (names, e.g.
`Lyrics,Dynamic`), which rebuild the loss locally rather than editing the shared
`CamVidModel` - every other user of that model is unaffected. `ignore_index` is
threaded into `per_class_iou`/`evaluate` too: scoring ignored pixels would count
every unsupervised region as a prediction error and make the metric meaningless.
The flags are read with `getattr` because callers (including the existing tests)
build the `Namespace` directly rather than through argparse - the first version
broke six passing tests by assuming otherwise.

**An operational lesson worth recording.** Several runs were being starved and SSH
itself began refusing connections (load average 43). The cause was **orphaned
remote processes**: a local `timeout` kills the ssh *client*, but the remote
`pytest`/training process keeps running. Three abandoned test runs had accumulated.
Killing them dropped load 43 -> 11 immediately. When driving a remote box this way,
a timed-out command is not a stopped command, and the remote side must be checked
and cleaned explicitly.

**Experiment matrix (running/planned).** E0 is the frozen comparison §14.1 insists
on - without it none of the rest is interpretable:

| id | data | loss | question |
|---|---|---|---|
| E0 | synthetic only (`detector_masks_v4`, 2,582 train / 307 valid) | all classes | baseline: reproduce current detector behaviour |
| E1 | synthetic + real-scan (ignore-masked) | `classes=[Lyrics,Dynamic]`, `ignore_index=255` | does real-scan data improve Lyrics/Dynamic? |
| E2 | synthetic + real-scan (`--background-outside`) | same | does ignore-masking actually matter, or is naive background fine? |
| E3 | synthetic + real-scan (ignore-masked) | all classes | does restricting the loss to 2 classes help or hurt the other six? |

E1-E3 are blocked on the Stage 3 verse-aware re-extraction finishing, since they
need the real-scan boxes. E0 needs none of it and is running now.

**Stage 3 OCR extraction complete (2026-08-25).** Both passes finished, all 12 shards
exit 0: 329 score files, 324 with at least one match, **20,785 matched boxes - 16,855
lyric and 3,930 dynamic**. Four score ids appear in both the 355 and 121 sets
(`IMSLP618494`, `IMSLP621830`, `IMSLP622484`, `IMSLP650688`) and keep their 355-pass
version, since the extractor skips a score whose output already exists.

**One asymmetry in the E1/E2 ablation, stated rather than buried.** Scan patches are
drawn at the same 0.7 positive ratio as E0's synthetic bank, so E1 and E2 differ only
in the masking policy - which is what makes them an ablation. But the policies do not
cost the same: under ignore-masking roughly 30% of scan patches are negatives drawn at
a random page location, and those are almost entirely `IGNORE`, so they contribute to
no loss term. E2's equivalent patches carry background supervision. E1 therefore
trains on meaningfully fewer *useful* scan patches than E2 from the same draws.

That is not a bug in the comparison, it is the ignore policy's actual cost - the design
above says negatives should come from the fully-labelled synthetic replay instead, and
this is what that means in patches. It does mean a raw E2 > E1 result must not be read
as "background-outside labelling is better" without checking whether E1 simply saw less
signal; raising E1's positive ratio to 1.0 is the follow-up that separates the two, and
it is deliberately not folded into this first comparison.

**A second data-loader bottleneck, found by running E0 (2026-08-25).** E0's first
launch produced its "20,656 patches from 2,582 images" banner and then nothing at
all for nine minutes: GPU at 0%, all eight workers pegged in `R` state at ~3.5
minutes of CPU each, `%wa` at 0.0, and the main process at 21 seconds - blocked
waiting for a first batch that never arrived. Load was 34 on 128 cores, so this was
not contention with the OCR shards.

The cause is a near-miss of the `ImageBlockSampler` fix already recorded above.
That fix made a page's patches arrive consecutively so the one-slot cache would turn
eight decodes into one, and it does. But only the *decode* was behind the cache.
`box_centres_by_class` sat after it and ran unconditionally on every draw - and it is
much more expensive than the `imread` it followed: seven full-page `mask == index`
comparisons plus a `connectedComponentsWithStats` for every class actually present,
on a full-resolution page. So the sampler removed 7/8 of the cheap work and none of
the expensive work, which is why the symptom survived a fix that looked like it
addressed exactly this.

It is a pure function of the mask, so it is now cached alongside the image and
computed once per image. Two tests pin it: one counts calls across eight consecutive
draws and asserts exactly one, the other asserts a different image does not inherit
the previous page's centres (which would draw patches at coordinates belonging to
another page). 30 tests in `test_detector_patches.py` pass.

The general lesson, and the reason this is written down rather than just fixed: the
first fix was verified by reasoning about the access pattern it changed, not by
measuring the epoch it was supposed to speed up. A cache in front of the second-most
expensive operation in a loop is indistinguishable from a working fix until someone
times it.

**Measured, rather than argued about again (2026-08-25).** After the centres fix E0
was still doing roughly two to four hours per epoch, so the loader was profiled
directly instead of reasoned about a third time:

| | |
|---|---|
| page size | 4138 x 2928 (12.1 MP) |
| `cv2.imread` | 6.2 s |
| `box_centres_by_class` | 3.8 s |
| `DetectorPatches.__getitem__`, mean over 8 consecutive draws | 1.2 s |

The last row is the fix working exactly as intended - about ten seconds of per-image
work spread over eight patches rather than paid eight times. It is also the point:
even perfectly amortised, a patch costs more than a second, every epoch re-pays it,
and the experiment matrix is four runs of ten epochs, so the corpus would be decoded
forty times over to draw patches that never change.

**`extract_patch_bank.py` (new, 13 tests).** Draws every patch once to disk. The draws
come from `DetectorPatches` itself rather than a reimplementation, so the positive
ratio, per-class centre choice and jitter stay identical to the live dataset by
construction. `PreExtractedPatches` reads the bank; `train_detector.py --pre-extracted`
selects it. Two consequences beyond speed, both of which matter more:

1. **Batches become properly shuffled.** `ImageBlockSampler` exists only to make the
   decode cache hit, and its cost is that a batch is drawn from one or two pages -
   not close to i.i.d. Patch files are tiny, so a plain `shuffle=True` is affordable
   and every batch mixes pages.
2. **E0-E3 become comparable by construction.** All four read the same bank, so a
   difference between them cannot come from having drawn different patches. Per-image
   seeds are hashed from `(seed, image_index)` rather than taken from one advancing
   RNG, so the bank does not depend on how the pool scheduled the work.

### The box has a pid limit, and onnxruntime does not respect thread hints

Sharding the OCR extraction 8 ways took the whole machine down: every `fork` on the
box began failing with `Resource temporarily unavailable`, including `pgrep` and
`tmux`, so the machine looked dead rather than misconfigured. Recovery needed a loop
over `/proc/[0-9]*` using only bash builtins, since nothing that forks could run.

The cause, measured rather than guessed:

- The container's `pids.max` is **3840**.
- One `RapidOCR()` opens **572 threads** by default, because
  `EngineConfig.onnxruntime.intra_op_num_threads` defaults to `-1`, meaning one thread
  per core - and this is a 128-core box, across three sessions (detect, classify,
  recognise).
- 8 shards x 572 = 4576, past the limit before any other work on the box.

`OMP_NUM_THREADS` does not bound this - already recorded above for the earlier
`pthread_create` failure, and it is the same root cause reappearing in a new place.
Only onnxruntime's own option does: passing `intra_op_num_threads: 4` takes a shard
from 572 threads to **200**, verified by counting `/proc/self/task` both ways. 8 shards
now sit at 1,333 of 3,840 pids with the whole machine responsive.

`--ocr-threads` (default 4) now exposes this on
`ocr_first_text_ground_truth.py`, and `extract_patch_bank.py` calls
`cv2.setNumThreads(1)` in each pool worker for the same reason - OpenCV also sizes its
pool from the core count inside every pool member at once.

The general point worth keeping: on a many-core box, any library that defaults its
thread pool to "one per core" is a pid-limit hazard the moment it is run more than
once concurrently, and the symptom (the machine stops forking) points nowhere near the
cause.

Fixing it also turned out to be a throughput fix, not only a stability one. With 8
shards each spawning 572 threads the box was thrashing on context switches: load sat
around 51 and OCR completed roughly 0.3 scores a minute. With threads bounded, 8
shards run at load ~11 and around 7 scores a minute - the 355-score pass went from an
estimated 19 hours to well under one. Oversubscription was costing more than an order
of magnitude, and it read as "OCR is just slow" for as long as nobody counted threads.

**What the bank actually bought (measured, same corpus, same 1,291-batch epoch):**

| | before | after |
|---|---|---|
| batches in the first 60 s | ~5 | **450** |
| GPU utilisation | 0% | 42% |
| time per epoch | ~2-4 h | **~3 min** |
| 10-epoch run | ~30 h | **~30 min** |

The bank is 20,656 train and 2,456 validation patches (2,582 and 307 pages x 8), built
in about 12 minutes once. That one-time cost replaces the forty corpus decodes the
four-experiment matrix would otherwise have done, and it is what makes running E1-E3
as real comparisons affordable rather than aspirational.

### E0 baseline — complete (2026-08-25)

10 epochs over the patch bank, synthetic data only (`detector_masks_v4`, score-disjoint
2,582 train / 307 valid pages), all classes in the loss, ~35 minutes wall.
Weights `/workspace/b0/detector_e0_baseline.pth`, history
`/workspace/b0/detector_e0_history.json`. Training loss 0.2358 (epoch 1) -> 0.0425.

**Validation IoU, epoch 10 — the numbers E1-E3 are to be compared against:**

| class | epoch 1 | epoch 10 |
|---|---|---|
| background | 0.990 | 0.995 |
| MeasureNumber | 0.983 | 0.992 |
| Dynamic | 0.971 | 0.983 |
| Tempo | 0.846 | 0.966 |
| Lyrics | 0.899 | 0.955 |
| StaffText | 0.858 | 0.915 |
| Expression | 0.611 | 0.906 |
| Fingering | 0.875 | 0.875 |

Two things to carry into reading E1-E3 rather than discover afterwards:

1. **This is patch IoU, not the page-level precision §1 was about.** Every class scores
   0.87+ here while §1 recorded whole-page precision collapsing for five of seven
   classes. Both can be true: a patch is drawn near a box 70% of the time, so this
   measures "given text is roughly here, is it segmented correctly", not "how often
   does the detector fire somewhere on a whole page with no text". Improvements on
   this metric do not automatically transfer, and reporting an E1 win here as if it
   answered §1 would be exactly the mistake §1 was closed over.
2. **Fingering is the only class that did not move** (0.875 at both epoch 1 and 10) and
   is the weakest at epoch 10 - consistent with §2's rarity finding surviving
   phase18's "more distinct scores" fix.

### E1-E3 results — real-scan data helps substantially, and E3 is the configuration to keep

**Two measurement bugs had to be fixed before any of this could be read**, and both
produced plausible-looking numbers rather than obvious breakage:

1. **`evaluate()` dropped `ignore_index`.** `per_class_iou` has always supported it, and
   its own docstring says it "must be passed through here as well as into the loss -
   scoring those pixels would count every ignored region as a prediction error". The
   training loop passed it; `evaluate` accepted the argument and ignored it. Since
   ~98% of an ignore-masked scan page is `IGNORE`, every prediction there scored as an
   error, so E1-E3's validation blocks were meaningless while their *training* IoU
   looked normal. Three tests now cover it, including one that pins what the bug did
   so the fix cannot be quietly reverted.
2. **E0 and E1-E3 were validated on different sets.** E0 used the synthetic bank,
   E1-E3 the mixed bank. Whichever way those numbers fell they could not answer
   whether scan data helped, because they were scores on tests of different difficulty.
   `eval_detector.py` (new, 5 tests) scores any checkpoint against any bank, so all
   four models are now measured on the same data - and on each bank *separately*,
   since "did scans cost anything on synthetic pages" and "did scans help on real
   ones" are two questions a mixed number averages into neither.

**Validation IoU on the synthetic bank (`patch_bank_v4/valid`):**

| class | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| background | **0.995** | 0.000 | 0.459 | 0.994 |
| MeasureNumber | **0.992** | 0.784 | 0.006 | **0.992** |
| Dynamic | 0.983 | 0.983 | 0.984 | **0.984** |
| Tempo | **0.966** | 0.045 | 0.847 | 0.822 |
| Lyrics | **0.955** | 0.963 | 0.955 | 0.941 |
| StaffText | **0.915** | 0.827 | 0.827 | 0.913 |
| Expression | **0.906** | 0.002 | 0.806 | 0.897 |
| Fingering | 0.875 | 0.875 | 0.875 | 0.875 |

**Validation IoU on the real-scan bank (`scan_bank_ig/valid`)** - only Lyrics and
Dynamic are labelled there, so only they can be scored:

| class | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| Lyrics | 0.581 | 0.963 | 0.822 | **0.966** |
| Dynamic | 0.696 | **0.957** | 0.902 | 0.954 |

**What this actually says:**

- **Real-scan data helps, and by a lot.** The synthetic-only baseline scores Lyrics
  **0.581** and Dynamic **0.696** on real scans; adding scan data takes them to
  **0.966** and **0.954**. This is the first direct evidence in this project that the
  synthetic/real gap is real, large, and closable with the OCR-first ground truth -
  and note that E0 is not bad at Lyrics in general (0.955 on synthetic), it is bad at
  Lyrics *on real scans specifically*. That is a domain gap, not a weak class.
- **Ignore-masking matters, and the ablation is decisive.** E2 differs from E1 only in
  labelling unmatched pixels background instead of ignore, and it loses 0.14 of Lyrics
  IoU on real scans (0.822 vs 0.963). The asymmetry flagged before the run cuts
  *against* this conclusion rather than explaining it: E2 sees strictly more supervised
  patches than E1 from identical draws, and still does worse. Teaching the model that
  OCR-missed lyrics are background is worse than teaching it nothing there.

  Since the asymmetry was predicted rather than measured, it has now been counted:
  sampling 600 patches from each scan bank, **77.8% of E1's carry a label** and the
  remaining ~22% are pure `IGNORE` and contribute to no loss term at all. So E1 learned
  from roughly 7,120 useful scan patches against E2's full 9,152 - a 22% data
  disadvantage - and still won by 0.14 IoU. The handicap is real and the result
  survives it.
- **Restricting the loss to two classes is badly harmful, and E3 answers its question
  cleanly.** E1 and E2 destroy every class the loss does not cover - E1's background
  falls to 0.000 and Expression to 0.002, E2's MeasureNumber to 0.006 - because those
  classes receive no gradient at all. E3, identical to E1 except that all classes stay
  in the loss, keeps essentially all of E0's synthetic performance (background 0.994 vs
  0.995, MeasureNumber 0.992 vs 0.992, StaffText 0.913 vs 0.915, Expression 0.897 vs
  0.906) *and* gets the full real-scan gain.
- **E3 is the best configuration *on this metric*** - scan data mixed in,
  ignore-masked, all classes in the loss - with only Tempo (0.966 -> 0.822) and a
  slight synthetic Lyrics dip (0.955 -> 0.941) against E0. **This conclusion does not
  survive the page-level measurement below. Read that section before acting on this
  one.**

**Two cautions on reading the table.**

*Fingering is 0.875 in all four runs and at both epoch 1 and epoch 10.* An identical
value across four independently trained models is not a measurement, it is an artifact
- 0.875 is exactly 7/8, consistent with the class appearing in so few validation
patches that its score is decided by a single batch. Fingering should be treated as
unmeasured here, not as "unchanged", and §2's rarity finding is the reason to expect
exactly this.

*This is patch IoU, not the page-level precision of §1.* The caution recorded under E0
applies to the whole table: patches are drawn near a box 70% of the time, so this
measures segmentation given that text is roughly present, not false-positive rate
across a whole page. The real-scan gain above is strong evidence for the data, and
still not a §1 answer.

### The page-level measurement contradicts the patch measurement — E3 halves precision

`detector_box_eval.py` runs the model over whole pages, recovers boxes the way
inference actually has to, and matches them against real box ground truth. This is
§1's metric. Run over the same 307-page score-disjoint synthetic validation set:

| | E0 precision | E3 precision | E0 F1 | E3 F1 | gt boxes |
|---|---|---|---|---|---|
| Lyrics | **75.9%** | 71.1% | **84.4%** | 80.6% | 3,555 |
| Dynamic | **77.5%** | 64.4% | **84.9%** | 76.4% | 429 |
| MeasureNumber | **91.6%** | 83.0% | **94.4%** | 90.7% | 78 |
| StaffText | 11.5% | 9.7% | 19.8% | 17.1% | 106 |
| Tempo | 12.9% | 0.5% | 21.8% | 1.0% | 53 |
| Expression | 3.4% | 5.1% | 6.5% | 9.6% | 44 |
| Fingering | 0.0% | 0.0% | 0.0% | 0.0% | 12 |
| **overall** | **56.8%** | **29.0%** | **70.7%** | **44.1%** | 4,277 |

**Adding the scan data halved page-level precision (56.8% -> 29.0%) and cut F1 from
70.7% to 44.1%.** Recall barely moved (93.7% -> 92.1%), so E3 is not missing text; it
is firing far more often where there is none. That is precisely §1's failure mode, made
worse by the data gathered to address it.

**...but E1 and E2 then showed that conclusion was too broad, and the real story is a
precision/recall trade the aggregate was hiding.** Box-evaluating all four models on
the two classes the scan data actually covers:

| Lyrics | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| precision | 75.9% | 77.3% | **83.5%** | 71.1% |
| recall | 95.1% | **96.8%** | 93.5% | 93.0% |
| F1 | 84.4% | 86.0% | **88.2%** | 80.6% |

| Dynamic | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| precision | 77.5% | **90.9%** | 89.3% | 64.4% |
| recall | **93.7%** | 92.8% | 93.7% | 93.7% |
| F1 | 84.9% | **91.8%** | 91.5% | 76.4% |

Three things fall out of this, and none of them are visible in the overall row:

1. **On the classes it covers, scan data helps at page level too.** E1 and E2 both beat
   the synthetic baseline on Lyrics and Dynamic F1, Dynamic by ~7 points. The earlier
   "scan data halves precision" reading was true of E3 and false as a general claim.
2. **E1 vs E2 is exactly the predicted precision/recall trade, and it confirms the
   negative-supervision mechanism.** E1 (no negatives outside boxes) has the higher
   *recall* - 96.8% vs 93.5% on Lyrics - and the lower precision. E2 (full negatives)
   has the higher precision - 83.5% vs 77.3% - and the lower recall. Ignore-masking
   buys recall by withholding "no text here"; background-outside buys precision by
   asserting it, at the cost of teaching that OCR-missed lyrics are background. Note
   this also resolves the apparent contradiction with the patch table, where E1 beat E2
   on real-scan Lyrics IoU 0.963 to 0.822: that metric rewards finding text, this one
   penalises firing where there is none, and the two models sit on opposite sides of
   the same trade.
3. **E1/E2's catastrophic overall row (1.2% and 1.0% precision) is entirely the five
   classes their loss excluded.** Those get no gradient, predict everywhere, and swamp
   the aggregate. It is a real defect - such a model is unusable as a general detector -
   but it says nothing about the scan data.

**Which leaves E3 as the genuinely bad result, and for an interesting reason.** E3 is
the only run that had both scan data and an all-class loss, and it is the *worst* model
on Lyrics and Dynamic - below the synthetic baseline it started from. So the failure is
not "scan data hurts" and not "all-class loss hurts"; it is the combination. Asking one
loss to serve fully-labelled synthetic pages and 2%-labelled scan pages at once degrades
both the classes the scan data covers and the classes it does not.

The mechanism is consistent with the design: an ignore-masked scan page supervises
**1.98% of its pixels** (measured when the masks were built). Every scan patch
therefore contributes almost no evidence for "there is no text here", while
contributing strong evidence for "text looks like this". Mixed into training, the scan
data dilutes negative supervision corpus-wide, and a detector with weaker negative
evidence fires more. E1's Tempo precision collapsing to 0.5% is the same effect at its
most extreme.

**The honest summary of the matrix**, then, is neither the patch table's "scan data is
a large win" nor the first box-eval reading's "scan data halves precision":

- The OCR-first scan data **improves the classes it covers**, on both metrics, when the
  loss is restricted to those classes (E1, E2).
- **Ignore-masking and background-outside sit on opposite sides of a precision/recall
  trade**, and neither dominates: +recall/-precision for ignore, the reverse for
  background.
- **Mixing sparse scan supervision into an all-class loss (E3) is actively harmful**,
  worse than either the baseline or either ablation.
- The 2-class runs are **not shippable as they stand**, because the excluded classes
  collapse completely.

The caution attached to the patch table is what kept this readable: a large win on a
metric nobody had tied to the decision was not evidence the decision should change, and
it took the page-level numbers - and then the *per-class* page-level numbers - to see
what was actually happening. The aggregate row alone would have produced the wrong
conclusion twice, in opposite directions.

### E4/E5 — the middle masking policy (running)

The trade in (2) above is not fundamental. Ignore-masking gives up negative supervision
because it cannot tell "no text here" from "OCR missed the text"; but for *blank paper*
that distinction is free. A pixel with no ink cannot be a missed lyric.

`scan_text_masks.py --background-blank` (new, 6 further tests, 18 in the file) labels
unmatched blank-paper pixels background and leaves unmatched *inked* pixels `IGNORE`,
since ink outside a box is genuinely ambiguous - notation, or the missed lyric the
scheme exists to protect. The prediction it makes is specific: E4 should keep E1's
recall while gaining E2's precision, because it supplies the negatives E1 lacks without
making E2's false claim.

Built over the real corpus it does what it was designed to, and the numbers are worth
recording because they show how much of the "unsupervisable" page was in fact free:

| masking policy | supervised pixels | Lyrics px | Dynamic px |
|---|---|---|---|
| ignore-outside (E1, E3) | 1.98% | 1.8312% | 0.1504% |
| **background-on-blank-paper (E4, E5)** | **86.24%** | 1.8312% | 0.1504% |
| background-outside (E2) | 100.00% | 1.8312% | 0.1504% |

The class fractions are identical across all three - same boxes, same pages; only the
treatment of everything else differs. So ignore-masking was discarding ~84 points of
genuinely certain negative supervision to protect the ~14% of the page where ink makes
the question real.

| id | data | loss | question |
|---|---|---|---|
| E4 | synthetic + scan, blank-background | `Lyrics,Dynamic`, ignore 255 | does the middle policy beat both ablations on its own classes? |
| E5 | synthetic + scan, blank-background | all classes, ignore 255 | does restoring negatives rescue the all-class loss that sank E3? |

E5 is the one that matters for shipping: it asks whether a single all-class detector can
absorb scan data at all, which E3 says it cannot under pure ignore-masking.

**Remaining after that:**

1. Raise E1's positive ratio to 1.0 (bank already built, `scan_bank_p10`). Now expected
   to *lower* page-level precision, since it removes the last scan negatives - it tests
   the mechanism as much as the ratio.
2. Give Fingering a validation set that can score it: 12 ground-truth boxes and 0.0%
   across every model is not a measurement.
3. If E5 fails too, the structural answer is separate treatment rather than one shared
   loss - the scan-supervised classes and the synthetic-only classes are being asked to
   share a loss that fits neither.

### The OSSQ scanned track is systematically mislabeled — root cause found (2026-08-25)

**56.7% of scanned staves are paired with the wrong music.** Not degraded - wrong.

Found by asking where `stage2_scans_best`'s remaining error lived. The per-branch
accuracies already in `stage2_train.log` put pitch at 0.857 against 0.914 for rhythm and
0.966 for articulation, and pitch is the branch a misaligned crop damages most while
leaving the others intact. `domain_gap.py` then scored each staff against its own
synthetic twin, over 900 staves sampled evenly across all 9 validation scores:

| | |
|---|---|
| mean accuracy | synthetic **96.1%** -> scanned **46.8%** |
| unchanged (<=10 points) | 362 (40.2%) |
| collapsed (>50 points) | **510 (56.7%)** |
| per-score collapse rate | 63%, 66%, 68%, 92%, **95%** - every score, none spared |

**The cause is one join in `convert_ossq.py`:**

```python
segments = sorted((work / "musicxml" / "unaligned").glob("*.musicxml"))
crops = work / "images" / track / "partwise"
score_id, page, system = segment_path.stem.split(":")
image = crops / CROP_NAME.format(score=score_id, page=int(page), system=int(system), ...)
```

Symbols come from `musicxml/unaligned` for **both** tracks and are joined to crops by
`(page, system)`. That directory is keyed to the *synthetic* pagination. The scanned
pages paginate differently - `sq8907120` renders to 24 synthetic pages and scans to 22 -
so the same `(page, system)` names different music in each track. The two directories
hold the *same number of segments* (109 each), so the join finds a crop, the part count
matches, every guard passes, and the pairing is silently wrong.

**The dataset already ships the right source.** `musicxml/scanned/systemwise` holds 109
segments with max page 0022 - matching the scanned crops exactly, against
`unaligned`'s 0024.

Proof, on the collapsed staff `sq8907120_0011_0003_3` (100% synthetic -> 3% scanned):

```
scanned/systemwise :  D4 F3 F3 F3 F3 F3 F3 G3 G3 G3 G3 G3 G3 C4 ...
model read of crop :  _  .  D4 F3 F3 F3 F3 F3 F3 G3 G3 G3 G3 G3 ...   <- matches
training reference :  _  .  _  A4 C5 ...                              <- unaligned
```

**The model was reading the scanned crop correctly and being scored against the wrong
answer.** The 3% was never a failure to read; it was a mislabeled target.

**Consequences.**

- `train_scans.py`'s mixture was 32,982 OSSQ scanned of 42,735 total, so **~77% of last
  night's fine-tune was trained on data that is more than half mislabeled**.
  `stage2_scans_best` (0.93004) must therefore be treated as suspect, not as the best
  base to build on - an earlier note in this file called it exactly that, and that claim
  does not survive this finding.
- The synthetic track is unaffected: `unaligned` *is* its correct pagination, which is
  why synthetic mean accuracy is 96.1% and why the failure stayed invisible for so long.
- This is the same class of bug as the LayoutBreak off-by-one (§7) and §28.1's stripped
  directions: a positional join between two sequences that are the same length and
  different things. Bar-count-style guards cannot see it, because nothing about the
  count is wrong.

**Verified by eye before acting**: the two crops for that staff show plainly different
music - synthetic is alto clef, two flats, sparse notes with rests and `pp`; the scan is
dense sixteenth-note runs with `sf`, `ff` and `decresc.`
`ossq_pair_review_server.py` (new, 12 tests) serves the paired crops worst-first so this
is checkable without reading any code.

### Beams reach the output, and repeats are recovered — two gaps closed from one review session

Both of these came from human review of the pair page rather than from any metric, and
neither would have surfaced from accuracy numbers.

**1. The generator could not write beams.** The structured heads predict the whole
six-level beam vector at 0.9508 exact-match, and `music_xml_generator.py` contained
**zero** beam references - so every prediction was discarded on the way out, and every
render showed MuseScore's own automatic beaming instead (§27.6: a round trip rewrites
1.7% of notes, its largest pattern turning backward hooks into full beams).

`build_beams` now emits `<beam number="N">` per level. Verified end to end on a real
staff: 20 elements, begin 5 / continue 10 / end 5, all level 1 - five coherent groups.
Four behaviours are tested because each fails silently: hooks use MusicXML's spelling
(`forward hook`, not our `forward_hook`, which MuseScore ignores); `flag` and
`not_applicable` emit nothing, since absence *is* how MusicXML says "not beamed";
inapplicable levels do not renumber later ones, or a 16th beam is written as an 8th; and
notes without structured labels emit exactly what they did before, so checkpoints trained
without the heads are unaffected.

*The unit tests passed while the real path emitted zero beams* - the end-to-end check
caught that my probe called `attach_sidecar(symbols, path)` with the arguments reversed.
Nine green unit tests were not evidence the feature worked.

**2. Repeats and barlines are recoverable after all.** Review kept reporting "correct but
missing final repeat", "missing a double repeat between measures 3 and 4". Measured: the
whole scores carry **8,617 `<barline>`, 3,740 of them repeats**; the segments carry
**zero** - the same total loss as §28.1's directions and §27.20's slur placement, from
the same round trip.

`barline_placement.py` (new, 12 tests) recovers them the way those two already do, with
one difference that needed checking rather than assuming: barlines belong to *measures*,
not notes, so the join needs measure-level alignment as well as note-level. Verified over
the parts whose note signatures already match: **64 of 64 also have identical measure
counts, zero disagreements.** The note alignment remains the gate - a part whose notes do
not concatenate is skipped entirely rather than joined on measure counts alone, because
**a wrong repeat is worse than a missing one**: it changes how the music is played, not
merely how it looks.

Wired into `convert_ossq.py` beside the slur and dynamics indexes, and the corpus
rebuilt. The check that matters is token counts, not element counts - the index producing
`<barline>` elements and the tokeniser emitting `repeatEnd` are different pipelines:

| corpus | staves | repeat / special-barline tokens |
|---|---|---|
| `phase7clef` (before) | 3,912 | **0** |
| `phase7bar` (after) | 3,912 | **588** |

repeatEnd 176, doublebarline 152, repeatStart 124, repeatEndStart 68,
bolddoublebarline 68. Same staff count, so nothing was dropped to get them. This was
never a vocabulary limitation - every one of those tokens already existed.

### Structured heads on the final base — beaming recovered at 95%, dynamics did not train

Trained over the clef-corrected core with everything else frozen (326 core parameters
loaded, 22 head tensors trainable), 3 epochs on `phase7clef`, then evaluated on its
held-out split - 3,904 sequences.

| head | macro F1 | support |
|---|---|---|
| **exact beam vector** | **0.9508** | 56,741 |
| beam level 1 | 0.9562 | 56,741 |
| slur spans | 0.9290 | 12,998 |
| slur sides | 0.9094 | 6,081 |
| beam level 3 | 0.8480 | 2,107 |
| hooks | 0.8162 | 1,285 |
| beam level 2 | 0.8130 | 14,372 |
| ties | 0.8032 | 5,361 |
| stems | 0.7189 (micro 0.9483) | - |
| beam level 4 | 0.5972 | **8** |
| **dynamics** | **0.1030** | 4,517 |

**Beaming is recovered.** `exact_beam_vector` 0.9508 means the *entire* six-level beam
state is correct for 95% of notes, not merely one level - which is the number that
matters, since a note's beaming is the whole vector. This is the capability the review
page exposed as missing: renders showed MuseScore's auto-beaming because
`music_xml_generator.py` emits no `<beam>` and the tokens never carried one. The labels
were always in the sidecars; now a model predicts them.

**Two heads must not be read as working, and the loss curve alone would have said they
were.** Both were flagged from the training log before this evaluation ran, which is the
point of running it:

- **`dynamics` is at F1 0.103** on 4,517 supported positions, with its training loss
  moving only 2.6% (0.2540 -> 0.2474) despite the *largest* support of any head at
  913,207 positions.

  **Diagnosed, and my first explanation for it was wrong.** I assumed
  `dynamics_placement.py`'s positional join was failing to deliver marks into the
  segments - the same total-loss pattern as slurs and barlines. Measured instead, over
  800 sidecars: the marks **are** arriving. The distribution is

  | none | sf | p | f | pp | ff | fp | mf | sfp | other | fz |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 20,316 | 396 | 371 | 267 | 76 | 49 | 12 | 7 | 5 | 5 | 4 |

  94.5% of notes carry no dynamic, which is musically correct, and the labelled 5.5%
  spread across eleven classes of which five have **fewer than thirteen examples**.
  Macro-F1 averages classes equally, so a head that learns `p`, `f` and `sf` and cannot
  possibly learn `fz` (4 examples) scores near 0.10 regardless of how well it does the
  learnable part. This is §28.1's phase16/17 finding - "mf/mp/ppp stuck at F1=0.000" -
  reproduced exactly, and it is a *rarity* problem, not a delivery problem.

  The correction matters for what to do next: recovering more dynamics from the whole
  scores would not help, because they are already all here. What would help is either
  reporting this head on the classes that have support (as `detector_box_eval`'s
  `priority` row does for the detector), or merging the rare marks into coarser classes.
  Both are reporting/labelling changes rather than data work.
- **`beam level 4` has 8 supported positions** in the whole validation split. Its 0.5972
  is not a measurement. Level 4 beams are 64th-note subdivisions and are simply rare;
  the honest reading is "unmeasured", the same trap as Fingering's 0.875 (= 7/8) earlier
  in this file.

**`stems` shows the macro/micro split worth naming**: macro 0.7189 against micro 0.9483.
The micro figure says almost every stem is right; the macro says one of its classes is
not, and averaging classes equally exposes what averaging notes conceals.

### Adding the OSSQ synthetic track costs scan accuracy — measured, not assumed

The synthetic track (42,088 staves, clef-corrected) had never been in any mixture. Added
at full weight to the clef-corrected run, it takes the real-scan share of the mixture
from ~92% to **43.7%** - scans become the minority.

Validation was deliberately kept **scans only**. This is the whole reason the result is
readable: had synthetic staves been added to the held-out set alongside the training
set, a model drifting toward synthetic would have shown a *rising* aggregate while scan
accuracy fell, and the number would have concealed the regression it exists to detect.

Six epochs, against the 0.96906 / 0.9391 baseline it warm-started from:

| epoch | overall | pitch |
|---|---|---|
| baseline | **0.96906** | **0.9391** |
| 1 | 0.96863 | 0.93870 |
| 2 | 0.96784 | 0.93736 |
| 3 | 0.96772 | 0.93650 |
| 4 | 0.96794 | 0.93716 |
| 5 | 0.96771 | 0.93700 |
| 6 | 0.96780 | 0.93772 |

**It dips over three epochs and then holds flat, about 0.0013 below baseline on overall
and 0.0018 on pitch.** Not a transient adaptation dip - three further epochs produced no
recovery. This is §23's warning behaving exactly as written: adapting on data from a
domain we are not trying to fit specialises the model away from the one we are, and 15%
pdmx replay does not prevent it. It is also the second measurement of the same effect
today, the first being that adding OSSQ to a Lieder-heavy mixture cost 1-2 points of
Lieder accuracy per branch.

The cost is small in absolute terms. What makes it worth acting on is that it is
**paid for nothing**: the synthetic track is 42,088 staves of the easier domain, and
the metric that matters got worse rather than better.

**Follow-up, pre-registered before the numbers arrived**: retry at
`--synthetic-weight 0.4` (16,835 synthetic staves, real-scan share back to **61.7%**),
on the theory that a smaller amount might act as regularisation where a swamping amount
specialises.

**Run, and it is also below baseline.** Four epochs, oscillating 0.9674-0.9685 and never
reaching the baseline it warm-started from:

| | baseline | synthetic w=1.0 (best) | synthetic w=0.4 (best) |
|---|---|---|---|
| overall | **0.96906** | 0.96863 | 0.96847 |
| pitch | **0.9391** | 0.93870 | 0.93838 |

Lowering the weight does reduce the damage - w=0.4 tracks slightly above w=1.0 at
matched epochs - but it does not turn the sign around. **The conclusion is that the
OSSQ synthetic track has nothing to add to scan accuracy at any weight tried**, and
`scans_clef_best.pth` (0.96906) stands as the best base. Both experiment models are kept
(`scans_synthetic_w1.pth`, `scans_synthetic_w04.pth`) so the comparison is reproducible.

This is a genuinely useful negative: the largest untapped data source in the project
turns out not to be a lever, and knowing that stops it being tried again. It also
sharpens what the remaining headroom is - the base is not data-limited on staves of this
kind, so improving it further means better data of a *different* kind (harder scans,
more Lieder, or the excluded OSSQ segments) rather than more of the same.

**An operational note that cost two false starts**: `train.py` refuses to overwrite an
existing model destination, and the run id derives from git state - so every run in a
session with an unchanged working tree collides on the same filename, and the second one
exits immediately with "Model already exists" having trained nothing. Killing a tmux
session *does* trigger `train.py`'s save path, so an interrupted run leaves exactly the
file that blocks its successor. Renaming each finished model to what it actually is
(rather than leaving the generic `pytorch_model_<runid>.pth`) both fixes the collision
and makes the model inventory readable.

### Shipping decision: two detectors, E2 for vocal and the instrumental model for the rest

**Decided 2026-08-25 (user).** The application offers a "with/without lyrics" toggle, and
each side gets its own detector:

| toggle | weights | loss | trained on |
|---|---|---|---|
| **with lyrics** (vocal) | `detector_e2.pth` | Dice over `Lyrics,Dynamic` | 29,808 patches, Lieder scans + synthetic, background-outside masks |
| **without lyrics** (instrumental) | `detector_instr_bg.pth` | Dice over `Dynamic,Tempo,StaffText,Expression` | 18,608 patches, OSSQ scans, background-outside masks |

E2 is the best model measured on the priority classes - **88.5% F1 on Lyrics+Dynamic
pooled**, against 86.6% for E1, 85.8% for E4 and 84.5% for the synthetic-only baseline.

**What this decision costs, stated plainly so it is not discovered later.** E2's loss
covers two classes, so the other five receive no gradient and predict everywhere: at page
level its overall precision is 1.0%, with `Expression`, `Fingering`, `MeasureNumber`,
`StaffText` and `Tempo` all at 0.0% F1. Four of those were scoped out as unwanted
(Fingering) or derivable (MeasureNumber) or hand-fixable (Tempo). The two that were
called "nice to have" - StaffText and Expression - are not degraded by this choice, they
are **dead**. That is a stronger statement than the earlier framing of the trade and is
the actual consequence of picking E2.

The instrumental model covers Tempo, StaffText and Expression on instrumental pages
(0.853, 0.783, 0.694 IoU), so the classes E2 abandons are not lost from the *product* -
they are lost from vocal scores specifically. Whether that matters is a question about
Lieder scores with staff text on them, which nothing here has measured.

**What still has to be true before this ships:**

1. ~~**The instrumental model has never been evaluated at page level.**~~ **Resolved
   2026-08-25, and it changed the answer.** The page-level evaluation was run, and the
   model named in the table above is *not* the one that was originally trained.

   The first instrumental model was trained with **blank-paper** masking - after this
   document had already established, from the vocal E-series, that background-outside
   wins. That was my error, not a considered choice. At page level it is unusable:

   | class | blank-paper P / R / F1 | background-outside P / R / F1 | Δ F1 |
   |---|---|---|---|
   | Dynamic | 0.12% / 32.48% / 0.0025 | **43.11% / 75.72% / 0.5494** | +0.5469 |
   | Tempo | 0.15% / 52.65% / 0.0030 | **13.15% / 68.94% / 0.2209** | +0.2179 |
   | StaffText | 0.15% / 41.37% / 0.0030 | **22.20% / 58.63% / 0.3220** | +0.3190 |
   | Expression | 0.14% / 42.47% / 0.0028 | **9.71% / 26.25% / 0.1418** | +0.1390 |

   Blank-paper predicted 911,859 Dynamic boxes against 3,501 real ones - 4,378 boxes per
   page. It is not a low-precision model, it is a model that says "text" everywhere, and
   its ~40% recall is an artefact of covering the page rather than of finding anything.

   Two things are worth keeping from this. First, the retrain **dominates on both axes** -
   higher precision *and* higher recall on every class - so unlike the E-series result
   (§ignore-masking vs background-outside), this is not a precision/recall trade and needs
   no judgement call. Second, patch IoU again failed to predict it: the blank-paper model
   scored a respectable 0.545-0.853 valid IoU on exactly the classes where it turns out to
   emit 4,378 boxes per page. That is now **twice** in this project that the patch table
   ranked a model the page-level metric then reversed. Patch IoU should be treated as a
   training-progress signal only, never as a selection criterion.

2. **The instrumental model is still not shippable on its rare classes.** Even retrained,
   Expression (9.7%) and Tempo (13.2%) precision mean most predicted boxes are wrong.
   Dynamic (43.1% / 75.7%) is the only class carrying its weight. `MeasureNumber` and
   `Lyrics` receive no gradient here and predict spuriously (30,316 and 36,294 boxes
   against zero ground truth) - harmless only because the toggle means this model never
   sees a vocal page, which makes the toggle a correctness component exactly as noted
   below.
3. **Nothing measures how the toggle is chosen.** A vocal score run through the
   instrumental model loses its lyrics entirely; an instrumental score run through E2
   gets lyric false positives on every page. The selection mechanism is now a correctness
   component, not a UI convenience.

### Head-to-head against upstream homr — the number the whole effort is judged on

Everything above measures our own runs against each other. This measures the work
against the thing it is meant to improve: the **pinned upstream checkpoint**
(`pytorch_model_426-...`), scored through the identical loader and metric on identical
held-out staves.

| branch | Lieder (n=362) | | | OSSQ scanned (n=600) | | |
|---|---|---|---|---|---|---|
| | homr | ours | Δ | homr | ours | Δ |
| pitch | 0.5316 | **0.5666** | +3.50 | 0.9382 | **0.9799** | +4.17 |
| rhythm | 0.5764 | **0.6052** | +2.88 | 0.9069 | **0.9679** | +6.10 |
| lift | 0.6210 | **0.6440** | +2.30 | 0.9221 | **0.9848** | +6.27 |
| articulation | 0.6493 | **0.6726** | +2.33 | 0.9613 | **0.9838** | +2.25 |
| slur | 0.6312 | **0.6523** | +2.11 | 0.9519 | **0.9803** | +2.84 |
| position | 0.6251 | **0.6533** | +2.82 | 0.9720 | **0.9902** | +1.82 |

**Every branch improves on both domains.** Largest instrumental gains are rhythm and
lift (+6.10, +6.27); the largest vocal gain is pitch (+3.50), which is the branch the
pagination and clef corrections were aimed at.

Three things must be said with these numbers or they will be over-read:

1. **Absolute levels are not comparable across domains.** OSSQ staves are single quartet
   lines; Lieder staves are dense piano-and-voice writing. Compare within a domain only.
2. **This metric is stricter than `eval_accuracy`** - free-running decode with both sides
   padded to the longer length by a non-matching sentinel, so a length disagreement
   counts against the model rather than being scored on the overlap. Both checkpoints are
   scored identically, so the deltas hold; the levels do not transfer to any
   `eval_accuracy` figure elsewhere in this file.
3. **It is a combined delta.** Corrected pagination, carried clefs, restored barlines and
   scan-domain adaptation all landed in one checkpoint. This establishes that the
   programme worked; it attributes nothing to any single change, and the ablation that
   would has not been run.

### The instrumental detector — the "without lyrics" half of the two-model split

The E0-E5 matrix established that one detector with one loss cannot serve both
supervision regimes, and that no masking policy fixes it. The structural answer is two
models, which is also the product shape the application wants ("with/without lyrics").
This is the instrumental half, and the OSSQ text extraction is what made it possible.

Corpus: 2,625 OSSQ pages, 93 scores, split 81/12 score-disjoint. **0.0000% lyrics
pixels** by construction, against Lieder's 1.83% - a genuinely instrumental corpus rather
than a vocal one with the lyrics ignored. 18,608 training patches, 2,392 validation.

Loss covers `Dynamic,Tempo,StaffText,Expression`. **Lyrics is deliberately absent from
the loss and from the data**, which is the point of the split: a detector that cannot
predict lyrics cannot emit a false lyric on a quartet page, and on instrumental scores
every lyric prediction the general detector makes is a false positive by definition.

First epoch, validation IoU:

| class | IoU |
|---|---|
| Tempo | 0.809 |
| Expression | 0.700 |
| StaffText | 0.645 |
| Dynamic | 0.601 |

**Tempo, StaffText and Expression are the three classes that had no real-scan
supervision of any kind before this**, and whose page-level F1 in the synthetic baseline
was 21.8%, 19.8% and 6.5% respectively. These IoU figures are not comparable to those F1
figures - different metric, different corpus - and the comparison that will matter is
`detector_box_eval` on instrumental pages, which needs instrumental box ground truth that
does not exist yet. What can be said now is narrower and still worth saying: the classes
the general detector could never learn are learnable when the data actually contains
them.

**A split trap worth recording**, since it would have passed silently: every OSSQ page
lives in a directory named `original`, so `detector_split`'s folder rule reports one
score for the entire corpus and puts all of it on one side of the split while still
printing a clean score-disjoint summary. `--score-from mask` reads the score from the
mask filename instead, and works for all three corpora.

**Complete, 10 epochs** (`detector_instrumental.pth`, loss 0.4814 -> 0.1965):

| class | epoch 1 valid | epoch 10 valid | epoch 10 train |
|---|---|---|---|
| Tempo | 0.809 | **0.853** | 0.889 |
| StaffText | 0.645 | **0.783** | 0.891 |
| Expression | 0.700 | 0.694 | 0.892 |
| Dynamic | 0.601 | **0.545** | 0.611 |

Two things in that table are more informative than the headline:

- **Dynamic regressed** (0.601 -> 0.545) and is the weakest of the four, which is
  awkward because it is one of the two priority classes. A plausible reason is visible
  in the corpus itself: OSSQ dynamics are small italic glyphs, and they are the same
  marks the OCR extraction had the most trouble reading - the class is genuinely hard
  *on these pages*, as distinct from hard in general. Lieder dynamics, which are larger
  and better separated, reached 0.983 patch IoU in E2. That points at combining the
  corpora for Dynamic rather than treating this number as the ceiling.
- **Expression and StaffText show a large train/valid gap** (0.892 vs 0.694, 0.891 vs
  0.783) on 12 validation scores. That is overfitting on the two rarest classes in the
  smallest corpus, and it means these validation numbers should be treated as provisional
  until there are more instrumental scores.

`Fingering`, `MeasureNumber` and `Lyrics` report "no data this epoch" throughout, which
is the corpus behaving as designed rather than a fault: there are no lyrics on a string
quartet page, and that absence is the reason this model exists.

### The clef-corrected continuation — plateaued, and the sequence of runs so far

Continued from the phase7fix run's epoch-7 checkpoint (0.96798) on `phase7clef`, warm
started rather than restarted, since the two corpora differ only in the 837 staves that
gained a clef.

| branch | phase7fix (epoch 7) | clef-corrected (best) |
|---|---|---|
| pitch | 0.9351 | **0.9391** |
| rhythm | 0.9592 | **0.9629** |
| lift | 0.9675 | **0.9700** |
| position | 0.9838 | **0.9855** |
| articulations | 0.9806 | **0.9817** |
| slurs | 0.9734 | **0.9762** |
| overall | 0.96798 | **0.96906** |

Plateaued clearly: epochs 3-7 span 0.9685-0.9691, a range of 0.0006. Best checkpoint
`checkpoint-8262`, saved as `/workspace/b0/scans_clef_best.pth`.

**The gain is small and that is the expected size.** Correcting 2.4% of staves moved the
aggregate by about 0.001. It would have been a warning sign if it had moved much more -
a clef token is one symbol at the head of a sequence, and the pitches around it were
already right. The reason to fix it was never the aggregate; it was that a consumer of
these tokens reconstructs the wrong staff, which is how it was found.

**Where the runs stand:**

| run | corpus | validation | best |
|---|---|---|---|
| contaminated | phase7 (56.7% mislabeled) | Lieder only | 0.9294 |
| Lieder-only | Lieder + replay | Lieder only | 0.9268 |
| pagination-fixed | phase7fix | mixed scans | 0.96798 |
| clef-corrected | phase7clef | mixed scans | **0.96906** |

The first two are not comparable to the last two - different validation sets - which is
itself the point recorded above: a Lieder-only held-out set never measured the OSSQ half
at all.

### A second, independent data bug: 2.4% of staves had no clef — found by eye, invisible to every metric

Found during review: a bass-clef scan whose rendered label came out in treble, notes
stacked on ledger lines far below the staff. The label's first token was
`keySignature_-3`; a correct staff starts `clef_F4`.

**Why no measurement would ever have caught it.** Pitches in this format are absolute
(`B3`), so a missing clef costs essentially nothing in pitch accuracy and moves no
number in any training log. It only becomes visible when something *reconstructs* the
staff - which is what rendering the label for human review does, and why this surfaced
the day that page was built rather than in any of the accuracy work before it.

Scale, before the fix - and note it is **not** the pagination bug, since it hits both
tracks identically:

| track | staves | no leading clef |
|---|---|---|
| scanned train | 34,511 | 837 (2.43%) |
| scanned valid | 3,912 | 117 (2.99%) |
| synthetic train | 42,090 | 1,012 (2.40%) |
| synthetic valid | 4,943 | 140 (2.83%) |

**Cause**: MusicXML states a clef in `<attributes>` where it is established or changes,
so a systemwise segment cut from the middle of a part can legitimately begin without
one. Engraved music restates the clef on every system, so the crop shows one and the
label does not, and neither file alone reveals the disagreement. Confirmed directly: the
segment for the reported staff has no `<clef>` for *any* of its four parts.

**Fix**: `ensure_clef` carries the clef forward from the previous segment of the same
part. The clef in effect is not ambiguous - it is whatever that part last established -
and this is the same recovery `slur_placement.py` and `dynamics_placement.py` already do
for their own upstream losses. Three properties are tested because each would fail
silently: an existing clef is never overwritten (a genuine mid-piece clef change must
survive), the carry resets per work (a clef must never cross scores), and the carried
element is deep-copied on insert (otherwise one segment's tree mutates another's).

**Verified**: 837 -> **7** on train, 117 -> **1** on valid, with the staff count
unchanged, so nothing was dropped to achieve it. The residual 8 are staves where no
earlier segment of that part ever stated a clef, so there is nothing to carry and they
are correctly left alone rather than guessed at.

**Human review, at 52 verdicts: `label-ok` on every one, zero mismatched pairs** - on
worst-scoring-first ordering, i.e. the staves most likely to be wrong. That is
independent visual confirmation of the same conclusion the 56.7% -> 7.9% collapse
measurement reached from the accuracy side. The review's remaining complaints are all
about elements that were never in the segments to begin with (repeats, slurs), which is
§28.1's loss pattern rather than a pairing fault.

### Training on the corrected corpus — first numbers, and what they do not yet prove

Five epochs in, on 43,931 training staves (corrected OSSQ + Lieder + pdmx replay):

| branch | contaminated run | Lieder-only run | **corrected run** |
|---|---|---|---|
| pitch | 0.8569 | 0.8513 | **0.9351** |
| rhythm | 0.9140 | 0.9122 | **0.9592** |
| lift | 0.9304 | 0.9284 | 0.9675 |
| position | 0.9634 | 0.9616 | 0.9838 |
| articulations | 0.9660 | 0.9630 | 0.9806 |
| slurs | 0.9485 | 0.9453 | 0.9734 |
| overall | 0.9294 | 0.9268 | **0.9666** |

Pitch - the branch identified as the bottleneck, and the one a misaligned crop damages
most - gains **7.8 points**.

**These columns are not strictly comparable, and the gap is not yet evidence of what it
looks like.** The first two runs validated on the 362 Lieder staves; this one validates
on the mixed set (362 Lieder + 792 OSSQ) introduced with the fix. Some of the
improvement is therefore arithmetic rather than learning: the OSSQ portion of the
validation set previously *could not* be scored well by anything, because its labels
disagreed with its images, and now it can.

Separating the two needs the corrected model scored on the Lieder-only 362 - if Lieder
accuracy also rose, the corrected OSSQ data is teaching something transferable; if it
held flat, the gain is confined to OSSQ no longer being mislabeled.

**Run, and the answer is the second one - with a caveat that matters more than the
result.** Both models scored on the same 362 Lieder staves:

| branch | Lieder-only model | scan run (epoch 7) |
|---|---|---|
| pitch | **0.5526** | 0.5347 |
| rhythm | **0.5858** | 0.5731 |
| lift | **0.6266** | 0.6160 |
| articulation | **0.6531** | 0.6434 |
| slur | **0.6331** | 0.6258 |
| position | **0.6335** | 0.6218 |

The Lieder-only model is better on Lieder in **every branch**, by 1-2 points. So the
0.9294 -> 0.9666 headline is substantially "OSSQ stopped being mislabeled" rather than
transferable learning, exactly as the caveat suspected.

**The caveat: this is not a clean test of the data, it is a test of the mixture.** Lieder
is 85% of the Lieder-only run's training set and 7.6% of the scan run's. A model trained
mostly on quartets being slightly worse at Lieder is §23's specialisation warning
behaving exactly as documented, not evidence that the corrected OSSQ data is unhelpful.
Distinguishing those would need the scan run's mixture re-weighted toward Lieder, which
has not been done.

What it does establish, and what matters for the next decision: **adding a large
non-Lieder corpus costs 1-2 points of Lieder accuracy**, and pdmx replay at 15% did not
prevent it. That is the price of the current mixture, now measured rather than assumed -
and it is the reason the synthetic track, which would push the mixture further still from
both scan domains, needs its scan accuracy watched separately rather than folded into an
aggregate.

*(Absolute numbers here are far below the trainer's 0.93+ because this metric is
free-running generation with a length-mismatch penalty, not teacher-forced token
accuracy. Both models are scored identically, so the comparison holds; the levels are not
comparable to `eval_accuracy` anywhere else in this file.)*

**A measurement bug found and fixed on the way.** The first attempt returned
byte-identical numbers for both models across all six branches - which reads as "these
models are equivalent" and is in fact "neither checkpoint was loaded". Two classes are
named `Staff2Score`: `homr.transformer.staff2score` is the ONNX inference path and never
reads `config.filepaths.checkpoint`, while
`training.architecture.transformer.staff2score` does. `base_predictions.py` imported the
first. It now imports the second, fails loudly on a missing checkpoint, and prints the
weights it is using; three tests pin the asymmetry, including one asserting the ONNX
class does *not* read the checkpoint so that a future change there surfaces as a test
failure rather than as silently identical results.

This does **not** affect the `domain_gap` results above (46.8% -> 89.9%, 56.7% -> 7.9%):
those compared two *label sets* through one constant model, so the model's identity was
irrelevant to the comparison - what differed was `phase7` against `phase7fix`. Checked
rather than assumed.

### The OSSQ fix, and its verification (2026-08-25)

`convert_ossq.py` now resolves the segment directory per track via `segments_dir`:
`musicxml/unaligned` for synthetic, `musicxml/scanned/systemwise` for scanned. A missing
directory **raises** rather than falling back, because falling back is precisely the
original failure and would look like a successful conversion. Six tests, including one
asserting the two tracks never resolve to the same directory.

**Single-case proof**, on the staff whose crops were checked by eye:

```
model read of scanned crop : D4 F3 F3 F3 F3 F3 F3 G3 G3 G3 G3 G3 G3 C4
OLD label (phase7)         : A4 C5 E4 C4 A3 E4 D4 B3 F3 F3 F3 F4 F4 F4   ->  0/14
NEW label (phase7fix)      : D4 F3 F3 F3 F3 F3 F3 G3 G3 G3 G3 G3 G3 C4   -> 14/14
```

**Corpus-wide verification** - same 900 staves, same checkpoint, only the labels changed,
so this isolates the fix:

| | before | after |
|---|---|---|
| mean accuracy, scanned | 46.8% | **89.9%** |
| median drop per staff | 76.7% | **0.0%** |
| quartiles | 0.0% / 88.2% | **0.0% / 0.0%** |
| unchanged (<=10 points) | 40.2% | **86.0%** |
| collapsed (>50 points) | **56.7%** | **7.9%** |

The typical staff now reads identically under both renderings. The rebuild also yields
*more* data than before - 34,510 train and 3,911 valid against 32,982 and 3,571 -
because correct pagination lets more systems pair one-for-one.

**The residual 7.9% is expected and is the real domain gap**: §27.14's staff-count
miscounts (five to nine staves detected in a four-part system) plus genuinely hard
scans. That is the population the "spread vs concentrated" question was originally
about, now visible for the first time without 56.7% of mislabeled pairs on top of it.

**This project had already met this hazard once and guarded it elsewhere.** The commit
the training box is checked out at is `d0bcb56`, *"Refuse OSSQ scores whose render
disagrees with the reference pagination"*, whose message reads: "The page indices in the
reference come from the MusicXML layout; the images come from rendering the same score.
Those are separate omr-data-preprocessor steps, and they only agree if the same
MuseScore produced both." That guard was added to the page-level benchmark. The
identical hazard - two independently produced paginations joined by index - sat
unguarded in `convert_ossq.py`, where it was worth 56.7% of the scanned corpus. Worth
noting as a pattern: a hazard understood well enough to guard in one place is not
thereby handled in the others, and this class of bug is invisible precisely where nobody
went looking.

**Two follow-on changes to `train_scans.py`**, so the fix cannot be undone by re-running
the old configuration:

- `OSSQ_SCANNED_INDEX` now names `phase7fix`, with a comment saying why `phase7` is not
  referenced any more - a path constant is exactly where a corrected corpus quietly
  reverts.
- **Validation now spans both domains** (`mixed_valid_index.txt`: 362 Lieder + 792 OSSQ,
  sampled evenly across all 9 held-out scores). The previous run validated on the 362
  Lieder staves alone, so *nothing in its reported accuracy ever measured the OSSQ half*
  - which is a large part of why 56.7% of that half being mislabeled never showed up as
  a falling number. Verified score-disjoint against both training indexes before use:
  zero overlapping scores on either corpus.

**A metric artifact worth not misreading**: the "share of all lost notes falling in the
worst 10% of staves" line reports 116.0% after the fix. A share above 100% is not a
finding - with total lost notes now small, staves where the *scan* scores higher than
the synthetic contribute negative loss and the ratio stops being a share. The line is
meaningful when the gap is large and meaningless once it closes.

### E4/E5: the middle masking policy did not work, and the prediction it was built on failed

E4 was built on a specific, falsifiable prediction recorded before the run: the
blank-paper policy "should keep E1's recall while gaining E2's precision, because it
supplies the negatives E1 lacks without making E2's false claim". Measured on the
priority classes (Lyrics + Dynamic pooled, box-weighted):

| run | policy | precision | recall | F1 |
|---|---|---|---|---|
| E2 | background-outside | **84.1%** | 93.5% | **88.5%** |
| E1 | ignore-outside | 78.5% | **96.4%** | 86.6% |
| **E4** | **blank-paper background** | **78.8%** | 94.3% | 85.8% |
| E0 | synthetic only | 76.1% | 95.0% | 84.5% |

**The prediction failed.** E4 landed on E1's precision (78.8% against 78.5%), not E2's.
Taking supervised pixels from 1.98% to 86.24% - 84 points of negative evidence that is
*certainly* correct - bought no precision at all.

That is informative rather than merely disappointing. It means E2's precision advantage
does **not** come from having more negative supervision, which was the whole premise.
What E2 does that E4 refuses to do is label the *ambiguous inked* regions - notation,
and any text the OCR missed - as background. So the advantage comes from suppressing
firing on inked non-text, and the blank paper the middle policy recovers was never where
the false positives were. In hindsight that is obvious: a detector does not fire on blank
paper.

**E2 remains the best model on the priority classes**, and the honest summary of the
whole masking question is that the naive policy won, for a reason none of the design
reasoning anticipated.

**E5 settles the shipping question, negatively.** E5 was the run that mattered: does
restoring negative supervision rescue the all-class loss that sank E3? It does not.

| run | scan masking | loss | priority F1 |
|---|---|---|---|
| E2 | background-outside | Lyrics,Dynamic | **88.5%** |
| E1 | ignore-outside | Lyrics,Dynamic | 86.6% |
| E4 | blank-paper | Lyrics,Dynamic | 85.8% |
| E0 | *(no scan data)* | all classes | 84.5% |
| E3 | ignore-outside | all classes | 80.1% |
| **E5** | **blank-paper** | **all classes** | **77.7%** |

The pattern is now unambiguous across six runs: **every** 2-class run beats the
synthetic-only baseline on the priority classes, and **every** all-class run with scan
data falls below it - including the one with 86% of its pixels supervised. Adding
sparse, two-class real-scan supervision to a loss that also has to serve six
fully-labelled synthetic classes damages the two classes the new data is *about*.

E5 does keep the other classes alive where E1/E2 destroy them (overall precision 53.9%
against their ~1%), but that is the consolation prize: it is still below E0's 56.8%
overall, so it is worse than the baseline at everything.

**So the structural answer recorded as the fallback is now the finding.** One model with
one loss cannot serve both supervision regimes, and no masking policy fixes that -
the three policies tried span 1.98%, 86.24% and 100% supervised pixels and all three
land in the same place. The scan-supervised classes (Lyrics, Dynamic) and the
synthetic-only classes (Tempo, StaffText, Expression, MeasureNumber, Fingering) need
separate treatment.

That is independent support for the two-model toggle proposed for the application
("with/without lyrics"): the split is not only a product convenience, it is what the
measurements say the data requires.

### OSSQ instrumental text extraction — complete (2026-08-25)

All 8 shards exit 0. **93 scores, 2,625 pages carrying text, 36,897 confirmed boxes** -
1.8x the entire Lieder corpus, and the first real-scan supervision Tempo, StaffText and
Expression have ever had:

| kind | matches |
|---|---|
| dynamic | 29,796 |
| stafftext | 3,561 |
| tempo | 2,161 |
| expression | 1,379 |

**A silent-failure bug found on the way, worth recording for the class of bug it is.**
The first run produced 40 matches over 167 pages, nearly all of them from title pages.
Three hypotheses were tested and all three were wrong: detection resolution (0 lines at
`limit_side_len` 2000, 4000 and 6000), the colon in OSSQ's filenames (0 either way), and
detection thresholds (0 at every `box_thresh`/`thresh` setting).

What settled it was rendering a page and looking at it: an obvious header, measure
numbers, and clearly legible `f`, `p`, `sf`. Zero OCR lines on *that* is not a
sensitivity story. The cause is that **OSSQ pages are palette-mode PNGs (`mode=P`) and
RapidOCR's own file loader returns zero boxes on them, with no error**; every Lieder
page is `mode=RGB`, so it failed in exactly one corpus. The same page decoded by
`cv2.imread` and passed as an array yields 39 boxes. `ocr_page` now decodes the image
itself, which also means one loader for both corpora rather than two that can disagree.

Left unfixed, this would have written 2,876 all-`IGNORE` masks and trained on nothing
while reporting success at every stage - the mask builder's own summary line reads
"N page masks written, 0 skipped" whether or not any mask contains a label. The general
shape: an input-loading failure that returns *empty* rather than raising looks exactly
like "the data has none of what you're looking for", and that is a conclusion one can
act on for a long time.

### Building the best Stage 2 model: what the artifacts actually are, and the order to do it in

**`phase20`-`phase22` are not Stage 2 models.** This is worth stating plainly because
the naming invites the opposite reading. The pieces are separate and stack:

| artifact | what it is | current best |
|---|---|---|
| `/workspace/b0/stage2_scans_best/` | the Stage 2 **base transformer**, real-scan fine-tuned | `eval_accuracy` **0.93004** |
| `phase20/profile_context_weights.pth` | §7.2/§7.3 score-profile conditioning | phase20 (phase21 unfroze and erased the gain, §3) |
| `phaseN/heads.pth` | structured beam / stem / slur heads | phase13 latest; phase11 holds the focal/weights variants |

Every one of our additions is an off-by-default, zero-gated extension of the *same* base
- `enable_structured_heads`, `enable_profile_context`, `duration_adherence_weight=0.0`,
`cross_staff_coherence_weight=0.0`, all under the standing rule in `configs.py` that "a
checkpoint trained without them must keep loading and behaving exactly as before". So
there is no separate "our model, with heads" base to fine-tune. There is one base, plus
attachments to it.

Which means last night's fine-tune is not only an upstream-shaped PR candidate: it is
also *our* best base, because the attachments bolt onto exactly that checkpoint.

**The ordering, and why it is not arbitrary.** `train_structured_heads.py` trains the
heads **over a frozen core**, deliberately - its own docstring: "Only the new
projections move. If anything else did, a gain could be the heads or the core drifting
to suit them, and the answer would not be an answer." Heads therefore learn to read one
specific core's decoder features. Moving the core afterwards invalidates them.

So the base must be finished *before* the heads are trained, or the heads get trained
twice and the first run is discarded. The same applies to phase20's profile-context gate,
which is also fitted against a specific core. Sequence:

1. **Finish the base.** Currently `stage2_scans_best`, 0.93004.
2. **Train the structured heads over that frozen core.** This is the run that produces
   beaming, and it is cheap - only the projections move.
3. **Re-fit profile context** on the final base, using phase20's recipe (the one that
   worked), not phase21's.

`phase13/heads.pth` and `phase20/profile_context_weights.pth` are both currently paired
with the *pre-fine-tune* core, so neither is valid against `stage2_scans_best` as things
stand. That is not a defect - it is the expected consequence of step 1 having happened.

**What is actually available to push the base with (measured, 2026-08-25):**

| source | used by the run | on disk |
|---|---|---|
| IMSLP/Lieder pairs | 3,353 train + 362 valid = **all 3,715** | 3,715 |
| OSSQ scanned partwise crops | 32,982 train + 3,571 valid = 36,553 | 44,682 |

There is **no unused Lieder data** - the recovery work already fed all of it in. On the
OSSQ side ~8,100 crops (18%) are outside the phase7 indexes, of which 1,816 are on the
explicit `excluded_segments_scanned_abc_partwise.txt` exclusion list.

**So "push the base further" has no free data lever**, and the run plateaued by epoch 5
(epochs 5-13 within ±0.0006). Adding ~19% more crops to a run that converged early is
not obviously worth a training cycle, and the honest next step is to find out *where*
0.93004 is losing accuracy before buying data aimed at nothing in particular - the same
discipline §14.7 applied to GPU capacity. Two candidate levers, in the order the
evidence supports:

1. **Error analysis of `stage2_scans_best` on its held-out set**, broken down by symbol
   class and by corpus (Lieder scans vs OSSQ scans). Cheap, and it decides whether the
   remaining 7% is rhythm, pitch, or something structural.
2. **Content-fingerprint recovery of the excluded OSSQ segments**, which is the proven
   technique from the Lieder side (+954 systems, +35%) applied to the 1,816 exclusions -
   but only if (1) shows the base is data-limited rather than saturated.

**(1) is already answered, and it cost nothing to answer.** `HomrTrainer` tracks
per-branch token accuracy alongside the headline number, so the breakdown was already in
`stage2_train.log` and needed no GPU and no new tooling:

| branch | epoch 1 | best |
|---|---|---|
| articulations | 0.9561 | 0.9660 |
| position | 0.9501 | 0.9634 |
| slurs | 0.9383 | 0.9485 |
| lift | 0.9182 | 0.9304 |
| rhythm | 0.8975 | 0.9140 |
| **pitch** | 0.8367 | **0.8569** |
| overall | 0.9161 | 0.9294 |

**Pitch is the bottleneck, by a wide margin** - 0.857 against 0.914 for the next worst,
and ~11 points below the four heads that are effectively done. The headline 0.93 is an
average over branches that are not remotely equal, and it hides the fact that roughly
half the remaining error mass sits in one branch.

Two consequences for step 1:

- **A general "more crops" run is the wrong shape.** It spends a training cycle spread
  evenly over six branches, four of which are at 0.95+ and cannot contribute much. What
  the base needs is whatever fixes *pitch* specifically.
- **Pitch is also the branch most sensitive to crop alignment**, which is what makes
  §27.14's scanned-staff miscount (five to nine staves detected in a four-part system)
  the first suspect rather than image difficulty: a shifted crop-to-part pairing moves
  every pitch on the staff while leaving rhythm, articulation and position largely
  intact - exactly the asymmetry in the table. `domain_gap.py` exists to tell these
  apart ("spread" = the domain is hard, "concentrated" = the crops are wrong) by scoring
  each staff against its own synthetic twin, and it should be run on this checkpoint
  before any decision to add data.

That last point is the same trap this session hit twice elsewhere: a plausible cause
(scans are harder) that would send the work at augmentation and more data, sitting next
to a mechanical cause (the crops are misaligned) that more data cannot touch. The
measurement distinguishing them already exists.

### Stage 2 renders and the review site (2026-08-25)

3,715 pairs (2,761 extracted + 954 recovered) rendered for review; the combined
manifest matches the pairs on disk exactly. The first pass rendered 2,200 and failed
1,515, in whole batches of 200 at a time with empty stderr.

The batching was the cause, not the files: `render_batch` runs `xvfb-run -a mscore -j`
over a whole batch under `check=True`, so a single failure inside the batch fails all
200 and reports nothing useful. Re-running with `--batch-size 20` recovers them at
about 78% per batch. A file that "failed" in a batch of 200 renders perfectly well on
its own - verified directly.

Worth remembering as a pattern rather than a one-off: batching for throughput and
`check=True` for safety combine into "one bad item silently discards 199 good ones",
and the log line it produces (`batch at 400 FAILED:` with empty stderr) actively points
away from that being what happened.

Review server is back up on port 8792 over the combined manifest: 3,715 pairs across
161 scores, plus the 20,785 Stage 3 text matches mounted at `/text`.
pages, and the two are laid out differently on disk: synthetic images sit in a
`{score}_p{n}-s{n}` folder, real-scan pages in a plain `{score}` folder. `score_of`
derives the score from the parent folder name, so both resolve correctly - but nothing
tested it, and if it had failed on the scan layout the "score-disjoint" guarantee
(§13.5) would have silently applied to only half the corpus while still reporting a
clean split. `tests/test_detector_split.py` (new, 7 tests) pins both layouts and the
mixed case.

### Packaging the corpora and models for distribution

Release mechanics live in **`DATASET_DISTRIBUTION.md`** - what ships, how it is laid out,
and the commands that reproduce each headline number. The packager is
`training/omr_datasets/package_dataset.py` (23 tests).

Three defects stood between "the corpora exist" and "someone else can use them", and all
three pass a naive file count, which is the only reason they are worth a section:

1. **The staff crops are symlinks.** `convert_ossq.py` calls `link_image`, so a corpus
   directory holds links into the build machine's scratch space rather than pixels.
   `rsync -a` faithfully copies the links: 38,421 files arrive, the count is exactly
   right, and every one is dangling. `rsync -aL` is required, and `-a` alone fails
   silently. This is the same class of error as the pagination bug - equal counts on both
   sides, nothing raising, everything wrong.
2. **Index files carry `/workspace/b0/...` absolute paths.**
3. **Box ground-truth JSON references pages by absolute path.**

`package_dataset.py` materialises the links, rewrites both path classes, writes a
manifest, and then re-reads from disk to verify - the failure mode being guarded against
is a packaging step that reports success over a broken dataset.

Two decisions inside it are worth recording because the obvious alternative is wrong:

- **Paths are rewritten relative to the index that names them, not to the dataset root.**
  The indexes describe the build machine's layout while the files ship laid out beside
  their index, so there is no shared prefix and a plain `relative_to()` silently finds
  nothing to do, leaving every row absolute. Caught only by running the packager against
  the real corpus rather than the fixtures - the unit tests had been written against the
  layout I assumed.
- **An unresolvable path is left absolute rather than guessed**, so `verify()` reports it.
  A plausible path resolving to the *wrong* file is precisely the silent failure this
  module exists to prevent.

The manifest also excludes itself from its own digest: it is computed over a tree without
a manifest and checked against a tree with one, so a self-inclusive digest could never be
reproduced by the person checking it.

## Structured heads in production, and the refinement UI

**Decided 2026-08-25 (user): heads on in production, and their probabilities exposed as
multiple-choice refinements.** This section records what that requires and what was
deliberately scoped out, because the gap between "the heads work" and "the heads do
anything for a user" is larger than the evaluation numbers suggest.

### The chain is built at both ends and missing in the middle

| stage | state |
|---|---|
| heads train, produce per-head logits | works - `exact_beam_vector` 0.9508 |
| `music_xml_generator.build_beams` emits `<beam>` | works |
| `generate()` runs the heads and returns them | **missing** |
| something populates `Note.notation` at inference | **missing** |

`structured_logits` is produced only by the *training* forward pass and consumed only by
training and evaluation scripts. `generate()` - the autoregressive path inference actually
uses - returns `symbols` and nothing else, and `Note.notation` defaults to `None`. So
`build_beams` is currently a no-op in production: it returns early on every note.

This is worth stating plainly because every measured number in this document is real and
none of it reaches a user. The work is plumbing, not modelling.

### What ships as a choice, and what only ships

These are different questions, and conflating them would either hide good output or ask
users to arbitrate noise.

| head | macro F1 | written to MusicXML | offered as a choice |
|---|---|---|---|
| beam level 1 | 0.9562 | yes | **yes** |
| slur spans | 0.9290 | yes | **yes** |
| slur sides | 0.9094 | yes | **yes** |
| beam level 3 | 0.8480 | yes | **yes** |
| hooks | 0.8162 | yes | **yes** |
| beam level 2 | 0.8130 | yes | **yes** |
| ties | 0.8032 | yes | no |
| stems | 0.7189 (micro 0.9483) | yes | no |
| beam level 4 | 0.5972 (n=8) | yes, when applicable | no |
| dynamics | 0.1030 | **no** | no |

Beams and slurs are offered. Stems and ties are good enough to improve the output but not
to ask a user to arbitrate - a refinement UI implies the model has an opinion worth
choosing between, and at 0.72/0.80 that claim is weaker than the interface would suggest.
Beam level 4 has support 8; its F1 is noise and must never be presented as a distribution.
Dynamics at 0.1030 did not train and is not emitted at all.

### Surfacing: threshold-gated, not always-on

Alternatives appear only where the model is genuinely uncertain. At 0.9508 exact-beam
accuracy, offering a choice on every note would bury the ~5% worth reviewing under 95%
noise. The threshold is a tunable, and choosing it needs the confidence distribution on
real pages - not yet measured, and it should not be guessed.

The known cost: a confident-but-wrong prediction is never surfaced. That is the accepted
trade, and it is the reason the threshold should be set from a measured distribution
rather than picked.

### Why `/v1/regenerate` is the right seam

OTS already has `/v1/regenerate`: rebuild MusicXML from an edited token sequence, no
image, no GPU. A refinement is exactly that shape - the user picks an alternative, the
score is rebuilt, no re-recognition. The contract needs extending, since regenerate
currently validates six-field string symbols and structured notation is not among those
six fields.

### ~~Still unproven~~ Resolved 2026-08-25: the heads export cleanly

They do survive export, and more easily than expected, because two things turned out to
be true that were not obvious until checked:

1. The heads are a **non-autoregressive** projection of the hidden state
   (`structuredHeadsAutoregressive: false` in the capability manifest).
2. The decoder graph **already emits `hidden`** as an output.

So the heads export as their own small graph (132 KB) consuming `hidden`, and the decoder
export is untouched. A deployment missing that file behaves exactly as it did before -
the same rule `configs.py` applies to enabling the heads at all.

Verified with the real trained weights, not a fresh module: `heads_clef.pth` loads its 22
tensors with nothing missing or unexpected, and against 50 random hidden states the
exported graph gives a **max absolute logit delta of 5.7e-06** and **zero disagreements
in decoded notation**. The second number is the one that matters - logit drift at float32
rounding is harmless, a different decoded beam state is a different score.

`convert_structured_heads()` in `training/onnx/convert.py`, 6 tests.

One incidental fix: `convert.py` imported the segnet model and the encoder at module
scope, pulling in `pytorch_lightning` and `timm`. Neither is needed to export the decoder
or the heads, and at module scope they made this work untestable without the whole
segmentation stack. Made local to the functions that use them, which also cleared one
pre-existing test failure and three collection errors.

**Follow-up attempt at a full-model export, and a mistake worth recording.** With the real
checkpoint now local, I tried exporting the whole pipeline - not just the heads - by
splitting `scans_clef_best.pth` into the two files `convert_encoder`/`convert_decoder`
already expect (`training/onnx/split_pinned_checkpoint.py`, 7 tests; both halves load
`strict=True` into fresh modules, so the split itself is exactly right: 184 encoder + 141
decoder tensors from 326, the one dropped key being `decoder.note_mask`, a buffer
`ScoreDecoder` derives from config rather than a learned tensor).

The exports it produced are not trustworthy. Encoder: 65 KB, against a real encoder of
52.8 MB - roughly 800x too small to hold a ConvNeXt's weights, so something in the export
path silently failed to embed them. Decoder: 188 MB against a real 47.3 MB - 4x too large,
which is its own, different anomaly. Neither number was sanity-checked before I ran the
export, which is the same mistake this document has flagged before: a step that reports
success is not the same as a step that worked, and I did not verify decoded output against
the torch model the way the heads export was verified (max logit delta, zero decoded-
notation disagreements). That verification is real work still to do, not a formality.

**A real mistake in the process, not just a modelling gap.** `convert_encoder`/
`convert_decoder` write to `config.filepaths.encoder_path`/`decoder_path`, which are fixed
to `homr/transformer/{encoder,decoder}_pytorch_model_426-....onnx` regardless of whose
weights were loaded into them - the filename is keyed to the *architecture*, not the
checkpoint. That path doubles as the live cache `download_weights` populates, and both
functions do guard it (`if exists and not overwrite: warn and refuse`) - the mistake was
mine, passing `overwrite=True` without registering that the file I was about to replace
was the real upstream cache and not a scratch copy of my own. The guard did exactly what
it was built to do; I overrode it without checking what it was protecting.

Caught by comparing file sizes against timestamps immediately after, not by the guard,
since I had disabled it. Recovered by fetching the two zips directly from `liebharc/homr`'s
release and restoring the original files; nothing tracked in git was affected, since
`*.onnx` is gitignored.

**Fixed 2026-08-26.** All three `convert_*` functions now take an optional `out_dir`;
omitted, behaviour is exactly what it was, so the pinned-checkpoint export flow this was
inherited from is untouched. Verified two ways: unit tests against a mocked `Config`
confirm the redirected path is actually used (3 tests), and I re-ran the exact export that
caused the incident with `out_dir` pointed at scratch - the live cache's file listing and
timestamps are byte-identical before and after.

**Both anomalies turned out to be real, and both are now resolved or understood - neither
was a correctness bug in the weights.** Debugged directly on the GPU instance, against
the real checkpoint and a torch/CUDA build matched to the one that trains it, rather than
in the local CPU venv - the instance already had every dependency installed and needed no
resolving.

- **The 65 KB encoder was never broken.** It was correct weights (the graph's
  initializers hold exactly the expected ~51 MB, confirmed by summing `raw_data` across
  every tensor) split into a small graph file plus a sibling `.onnx.data` file. A newer
  torch silently upgrades the requested opset (17 -> 18) and, once upgraded, routes
  through the torch.export/onnxscript exporter instead of the legacy TorchScript one -
  and that exporter defaults to externalizing large initializers. `convert_decoder`
  already pinned `dynamo=False` to avoid exactly this; `convert_encoder` did not. Fixed by
  adding the same flag. Reproduced the anomaly byte-for-byte on the instance first (to
  rule out an environment difference from the local venv), then re-exported with the fix
  and got a single 50.3 MB file. Verified against torch on a real image tensor: max
  absolute delta **8.1e-06**.

- **The 188 MB decoder is not a bug - the real release is quantized, and this project has
  never had tooling for that step.** Comparing initializers by shape: the same
  `(512, 4096)` weight is 8,388,608 bytes in my fp32 export and exactly 2,097,152 bytes
  (4x smaller, 1 byte/element) in the real release. The real graph's node types confirm
  it: `DynamicQuantizeLinear` (55x), `MatMulInteger` (70x), `DequantizeLinear`, plus
  `SkipLayerNormalization` and `MultiHeadAttention` - fused kernels from ONNX Runtime's
  transformer optimizer. The pinned decoder went through **ORT's transformer-fusion pass
  and dynamic int8 quantization** after export, a pipeline stage `training/onnx/convert.py`
  has never encoded anywhere. This isn't a `dynamo=` flag away; it needs
  `onnxruntime.transformers.optimizer` and `quantize_dynamic` run against our export,
  matching whatever recipe produced the pinned release - a real, separate piece of work,
  not a one-line fix.

  Verified the fp32 export itself is correct on its own terms, since "correct but
  unquantized" and "wrong" are different findings and only one of them is true here:
  decoder ONNX vs torch on a real generation step (context, KV cache, and the
  `staff_context_emb` input all included), max absolute delta **2.4e-06** on the rhythm
  logits, and identical argmax.

  **The quantization step itself is now built** - `quantize_decoder()`, wrapping
  `onnxruntime.quantization.quantize_dynamic`. Run against the real export it reproduces
  the footprint almost exactly: **47.6 MB against a real 47.3 MB**. It does *not*
  reproduce the real graph's `SkipLayerNormalization`/`MultiHeadAttention` operator
  fusion - ONNX Runtime's transformer optimizer, a separate step aimed at inference speed
  rather than size, not attempted.

  **Accuracy is only spot-checked on random inputs, not yet on real staff data, and that
  distinction matters here more than usual.** 20 trials of random tokens/context against
  the matching fp32 torch model: **1/40 argmax mismatches**, max logit delta **0.178** - a
  real change from quantization noise, not rounding. Random inputs are close to
  worst-case for this check: with no real signal separating classes, logits sit close
  together, and small quantization noise flips an argmax more easily than it would
  against a trained model's confident, well-separated real output. Whether decoding
  decisions hold up on real pages is not yet known.

  **Resolved 2026-08-26 with the real-data check.** Ran the actual production
  `Staff2Score` class (`homr/transformer/staff2score.py` - not a stand-in) over 15 real
  held-out validation crops, three ways: the torch reference, the fp32 ONNX pipeline, and
  the quantized-decoder ONNX pipeline, all through the exact same encoder and heads.

  | comparison | token-sequence | beam-level (571 notes) |
  |---|---|---|
  | torch vs fp32-onnx | 15/15 exact match | 569/571 agree (99.6%) |
  | torch vs quantized-onnx | 0/15 mismatches | 555/571 agree (97.2%) |

  This is a materially better result than the random-input spot-check suggested (1/40
  argmax mismatches there), for the reason predicted: real staff images produce
  confident, well-separated logits that quantization noise rarely flips, unlike random
  noise inputs where classes sit close together. Every one of the 15 crops produced the
  *same overall token sequence* under quantization; the 16 beam-level disagreements are
  isolated per-note calls, not sequence-altering. Even torch-vs-fp32 (numerically
  identical to 1e-6) shows 2 disagreements from the same cause - a near-tied logit can
  flip on rounding-level noise regardless of quantization.

  **Broadened 2026-08-26, same day: the first pass covered one score by accident.**
  `sorted(glob(...))[:15]` put 15 consecutive crops from a single piece (`sq10307350`) in
  front, not a spread across the 9 distinct scores in this validation split - caught only
  when asked directly whether the crops had actually been tested, not by anything in the
  script itself. Re-ran with 3 crops sampled from each of the 9 scores (27 total):

  | comparison | token-sequence | beam-level (855 notes) |
  |---|---|---|
  | torch vs fp32-onnx | 27/27 exact match | 848/855 agree (99.2%) |
  | torch vs quantized-onnx | 0/27 mismatches | 833/855 agree (97.4%) |

  Consistent with the narrower first pass, and the broader sample adds the number that
  actually settles the shipping question: of the 22 beam-level disagreements between
  quantized-onnx and the torch reference, **22/22 (100%) fell on a note the quantized
  model had already flagged as uncertain** (below the confidence threshold, `structured_
  decode.py`'s own gate). Zero fell on a note it was confident about. Quantization never
  silently changed a decision the model was sure of in this sample - every place it
  landed differently from fp32 was already going to reach the user as a reviewable
  alternative regardless of which side it fell on, because that is precisely what "below
  threshold" means.

  **Net position: the quantized decoder is validated across every score in this held-out
  split and ships as the default.** It matches production's own real-world precedent
  (upstream's release is quantized, not fp32), it is 4x smaller, and nothing in either
  test round argues for preferring the fp32 decoder - it remains available as a fallback,
  not because the data asks for one.

### What is now the real blocker

Not the model. `base_url` in `homr/main.py:370` is a **hardcoded local variable** pointing
at `liebharc/homr`'s release tag, with no environment override and no config seam. Serving
our own weights means changing that line - so the redirect has to live in the commit OTS
pins, which is why the code pin and the weights pin are less independent than they look.

Also worth knowing before any model swap: `download_weights` decides freshness purely by
`os.path.exists(model)` - no version, no checksum. The Dockerfile bakes weights into the
image so a rebuild is clean, but any deployment with a persisted cache will silently keep
serving the old model.
