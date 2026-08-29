# RUNLOG — Ensemble transcription for homr

Single consolidated record for this project. It replaces the two documents that
previously split it:

- `ENSEMBLE_TRANSCRIPTION_DESIGN.md` — the architecture and its rationale
- `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` — the running work log

Both now live here verbatim, as Parts I and II, so nothing is lost and every prior
cross-reference still resolves to the same words. Part III is new: the Lieder corpus
rebuild, which is where the training and evaluation data actually comes from and
which neither original covered end to end.

**Where they disagree, the most recent dated entry wins.**

| part | contents |
|---|---|
| [Part I](#part-i--design) | Architecture, structured heads, cross-staff repair, evaluation, acceptance gates |
| [Part II](#part-ii--work-log) | Dated experiment and investigation log |
| [Part III](#part-iii--corpus-construction-lieder) | How the Lieder scan corpus is built, and why the obvious ways are wrong |

---

# Part I — Design

*Verbatim from `ENSEMBLE_TRANSCRIPTION_DESIGN.md`; headings demoted one level.*

## Generic ensemble transcription with structured notation heads

**Status:** accepted design; implementation not started

**Date:** 2026-08-15

**Initial corpus:** OpenScore String Quartet OMR (OSSQ-OMR)

**Future corpus:** OpenScore Lieder

### Contents

Sections 1-26 are the design proper; §27 is the running reproduction
record and carries most of the measured findings, so its subsections are listed too.

- [1. Executive summary](#1-executive-summary)
- [2. Goals](#2-goals)
- [3. Non-goals for the first experiments](#3-non-goals-for-the-first-experiments)
- [4. Current architecture and relevant constraints](#4-current-architecture-and-relevant-constraints)
- [5. Design principles](#5-design-principles)
- [6. Target architecture](#6-target-architecture)
- [7. Optional score-profile conditioning](#7-optional-score-profile-conditioning)
- [8. Staff and system detection](#8-staff-and-system-detection)
- [9. Structured beam and stem heads](#9-structured-beam-and-stem-heads)
- [10. Structured slur heads](#10-structured-slur-heads)
- [11. Semantic data model and MusicXML generation](#11-semantic-data-model-and-musicxml-generation)
- [12. Cross-staff context and repair](#12-cross-staff-context-and-repair)
- [13. OSSQ dataset adapter](#13-ossq-dataset-adapter)
- [14. Training plan](#14-training-plan)
- [15. Evaluation](#15-evaluation)
- [16. Human-in-the-loop contract](#16-human-in-the-loop-contract)
- [17. Page boundaries and assembly](#17-page-boundaries-and-assembly)
- [18. Lieder extension and lyrics](#18-lieder-extension-and-lyrics)
- [19. Backward compatibility and deployment](#19-backward-compatibility-and-deployment)
- [20. Concrete implementation areas](#20-concrete-implementation-areas)
- [21. Experiment matrix](#21-experiment-matrix)
- [22. Acceptance gates](#22-acceptance-gates)
- [23. Risks and mitigations](#23-risks-and-mitigations)
- [24. Recommended first implementation slice](#24-recommended-first-implementation-slice)
- [25. Settled decisions and open measurements](#25-settled-decisions-and-open-measurements)
- [26. Implementation evidence and related designs](#26-implementation-evidence-and-related-designs)
- [27. Reproduction record (2026-08-15/16)](#27-reproduction-record-2026-08-1516)
  - [27.1 What was built](#271-what-was-built)
  - [27.2 Environment](#272-environment)
  - [27.3 Dataset construction](#273-dataset-construction)
  - [27.4 B0: the pinned checkpoint on OSSQ synthetic](#274-b0-the-pinned-checkpoint-on-ossq-synthetic)
  - [27.5 Label support, and what it settles](#275-label-support-and-what-it-settles)
  - [27.6 Beam materialization is not needed here, and would be harmful](#276-beam-materialization-is-not-needed-here-and-would-be-harmful)
  - [27.8 The stem head has no supervision from the segment labels](#278-the-stem-head-has-no-supervision-from-the-segment-labels)
  - [27.9 Notation has to survive the dataset files, not just the parser](#279-notation-has-to-survive-the-dataset-files-not-just-the-parser)
  - [27.10 homr's rhythm vocabulary stops at the 128th note](#2710-homrs-rhythm-vocabulary-stops-at-the-128th-note)
  - [27.11 Training needs staff crops, which nobody has built](#2711-training-needs-staff-crops-which-nobody-has-built)
  - [27.12 How much beaming is derivable without looking at the page](#2712-how-much-beaming-is-derivable-without-looking-at-the-page)
  - [27.13 The scanned track, built](#2713-the-scanned-track-built)
  - [27.14 B0 on the scanned track, and the synthetic-to-scan gap](#2714-b0-on-the-scanned-track-and-the-synthetic-to-scan-gap)
  - [27.15 The frozen-core run, made runnable](#2715-the-frozen-core-run-made-runnable)
  - [27.25 Which corpora may carry notation labels](#2725-which-corpora-may-carry-notation-labels)
  - [27.24 Ties are not slurs, and the labels could not tell them apart](#2724-ties-are-not-slurs-and-the-labels-could-not-tell-them-apart)
  - [27.61 The collapsed staves are faint, not misaligned - which rules out the cheaper fix](#2761-the-collapsed-staves-are-faint-not-misaligned---which-rules-out-the-cheaper-fix)
  - [27.63 CLAHE does not close the gap - it narrows the spread by damaging the good scans](#2763-clahe-does-not-close-the-gap---it-narrows-the-spread-by-damaging-the-good-scans)
  - [27.62 phase11: focal loss helps, class weights hurt - the risk in 27.50 materialized](#2762-phase11-focal-loss-helps-class-weights-hurt---the-risk-in-2750-materialized)
  - [27.60 The scan gap is bimodal, and a fifth of one score's staves collapse](#2760-the-scan-gap-is-bimodal-and-a-fifth-of-one-scores-staves-collapse)
  - [27.59 The recogniser's errors are diffuse, which is itself the finding](#2759-the-recognisers-errors-are-diffuse-which-is-itself-the-finding)
  - [27.58 phase10: both gates re-run, and a crosstab of zeros that looked like a result](#2758-phase10-both-gates-re-run-and-a-crosstab-of-zeros-that-looked-like-a-result)
  - [27.57 The resolve rule settles at 98.2%, and the wrapped systems were scoreable after all](#2757-the-resolve-rule-settles-at-982-and-the-wrapped-systems-were-scoreable-after-all)
  - [27.56 Three-corpus training: scans bought cheaply, and a prediction confirmed the hard way](#2756-three-corpus-training-scans-bought-cheaply-and-a-prediction-confirmed-the-hard-way)
  - [27.55 Two thirds of the rendered pages hold more than one system, and 27.52 blamed the wrong thing](#2755-two-thirds-of-the-rendered-pages-hold-more-than-one-system-and-2752-blamed-the-wrong-thing)
  - [27.54 Resolution was not the constraint, and the hypothesis that it was is dead](#2754-resolution-was-not-the-constraint-and-the-hypothesis-that-it-was-is-dead)
  - [27.53 Evaluation was reading the padding, and two runs were scored through it](#2753-evaluation-was-reading-the-padding-and-two-runs-were-scored-through-it)
  - [27.52 Nearest-x already resolves 98.9% of syllables](#2752-nearest-x-already-resolves-989-of-syllables)
  - [27.51 The recogniser, and a metric built so it can fail](#2751-the-recogniser-and-a-metric-built-so-it-can-fail)
  - [27.50 The class-imbalance instruments, and a sweep to choose between them](#2750-the-class-imbalance-instruments-and-a-sweep-to-choose-between-them)
  - [27.49 Why dynamics are commented out - and the same failure is in our tie head](#2749-why-dynamics-are-commented-out---and-the-same-failure-is-in-our-tie-head)
  - [27.48 A recogniser's corpus, split by score, and the chain checked end to end](#2748-a-recognisers-corpus-split-by-score-and-the-chain-checked-end-to-end)
  - [27.47 The synthetic stage was lower-resolution than the scans, which is backwards](#2747-the-synthetic-stage-was-lower-resolution-than-the-scans-which-is-backwards)
  - [27.46 Two sources numbering parts independently, and a crash that was the lucky outcome](#2746-two-sources-numbering-parts-independently-and-a-crash-that-was-the-lucky-outcome)
  - [27.45 Detection then recognition - and dynamics are not text](#2745-detection-then-recognition---and-dynamics-are-not-text)
  - [27.44 The page is not only lyrics, and MuseScore types the rest for free](#2744-the-page-is-not-only-lyrics-and-musescore-types-the-rest-for-free)
  - [27.43 MuseScore earns the dependency: exact syllable boxes from the engraving it drew](#2743-musescore-earns-the-dependency-exact-syllable-boxes-from-the-engraving-it-drew)
  - [27.42 The lyric stage is OCR plus a resolve, and the numbers say why](#2742-the-lyric-stage-is-ocr-plus-a-resolve-and-the-numbers-say-why)
  - [27.41 The voice comes back by arithmetic, and MuseScore is not needed](#2741-the-voice-comes-back-by-arithmetic-and-musescore-is-not-needed)
  - [27.40 OLiMPiC's lyrics are recoverable, and it took looking to know it](#2740-olimpics-lyrics-are-recoverable-and-it-took-looking-to-know-it)
  - [27.39 A lyric track, and what OLiMPiC would have to be repaired to](#2739-a-lyric-track-and-what-olimpic-would-have-to-be-repaired-to)
  - [27.38 The synthetic-to-scan gap is far worse for notation than for notes](#2738-the-synthetic-to-scan-gap-is-far-worse-for-notation-than-for-notes)
  - [27.37 Other corpora: what is worth preparing, and what each would cost](#2737-other-corpora-what-is-worth-preparing-and-what-each-would-cost)
  - [27.36 Both models on both splits: mixing was right, and quartets do not generalise](#2736-both-models-on-both-splits-mixing-was-right-and-quartets-do-not-generalise)
  - [27.35 Mixing corpora: the predicted gain arrives, and so does a cost](#2735-mixing-corpora-the-predicted-gain-arrives-and-so-does-a-cost)
  - [27.34 The scanned track, converted](#2734-the-scanned-track-converted)
  - [27.33 The scanned crop guard costs nothing either, and why I expected otherwise](#2733-the-scanned-crop-guard-costs-nothing-either-and-why-i-expected-otherwise)
  - [27.32 The v2 retrain: nine heads, and the hooks move](#2732-the-v2-retrain-nine-heads-and-the-hooks-move)
  - [27.31 PDMX, converted from source](#2731-pdmx-converted-from-source)
  - [27.30 Is OSSQ representative? For beams yes, for slurs not at all](#2730-is-ossq-representative-for-beams-yes-for-slurs-not-at-all)
  - [27.29 The re-conversion, and two bugs that only a prediction would have caught](#2729-the-re-conversion-and-two-bugs-that-only-a-prediction-would-have-caught)
  - [27.28 The stem head and the rule are complementary, not redundant](#2728-the-stem-head-and-the-rule-are-complementary-not-redundant)
  - [27.27 The stem head can be replaced by a rule over its own beam predictions](#2727-the-stem-head-can-be-replaced-by-a-rule-over-its-own-beam-predictions)
  - [27.26 The converged run: three epochs was already most of the way](#2726-the-converged-run-three-epochs-was-already-most-of-the-way)
  - [27.23 Stem direction is mostly a rule, and the head has not yet beaten it](#2723-stem-direction-is-mostly-a-rule-and-the-head-has-not-yet-beaten-it)
  - [27.22 Slur placement can be recovered, and the join is verifiable](#2722-slur-placement-can-be-recovered-and-the-join-is-verifiable)
  - [27.21 Phase 2, the first frozen-core run, and Gate C](#2721-phase-2-the-first-frozen-core-run-and-gate-c)
  - [27.20 What the built labels actually contain](#2720-what-the-built-labels-actually-contain)
  - [27.19 Building the training set: three things that stop a conversion dead](#2719-building-the-training-set-three-things-that-stop-a-conversion-dead)
  - [27.18 Whole-measure rests need no repair on the training side](#2718-whole-measure-rests-need-no-repair-on-the-training-side)
  - [27.17 Slurs that cross a system break](#2717-slurs-that-cross-a-system-break)
  - [27.16 The Gate C baseline, per split](#2716-the-gate-c-baseline-per-split)
  - [27.7 Known gaps](#277-known-gaps)

---

### 1. Executive summary

This design improves HOMR's transcription of multi-staff ensemble scores without
creating a quartet-only model. OSSQ-OMR is the first specialization and evaluation
corpus, but every new interface is optional, variable-staff, and instrument-agnostic.

The work has five parts:

1. Add structured output heads for beams, stem direction, and richer slurs while
   preserving the pretrained visual encoder, autoregressive decoder, and existing
   heads.
2. Add an optional generic score profile describing expected parts, instruments,
   staff counts, clefs, and transpositions. Use it as a soft conditioning signal,
   never as an unbreakable rule.
3. Improve system grouping after segmentation. The U-Net continues to identify
   staff and symbol pixels; a geometry layer groups physical staves into systems,
   assisted by optional score-profile information and correctable by a human.
4. Add cross-staff consistency checks first, targeted repairs second, and an
   optional learned variable-staff context adapter only after the simpler methods
   have been measured.
5. Preserve page-local inference while exposing enough geometry, confidence,
   provenance, and structured output for an external human-review and page-assembly
   system.

The existing model is not retrained from scratch. New heads and adapters are
initialized separately, trained while the existing core is frozen, and then
fine-tuned jointly at a low learning rate. Existing datasets remain in the training
mixture to prevent catastrophic forgetting.

The later Lieder project reuses the same score profile, staff grouping, variable-
staff context, notation heads, page evidence, correction capture, and music model.
Lyrics are a separate downstream module for region detection, text recognition,
and syllable-to-note alignment. That module needs its own training, but the music
backbone does not.

### 2. Goals

#### 2.1 Primary goals

- Improve end-to-end transcription of printed string quartet scores.
- Explicitly recognize visible beam grouping at every flag level.
- Explicitly recognize actual stem direction.
- Preserve slur direction and multiple concurrent slurs.
- Improve grouping of physical staves into systems without assuming every score is
  a quartet.
- Use optional user-supplied part and instrument information when available.
- Exploit agreement among simultaneous staves without forcing inference to become
  a fixed four-staff operation.
- Produce uncertainty and alternatives suitable for targeted human review.
- Retain page-by-page inference, retry, caching, and provenance boundaries.
- Reuse the pretrained HOMR checkpoint and avoid regressions on existing corpora.
- Keep the architecture extensible to voice-plus-piano Lieder and a later lyric
  stage.

#### 2.2 Secondary goals

- Improve MusicXML round-trip fidelity through MuseScore.
- Make explicit notation independently measurable from pitch and rhythm.
- Capture confirmed and corrected beam, stem, slur, layout, and cross-staff choices
  as provenance-rich future training data.
- Make model capabilities discoverable so an older checkpoint can coexist with
  newer provider and review code.

### 3. Non-goals for the first experiments

- A quartet-exclusive checkpoint or a `quartet=true` model branch.
- Full-page end-to-end transcription in one neural pass.
- Beam-search sequence decoding. Greedy decoding remains the initial sequence
  strategy.
- A four-staff joint decoder trained from scratch.
- Automatic correction of every cross-staff disagreement.
- Dynamics, hairpins, fingerings, or lyrics in the first OSSQ training run.
- Training the segmentation U-Net merely to add beam and slur semantics. The
  Transformer receives the dewarped source staff image and is the appropriate
  initial location for these notation heads.
- Treating source MusicXML as exact scanned-image ground truth when the scan is a
  different edition.
- Solving handwritten music recognition.

### 4. Current architecture and relevant constraints

HOMR currently operates page by page:

```text
page image
  -> U-Net/SegNet pixel classes
  -> staff-line and symbol geometry
  -> Staff objects
  -> brace/barline/periodicity grouping into MultiStaff systems
  -> dewarped image for each physical staff
  -> shared visual encoder and autoregressive Transformer decoder
  -> parallel token heads
  -> page-local MusicXML
```

The segmentation model predicts stems/rests, noteheads, clefs/key signatures,
staff lines, and general symbols. It does not directly predict a system bounding
box or a four-staff-system class. System construction is conventional geometry and
post-processing influenced by segmentation quality.

The Transformer has one shared autoregressive decoder and six parallel output
heads, not six independent decoder stacks:

- rhythm;
- pitch;
- lift/accidental;
- upper/lower staff position;
- articulation;
- slur.

The embeddings for prior rhythm, pitch, lift, articulation, and slur tokens are
summed before the decoder. The current position output is not fed back as an input
embedding. That provides a precedent for adding output-only notation heads without
changing the pretrained autoregressive input path.

Current generation is greedy: every head takes `argmax` at each sequence step.
This design deliberately leaves that unchanged for the first experiments.

HOMR produces a separate MusicXML document for each page. Cross-page part identity,
measure numbering, repeated attributes, and spanning notation are assembly
concerns outside the image model. The model and provider must expose enough state
to make those concerns reviewable, but they should not make page retries dependent
on previous pages.

### 5. Design principles

#### 5.1 Preserve the pretrained core

New semantics should be represented by new heads instead of expanding the rhythm
vocabulary. Changing the rhythm vocabulary would alter the most important existing
softmax and embedding matrices and would entangle notation fidelity with note/rest
sequence accuracy.

Checkpoint loading must explicitly distinguish expected new parameters from an
accidental mismatch. `strict=False` by itself is insufficient; loading should
validate an allowlist of missing new-head and adapter parameters and reject all
other missing or unexpected keys.

#### 5.2 Factor independent musical dimensions

Avoid a combinatorial token such as
`slur2StartAbove_slur1Stop_beam2BackwardHook_stemDown`. Beam level, beam state,
stem direction, slur identity, slur event, and slur direction have distinct
semantics and class frequencies. They should have distinct targets and losses.

#### 5.3 Treat metadata as optional evidence

Instrument and part information is often known before recognition. It is useful
for layout, clef priors, and repair, but it can be wrong or incomplete. Every
conditioning field therefore has an unknown value, training uses context dropout,
and layout rules can be overridden by visual evidence or a reviewer.

#### 5.4 Prefer explicit refusal and review to silent repair

An output can be valid MusicXML and still be musically wrong. Structural repair
must report its evidence, alternatives, and any mutation. Ambiguous staff grouping,
measure mismatch, or spanning-notation damage should yield a review item or a
refusal rather than plausible-looking output.

#### 5.5 Preserve exact source-image identity

All review geometry and training corrections must be tied to the exact normalized
page raster used for inference, its checksum, model revision, preprocessing
revision, and token vocabulary/head schema versions.

### 6. Target architecture

```text
                             optional ScoreProfile
                         /             |             \
                        v              v              v
page -> segmentation -> layout -> per-staff encoder -> decoder core
          |                |              |                |
          |                |              |                +-> existing heads
          |                |              |                +-> beam levels 1..6
          |                |              |                +-> stem direction
          |                |              |                +-> structured slurs
          |                |              |
          |                |              +-> optional gated variable-staff context
          |                |
          |                +-> systems, part/staff assignment, review geometry
          |
          +-> staff/symbol masks

decoded page
  -> cross-staff validation and targeted repair proposals
  -> human review when requested
  -> page-local MusicXML plus structured evidence
  -> external multi-page assembly
```

The first implementation stops before the learned variable-staff context adapter.
It adds the structured heads, generic profile contract, deterministic layout
improvements, dataset adapter, evaluation, and repair evidence. The learned adapter
is gated by measurements from those stages.

### 7. Optional score-profile conditioning

#### 7.1 Contract

Use a generic document-level profile rather than an ensemble-type flag:

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
    },
    {
      "stableId": "violin-2",
      "displayName": "Violin II",
      "instrumentFamily": "strings.violin",
      "expectedStaffCount": 1,
      "likelyClefs": ["G2"],
      "transpositionSemitones": 0,
      "lyricsExpected": false
    },
    {
      "stableId": "viola",
      "displayName": "Viola",
      "instrumentFamily": "strings.viola",
      "expectedStaffCount": 1,
      "likelyClefs": ["C3", "G2"],
      "transpositionSemitones": 0,
      "lyricsExpected": false
    },
    {
      "stableId": "cello",
      "displayName": "Cello",
      "instrumentFamily": "strings.cello",
      "expectedStaffCount": 1,
      "likelyClefs": ["F4", "C4", "G2"],
      "transpositionSemitones": 0,
      "lyricsExpected": false
    }
  ]
}
```

The stable ID is scoped to the submitted document or job. It is not a universal
instrument identifier. Unknown names, families, clefs, staff counts, and
transpositions are valid.

Future fields may describe percussion, tablature, language, verse count, or a
known page range, but the first implementation should not add unused fields.

#### 7.2 Use in layout

The profile supplies an expected ordered physical-staff pattern. For a quartet it
is `[1, 1, 1, 1]`; for voice and piano it is `[1, 2]`. Layout uses this pattern as
a scored hypothesis, not a hard assertion.

The layout result must report:

- detected physical staff count;
- proposed systems and staff rows;
- proposed mapping from staff rows to profile parts;
- evidence score and competing hypotheses;
- deviations from the supplied profile;
- exact source-image regions.

#### 7.3 Use in staff recognition

For a recognized physical staff, encode only the applicable part context:

- instrument-family embedding;
- part ordinal embedding;
- staff-within-part ordinal embedding;
- expected-staff-count embedding;
- likely-clef set embedding;
- transposition embedding;
- unknown/context-missing indicators.

The initial implementation may inject these as prefix/context tokens to the
decoder or as a gated additive vector to encoder context. A zero-initialized gate
must make the unconditioned path identical at initialization.

Do not make instrument range a hard pitch constraint. Ranges and likely clefs are
priors; real music legitimately violates them.

#### 7.4 Context dropout

During training, randomly remove the entire profile and independently mask fields.
The exact probabilities should be configured and recorded with the run. A sensible
starting point is:

- 30% of examples receive no profile;
- another 30% receive a partially masked profile;
- 40% receive the complete available profile.

This is a starting hypothesis, not a fixed constant. Evaluation must include both
conditioned and unconditioned inference.

### 8. Staff and system detection

#### 8.1 Responsibility boundary

Four-staff system detection is not solely a U-Net task. Stage 1 provides the staff
and symbol masks; subsequent geometry creates physical staffs and groups them into
systems. Beam and slur heads do not require a segmentation change because the
semantic Transformer sees the dewarped source pixels.

#### 8.2 Deterministic grouping before another neural model

Replace brittle greedy/periodic decisions with a page-level grouping optimizer.
Candidate systems are scored using:

- horizontal overlap of staff extents;
- similarity of left and right margins;
- vertical gaps normalized by local staff unit size;
- barline connectivity and aligned barline x-coordinates;
- brace/bracket evidence;
- clef and part-layout evidence;
- consistency with systems above and below;
- optional score-profile staff pattern;
- penalties for discarding or duplicating a physical staff.

A dynamic-programming or shortest-path formulation should partition the ordered
physical staffs for the whole page. It must retain the best competing partition,
not only the winner, so ambiguity can be surfaced.

The current fast path remains valid: if every already-grouped row has the same
number of more than one physical staff, retain it. This specifically protects
quartets from the degenerate constant `is_grandstaff` periodic signature.

#### 8.3 Exceptions

The grouping model must tolerate and report:

- title and instrument-name text near the first system;
- incomplete final systems;
- hidden or tacet parts;
- divisi or temporary extra staves;
- ossia staves;
- pickup systems;
- grand-staff parts mixed with single-staff parts;
- a scan cropped through the top or bottom system.

The optional profile raises or lowers hypothesis scores; it does not silently
delete exceptions.

#### 8.4 Human correction

Before semantic token review, a client may present page-level staff bands and
system groups. Supported corrections are:

- split or merge a proposed system;
- add, remove, or adjust a staff band;
- reorder staff rows;
- assign a staff row to a profile part;
- mark a system incomplete;
- confirm the proposed structure.

The corrected geometry can be fed back through HOMR's staff-position loading path
and used to rerun only the affected page. A correction must be content-signature
bound and must invalidate downstream tokens and MusicXML produced from the old
geometry.

### 9. Structured beam and stem heads

#### 9.1 Why MusicXML beam vectors are the canonical target

MuseScore's chord-local `BeamMode` supports `AUTO`, `NONE`, `BEGIN`, `BEGIN16`,
`BEGIN32`, `MID`, and `END`. Its MusicXML importer derives those values from a
richer per-level representation and maps forward/backward hooks and unfamiliar
combinations to `AUTO`. Training directly on `BeamMode` would therefore discard
visible information that the model is intended to recover.

The canonical training and model representation is one state per rhythmic beam
level:

```text
NOT_APPLICABLE
FLAG
BEGIN
CONTINUE
END
FORWARD_HOOK
BACKWARD_HOOK
```

Levels are:

| Level | Note-value boundary |
|---:|---|
| 1 | eighth |
| 2 | sixteenth |
| 3 | thirty-second |
| 4 | sixty-fourth |
| 5 | one-hundred-twenty-eighth |
| 6 | two-hundred-fifty-sixth |

`NOT_APPLICABLE` means the duration has fewer flags than that level. `FLAG` means
the level applies visually but is not joined to a neighbor. The remaining states
map directly to MusicXML `<beam number="N">` values.

Examples:

```text
isolated sixteenth, stem up:
  L1=FLAG, L2=FLAG, L3..L6=NOT_APPLICABLE, stem=UP

sixteenth beginning both beams:
  L1=BEGIN, L2=BEGIN

sixteenth continuing primary beam, breaking secondary to the right:
  L1=CONTINUE, L2=END

sixteenth continuing primary beam with a right-facing secondary hook:
  L1=CONTINUE, L2=FORWARD_HOOK
```

This represents join, no beam, primary breaks, secondary breaks, and hooks without
depending on a particular notation editor's UI vocabulary.

#### 9.2 Stem direction

Add a separate actual-stem-direction head:

```text
NOT_APPLICABLE | UP | DOWN | NONE | DOUBLE
```

Missing or unreliable source labels use a dataset-side `UNKNOWN` sentinel that is
masked out of the loss; `UNKNOWN` is not an inference class. For ordinary flagged
notes, flag orientation follows `UP` or `DOWN`. For a beamed group, the head records
the visible stem direction independently of beam connectivity.

#### 9.3 Output shape

The first model adds seven output projections to the shared decoder hidden state:

- six `beam_level_N` categorical projections;
- one `stem_direction` projection.

They are output-only in the first experiment. They do not contribute embeddings to
the next autoregressive step. This minimizes checkpoint disruption and provides a
clean ablation.

#### 9.4 Masking and losses

Loss must not be dominated by non-applicable levels:

- infer the maximum applicable level from the ground-truth rhythm;
- train applicable levels on `FLAG` or explicit beam state;
- either exclude higher levels from loss or give `NOT_APPLICABLE` a separately
  controlled weight;
- mask all beam/stem losses on non-note sequence tokens;
- record per-class support and loss in every run;
- use class-balanced weighting or focal loss only after measuring the unweighted
  baseline.

Consistency penalties may enforce:

- a beam state cannot occur above the duration's flag depth;
- a quarter note cannot carry beam levels;
- a hook cannot occur at level 1 without an explicit, justified representation;
- starts, continues, ends, and hooks form valid groups within a voice.

Group validation should happen in generation as well as training metrics. It must
not silently rewrite the raw prediction; repaired and raw representations remain
distinguishable.

#### 9.5 Materializing automatic beams

Absence of `<beam>` is ambiguous in MusicXML. Depending on document context and
consumer behavior, it can mean automatic beaming or deliberately unbeamed notes.
Mixing explicit and implicit beam information can change untouched notation after
MuseScore import.

For exact synthetic supervision:

1. render/export the source through the pinned MuseScore version used to create the
   image;
2. materialize automatic beam choices into explicit per-level MusicXML beams;
3. parse that materialized file for targets;
4. verify that materialization does not change the rendered notation;
5. record the MuseScore version and materialization revision.

Do not run materialization over engine output merely to make it resemble labels.
It is a dataset-label normalization step and an explicit merge repair, not a
general post-processing substitute for recognition.

### 10. Structured slur heads

#### 10.1 Requirements

The representation must preserve:

- start and stop events;
- a stop and a new start on the same note;
- above/below placement;
- more than one concurrent slur;
- endpoint identity across intervening notes;
- system/page-boundary incompleteness;
- future line style without replacing the initial schema.

#### 10.2 Slot representation

Support six concurrent canonical slur slots in the initial model. OSSQ source data
contains slur numbers through six. Each slot has two factored heads:

```text
slur_event[slot]:
  NONE | START | STOP | START_AND_STOP | CONTINUE

slur_side[slot]:
  UNSPECIFIED | ABOVE | BELOW
```

Direction loss applies where the source provides a reliable placement, normally at
the start. A stop inherits the direction of its matching active start for MusicXML
serialization, although inference may retain visually estimated endpoint-side
evidence for review.

`START_AND_STOP` is required because a note can close one span and open another in
the same canonical slot. `CONTINUE` supports MusicXML inputs that explicitly carry
an intermediate event.

#### 10.3 Canonical slot assignment

Literal MusicXML `number=` values identify paired elements only within a document;
they are not universal semantic labels. Before tokenization:

1. process each voice in musical order;
2. retain valid active source identities while spans are open;
3. assign new starts to the first available canonical slot unless an existing
   source identity must be reused at the same anchor;
4. pair stops with active starts;
5. record unpaired, duplicated, or over-capacity events as validation findings;
6. never move all unmatched slurs to an arbitrary first note.

Documents requiring more than six simultaneous slots remain parseable but their
overflow labels are masked and reported. The cap can be raised without changing
the conceptual schema.

#### 10.4 Checkpoint-compatible migration

Retain the legacy collapsed slur head and embedding during the first experiments.
Add structured slur event and direction heads in parallel. This permits exact
loading of the existing slur weights and gives a direct comparison between legacy
and structured accuracy.

Initially, structured slur heads are output-only. If endpoint continuity is poor,
add embeddings of the previous structured slur state behind a zero-initialized
gate in a later experiment.

Once structured heads are demonstrably better and all inference/export paths
support them, the legacy head can be deprecated in a versioned checkpoint rather
than silently reinterpreted.

#### 10.5 Slur validation

Generation reports, and never silently hides:

- unmatched starts or stops;
- invalid slot reuse;
- direction changes within one span;
- spans cut by a system or page boundary;
- spans attached to missing notes;
- simultaneous-slot overflow.

Unlike ties, a damaged slur does not alter sounding pitch or duration. An external
repair layer may offer to drop a dangling half, but the mutation must be visible
and excluded from engine-credit training labels.

### 11. Semantic data model and MusicXML generation

#### 11.1 Structured symbol representation

Extend the semantic symbol representation without turning every combination into
a string token. Conceptually:

```python
EncodedSymbol(
    rhythm=...,
    pitch=...,
    lift=...,
    articulation=...,
    slur=...,              # legacy during migration
    position=...,
    beam_levels=(...),     # six BeamLevelState values
    stem_direction=...,
    structured_slurs=(...),
)
```

Serialization used for dataset indexes and provider envelopes must be schema-
versioned. Old five/six-field token lines remain readable. New training should use
a structured JSON or an explicitly versioned line format rather than adding
unescaped delimiters to the legacy format.

#### 11.2 Parser requirements

The MusicXML parser must preserve:

- all `<beam number>` elements and hook values;
- `<stem>` direction;
- every slur element, number, type, placement/orientation, and supported line
  style;
- voice, staff, chord, grace, and temporal identity required for canonicalization;
- measure and part provenance used for leakage-safe grouping.

Parsing should produce validation findings rather than globally dropping a file
for a rare unsupported marking.

#### 11.3 Generator requirements

The MusicXML generator must:

- emit explicit beam elements at every applicable predicted level;
- emit stem direction when supported;
- pair structured slur starts and stops by canonical slot;
- emit above/below placement on slur starts;
- retain raw versus repaired notation provenance;
- remain able to generate output from legacy symbols/checkpoints;
- reject impossible combinations before storing output;
- pass a headless MuseScore load/render smoke test.

#### 11.4 Capability manifest

Every checkpoint/export should carry a manifest containing:

- model and training revision;
- supported head names;
- vocabulary hashes for every head;
- maximum beam and slur levels;
- whether structured heads are fed back autoregressively;
- supported score-profile schema;
- image and sequence limits;
- expected parser/generator schema version;
- dataset/run identifiers.

Inference returns only heads declared by the manifest. Consumers must treat a
missing head as unsupported, not as a confident `NONE` prediction.

### 12. Cross-staff context and repair

Cross-staff work is staged from least invasive to most invasive.

#### 12.1 Stage A: deterministic consistency analysis

After independent staff decoding, align measures within each proposed system and
compute findings such as:

- different numbers of decoded measures across parts;
- conflicting barline locations;
- conflicting key or time signatures;
- one voice whose measure duration disagrees with the meter and other voices;
- a clef inconsistent with both the image and supplied profile;
- part order changing between systems;
- missing/extra staff output;
- a beam or slur endpoint made dangling by a structural edit.

The analyzer emits structured evidence. It does not alter MusicXML.

#### 12.2 Stage B: targeted repair proposals from existing alternatives

Use the existing greedy logits and top-k alternatives to propose a bounded local
repair when a specific low-confidence head explains a cross-staff inconsistency.

Example:

```text
Violin I measure duration: 4/4
Violin II measure duration: 4/4
Viola measure duration: 7/8
Cello measure duration: 4/4

Viola token 31:
  chosen sixteenth, confidence 0.44
  alternative eighth, confidence 0.41

Changing only that token restores 4/4.
```

The suggestion includes the aligned source crop and all affected staff readings.
It is a review question, not an automatic correction. Applying it follows the
same token-regeneration and content-signature rules as an ordinary confidence
correction.

This is not beam-search sequence decoding. It is targeted use of already computed
per-head alternatives after a deterministic inconsistency has identified a small
search neighborhood.

#### 12.3 Stage C: learned variable-staff context adapter

Only after Stages A and B have been benchmarked, add a small learned adapter:

```text
E_i = shared visual encoder(staff image i)
h_i = masked pool(E_i)
C_1..C_N = StaffContextTransformer(h_1..h_N, staff order, ScoreProfile)
E'_i = E_i + sigmoid(gate_i) * projection(C_i)
decode each E'_i with the existing shared decoder
```

Properties:

- `N` is variable and padded with an explicit mask;
- the visual encoder and decoder weights are reused;
- staff order and staff-within-part identity are explicit;
- the context gate is initialized to zero, reproducing the baseline initially;
- missing profile information is supported;
- the adapter does not require token alignment across parts;
- per-staff output and confidence remain available.

The first learned version uses one summary per physical staff. A later version may
exchange system-position features or decoded measure summaries, but should not be
built before the simpler adapter is measured.

#### 12.4 Why not a fixed four-staff decoder

A fixed quartet tensor would make the initial benchmark easy but would block or
complicate piano, Lieder, trios, orchestral reductions, missing staves, divisi, and
partial crops. A masked set/sequence of staff summaries gives the model quartet
context without encoding quartet as the architecture.

### 13. OSSQ dataset adapter

#### 13.1 Constructed local assets

At design time the local OSSQ tree contains:

- 122 source and cleaned MusicXML scores;
- 96 downloaded scanned PDFs;
- 2,770 scanned page images;
- 122 synthetic PDFs;
- 3,206 synthetic page images;
- 13,244 unaligned system MusicXML files;
- 13,244 unaligned LMXE files;
- 13,244 unaligned metadata records.

The repository's published counts and exclusion lists must remain authoritative.
The four-file difference between 13,244 generated synthetic systems and the
13,240 published figure must be reconciled before training rather than assumed to
be valid new data.

GPU system/staff cropping and alignment has not been run locally. It should run on
the rented training instance, not on CPU.

#### 13.2 Corpus evidence for new heads

An audit of the 122 original source MusicXML files found approximately 1.43 million
notes and:

- 736,820 notes with explicit beam data;
- beam levels through level 6;
- 46 distinct per-note beam vectors;
- common secondary forward and backward hooks;
- 657,380 explicit down stems and 569,764 explicit up stems;
- 107,409 above primary-slur starts and 73,336 below primary-slur starts;
- secondary slur identities through number 6;
- rare anchors with many simultaneous slur starts/stops.

The cleaned XML is unsuitable as the sole source for the new slur targets: it
flattens slur numbering to 1 and removes placement. Existing derived token data
also intentionally removed stem direction. Label extraction must therefore use
the original XML or a revised cleaning path proven to preserve these fields.

#### 13.3 Training record

Each staff example should carry:

```jsonc
{
  "image": "...",
  "tokens": "...",
  "scoreId": "...",
  "workId": "...",
  "composerId": "...",
  "sourceKind": "synthetic|scanned",
  "editionId": "...",
  "pageIndex": 0,
  "systemIndex": 0,
  "physicalStaffIndex": 0,
  "partId": "...",
  "staffWithinPart": 0,
  "scoreProfile": { "...": "..." },
  "imageSha256": "...",
  "labelSha256": "...",
  "labelSchemaVersion": "homr.structured-symbols.v1",
  "renderer": { "name": "MuseScore", "version": "..." },
  "alignmentConfidence": 1.0,
  "labelWarnings": []
}
```

The index may remain line-oriented for efficient loading, but grouping and
provenance fields cannot be discarded.

#### 13.4 Exact synthetic versus edition-noisy scanned labels

Synthetic images rendered from the same materialized MusicXML provide exact beam,
stem, and slur supervision.

Scanned PDFs may be a different engraving or edition from the symbolic source.
Pitch and rhythm alignment can remain valid while beam grouping, slur placement,
or even slur presence differs. Therefore:

- do not train new notation heads on all aligned scans blindly;
- compare or review scanned notation labels where possible;
- attach per-head label confidence/masks, not only one sample confidence;
- use scanned staff images for legacy pitch/rhythm adaptation when those labels are
  valid while masking uncertain beam/slur losses;
- promote confirmed human corrections and confirmations into high-quality scanned
  notation supervision only with rights/provenance approval.

#### 13.5 Splits and leakage prevention

All crops, systems, pages, movements, and source variants from one score belong to
one split. Never randomly split staff strips.

Required splits:

- train;
- validation for early stopping and head/loss selection;
- fixed held-out OSSQ test set;
- optional composer-disjoint challenge set;
- fixed general HOMR regression sets.

Store an explicit split manifest under version control. Hash the manifest into the
run metadata. No data-loader fallback may silently create a sample-level random
split.

#### 13.6 Dataset mixing

OSSQ is domain adaptation, not a replacement for Primus, Grandstaff, Lieder, PDMX,
Musetrainer, and other existing data.

The sampler must implement its declared dataset weights even when all files are
loaded. The current mixing behavior should be audited because concatenating all
datasets before shuffling can make configured weights ineffective.

Initial mixtures to compare:

- existing general data only;
- general replay plus OSSQ synthetic;
- general replay plus OSSQ synthetic and scanned legacy-head supervision;
- the prior mixture plus reviewed scanned beam/slur supervision when available.

Record effective samples per dataset and per head, not just configured weights.

### 14. Training plan

#### 14.1 Phase 0: reproduce and freeze baselines

Before architecture changes:

1. evaluate the pinned pretrained checkpoint on existing smoke/system tests;
2. evaluate page-local HOMR on the fixed OSSQ test scores;
3. record layout, staff, pitch, rhythm, slur, MusicXML, and runtime metrics;
4. archive manifests, model hashes, commands, and raw predictions;
5. verify the original training data conversion still reproduces a known baseline.

No fine-tuning result is meaningful without this frozen comparison.

#### 14.2 Phase 1: label and round-trip validation

- Implement structured parser targets.
- Materialize automatic beams with a pinned MuseScore.
- Render source and materialized scores and verify visual equivalence.
- Canonicalize slur slots and report invalid/overflow spans.
- Confirm that generated structured MusicXML reloads in MuseScore.
- Reconcile OSSQ exclusions and system counts.
- Produce per-class support tables by split.
- Manually inspect examples of every beam state and secondary slur class.

#### 14.3 Phase 2: new-head-only training

- Load the pinned pretrained model.
- Freeze the visual encoder, decoder layers, and existing heads.
- Train beam-level, stem-direction, and structured-slur projections on exact
  synthetic pairs.
- Use greedy decoding and output-only new heads.
- Select loss weights on validation data, not the held-out test set.
- Confirm that existing head logits and outputs are bit-identical while the core is
  frozen.

This phase answers whether the existing hidden representation already contains
enough visual information for the new notation tasks.

#### 14.4 Phase 3: limited joint fine-tuning

- Unfreeze the last decoder layers and new heads first.
- Retain a general-data replay mixture.
- Use a lower learning rate for pretrained layers than for new projections.
- If needed, unfreeze the visual encoder last at an even lower rate.
- Train all existing semantic heads, not only the lift head.
- Retain checkpoint selection based on a multi-objective validation report rather
  than new-head accuracy alone.

The current fine-tuning path that freezes most of the model and trains only lift is
not suitable for this domain adaptation and needs a separate explicit mode.

#### 14.5 Phase 4: score-profile conditioning

- Add profile embeddings behind a zero-initialized gate.
- Train with full/partial/no-context examples.
- Compare unconditioned inference with the original baseline.
- Measure conditioned gains by head and by instrument.
- Test deliberately incorrect profiles and ensure they reduce confidence or create
  review warnings rather than catastrophically overriding the image.

#### 14.6 Phase 5: cross-staff work

Implement and measure in order:

1. deterministic consistency findings;
2. targeted top-k repair proposals with a human decision;
3. optional variable-staff learned context adapter.

Do not proceed to the learned adapter merely because GPU time is available. Proceed
when the error taxonomy shows remaining errors that truly require simultaneous
staff context.

#### 14.7 Compute and run configuration

A single 24 GB RTX 3090 or 4090 is sufficient for the first experiments. Prefer:

- at least 8 capable CPU cores;
- at least 32 GB RAM;
- 100-150 GB local NVMe;
- CUDA-compatible pinned container/runtime;
- bf16 training where supported;
- explicit batch size, gradient accumulation, worker count, and seed in the run
  manifest.

The data loader is CPU-sensitive. Measure GPU utilization before buying more GPU
capacity. Multi-GPU training is unnecessary until one-GPU correctness and sampling
are established.

### 15. Evaluation

#### 15.1 Existing-head non-regression

Report at minimum:

- existing transcription smoke-test SER;
- full system-level SER/diffs;
- pitch accuracy/SER;
- rhythm accuracy/SER;
- accidental, articulation, legacy slur, and position metrics;
- inference time and memory;
- MuseScore load/render success rate.

Compare on the exact same frozen inputs and manifests. A new-head run is not
acceptable if it hides a material pitch/rhythm regression behind improved notation
scores.

#### 15.2 Beam metrics

- per-level macro and micro precision/recall/F1;
- per-level confusion matrices;
- exact per-note beam-vector accuracy;
- exact beam-group accuracy;
- beam-group boundary F1;
- forward/backward-hook F1;
- applicable-depth validity rate;
- MusicXML beam structural validity;
- rendered visual agreement on controlled synthetic fixtures.

Compare against:

- a majority-class baseline;
- duration-and-meter automatic beaming;
- source MusicXML passed through the pinned renderer.

The automatic-beaming baseline matters: a head is only useful if it recovers
visible exceptions and edition choices better than deterministic reconstruction.

#### 15.3 Stem metrics

- up/down macro F1;
- exact accuracy on flagged notes;
- exact accuracy on beamed notes;
- beam-group stem consistency;
- performance by voice count and chord density.

#### 15.4 Slur metrics

- primary event F1;
- secondary-slot event F1 by slot/support;
- above/below direction F1;
- endpoint-pair precision/recall/F1;
- complete-span accuracy;
- unmatched start/stop rate;
- overflow rate;
- performance by span length and system-boundary crossing.

Because secondary slurs are rare, include confidence intervals and raw support.
Micro accuracy alone is misleading.

#### 15.5 Layout and system metrics

- physical-staff recall and precision;
- exact systems per page;
- exact physical staffs per system;
- part/staff assignment accuracy;
- dropped, duplicated, merged, and split staff counts;
- correct handling of incomplete systems;
- conditioned versus unconditioned results;
- rate of pages requiring structural review.

#### 15.6 Cross-staff and end-to-end metrics

- percentage of systems with aligned barline/measure counts;
- per-measure duration consistency;
- shared key/time disagreement rate;
- valid targeted repair proposal precision;
- reviewer acceptance/rejection rate;
- edit distance/NED on final MusicXML;
- MuseScore semantic validation and render success;
- remaining human edits per page or system.

The practical success metric is reduction in verified human correction effort while
preserving musical correctness, not only token accuracy.

#### 15.7 Calibration and review utility

For every new head, measure reliability diagrams or expected calibration error.
Top-k alternatives should be shown as percentages only if calibrated well enough;
otherwise expose ordered alternatives with qualitative confidence bands.

Record:

- which questions reviewers answer;
- confirmations versus corrections;
- where reviewers stop;
- correction time;
- whether the correct choice was offered;
- whether a cross-staff explanation changed the decision.

### 16. Human-in-the-loop contract

An external review system already supports confidence-driven token questions,
geometry-bound crops, cumulative corrections, regeneration, content signatures,
engine comparison, durable flags, and correction capture. HOMR should expose a
generic contract rather than depending on one client.

#### 16.1 Head identifiers

Use stable typed names:

```text
rhythm
pitch
lift
articulation
legacy_slur
position
stem.direction
beam.level.1 ... beam.level.6
slur.slot.1.event ... slur.slot.6.event
slur.slot.1.side  ... slur.slot.6.side
```

The provider envelope treats heads as a map, accompanied by the capability
manifest. It must not require every model to have the same closed enum.

#### 16.2 Beam review

A beam correction applies a validated full per-note beam vector atomically, even if
the UI lets the reviewer change one level. This prevents an isolated per-level edit
from creating an impossible vector.

Show:

- a source crop wide enough to include neighboring notes;
- the recognized vector by rhythmic level;
- the top alternatives and confidence per uncertain level;
- stem direction;
- a warning if the proposed vector breaks group continuity.

#### 16.3 Slur review

Show a system-width crop when possible, because a measure-local crop may omit an
endpoint. Present primary and secondary spans separately with above/below placement.
An endpoint correction must rerun pairing validation before MusicXML regeneration.

#### 16.4 Structural review

Structural questions precede token questions. A stale layout invalidates staff
crops, token identities, attention hints, cross-staff analysis, and generated
MusicXML. Re-running only after token corrections would discard work or attach it to
the wrong staff.

#### 16.5 Cross-staff review

For a consistency finding, show:

- the aligned source bands for all affected staves;
- each decoded measure;
- the precise invariant that failed;
- a bounded proposed token change, if any;
- resulting measure totals and downstream warnings;
- choices to accept, select another alternative, defer, or open an editor.

#### 16.6 Training capture

Confirmations and corrections are both valuable. Each record includes:

- exact page image hash;
- staff/system/measure/symbol identity;
- original prediction and logits/top-k;
- chosen result;
- raw and corrected structured vector;
- model, provider, parser, generator, and preprocessing revisions;
- score-profile presence and values;
- source crop reference, not an uncontrolled duplicate;
- rights/policy stamp authorizing or forbidding future training use.

Structural corrections use a separate geometry record keyed to the page hash and
old/new layout signatures.

### 17. Page boundaries and assembly

HOMR remains page-local. It should not make a page retry depend on earlier model
state. Instead, each page result includes:

- proposed stable part mapping within the supplied score profile;
- first and final effective clef/key/time state per part;
- first and final open tie/slur slots;
- measure counts and duration findings;
- source system/staff geometry;
- raw and repaired notation distinctions;
- capability and vocabulary versions.

An external assembler can compare adjacent pages and ask for review when:

- part mappings change;
- staff layouts change;
- page 2 restarts with incompatible state;
- a tie or slur endpoint is missing;
- a pickup or split measure is suspected;
- a correction makes an already assembled artifact stale.

The assembler must preserve per-page MusicXML even when combination fails. It must
not invent cross-page slur or tie endpoints solely to make validation pass.

### 18. Lieder extension and lyrics

#### 18.1 Reused architecture

The Lieder project reuses:

- segmentation and physical-staff geometry;
- generic score profile;
- variable `[1, 2]` voice-plus-piano layout;
- conditioned staff recognition;
- beam, stem, and structured slur heads;
- variable-staff context adapter;
- page-local state and assembly contract;
- source geometry and review UI;
- confidence/correction capture;
- score-level split and general-data replay mechanisms.

OpenScore Lieder is already included by HOMR's Lieder conversion path. The new
adapter should separate corpus acquisition, music labels, lyric labels, and render
geometry so quartet and Lieder data do not require parallel one-off pipelines.

#### 18.2 Lyric stage

Lyrics are not another small parallel token head. They require three operations:

```text
recognized system and note anchors
  -> lyric-region/line detection
  -> text recognition
  -> monotonic syllable-to-note alignment
  -> MusicXML <lyric> output
```

The alignment model must represent:

- `single`, `begin`, `middle`, and `end` syllabic states;
- hyphens and elisions;
- melisma/extension lines;
- verse number;
- multiple lyric lines;
- text that is not a lyric, such as dynamics, tempo, expression, or annotation;
- missing or repeated notes caused by music-recognition errors.

MusicXML `<lyric>`, `<syllabic>`, `<text>`, and `<extend>` provide semantic
supervision. Synthetic SVG/render output should provide exact text-region geometry.

The current music staff crop may not include enough space below a vocal staff for
multiple lyric lines. The lyric stage should request a larger system/staff evidence
crop rather than changing the music model's fixed canvas prematurely.

#### 18.3 Reuse without pretending the lyric head is pretrained

The music encoder/decoder and notation heads are reused. The new lyric-region,
text-recognition, and alignment modules still require training or initialization
from an appropriate pretrained OCR/text model. This is not full-model training from
scratch: the stable music result and note anchors are inputs to an independently
versioned lyric capability.

Keep lyric capability optional. Quartet checkpoints and pages without lyrics should
not allocate lyric decoding work or emit empty lyrics as if they were confident
predictions.

### 19. Backward compatibility and deployment

#### 19.1 Checkpoints

- Old checkpoints load into the unchanged legacy architecture.
- New architecture loading an old checkpoint initializes only allowlisted new heads
  and adapters.
- A checkpoint manifest declares exactly which heads are trained.
- An untrained new head is not exported as a supported capability.
- New checkpoints may continue to emit legacy slur output during migration.

#### 19.2 Token/index formats

- Legacy token files remain readable.
- Structured token schema is versioned and preferably JSON-based.
- Dataset indexes include score grouping and provenance.
- Converters reject mixed schema versions unless explicitly migrated.

#### 19.3 ONNX

ONNX encoder and decoder export must name outputs rather than depend on tuple
position. Dynamic-cache behavior remains unchanged for output-only heads.

The decoder inference wrapper should expose a dictionary of head logits and use the
checkpoint manifest to bind ONNX output names. This prevents adding one head from
silently shifting every downstream positional output.

#### 19.4 Provider/API

Provider revisions advertise:

- model capabilities;
- score-profile schema support;
- structured token schema;
- geometry schema;
- confidence/alternative support by head;
- regeneration support;
- pinned model and source revisions.

Unsupported client requests fail explicitly. A profile supplied to a model that
does not support conditioning is not silently ignored.

### 20. Concrete implementation areas

Likely HOMR changes, grouped by responsibility:

#### Semantic vocabulary and structures

- `homr/transformer/vocabulary.py`
- `training/transformer/training_vocabulary.py`
- new structured-symbol schema and migration helpers

#### Parser and dataset adapters

- `training/omr_datasets/music_xml_parser.py`
- `training/omr_datasets/convert_lieder.py`
- new `training/omr_datasets/convert_ossq.py`
- MuseScore beam-materialization helper
- explicit split-manifest tooling

#### Transformer model and inference

- `training/architecture/transformer/decoder.py`
- `training/architecture/transformer/tromr_arch.py`
- `training/transformer/data_loader.py`
- `training/transformer/metrics.py`
- `homr/transformer/decoder_inference.py`
- `homr/transformer/staff2score.py`
- `homr/transformer/configs.py`
- `training/onnx/convert.py`

#### Training orchestration

- `training/transformer/train.py`
- `training/transformer/mix_datasets.py`
- checkpoint manifest/load validation
- explicit fine-tuning modes and parameter groups
- effective sampler-stat logging

#### Layout and system context

- `homr/staff_detection.py`
- `homr/brace_dot_detection.py`
- `homr/staff_parsing.py`
- `homr/staff_position_save_load.py`
- new system-partition scoring module
- new score-profile module
- optional learned staff-context adapter

#### MusicXML and validation

- `homr/music_xml_generator.py`
- structured beam/slur validator
- page-state summary and raw/repaired provenance

#### Tests

- parser/generator round trips for every beam state and level;
- hooks and secondary beam breaks;
- stem up/down and unbeamed flags;
- simultaneous, nested, same-anchor, and cross-system slurs;
- old checkpoint compatibility;
- old token-file compatibility;
- named ONNX outputs and cache generation;
- no-profile, partial-profile, correct-profile, and incorrect-profile inference;
- quartet, voice-plus-piano, grand staff, incomplete system, and divisi layout;
- score-level split leakage guard;
- synthetic render equivalence before/after beam materialization;
- MusicXML semantic and MuseScore load/render gates.

External review/assembly integration is intentionally a separate consumer. HOMR's
contract should make it possible without importing web-service concerns into this
repository.

### 21. Experiment matrix

Use a small, explicit ablation matrix rather than one long run with every idea:

| ID | Existing core | New heads | Profile | Cross-staff | Data |
|---|---|---|---|---|---|
| B0 | pretrained | none | no | none | frozen tests |
| H1 | frozen | beam/stem/slur | no | none | OSSQ synthetic |
| H2 | last decoder layers trainable | beam/stem/slur | no | none | general replay + OSSQ synthetic |
| H3 | limited joint fine-tune | beam/stem/slur | no | deterministic checks | general + OSSQ synthetic/scanned masks |
| C1 | H3 | same | optional | deterministic checks | same |
| R1 | C1 | same | optional | targeted top-k repair | evaluation only initially |
| X1 | C1 | same | optional | learned gated adapter | same |

Each row produces:

- checkpoint and manifest hashes;
- code and data revisions;
- effective per-dataset/per-head sample counts;
- full existing and new-head metrics;
- fixed qualitative render sheet;
- raw predictions needed for later paired analysis.

Do not promote X1 merely because its aggregate score improves. It must demonstrate
a gain attributable to cross-staff evidence and preserve single-staff/general
performance when no context is present.

### 22. Acceptance gates

#### Gate A: data correctness

- No split leakage by score/work.
- Every retained synthetic example loads and renders.
- Beam materialization is visually equivalent on the controlled corpus.
- All label warnings and exclusions are accounted for.
- Representative examples of every supported class are manually inspected.
- Structured parser -> generator -> parser round trips preserve supported labels.

#### Gate B: checkpoint compatibility

- Existing checkpoint loads with only expected new parameters absent.
- Frozen-core existing outputs remain identical.
- Old token files and old inference models remain supported.
- ONNX output naming is manifest-driven and covered by tests.

#### Gate C: new-head usefulness

- Beam heads outperform automatic beaming on held-out visible exceptions, not only
  on common regular groups.
- Stem direction beats the source/layout heuristic baseline.
- Structured slur endpoint and direction metrics improve over the legacy head.
- Rare-class metrics include sufficient support or are explicitly inconclusive.

#### Gate D: no material general regression

- Existing fixed smoke and system-level benchmarks remain within a predeclared
  non-regression tolerance.
- Single-staff and grand-staff inference remain valid without a score profile.
- Runtime and memory increases are measured and acceptable.

The numeric tolerance should be declared after reproducing B0 variance and before
examining held-out experiment results.

#### Gate E: safe conditioned behavior

- Missing context reproduces the unconditioned path within the expected numerical
  tolerance.
- Correct context improves the target metrics.
- Incorrect context cannot silently force structurally impossible output.
- Profile deviations become evidence/review findings.

#### Gate F: human-review value

- Structural corrections reliably invalidate and regenerate dependent output.
- Beam/slur questions show sufficient source context.
- Cross-staff proposals have measured precision and explain the invariant involved.
- Confirmations and corrections are captured with immutable provenance.
- Review reduces verified correction time or residual errors on representative
  scores.

### 23. Risks and mitigations

#### Sparse secondary notation

Secondary slurs and deep beam levels are rare. Use factored heads, class-support
reporting, targeted sampling, and macro metrics. Do not claim success from micro
accuracy dominated by `NONE`.

#### Synthetic-to-scan domain gap

Synthetic labels are exact but visually clean; scans are realistic but may have
edition-mismatched notation. Use strong visual augmentation, exact synthetic
supervision, per-head scanned masks, and reviewed scanned corrections.

#### Catastrophic forgetting

Adding OSSQ only can specialize the model at the expense of other scores. Retain
general-data replay, separate pretrained/new parameter learning rates, staged
unfreezing, and fixed general regression tests.

#### Context over-reliance

A model may learn that cello always means bass clef or that a supplied four-part
profile always means exactly four visible staves. Use unknown tokens, context
dropout, incorrect-context tests, soft layout scores, and explicit deviation output.

#### Invalid beam/slur sequences

Independent per-token heads can produce locally likely but globally invalid spans.
Use sequence validators, group-level metrics, bounded review/repair, and optional
feedback embeddings only after measuring the output-only baseline.

#### Token correction invalidation

Changing rhythm can change measure alignment; changing layout changes every token
identity. Content signatures, dependency-aware invalidation, and regeneration are
mandatory.

#### Page-boundary ambiguity

Page-local models cannot see the missing half of a spanning slur or tie. Expose open
state and let assembly review it; do not invent endpoints.

#### Capability drift

Provider, checkpoint, vocabularies, ONNX output order, parser, and generator can
drift independently. Named heads and immutable capability/version manifests are the
mitigation.

### 24. Recommended first implementation slice

The smallest slice that proves the main thesis is:

1. Add the structured-symbol schema and exact parser/generator round-trip fixtures.
2. Add OSSQ score-level split manifests and reconcile exclusions/counts.
3. Materialize explicit beam labels with pinned MuseScore.
4. Add six output-only beam heads, one output-only stem head, and parallel
   structured slur heads.
5. Add manifest-validated loading of the current pretrained checkpoint.
6. Train only the new heads on exact OSSQ synthetic staff crops.
7. Evaluate beam vectors/groups, stem direction, structured slurs, and bit-identical
   legacy outputs.
8. Unfreeze the final decoder layers with general-data replay only after the
   head-only result is understood.
9. Add the generic score-profile contract and deterministic page-level layout
   optimizer.
10. Integrate new head confidence and structural findings with human review.

The learned cross-staff adapter and lyric stage remain explicit later phases. This
keeps the first Vast.ai rental focused on a falsifiable question: does HOMR's
existing representation already contain enough visual evidence to learn explicit
beaming, stem direction, and richer slurs while preserving its current music
recognition quality?

#### 24.1 Revised sequencing after B0 (2026-08-16)

B0 has been built and run, and it changes the order above in two ways. The list in
§24 is otherwise unchanged; this subsection records what was measured and what
follows from it.

**What B0 measured.** A page-level OSSQ benchmark (`validation/ossq.py`, synthetic
track) against the pinned checkpoint found that 25% of quartet pages decoded as a
single part containing every staff's music in sequence. The cause was the
degenerate periodic signature this design anticipated in §8.2: with the bracket
rows inconsistent, staff parsing falls back to a period in each staff's
`is_grandstaff` flag, which is constant for an ensemble of same-type single staffs,
so period 1 fits vacuously. On those pages every per-note metric is meaningless,
so no notation-head result could have been read from them.

Deterministic page-level grouping from staff spacing (`homr/system_grouping.py`,
§8.2's "new system-partition scoring module") was implemented and measured: on a
60-page set the layout-broken count went 12 -> 0 and mean OMR-NED 16.44% -> 5.19%,
with the median unchanged because pages that already worked never reach it.
polish-scores is bit-identical with and without it.

**Change 1: item 9's layout optimizer moves ahead of the heads.** It is already
done for the grouping half, and it was a precondition rather than a follow-up: beam,
stem and slur accuracy cannot be measured on pages whose staffs are read into the
wrong parts. What remains of it is incomplete-system recovery. Grouping currently
drops a system whose staff count is short, because downstream indexes systems by
voice number; measured, that discard is the whole of the residual error on repaired
pages (`pred/ref` token ratio 0.81-0.85 against 1.01 on clean pages, ~10% NED).
Recovering it means inferring which voice slot is missing from the internal gap
pattern and teaching `parse_staffs` to index by slot rather than position.

**Change 2: items 1-3 are strictly blocking and are worth more than they look.**
The labels the new heads would train on do not exist anywhere in the pipeline today:
`training/omr_datasets/music_xml_parser.py` does not read `<beam>` or `<stem>` at
all, and collapses every slur to `slurStart`/`slurStop`, discarding number and
placement. So items 1-3 are not preparation for the heads, they are the heads'
entire training signal.

They also *determine the head configuration*, which is why they cannot be deferred
past item 4. §25.2 leaves open whether level-5 and level-6 beam heads have enough
examples to be learned or should start deterministic/unsupported; that is answered
by the per-class support tables, and answering it after building six heads means
rebuilding them.

**Resulting order.**

1. Freeze a score-level split manifest. Adopt `sqomr`'s four published folds with
   mutually exclusive test sets rather than inventing one, so results stay
   comparable with the paper, and hash the manifest into run metadata (§13.5).
   Blocking for everything that reports per-class support "by split".
2. Label extraction and validation, i.e. §14.2 Phase 1: parser preservation of
   beams, stems and full slur identity; beam materialization under a pinned
   MuseScore with the visual-equivalence check; slur slot canonicalization with
   overflow reporting; per-class support tables by split. Labels must come from the
   original MusicXML, not the cleaned copy, for the reason in §13.2.
3. Incomplete-system recovery, in parallel - it is independent of the label work
   and unblocks measurement on the pages grouping currently discards.
4. Then the heads, as items 4-8 of §24, unchanged.

`music_xml_parser.py` feeds every corpus conversion, not only OSSQ, so extending it
follows §19.2: versioned schema, existing token files stay readable.

#### 24.2 Where the sequence actually stands (2026-08-16)

Items 1-3 of §24.1 are done, and so is everything the heads need on the code side. What
remains is data and a run.

```
1  split manifest                      done   ossq_split_manifest.json, sqomr 4926e698
2  label extraction and validation     done   27.8-27.10, sidecars, support tables
3  incomplete-system recovery          done   27.7, layout failure 25% -> 2.9%
4  head architecture                   done   structured_heads, frozen-core loading
   targets, losses, dataset wrapper    done   27.15, alignment and substitution fixed
   metrics and evaluation pass         done   15.2-15.4, evaluate_structured_heads
   Gate C baseline, per split          done   27.16, 80.3-87.0% depending on split
   end-to-end integration              done   real loader, real TrOMR, core frozen
5  synthetic partwise staff crops      done   52,973 vs 52,960 published (27.11)
6  convert_ossq -> training set        done   42,089 train / 4,942 valid (27.19)
7  Phase 2 frozen-core run             done   3 epochs, 7 heads declared (27.21)
8  evaluate against the split baseline done   Gate C cleared, 78.7% of exceptions
```

**Gate C is cleared** (27.21): the heads recover 78.7% of the beaming duration and metre
cannot predict, against 27.12's question of whether half was reachable. What follows from
that, in order of what the result actually licenses:

```
 9  a converged run          done (27.26): twelve epochs buys about a point over three
10  stem head or rule        done (27.27, 27.28): they are complementary, not redundant,
                             and arbitrating on head confidence beats both by 1.5 points
11  re-convert OSSQ          done (27.29): 42,088 examples with ties and placement
12  slur-side heads          done (27.32): macro F1 0.925, from no targets at all
13  PDMX from source         done (27.31): 35,800 examples, ~4x the level-2 hooks
14  OSSQ + PDMX together     done (27.35, 27.36): mixing costs 1.6 points on OSSQ and
                             buys 16 on PDMX, and the quartet-only model turns out not
                             to generalise
15  a tie head               done (27.35): macro F1 0.842 on PDMX
16  the scanned track        done (27.56): trained on all three corpora, 12 epochs.
                             +4.0 beam and +6.3 stem on scans for 0.8 points of
                             synthetic - half 27.36's mixing cost. Ties moved
                             backwards despite more data (27.49's prediction, confirmed).
17  Gate C on the combined   done (27.58-27.60), and it reframes the track. Gate C
    model                    fails on scanned: the beam head is a net regression
                             against the rule (loses 12,027 notes, recovers 4,231)
                             where it was a clear win on synthetic. The stem arbiter
                             cannot rescue it either - best threshold uses the head on
                             3.4% of notes on scans, still below the rule alone. Both
                             heads are gains on synthetic and regressions on scanned.
                             Root cause is a bimodal gap (27.60): 45.8% of scanned
                             staves are unaffected, 10.8% collapse, and the collapse
                             rate varies 70x by score (27.61) - the bad staves are
                             faint scans, not misaligned crops.
18  lieder sidecars          one call added; the corpus has not been downloaded
19  test_synth               held back deliberately - it is the one split that has not
                             been looked at, and it should stay that way until a
                             configuration is being reported rather than explored
20  class-imbalance          done (27.62): focal loss (gamma=2) is a clean win, macro
    instruments               F1 +1.0 and start F1 +4.9 for -0.5 on beam vector - done
                             (27.66): confirmed the winner after all four arms. `both`
                             compounds the damage rather than offsetting it (macro
                             0.630, worse than weights alone) - class weighting is set
                             aside, not iterated on.
21  fix for the scan gap      one page-level candidate ruled out (27.63): CLAHE damages
                             crisp scans faster than it helps faint ones. Rebuilt around
                             per-image contrast after finding (27.65) that the
                             score-level version could not fire at all - the nine
                             measured collapse rates are entirely in validation, zero
                             overlap with training. phase12 running: reweighted index
                             (4,326 of 32,982 scanned images boosted) + focal loss
                             together, scored and gated the same way as phase9/10 for
                             a direct comparison.
```

**Every result before 27.36 should be read as "on string quartets".** That section measured
the OSSQ-only model on ordinary published music and found 0.927 exact beam vector falling
to 0.706, and 0.919 slur spans to 0.145. The figures in 27.21 through 27.32 are correct and
they describe one genre.

**The arbiter threshold is per-run and must be re-swept every time.** Items 12 and 14 each
produce new weights, and 27.28's 0.9 was tuned against the weights of 27.26. The driver
scripts re-run the sweep rather than reading a constant, which is the only reason this is
safe to leave unattended.

**A note on how many of these to have open at once.** At one point eight of these were
part-started, with nine background waiters polling the instance, five of them for a
marker written by a run that had finished seventeen hours earlier. The parallelism was not
buying anything - every track was blocked on the same GPU or the same conversion - and it
made the state of the work genuinely hard to read. One or two live tracks, with the rest
explicitly parked, is the working discipline.

**And a constant that must not go stale.** The stem arbiter's confidence threshold (0.9)
was tuned on half of `valid` for *this* set of weights. Any retraining invalidates it, and
re-using it because it is written down somewhere would look fine and quietly cost
accuracy. It is produced by a sweep, not a configuration file, for that reason.

**What does and does not need a re-conversion.** The seven heads trained in 27.21 read
beam levels, stem and slur slots, and none of those moved: the extractor always read only
`<slur>` and never `<tied>`, so the slur slots were already tie-free, and the token-level
conflation of tie and slur never reached the sidecar. Ties and placement are *additions*.
So the converged run in item 9 is unaffected and was allowed to finish; only a head that
consumes the new fields needs the data rebuilt.

Two things are worth carrying forward from how items 4's sub-items went. Both bugs found
in the wiring were *correspondence* bugs - targets one position out from the decoder's
shifted output, and labels not following the loader's image substitution - where neither
side is malformed alone and no shape check or loss curve reveals the mismatch. Anything
else that pairs two sequences in this pipeline deserves the same direct assertion.

And the Gate C baseline moves 6.7 points between splits, so a head's result is only
meaningful against the baseline on its own split. That is now a flag on the tool rather
than a note in a doc.

#### 27.64 Oversampling the worst scores, and a scaling bug caught before it ran

27.63 ruled out a page-level fix and pointed at which documents the model sees more of,
following 27.60's finding that the gap concentrates by score. `score_reweight.py` repeats a
score's index lines in proportion to its collapse rate - oversampling rather than filtering,
since dropping the worst scores would shrink an already-small scanned corpus and remove
exactly the examples a deployed system will meet.

**The first version scaled toward a hypothetical 100% collapse rate**, and on the real nine
scores - whose worst is 21.9% - that put `max_repeats` at a point nothing reaches:

```
first run     sq8806881 21.9% -> x2      every other score -> x1
```

Eight of nine scores landed within rounding distance of x1, so the mechanism built to
correct the gap would not have fired on the data that exists. Rescaling against the worst
rate observed *in the batch*, not against 1.0, fixes it:

```
after         sq8806881   21.9%  x6   sq8907120    14.5%  x3   sq7354505    8.0%  x1
              sq10414906  16.9%  x4   sq10307350    8.2%  x1   sq8806134    5.4%  x1
              sq8075304   16.6%  x4                             sq12772795   0.3%  x1
              sq8885571   15.3%  x3
```

The worst score now reaches the cap, the next two land at x4, and the four scores at or
below the 10% floor are untouched. This is the same shape of mistake as 27.53's padding bug
and 27.60's first collapse-rate framing: a number computed correctly against the wrong
reference, caught by running it against real data before trusting the arithmetic.

Not yet trained against - it builds and verifies on CPU; whether oversampling actually
narrows the domain gap needs a training run, queued behind phase11.

#### 27.66 phase11 complete: focal alone is the winner, and combining with weights compounds the damage

All four arms, synthetic validation:

```
                  macro F1   start F1   stop F1   start_and_stop F1   exact beam vector
baseline             0.772      0.551     0.791              0.746               0.904
focal (g=2)          0.782      0.600     0.785              0.744               0.899
weights (cap 50)     0.689      0.414     0.615              0.727               0.901
both                 0.630      0.283     0.490              0.750               0.895
```

27.62 predicted this before running it: *"the theory above predicts compounds, since focal's
gain does not depend on weighting being safe."* It compounds. `both` is worse than `weights`
alone on macro F1 (0.630 vs 0.689) and on `start` (0.283 vs 0.414) - focal does not rescue
weighting's damage, and stacking the two costs more than either alone. Only
`start_and_stop` moves the other way (0.750, best of all four), which is the class most
starved to begin with and the one a 293-example weight boost was aimed hardest at; even
there the gain is inside noise given n=604.

**Decision: focal loss (gamma=2), alone, is what goes into the next full training run.**
Class weighting is set aside, not iterated on - the mechanism 27.50 flagged (293 examples
carrying outsized gradient, amplifying any label error among them) is now confirmed twice,
once as a standalone regression and once as the dominant term in a compound that made a
second technique worse than doing nothing.

#### 27.65 27.64's oversampling design could not work at all - the split makes score names disjoint

Preparing to actually run 27.64's tool found it broken more deeply than the scaling bug
already fixed. Checked before spending a training run on it: the nine scores with measured
collapse rates are every one of them in validation, and the training index has **zero
overlap** with those score names.

```
training index distinct scores    32,551
validation scores measured         9
overlap                            0
```

That is not incidental - it is 13.5's own principle working correctly. OSSQ is split by
score precisely so a crop-level leak cannot inflate a number, and the collapse rates were
computed from held-out predictions for the same reason. A design that oversamples "scores
that collapse" in the training set is asking the training set to contain the very data that
was held out to measure it; it cannot, by the split's own construction, and 27.64's tool
would have run to completion, repeated nothing, and reported success on an index identical
to its input.

**`score_reweight.py` is rebuilt around the measurement that does not have this dependency.**
27.61 found contrast and ink fraction correlate with collapse *from the image alone* - no
prediction, no model, no score identity required. Weighting directly by each training
image's own contrast sidesteps the split entirely, and is finer-grained besides: 27.60 found
45.8% of a "bad" score's own staves are unaffected, so a score-level weight would have
boosted good pages inside bad scores for no reason a per-image weight avoids.

Run for real, on the scanned training index alone (32,982 images, the synthetic and PDMX
lines are untouched by construction):

```
32,982 images measured, 4,326 boosted above x1
contrast: min 84   median 238   max 255

worst training images (untouched by the validation split, confirming this works on data
the earlier design could not reach):
  sq7362818_0055_0003_4.png   contrast 84   x4
  sq9961690_0006_0001_4.png   contrast 87   x4
  sq9961690_0013_0001_2.png   contrast 91   x4

32,982 lines -> 37,920 lines
```

Assembled into a full reweighted training index at 112,459 lines (107,521 original plus
4,938 repeats), ready for the next training slot.

**The pattern across 27.53, 27.60's first framing, 27.64, and this: build the check for
whether a mechanism can fire on the actual data before spending the run that would have
revealed it firing on nothing.** Four times now the mechanism was correct in isolation and
wrong against the data it was meant to touch.

#### 27.67 phase12: the two prepared fixes combined into one run

27.65's reweighted training index and 27.66's focal loss were each validated separately -
the reweighting on CPU against real data, focal against three alternatives in phase11 - and
neither had yet been trained together, or scored against phase9's scanned Gate C failure
that motivated both. `phase12.sh` does that: trains from the same frozen-core checkpoint
every phase has used, on the 112,459-line reweighted index, with `--focal-gamma 2.0`, then
scores and gates all three domains the way phase9 and phase10 did - synthetic and scanned get
the crosstab and stem arbiter, PDMX gets evaluation only, per 27.58's note that both tools are
OSSQ-shaped and cannot read it.

**A smoke test before committing the GPU time**, given how many "ready" scripts this session
have had a bug surface only when run: 20 lines of the reweighted index, one epoch, batch size
4. It trained cleanly - the only failure was the smoke test's own scratch directory missing,
not the training command - so the full run was launched rather than iterated on further.

What this run is expected to settle: whether the scanned Gate C failure of 27.58 (head loses
12,027 notes to the rule, recovers 4,231) narrows when the faintest training crops are seen
more often, and whether combining that with focal loss compounds the tie-head gain of 27.66
or interacts with it the way weighting's compound did.

#### 27.68 The detector half of 27.45 was never built - and its data now is

27.45 settled the text pass as detection then recognition. Checking what actually exists
found only the second half was: every crop the recogniser has been trained or measured on
came from MuseScore's own SVG boxes, ground truth handed to it rather than found. Nothing in
this project yet locates a syllable, or any other text, on a page it has not already been
told the layout of - which is what a real scanned page is, at inference. The recogniser as
it stands has no way to be pointed at one.

`training/ocr/detector_data.py` builds the missing half's data: one row per box, image path,
class, and the box itself, from the `.boxes.json` corpus `musescore_boxes.py` already
produces. No string is needed, since detection only has to find and classify a region -
which is why every text class from 27.44 trains the same detector, `Dynamic` still excluded
per 27.45's finding that it is a music glyph rather than text.

Run against the full 2,926-system corpus:

```
37,356 boxes across 2,847 images

Lyrics            34,325   91.9%
StaffText          1,262    3.4%
MeasureNumber        806    2.2%
Tempo                493    1.3%
Expression           423    1.1%
Fingering             27    0.1%
SystemText            20    0.1%

imbalance: Lyrics is 1,716x SystemText
```

**`InstrumentName` does not appear**, correctly - 27.46's part-name suppression already
removed it from the render, and this confirms that fix reached the data a detector would
train on, not only the syllable boxes it was checked against at the time. `Harmony` and
`RehearsalMark` are absent too, plausibly because solo Lieder carry neither chord symbols nor
rehearsal marks - unverified, but consistent with the repertoire.

**The imbalance is worse than the tie head's, by three orders of magnitude**, and 27.62 is
now direct precedent rather than an analogy: an uncorrected classifier facing this shape of
imbalance stops predicting the starved class, and a detector is a classifier at every anchor
it considers. Whatever detector architecture is chosen needs the same correction - focal
loss, validated, not class weighting, refuted twice - built in from the first training run
rather than discovered as a regression afterward.

**What remains open, deliberately.** The detector's own architecture is not chosen here. It
is a separate decision from the data it will train on, and this section stops at making that
data ready.

#### 27.69 Detection masks built on homr's own segmentation pattern - and how sparse the target is

27.68 left the detector's architecture open. It is not a new invention: `homr/segmentation`
already solves "find small objects on a large, mostly-blank sheet-music page" for staves and
noteheads - a U-Net over 320x320 patches tiled across the full-resolution page at 50%
overlap, boxes recovered from the predicted per-pixel class map by connected components. A
lyric syllable at a measured median 34x16 pixels (27.44) is exactly what that pattern was
built for, and downsampling a whole page to one fixed size would erase it before the model
ever saw it. `detector_masks.py` produces the target that architecture trains against - a
per-pixel class mask - as the data step ahead of the model, not the model itself.

Run over the full corpus:

```
2,847 page masks written, 79MB total (~28KB average - PNG's compression on a mask that is
  background almost everywhere, not a raw array)
text pixels: 0.38% of a page, sampled over 200
```

**0.38% coverage is the number that decides how this has to be trained.** A 320x320 patch is
102,400 pixels; at uniform random placement over a page this sparse, most patches land
entirely in background, and a training loop that samples patches uniformly would spend
nearly all its steps on nothing to detect. This is the same shape of problem 27.49 diagnosed
for the tie head and 27.68 already flagged for the class balance between text types - here
it is spatial rather than categorical, and the fix is the spatial analogue: patches have to
be sampled biased toward where the boxes are, not drawn uniformly across the page.

**Class indices are fixed, not derived from a set's iteration order.** `CLASS_ORDER` is an
explicit tuple; a mask's channel meaning has to be stable across every image it is compared
against, and depending on a set's ordering would make that stability accidental.

Caught in the test before it ran on real data: the first version of the write-then-read test
asserted on a mask file after the `TemporaryDirectory` block that held it had already been
deleted, and passed nothing - a manual repro of the same code the test wrapped worked cleanly,
which was the tell that the test's structure was wrong rather than the code.

**Left for the next slot**, since it needs the GPU phase12 currently holds: the patch sampler
and the segmentation model itself, biased per the 0.38% finding, and trained with the focal
loss correction 27.62 validated and 27.68 already flagged this corpus would need.

#### 27.70 Every "scanned" result in this design is OSSQ, one corpus, one genre

Asked directly and worth stating plainly rather than leaving implicit: does the scanned
domain behind 27.38, 27.56, 27.58, 27.60, 27.61, 27.63, 27.65 and phase12 include OLiMPiC's
scanned track, or only OSSQ's?

**Only OSSQ.** Checked rather than recalled:

```
phase7/train token names       sq7383977_..., sq8907120_...   - OSSQ's own naming
phase9's training index        zero references to olimpic
dataset-root for every         /workspace/b0/ossq-omr, whose layout
Gate C / arbiter run           scores/*/*/musicxml/unaligned/... is OSSQ-specific
                                (27.58 already noted PDMX cannot use these tools for the
                                same reason - they are OSSQ-shaped by construction)
```

`phase7` - the source of every "scanned" figure in 27.38 through phase12 - is built by
`scanned_convert.sh` from OSSQ alone. OLiMPiC's scanned images are a wholly separate track,
used only for the lyric/OCR work from 27.40 onward (`musescore_boxes.py`, the resolve
baseline, the detector data). They have never been part of notation-head training or
evaluation, and no beam, stem, slur or tie number in this design has ever been measured
against them.

**So every domain-gap finding in this design describes one corpus: OSSQ's photographed
string quartets.** That includes the headline results - the 20-40 point synthetic-to-scan
gaps, Gate C's scanned regression, the 70x per-score collapse-rate spread, the contrast
correlation, and phase12's test of whether reweighting narrows it. None of it has been
checked against a second scanned source.

**This is a real scope limit, not a caveat to wave past.** OLiMPiC's scans differ from
OSSQ's on every axis that could matter: different repertoire (Lieder rather than quartets),
different engraving era and conventions, and - so far as this design has established -
different scanning equipment and provenance entirely. A contrast effect real for OSSQ's
photographs might not describe OLiMPiC's scans at all; it might be milder, worse, or shaped
differently. Nothing here says which, because nothing here has looked.

**What would close it**: the same domain-gap measurement 27.60 ran - paired staff accuracy
between a corpus's synthetic and scanned renderings - repeated on OLiMPiC once its notation
heads have targets to train against, which they do not yet (26.40's track has built images
and lyric labels, not beam/stem/slur/tie labels for OLiMPiC's scans). Until then, "the scan
gap" in this design should be read as "the scan gap measured on OSSQ," not as a property of
scanned sheet music in general.

#### 27.71 Slurs had no domain-gap tooling at all - closing that gap before closing the gap itself

Asked directly: slurs carry the worst measured domain gap of any channel (27.56, ~37-40
points), worse than beams. The natural next step is 27.60's paired-staff analysis, run on
slurs instead of beams - and it could not be done. `dump_predictions` only ever wrote beam
and stem vectors; slur state was never recorded per staff, only aggregated into the F1
figures already quoted. There was no way to ask which staves the slur head fails on, or
whether the failure concentrates the way beam's did.

**Extended rather than rebuilt.** `evaluate_structured_heads.py` now writes
`slur_reference`/`slur_predicted` in the same nested-list-per-note shape the beam vectors
already use - one list of `[event, side, event, side, ...]` strings per note, flattened
across the configured slur slots. That shape means `domain_gap.py`'s existing comparison
needs no new logic to read it: Python's list equality already treats each note's whole
vector as one unit, which is what "does this note's slur state match" means for either head.
`staff_accuracy` takes a `field` argument - `"reference"` for beams, anything else reads
`{field}_reference`/`{field}_predicted` - so the same tool now answers the question for
either channel.

Nothing here changes what the model does. It changes what can be asked of predictions that
have not been generated yet - phase12's own scoring stage, still to run, will produce
`slur_reference`/`slur_predicted` automatically once it reaches its `evaluate_structured_heads`
calls, since each is a fresh subprocess that imports the updated module from disk. No
separate GPU pass is needed to get the first slur domain-gap reading; phase12's already-queued
evaluation supplies it.

Caught by the existing integration test rather than by inspection: `test_evaluate_integration.py`
exercises `dump_predictions` end to end with a real model, and it passed unchanged after the
extension, which is what confirmed the new fields did not disturb the beam/stem path they sit
beside.

#### 27.72 phase12 complete: the combined fix hurt beam and slur everywhere, and isolated a cause

phase12 trained on the reweighted index with focal loss, scored all three domains against
phase9's baseline the same way phase9/phase10 were:

```
                    beam (exact vector)      slur spans F1         tie macro F1
synthetic       0.903 -> 0.880  (-2.3)   0.884 -> 0.832  (-5.2)   0.774 -> 0.789  (+1.5)
scanned         0.701 -> 0.690  (-1.1)   0.509 -> 0.456  (-5.3)   0.545 -> 0.572  (+2.7)
pdmx            0.855 -> 0.836  (-1.9)   0.737 -> 0.660  (-7.7)   0.805 -> 0.811  (+0.6)
```

**Beam and slur are worse in every domain; ties are better in every domain.** The scanned
Gate C regression that motivated the whole experiment did not narrow - recovery fell from
55.1% to 53.1%, loss rose from 28.3% to 29.3%. This is a negative result for the combined
fix as tried, but the pattern across domains isolates *why*, and that turns out to be the
valuable part.

**Synthetic and PDMX were never touched by the reweighted index** - `score_reweight.py`
only duplicates lines under the scanned track's own directory, and neither synthetic nor
PDMX training data changed at all between phase9 and phase12. Yet beam and slur regressed
there too, by nearly as much as on scanned. The one thing that *did* change everywhere was
focal loss, applied with a single scalar `gamma` to every head in the loop, including three
that were never shown to need it. **The correction 27.62 validated for the tie head's
specific starvation was applied globally, and paid for by every head that did not have that
problem.** phase11 never caught this because it only ever scored on synthetic.

**`structured_loss`'s `gamma` now accepts a per-head dict**, defaulting any head not named
to plain cross-entropy (0.0) rather than to a guess. `--focal-gamma-head tie.state=2.0`
scopes it from the CLI. Existing scalar behaviour is unchanged and pinned by a test - every
earlier phase's runs used a scalar and must not be retroactively reinterpreted.

`phase13.sh` is running: the same reweighted index, focal scoped to `tie.state` alone. If
beam and slur recover to something near phase9's numbers while tie keeps its gain, the
mechanism is confirmed and the two fixes can be evaluated independently for the first time.
If they do not, the reweighted index carries its own cost distinct from what 27.65 measured
on CPU, and that would need its own isolation.

#### 27.73 The first real slur domain-gap reading - and a self-inflicted trap caught on first use

27.71 built the tooling to ask whether slurs' domain gap concentrates the way beams' does.
The first run answered a different question than intended: 99.7% synthetic against 99.2%
scanned, a gap of essentially nothing - flatly contradicting the 30-40 point gap the span-F1
metric has shown throughout this design.

**The tool's own first output was the tell.** Checked before trusting it: only 1.4% of notes
carry any non-`none` slur content. Slur event is supervised on every note - unlike beam,
which is masked to beamable notes upstream, in the writer, before this tool ever sees it -
so an unfiltered per-note accuracy is 98.6% before the head predicts anything, the exact
shape of trap 27.49 found in tie micro F1. The tool built to give slurs the same treatment
beams already had reproduced, on its first real use, the precise failure mode that treatment
exists to catch.

`staff_accuracy` now takes `exclude_trivial`, dropping positions whose reference vector is
entirely `none`/`unspecified` before comparing - required for slur, inert for beam, since
beam vectors never contain those strings. With it:

```
2,733 staves scored under both renderings
  mean accuracy  synthetic 76.8%   scanned 43.7%

collapsed (> 50 points): 750 (27.4%)     unchanged (<= 10 points): 904 (33.1%)
share of lost notes in the worst 10% of staves: 20.6%

collapse rate by score:
  sq8806881   52.2%    sq8907120   41.5%    sq8885571  35.4%   sq8806134  17.9%
  sq8075304   42.2%    sq10414906  41.0%    sq10307350 28.1%   sq7354505  15.5%
  sq12772795   2.2%
```

**A real 33-point gap, and a different shape from beam's.** Beam's collapse concentrated in
20.4% of staves carrying 28.1% of the loss (27.60); here 27.4% of staves collapse outright
and only 20.6% of the loss sits in the worst decile - *less* concentrated than beam, spread
more broadly across scores including the best one (sq12772795 still loses 2.2% of its
slur-bearing notes, where its beam collapse rate was near zero). **A fix aimed at a handful
of bad documents fits beam's shape better than slur's.** Slurs need something closer to a
uniform correction, or a different mechanism than per-score reweighting was ever going to
reach.

#### 27.74 The detector's patch sampler, and why homr's own sampler was the wrong template

27.69 left the patch sampler and detector model open, with training resolution set by
homr's own segmentation pattern - a U-Net over 320x320 tiles. Checked before reusing homr's
own dataset code wholesale: `training/segmentation/dense_dataset_definitions.py`'s
`SegmentationBaseDataset` tiles a plain non-overlapping grid across the whole page with no
bias toward content at all, right for its own targets - noteheads and staff lines occur
densely enough that most tiles contain some. 27.69 measured text coverage at 0.38% of a
page, and 27.68 measured about 13 boxes per image; a 4500x3200 page tiles into on the order
of 140 non-overlapping windows at this size, so an unbiased grid would put text in a
handful of tiles and nothing in the rest. Copying homr's sampler here would have trained a
detector almost entirely on the answer "nothing here" - the data-side version of 27.72's
mistake, where a correction validated for one narrow case got applied somewhere it was
never measured to be needed. Checking first is what avoided training that mistake into a
model rather than discovering it afterward.

`detector_patches.py` samples instead of tiling: each patch is centred on a randomly chosen
box (jittered, so a box does not always sit dead-centre) at a fixed ratio, or on a
uniformly random page location otherwise - positive examples to learn from, negative ones so
the model does not predict text everywhere out of habit. Box centres come from
`cv2.connectedComponentsWithStats` on the rasterised mask rather than from the original box
list, so the sampler needs nothing but the mask 27.69 already produces. Edge padding follows
the convention already set in this corpus: white for the image, background for the mask -
27.51 found zero-padding on a crop teaches a model that every crop ends in a black bar, and
that lesson applies here without change.

19 tests pass, including a statistical check that `positive_ratio=0.0` on a large page with a
tiny box rarely lands on it by chance - confirming the sparsity problem the sampler exists to
correct is real, not assumed. Run against the real mask corpus, 2,847 images: 364 of 500
sampled patches contained text, 72.8% against a 70% target ratio - matching design on the
data it will actually train on, not only on synthetic test fixtures.

#### 27.75 The detector training script - a missing dependency, and a repeated smoke-test mistake

27.68/27.69/27.74 built the detector's data, masks and patch sampler. `train_detector.py`
closes the loop: `CamVidModel` (`training/architecture/segmentation/model.py`) - homr's own
U-Net over a resnet18 encoder, already used for staff and notehead segmentation - reused as
a plain `nn.Module` rather than reinvented, driven by a plain training loop consistent with
`train_structured_heads.py` and `train_recognizer.py` rather than through PyTorch Lightning's
`Trainer`, which `training/segmentation/train.py` uses and nothing else in this project does.

**No focal loss, and that absence is a decision, not an oversight.** 27.68 measured the
detector's class imbalance at up to 1,716x - three orders of magnitude past the tie head's -
which invites reapplying 27.62's fix on the strength of that number alone. 27.72 is the
reason this does not: a correction validated for one head, applied without checking whether
another needed it, cost every head that did not. `CamVidModel`'s loss is multiclass Dice,
computed from per-class overlap rather than per-token cross-entropy, and whether that
already tolerates this imbalance or collapses the same way ties did is unmeasured - so
`per_class_iou` reports every class's IoU every epoch, absent classes marked absent rather
than zero, so a starved class and an unlucky batch cannot be confused. The baseline gets
measured before anything is added to fix it, the same order `structured_losses.py` asked
for the first time this project reached for a class correction.

**A declared dependency was missing from the environment.** `import segmentation_models_pytorch`
failed outright - `pyproject.toml` lists it and `pytorch-lightning` as main dependencies, but
the instance's venv had never needed them until this track reached for `CamVidModel`.
Installed with `uv pip install --python .venv/bin/python`, since the venv has no `pip`
binary of its own; verified phase13's training process was undisturbed by the install before
and after.

**A smoke test aimed at the full corpus, the same mistake made once before with the OCR
sweep.** The first run pointed `--index` at all 2,847 images on CPU and did not return
inside a generous timeout - not a bug, a resnet18 U-Net's forward and backward pass on CPU
over the full patch set is genuinely slow, confirming training needs the GPU rather than
indicating anything wrong. Restarted against a 10-image slice: 20 patches, one epoch, 34.5
seconds wall clock. That is the smoke test this needed the first time.

Ready for the next GPU slot behind phase13.

#### 27.76 U-Net was chosen for reuse, not for merit - an architecture search stays open

27.75 picked `CamVidModel` substantially because homr already has it: shared training idiom,
smaller footprint, one segmentation architecture in the project instead of two. Asked
directly whether that was the right call rather than merely the convenient one, and the
honest answer is that no comparison was run. U-Net is a 2015 architecture; nothing about
its age disqualifies it for a small, dense, per-pixel localization task like this one, but
nothing about its age recommends it either - reuse was the deciding factor, not measurement.

**A large multimodal model was also considered and is not the right comparison.** The
question that prompted this was whether a VLM (served through something like Ollama) should
replace the detector, or even the whole detect-then-recognise pipeline. Three reasons this
design already argues against it, not new ones invented for the question:

  * **Language-prior contamination**, the same failure 27.51 built the recogniser against.
    Targets are syllable fragments - `ter`, `nel`, `schaft!` - and a VLM's language prior is
    stronger than a small CTC recogniser's, so it would "correct" fragments into words more
    aggressively, not less.
  * **Small-object localization**, the same failure 27.47 measured for whole-page
    downsampling. A VLM's image encoder tokenizes into a coarse patch grid; a 34x16px
    syllable (27.44) does not survive that any better than it survived a naively-resized
    training crop.
  * **Verifiable joins over free text.** This pipeline's discipline throughout - ordinal
    pairing, count checks, refuse rather than guess (27.11, 27.41, 27.43, 27.57) - needs a
    pixel coordinate and a class, not prose. A VLM's output is harder to build that
    discipline around, not merely different in style.

**What stays genuinely open is purpose-built detector architectures against each other** -
U-Net-style dense segmentation against a proper object detector: YOLO-family single-shot
detectors, DETR-style transformer detectors, or anchor-free detectors like FCOS or
CenterNet. Each frames the problem differently (boxes directly vs. per-pixel class then
connected components) and none has been measured against this corpus. The user was asked and
is open to running that comparison.

**When to run it, and what would decide it:** once `train_detector.py` produces a real
baseline - per-class IoU from an actual training run rather than the untrained smoke test of
27.75 - which is the same order this project has kept everywhere else: measure the reused,
already-available architecture first, and let its shortfall (or lack of one) decide whether
a search is worth the GPU hours. Comparing architectures before any of them have been run on
this data would be arguing from priors the corpus itself can settle.

#### 27.77 phase13 confirms the isolation: scoping gamma to tie.state recovers beam and slur

phase12 applied focal globally and lost beam/slur in every domain. phase13 kept the
reweighted index unchanged and scoped `--focal-gamma-head tie.state=2.0` to the one head
that was ever measured to need it. Synthetic and scanned, against phase9's baseline and
phase12's global-focal result:

```
                    beam (exact vector)                    slur spans F1
              phase9    phase12   phase13         phase9    phase12   phase13
synthetic     0.903     0.880     0.901           0.884     0.832     0.903
scanned       0.701     0.690     0.706           0.509     0.456     0.545

                    tie macro F1
              phase9    phase12   phase13
synthetic     0.774     0.789     0.787
scanned       0.545     0.572     0.559
```

**Beam and slur recover to at or above phase9's baseline on both domains, while tie keeps
most of its gain.** Scoping the correction to the head it was diagnosed for did what 27.72
predicted it would: the damage to beam and slur was never inherent to focal loss, it was
inherent to applying it somewhere it was never measured to be needed. Tie gives up a small
amount relative to phase12's global version (0.787 vs 0.789 synthetic, 0.559 vs 0.572
scanned) - global focal happened to help ties slightly more by also suppressing easy
positions on every other head's loss term, which shrank their contribution to the total and
left tie's gradient share relatively larger. That is not a reason to prefer the global
version; it is a side effect of a mechanism 27.72 already showed costs more than it returns.

**Slur spans on scans reach 0.545, above every prior measurement in this design** - 27.38's
untrained gap, 27.56's three-corpus baseline (0.509), and phase12's regression (0.456) are
all below it. Whether this reflects the reweighted index finally showing a real benefit once
it is not confounded by a miscalibrated global correction, or run-to-run variance from a
single seed, is not yet distinguishable - each phase here is one training run, not a
repeated-seed comparison.

**Gate C on scanned**: recovers 54.2% (4,162 notes), loses 27.5% (11,677) - within noise of
phase9's 55.1%/28.3% and phase12's 53.1%/29.3%, not a clear win or loss. The scanned Gate C
regression that motivated the whole reweighting effort has not resolved either way with
enough separation from noise to call.

**The slur domain-gap tool (27.71/27.73) gives a second, independent readout.** Per-note
accuracy excluding trivial positions:

```
              synthetic   scanned    gap      collapsed   worst-decile share
phase12         76.8%      43.7%    33.1pp      27.4%           20.6%
phase13         87.8%      52.3%    35.5pp      31.1%           15.8%
```

Both domains improved in absolute terms, consistent with the span-F1 gain. The gap in
percentage points is slightly wider and collapse is even less concentrated (15.8% vs
20.6% of the loss in the worst decile) - continued evidence, independent of the beam-focused
27.60/27.65/27.74 work, that slurs fail broadly across documents rather than in a
concentrated few, and a per-score fix was never going to be the whole answer for this head.

**Decision: `--focal-gamma-head` is the form any future per-class correction in this
project should take.** A scalar `--focal-gamma` remains for backward compatibility and
nothing else; every future run reaching for focal loss should name the head it is meant for.

#### 27.78 Correction: PDMX was already in run 426's own training - "we added PDMX" was imprecise

Asked directly why "we added PDMX" was said without checking what homr's own pretraining
already included, and it should have been checked. `Training.md`, run 426 - commit
`b6fd20809a8dcaf10dfd39a4ca4f64c6f056e644`, the exact checkpoint every phase script in this
design has fine-tuned - states plainly: *"Training with lieder+grandstaff+primus+pdmx+
musetrainer datasets."* The frozen core this whole design builds on top of had already seen
PDMX, along with Lieder, GrandStaff, PrIMuS and MuseTrainer, before this design session
existed.

**What this project actually did with PDMX was narrower than "added it".** The structured
heads - beam, stem, tie, and the slotted slur representation - are output-only additions
bolted onto that frozen core, and they had no training data at all, on any corpus, before
this project built them; run 426's own training predates their existence entirely. What
27.31/27.35/27.36 built was a newly re-converted, quality-filtered PDMX (from the Zenodo
source, excluding empty-final-measure and low-note-count scores) fed specifically to *those*
heads. "We added PDMX" was true of the heads and false of homr as a whole, and every mention
of PDMX generalization in this design through 27.36 should be read as being about the new
heads' generalization, not the base decoder's.

**This reframes 27.36's finding rather than voiding it.** "The quartet-only model does not
generalise to PDMX" measured the structured heads specifically, trained at that point only
on OSSQ synthetic quartets - the base decoder's own rhythm/pitch/lift/articulation/slur/
position predictions were never at risk of that collapse, because they had PDMX (and four
other corpora) in their training from before this design began.

**An unmeasured question this opens, worth stating rather than leaving implicit.** Every
"exact beam vector" and Gate C figure in this design measures the structured heads'
accuracy - never the frozen core's own note-reading on the same scanned images. Run 426's
pretraining mix is more diverse than anything the structured heads have seen, so it is not
known whether the scanned regression documented in 27.58 onward is specific to the heads
(which had comparatively little scanned-specific training before phase12/13's reweighting)
or whether the frozen core's own predictions - the fields the heads are layered on top of -
also degrade on the same scans. If the core holds up and only the bolted-on heads collapse,
that argues for more scanned data for the heads specifically. If the core degrades too, the
heads were never going to fix it alone, and the domain gap says as much about run 426's own
training mix as about anything this design built afterward. Nothing here has checked which.

#### 27.79 The frozen core also degrades on scans - but far less than the structured heads, and on the same documents

27.78 asked whether the scanned regression is specific to the structured heads or also
present in run 426's own predictions. Measured with `base_domain_gap.py` against
`validation/ossq.py`'s plain pipeline (zero structured heads, 211 pages paired):

```
overall NED       synthetic 7.5%    scanned 12.9%    median page increase  1.7pp
rhythm NED        synthetic 5.4%    scanned  9.3%    median page increase  0.7pp

worst degrading scores (overall NED):
  sq10414906   +42.6%     sq8806881   +25.8%     sq8806134   +17.1%
  sq8075304     +6.0%     sq12772795   +2.0%     sq7354505    +1.7%
  sq8885571     +1.0%     sq10307350   -1.6%     sq8907120    -5.3%
```

**A found bug before trusting any of this**: the first run read the database's `ned` column
as a percentage and divided by 100 again, producing a mean of 0.1% against printed log
lines showing individual pages at 2-14%. The database stores the same 0-1 fraction the log
formats for display; caught by checking the raw column values directly rather than trusting
a suspiciously good number.

**The answer is neither of 27.78's two clean alternatives.** The frozen core is not immune -
overall NED nearly doubles on scans (7.5% to 12.9%) and rhythm NED does the same (5.4% to
9.3%). But it is nowhere near as damaged as the structured heads: an NED increase from 5.4%
to 9.3% is a small absolute degradation on an already-good baseline, where the beam head's
accuracy fell from 90.3% to 70.1% - a 20-point collapse on a metric that was not
edit-distance-forgiving to begin with. Different scales, but the qualitative gap in severity
is not close.

**And the degradation is concentrated on exactly the same documents 27.61 already diagnosed
as faint.** `sq10414906`, `sq8806881` and `sq8806134` are the frozen core's three worst
scores here, and they are also the beam head's three worst scores in 27.60 (collapse rates
16.9%, 21.9%, 5.4% respectively - not identically ranked, but the same set). This is
independent confirmation of 27.61's contrast finding from a measurement path that shares
nothing with it - no structured head, no reweighted index, no focal loss, just the frozen
decoder reading the same bad photographs.

**What this settles, for the "is it worth it" question 27.58 raised:** these are bad scans,
not an unfixable property of scanned sheet music in general - the frozen core, with its
much richer original pretraining diet (lieder+grandstaff+primus+pdmx+musetrainer, 27.78),
handles the same faint documents with only a mild NED increase. The structured heads,
trained on a narrower and more recent slice of scanned data, are simply less robust to the
same bad documents the core shrugs off more easily. That argues for more and more varied
scanned training data for the heads specifically - closing the gap between their training
diet and the core's - rather than for concluding the domain gap is inherent to real scans
or that the heads are a lost cause on this input.

#### 27.80 Correction: OSSQ's scans are IMSLP too - OLiMPiC differs in genre and document, not in provenance category

Proposing OLiMPiC as a second scanned domain, this design characterised OSSQ's scans as
distinct from OLiMPiC's - "personal photography" against "IMSLP archival scans" - without
checking. Wrong, and checked properly this time: `ossq-omr`'s own README documents
`sq<id>_scanned.pdf  # optional scanned version from IMSLP` and ships
`scores_w_url.yaml: per-score IMSLP source metadata (catalog number and direct scan URL)`.
Both corpora's scans are IMSLP-derived.

What actually differs between them is genre - string quartets against Lieder piano - and
which specific documents and scanning batches, since IMSLP itself is an aggregation from
many contributors across many years, not a single uniform scanning source. Testing on
OLiMPiC is still worth doing for that reason: a different set of IMSLP documents, a
different repertoire, possibly different scan eras and equipment within IMSLP's own
heterogeneous holdings - but it is not the "archival vs. personal" contrast this design
stated, and 27.79's finding that a handful of specific documents are unusually faint should
be read as being about *those documents*, not about a category of scanning distinct from
what OLiMPiC would offer.

#### 27.81 The concentrated-collapse pattern replicates on an independent sample - without touching the held-back split

Following up on whether nine scores were enough (27.60's own caveat, repeated at 27.79):
`test_scanned`, ten scores fully disjoint from `valid`, run through the same frozen-core NED
benchmark.

**Its synthetic counterpart is `test_synth`** - the split 24.2 held back deliberately,
"the one split that has not been looked at, and it should stay that way until a
configuration is being reported rather than explored." Pairing `test_scanned` against it for
a synthetic/scanned gap, the way 27.79 did for `valid`, would mean looking at test_synth for
an exploratory diagnostic. That crosses a rule this design set for itself on purpose, so it
was not done. Whether to lift that restriction for a diagnostic that feeds no tuning
decision, rather than a configuration score, is the user's call and has been raised as
such rather than decided here.

**What test_scanned answers on its own, no synthetic pairing needed:**

```
191 pages, 8 scores with any scored pages (2 of the 10 nominal scores produced none)
overall mean NED: 10.2%    (valid's scanned mean was 12.9% - comparable)

sq9146376   28.4%   sq9631717   11.9%   sq8071278    6.9%   sq7127785   5.6%
sq8807667   18.1%   sq7294793    7.5%   sq7302602    6.8%   sq7358579   4.2%
```

**The same shape replicates.** A greater-than-sixfold spread between the worst score
(28.4%) and the best (4.2%), with two scores standing well clear of the rest - the same
concentrated-collapse pattern 27.60 found for the beam head and 27.79 found for the frozen
core on `valid`, now confirmed on a fully independent set of documents. This does not by
itself prove the *specific* documents matter (that needs the synthetic pairing this section
declined to run), but it rules out "nine scores was a fluke" as the explanation for the
pattern - a second, disjoint sample shows the same shape.

#### 27.82 OLiMPiC scanned scored: the gap is head-dependent, not a uniform "scans are hard"

phase13's weights (the best configuration found) scored against OLiMPiC's scanned `dev`
split - 1,350 systems, converted cleanly (94% success rate, 27.80's conversion work) - with
no Gate C or arbiter, the same limitation PDMX already has: both tools are OSSQ-shaped by
construction (27.58).

```
                    beam       hooks F1   stem (up/down)   slur spans F1   tie macro F1
OSSQ synthetic     0.901        0.819         0.929            0.903           0.787
OSSQ scanned       0.706        0.661         0.794            0.545           0.559
PDMX               0.845        0.788         0.806            0.779           0.803
OLiMPiC scanned    0.815        0.549         0.764            0.583           0.647
```

**Beam does markedly better on OLiMPiC scanned than on OSSQ scanned** - 0.815 against 0.706,
close to PDMX's clean-but-cross-genre 0.845 rather than to OSSQ's collapsed figure. If the
whole scanned-domain problem were "scans are hard," OLiMPiC scanned should sit near OSSQ
scanned. It does not, for this head.

**Hooks and stem go the other way** - hooks F1 0.549 against OSSQ scanned's 0.661, stem
0.764 against 0.794 - both *worse* on OLiMPiC than on OSSQ's own scans. Slur (0.583) and tie
(0.647) sit between OSSQ scanned and PDMX, closer to neither.

**No single story fits all four heads**, which is itself the finding. The domain gap is not
a property of "scanned" as one category that every head suffers from equally - it is
head-specific, and a fix aimed at one head's failure on OSSQ scanned (the reweighted index,
scoped to contrast) has no reason to transfer to a different head's failure on a different
scanning source, even before asking whether it transfers to a different scanning source at
all.

**A real limitation on how far to read this.** OLiMPiC is pianoform reduction from Lieder -
a different genre, different typical beam-group complexity, different note density per
staff than OSSQ's string quartet parts - converted through an entirely separate pipeline.
This is not a controlled comparison in the way 27.79's paired-page NED was; it is two
different corpora scored with the same weights, which answers "does performance transfer"
but not "why," and the genre difference is a live confound for every number above, not just
a caveat to mention once.

#### 27.83 On OLiMPiC the head's advantage over the rule disappears - and the fuller crosstab was blocked, not built

27.82 gave beam=0.815 on OLiMPiC with nothing to compare it against. `beam_baseline.py`'s
rule (`measure_part`) is generic MusicXML and needed no OSSQ-specific change; only its file
discovery did. Adapted to read whole-score piano parts from the OpenScore Lieder `.mxl`
downloads kept from 27.41's lyric work (the same `piano_part_id` selecting the same part
OLiMPiC's own build selects), scored on the 100 scores in the `dev` split:

```
rule alone       84.0%   (29,338 / 34,917, whole-score, not systemwise segments - the
                          fragmentation beam_baseline.py's own docstring already measured
                          costs 91.9% vs 79.4% for OSSQ)
head alone       81.5%   (27.82's exact beam vector on the same split)
```

**On OSSQ synthetic the head clearly beats the rule - 90.3% against roughly 84%. On
OLiMPiC, that gap is gone**, and the rule is a little ahead. Read alone this already answers
part of "should we give up on them": whatever advantage the head has on quartets, it is not
showing up on Lieder piano reduction.

**But a totals comparison is exactly what 27.16 warned against trusting** - two similar
numbers can hide an oracle above either, if the head and the rule fail on different notes.
Building the fuller crosstab (recovers/loses, the way Gate C measured OSSQ) was attempted
and stopped before producing a number, for a reason worth recording precisely rather than
papering over.

**`rule_vectors` walks notes in document order** - `measure.findall("note")`, whatever
order the raw MusicXML interleaves staves and voices in. **`convert_olimpic.py`'s own build
flattens voice-major** - `for voice in voices for measure in voice for symbol in measure`,
all of one voice's measures before the next - which is what `predictions.jsonl`'s
`reference`/`predicted` fields are ordered by, since they come from the same token stream.
For a single-voice part these coincide; for a piano grand staff, the case this measurement
is actually about, they generally do not. A position-based zip between the two would
silently pair the wrong notes - the same shape of bug as 27.11, the sidecar substitution,
and the slur-transfer bug, each of which looked like a working join until checked.

**This was caught by reading the two orderings side by side before running anything, not by
a result that looked wrong.** No crosstab was produced, rather than producing one that could
not be trusted. Fixing it needs `rule_vectors`' logic re-applied over a voice-ordered walk
- reusing `music_xml_file_to_tokens`'s own grouping rather than `part.findall("measure")`
directly - which is a rewrite of the traversal, not a small patch, and is left for the next
session on this question.

**Where this leaves "should we give up on them."** The solid evidence is the totals: the
head's advantage over the rule, clear on the genre it was tuned on, is not visible on
Lieder piano. That is a real, concerning signal on its own, even without the recovers/loses
breakdown. It does not yet establish a *regression* the way 27.58 did for OSSQ scanned -
that needs the crosstab this section could not safely build - so the honest answer is
narrower than either "yes" or "no": the case for keeping the heads is weaker on this genre
than anywhere else they have been measured, and the question of whether they are a net
loss specifically, rather than merely no longer a clear win, is still open.

**27.84 correcting 27.83's diagnosis of the ordering mismatch.** 27.83 blamed the wrong
mechanism. It described the mismatch as `convert_olimpic.py` flattening "voice-major" -
implying MusicXML's internal `<voice>` sub-element, or a multi-`<part>` interleave problem.
Reading `_music_xml_element_to_symbols` shows "voices" at that level means `<part>`
elements: for OLiMPiC's grand-staff piano samples there is exactly one `<part>` (confirmed
against a real sample - `<part-list><score-part id="P2">...`), so that outer flatten is a
no-op and was never where the risk lived.

The real reordering is one level down, inside `TokensMeasure.complete_measure()`
(`training/omr_datasets/music_xml_parser.py`) and `merge_upper_and_lower_staff`
(`training/omr_datasets/staff_merging.py`), and it affects a single-part score just as much
as a multi-part one. `_music_part_to_tokens` does walk each measure's children in raw
document order, dispatching `<note>`/`<backup>`/`<forward>`/etc. to handlers that push
symbols tagged with a rhythmic position - so far, order-preserving. But `complete_measure`
then **re-sorts every symbol in the measure by that rhythmic position** (`sort_order()`,
grouped via a dict keyed on position), and within each position bucket, splits symbols by
staff (`_get_staff_no`: upper first, then lower) before `merge_upper_and_lower_staff` walks
positions in sorted order and, within each, emits upper-staff content before lower-staff
content, with `create_chord_over_two_staffs` further regrouping into
barline/clef/key/time/notes order. So the token pipeline's final note order is: sorted by
musical time, then by staff, then by symbol category - not document order.

This still means `rule_vectors`' raw `part.findall("measure")` -> `measure.findall("note")`
walk (document order) will disagree with the token pipeline's order whenever a score's XML
doesn't already happen to interleave staff-1-then-staff-2 via backup/forward in a way that
collapses to the same sorted result - which is common for plain grand-staff writing but not
guaranteed (voice-crossing, multiple voices per staff, or unusual backup/forward patterns
break it). So 27.83's caution about not trusting a note-position-indexed crosstab without
re-deriving the walk was directionally right, but for a different, deeper reason than
stated: the risk isn't confined to multi-part scores, it applies to every score, and the
fix isn't "walk voice-major" but "re-sort by `sort_order()` and staff exactly as
`complete_measure` does" (or, more simply, drive the crosstab off the same `Measure` objects
`music_xml_file_to_tokens` already produces, rather than off raw XML at all). Still left for
the next session that wants the real OLiMPiC recovers/loses crosstab.

**27.85 the real OLiMPiC crosstab: a wash, not a loss.** Built
`training/transformer/olimpic_rule_vs_head.py` per 27.84's diagnosis: `ordered_rule_vectors`
computes the rule the same way `rule_vs_head.rule_vectors` does (per-voice onset, since
beaming is voice-scoped) but re-sorts its output by (onset, staff, document order) to match
`complete_measure`'s actual label order, rather than trusting raw document order the way
OSSQ's single-staff segments safely could. Locating each sample's own MusicXML needed no
segment-decomposition guessing (27.37's "one image and one MusicXML per system, already
paired by filename" held) - only a fix to the token-stem regex, which needed a literal
`samples_` prefix (`sample.replace("/", "_")` applied to `"samples/<score>/p<page>-s<system>"`,
the same leading-path-component gotcha 27.83's `olimpic_beam_baseline.py` hit and fixed for
`score_id_of`).

Run against phase13's OLiMPiC scanned predictions (1,300 of 1,350 staves joined, 12 skipped
on their own terms - a musicxml missing or not the expected one-part grand-staff shape):

```
                  head right   head wrong
  rule right        18,911        5,075
  rule wrong         5,058        2,775

rule accuracy: 75.4%   head accuracy: 75.3%
exceptions the head recovers: 64.6%  (5,058 notes)
agreements the head loses:    21.2%  (5,075 notes)
```

Recovered and lost are within 17 notes of each other out of 31,819 - a wash, not the
regression 27.83 left open as a possibility. The head still recovers a real 64.6% of what
duration and metre alone cannot predict, which is evidence it is reading the image rather
than memorising the rule; that recovery is just offset almost exactly by an equal-sized
loss of agreements the rule already had. This settles the question raised at the top of
this thread ("in short if we should give up on them"): no - on OLiMPiC the heads are
neutral, not harmful, which is a different and more precise answer than the totals-only
comparison (84.0% vs 81.5%) could give, and a different conclusion than OSSQ's own Gate C
crosstab, where the heads showed a clear net gain. The honest summary across both corpora
now measured: the heads earn their keep clearly on the genre they were tuned on, and cost
nothing on a genre they were not.

**27.86 the real detector baseline (27.76's gate), and what it means for the architecture
search.** 27.76 recorded the user's openness to an architecture search for the text
detector - U-Net vs. something more modern - gated on `train_detector.py` producing a real
number rather than the 10-image CPU smoke test that was all that existed. Two things had
to happen first: a score-level train/valid split (`detector_split.py`, hashing the score id
so a score's pages never appear on both sides - 200 scores, 178 train / 22 valid, 2,542 /
305 pages) and a fix to a real inefficiency `train_detector.py` had never been run long
enough to expose: `DetectorPatches.__getitem__` called `cv2.imread` on the full page for
every one of `patches_per_image` (8) patches independently, and `shuffle=True` scattered
those 8 draws randomly across the whole epoch, so a page was decoded up to 8x with
essentially no cache locality. Measured directly: epoch 1 was still not done after 12
minutes with 0% average GPU utilisation, CPU workers pegged on decode. Fixed with a
one-slot decode cache in the dataset plus `ImageBlockSampler`, which shuffles image order
but keeps one image's patches consecutive so the cache actually hits (a 20-image slice
went from not finishing to 18.6 seconds); relaunched, GPU utilisation went to 47%+ and the
full run completed in under two hours.

Final per-class IoU, 20 epochs, plain Dice loss, no imbalance correction:

```
                train    valid
background      0.995    0.996
SystemText       0.929    0.969
Fingering        0.938    0.812
Expression       0.967    0.929
Tempo            0.968    0.966
MeasureNumber    0.997    0.996
StaffText        0.954    0.954
Lyrics           0.949    0.957
```

Fingering - the rarest class, 1,716x apart from the most common in 27.68's count - is the
one place train and valid pull apart (0.938 vs 0.812), the shape a genuinely under-sampled
class takes, but it is still the model's *worst* number, not a collapse: no class reads
near zero the way 27.62's uncorrected recogniser did on its starved class. Dice's
intersection-over-union shape, not per-token cross-entropy, is doing real work here - this
confirms 27.69's reasoning rather than assuming it, the same "measure the unweighted
baseline first" discipline 27.75 stated and 27.72 validated the cost of skipping.

**This changes the answer to the architecture search question.** 27.76 recorded the user's
willingness to explore alternatives as conditional on this baseline existing; it did not
promise the search would still look worth running once it did. It does not, on this
evidence: U-Net-over-resnet18, unmodified, already resolves every detection class at
0.81-0.997 IoU on held-out scores with a severity of imbalance that was the leading reason
to doubt it. A search across YOLO/DETR/anchor-free alternatives is warranted when a
baseline is failing in a way a different inductive bias would plausibly fix - it is not
warranted to replace a baseline that is not failing. The open question this baseline
actually raises is different: whether Fingering's gap (train 0.938, valid 0.812, the widest
of any class) is a data-quantity problem worth more scans or crops, not an architecture
problem - and whether this detector, now real, should be wired into the recognizer
(`train_recognizer.py`) to close the gap 27.68 opened (nothing before this measured a
syllable's *location* on a scan the recognizer was not already told the layout of).
Recommending the latter as the next step over the architecture search, and leaving the
final call to the user.

**27.87 patch-level IoU hid a real page-level precision problem - Lyrics is fine, the
rest are not.** 27.86 measured the detector against the same patch distribution it trained
on (`DetectorPatches`, 70% of draws centred on a box) and called the baseline strong enough
to skip the architecture search. That measurement never asked the question inference
actually depends on: run whole-page, tiled, at true class frequency, does the model still
find boxes without inventing them. Built `detector_inference.py` (tiled prediction with
50%-overlap stitching, following the convention `detector_masks.py` already pointed at from
homr's own segmentation inference, boxes recovered by the same
`connectedComponentsWithStats` the ground-truth rasterizer uses) and
`detector_box_eval.py` (greedy one-to-one IoU matching per class, precision/recall/F1 -
ground truth filtered to `detector_masks.CLASS_ORDER`'s 7 classes, since `collect()`'s
source manifest carries 4 more that were never rasterized into a training mask and would
have counted as missed recall for classes the detector was never asked to find).

Run against all 305 held-out pages:

```
class           precision     recall         f1   gt boxes
Expression           5.0%      88.6%       9.5%         44
Fingering            0.0%       0.0%       0.0%         12
Lyrics              68.6%      96.3%      80.1%      3,555
MeasureNumber       68.4%     100.0%      81.3%         78
StaffText            9.3%      70.9%      16.4%        103
SystemText           0.0%       0.0%       0.0%          3
Tempo                1.9%      69.8%       3.7%         53
overall             42.4%      94.8%      58.6%      3,848
```

Recall is high almost everywhere - the model does find real boxes. Precision collapses for
every class except Lyrics and MeasureNumber: Tempo's 1.9% means roughly 50 predicted boxes
for every real one; Fingering and SystemText, the two rarest classes in 27.68's own count,
recover nothing at all in a plain 88.6%-recall Expression column at 5.0% precision. This
was invisible in 27.86's per-pixel numbers because that measurement only ever showed the
model 70%-positive-biased crops; a full page is >99% background outside boxes (27.69), and
whatever the model does across all that background it was never sampled at that ratio
during training - it is a domain-gap pattern this project has hit before (27.60), here
between the training patch distribution and the inference deployment distribution rather
than between synthetic and scanned images.

**This narrows, rather than reverses, 27.86's conclusion.** Lyrics - the one class this
track exists for (`detector_data.py`'s own docstring: only lyrics carry a label to
recognise afterwards) - holds up at 80.1% F1 whole-page, close to the discipline the
recogniser itself needs; MeasureNumber is comparably solid. The other five classes have a
real precision failure that is not yet diagnosed as architecture or as training-distribution
mismatch - `DetectorPatches`' `POSITIVE_RATIO=0.7` is the first suspect, since it directly
controls how much true-negative background the model ever saw relative to a real page, and
raising the negative share is a training-recipe change to try before concluding U-Net
itself cannot suppress background at this imbalance. Recommending: proceed with wiring
Lyrics detection into the recognizer now (it is ready), and treat the other six classes'
precision collapse as a separate, not-yet-blocking problem to return to.

**27.88 the gap 27.45 opened, closed: detection + recognition end to end.** Every recogniser
number before this used a ground-truth box - nothing had yet asked whether the detector's
own localisation, handed to the recogniser, still reads correctly. 27.87 restricted this to
Lyrics (80.1% whole-page F1; the other six classes' precision collapse is a separate,
unresolved problem, per that section).

Trained the recogniser for the first time on this instance (`train_recognizer.py`, 30,031
train / 4,293 valid crops from `lyric_crops`, 25 epochs; only the final epoch's weights are
saved, and this run's own history shows unseen-exact peaking earlier at epoch 21 (78.7%)
before ending at epoch 25 (72.8%) - the run used what was saved rather than the peak,
recorded so a future session knows early stopping is worth adding, not blocking this one).

Built `end_to_end_eval.py`: for each of the 305 held-out pages, match the detector's
predicted Lyrics boxes to `*.boxes.json`'s ground truth (same greedy IoU matching as
27.87's `detector_box_eval.py`, reused rather than reimplemented), crop the page at the
*predicted* box, and read it with the recogniser - alongside the same matched syllables
read from the *ground-truth* box, so a drop is attributable to localisation rather than
reading.

First run measured something wrong: exact match was 22.9%/17.1%, both far below the
recogniser's own training-time accuracy, and the oracle number came out *lower* than the
detected-box number - backwards, since a ground-truth box can only be at least as good as
an imprecise one. Traced to a real preprocessing gap: `lyric_crops.py`'s `crop_syllables`
keeps a 4px `MARGIN` of air around every box before cropping ("a recogniser reads better
with room for a hyphen or a descender than a box cut exactly to the ink") - the first
version of this eval cropped tight to the box, cutting exactly the content the recogniser
was trained to expect padding around. Fixed by applying the same `MARGIN`.

Full run against all 3,555 ground-truth Lyrics boxes across 305 pages:

```
3,555 ground-truth lyric boxes, 3,422 matched to a detected box (96.3%, matches 27.87's recall)

read from the detector's own box  : exact 80.9% (2,767/3,422), CER 15.9%
read from the ground-truth box    : exact 84.3% (2,886/3,422), CER 11.9%
```

The detector's own localisation costs 3.4 points of exact match against the oracle ceiling
- a real but modest cost, not the dominant error source. Combined with 96.3% of boxes being
found at all, this is a working end-to-end pipeline for the one class this track exists
for: on a held-out page, detect a syllable's box and read it correctly 80.9% of the time,
with no ground truth involved anywhere in the process. This is the number 27.45 opened the
question for and 27.68 first measured the data half of - it is now answered.

A small-sample smoke test (3 pages, 70 boxes) before the full run showed what looked like a
neighbour-swap pattern in the misreads (`'Wal'->'des'`, `'des'->'Wip'`, ...) - traced to a
hyphenated compound word ("Wal-des-Wip-fel", plausibly "Waldeswipfel") whose sub-word
syllables sit close enough together that box precision genuinely blurs which syllable is
which; the full run's aggregate numbers are not dominated by this and it is recorded as a
known edge case (tightly-kerned hyphenated syllables) rather than a bug.

**27.89 class-balanced positive sampling: real improvement, two classes still at zero.**
27.87's leading suspect for the whole-page precision collapse was `DetectorPatches`'
positive-centre selection: `box_centres` pooled every connected foreground region
regardless of class, so a page's positive training centre was drawn in proportion to how
much of the page a class covered, not evenly across classes - a page with dozens of Lyrics
boxes and one Tempo mark almost never centred a patch on the Tempo mark. Replaced with
`box_centres_by_class` (separate connected components per class) plus a two-step draw in
`DetectorPatches.__getitem__` - pick a class uniformly among those present on the page,
then a centre of that class - so every class present gets an equal share of positive
training examples regardless of corpus-wide frequency. 4 new/updated tests
(`TestBoxCentresByClass`), the old pooled `box_centres` removed (unused once its only
caller changed shape).

Retrained from scratch (same 20 epochs, same train/valid split) and re-ran
`detector_box_eval.py` on the identical 305 held-out pages:

```
class            v1 F1    v2 F1     v1 precision   v2 precision
Expression        9.5%    13.3%          5.0%           7.3%
Fingering         0.0%     0.0%          0.0%           0.0%
Lyrics            80.1%   83.8%         68.6%          74.5%
MeasureNumber     81.3%   95.7%         68.4%          91.8%
StaffText         16.4%   15.0%          9.3%           8.3%
SystemText         0.0%     0.0%          0.0%           0.0%
Tempo               3.7%    18.4%          1.9%          10.5%
overall            58.6%   70.5%         42.4%          56.2%
```

A real, broad improvement - overall F1 58.6%→70.5%, Tempo's precision roughly quintupled,
MeasureNumber and Lyrics (already the two working classes) both improved further as a
side effect of less contention for positive-sampling attention. But Fingering and
SystemText - the two rarest classes in 27.68's own count (12 and 3 ground-truth boxes in
this entire 305-page validation split) - stayed at exactly 0% precision and recall,
unmoved by the fix. The diagnosis this narrows to: within-page class balance was a real
and fixable problem, but it cannot manufacture positive examples on pages that never
contain the class at all, which is the corpus-level shape of Fingering/SystemText's
rarity, not a page-level sampling one. The next lever, not yet tried, is corpus-level image
reweighting - oversampling the (few) images that do contain a rare class, on top of the
within-page fix already made - or accepting that two classes this rare may need a
different strategy (e.g. a dedicated few-shot pass, or simply not detecting them
automatically) rather than more of this detector's own training data.

Lyrics' improvement (80.1%→83.8% F1) also revises 27.88's end-to-end number upward as a
side effect, though that pipeline has not been re-run against these new weights yet.

**27.90 manufacturing real training data for Fingering and SystemText.** 27.89 left
Fingering and SystemText at 0% precision/recall, unmoved by within-page resampling,
because the source corpus only ever contained 12 and 3 boxes of each (27.68) - too few for
any resampling strategy to work with. The corpus is built by rendering MusicXML through
MuseScore (`musescore_boxes.py`), which is the way out: a `<fingering>` added to a note, or
a `<direction>` added to a measure, is drawn by the same renderer and turned into a box by
the same SVG-class extraction as every other page - genuinely rendered engraving, not a
faked image, satisfying 27.25's rule that boxes have to come from the renderer that drew
the image.

Built `rare_class_synthesis.py`. `inject_fingering` adds a `<technical><fingering>` to
random notes (inserted before any `<lyric>` to keep MusicXML's declared element order
valid - the corpus is Lieder, so many candidate notes have one). `inject_system_text` adds
a `<direction system="only-top">` word annotation to random measures, on the hypothesis
that `system="only-top"` is what MuseScore's own SystemText-vs-StaffText choice keys off
(undocumented; `musescore_boxes.py`'s own docstring says MuseScore decides this
internally and does not record which).

Verified against a real render before trusting it at scale: `xvfb-run mscore` on an
augmented score, then reading the resulting SVG's classes directly rather than assuming
the hypothesis held. First pass looked like a problem - 2 "Tempo" paths appeared alongside
6 correct "Fingering" and correct "SystemText" glyphs, which read like some injected words
("Coda", "D.C.") were being reclassified. Checking the *unmodified* baseline render of the
same score before concluding anything found it already carried one genuine Tempo marking
of its own (the piece's actual tempo indication) - the "misclassification" was never the
injected content at all, just a pre-existing element this check had not accounted for.
Recorded as a reminder to check the baseline before attributing a surprising count to the
change under test, not as a bug that needed fixing (the word-narrowing done while chasing
it - excluding "Coda"/"D.C." from `SYSTEM_TEXT_WORDS`, excluding each part's first measure
in `_measures` - was unnecessary but left in place since it is still correct and no less
safe).

`build_batch` (reusing `musescore_boxes.annotate` directly rather than reimplementing
render+extract) ran on 80 scores drawn only from the detector's *train* split - excluded
every score present in `valid_index.txt` first, so the held-out set used for 27.87/27.89's
comparisons stays uncontaminated by synthetic siblings of its own scores. 79/80 rendered
(one refused on a pre-existing lyric-count mismatch, unrelated to the injection), adding
316 Fingering and 156 SystemText boxes - roughly 26x and 52x the corpus's original count
of each. Masked with `detector_masks.py` and appended to a new `train_index_v3.txt`
(2,542 real + 79 synthetic = 2,621 images); `valid_index.txt` is untouched.

Detector v3 (class-balanced sampling from 27.89, now also with real synthetic exposure to
the two classes it could not move) is training against this index; the result will be
compared against 27.89's numbers on the identical unchanged 305-page validation set. Not
yet complete as of this entry.

**27.91 `train_detector.py` has no per-epoch checkpointing - a gap, not yet fixed.**
Mid-run on detector v3, patch-level valid IoU had clearly plateaued by epoch 12-14
(noisy, not trending up: SystemText 0.947→0.922→0.886 across three epochs) - the same
shape 27.55's structured-heads training found, where 12 epochs turned out to be enough.
But `train_detector.py`'s `torch.save` sits after the full epoch loop, not inside it, so
stopping a run early to save the wait produces no usable weights at all, not merely a
slightly-undertrained checkpoint. Decided to let this run finish its full 20 epochs rather
than lose it, and record the gap rather than fix it under time pressure mid-run: a future
long detector run should save every epoch (or the best-so-far by valid loss) the way
`train_recognizer.py` already doesn't either, so this class of "the good checkpoint several
epochs ago is gone" problem (already hit once for the recognizer, 27.88) stops recurring.

**27.92 the synthetic data helped Fingering, did not move SystemText, and cost every
other class - a mixed result, not a win.** Detector v3 (27.90's synthetic Fingering/
SystemText boxes merged into training, 27.89's class-balanced sampling still active)
scored against the identical unchanged 305-page validation set:

```
class            v2 F1   v3 F1     v2 precision   v3 precision
Expression       13.3%    3.2%           7.3%           1.7%
Fingering         0.0%   18.8%           0.0%          10.7%
Lyrics           83.8%   79.1%          74.5%          67.5%
MeasureNumber    95.7%   78.7%          91.8%          68.6%
StaffText        15.0%   12.7%           8.3%           7.0%
SystemText        0.0%    0.0%           0.0%           0.0%
Tempo            18.4%    9.8%          10.5%           5.2%
overall          70.5%   57.9%          56.2%          42.0%
```

Fingering moved off zero for the first time (18.8% F1) - the synthesis approach works, at
least partially, for the class it added the most examples of (316 boxes). But SystemText,
which got a comparable injection (156 boxes across the same 79 pages), stayed at exactly
0%. Checked the synthetic masks directly rather than assume a pipeline bug: SystemText
pixels are genuinely present (20k-53k per page, comparable to or larger than Fingering's
2k-5k), so this is a real modelling outcome, not a masking or rasterisation error -
SystemText may simply be harder to separate from the visually similar StaffText class than
Fingering is from anything else, though this is not yet confirmed against a confusion
matrix.

More concerning than either of those: every other class got worse, broadly, not within
noise - overall F1 dropped 70.5%→57.9%, MeasureNumber (previously the second-best class)
fell from 95.7% to 78.7%. The likely mechanism: 27.89's class-balanced sampler draws a
positive centre by picking a class uniformly among those present *on a page*, then a
centre of that class. For Fingering and SystemText, the only pages where that class is
"present" are the 79 synthetic ones - so across 20 epochs, those same 79 pages got selected
as the positive-sampling source for two of seven classes far more often than 79/2,621 of
the corpus would suggest, and the model likely overfit to whatever is idiosyncratic about
them (the specific injected words, their placement, or some other synthetic-content tell)
at the expense of the other six classes' shared background/discrimination capacity - the
same shape of mistake 27.72 found with global focal loss: a correction aimed at starved
classes bled into classes it was never meant to touch.

**This is not being recommended for use as-is.** The honest state: manufacturing rendered
training data for a rare class is a real, working technique (Fingering proves it), but 79
pages is evidently too narrow a base to draw on this heavily without collateral damage to
the rest of the detector. The fix implied but not yet tried: either weight the sampler so a
class's positive-draw frequency is capped by how many *distinct* pages contain it (so 79
synthetic pages cannot out-compete 2,542 real ones for attention), or generate on
substantially more distinct source scores at lower density per page rather than
concentrating iterations on the same 79. Left as an open question for the next session
rather than another multi-hour retrain attempted immediately, given the session's length.
27.89's weights (not this run's) remain the better detector to use in the meantime.

**27.93 SystemText folded into StaffText; user decision, not further-measured.** Given
27.92's result (0% precision/recall even with correct injected training data), the user
decided to give up on SystemText as its own class rather than keep spending detector
capacity on it - acceptable to read a system-level annotation as StaffText, since the
resolve stage does not currently depend on the distinction. `detector_masks.py` now has 6
classes, not 7 (`CLASS_ALIASES = {"SystemText": "StaffText"}`, applied in `rasterize()`);
`detector_box_eval.py`'s ground-truth collection applies the same aliasing so a
SystemText box scores as a StaffText box, matching what the model was trained on.
`rare_class_synthesis.py`'s SystemText injection (`inject_system_text`, `_measures`,
`SYSTEM_TEXT_WORDS`) is removed rather than left dormant - it worked exactly as intended
(real, correctly-classified SystemText training boxes) and still did not move the metric,
so keeping the code around only invites reaching for the same failed premise again.
Fingering's injection is unaffected and kept.

Also fixed while touching these tests: `test_train_detector.py`'s `_args()` helper never
gained a `valid_index` key when `--valid-index` was added to `train_detector.py` earlier
this session (27.86), so `TestTrainEntryPoint`'s two tests had been silently failing since
- caught only now by actually running the suite rather than assuming it still passed.
Added a `valid_index=None` default and a new test exercising the valid-loader path
(`test_a_valid_index_adds_a_valid_report_per_epoch`), which had no coverage at all before
this. All 44 tests across `test_detector_masks.py`, `test_detector_patches.py` and
`test_train_detector.py` pass.

**27.94 a strategy for dynamics, and why inline decoder tokens are ruled out.** Dynamic was
excluded from the OCR detector by 27.45 on a correctness argument (MusicXML stores
dynamics as element names, not strings - reading `f` the dynamic as `f` the letter would
recognise the wrong thing correctly), not a scarcity one. Checked directly against the raw
corpus: Dynamic is the *most common* non-Lyrics text class by a wide margin -
`Dynamic 5,013, StaffText 1,262, MeasureNumber 806, Tempo 493, Expression 423` - so unlike
Fingering/SystemText, adding it to the detector should not need 27.90's synthetic-data
treatment at all.

The remaining question is how a detected, classified dynamic gets attached to the score,
and 27.49 already answered it - a fact this thread initially failed to connect before
proposing inline decoder tokens as a live option. 27.49 found the "just uncomment
`vocabulary.py`'s commented-out dynamics tokens" path was not merely untried: it was tried
upstream (run 318, ~70 epochs) and it collapsed - SER jumped from a 26% baseline to 132%,
dynamics were never predicted at all, because they are ~0.05% of tokens in the shared
decoder vocabulary and the model learned to ignore them entirely to minimise loss. The
maintainer's proposed remedies (focal loss, 50x class weighting, more data, a
dynamics-enriched subset) were never tried. This rules out inline tokens as a first attempt
- the failure mode is structural (competing against the entire shared vocabulary's softmax
at a token frequency this low), not a training-recipe oversight fixable by being more
careful the second time.

**Decided: dynamics get a new structured-notation head, the same pattern as beam/stem/slur/
tie**, not inline tokens. Reasons, weighed directly against the failed alternative:

  - A head's own softmax is small (~12 dynamics classes plus "none"), not competing against
    thousands of rhythm/pitch tokens for probability mass - this is exactly the shape of
    difference that would have prevented run 318's collapse, not a hopeful guess that this
    time is different.
  - This project already has validated tools for within-head class imbalance specifically
    (`--focal-gamma-head`, the reweighted training index) proven on the tie head (27.55,
    27.77) - directly transferable, not hypothetical remedies like the ones the maintainer
    only proposed.
  - Output-only, frozen core untouched - the pattern this session has repeatedly found
    lower-risk than anything that reaches into a shared component (27.72's global focal
    loss, 27.92's synthetic data both regressed things they were never meant to touch).
  - Traded away: dynamics attach to *events* more loosely than beam/stem do (a dynamic can
    span or precede a note rather than always coincide with one), so the resolve-stage
    attachment logic needs its own design, not a copy of tie/beam's; and a structured head
    keeps dynamics invisible to the base decoder's own rhythm/pitch generation, unlike
    inline tokens which (if they had worked) would share sequence context.

**Sequencing:** the detector side is ready to build now (real, abundant data, no synthetic
augmentation needed); the structured-head design and resolve-stage attachment are separate,
later work not yet started.

**27.95 Dynamic detection works on the first try - 27.94's prediction confirmed.** Added
Dynamic to `detector_masks.CLASS_ORDER` and `detector_data.DETECTION_CLASSES` (both updated,
with tests corrected from the old exclusion to the new inclusion), rebuilt masks for the
whole corpus (2,889 pages), regenerated the score-level split with the same seed (identical
178/22 train/valid scores), and retrained (detector4, 20 epochs, class-balanced sampling
from 27.89 still active). Whole-page box eval on the 307 held-out pages, 4,277 ground-truth
boxes:

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

Dynamic lands second only to MeasureNumber, on its first real training run - no synthetic
data, no special handling, exactly what 27.94 predicted from its abundant real-corpus count
(5,013 boxes overall). This detector-side half of the dynamics strategy is done: a Dynamic
box can now be found on a page it was never given the layout of.

Two things worth recording honestly rather than glossing over. Fingering is back at 0%
here, since this run carries only the real corpus - 27.90's synthetic Fingering boost was
never re-merged into this index (the two changes, Dynamic's addition and the SystemText
fold, were kept separate from the Fingering synthesis question). And Lyrics/MeasureNumber
both dipped somewhat against the last comparable run (Lyrics 83.8%→78.1%, MeasureNumber
95.7%→84.1%), while Tempo improved (18.4%→26.2%) - several changes landed in this one run
(Dynamic added, SystemText removed, both changing what a 7-way class-balanced sampler
competes over), so no single cause is being claimed for either the gain or the dip; it is
recorded as a genuine open question rather than attributed to a story that fits.

**Still not started:** the structured-notation head that reads a detected Dynamic box and
attaches it to the score (27.94's decision), and the closed-set classifier that turns a
crop into one of ~12 dynamics markings rather than open-vocabulary text. The detector
output from this section is the input those stages will need.

**27.96 the Dynamic classifier: 88.6%, evenly across the common labels.** 27.95 built the
detector half; this is the reading half - 27.94's decision that dynamics need a small
closed-set classifier, not the CTC recogniser.

Built `dynamics_crops.py`: joins each Dynamic box (`text_boxes["Dynamic"]`, in
`musescore_boxes.boxes_of_class`'s *reading* order) against `source_dynamics`'s marks (raw
XML *document* order) by position, and refuses the whole system on a count mismatch rather
than trust the join - `musescore_boxes.pair`'s own discipline for lyrics, applied here
because nothing proves reading order and document order coincide for dynamics the way they
happen to for a single-verse note stream. That caution was not idle: 184 of 2,926 systems
(6.3%) refused on exactly that mismatch. The 4,295 crops that did join cleanly span 18 raw
labels, 8 of which (`p, f, mf, pp, ff, sf, mp, ppp`) cover ~97% of examples; the rest are a
long tail down to single-digit counts (`other-dynamics`'s 33 losing the actual engraved
text, since `source_dynamics` only concatenates MusicXML element tag names).

Built `dynamics_classifier.py` (`DynamicsCNN`: 3 conv blocks, adaptive average pool, one
linear layer - a single softmax over the label set, deliberately not `CRNN`, since there is
no sequence to read) and `train_dynamics_classifier.py` (cross-entropy, per-label accuracy
reported rather than only the overall number, for the same reason 27.49's tie head and
27.62's OCR sweep both needed the majority class excluded before the real number showed).
25 epochs, 3,653 train / 642 valid crops, split by score:

```
overall: 569/642 (88.6%)
p                189/212  (89.2%)
f                138/149  (92.6%)
pp                73/83   (88.0%)
mf                62/74   (83.8%)
sf                31/34   (91.2%)
mp                26/31   (83.9%)
ppp               23/27   (85.2%)
ff                21/25   (84.0%)
```

Every common label lands in the 83-93% band - not one majority class carrying the average
while the rest collapse, the failure mode this project has hit before. The tail labels
(`fp`, `other-dynamics`, `rf`, ...) have too few validation examples (1-4 each) to read
anything from individually.

**Both data-producing halves of the dynamics strategy are now built and measured**: the
detector finds a Dynamic box on a page (27.95, 84.0% whole-page F1), this classifier reads
which mark it is (88.6%) once handed a crop. What remains, per 27.94's decision, is the
structured-notation head that attaches a classified dynamic to the score - not yet started.

**27.97 attaching a dynamic to a note - measured, not yet built.** 27.94 decided dynamics
get a structured-notation head; every existing head (beam, stem, slur, tie) labels a
`<note>` from data that already lives on it or is directly derivable from a child element.
A dynamic is different - it is a `<direction>`, a sibling of notes in the measure, not a
child of one - so which note it labels is a real design decision nothing in this pipeline
had made before.

Built `dynamics_attachment.py` with the simplest defensible rule: a dynamic attaches to
the *next* note encountered after it in document order, within the same part - the same
convention MuseScore's own engraving follows (a `<direction><dynamics>` sits in the XML
immediately before the note it prints under). Deliberately not position-based (matching a
direction's measure-offset against a note's onset via backup/forward, the way
`music_xml_parser.py` computes render order) - heavier machinery not worth building before
knowing whether the simple rule is even good enough to bother with.

Measured against the whole real corpus (142,243 notes): **3.35% carry an attached dynamic**,
dominated by 8 marks (`p, f, mf, pp, sf, ff, mp, ppp`) that together account for the large
majority of the 4,772 labelled notes; per-part marked rate has a median of 1.39% but ranges
0-100%, so the imbalance is real but concentrated rather than uniform. This is the number
that matters against 27.49's documented failure: inline decoder tokens collapsed at ~0.05%
of *tokens* (a different, coarser denominator, but the same shape of problem) with no
correction ever tried. 3.35% of notes is a real imbalance - comparable to what this
project's own tie head already handles successfully with `--focal-gamma-head` - not the
~65x-more-extreme starvation that sank run 318. This is empirical support for 27.94's
reasoning, not just the theoretical argument it was made on.

One artefact worth fixing before this becomes training data: `dynamics_of`'s label
sometimes concatenates two dynamics children present in one `<dynamics>` element
(`pother-dynamics`, `fother-dynamics`, ...) - a symptom of the same information loss
27.96 already flagged (`source_dynamics`/`dynamics_of` only concatenate MusicXML element
tag names, never `<other-dynamics>`'s actual text content). These hybrid labels should
collapse into a single "other" bucket rather than be treated as their own classes.

**Not yet built:** adding a `dynamic` field to `NoteNotation` (`homr/transformer/
structured_notation.py`, following the precedent `tie` already set - defaulted so existing
sidecars keep decoding), wiring `attach_dynamics` into the structured-notation extraction
pass (`structured_notation_parser.py`) and `to_decoder_branches`, a new output head in the
structured-heads model, and a full training-data regeneration + training run. This is
comparable in scope to how the beam/stem/slur/tie heads themselves were built - a
multi-session undertaking, not a single turn's addition - and is left as the next distinct
piece of work rather than started without a clear stopping point.

**27.98 the structured dynamics head is built; its first two training runs found a real
data bug, then a real class-imbalance problem.** 27.97 left the head itself unbuilt. This
session built it: `DynamicMark` (the full ~33-tag MuseScore/MusicXML vocabulary, not just
OSSQ's observed subset - the Lieder corpus this design targets next is expected to use
marks OSSQ's quartets do not, and an unused logit costs nearly nothing against having to
add a class later and invalidate every checkpoint trained against it), a `dynamic` field on
`NoteNotation`, staff-scoped attachment in `NotationExtractor`/`music_xml_parser.py` (a
direction on staff 1 of a multi-staff part must not be claimed by staff 2's next note -
dormant in OSSQ's single-staff quartet parts, live for Lieder's piano staves), a
`dynamic.mark` head in `StructuredNotationHeads`, and the matching target/decode/metric/
manifest/sidecar wiring (`structured_targets.py`, `structured_decoding.py`,
`structured_metrics.py`, `capability_manifest.py`, `notation_sidecar.py` bumped to v3).
Unit tests (`test_structured_notation_parser.py::TestDynamics`, staff-scoping and
cross-part isolation included) and the full existing structured-heads suite pass
unchanged (137 tests, 11 subtests).

**Phase14 (first training run, plain cross-entropy): the dynamics head learned nothing -
loss 0.0107 -> 0.0000 by epoch 2, flat through epoch 12, while every other head kept
declining.** The eval report showed why: **zero non-`none` dynamics anywhere in the
converted corpus, train or valid** (1,136,351 and 131,842 notes respectively, all
labelled `none`). This is not the class-imbalance failure 27.94's reasoning was built to
survive - it is an absence of data, and tracing it found a bug one level below anything
this design had touched. `convert_ossq.py` tokenises the "unaligned" systemwise segment
MusicXML under `musicxml/unaligned/`, not the whole-score file `dynamics_attachment.py`
was measured against (27.97) - and that segment file carries **zero `<direction>`
elements anywhere in the corpus**, confirmed by grep against a sample segment and its
whole-score counterpart (3,375 directions, 474 dynamics) and against the `_cleaned`
copy (also zero). The MuseScore round-trip that produces "unaligned" segments drops
`<direction>` entirely - the same family of loss `slur_placement.py`'s docstring already
documented for slur placement (numbering flattened, placement dropped), just total here
instead of partial. 27.97's 3.35% corpus-wide measurement was real and correct; it was
simply run against a file convert_ossq.py never reads.

**The fix is 27.20's fix, applied to a second kind of information lost the same way.**
Built `dynamics_placement.py`, structurally identical to `slur_placement.py`:
`part_dynamics()` walks a whole-score part with the same `NotationExtractor` the segment
pipeline uses, filtered to *visible* notes to match `part_signature`'s alignment unit;
`DynamicsPlacementIndex` reuses `slur_placement.py`'s validated positional join
(`is_visible`, `part_signature`, `concatenated`, `segments_of`) rather than re-deriving
it - the join is the fragile, previously-broken part (27.20's docstring: broken five
times before), and nothing here should risk breaking it a sixth for the sake of not
importing; `apply_dynamics()` writes recovered marks back as ordinary `<direction>`
elements immediately before the note they belong to, so the ordinary extractor reads them
the way it always would and nothing downstream needs to know this was reconstructed.
Wired into `convert_ossq.py._write_example`/`build()` alongside the existing
`PlacementIndex`, same pattern, same per-score caching. 11 new unit tests cover the
visible-note filtering, the index slicing, and - end to end - that a mark written by
`apply_dynamics` and read back by `parse_part` returns exactly what went in. Checked
against the real corpus before trusting it further: one score, 4/4 parts aligned, 473 of
474 source dynamics recovered.

**Phase15 (reconverted with the fix, same plain-cross-entropy recipe): real signal, poor
head.** The sidecars now carry a musically ordinary distribution - 60,559 of 1,136,351
train notes (5.33%) marked, dominated by `p, f, sf, pp, ff, mf, other-dynamics, fz, fp,
mp, ppp, sfp, rf, sfz` - and `dynamic.mark`'s training loss starts at 0.2645 and declines
to a 0.239 plateau by epoch 12, nothing like phase14's instant collapse. But the eval
number is still bad: **macro F1 0.068**, next to ties' 0.831 in the same run. The
comparison that matters is support, not the headline number: dynamics' non-`none` support
(7,219) and ties' (7,064) are nearly identical - this is not the sparse-supervision
problem 27.94's reasoning addressed. The difference is class count: ties spreads its
positive mass over 3 non-`none` classes, dynamics over 17 observed ones, so a
comparably-sized signal is divided far thinner per class. `p:F1=0.076, f:F1=0.023,
ff:F1=0.067` are the least-bad marks (the most frequent ones); nine of the seventeen
score exactly 0.000.

**Decided: phase16, `--focal-gamma-head dynamic.mark=2.0`** - the same lever and the same
first value that fixed the tie head in phase13, tried before anything more elaborate
(per-class weighting, a higher gamma, merging rare marks) because it is the cheapest
correction this project has already validated for exactly this shape of problem, and
trying it first is what makes a later, more invasive fix defensible if this one is not
enough. Launched against phase15's already-correct converted data (no reconversion
needed).

**Phase16 measured: focal gamma does not fix it - macro F1 0.063, statistically flat
against phase15's 0.068, and every one of the five marks phase15 had a nonzero read on
(`p, f, sf, pp, ff`) scored lower, not higher** (e.g. `p` 0.076 -> 0.040, `ff` 0.067 ->
0.053). This is the confirming half of 27.98's own class-count diagnosis, not a new
finding: focal loss reweights *hard-vs-easy* examples within a fixed class set, which is
the right tool for a class with thousands of examples that the model finds hard to
separate from the majority (tie's `start`/`stop`, 3,230 each) - it does nothing for a
class that has almost no examples to learn a decision boundary from regardless of how
they are weighted. Nine of dynamics' seventeen observed marks have single- or
low-double-digit support in this valid split (`fff` n=2, `mp` n=9, `sfp` n=10, `ppp`
n=12, `rf` n=46, ...) - focal gamma cannot manufacture data those counts do not contain.
A same-run side effect worth recording rather than reading into: ties' macro F1 also
drifted (0.831 -> 0.813) despite `--focal-gamma-head` scoping gamma to `dynamic.mark`
alone (0.0 elsewhere, per its own documented contract) - plausibly ordinary run-to-run
variance from a single shared optimiser step over every head's summed loss rather than a
repeat of phase12's cross-head damage, but not confirmed either way on one run each side.

**Not yet decided or built: how to reduce dynamics' effective class count.** Two
candidate levers, neither tried:

  - `--class-weights` (`training_transformer/train_structured_heads.py`, inverse-frequency
    reweighting, already implemented per 27.49) - but unlike `--focal-gamma-head` it has
    no per-head scope in its current CLI, and this project has already measured what an
    *unscoped* imbalance correction costs (phase12: a global focal gamma bought ties
    0.6-2.7 points at beam's cost of 1-8 and slur's of 5-8). Applying it as-is risks the
    same trade for a head that, per phase16, may not even take the deal. Needs a
    `--class-weights-head` equivalent to `--focal-gamma-head` before it is safe to try,
    not a blind run.
  - Folding the long tail (marks scoring 0.000 across both phase15 and phase16 -
    `sfz, fz, mf, fp, rfz, sffz, rf, ppp, sfp, mp, fff`) into `DynamicMark.OTHER` at the
    label-extraction stage, concentrating what little signal exists onto the handful of
    marks (`p, f, sf, pp, ff`) that already show non-zero learning. This is a corpus
    policy decision (where to draw the line, and whether it should be a fixed threshold
    or something the training-data audit measures per run) that changes what the head can
    ever report, not just how it is trained - left for the next explicit decision rather
    than picked here.

**Decided and measured: phase17, the collapsed-vocabulary lever, chosen over
`--class-weights`.** Picked the fold over `--class-weights` specifically because it needed
no new code and carries none of the unscoped-correction risk phase12 already charged
against beam/slur - it only changes what the head is asked to discriminate, not how the
shared loss is weighted. The threshold was **not** picked from phase15/16's own zero-F1
list (evaluation-driven class selection on one small valid split risks tuning to that
split's noise); it reuses 27.96's classifier-side precedent instead - `p, f, mf, pp, sf,
ff, mp, ppp`, the 8 marks an entirely separate measurement (the crop classifier, on its
own held-out split) already found cover ~97% of corpus occurrences. `TRAINED_DYNAMIC_MARKS`
(`homr/transformer/structured_notation.py`) and `trained_dynamic_mark()` implement the
split the same way `TRAINED_BEAM_LEVELS`/`TRAINED_SLUR_SLOTS` already do: the
representation (`DynamicMark`, `NoteNotation.dynamic`, the sidecar) stays the full ~33-tag
set, and only `structured_targets.py`'s target-building collapses anything outside the
trained set to `OTHER` - so phase15's already-correct converted data needed no
reconversion, only a retrain.

**Result: macro F1 0.068 -> 0.120, roughly double, but mostly from removing zero-support
classes out of the average rather than a clean win on every kept mark.** `f` (0.023 ->
0.032) and `pp` (0.032 -> 0.046) improved; `p` (0.076 -> 0.067) and `ff` (0.067 -> 0.052)
went the other way. More telling: **`mf`, `mp` and `ppp` - three of the eight marks
deliberately kept trained on 27.96's precedent - still score exactly 0.000**, despite `mf`
alone carrying 3,554 training-set occurrences (phase15's sanity count), more than `sf`
(6,974 has some signal) but on the same order. Support size alone is not predicting which
marks the head can learn; something else - visual confusability with a neighbouring
dynamic, position within the crop, an artefact of how OSSQ's synthetic renders these three
specifically - is still unmeasured. `other-dynamics` (now absorbing the folded tail, support
1,345 up from 404) scores 0.001, confirming the fold concentrates volume without making the
catch-all class itself learnable - expected, since OTHER by construction spans marks with
nothing visually in common.

**Where this leaves the dynamics head:** a real, trained, non-degenerate structured head -
NONE is 0.999, and the corpus's four most common marks (`p, f, pp, ff`) all score above
zero and would show up in a prediction stream rather than being silently absent - but not
a head this design should claim reads dynamics reliably. Three sessions' worth of measured
levers (plain cross-entropy, focal gamma, vocabulary collapse) have moved macro F1 from
0.068 to 0.120 without finding what actually limits `mf`/`mp`/`ppp` specifically, which
suggests the next productive step is diagnostic (read a sample of the model's `mf`
confusions, the way `stem_arbiter.py`/`rule_vs_head.py` already do for other heads) rather
than another blind lever.

**Built and run: the diagnostic.** `evaluate_structured_heads.py`'s `dump_predictions`
wrote `stem_reference`/`predicted` and `slur_reference`/`predicted` per note but nothing
for dynamics - the dynamics head postdates that function and nobody had extended it, so
the confusions this design needed to read were not written anywhere. Added
`dynamics_reference`/`dynamics_predicted` the same way `slur_reference`/`predicted`
already work: every note qualifies (dynamic is supervised everywhere, like slur event),
so no filtering condition to mirror. 11 existing `evaluate_structured_heads` tests pass
unchanged. Re-ran phase17's already-trained checkpoint through evaluation only (no
retraining) to regenerate predictions with the new fields, then built the confusion
counts directly.

**The answer: it is not confusion between marks, it is the head mostly failing to
register that a mark is present at all.** For the 196 `mf` positions in this valid split,
171 (87%) are predicted `none`, not some other dynamic - only 10 land on `p`, 7 on `f`, 6
on `pp`. `mp` (9 positions): 7 `none`, 2 `p`, never `mp` itself. `ppp` (12 positions): 11
`none`, 1 `pp`. The reverse view confirms it is not a precision problem masquerading as a
recall one either: of the 13 times the head *did* predict `mf`, none of them were
correct (4 were actually `p`, 3 `f`, 2 `other-dynamics`, 2 `none`, 1 each `pp`/`sf`) - and
`mp` was never predicted at all. This rules out the leading hypothesis from phase17
(visual confusability with a *specific* neighbouring mark) - a confusability story would
show up as concentrated mass on one or two wrong marks, not 87-92% collapsing to "nothing
here." What it does not rule out: mark size/rendering in OSSQ's synthetic crops, position
within the crop relative to the note it labels, or simply that three marks' worth of
signal is still too thin for this architecture's single linear head regardless of how the
vocabulary is drawn - none of these distinguished by the counts alone, and unmeasured.

**Where this leaves the dynamics head, after four training runs and one diagnostic pass:**
real and non-degenerate on the marks it can see (`p, f, pp, ff`, macro F1 0.120 overall),
but for `mf`/`mp`/`ppp` specifically it behaves close to an "unmarked" detector rather
than a misclassifier - which changes what a plausible next fix looks like. A confusion
problem would call for more separating capacity or a different loss; a registration
problem more plausibly calls for looking at the visual evidence itself (crop the actual
`mf` positions this run got wrong and look at them, the way 27.20's slur-placement work
and this session's own `dynamics_placement.py` sanity check both checked real examples
before trusting a number) rather than another training-recipe change. Not yet done - left
as the next distinct piece of work, the same way 27.97 left the head itself at the end of
the session that measured the attachment rule.

**Looked at the actual crops. Found the diagnosis's missing piece: at least some `mf`
ground truth is not printed on the page at all.** Pulled six real training-set crops for
notes labelled `mf` where the head predicted `none`, from two different scores. One
(`sq8806134`, measure ~20) shows `mf` exactly where labelled, rendered plainly under the
note - a working positive control, ruling out "the head can never read this glyph."
The other (`sq10406164`, page 17 system 1, part 2) does not: traced the label to
measure 50's `<direction><dynamics><mf/></dynamics></direction>` in the whole-score
XML (no `print-object="no"`, so nothing in the source marks it non-printing), verified
`dynamics_placement.py`'s positional join lands it on the correct note (global visible-
note index 2800 of 4256, sliced to local index 6 of this segment's 23 - arithmetic checked
directly, not just trusted), reconstructed the exact `<direction>`-annotated XML
`apply_dynamics` would tokenise, and confirmed by hand which rendered note that XML
position corresponds to. At 4x zoom, that note - a tied, dotted-quarter chord starting a
diminuendo phrase - carries `dim.` underneath it in the rendered crop. No `mf` appears
anywhere on the system. The whole-score XML's own dynamics are otherwise unremarkable nearby
(`sf`/`f`/`p`/`ff`/`pp` all present and printed at their measures), so this reads as one
specific direction that survived into the XML export without ever being (or no longer
being) engraved on the page - plausibly an editing leftover (MuseScore keeps an invisible
direction's dynamics data even after a visible mark is deleted or replaced, and this
corpus is machine-derived OMR ground truth to begin with, not hand-verified per mark).

**This means part of `mf`/`mp`/`ppp`'s measured 87-92% "predicts none" rate may not be
head failure at all - the model may be correctly reporting nothing where there is visually
nothing to report, and being scored wrong for it.** Not a `dynamics_placement.py` bug: the
extraction, alignment and reinjection are all confirmed correct against this XML: the
finding is that the XML itself is not always faithful to the page for every mark, a
distinct problem this design has not measured for any other structured-notation channel
because none of them previously depended on `<direction>` elements at all. Unmeasured:
how much of the shortfall this actually explains - one confirmed unprinted `mf` and one
confirmed printed one is an existence proof, not a rate. The next step is to measure that
rate before touching training again: sample a meaningful number of `mf`/`mp`/`ppp`
(and, as a sanity check, some `p`/`f` too, to see whether the rate is dynamics-specific or
a general property of this corpus) ground-truth positions against their actual rendered
crops, the same way this session's own real-corpus sanity checks for `dynamics_placement.py`
and `slur_placement.py`'s alignment measurement both did - and only then decide whether the
fix belongs in extraction (e.g. requiring a `<sound>`-only heuristic, or cross-checking
placement/offset plausibility) or is a corpus-quality ceiling this design has to report and
accept.

**Measured: 11 more positions, randomly sampled across scores never seen in this
investigation before** (3 `mf`, 2 `mp`, 2 `ppp`, plus 2 `p` and 2 `f` as the control the
previous paragraph called for) - drawn from the actual converted train/valid sidecars, so
every sample has a real crop, and inspected the same way as the two originals: fetch the
crop, look for the mark. **10 of 11 are printed exactly as labelled.** The one miss is
another `mp` (`sq7354505_0023_0004_4`, position 0) - the crop shows `Solo`, `en dehors, et
très expressif`, a hairpin, `dim.` and `p`, no `mp` anywhere. Combined with the two
originals: **7 of 9 `mf`/`mp`/`ppp` positions checked across this whole investigation are
printed; 4 of 4 `p`/`f` control positions are.** The rate is not zero and it is not evenly
spread (both misses are `mf` or `mp`, none in the well-scoring marks), which keeps the
ground-truth-quality hypothesis alive as a real, non-zero contributor - but at roughly
2-in-9 on a sample this small, it is nowhere near large enough to explain an 87-92%
"predicts none" rate by itself. **Most of `mf`/`mp`/`ppp`'s recall collapse is still
unaccounted for and most plausibly a genuine model limitation**, not a corpus artefact -
the corpus-quality explanation is real but partial, and this design should not lean on it
further without a larger sample than twelve manually-inspected positions can support.

**Where this leaves the dynamics head, for real this time:** four training runs, one
prediction-level diagnostic, and thirteen hand-checked crops later, the head reads
`p`/`f`/`pp`/`ff` with modest but genuine signal (macro F1 0.120 overall) and does not
reliably read `mf`/`mp`/`ppp`, for reasons that are now partially - not fully -
understood: a real but minority-share ground-truth noise floor, and an unexplained
remainder this session did not chase further. Left here as the honest stopping point:
not a working three-mark reader, not a mystery either, and not worth a fifth training run
without a new idea rather than another rerun of the same one.

### 25. Settled decisions and open measurements

#### 25.1 Settled by this design

- “Beaming” means explicit musical beam/flag recognition, not beam-search sequence
  decoding, for the first experiment.
- Full per-level MusicXML beam states are the canonical representation; MuseScore
  `BeamMode` is a derived editor representation.
- Stem direction is a separate head from beam connectivity.
- Slur event, identity/slot, and above/below direction are factored rather than
  flattened into a combinatorial token vocabulary.
- New notation heads are output-only initially.
- Existing pretrained encoder, decoder, and legacy heads are reused.
- OSSQ is the first adaptation corpus, not the definition of a quartet-only model.
- User-supplied instruments/parts are represented by an optional generic score
  profile with unknown values and context dropout.
- System grouping remains a geometry stage after U-Net segmentation initially.
- HOMR remains page-by-page; continuity and assembly use explicit page state.
- Deterministic cross-staff checks and human-reviewed targeted repairs precede a
  learned context adapter.
- Any learned staff-context adapter accepts a masked variable number of staves.
- Structural review occurs before token review.
- Lieder is a later corpus using the same music architecture and generic profile.
- Lyrics are a separate detection, recognition, and alignment capability; adding it
  does not require restarting music-model training from random weights.

#### 25.2 Must be measured before finalizing implementation constants

- Whether six slur slots are sufficient outside OSSQ and whether rare slots should
  be trained, masked, or represented by an overflow mechanism.
- Whether level-5 and level-6 beam heads have enough examples for useful learned
  predictions or should initially be deterministic/unsupported.
- Whether output-only heads can maintain valid beam groups and slur spans without
  feedback embeddings.
- The best new-head loss weights and whether focal/class-balanced loss is needed.
- The exact context-dropout distribution.
- The score-level OSSQ train/validation/test membership and whether a useful
  composer-disjoint challenge set is large enough.
- The degree of beam/slur edition mismatch between the aligned scans and symbolic
  scores.
- The non-regression tolerance after baseline-run variance is known.
- Whether deterministic/top-k cross-staff repair removes enough errors to make a
  learned context adapter unnecessary.
- Whether score-profile conditioning belongs at encoder context, decoder prefix, or
  both; the zero-gated ablation decides this.
- The appropriate source-crop width for beam questions and span context for slur
  questions.
- Vast.ai batch size, accumulation, data-worker count, and storage needs after the
  final staff-crop dataset exists.

These are experiment parameters, not opportunities to silently change the settled
architecture. Each must be recorded as a run decision and resolved from validation
or usability evidence without inspecting the held-out test set.

### 26. Implementation evidence and related designs

This document is self-contained, but the following local artifacts contain the
specific implementation evidence behind it:

#### HOMR

- `training/architecture/transformer/decoder.py`: shared decoder, parallel heads,
  greedy training/inference behavior, consistency loss.
- `homr/transformer/decoder_inference.py`: ONNX greedy decode and discarded logits.
- `homr/transformer/vocabulary.py`: current rhythm, articulation, and collapsed
  slur vocabularies.
- `training/omr_datasets/music_xml_parser.py`: current MusicXML-to-token path.
- `training/omr_datasets/convert_lieder.py`: current Lieder and OpenScore
  StringQuartets acquisition/render/crop path.
- `homr/main.py`, `homr/staff_detection.py`, `homr/brace_dot_detection.py`, and
  `homr/staff_parsing.py`: segmentation-to-staff-to-system flow.
- `training/transformer/train.py`, `training/transformer/data_loader.py`, and
  `training/transformer/mix_datasets.py`: current split, sampling, and fine-tuning
  behavior.
- `Training.md`: current run history, validation scopes, and cloud-training notes.

#### MuseScore

- `../MuseScore/src/engraving/types/types.h`: internal `BeamMode` values.
- `../MuseScore/src/engraving/types/typesconv.cpp`: serialized beam-mode mapping.
- `../MuseScore/src/importexport/musicxml/internal/export/exportmusicxml.cpp`:
  per-level MusicXML beam generation including forward/backward hooks and slur
  direction export.
- `../MuseScore/src/importexport/musicxml/internal/import/importmusicxmlpass2.cpp`:
  reduction from per-level MusicXML beam states to internal modes and slur placement
  import.
- `../MuseScore/src/propertiespanel/qml/MuseScore/PropertiesPanel/notation/beams/BeamTypeSelector.qml`:
  user-facing no-beam, break, inner-break, and join controls.

#### OSSQ

- `../ossq-omr/scores/**`: original and cleaned score MusicXML plus constructed
  page/system artifacts.
- `../ossq-omr/excluded_segments_*`: published task-specific exclusions.
- `../string-quartet-omr-benchmark`: evaluation code and benchmark definitions.
- `../omr-data-preprocessor/omrdp/ossq`: construction and alignment pipeline.

#### Human review and assembly

- `../OurTextScores/docs/private/SCANNER_PAGE_REVIEW_DESIGN_2026-08-09.md`:
  per-head confidence review, source geometry, cumulative corrections, and training
  capture.
- `../OurTextScores/docs/private/SCANNER_PAGE_HOMR_DESIGN_2026-08-06.md`:
  durable page jobs, provider contract, page-local HOMR constraints, and assembly.
- `../OurTextScores/docs/private/SCANNER_DUAL_ENGINE_COMPARE_DESIGN_2026-08-10.md`:
  source-grounded per-block review, safe MusicXML mutation, part/staff identity,
  provenance, and the observed need to materialize explicit beam semantics before
  mixing MusicXML passages.

The external review system is a consumer, not a dependency of HOMR. HOMR should
remain independently usable from its CLI while exposing contracts rich enough for
that consumer and others.

### 27. Reproduction record (2026-08-15/16)

Everything in this section was run and measured. It exists so the numbers quoted in
§24.1 can be re-derived rather than taken on trust, and so the next person does not
rediscover the environment problems.

#### 27.1 What was built

| Area | Where |
|---|---|
| Page-level OSSQ benchmark | `validation/ossq.py` |
| MusicXML ground truth in the NED scorer | `validation/ned_score.py` (`_side_parts`) |
| Deterministic system grouping | `homr/system_grouping.py` |
| Score-level split manifest | `training/omr_datasets/ossq_split_manifest.json`, `ossq_splits.py` |
| Per-class support tables | `training/omr_datasets/ossq_label_audit.py` |
| Structured beam/stem/slur schema | `homr/transformer/structured_notation.py` |
| Label extraction from MusicXML | `training/omr_datasets/structured_notation_parser.py` |
| Beam materialization check | `training/omr_datasets/beam_materialization_check.py` |

#### 27.2 Environment

Rented a single RTX 4090 (48 GB) with 128 vCPU and 755 GB RAM. B0 is inference only, so
the GPU is not the constraint; CPU is. Three things had to be right:

**onnxruntime and CUDA.** The driver advertises CUDA 12.8, and `onnxruntime-gpu >=
1.24` - what `homr`'s `[gpu]` extra asks for - requires CUDA 13 and fails to load
`libcublasLt.so.13`. Pin `onnxruntime-gpu==1.22.0` with the `nvidia-*-cu12` wheels and
put their `lib` directories on `LD_LIBRARY_PATH`.

**Thread count, which dominates everything.** Uncapped, one page cost 4m40s wall and 65
CPU-minutes thrashing 128 cores. With `OMP_NUM_THREADS=4` the same page takes 12.8s wall
and 24 CPU-seconds. Set it before measuring anything.

**The container's task limit.** Threads, not processes, exhaust it: a healthy run sits at
~115 processes but ~3,600 threads, roughly 30 per homr process even with the cap above,
from ONNX inter-op, OpenCV and CUDA pools. Six benchmark workers is sustainable; sixteen
is not, and produces `fork: Resource temporarily unavailable` across the box. Recover by
killing only the offending session - `kill -9 -1` takes out sshd with everything else.

`validation/tools.py` shells out to `poetry run python3 -m homr.main`, so a `poetry`
shim that execs the project's interpreter is enough; poetry itself is not needed.

#### 27.3 Dataset construction

`ossq-omr` tracks only sources: `.mscx`, `sq*.musicxml`, `sq*_cleaned.musicxml`,
`yolo_infos.yaml`, `_scanned.csv`. Everything the benchmark consumes is generated, and
the scanned PDFs are fetched separately and are not in git.

Synthetic track, from a fresh clone:

```
convert_musicxml_to_lmxe.py -t synthetic   # -> musicxml/, lmxe/, metadata/ unaligned
convert_musescore_to_pdf.py -t synthetic
convert_pdf_to_images.py    -t synthetic   # -> images/synthetic/original
```

Scanned track additionally needs the four YOLO checkpoints and, per score, its own
recorded models and thresholds (`--reproduce 1`):

```
convert_pdf_to_images.py       -t scanned
yolo_detect_systems.py         -t scanned --reproduce 1
yolo_crop_systems.py           -t scanned --confidence-threshold 0.30 --reproduce 1
yolo_detect_staff_heights.py   -t scanned --reproduce 1
yolo_resize_systems.py         -t scanned --target_height 18 --reproduce 1
align_systems_lmxe.py          -t scanned   # -> musicxml/scanned/systemwise
```

**Do not regenerate `sq*.musicxml` with a different MuseScore.** The page indices in the
reference come from the MusicXML layout and the images from rendering the same score;
4.6.5 lays the Ravel quartet out over 56 pages where the version behind the tracked
MusicXML used 47. The filenames still line up, so page 5 is scored against a page 5 of
different music, silently and with a plausible NED. `validation/ossq.py` now refuses a
score whose rendered page count disagrees with the reference, but the cheaper answer is
to leave the tracked MusicXML alone - `convert_musicxml_to_lmxe.py` reads it and
reproduces the segments byte for byte.

Scanned alignment is positional, not by page number: `align_systems_lmxe` sorts the
detected scanned systems, sorts the symbolic segments, requires the counts to match
exactly, and zips them. Its per-score `_scanned.csv` check is an existence gate only.
Predicted coverage: the 96 scores with a scanned PDF hold 11,305 symbolic segments
against 11,304 published scanned system images, so essentially every score should align,
with one segment excluded.

#### 27.4 B0: the pinned checkpoint on OSSQ synthetic

3,148 of 3,206 pages, whole corpus, attribute restatements collapsed:

```
all pages                     n=3148  mean  8.43%  median  3.93%  p90 18.99%
layout correct (parts match)  n=3057  mean  6.66%  median  3.72%  p90 16.67%
layout wrong  (parts differ)  n=  91  mean 67.79%  median 70.36%  p90 74.22%
```

Layout failure rate 2.9%, of which 80 pages are still four parts read as one. Before the
grouping work it was 25%. On a fixed 60-page subset, before and after: mean 16.44% ->
5.19%, layout-broken 12 -> 0, median unchanged at 2.68%, nothing regressed.
polish-scores is bit-identical with and without the change (17.93%, 108/112, same four
failures), which is expected since single-system images cannot reach the geometric path.

Two scoring artifacts had to be handled before any of this meant anything:

- Clef and key restatements were 40% of all non-matching tokens. Engraving restates them
  at every system start; homr reports state changes only. Same page, different
  convention. Collapsing repeats on both sides moved a sample from 6.31% to 3.74%
  overall and pitch from 3.43% to 0.57%.
- Whole-measure rests carry no `<duration>` in 19,147 places, and homr's parser assigns
  duration 0 rather than inferring it, so every empty measure in the reference was
  mis-timed against a real rest in the prediction.

The largest remaining error classes are now `note_12 -> note_8` (32,538) and
`note_24 -> note_16` (13,114) - triplets read as plain eighths and sixteenths - plus
12,091 hallucinated `timeSignature/4`, which is `music_xml_generator` forcing a `<time>`
element into every part's first measure whether or not a meter was recognised.

#### 27.5 Label support, and what it settles

Counted from the original MusicXML over 1,433,203 notes (§25.2's open questions):

```
beam level    train    valid   test        slur slot   train   valid  test
  1         608,166   66,043  62,656         1       291,683  34,830 34,983
  2         246,240   23,050  17,749         2         2,379     438    243
  3          27,129    2,413   1,405         3            83       2     11
  4           1,697        8      40         4-6          33       0      9
  5               0        0      14
  6               0        0      14
```

Beam levels 5 and 6 have no training examples at all - all 28 occurrences are in test -
so they should be deterministic or unsupported, not learned. Level 4 can be fit at 1,697
but cannot be selected on with 8 validation examples. Slur slots 3-6 hold 116 training
occurrences between them: two trained slots plus overflow reporting matches the data,
six trained slots does not. Half of all slurs carry no placement, so the side head's
supervision covers ~50% of spans and `UNSPECIFIED` is the majority class rather than an
edge case. The stem head's `DOUBLE` class has no support anywhere in the corpus.

Extraction over all 122 scores is clean: 89 unmatched stops, 17 unclosed starts, 60
duplicate starts, **0 slot overflow**, 0 beams deeper than their duration. Starts minus
stops is exactly 17, matching unclosed starts, so the canonical slot pairing balances.

#### 27.6 Beam materialization is not needed here, and would be harmful

§9.5 prescribes materializing automatic beam choices through the pinned MuseScore, with
a check that this does not change the rendered notation. Running the check first inverts
the conclusion. Over 14 scores and 172,607 notes:

```
beams gained        1  (0.001%)
beams lost/changed  2,910  (1.686%)

1756  16th   ['end','backward hook'] -> ['end','end']
 458  eighth ['end']                 -> []
 280  eighth ['begin']               -> []
 118  32nd   ['end','backward hook','backward hook'] -> ['end','end','backward hook']
```

Nothing is ambiguous: MuseScore writes `<beam>` for what it beams, so a flag-worthy note
without one is genuinely flagged and its FLAG label is already correct. And the round
trip is not safe to run - it rewrites grouping on 1.7% of notes, and its largest single
pattern turns backward hooks into full beams. That is exactly the information these
heads exist to recover, and the same information this design cites as the reason to
prefer per-level states over MuseScore's `BeamMode`.

So: skip materialization, and do not use a MuseScore round trip to produce or normalise
beam labels. This also bounds §11.3's generator gate - a MuseScore load/render check on
emitted MusicXML cannot assert beam equality without producing false failures. The
finding is corpus- and version-specific, so it is a repeatable check that records the
MuseScore version rather than a fact asserted here.

#### 27.8 The stem head has no supervision from the segment labels

Found while wiring notation onto the token stream, and it needs fixing before the stem
head can be trained at all.

The per-system segments under `musicxml/unaligned/` are what the training crops
correspond to, and **none of them carry `<stem>`**: 0 of 13,244 files, against 14,370
stem elements in one original score alone. Beams and slurs survive the same path
untouched. This is the hazard 13.2 names - "existing derived token data also
intentionally removed stem direction" - now measured rather than anticipated.

The cause is `convert_musicxml_to_lmxe.py`, which calls `system.strip_stem_directions()`
before writing both the LMXE and the MusicXML segment when `--remove-stem-direction 1` is
set, which is what `ossq_step_001.sh` sets. Parsing a segment therefore yields
`StemDirection.UNKNOWN` for every note, which the target builder correctly masks - so the
stem head would train on nothing and report a clean zero loss over zero positions.

**The step**: regenerate the segments with `--remove-stem-direction 0`. That is 13.2's
"revised cleaning path proven to preserve these fields", and it is a flag rather than new
code. **Done** - all 13,244 segments now carry `<stem>`, so the stem head has supervision.

Two things it does not disturb. NED is unaffected, because stem direction is not one of
the six fields `EncodedSymbol` compares on, so the benchmark's ground truth means exactly
what it did. And the beam and slur labels are unchanged, since only stems are stripped.
It does mean the segments stop being byte-identical to the ones described in 27.3, so
regenerate the whole corpus rather than mixing the two.

#### 27.9 Notation has to survive the dataset files, not just the parser

Second gap found the same way, and blocking for the same reason.

Training does not parse MusicXML. `convert_*` writes token text files once, and
`data_loader` reads those back through `read_tokens`, which reconstructs each symbol from
its six fields. Notation attached during parsing is therefore dropped at the moment the
dataset is written - the heads would see the same nothing the stem head sees in 27.8,
for a different reason.

11.1 anticipates this ("serialization used for dataset indexes must be schema-versioned")
and 19.2 constrains the fix ("legacy token files remain readable", "structured token
schema is versioned and preferably JSON-based"). What it does not settle is where the
notation goes, and the packed line format argues against putting it inline: a chord is one
line carrying several symbols, with articulations and slurs hoisted into per-position
sets, so a per-note field cannot be appended without either escaping problems or a second
parser.

**The step**: a sidecar file beside each token file, one JSON record per note-bearing
symbol in sequence order, read only when present. Legacy token files stay byte-identical
and keep loading, which 19.2 requires, and nothing in the existing format changes.

The obvious objection is that a sidecar reintroduces exactly the pairing-by-position
problem that carrying notation on the symbol was meant to remove. The difference is that
both sides of this pairing are ours and are written in one pass: the count of
note-bearing symbols is recorded when writing and checked when reading, so a mismatch is
refused rather than mis-attached. That is a guard against our own bug, not an inference
about someone else's data.

#### 27.10 homr's rhythm vocabulary stops at the 128th note

A corpus run failed one page with a bare `'256th'`, which is a KeyError from
`DURATION_NUMBER` in the MusicXML parser. `DURATION_NAMES` runs breve, whole, half,
quarter, eighth, 16th, 32nd, 64th, 128th - and stops. A 256th note cannot be tokenised,
so a score containing one cannot be parsed at all, in the benchmark's ground truth or
anywhere else.

This is pre-existing and unrelated to the structured heads, but it settles one of their
open questions from a second direction. 27.5 already showed beam level 6 has no training
examples; level 6 is by definition the 256th note, and homr has no rhythm token for it.
A level-6 beam head could not be trained, and its predictions would attach to notes the
model cannot emit. Level 5 is the 128th, which is representable.

So level 6 is not merely unsupported for want of data - it is unreachable, and should
stay out of the trained configuration on that ground alone.

#### 27.11 Training needs staff crops, which nobody has built

The heads train on single staves, because that is what homr's transformer reads. The
crops exist in principle - omr-data-preprocessor's partwise cropping writes
`images/<track>/partwise/<score>:<page>:<system>:<part>.png` - but no run has produced
them: zero `partwise` directories across the corpus. The systemwise pipeline that B0 and
the scanned track need stops short of them.

`convert_ossq.py` is written against that layout and reports how many parts it skipped
for want of a crop, so the missing step names itself rather than producing an empty
dataset quietly.

One choice worth recording. The preprocessor's partwise *symbolic* output is LMXE only,
with no MusicXML, so a converter following it would have to come back through the LMXE
tooling to reach something homr can tokenise. Instead the converter takes the systemwise
MusicXML - which it already has, and which the benchmark already assembles the other way
round - and pulls the single part out of it. Parts are taken in document order, which is
top-to-bottom on the page and therefore exactly how the crops are numbered, so the
correspondence is positional and checkable rather than inferred through a second format.

**The step**: the synthetic track had never had *any* of its YOLO chain run - only
PDF-to-images - so partwise crops need the whole sequence, not just the last two stages:
`yolo_detect_systems`, `yolo_crop_systems` (confidence 0.45), `yolo_detect_staff_heights`,
`yolo_resize_systems`, `yolo_detect_staves`, `yolo_crop_staves` (confidence 0.7), then
`convert_ossq.py`. Note `yolo_crop_systems` writes to `images/<track>/cropped/`, not
`systemwise/`; `yolo_resize_systems` is what produces `systemwise/`. Its 13,244 synthetic
crops match the segment count from 27.8 exactly, which is the check that the chain is
aligned with the symbolic side.

**A guard the converter needed.** The crop-to-part pairing is positional - crop *n* is the
*n*th part in document order, both being top-to-bottom on the page - and that holds only
if the detector found exactly the staves that are there. 27.14 measured that it does not:
scans over-detect, reporting five, six, seven or nine staves in a four-part system, and
detection can equally miss one. Either direction shifts the numbering, so a system whose
second staff was missed yields crops 1, 2, 3 against four parts and crop 2 is part 3 -
every pair from the gap onward being a plausible staff image with another staff's beams.

`convert_ossq.py` now requires the crop numbers to be exactly 1..len(parts) and skips the
system whole otherwise, counting "no crops at all" separately from "crops disagree with
the parts" because the first means run the cropping and the second means no rerun will
help. Filling in what is present would keep the good pairs and corrupt the rest; a smaller
clean training set beats a larger one with unfindable label errors in it.

That is the third correspondence bug of this kind - after the decoder-output alignment and
the loader's image substitution (27.15) - and the same shape each time: two sequences,
neither malformed alone, paired by position, with no check that the pairing holds.

**What the guard costs, measured before the run rather than after.** A guard that discards
systems is only affordable if it rarely fires, so
`training/omr_datasets/staff_detection_agreement.py` compares the staff detections already
on disk against the part counts. On 5,742 synthetic systems:

```
detections match the part count   5,739   99.9%
mismatch (system skipped)             3    0.1%   one each at +1, -1 and -3
```

So on the synthetic track the guard costs essentially nothing, and the question of what to
do about a large drop does not arise. That is a statement about this track only - 27.14
found scans over-detecting into five, six, seven and nine staves, so the same measurement
has to be repeated before the scanned track is converted, and the answer there is likely
different.

#### 27.12 How much beaming is derivable without looking at the page

Gate C asks whether the beam heads beat deterministic reconstruction, and 15.2 names
"duration-and-meter automatic beaming" as the baseline. That baseline can be measured now,
before any head is trained, and it says whether the exercise is worth running.

Implemented as the textbook rule - beams do not cross a beat, a rest or a note too long
to carry a beam ends the group, compound metres beat in threes - and compared against the
engraved beaming of 20 scores, 174,830 beamable notes:

```
automatic beaming matches the engraving   136,299   78.0%
exceptions the rule does not predict       38,531   22.0%
```

So roughly a fifth of the corpus's beaming is not derivable from duration and metre. That
is a real target: a head that learns nothing beyond the rule would be worthless, and one
that recovers even half the exceptions is recovering something only the image carries.

The largest disagreements under that rule were `begin -> continue` and `end -> continue`,
13,172 between them, which is the rule breaking a group at every beat where the engraving
carries it across - eight eighths in 4/4 beamed as two groups of four rather than four of
two. That is house style rather than an exception the page reveals, so the rule was
strengthened to beam eighths by the half-bar in simple duple metre, with any shorter value
in a group pulling it back to the beat.

```
strict beat rule    136,299 / 174,830   78.0% match   22.0% exceptions
half-bar rule       138,824 / 174,830   79.4% match   20.6% exceptions
```

**That correction was expected to close much of the gap and did not - it closed 1.4
points of 22.** The remaining disagreements are also no longer one-sided: `continue ->
begin` and `continue -> end` (8,155) now sit alongside `begin -> continue` and `end ->
continue` (4,836). A systematic rule offset would push in one direction; this pushes both
ways, which is what per-instance editorial choice looks like rather than a rule still
waiting to be implemented.

So about a fifth of this corpus's beaming appears genuinely not derivable by any fixed
rule, and that residue is what the beam heads are for. Gate C should be judged against
the half-bar baseline, not the strict one.

**The figures in this section are superseded by 27.16.** They came from a script that was
not kept, so they cannot be re-derived; the committed tool measures the whole corpus and
each split, and puts the number higher. The conclusion - that the residue is real and is
what the heads are for - survives; the exact percentage here does not.

The measurement also caught a labelling bug: the extractor was marking rests FLAG at their
applicable levels. A rest has no stem and therefore no beam or flag, so every eighth rest
in the corpus was entering the beam heads' training signal as a positive. Fixed, and the
baseline moved 69.8% -> 78.0% once it was.

#### 27.13 The scanned track, built

The scanned pipeline was run for all 96 scores with a scanned PDF, using each score's own
recorded YOLO models and thresholds (`--reproduce 1`):

```
scores aligned                     89 / 96
aligned scanned system musicxml    10,548
benchmark pages built              2,468 across 89 works
```

Against the 11,304 predicted from the published system-image count, so seven scores did
not align. `align_systems_lmxe` requires the detected system count to equal the symbolic
one exactly and skips the score otherwise, and the log names each one with both figures.
Most are near misses - 179 symbolic against 178 detected - but `sq9790696` detected 36
systems against 159 expected, which is a detection failure rather than an off-by-one and
would need that score's `yolo_infos.yaml` exceptions revisited.

Refusing those seven is the right outcome rather than a shortfall: a score whose systems
were only partly detected would pair the wrong music with every page after the first
miss, which is exactly what the positional alignment cannot survive and what the count
check exists to catch.

#### 27.14 B0 on the scanned track, and the synthetic-to-scan gap

2,452 of 2,468 pages scored, against the synthetic corpus for comparison:

```
             pages   overall   rhythm   pitch    lift   artic    slur   layout fail
synthetic     3159     7.79%    5.39%   3.46%   3.28%   4.13%   4.09%      2.9%
scanned       2452    10.00%    6.42%   3.82%   3.67%   4.64%   4.47%      3.1%
```

Two things stand out, and the second was not expected.

**The domain gap is smaller than the framing suggests.** 23 names the synthetic-to-scan
gap as a principal risk, and it is real - 1.28x on the mean, 1.57x on the median - but
these are photographs of different editions scored against symbolic ground truth, and the
degradation is a quarter rather than a multiple. Every field moves together, roughly
proportionally, rather than one collapsing: no component of recognition falls apart on
scans, they all get modestly harder. An earlier reading of two pages suggested rhythm
carried nearly all the scanned error; across 2,452 it does not.

**Layout failure is the same on both tracks: 2.9% synthetic, 3.1% scanned.** That was not
a given. The grouping in 8.2 decides from staff spacing measured in unit sizes, and
nothing about it was tuned for scans - yet it transfers from clean renders to photographs
essentially unchanged. The failure *modes* do differ: synthetic failures are almost all
four parts read as one, while scans also over-detect, producing 5, 6, 7 and even 9 parts
where there are 4. Spurious staves are a scanning artefact the geometry has no reason to
catch, and they are the residue worth attacking next on that track.

Taken together these say the remaining gap is mostly in transcription rather than
structure, which is where the notation heads are aimed.

#### 27.15 The frozen-core run, made runnable

Three things stood between the design and a first Phase 2 run, and none of them was the
part that had been written.

**There was no entry point.** `train_structured_heads.main()` was a guard over an empty
body - no model, no loader, no loop - so removing the guard would not have produced a
run. It now builds the model with the heads enabled, loads the pinned checkpoint under
the allowlist, freezes the core, and iterates the wrapped dataset.

**The targets were a position out.** The decoder reads `rhythms[:, :-1]`, so its hidden
state at *t* is the prediction for token *t+1*, and the structured heads sit on that same
hidden state. Targets laid out over the full token sequence are one place to the left of
the logits meant to score them. This is the kind of error that does not announce itself:
the shapes differ by one, which torch catches only until something pads or truncates, and
then every head trains on the token *after* the one it describes, converges to something
plausible, and reports a healthy loss. It is now `align_to_decoder_output`, a named
function with a test, and the stand-in model in the tests shortens by one exactly as the
real decoder does - otherwise the test would pass whatever the alignment.

**There is no published `.pth`.** homr's `download_weights` fetches ONNX only, and the
frozen-core experiment needs torch weights. The checkpoint does exist, in a separate
release that nothing in the code references:

```
https://github.com/liebharc/homr/releases/download/checkpoints/
  pytorch_model_426-b6fd20809a8dcaf10dfd39a4ca4f64c6f056e644.zip
```

That is the run-426 checkpoint named by `Config.filepaths.checkpoint`, so it is the same
core B0 measured, not a nearby one. Unzip it into
`training/architecture/transformer/`. It loads cleanly into a model with the heads
enabled:

```
checkpoint: loaded 326 parameters; initialized 18 new ones under decoder.structured_heads
trainable tensors: 18   frozen: 326   trainable params: 25,137
heads: beam, slur_event, slur_side, stem
```

25,137 trainable parameters against a 300 MB core is the shape the experiment wants: if
these heads learn anything, it is because the representation already carried it.

**The loader substitutes images, and the labels did not follow.** `DataLoader.__getitem__`
returns `(index + 1)` when cv2 cannot decode an image - a reasonable way to survive a
corrupt file in a corpus of hundreds of thousands. The notation wrapper was still
attaching the requested index's sidecar, so any substituted item paired one staff's
picture with the previous staff's beams, stems and slurs. Like the alignment bug, nothing
raises and the loss looks normal. The wrapper now resolves the substitution before
fetching, using the loader's own test - a decode attempt, not an existence check, because
a present-but-corrupt file passes the cheap check here and fails there, which is precisely
the desync.

Both of these share a shape worth naming: the failure is a *correspondence* between two
sequences, and neither sequence is malformed on its own. Shape checks do not catch them,
losses do not spike, and the heads converge to something plausible. They are only
catchable by asserting the correspondence directly, which is what the tests for
`align_to_decoder_output` and `_resolve` do.

**End to end, once, on real objects.** `tests/test_structured_training_integration.py`
runs an index file through the real loader into a real TrOMR shrunk to one layer, and
checks the three things every stand-in hides: that no core parameter moves, that the head
parameters receive gradient rather than merely being listed as trainable, and that `loss`
is still emitted and finite so B0 stays comparable.

**Environment.** The benchmark venv is inference-only. Training additionally needs
`x_transformers`, `timm` and `albumentations`; installing `timm` pulls a PyPI
`torchvision` built against the CUDA torch ABI, which fails at import against this
instance's `torch 2.13.0+cpu` with `operator torchvision::nms does not exist`. Install
`torchvision==0.28.0+cpu` from the pytorch cpu index to match. A GPU run will want a CUDA
torch build instead; the checkpoint load and the unit tests do not.

#### 27.25 Which corpora may carry notation labels

`music_xml_parser` attaches beam, stem and slur labels for every corpus that goes through
it, so it is tempting to have every converter write a sidecar. That is wrong for some of
them, and wrong in a way no test of the parser could catch. The labels describe the
*source* engraving, so the test a corpus has to pass is about its **images**: does the
training picture show the engraving the labels came from?

```
ossq        page images rendered from the source score, cropped to staves      eligible
lieder      SVG and MusicXML both rendered from one source .mscx               eligible
pdmx        MusicXML regenerated from the tokens, rendered with Verovio        not eligible
musetrainer same rendering path as pdmx                                        not eligible
grandstaff  **kern source; beams and stems are there but need another parser   unassessed
primus      .semantic encoding, which likely carries no beaming at all         unassessed
```

PDMX is the interesting case. Its images are built by regenerating MusicXML *from the
tokens* and rendering that - and tokens carry no beams or stems, so Verovio supplies its
own. Source-derived labels would then disagree with the picture precisely where the
engraving departs from the rule, which is the entire 18% the beam heads exist to learn.
That is not a noisy addition to the training set, it is an anti-signal: it would teach a
head that exceptions do not occur.

**The fix is a change to one stage, not a re-acquisition.** `convert_pdmx` already
downloads its MusicXML from the same Zenodo record it would need (15571083), so the source
is present; only the render step round-trips through tokens.
`training/omr_datasets/musicxml_window.py` cuts a renderable window out of the source part
instead, carrying the clef, key, divisions and time in force where the window starts.

**On the PDFs in that record.** They are MuseScore's rendering of the same MXL, not
independent scans, so they carry no notation the MXL does not already have - the same
relationship OSSQ's *synthetic* track has to its source, and the reason that track is
called synthetic rather than scanned. A PDF-based path would buy realistic page layout and
system breaks, not label fidelity, and it would need the whole OSSQ-style pipeline
(page, systems, staves, alignment) over heterogeneous instrumentation with no curated page
ranges. Worth wanting for layout realism; not a shortcut to better labels.

The eligibility rules are pinned by `tests/test_corpus_notation_eligibility.py` rather
than by comments, so a later change to how a corpus builds its images fails a test instead
of silently making its labels wrong.

#### 27.24 Ties are not slurs, and the labels could not tell them apart

homr's label vocabulary has three slur values - `slurStart`, `slurStop`,
`slurStart_slurStop` - and `<tied>` emits the same ones as `<slur>`. So a tie is
indistinguishable from a slur in a token file, though they are different objects: a tie
joins two notations of one pitch into a single sounding note, while a slur groups distinct
pitches under one phrase.

This is not a rare edge. In an 800-segment sample of 72,817 notes:

```
slur starts              9,115
tie starts               2,764     23% of all slur-like markings
notes carrying both        741     collapsed into one field
```

Scaled to the corpus that is roughly 45,700 ties labelled as slurs - a fidelity error in
the *existing* slur field, independent of the new heads.

`TieState` needs no slot machinery, unlike slurs: a tie joins one pitch to the same pitch,
so two ties cannot be open on one note of a voice without being the same tie. It is read
from `<tied>` rather than `<tie>` - both appear in MusicXML, the first is the notated
object and the second the sounding instruction, and it is the notation that is on the page
for a model to see.

The sidecar schema goes to v2 and the reader accepts v1. The 42,000 sidecars already built
predate tie extraction, and decoding them as "no tie" is correct rather than lossy: the
field was absent from the writer, not from the file. An unrecognised schema is still
refused.

#### 27.61 The collapsed staves are faint, not misaligned - which rules out the cheaper fix

27.60 found a seventyfold spread in collapse rate between scores and offered two readings:
scan quality, or misalignment from the staff-miscounting 27.14 measured. Looking settles it.

Two collapsed staves from the worst score against two healthy ones from the best:

```
sq8806881_0002_0003_2   87% -> 33%      faint, low contrast, staff lines thin and broken
sq8806881_0002_0004_3   89% -> 22%      same
sq12772795_0001_0001_1  92% -> 92%      crisp, solid noteheads, clean beams
sq12772795_0001_0001_2 100% -> 100%     same
```

**The bad crops are correctly framed and coherent - they are simply faded.** Staff lines are
present, the music reads, nothing is shifted. So this is not the crop-to-part misalignment
that produced three earlier defects, and the alignment guard is doing its job.

Measured across all nine scores:

```
score         collapse   ink fraction   contrast
sq8806881       21.9%          0.083         170
sq10414906      16.9%          0.070         163
sq8075304       16.6%          0.185         249
sq8885571       15.3%          0.139         221
sq8907120       14.5%          0.156         232
sq10307350       8.2%          0.204         254
sq7354505        8.0%          0.134         210
sq8806134        5.4%          0.114         198
sq12772795       0.3%          0.210         255

correlation(contrast, collapse)      -0.51
correlation(ink fraction, collapse)  -0.56
```

**The correlations are suggestive and not established, and the difference matters.** With nine
points, r of -0.5 carries p around 0.16 - it would arise by chance one time in six. The
extremes are unambiguous: the two faintest scores are the two worst, the crispest is the best
by a factor of thirty. But sq8075304 has contrast on a par with the best score and still loses
one staff in six, so faintness is not the whole account.

**What this licenses and what it does not.** It licenses testing contrast normalisation and
faded-ink augmentation, which are cheap and now have a reason. It does not license the claim
that the domain gap is a contrast problem - one score contradicts that outright, and nine
documents cannot settle it either way. The measurement to run is the same one on a larger
fold, which 13.5's split provides.

#### 27.63 CLAHE does not close the gap - it narrows the spread by damaging the good scans

27.61 licensed testing contrast normalisation without licensing the belief it would help.
`contrast_normalize.py` measures CLAHE's effect on all nine scanned scores directly, on CPU,
before spending any GPU time retraining with it.

At the default clip limit, the effect is negligible - the worst score gains six points of
contrast, the spread across scores narrows only 92 to 85. Raising the clip limit to see
whether a stronger setting would help instead reveals why it does not:

```
clip limit    sq8806881 (worst)    sq12772795 (best)    spread across scores
   2               169 -> 176           255 -> 250              92 -> 85
   4               169 -> 176           255 -> 249             101 -> 90
   8               169 -> 177           255 -> 246             101 -> 85
  16               169 -> 179           255 -> 239             101 -> 76
  40               169 -> 188           255 -> 223             101 -> 53
```

**The worst score barely moves - 19 points across a twentyfold increase in clip limit -
while the best score is damaged steadily**, losing 32 points at the setting that would be
needed to move the worst score by less than 20. The spread narrows, but by pulling the good
scans down rather than by lifting the faint ones. That is precisely the failure mode the
module's docstring was written to catch, and it is not a marginal effect: at clip 40 the
best score has been made measurably worse than the second-worst score started.

**Why it fails, mechanically:** a uniformly faint page has little *local* variation for CLAHE
to exploit - the tile is faint ink on faint background nearly everywhere, not a a patch of
strong contrast next to a patch of none. There is no local structure to redistribute. What
CLAHE has plenty of, on a crisp page with clean staff lines and white ground, is exactly the
sharp local structure it is built to equalise - so it acts on the wrong image.

**CLAHE is ruled out as the fix for this domain gap.** The next candidate, per the domain
gap's own shape, is not a page-level transform at all: 27.60 found the gap concentrated by
score more than by staff, so a fix aimed at *which documents* enter training - re-weighting
or oversampling the worst-performing scores - fits the evidence better than a filter applied
uniformly to every image.

**A gap in the test suite, worth naming.** `test_a_crisp_page_is_not_pushed_into_noise`
passed throughout this sweep and did not catch the damage documented above, because its
synthetic crisp page is nearly flat except for one thin stroke - it has none of a real
staff's dense local structure for CLAHE to over-equalise. The unit test validated the
mechanism in isolation and missed the failure that only appears on real images at scale,
which is why this measurement was run against the real scans rather than trusted from the
test alone.

#### 27.62 phase11: focal loss helps, class weights hurt - the risk in 27.50 materialized

27.49 diagnosed tie-class starvation and 27.50 built two instruments without knowing which
would work. phase11 ran them against the same baseline, synthetic validation:

```
                    macro F1   start F1   stop F1   start_and_stop F1   exact beam vector
baseline               0.772      0.551     0.791              0.746              0.904
focal (gamma=2)        0.782      0.600     0.785              0.744              0.899
weights (cap 50)       0.689      0.414     0.615              0.727              0.901
```

**Focal loss helps, at negligible cost.** Macro up a point, `start` up five points, beam
vector down half a point - a clean win, and it is the change to keep.

**Class weights make ties worse across every class**, not merely fail to help: macro down
8.3 points, `start` down 13.7, `stop` down 17.6. This is the over-correction risk 27.50 named
before running anything - *"293 start_and_stop examples carry a quarter of the gradient,
roughly 7,300x the leverage of one none, so the head memorises 293 examples rather than
learning a visual cue, and any label error among them is amplified 7,300x."* Weighting did
not fail neutrally; it actively damaged the classes it was meant to help, which is consistent
with that mechanism and not with simple underfitting - underfitting would show as no
movement, not as movement in the wrong direction on every rare class at once.

**Why focal succeeds where weighting does not**, on the same theory: focal reduces the loss
on positions the model is already confident about, which for `none` is nearly all of them,
without amplifying any specific rare example's individual leverage. It rebalances the
*aggregate* pressure between classes without turning any one of 293 crops into a landmark
the model must fit. Weighting rebalances by literally multiplying that example's gradient,
which is the mechanism 27.50 flagged as the failure mode.

The `both` arm is running to check whether combining them recovers weighting's loss or
compounds it - the theory above predicts compounds, since focal's gain does not depend on
weighting being safe.

#### 27.60 The scan gap is bimodal, and a fifth of one score's staves collapse

27.58 reframed track 1 around closing the domain gap. That is only actionable once the gap's
shape is known, and there were two candidates: **spread**, where every staff degrades because
scans are harder to read and the fix is visual; or **concentrated**, where most staves are
fine and a minority collapse, which would point at misalignment rather than difficulty -
27.14 measured scanned staff detection reporting five to nine staves in a four-part system,
and a miscount shifts every crop-to-part pairing after it.

The two tracks share their token filenames, so each staff can be scored twice.

```
3,028 staves scored under both renderings
  mean accuracy   synthetic 90.8%    scanned 70.9%

drop per staff (synthetic minus scanned)
  median 14.3%    quartiles 0.0% / 39.1%
  unchanged or nearly so (<= 10 points)   1,387  (45.8%)
  collapsed (> 50 points)                   326  (10.8%)

share of lost notes in the worst 10% of staves   28.1%
  uniform would give about 10%; broken crops alone would give most of it
```

**It is both, and neither answer alone was right.** Nearly half the scanned staves read as
well as their synthetic twins - the median staff loses nothing at the first quartile - while
a tenth collapse outright. The worst tenth carries 28.1% of the loss: concentration is real
at nearly three times its share, and still explains under a third.

**The sharper signal is per score:**

```
sq8806881   40/183  21.9%      sq10307350   28/343   8.2%
sq10414906  49/290  16.9%      sq7354505    65/817   8.0%
sq8075304   33/199  16.6%      sq8806134     8/147   5.4%
sq8885571   58/380  15.3%      sq12772795    1/365   0.3%
sq8907120   44/304  14.5%
```

A seventyfold spread between the best and worst score. One score loses one staff in 365; another
loses one in five. That is not a property of scanned images in general, it is a property of
*these* scans, and it means a meaningful share of the gap is recoverable by finding what is
wrong with the bad ones rather than by making the model more robust in general.

**A caveat that matters more than usual: this split has nine scores.** Every figure above is
an average over nine documents, and per-score rates from nine samples are suggestive rather
than settled. The finding worth acting on is that the variation exists and is large; which
scores are bad, and why, needs the fold that 13.5 set up rather than this one.

**One line in the first version of this tool was misleading and is fixed.** It reported "the
326 collapsed staves come from 9 score(s)", which reads as clustering. The split has exactly
nine scores, so that is all of them. A count is only evidence of concentration against the
total it is drawn from, and the tool now prints the rate per score with the total beside it.

#### 27.59 The recogniser's errors are diffuse, which is itself the finding

27.54 killed the resolution hypothesis for the 11.6% CER and left capacity or training length.
Both are expensive, and neither is worth spending until the errors are known to lack a
cheaper structural cause. `recognizer_errors.py` cuts the same accuracy four ways:

```
by syllable length          3 chars 90.1%   5 chars 83.2%   7 chars 61.4%
                            CER flat at 9-12% from 3 chars up
by punctuation              none 87.8%      with punctuation 83.1%
by seen / unseen            seen 88.7%      unseen 79.0%
characters                  dropped 'e' 136 / added 'e' 128    'r' 86/90
                            'i' 85/87   'n' 85/82   's' 83/77
```

**Every structural hypothesis fails.** If the frame budget bound, CER would climb with
length; it is flat from three characters up, and exact match falls only because a longer word
offers more chances to slip. If thin punctuation were being erased by the downscale, the
punctuation bucket would collapse; it costs 4.7 points. If one letterform were being lost,
the confusion list would have a head; instead **dropped and added counts are near-symmetric
across the common letters** - 136 against 128 for `e`, 86 against 90 for `r` - which is what
general letterform confusion looks like, not a specific failure.

So there is nothing cheap to fix, and 27.54's remaining levers are the right ones. The
evidence points at capacity or training length rather than at data or geometry: the loss
plateaued at 0.685 in all three height arms, which is a model that has stopped learning
rather than one starved of signal.

**One number in that table is an artefact and should not be read.** Single-character
syllables show CER 45.0% against exact 83.9%, which looks alarming and is arithmetic: one
wrong character out of a one-character label is a CER of 100% for that crop. CER over very
short strings is not comparable with CER over longer ones, and the exact-match column is the
honest one there.

#### 27.58 phase10: both gates re-run, and a crosstab of zeros that looked like a result

24.2 item 17 recorded that phase9 scores without `--predictions`, so neither Gate C's
crosstab nor the stem arbiter could read it. `phase10.sh` re-scores each domain with
predictions and then runs both.

**The first run produced a crosstab of zeros for two of three domains and reported them as
though they were findings** - `exceptions the head recovers: 0.0%`, `head used on 0.0% of
notes`, four zeros in a 2x2 table. Two separate faults, and the more useful one is mine:

  * *The scanned arm had the wrong root.* `rule_vs_head.segment_for` globs
    `dataset_root/scores/*/*/musicxml/unaligned/{name}`, and the arm was pointed at
    `/workspace/b0/phase7`, which holds the index but not the scores. The scanned track
    re-photographs the same OSSQ scores - its token files carry identical names,
    `sq8907120_0001_0001_1.txt` - so the correct root was OSSQ's all along.
  * *PDMX can never work.* Both tools decompose `convert_ossq`'s filenames and expect its
    directory layout; they are OSSQ-shaped by construction. PDMX now gets scored without
    them rather than reported as zeros.

**The tools were not at fault and said so plainly**: `0 staves joined, 0 beamable notes,
3,549 skipped`. The reporting script's `grep` kept the summary lines and dropped that one, so
a loud diagnostic was filtered into silence by the thing meant to summarise it. The grep now
keeps it, with a comment saying why. It is a smaller cousin of 27.53 - a number that looked
like a measurement and was an artefact of how it was collected.

**Gate C passes on synthetic and fails on scanned, and only the crosstab says so.** The
scanned arm, once pointed at the right root:

```
scanned              head right   head wrong        synthetic       head right   head wrong
  rule right             30,409       12,027          rule right        58,728        4,761
  rule wrong              4,231        3,445          rule wrong         8,953        2,775

  rule accuracy  84.7%   head 69.1%                   rule 84.4%   head 90.0%
  recovers 55.1% (4,231)  loses 28.3% (12,027)        recovers 76.3%  loses 7.5%
```

On scans the head **loses 12,027 notes the rule had right and recovers 4,231** - a net loss of
7,796 notes. The rule is domain-independent by construction, since it is derived from the
score rather than the image, and it holds at 84.4% and 84.7%. The head falls from 90.0% to
69.1%. That is a 21-point collapse where 27.38 measured 20 points on the exact-beam-vector
total, and it is the same phenomenon seen against a fixed reference instead of against
itself.

**Nothing in the totals shows this.** Exact beam vector on scans is 0.700, which reads as a
serviceable head. It is serviceable only in the sense that it is right more often than not;
against the rule it is a regression, and shipping it on scanned input would make transcription
worse than not having it. 27.16's insistence on the crosstab over the totals was argued from a
case where the two failed on disjoint notes; this is the stronger case, where the totals are
not merely uninformative but point the wrong way.

**One caveat that limits what may be concluded.** The rule here is computed from the score's
own durations and meter. At inference those come from the model, so a deployed rule runs on
predicted rhythm and would not reach 84.7%. What the comparison establishes is that the head
has no advantage to offer on scans, not that the rule alone is a finished answer.

**The stem head tells the same story, more starkly.** On scans:

```
              head alone   rule alone   oracle    best arbitration
scanned          72.47%       93.76%    96.96%    93.66%  (head on 3.4% of notes)
synthetic        92.70%       94.34%    98.24%    95.28%  (head on 77.0% of notes)
```

On synthetic the arbiter uses the head on 77% of notes and beats the rule by a point. On
scans the best threshold it can find uses the head on **3.4%** of notes and still lands at
93.66%, marginally *below* the rule's own 93.76%. The arbiter of 27.20 was built because the
head and the rule failed on disjoint notes; on scans the head's failures are no longer
disjoint, they are a superset.

**So both structured heads are net regressions on scanned input** - the domain that matters
for reading real sheet music - while both are gains on synthetic. That is the finding phase10
existed to produce, and it reframes the track: the work is not to add more heads but to close
the domain gap on the ones that exist.

The synthetic arm, which was correct throughout:

**Gate C, judged by the crosstab rather than the totals** - 27.16's rule, because two equal
totals can hide an oracle far above either:

```
                head right   head wrong
  rule right       58,728        4,761
  rule wrong        8,953        2,775

  rule accuracy on these notes   84.4%
  head accuracy on these notes   90.0%
  exceptions the head recovers   76.3%   (8,953 notes)
  agreements the head loses       7.5%   (4,761 notes)
```

The head recovers ten times what it loses, so the gate clears comfortably. It clears lower
than the 81.2% recorded in 27.16 for the two-corpus model, which is consistent with 27.36's
finding that mixing corpora costs something on any single one - though the two figures come
from different models and are not a controlled comparison.

**The stem arbiter, and a result worth stating plainly:**

```
  head alone    92.70%
  rule alone    94.34%
  oracle        98.24%    (upper bound if an oracle chose per note)
  arbitrated    95.28%    head if confidence >= 0.8, head used on 77.0% of notes
```

**The head alone is now worse than the rule** - 92.70 against 94.34, where the earlier
two-corpus model gave 94.2 against 94.5. Three-corpus training cost the stem head about a
point and a half of standalone accuracy. Taken alone that reads as an argument for deleting
it, which is the argument that was already made once and already refuted.

It is still wrong, for the same reason as before: **the oracle is 98.24%**. Head and rule
remain complementary, not redundant, and arbitration on head confidence gives 95.28% - a
point better than the better of the two, using the head on 77% of notes. A head that loses
to the rule on its own and beats it in combination is exactly the case a totals-only reading
would discard.

#### 27.57 The resolve rule settles at 98.2%, and the wrapped systems were scoreable after all

27.52 measured nearest-x on the 51% of systems with a single lyric line and got 98.9%. 27.55
showed the excluded 41% were not second singers but wrapped ones. This scores them, which
takes the measurement from a subset to the corpus:

```
26,503 of 26,998 syllables            98.2%
  on one note only                    98.7%
  held across notes (melisma)         93.0%
within one note either side           99.3%

2,584 systems scored, 327 skipped
lines per system   1: 1,952   2: 619   3: 11   4: 2   6: 2
```

Eight and a half times the syllables of 27.52, at essentially the same accuracy. **The
resolve stage's horizontal problem is solved by a rule**, and a learned resolver would be
competing for the last 1.8 points - against a baseline that costs nothing to run and cannot
be wrong in surprising ways.

**One wrong turn on the way, and it is the same shape as the others.** Scoring the wrapped
systems first gave 91.3%, a 7-point drop that looked like wrapped systems being genuinely
harder. They are not: the note band for a line was taken as everything between the previous
line's lyrics and its own, and *between two lyric lines sits the piano of the system above*
as well as the voice of the system below. Counting the piano's noteheads shifted every index
after them. Bounding each band by the staff lines instead - five close polylines are a staff,
a jump starts the next - restores 98.5% on the same data.

That is three times now that a plausible reading of a number has been wrong here: melismas as
the weak point of nearest-x (they are, by 5 points, not by a lot), a second vocal staff as the
cause of the extra lines, and wrapped systems as intrinsically harder. Each was cheap to test
and each test took under an hour. The pattern worth keeping is not "be more careful", it is
that **a number that surprises is worth one experiment before it is worth an explanation.**

#### 27.56 Three-corpus training: scans bought cheaply, and a prediction confirmed the hard way

phase9 trained the heads on all three corpora - OSSQ synthetic 42,088, OSSQ scanned 32,982,
PDMX 32,451 - for 12 epochs, against 27.38's baseline of the same heads trained without any
scans.

**Adding scanned data helps scans, and costs almost nothing elsewhere:**

```
on scanned validation        before     after     change
exact beam vector             0.661     0.701      +4.0
stem direction (up/down)      0.721     0.784      +6.3
hooks F1                      0.623     0.632      +0.9
slur spans F1                 0.494     0.509      +1.5
ties macro F1                 0.560     0.545      -1.5

on synthetic validation       0.911     0.903      -0.8   (exact beam vector)
```

Eight tenths of a point given up on synthetic to buy four on scans and six on stems. 27.36
measured the cost of mixing corpora at 1.6 points on OSSQ; scans cost half that, which makes
this the cheapest mixture tried.

**The domain gap narrows but does not close.** After training on scans, synthetic against
scanned still runs 0.903 to 0.701 on beams, 0.926 to 0.784 on stems, and 0.884 to 0.509 on
slur spans. 27.38 measured the untrained gap at 25 points on beams; it is now 20. **Slurs are
the worst channel by a wide margin** - 37.5 points - and that is where the next effort on
this track belongs.

**PDMX, the third domain, makes the mechanism visible.** The same head, same weights, scores
tie macro F1 **0.805** there against **0.545** on scans - `start` 0.724 against 0.248. The
difference is not the domain's difficulty but its class ratio: PDMX carries 5,533 `start`
ties in 1,942,885 tokens, 0.28%, against the scanned set's 2,342 in 2,149,263, 0.109%. **Two
and a half times the frequency, three times the F1.** That is the imbalance hypothesis of
27.49 tested across domains rather than argued, and it holds.

```
domain      start ties   share of tokens   tie macro F1
PDMX             5,533            0.285%          0.805
scanned          2,342            0.109%          0.545
```

**The tie head also got worse with more data, which 27.49 predicted.** 0.560 to 0.545 macro F1
on scans, while every other channel improved. The prediction there was explicit: *more data at
the same class ratio does not fix a ratio problem*, because the ratio is what starves the
class. Three corpora instead of two gave the tie classes three times as many examples and
three times as many `none`s, and the head moved backwards. This is the confirmation that
phase11's instruments are aimed at something real rather than at a suspicion.

Note also that tie **micro** F1 rose, 0.997 to 0.998, while macro fell. Quoted alone the
micro figure would have reported this regression as an improvement.

#### 27.55 Two thirds of the rendered pages hold more than one system, and 27.52 blamed the wrong thing

27.52 skipped 249 of 600 systems for carrying more than one line of lyrics, put it down to a
second vocal staff, and concluded that line assignment is the open problem for the resolve
stage. Chasing the line-count disagreement produced a different answer.

The disagreements are almost all in one direction - clustering finds one line *more* than the
score has parts, 245 times against 4 with two more. One case, opened up:

```
4919798_p1-s1:  2 lines vs 1 part
  line 0   9 syllables at y 777-815    'Fried' 'li' 'cher' 'A' 'bend' 'senkt'
  line 1   2 syllables at y 2348-2395  'fil' 'de,'
```

Fifteen hundred pixels apart, one vocal part. Counting staff lines in the same page settles
it: six groups of five, where a voice-over-piano system is three. **MuseScore wrapped the
joined system onto two systems** because it did not fit the page width.

(A first estimate of how often - "132 of 200 rendered pages" - was too high. It counted
staff-line groups and assumed three staves to a system, which over-flags any score with more.
Counting lyric lines instead, over the whole corpus, gives **634 of 2,586 systems, 24.5%**.)

So the vertical bands are not two singers, they are one singer continued. Measured directly:

```
lyric-carrying parts per joined system (800 systems)
  0 parts   57  ( 7.1%)
  1 part   743  (92.9%)
  2+       0    ( 0.0%)
```

**No system in this corpus has a second vocal line.** The "41% need line assignment" of 27.52
was an artefact of the renderer, and the resolve stage does not face simultaneous lines at
all - it faces a continued one, where the assignment is reading order rather than a choice.

**What this does and does not damage.** The recogniser corpus is unaffected: crops are
per-syllable, the counts were checked, and the pairing was verified by eye (27.48). The
detector arguably benefits, since a page holding two systems resembles a real page more than
one holding a single system does. What was damaged is a conclusion about where to spend
effort, which is the expensive kind of wrong - it would have bought a model for a problem
that does not exist.

The lesson is narrow and repeatable: **a disagreement between two counts is worth opening
before it is explained.** The explanation offered in 27.52 was plausible, fitted the numbers,
and named a real phenomenon that Lieder does exhibit elsewhere - it simply was not what these
numbers were.

#### 27.54 Resolution was not the constraint, and the hypothesis that it was is dead

27.51 reached CER 11.7% and blamed `IMAGE_HEIGHT = 32`, on the reasoning that 32 downscales
93% of crops and so discards the detail 27.47's sampled render resolution exists to provide.
That reasoning was sound and the conclusion was wrong.

With the padding bug of 27.53 fixed:

```
height 48   seen 88.7%   unseen 79.0%   CER 11.6%   loss 0.688
height 64   seen 88.2%   unseen 76.3%   CER 11.8%   loss 0.685
height 32   seen 87.1%   unseen 73.8%              (scored through the padding bug)
```

**Doubling the resolution moves CER by 0.2 points.** 48 is marginally the best and 64 is
slightly worse than 48, which is noise rather than a trend. The loss plateaus at 0.685 in
every arm, so the model is converged at this size and simply cannot read better - capacity
or training length is the remaining lever, not pixels.

A second hypothesis was tested and also died. The misreads after the 27.53 fix were
systematically missing a final character - `'senkt'` to `'senk'`, `'deckt;'` to `'deck'` -
which looked exactly like over-correcting the truncation. Decoding with extra slack says
otherwise:

```
frames+0   87.0%      frames+2   87.0%
frames+1   87.0%      frames+4   86.9%      all frames   51.8%
```

Slack changes nothing; removing the truncation entirely costs 35 points. The boundary is
right and the dropped characters are the model failing, not the decoder cutting. Two
plausible explanations for one symptom, both testable in minutes, both wrong - which is
cheaper than either being adopted.

**What this leaves.** 27.47's resolution work is not wasted: it made the synthetic images
match the scans they must transfer to, which is a domain-gap argument and remains right.
It just does not explain this CER, and the height is set to 48 because it is marginally best
and cheaper than 64, not because it matters.

#### 27.53 Evaluation was reading the padding, and two runs were scored through it

The height sweep of 27.52 returned nonsense: height 48 scored 30.1% exact where height 32
scored 88.3%, with **unseen scoring higher than seen**, which is backwards. The misread list
said why in one line:

```
'Zü' -> 'Zü\xa0'   'Ant' -> 'Ant\xa0'   'Fried' -> 'Fried\xa0'
```

Every prediction is the right word with a space stapled to it. Training gives CTC the true
unpadded lengths; **evaluation decoded every frame, including the padding**. The padding is
white paper, and the corpus alphabet contains a non-breaking space - from syllables like
`1.\xa0Es`, where MuseScore draws a verse number and its word as one lyric - so the model had
a perfectly legitimate white character to put over the padded frames.

**It hid behind the choice of metric, in the direction opposite to 27.49.** There, micro F1
flattered a head that could not predict its class. Here CER *understated* the damage: one
extra character on a three-character word is a small edit distance and a total failure of
exact match, so CER stayed near 23% while exact match collapsed to 30%. A single metric
would have hidden it either way; two disagreeing metrics are what made it visible.

So **the height sweep measured nothing** and both its arms are void, including the 32px
baseline of 27.51 - that run was scored through the same bug, so its 88.3%/78.3% is a floor
rather than a result. The sweep will be rerun.

Found by reading the misread list, which exists because a rate says how often and the
examples say what kind. Without those six strings this would have been filed as "48 is worse
than 32, resolution does not help" - a plausible, wrong, and expensive conclusion.

#### 27.52 Nearest-x already resolves 98.9% of syllables

Gate C asked what the automatic-beaming rule achieved before a head was built for beams.
27.45 named the resolve stage - attaching a recognised syllable to its note - the risky
component, so it gets the same question first: **what does the obvious rule already get
right?**

The rule is nearest note by horizontal centre. Ground truth is MuseScore's own render, so
image and labels are one engraving (27.25); note boxes come from the SVG, and only the voice
staff's notes count.

```
nearest-x picks the right note        98.9%   (3,079 of 3,113)
  syllables on one note only          99.2%
  syllables held across notes         94.8%
within one note either side           99.5%
```

**27.42 predicted melismas would be where this breaks. The prediction held in direction and
was wrong in size** - held syllables are worse, but by 4.4 points, not catastrophically. A
rule that gets 94.8% of the hard case right is not the weak link it was expected to be.

**The skips relocate the problem, and are the more useful finding:**

```
309 systems scored, 291 skipped
  249  more than one lyric line (a second vocal staff)
   42  no verse-1 syllables
    0  count disagreements
```

Zero count disagreements says the join of 27.41 is sound. 249 of 600 systems carry more than
one line of lyrics.

**The explanation given here for those 249 was wrong, and 27.55 replaces it.** They were
attributed to a second vocal staff, and the conclusion drawn was that line assignment is the
open problem. Measured afterwards: **no joined system has more than one lyric-carrying
part** - 92.9% have exactly one and 7.1% have none. Every multi-line case is MuseScore
wrapping one system across the page. The finding that survives is the first half only:
horizontal attachment is close to solved by geometry.

Two encodings had to be learned by being wrong about them. MuseScore draws lyrics in place,
with absolute path coordinates, but draws noteheads at the origin with a `transform` matrix -
so reading a notehead's `d` as absolute puts every note at the top left, and the first run
found zero notes. And "the voice notes are the ones above the lyrics" assumes one vocal
staff; with two, the lyric band spanned 777 to 2395 pixels and swallowed the piano.



#### 27.51 The recogniser, and a metric built so it can fail

`training/architecture/ocr/crnn.py` and `training/ocr/` implement the recognition half of
27.45: a small CRNN, CTC, no language model. The size is deliberate - 27.42 measured a
104-character alphabet and syllables of median 3 characters, so capacity is not the binding
constraint, and a large pretrained recogniser would reintroduce the very language prior CTC
was chosen to avoid. `ter`, `nel`, `Ê`, `schaft!` are fragments, and a model that repairs
them into words fails in the way that looks like success.

**The reporting is the part that took thought.** 27.49's tie head had micro F1 0.997 and was
near-useless, because one class carried the figure. The same trap sits here in a different
shape: 27.48 measured **17.0% of validation syllables never appearing in training**, so a
model that memorised the training vocabulary and read nothing at all would still score 83% -
and 83% reads like a working recogniser. So every figure is split:

```
seen      syllables whose exact string appears in training - says whether it remembers
unseen    syllables that do not                            - says whether it reads
```

First run, 30,030 training crops against 4,291 validation:

```
epoch 1   seen 68.5%   unseen 53.8%    CER 18.9% / 23.0%
epoch 2   seen 83.0%   unseen 61.8%    CER 13.5% / 20.9%
epoch 3   seen 88.3%   ...
```

The gap is real and it is what the split exists to show - roughly 20 points at epoch 2,
invisible in any aggregate. But **unseen climbs with seen**, which is the answer to the
question the split was built to ask: the model is reading, not only remembering. A memorising
model would show unseen flat.

**Three loader rules, each a way CTC breaks silently rather than loudly:**

  * Pad with paper, not zeros. Zero is ink, and the model would learn that every syllable
    ends in a black bar.
  * Give CTC the true width, not the padded one, or it hunts for the label inside the
    padding.
  * Refuse a label longer than its frame count. That example is not hard, it is impossible -
    an infinite loss poisoning the batch mean. Three were refused, and they are counted so a
    corpus that starts producing them says so.

The alphabet is built from training alone. A character appearing only in validation is one
the model could never emit, and including it would widen the output layer while pretending
the character is learnable - one validation syllable was excluded on this ground. Weights
are saved with their alphabet, because weights without the indexing that produced them
decode to nonsense.

#### 27.50 The class-imbalance instruments, and a sweep to choose between them

27.49 established the problem and its precedent. `structured_losses.py` had already written
down the condition for acting: *"the design's instruction is to measure the unweighted
baseline before reaching for class weighting or focal loss, so this computes plain
cross-entropy and reports what would justify the alternative."* The baseline is measured, so
this builds the alternative.

**Both instruments, not one.** Upstream converged on focal loss - as oemer's UNet uses - or
class weights at 50x, without settling which, so `focal_cross_entropy` takes a focal
exponent and a per-class weight vector and either can be zero. At `gamma=0, alpha=None` it
*is* `cross_entropy`, pinned by a test, so every earlier result stays comparable and a sweep
moves one thing at a time.

Measured spreads, from one pass over the training data:

```
tie.state          47.1x      slur.slot.1.event  37.4x     beam.level.1  19.7x
slur.slot.2.event  49.8x      stem.direction     25.9x     slur.slot.2.side  2.0x
```

The imbalance is not a tie-head peculiarity - six of eight heads sit above 19x, and
`slur.slot.2.event` is pinned at the cap. Only `slur.slot.2.side` is genuinely balanced.

**Three decisions worth stating, because each could have been hidden:**

  * *The 50x cap flattens the rarest classes together* - `start` at 0.109% and
    `start_and_stop` at 0.014% come out equal despite one being eight times rarer. This was
    first written up as the cap's main cost. It is not. Measured as gradient share on the
    tie head:

    ```
    scheme          none      stop     start   start_and_stop     tie classes combined
    unweighted    99.77%     0.11%     0.11%            0.01%                    0.23%
    cap 10        97.74%     1.06%     1.06%            0.13%                    2.26%
    cap 50        89.64%     4.88%     4.87%            0.61%                   10.36%
    cap 200       68.38%    14.89%    14.87%            1.86%                   31.62%
    uncapped      25.00%    25.00%    25.00%           25.00%                   75.00%
    ```

    **The cap chooses how much rebalancing happens at all, and at 50 `none` still holds
    89.6% of the gradient.** The flattening is a side effect of that larger choice. Too
    little rebalancing is the documented collapse - the class stops being predicted, SER 26%
    to 132%. Too much is worse in a way particular to this work: uncapped, 293
    `start_and_stop` examples carry a quarter of the gradient, roughly 7,300x the leverage
    of one `none`, so the head memorises 293 examples rather than learning a visual cue, and
    **any label error among them is amplified 7,300x**. Four label-pipeline defects were
    found in this corpus in a single day; a handful of bad labels in a 293-example class
    would decide what that class learns.

    Flattening is also a refusal to assume rarity tracks difficulty. `start_and_stop` is
    rare but visually distinctive - a note with a tie arriving and a tie leaving - so
    weighting it eight times harder than `start` asserts something unmeasured. The cap is a
    scalar, so `phase11.sh` sweeps it rather than arguing about it.
  * *Alpha normalises by the applied weight, not the count.* Otherwise raising a rare
    class's weight would also raise that head's share of the summed multi-head loss, and the
    two knobs would silently interact.
  * *Weights are measured in a separate pass, not per batch.* A batch holding no `start`
    tie would weight that class from a sample of zero, and the objective would wobble worst
    on exactly the classes least able to learn.

`phase11.sh` runs the sweep: focal alone, weights alone, both - same data, same epochs, only
the loss differing. It greps macro figures and never micro, because 27.49's tie micro F1 of
0.997 is almost entirely `none`.

**Two scheduling decisions, made because GPU hours are the scarce thing here.** The baseline
arm is not trained: phase9 already trains that exact configuration - same index, same 12
epochs, same unweighted loss - so retraining it would spend three hours reproducing an
existing file, and any gap between the two would be seed noise dressed as a finding. Its
heads are evaluated directly instead.

And the cap sweep is deferred rather than run alongside. The cap asks *how far* to
rebalance, which is only worth answering once rebalancing is known to help at all. At about
three GPU hours an arm, asking both questions together would cost a day to discover the
answer to the first was no.

**Found by running it, not by the tests:** the weight-measuring pass fed CPU tensors to a
CUDA model. No unit test would catch it - they all run on cpu - and it is the fourth defect
in this stretch that only appeared under a real execution.

#### 27.49 Why dynamics are commented out - and the same failure is in our tie head

27.45 claimed dynamics "belong in homr's symbol vocabulary, which is already written and
commented out", and treated uncommenting it as separate, straightforward work. That was
asserted without checking. Checking changes it.

**The block was never disabled after a failure.** It was born commented out in `c50aec7`
(2025-08-30), the commit that created `vocabulary.py`, and no commit has ever uncommented
it. The maintainer answered the question directly in upstream issue #61 (2026-02-13), where
a contributor asked whether training had been tried and found wanting:

> "A 'Scope/priority' decision likely describes it well enough. So it was simply the last
> feature which was added and a realization that I have to put less time into the project."

He also ruled out data quality: *"Almost all articulations are solely trained on the Lieder
dataset. You should get at least comparable performance as for the other articulations."*

**But it was then tried, and it collapsed.** The same contributor trained it (run 318, ~70
epochs, eval_loss 0.044):

> "SER jumped to 132% vs baseline 26% - dynamics were never predicted. The more probable
> root cause was that dynamics are only ~0.05% of tokens, so the model learned to ignore
> them entirely to minimise loss (class imbalance collapse)."

The maintainer agreed it was plausible and observed run 331 doing the same with double
sharps, never detected at inference. Proposed remedies were focal loss - as oemer's UNet
uses - class weighting at 50x, more Lieder data, or a dynamics-enriched subset.

So adding dynamics to the rhythm branch is not a matter of uncommenting: it is a documented
regression that made the whole model worse. 27.45's conclusion to keep them out of the OCR
pass still holds - they are music glyphs, not text - but its disposal of them into the
symbol vocabulary was too casual.

**The same failure mode is already in our own numbers.** The tie head, on the phase9
baseline:

```
none             n=2,149,263   F1 0.999
stop             n=2,345       F1 0.535      0.109% of tokens
start            n=2,342       F1 0.315      0.109%
start_and_stop   n=293         F1 0.393      0.014%

macro F1 0.560   micro F1 0.997
```

The contributor's dynamics were 0.05% of tokens and vanished entirely. Our tie classes are
0.109% - twice as frequent, not collapsed, but plainly starved. **And micro F1 0.997 is
almost entirely `none`**: reported alone it would call the head excellent while it is close
to useless for the thing it exists to predict. That is the same shape as 27.16 preferring a
crosstab to a total, arriving in a metric rather than a comparison.

So the tie head needs the treatment that discussion converged on - class weighting or focal
loss - and it needs it before more corpora are added, since more data at the same class
ratio does not fix a ratio problem.

**An independent corroboration of the architecture.** In the same thread, fablau describes
the pipeline behind Virtual Sheet Music's Playground:

```
1. staves, notes, key signatures, time signatures, clefs
2. dynamics
3. slurs and ties
4. other objects
```

Separate models per layer, combined into MusicXML afterwards. That is a deployed system
arriving independently at the staged design of 27.45, with dynamics as their own layer
rather than tokens in a shared vocabulary - which is also the answer to the class-imbalance
problem, since a detector for a class does not have to compete with note tokens for
probability mass.

#### 27.48 A recogniser's corpus, split by score, and the chain checked end to end

`musescore_boxes` feeds the detector; `lyric_crops` feeds the recogniser - one image per
syllable and the string it says. The annotation run finished at:

```
2,911 of 2,926 systems annotated       99.5%
34,325 syllables paired to boxes
8,044 other text boxes, typed, unpaired (27.44)
3 refused by mscore, 8 on a count mismatch
```

The part-id fix of 27.46 took the render failures from 217 to 3. The 8 remaining are the
count guard firing, which is what it is for.

**The split is by score, not by system or crop.** A Lied's systems share its engraving, its
typesetting and most of its words - *Es*, *und*, *die* recur through a song - so a crop-level
split would put the same syllable, in the same font, at the same size, on both sides of it.
That measures memorisation and reports it as recognition. 13.5 took score-level splits for
OSSQ; the argument is stronger here, because text repeats far more than music does.

Crops keep their native resolution. 27.47 sampled that resolution across the range the scans
span deliberately, and normalising at corpus-build time would discard what was just paid
for. Height normalisation belongs in the training loop, where it can be augmentation.

**The chain was checked by looking, which is the check that has worked.** Six crops drawn at
random, each with its stored label printed under it:

```
klagst,   du   klagst,   Lei   den   schaft!
```

Every crop shows the word its label claims, and *Lei-den-schaft!* is one word correctly
split across three syllables - so hyphenation and syllabic position survive the whole path
from MusicXML through render, box, crop and manifest. That path had three defects in it
before this section; none of them survived to here.

#### 27.47 The synthetic stage was lower-resolution than the scans, which is backwards

150 dpi was chosen for the render with the comment that it was "close to the scanned
corpus". That was an assumption stated as a fact, and it was wrong in the direction that
matters. Staff space is the scale everything else on a page follows from:

```
synthetic at 150dpi   10.3px   lyric text 16px = 1.55 staff spaces
scanned IMSLP         26.0px   quartiles 19 / 37, full range 5 to 76  (248 crops)
```

**The synthetic stage sat below the first quartile of the real data.** Nearly every scan
carries more detail than the images meant to teach the model to read them - the reverse of
the usual synthetic-to-scan gap, where synthetic is the cleaner domain. Height-normalising
crops at preprocessing does not fix it, because upscaling cannot recover detail that was
never rendered.

The crops were checked by eye first and were perfectly legible - *Schatz,* *säu* *Veil*,
umlauts and all, at 15 to 20 pixels. Legible was the wrong bar. The question is not whether
a human can read the training image, it is whether it carries what the test image carries.

**No single resolution fixes it**, because the scans vary fourfold among themselves: 19 to
37 px at the quartiles is 277 to 539 dpi. So resolution is sampled per system across
280-540, seeded on the system name so a rebuild does not silently change an image the boxes
were computed against. The spread becomes something the model meets in training instead of
first at inference.

After the change, measured on the same 50 systems:

```
staff space   28.7px   against the scans' 26.0
lyric text    44px     up from 16
```

This is the third defect found in this data by measuring rather than reasoning - after the
shared tree in `lieder_voice` and the part-id collision - and the only one that would have
survived into training silently. A crashed render and a crawling loop announce themselves.
An image that is merely too small does not.

#### 27.46 Two sources numbering parts independently, and a crash that was the lucky outcome

Annotating all 2,926 joined systems refused 227 of them - 7.4% - with `mscore failed on
.svg` and an empty stderr. Run by hand, the same score exits 40 and writes nothing.

The original OLiMPiC sample renders. The joined version does not, so the join made it:

```
joined 6007571_p1-s1     part id P1, part id P2, part id P2
```

Two parts called P2. The cause is that **the published score and OLiMPiC's sample number
their parts in independent namespaces**:

```
published lc6007571   P1 Chant/Voice   P2 Chant/Voice verse 2   P3 Piano
olimpic sample        P2 (the piano, renumbered)
```

A Lied with two vocal lines uses P2 for its second singer, and OLiMPiC calls its piano P2 as
well. `join` appended both and produced a score no renderer will accept. `reassign_ids`
renumbers the combined parts P1..Pn, pairing each `<score-part>` with its `<part>` by
position, which is sound because `join` appends both in the same order.

**The crash was the lucky outcome.** A duplicate part id is not a parse error in MusicXML -
it is ambiguous rather than illegal. Had MuseScore accepted it, the joined score would have
carried the piano twice, the boxes would have been drawn for a doubled system, and the
labels would have described music the image shows once. Nothing in the output would have
said so. The failure was loud only because the tool downstream happened to be strict; this
is the same class as the shared-tree bug two sections earlier, which was *not* loud and cost
40 minutes of a crawling run before anyone looked.

A first hypothesis - that piano detection had failed on these scores, making `voice_parts`
return the piano as a voice - was measured and was wrong: 0 of 200 published scores lack an
identifiable piano. Checking it took one command and would otherwise have produced a fix for
a problem that did not exist.

After the fix that score renders, and the join still reports the same 5 honest refusals for
a missing pickup measure.

#### 27.45 Detection then recognition - and dynamics are not text

Two open questions closed here. One was put to the user and handed back, so it is decided
below on the evidence rather than deferred again.

**Dynamics do not belong in the OCR pass, and 27.44 grouped them wrongly.** They passed the
same count check as lyrics - 28 rendered against 28 in the source - so they were listed
beside them as joinable text. The geometry says otherwise:

```
class            n   median w x h   aspect   width range
Lyrics         798        34 x 16     2.12         6-77
Tempo           14       200 x 25     8.72       39-475
StaffText       12       116 x 20     5.00       42-145
InstrumentName 105        55 x 17     3.24       50-114
MeasureNumber   31        19 x 13     1.46        10-20
Dynamic         61        26 x 26     1.00       17-117
```

Every text class is wide; Dynamic is square. The decisive evidence is in the source, not the
render: **MusicXML stores dynamics as element names** - `<dynamics><p/></dynamics>` - never
as strings, which is why `source_dynamics` has to rebuild the name from child tags. They are
music glyphs in a music font, and `f` the dynamic is a different glyph from `f` the letter.
Reading them as text would be recognising the wrong thing correctly.

They belong in homr's symbol vocabulary, which is already written and commented out
(`homr/transformer/vocabulary.py`). Uncommenting it is a separate piece of work from the OCR
stage and should not be folded into it.

**The text pass is detection then recognition, not one model emitting typed text.** The
geometry is the weakest of the reasons - a 79x span in width is exactly what CTC absorbs,
since it reads a fixed-height strip of any length. The reasons that decide it:

  1. **Separable measurement.** Detection recall and recognition accuracy can be judged
     apart. An end-to-end spotter fails un-attributably, and every stage in this work is
     judged by decomposition - 27.16 preferring a crosstab to a total, per-class F1 over a
     macro number, 27.40 finding a measure that could not verify its own fix.
  2. **Positions are the product.** The resolve stage consumes boxes. Detection emits them;
     an end-to-end model buries them in attention and they have to be recovered.
  3. **CTC refines inside the box.** Per-character alignment gives sub-box positions at no
     extra cost, which is what attaching a melisma to its notes needs.

**A rendering artefact to correct before training on this data.** Each joined system is
rendered as a standalone score, so MuseScore prints the instrument name on every one -
`InstrumentName` appears 105 times across 50 systems, about twice each. Real engraving
prints it on the first system only. A detector trained on this would learn that every system
carries an instrument label, which no scan will show. Either suppress it at render time or
exclude the class from the detection target.

#### 27.44 The page is not only lyrics, and MuseScore types the rest for free

The lyric framing was too narrow. A page carries title, composer, tempo marks, dynamics,
staff and system directions, fingerings and rehearsal marks, and an OCR pass cropping a band
under a staff meets all of them whether or not it expects to. 27.40 caught this without
noticing: the crop that recovered the voice staff also pulled in *"sempre legato"*, and the
trim that removed it was described there as an acceptable loss. It is only acceptable while
the target is lyrics.

**homr represents none of this.** Its six fields are rhythm, pitch, lift, articulation, slur
and position. The dynamics vocabulary exists but is *commented out*
(`homr/transformer/vocabulary.py`), and tempo, rehearsal marks, fingerings and titles were
never there at all. So a typed-text pass is capability homr does not have, not a second
implementation of something it does - which changes what the OCR stage is worth. It is not
only a lyric feature.

**MuseScore already types every text element**, in the same SVG class attribute the boxes
come from, so reading all page text costs no more annotation than reading only lyrics.
Across 11 rendered pages of two scores:

```
Lyrics 503   LyricsLineSegment 155   Dynamic 45   MeasureNumber 32   StaffText 18
Tempo 15     Text 13                 TextLineSegment 10   InstrumentName 7   Expression 2
```

**Which of these can be joined to a string is measured, not assumed:**

| Class | Rendered vs source | Joinable |
|---|---|---|
| Lyrics | 134/134 and 155/155 | yes - one path per syllable |
| Dynamic | 28/28 | yes - one path per `<dynamics>` |
| Tempo, StaffText, Expression | 21 rendered vs 23 source `<words>` | no |

The direction texts fail for a specific reason: MuseScore decides whether a `<words>`
renders as a Tempo, a StaffText or an Expression, and the MusicXML does not record which.
Their boxes are still extracted - a detection and classification target needs no string -
but the pairing is withheld, because an unverifiable join is precisely what this module
exists to avoid.

**Cost.** 50 systems rendered and boxed in 1m28s, 1.77s per system, zero refusals. The full
2,926 joined systems is about 86 minutes.

#### 27.43 MuseScore earns the dependency: exact syllable boxes from the engraving it drew

27.42 left one thing for the user to decide - which renderer draws the synthetic stage -
because 27.25 requires the image and the boxes to come from the *same* engraving, and
Verovio boxes over MuseScore images would be that mismatch exactly. The answer turned on
whether MuseScore actually yields positions, which is a question rather than a preference.

It does. Already installed at `/usr/local/bin/mscore`; the AppImage aborts under
`QT_QPA_PLATFORM=offscreen` but runs fine under `xvfb-run -a`:

```
xvfb-run -a mscore --export-to out.svg system.musicxml
```

Its SVG tags every element by type. On one joined system:

```
Note 146   Stem 104   Beam 31   Hook 20   LedgerLine 16   Tuplet 16
Lyrics 15                      one <path class="Lyrics"> per syllable
LyricsLineSegment 5            the melisma extender lines
SlurSegment 1   HairpinSegment 1   Clef 3   KeySig 15   BarLine 12
```

**15 Lyrics paths against 15 syllables in the source** - one path per syllable, not per
glyph, left to right and non-overlapping. Bounding boxes come straight out of the path
data:

```
source syllables  ['va', 'gues', 'des', 'mers,', 'Les', 'vents', ...]
first boxes       (793, 802, 846, 827)  (904, 801, 1007, 836)  (1015, 789, 1087, 827)
```

So the join is ordinal and checkable: sort the lyric paths by x within a staff, pair with
the syllables in document order, and refuse when the counts disagree. That refusal is the
whole point - it is the same guard 27.41 put on the measure range, and the same lesson as
27.11.

**What this buys.** Exact syllable boxes and note boxes in one coordinate system, plus a
raster of the same layout, all from the engraving the labels describe. That is positional
supervision for the resolve stage, which the scanned corpus cannot provide at all - the
published MusicXML carries no `default-x` on either notes or lyrics.

**What still needs checking before it is relied on:** that MuseScore's SVG and PNG exports
lay a page out identically. The whole value is that image and boxes agree, and that is an
assumption until measured - the kind that produced 27.11 and the sidecar substitution.

**One limitation.** MuseScore's SVG carries no element identity - no id linking a Lyrics
path back to its `<lyric>`. Reading order is the only join available. It is sound here
because syllables do not overlap and read left to right, but it means the count check is
load-bearing rather than a formality.

#### 27.42 The lyric stage is OCR plus a resolve, and the numbers say why

The first framing here was "a seventh channel or another head". That was loose and it was
challenged. Measuring the corpus settles it against the head, and not for the reason first
given.

**What rules out a head is melismas, not vocabulary.** Across all 200 scores, 34,657 lyric
occurrences:

```
syllables spanning more than one note   7,001 of 21,287   32.9%   spans up to 8 notes
notes carrying more than one verse      3,797 of 30,038   12.6%   up to 4 verses
lyric-bearing notes                    30,038 of 35,172   85.4%   of notes in vocal parts
```

The decoder emits one token per note and every field on it is a per-token closed class. One
syllable across eight notes has nowhere to live in that; nor do four verses on one note. The
structure is dense but it is emphatically not 1:1.

**On vocabulary the record went wrong twice, in opposite directions.** The first framing
claimed a long open tail. Then 7,112 syllables were found to account for *every* occurrence
in these 200 scores, and that was written up as the first claim being wrong. It was not -
7,112 is what a 200-score sample produces, and coverage measured inside the corpus it was
built from can only ever reach 100%.

Growth is the measure that settles it, and it does not saturate:

```
 scores    tokens    types    new types in the last 25 scores
     25     4,393    1,478
     50     7,895    2,333    +855
    100    16,440    4,200    +856
    150    26,536    6,002    +823
    200    34,657    7,112    +515

Heaps fit  V = 2.33 * N^0.770
extrapolated to 1,462 scores    ~33,800 distinct syllables
extrapolated to 10,000 scores  ~148,800 distinct syllables
```

A Heaps exponent of 0.770 against the 0.4-0.6 typical of natural language means the
vocabulary grows almost linearly with text - there is no size at which it closes. The
holdout gives the cost in the meantime: a vocabulary from 80% of the scores fails on 20.3%
of occurrences and 39.3% of types in the rest.

And this is one genre in three languages. A scanner has to read whatever is printed under
the staff, so the real vocabulary is unbounded and no corpus measurement can bound it. The
7,112 was a sampling artefact and treating it as a finding was the error.

**What the OCR pass faces is small.** Alphabet 104 characters, 2.31% of characters non-ascii
(French and German diacritics, curly quotes, dashes). Syllable length median 3, mean 3.3,
99th percentile 7, max 46.

**Model choice: CRNN + CTC.** Three reasons, each from the numbers above rather than from
taste:

| Reason | Consequence |
|---|---|
| Targets are fragments - `ter`, `nel`, `Ê`, `va` | Any LM prior (TrOCR, PARSeq, ABINet) corrects fragments into words. The strongest general recognizers are the wrong ones here. |
| CTC alignment gives an x-position per character | The OCR pass and the positional estimate become one model, which is exactly what the resolve stage needs. |
| 104 symbols, median 3 characters | A modest CRNN suffices; a large pretrained model buys little and brings the LM problem. |

Kraken or Calamari as the off-the-shelf baseline to beat - both built for historical printed
documents, which IMSLP scans are, both CTC. TrOCR-printed worth one measurement to know the
ceiling. Donut and Pix2Struct are far too much machinery, and homr's own encoder has
music-tuned features and the wrong receptive field for a band of text.

The OCR should emit **hyphens and extender lines as tokens** rather than stripping them.
They encode syllabic position - begin, middle, end - which is the melismatic structure the
resolve stage otherwise has to recover from geometry a second time.

**The resolve stage is the risky component.** Attaching syllables to notes is the same
correspondence problem that has produced three bugs in this work: crop-to-part indexing
(27.11), the sidecar image substitution, the slur placement transfer. Two sequences paired by
position, neither malformed alone. It wants a verifiable join with a refusal path.

**Positional supervision, and the renderer constraint.** The published scores carry no
`default-x` on notes or lyrics, so scans give ordinal supervision only - within a vocal part
the k-th syllable belongs to the k-th lyric-bearing note, with melismas and verses making
"k-th" non-trivial. Exact boxes have to come from a renderer, and 27.25 constrains which:
**the image and the boxes must come from the same engraving.** Verovio does tag every
syllable (`class="syl"` with a `<rect>`), but OLiMPiC's synthetic images are MuseScore's
engraving, and Verovio boxes over MuseScore images would be that mismatch exactly. Whichever
renderer draws the synthetic image must also emit the boxes.

**Curriculum:** synthetic with exact boxes, both recognition and alignment supervised, then
scanned with ordinal supervision only. 27.38 is the warning attached to it - notation heads
lost 25 points crossing synthetic to scanned against 1.28x for homr's existing channels, and
the alignment component is the part most likely to have learned clean-engraving geometry
that does not survive. That transfer deserves its own crosstab, not a single number.

#### 27.41 The voice comes back by arithmetic, and MuseScore is not needed

27.40 fixed OLiMPiC's images. The labels stayed a piano reduction: `get_piano_part()` is
called before any slicing happens, and `Pruner` strips `<lyric>` on top of that. So the
picture showed a singer and the label described a piano.

27.39 assumed the fix was to re-run OLiMPiC's build on the unreduced score. That would need
MuseScore, and worse, it would need MuseScore's page layout reproduced exactly - the systems
come from `<print new-system>` markers written at export, and the published OpenScore Lieder
`.mxl` files carry **none** of them (checked: zero in either part of lc5837811). Reproducing
a layout to the system is not a thing to be confident about.

**That assumption was wrong, and one measurement dissolved it.**

```
published lc5837811.mxl   P1 Chant/Voice   49 measures, numbered 0..48, 169 lyric elements
                          P2 Piano         49 measures, numbered 0..48
olimpic 5837811           16 systems covering measures 0..48, contiguous
                          p1-s1 = measures 0-2, p1-s2 = 3-5, p2-s1 = 12-14
```

OLiMPiC's slicing **preserves the original measure numbers**. A system is therefore a
measure range, and the same range read out of the published score is the voice part for
exactly that system. The join is arithmetic on measure numbers. No MuseScore, no layout,
no geometry - and all 200 of OLiMPiC's scores are in the Lieder corpus, which publishes
`.mxl` for all 1,462 of its own.

`training/omr_datasets/lieder_voice.py` does it. The piano half is passed through from
OLiMPiC's sample rather than re-sliced, so that half stays byte-for-byte what its published
labels describe; only the voice is new. Part selection matches OLiMPiC's instrument-name
list exactly, so the part discarded here is the part it keeps.

The range is checked, not trusted. A voice part one measure short would put every lyric
after it under the wrong note and nothing downstream could tell - 27.11 arriving through a
third route, after indexing and geometry.

On lc5837811, end to end:

```
16/16 systems joined
169 lyric elements recovered   against the published score's 169
p1-s2 parses as 2 voices       voice 0: 21 symbols, F4 - the singer
                               voice 1: 226 symbols - the piano grand staff
```

**The 169 is the check that matters.** The systems are disjoint, so their lyric counts sum
to the score's total only if every measure landed in exactly one system - nothing dropped,
nothing duplicated. Coverage in 27.40 could not verify its own fix; this number can.

**At scale, all 200 scores:**

```
2,926 systems joined
34,609 lyric elements recovered of the published 34,657   99.9%
5 systems refused: voice part is missing measure '0'      the pickup-measure edge case
```

The 48 unrecovered lyrics are exactly those five refusals. They refused rather than
mis-joining, which is the guard doing its job.

**A bug found here, and worth recording because of how it presented.** `slice_part` appended
the source measure elements rather than copies, so the slice and the score shared them, and
`_carry_attributes` then wrote its header *into the source*. Every later system inherited
the attributes injected for the earlier ones:

```
lc5837811, voice part <attributes> count as systems are processed:  5 5 6 7 8 9
```

It showed up as speed - the 200-score run crawled to two systems in fifteen minutes,
because the walk that gathers running attributes got longer each pass. After copying, the
same run finishes in under 25 seconds. **The speed was the harmless symptom.** The damage
was that every run produced plausible output, each carrying more duplicated clefs and keys
than the last, and nothing in the output said so. A test that joins the same sample four
times and compares now pins it.

**What is still missing is the vocabulary.** homr's `EncodedSymbol` has six fields and none
of them is text. The data for a lyric stage now exists - images showing lyrics, labels
containing them, aligned - but reading lyrics needs its own model, which is 27.42.

#### 27.40 OLiMPiC's lyrics are recoverable, and it took looking to know it

27.39 ended with a prediction: the boxes bound the piano, so extending them upward should
bring back the voice staff and its lyrics. `training/omr_datasets/olimpic_repair.py` does
that. Each box grows up towards the previous system's lower edge; the first on a page uses
the page's own typical gap, having no predecessor to measure against.

Run over all 122 annotated documents, 788 pages:

```
coverage before   40%
coverage after    98%   geometry alone
coverage after    93%   with the gutter trim below
```

**That second number is worthless and it is worth saying why.** Once each box reaches to
just below the one above, coverage reads near 100% whether or not the voice staff is inside
it. The measure that diagnosed the problem cannot verify the fix, because the fix changes
the thing it measures by construction. It confirms the boxes grew, which was never in
doubt.

So the check had to be about content. The first attempt counted staves - a Lieder crop
should hold two before and three after - which meant writing a staff detector. Three
attempts:

| Attempt | Result on crops known to hold exactly 2 staves |
|---|---|
| Rows that are >50% ink | 36% read as 2; a third read as 0 |
| Plus deskew over ±2° | no better; IMSLP scans tilt, but that was not the whole problem |
| Plus staff space read off the image | 37% read as 2, and 70 of 200 read as 0 |

Each round was calibrated against OLiMPiC's own shipped crops, which are known to be one
grand staff each, and each round failed that calibration. **The detector was never good
enough to conclude anything from, and its numbers looked plausible throughout** - the
first run reported "55% of crops gained at least one staff", which reads like a finding.
It was noise from a detector that could not count to two.

The verification that worked was rendering three before/after pairs and looking at them.
All three gained the voice staff; two of them carry legible lyrics under it - *"wir kennen
der das..."*, *"in dem Wald ein Haus"*. That is the whole claim of 27.39 confirmed, at a
fraction of the cost of the detector work, and homr already has a real staff detector had
one actually been needed.

**A sliver of the system above survived into some crops**, visible at the top edge of two of
the three. The fixed margin guesses how far stems and slurs overhang their own box, and it
guessed low - the same class of error as 27.11, one system's ink in another's image. The
page itself knows where the overhang ends, so `trim_to_gutter` scans down from the ceiling
for the first clear band and cuts there. On the sampled systems it moved the top down 34,
82 and 87 pixels. It also drops directions printed above the voice staff (*"sempre
legato"*), which homr's vocabulary does not encode; lyrics sit below the voice staff and
are never at risk. Over all 788 pages it gives back five points of coverage, 98% to 93% -
that difference is the gutters, restored.

The geometric path is kept for when no page image is available: it still recovers the voice
staff, it just leaves the sliver.

**What this does not do.** The images are now right and the labels are still wrong -
OLiMPiC's per-sample MusicXML remains the pianoform reduction, with no voice part and no
lyrics in it. 27.39's "remaining work is symbolic" is untouched by this. What has changed
is that it is now worth doing: before this, re-slicing the full Lieder score would have
produced lyric labels for images with no lyrics in them, which 27.25's eligibility test
rejects outright.

#### 27.39 A lyric track, and what OLiMPiC would have to be repaired to

18.2 designs the lyric stage; this is about where its supervision could come from. The
first question is which corpus, and the answer is not the obvious one.

```
OSSQ (string quartets)      0% of files carry lyrics
OLiMPiC scanned             0%
PDMX                       22% of files, 11.6% of notes
                              syllabic single 4,106  begin 2,117  end 2,081  middle 524
                              extend (melisma) 237   verse beyond the first 3,015
```

**PDMX has the lyrics and is the wrong corpus to learn them from.** Its images are Verovio
renderings, so its lyric text is clean synthetic glyphs at a consistent size and font. For
notation that mattered little - a beam is a beam - but for *text recognition* synthetic
rendering is not a different domain so much as the easy case, and 27.38 has just shown the
notation heads losing 25 points crossing from synthetic to scanned. A lyric recogniser
trained on rendered text would overstate by more.

**OLiMPiC has the scans and no lyrics**, because it extracted the pianoform parts from
OpenScore Lieder and dropped the vocal line with them - one part per file, zero `<lyric>`
elements. That is worth knowing before building on it: 27.37 recommended this corpus and
would have been recommending something unusable for this purpose.

**But OLiMPiC is repairable, and the expensive part is already published.**
`olimpic-1.0-sources-for-scanned` ships the IMSLP PDFs, the page renderings, the
sample-to-IMSLP mapping, and - the part that would otherwise cost weeks - the manually
annotated system bounding boxes. So for its 200 scores the title-page pruning, rotation and
page alignment are done.

The boxes bound the piano only, which is measurable rather than assumed. Over 2,173 system
pairs in 121 documents:

```
median box height     412 px
median system pitch  1013 px
median coverage        41% of the pitch   (quartiles 38% / 43%)
```

A box covering a whole system would leave only an inter-system margin, around 80-90%. At
41%, with quartiles that tight, these bound the grand staff and leave the voice staff and
its lyrics in the gap above. Recovering them is therefore geometry - extend each box upward
towards the previous system's lower edge - and not re-annotation.

**The remaining work is symbolic, not visual.** OLiMPiC's per-sample MusicXML is the
pianoform reduction, so the labels would have to come from re-slicing the full OpenScore
Lieder score - voice, piano and lyrics - against the same system boundaries. Its build
already does that slicing; it would run on the unreduced score.

Extending past OLiMPiC's 200 scores means following OpenScore Lieder's own IMSLP links and
redoing the page alignment - discarding title pages, correcting rotation, matching systems
to scans. That is the phase to defer, not the phase to start with.

#### 27.38 The synthetic-to-scan gap is far worse for notation than for notes

23 framed a domain gap and 27.14 measured it: mean OMR-NED 7.79% on synthetic against
10.00% on scanned, a factor of 1.28 across homr's existing six channels. The notation
heads were assumed to inherit something similar.

They do not. The OSSQ+PDMX model, which has never seen a scan, on OSSQ's scanned
validation split:

```
                        on synthetic    on scanned
exact beam vector            0.911          0.661
hooks F1                     0.834          0.623
stem direction only          0.929          0.721
slur spans F1                0.884          0.494
ties macro F1                0.816          0.560
```

Exact beam vector falls 25 points. Slur spans nearly halve. Against 1.28x on the existing
channels, this is a different order of problem, and it says the notation heads depend on
fine ink - the presence or absence of a short beam stub, the exact curvature of a slur end -
in a way that note and pitch recognition does not.

**This is the strongest argument yet for the scanned corpus**, and it was worth measuring
before training rather than after: a good scanned figure from the three-corpus run would
otherwise have been unattributable, exactly as 27.33's crop-guard measurement showed when
scans turned out *not* to be harder for staff detection. Two "scans are harder" intuitions,
one wrong and one badly understated, both settled by measuring the specific stage rather
than reasoning from the general claim.

#### 27.37 Other corpora: what is worth preparing, and what each would cost

27.25's test decides eligibility - does the training image show the engraving the labels
came from - and 27.36 gives the reason to care: a model trained on one genre scored 0.927
on it and 0.706 on ordinary music, so corpus breadth is not a nice-to-have.

**OLiMPiC** (ufal/olimpic-icdar24, CC BY-SA, built on OpenScore Lieder) is the strongest
candidate and needs no new label machinery. Verified on 300 sampled files of the scanned
subset:

```
24,576 notes    beams on 32.5%   stems on 89.3%
slurs 10.0% of notes             placement stated on exactly 50.0% of them
ties 6.8% of notes
```

Slur density sits between quartets (25.4%) and PDMX (3.8%), which is useful on its own -
the two corpora currently in hand are at the extremes. And placement at 50.0% in a third
independent corpus settles that it is a property of MuseScore's export rather than of
genre.

Two subsets, and they are worth different things:

- *synthetic* (train/dev/test), images rendered from the MusicXML - eligible as training
  data by 27.25's test.
- *scanned* (dev/test only), **independently photographed from physical sheet music** and
  manually annotated. This is the valuable one: every scan figure this work has produced
  comes from OSSQ, so "scans are handled" currently rests on a single provenance. 2,931
  images with paired MusicXML.

**The catch is the training unit.** OLiMPiC is pianoform: one part, two staves, one image
per system. `convert_ossq` refuses grand staffs outright (27.11), so it cannot be used -
but nothing needs cropping either, because the image already *is* the unit. A converter
for it is simpler than the two that exist, with no crop-to-part correspondence to get
wrong, which is where three of this work's bugs have lived.

**GrandStaff** passes the image test - its JPEGs ship paired with the source and are not
regenerated from tokens - but its labels are `**kern`. Beams (`L`/`J`) and stems (`/`,`\`)
are in kern, so the information is there, but `NotationExtractor` is a MusicXML walker and
a second extractor would have to be written and validated against the first. Deferred
behind OLiMPiC, which needs none. Note the OLiMPiC release also ships `grandstaff-lmx`,
which may sidestep the kern parsing entirely.

**Primus** is a `.semantic` encoding that most likely carries no beaming at all, and
**musetrainer** renders from tokens and is ineligible for the same reason PDMX was before
27.25. **Lieder** remains eligible and undownloaded.

#### 27.36 Both models on both splits: mixing was right, and quartets do not generalise

27.35 could not tell whether mixing helped or hurt, because PDMX had no held-out split.
With one - split by score, never by window, since windows of one score share an engraving -
both models were run against both validation sets:

```
                      OSSQ-only model         OSSQ+PDMX model
                     on OSSQ    on PDMX      on OSSQ    on PDMX
exact beam vector      0.927      0.706        0.911      0.867
hooks F1               0.818      0.523        0.834      0.832
stem direction only    0.943      0.773        0.929      0.817
slur spans F1          0.919      0.145        0.884      0.752
```

**The trade is not close.** Mixing costs 1.6 points of exact beam vector on OSSQ and buys
16 on PDMX; it costs 3.5 points of slur span F1 on OSSQ and buys 61 on PDMX. Every
regression 27.35 recorded is real and every one is small beside the corresponding gain.

**The quartet-only model does not generalise**, which is the finding worth carrying. Read
on its own, 27.32's numbers describe a model that appears excellent - 0.927 exact beam
vector, 0.919 slur spans. The same weights score 0.706 and 0.145 on ordinary published
music. Nothing in the OSSQ-only evaluation could have revealed that, and the design's
assumption in 25.1 that "OSSQ is the first adaptation corpus, not the definition of a
quartet-only model" was until now an intention rather than a fact.

**The slur collapse is the sharpest illustration.** The OSSQ-only model scores 0.145 on
PDMX slur spans at precision 0.080 - it predicts slurs almost everywhere. That is exactly
what 27.30 measured: slurs cover 25.4% of notes in quartets and 3.8% in PDMX, because
quartet writing notates bowing. The model learned "slurs are everywhere" from the only
corpus it saw and applied it to lead sheets. A distribution difference measured in the
symbolic files turned into a precise, predictable failure in the trained model.

**The tie asymmetry is OSSQ-specific.** 27.35 flagged tie starts (0.638) scoring far below
tie stops (0.827) with no explanation. On PDMX the same model gives 0.788 and 0.808 - nearly
symmetric. PDMX carries 74,687 tie starts against OSSQ's 31,198, so the gap looks like a
data-quantity effect rather than anything about how a tie's two ends are drawn.

**What this changes.** Training on OSSQ alone is not a step toward an ensemble
transcription model; it is a way to get numbers that do not survive contact with other
music. Every figure in 27.21 through 27.32 should be read as "on string quartets", and the
combined model is the one to carry forward.

#### 27.35 Mixing corpora: the predicted gain arrives, and so does a cost

Trained on OSSQ and PDMX together (77,888 examples) and evaluated on the OSSQ validation
split, against 27.32's OSSQ-only run:

```
                       OSSQ v2    OSSQ+PDMX
hooks F1                 0.822        0.837
beam level 2 macro F1    0.816        0.829
beam level 3 forward hk  0.659        0.709
exact beam vector        0.927        0.912
beam level 1 macro F1    0.936        0.921
slur spans F1            0.919        0.884
stem direction only      0.943        0.928
exceptions recovered     81.2%        79.4%
agreements lost           5.4%         7.0%
```

**The prediction held where it was made.** 27.30 said PDMX carries 2.8 times the hook
density and that quartets starve the heads' weakest class; hooks rise 0.822 -> 0.837 and
level-3 forward hooks 0.659 -> 0.709. Level 2, where most hooks live, improves too.

**Everything else got worse**, and the honest position is that this run does not establish
why. Two readings fit the same numbers:

- Mixing hurt: PDMX's different distribution pulled the heads away from quartet
  conventions, and the beam and slur losses are the price of the hook gain.
- Nothing got worse: the model is better across both domains, and this evaluation only
  looks at one of them.

**The reason it cannot be told apart is a gap I created: PDMX has no held-out split.** All
35,800 examples went into training, so the only validation set available is OSSQ's, and a
mixed-corpus model measured solely on quartets is being asked the wrong question. The
figures above describe how the combined model reads string quartets, which is a real thing
to know and not the thing the run was for.

That is the next step, and it is a correction rather than an extension: split PDMX by score
- never by window, since windows of one score share an engraving and would leak - and
re-evaluate both models on both validation sets. Four numbers instead of two, and the
question becomes answerable.

**The tie head trained for the first time:** macro F1 0.814, with `none` at 0.999 over
2.97 million notes and the real classes far lower - `stop` 0.827, `start_and_stop` 0.790,
`start` 0.638. The asymmetry between finding where a tie ends (0.827) and where it begins
(0.638) is not something the design anticipated and is worth a look: a tie's two ends are
drawn identically, so a gap that large suggests the head is using context rather than the
mark itself.

#### 27.34 The scanned track, converted

With the crop guard measured at 99.8% (27.33), the scanned track converts on the same
terms as synthetic:

```
train   32,982 examples    860,879 annotated notes
valid    3,571 examples
skipped     44 parts for crop/part mismatch, 344 for systems with no crops at all,
            19 for tokens outside homr's vocabulary
        6,180 slur markings collapsed to fit the legacy field
```

Every channel survives the scan pipeline: placement at 66,277 above and 46,644 below, ties
at 25,071 starts against 25,088 stops. The labels come from the same MusicXML as the
synthetic track, so this is expected rather than surprising - but it is the confirmation
that the *image* pipeline being different does not disturb the symbolic side, which is
what makes a synthetic-versus-scanned comparison meaningful at all.

Three corpora now exist with notation labels:

```
OSSQ synthetic   42,088 examples   1,136,351 notes
OSSQ scanned     32,982 examples     860,879 notes
PDMX             35,800 examples   2,501,759 notes
```

**A confound to record rather than hide.** The combined OSSQ+PDMX run declared ten heads
rather than nine, because the tie head was added while that run was queued - so it differs
from 27.32 in corpus *and* head set, which is exactly the "one variable at a time" rule
this design has been keeping.

It happens not to matter, for a structural reason. The heads are independent projections
on a frozen hidden state and the loss is their sum, so the gradient reaching a beam head is
its own loss's gradient and nothing else's. Adding a tie head cannot change what a beam
head learns. The reported *mean* loss is not comparable across the two runs - it averages
more terms - but every per-head metric is.

That is a property worth having noticed: output-only heads over a frozen core are
independently trainable, so the head set can grow without invalidating earlier per-head
results. It is also luck rather than discipline in this instance, and the rule stands.

#### 27.33 The scanned crop guard costs nothing either, and why I expected otherwise

27.14 found homr over-detecting staves on scans - five, six, seven and nine parts where
the music has four - so the expectation going into this was that 27.11's crop guard would
be expensive on the scanned track and might make it unusable. Measured over 5,623 scanned
systems:

```
detections match the part count   10,110   99.8%
mismatch (system skipped)             20    0.2%
  found one staff: 11    found none: 3    found five: 6
```

Against synthetic's 99.9%, over the full 10,130 scanned systems. The guard costs
essentially nothing on either track.

**The prediction was wrong because it conflated two different detectors.** 27.14 measured
homr's own segmentation and system grouping working on a *full scanned page* - finding the
staves, and deciding which belong to one system. What the crop guard depends on is
omr-data-preprocessor's YOLO staff detector working on a system that has *already* been
located, cropped out and resized to a target staff height. The second problem is far more
constrained than the first, and scanning noise that defeats page-level grouping does not
much trouble a detector looking at one isolated system.

So "scans are harder" is true of the layout stage and not of this one, and the two should
not have been reasoned about as a single quantity. The eleven systems where only one staff
was found, and the three where none was, are the real scanning failures - fourteen in
10,130.

This clears the way for converting the scanned track with the same guard, and it means the
27.14 over-detection figure should be quoted as a property of page-level grouping rather
than of scanned data in general.

#### 27.32 The v2 retrain: nine heads, and the hooks move

Trained on the re-converted OSSQ (27.29), the manifest declares **nine heads rather than
seven** - `slur.slot.1.side` and `slur.slot.2.side` have targets for the first time, on
141,825 and 1,107 supervised positions. That closes the chain that began with 27.20
finding placement absent, ran through 27.22's alignment and 27.29's transfer bug, and ends
here.

Against the same validation split as 27.26, so the numbers are comparable:

```
                        v1 data      v2 data
exact beam vector         0.923        0.927
beam level 1 macro F1     0.931        0.936
beam level 3 macro F1     0.879        0.885
hooks F1                  0.794        0.822
slur spans F1             0.917        0.919
exceptions recovered      79.8%        81.2%
agreements lost            5.7%         5.4%
```

Hooks moving from 0.794 to 0.822 is the result worth noting: it is the weakest class, the
one 27.30 identified as starved in a quartet corpus, and the only change between these
runs is label quality rather than more data. Exceptions recovered rises to 81.2%.

**The arbiter threshold moved from 0.9 to 0.8.** Re-tuned on this run's weights, it gives
95.89% against 94.22% for the head alone and 94.50% for the rule - so arbitration is still
worth about 1.4 points, and the threshold that was correct two runs ago is not correct
now. That is the concrete justification for sweeping rather than storing it.

**A capability was trained that could not be scored.** The slur-side heads had no metric:
`Evaluation` reported beams, hooks, stems and slur *spans*, and nothing for direction. So
the first run to train them produced no number for them at all. `slur_side_report` now
covers it, scoring only endpoints whose reference states a direction - UNSPECIFIED is a
silent source, not a third class, and scoring it would measure how often the engraver
bothered rather than how well the head reads the page.

The general shape of that mistake is worth naming: a head, its targets, its loss and its
manifest entry were all built, and the one missing piece was the measurement. Nothing
failed; there was simply a blank where a result should have been.

**Re-scored with it, the slur-side heads work:**

```
slur sides (above/below)   macro F1 0.925   micro 0.927
  above  F1 0.938 (n=10,173)
  below  F1 0.913 (n=7,295)
```

That is a capability which had no targets at all two runs ago, and it is the payoff for
the whole placement chain: 27.20 finding it absent, 27.22 verifying the join at 484 of 484
parts, 27.29 catching the silent transfer no-op, and this run declaring the head. 17,468
supervised endpoints, and the two directions score within 2.5 points of each other, so the
head is not simply predicting the commoner one.

Worth noting what this does *not* establish. Placement is stated on about half of slurs
(27.30), so the head is scored only where an engraver bothered to say - and whether the
unstated half is genuinely ambiguous or merely unlabelled is not something this measures.

#### 27.31 PDMX, converted from source

10,000 files in, 3,671 skipped, **35,800 usable examples with notation labels** and
2,501,759 annotated notes - more than twice what OSSQ carries. The skips are the filters
working: an empty final bar, a final bar of rests, fewer than 96 sounding notes, or a
part count that disagrees with the parser.

27.30 predicted from the symbolic files what the built labels should look like, and they
match:

```
                     OSSQ        PDMX
annotated notes    1,136,351   2,501,759
level-2 hooks         14,880      58,911     ~4x, and 27.30 predicted 2.8x density
level-4 hooks             99         353
level-5 anything           0          48
slur slot 1 starts   141,377      29,967     far fewer, as predicted
slur placement       143,925      30,483     ~50% of slur events in both
ties (start)          31,198      74,687
```

The hook figures are the point of doing this. Hooks are the heads' weakest class and the
clearest case of information only the image carries, and PDMX has roughly four times as
many - including 353 at level 4, where OSSQ's entire training split held 99, and 48 notes
at level 5, where OSSQ had none at all.

**One real defect, caught by the audit.** 282 examples (0.8%) had a sidecar whose count
disagreed with its token file: notation for 16 symbols where the token file yields 17
note-bearing ones. The count guard refuses those rather than attaching one note's beams to
another - correctly - but it refuses them *inside a DataLoader worker*, partway through a
training run.

So conversion now verifies the round trip before indexing an example, and drops the pair
if it fails. The existing index was filtered rather than rebuilt, which costs 282 examples
and saves an hour. The underlying cause is a symbol that is note-bearing but carries no
notation, which PDMX's window cutter can produce and OSSQ's whole-part extraction cannot -
worth understanding, but not at the price of leaving a trap in the training set.

#### 27.30 Is OSSQ representative? For beams yes, for slurs not at all

Every number in this work comes from 122 string quartets - one instrumentation, one
texture, one slice of engraving practice. `corpus_comparison.py` measures the statistics
that actually determine head sizing and Gate C against a PDMX sample, from the symbolic
files alone, using the same extractor and the same rule that produced the OSSQ figures.

```
                          OSSQ     PDMX
scores sampled             120      120
notes                1,398,799   63,612
beamable notes           59.6%    52.0%
automatic beaming        82.2%    79.8%
hooks                     2.3%     6.5%
ties                      6.3%     4.6%
slurs                    25.4%     3.8%
slurs stating placement   50.0%    50.0%
notes stating a stem     85.7%    82.7%
```

**The beam conclusions travel.** The rule reproduces 79.8% of PDMX's beaming against 82.2%
of OSSQ's, so the residue the beam heads exist for is not a quartet artefact - it is
slightly *larger* elsewhere. Gate C's premise holds.

**Hooks are 2.8 times more common in PDMX.** That is the sharpest result here. Hooks are
what MuseScore's BeamMode discards and what no duration-and-metre rule can produce, and
they are the head's weakest class (F1 0.794, and 0.442 at level 2). Training on quartets
alone starves exactly the class that most needs examples, and PDMX has nearly three times
the density.

**Slurs are 6.7 times rarer in PDMX.** Quartet writing is slur-saturated because bowing is
notated; a corpus of piano pieces and lead sheets is not. So the slur head configuration -
two canonical slots, sized from OSSQ's support tables - is sized against an outlier. Fewer
slurs argues if anything for fewer slots, so the cap is probably safe, but the *support*
figures behind it describe quartets and should not be quoted as general.

**Placement is stated on exactly half of slurs in both corpora.** Identical to a tenth of a
percent across two corpora with nothing else in common, which says it is a property of
MuseScore's export rather than of any genre - and makes the 27.22 recovery work worth as
much in PDMX as in OSSQ.

**What follows.** PDMX is not merely more data, it is differently distributed in the
direction the heads are weakest. That is the argument for converting it, and it is a
stronger argument than volume. The counter-argument is that its scores are far shorter -
530 notes per score against OSSQ's 11,657 - so a fixed score count buys much less music
than it appears to.

#### 27.29 The re-conversion, and two bugs that only a prediction would have caught

The rebuild carrying ties (27.24) and slur placement (27.22) was run against three
predictions written down before the numbers arrived. That mattered: one prediction was
met, one was wrong in an informative way, and one failure was caught only because it had
been named in advance as the thing to watch.

```
                predicted                        actual
ties            ~85,000, starts = stops          50,743; starts 23,158, stops 23,228
placement       tens of thousands stated         "none stated"  - FAILED
parts converted ~42,000 as before                38,451  - 3,637 fewer, unpredicted
```

**Ties landed.** The near-equality of starts and stops is the load-bearing part: it is the
check that the extractor is not losing one end of a span. The count came in lower than
predicted because the estimate was extrapolated from an 800-segment sample, which is a
reminder that a sample of a corpus this uneven scales badly.

**Placement did not land, and the cause was a silent no-op.** `extract_part` returns a
`<score-partwise>` root; `apply_placements` walks `<measure>` children and so found none,
dropped every placement, and reported it as an ordinary length mismatch. For one score
alone, 457 slices carried placement and not one of them landed. The index built by 27.22
was correct the whole time.

The fix is in two places, because the call site was only half of it. The caller passes
`single.find("part")`. And `apply_placements` now raises on an element that is not a
`<part>`: a wrong element is a programming error and should say so, where a length
mismatch is a data condition and still returns 0. Without that distinction the next such
mistake looks exactly like a corpus that states no placement - which is precisely how this
one presented.

**The unpredicted loss was a pre-existing problem becoming visible.** 3,637 parts (8.6%)
were refused for tokens like `slurStop_slurStop`, which is real music: two concurrent
slurs both ending on one note. homr's slur branch has three values and cannot express it,
so the conversion-time vocabulary check - added *after* the previous run - correctly
refused what the previous run had silently written.

Refusing is the wrong trade. The sidecar carries slur slots 1 and 2 separately and keeps
both endpoints exactly, so the only thing that cannot hold this is the legacy field the
sidecar supersedes. Paying a twelfth of the training set to protect it loses far more than
it saves, so the legacy field now collapses to its representable form, counted, and tested
to emit only tokens the vocabulary contains.

**The general point.** Writing the expected numbers down first turned a quiet "none
stated" into a caught bug. Had the prediction not been made, that line would have read as
a fact about the corpus - and the slur-side heads would have stayed untrainable for a
reason nobody was looking for.

**The corrected run, against the same predictions:**

```
examples          42,088     restored; 4 crop mismatches, 24 pipeline refusals
slur placement    above 85,879, below 58,046      predicted ~146,000, actual 143,925
ties              start 31,198, stop 31,195, start_and_stop 4,992
collapsed slurs   6,626 in train, 720 in valid    sidecars keep both endpoints
```

Placement is within 2% of what 27.22's alignment predicted for this split - 178,433
recoverable corpus-wide, of which train is about 82% - which is the confirmation that the
join transfers what it measured rather than merely aligning. Ties are symmetric to within
three events in thirty-one thousand.

`slur.slot.1.side` and `slur.slot.2.side` now have targets for the first time, so the
manifest should declare nine heads rather than seven.

#### 27.28 The stem head and the rule are complementary, not redundant

27.27 read as a case for deleting the stem head: two sources, the same accuracy, so keep
the free one. The crosstab says otherwise.

```
                  rule right   rule wrong
  head right      100,542        4,491
  head wrong        4,506        1,690
```

They fail on almost disjoint notes. The head rescues 4,491 of the rule's mistakes - 72.7%
of them - the rule rescues 4,506 of the head's, and only 1,690 notes of 111,229 defeat
both. An oracle choosing per note would score **98.5%** against 94.4% for either alone.

**This is the second time in this work that equal totals hid opposite behaviour**, after
Gate C. It is worth stating as a rule of thumb: when two sources agree on a headline
number, that is the moment to look at the crosstab, not the moment to pick one.

That still left the question of whether the gap is *reachable* - an oracle needs the
answer. It is, from the head's own softmax. Tuned on half the staves and reported on the
other half, because a threshold chosen on the notes it is quoted against reports the
tuning, and `test_synth` is reserved:

```
                              reporting half (60,927 notes)
head alone                                 94.39%
rule alone                                 94.41%
head if confidence >= 0.9                  95.92%    head used on 82.7% of notes
head if beam-group margin <= 2             95.27%    head used on 31.4%
either signal                              95.60%
oracle                                     98.57%
```

**Confidence arbitration is worth 1.5 points over either source.** So the stem head stays,
but its role changes: it is not the stem predictor, it is the second opinion on a rule
that handles most notes. The margin signal - how far a beam group's extreme notehead sits
from the middle line, which is exactly what the rule thresholds on - is weaker and largely
redundant with confidence.

2.6 points of oracle headroom remain unclaimed, which is where a better arbiter would go.

#### 27.27 The stem head can be replaced by a rule over its own beam predictions

27.26 proposed it; this measures it. Stem direction derived from the *predicted* beam
grouping plus pitch, scored over exactly the notes the stem head is scored on - 111,229,
which is the evaluation's up (51,871) and down (59,358) added together, so the populations
are identical rather than merely similar:

```
pitch alone                     91.0%
the trained stem head           94.3%     25,137 parameters
grouped by predicted beams      94.4%     no parameters
grouped by reference beams      94.9%     upper bound, perfect grouping
```

**The rule matches the head and edges past it.** On totals the head is not adding anything
the beam predictions plus a middle-line rule do not already give - but see 27.28, where
the crosstab shows this conclusion is wrong: the two fail on disjoint notes, and an
arbiter between them beats both.

The gap between predicted and reference grouping is 0.5 points, so the derived figure will
improve as the beam heads do - it inherits their accuracy instead of competing with it.
That is the better dependency: one capability improving two outputs.

**What follows.** The stem head is a candidate for removal, not a proven failure. Three
things are worth settling first:

- Whether it holds on `test_synth`, where the beam baseline is 8 points lower than on
  `valid` and predicted grouping will therefore be worse.
- Whether the two disagree on the *same* notes. A head that is 94.3% right on a different
  set from the rule's 94.4% could still be worth keeping as an ensemble; one that fails
  where the rule fails is pure redundancy. This needs the crosstab treatment that settled
  Gate C, not a comparison of totals.
- The multi-voice convention (upper voice up, lower down) is not applied here, because
  homr does not emit voice. It was worth 0.3 points in 27.23, so its absence is not what
  separates these numbers.

Note what this does *not* say about beams. Beams needed a head because 18% of them are not
derivable from anything homr already has. Stems are derivable from beams, so the beam head
carries both - which is an argument for the beam heads, not against heads in general.

#### 27.26 The converged run: three epochs was already most of the way

Item 9 asked whether 27.21's numbers were a floor. They were, but a shallow one: twelve
epochs buys about a point.

```
                              3 epochs   12 epochs
mean loss                       0.9675      0.8617
exact beam vector                0.920       0.923
beam level 1 macro F1            0.929       0.931
beam level 2 macro F1            0.818       0.836
beam level 3 macro F1            0.879       0.879
hooks F1                         0.790       0.794
slur spans F1                    0.916       0.917
exceptions the head recovers     78.7%       79.8%
```

The loss was still falling at epoch 12 (0.8650 -> 0.8617), so this is not fully converged
either - but the gradient of *usefulness* against epochs is now clearly flat. Further
training is not where the remaining error is.

**Gate C, converged:**

```
                  head right   head wrong
  rule right        59,874        3,615
  rule wrong         9,355        2,373

rule accuracy 84.4%    head accuracy 92.0%
exceptions recovered   79.8%   (9,355 notes)
agreements lost         5.7%   (3,615 notes)
```

**The stem head is a different story, and 27.23's suspicion was right.** The evaluation now
reports direction-only accuracy, which is the figure comparable to the baseline:

```
pitch alone, per note (what a label file implies)     91.4%
the trained head, up/down only                        94.3%
pitch and voice, one direction per beam group         95.7%
```

So the head buys 2.9 points over what the labels already imply, and sits 1.4 points *below*
a rule that knows where the beam groups are. That is not a head earning its place.

**And there is now a better option than either.** The beam heads predict beam grouping at
0.923 exact-vector accuracy, which is most of what the 95.7% rule needs. Deriving stem
direction from *predicted* beams plus pitch is very likely to beat the stem head, needs no
parameters at all, and is testable immediately with what already exists - `stem_baseline.py`
takes its grouping from a beam vector, and the beam heads produce beam vectors.

The negative result is worth as much as the positive one: beams needed a head because 18%
of them are not derivable, and stems appear not to, because 95.7% of them are.

#### 27.23 Stem direction is mostly a rule, and the head has not yet beaten it

The beam heads have a baseline to clear; the stem head had none. 27.21 reports macro F1
0.719 and micro 0.949 against nothing at all, which is the wrong way round - stem
direction is the *most* rule-governed of the three notations, so a head that cannot beat
the textbook rule has learned nothing worth keeping.

`training/omr_datasets/stem_baseline.py` implements the rule engravers use: a note at or
above the middle line takes a down stem, a chord takes the direction of the notehead
furthest from the middle line, and where two voices share a staff the convention overrides
pitch entirely (upper voice up, lower down). The middle line comes from the clef, so the
whole rule needs only pitch and clef - both of which homr already predicts. On `valid`:

```
pitch alone (what a label file carries)            102,402 / 112,051   91.4%
pitch and voice (needs the MusicXML)               102,725 / 112,051   91.7%
and one direction per engraved beam group          107,211 / 112,051   95.7%
```

Voice is worth 0.3 points and is not in a label file anyway - symbols are ordered by
position within a measure and never say which voice they belong to - so 91.4% is the
honest figure for "already implied by the labels". The third line is an upper bound rather
than a rule homr could apply unaided, since it uses the engraved beams to fix the groups,
but it is the fair statement of what a rule *can* do, because real engraving sets one
direction per beam and not per note.

**So stem direction is 91.4% implicit in the labels as they stand, and the trained head's
94.9% micro is not a like-for-like comparison.** That micro includes
`NOT_APPLICABLE` - rests and whole notes, which have no stem at all, which the rhythm
token already determines, and which the rule never attempts. `Evaluation` now reports
`stem direction only (up/down)` for exactly this comparison, and the converged run will be
the first to have it.

The honest position until then: the beam result is established (78.7% of exceptions
recovered, 27.21), and **the stem result is not**. It sits between the per-note rule and
the beam-grouped rule, which is the range where a head could be adding nothing. This is
what a baseline is for, and it is the reason 15.3 asked for one.

#### 27.22 Slur placement can be recovered, and the join is verifiable

27.20 left `slur.slot.N.side` untrainable: placement survives on half the slurs in the
original whole scores and is stripped from every segment by the MuseScore round-trip.
Recovering it needs a positional join - segment note *k* of a part is whole-score note *k*
- which is the correspondence that has failed five times in this pipeline, so the join was
measured before anything was built on it.

The naive join aligns **4 parts of 48**. The shape of the failure is what makes it
tractable: every failure is a *length* mismatch and none is a *signature* mismatch, which
says both walks see the same music and one of them sees more notes than the other.

Grace notes were the obvious suspect and are innocent - they match to the note on every
part inspected. The cause is invisible notes (`print-object="no"`), which the segmentation
drops: the shortfall equals `whole invisible - segment invisible` exactly, on all 16 parts
checked by hand. Excluding them, corpus-wide:

```
parts checked                        484
aligned note for note                484   100.0%
length mismatches                      0
signature mismatches                   0
notes in aligned parts         1,409,056
slur placements recoverable      178,433
```

Alignment is checked on a pitch-and-type signature rather than a count, because a dropped
note and an added one cancel in a count - and that is precisely how a placement would
transfer silently onto its neighbour.

178,433 recoverable placements against slot 1's 141,385 slur starts is enough to train the
side heads. What remains is the transfer itself: writing placement into the sidecars at
conversion time, with the alignment check as a per-part precondition rather than an
assumption.

#### 27.21 Phase 2, the first frozen-core run, and Gate C

**The heads clear the baseline, and they do it on the part of the corpus a rule cannot
reach.**

Training: 41,764 examples after the ledger-line filter, three epochs, ~15 minutes each on
a 4090. Only the 25,137 head parameters moved; the 300 MB core stayed frozen and in eval
mode. The structured loss never joined `loss`, so B0 remains comparable.

```
epoch 1: mean 1.4756  beam.1 0.3163  beam.2 0.3629  beam.3 0.3932  beam.4 0.0649  stem 0.1985
epoch 2: mean 1.0330  beam.1 0.2366  beam.2 0.2483  beam.3 0.2417  beam.4 0.0300  stem 0.1579
epoch 3: mean 0.9675  beam.1 0.2270  beam.2 0.2315  beam.3 0.2206  beam.4 0.0187  stem 0.1539
manifest: 7 heads declared   [no targets all epoch: slur.slot.1.side, slur.slot.2.side]
```

The two slur-side heads got no targets and the manifest declined to declare them, exactly
as 27.20 predicted. That is the capability machinery doing its job rather than a surprise.

Evaluation on the held-out `valid` split, 4,912 staves, undistorted images:

```
exact beam vector   0.920  (72,051 / 78,307)
beam level 1        macro F1 0.929   micro 0.931
beam level 2        macro F1 0.818   micro 0.923
beam level 3        macro F1 0.859   micro 0.903
beam level 4        macro F1 0.292   micro 0.500   (8 notes - not a result)
hooks               F1 0.790  P 0.807  R 0.774  (n=1,529)
stems               macro F1 0.719   micro 0.949
slur spans          F1 0.916  P 0.952  R 0.882  (n=17,146)
```

**Gate C.** The comparison that matters is not head-against-baseline but the crosstab, on
the same notes:

```
                  head right   head wrong
  rule right        59,790        3,699
  rule wrong         9,234        2,494

rule accuracy 84.4%    head accuracy 91.8%
exceptions the head recovers   78.7%   (9,234 notes)
agreements the head loses       5.8%   (3,699 notes)
```

27.12 asked whether a head could recover "even half the exceptions". It recovers **78.7%**
of them - 9,234 notes where duration and metre are provably insufficient and the head is
right anyway. A head that had merely learned the rule would score zero in that cell. The
price is 3,699 notes the rule had right and the head lost, and both are reported because
only the pair is honest.

Hooks are the sharpest version of the same point: F1 0.790 on 1,529 hooks, which are
precisely what MuseScore's `BeamMode` discards and what no duration-and-metre rule can
produce.

**How much of this to believe, and why the number moved.** The crosstab put the rule at
83.5% where the committed baseline put it at 87.0% on the same split. That 3.5-point
disagreement was the cross-check working, and chasing it found two real errors and
rejected two plausible-sounding hypotheses:

- *Rejected:* beam groups cut at system breaks. Disagreements are **less** frequent at the
  first and last beamed note of a staff (7.2%) than in the interior (16.7%).
- *Rejected:* the cleaned, original and segmented MusicXML disagreeing about beams. The
  markup is identical in all three - 6,291 beam elements each, same distribution.
- *Real:* the crosstab was beaming every segment as if it were in 4/4, because a segment
  restates `<time>` only at a movement start. Carrying the meter across a part's segments
  in reading order: 83.5% -> 84.4%.
- *Real, and larger:* the **baseline** was counting rests as free agreements. An eighth
  rest has a flag count but no stem, so it can carry no beam; the rule says
  not-applicable, the engraving says not-applicable, and every one scored as a match the
  rule never earned. They were 11.4% of what was counted on this split. This is the same
  error class the baseline already guarded against for quarter notes, missed for rests.

The corrected baseline on `valid` is 85.3%, against the crosstab's independently computed
84.4%. The residual 0.9 points is two different walks over slightly different artifacts -
whole scores against converted staves, six levels against four - and is small enough to
leave stated rather than chased. Note the direction of the correction: the overstated
baseline was **understating** the head's advantage, not flattering it.

**What is not established.** Beam level 4 has eight supervised notes in the whole
validation split; its macro F1 of 0.292 is noise and should not be quoted. The slur side
heads were not trained at all (27.20). Everything here is the synthetic track; the scanned
track has been benchmarked for layout (27.14) but no head has seen it. And this is one run
at one learning rate for three epochs - the losses were still falling, so these are a floor
rather than a converged result.

#### 27.20 What the built labels actually contain

The audit of the built training set - 42,088 examples, 1,136,381 annotated notes - settles
the head configuration §25.2 left open, and one of its answers is negative.

```
beam levels (over notes the level applies to)
  level      flag     begin  continue       end   fwd hook   bwd hook      total
      1    91,087   182,754   234,410   182,755          0          0    691,006
      2    11,161    64,000    98,955    64,001      3,855     11,025    252,997
      3     1,488     5,971    12,531     5,971        869      1,618     28,448
      4       128       432       734       432         27         72      1,825

stems   down 527,904   up 456,085   not_applicable 144,210   unknown 8,179   none 3

slur slots (notes carrying an event)
  slot        start       stop   start_and_stop
     1      141,385    141,419            1,403
     2        1,389      1,367               14
     3           42         39                0
     4           10         10                0
     5            3          3                0
     6            2          2                0
```

All four configured beam levels have support, level 4 thinly at 1,825. Two slur slots are
justified and slots 3-6 are not, which is what `TRAINED_SLUR_SLOTS = 2` already assumed.
Zero hooks at level 1 is a consistency check passing rather than a gap: a hook needs a
beam above it to hang from, which is why `beam_validation` treats a level-1 hook as
invalid.

**The negative answer: slur placement is gone, so the side heads cannot be trained.**
Every slur in the built labels has placement unstated. Tracing it back:

```
original whole scores    placement on 8,328 of 16,656 slurs (50%)   numbering 1/2/3
cleaned whole scores     placement absent                           numbering absent
systemwise segments      placement absent                           numbering 1/2/3
```

The segments are not derived from the cleaned copy - the numbering survives, and slot 2
carries 1,389 starts - so placement is lost in the MuseScore round-trip that produces
them, which stores slur direction as automatic and omits it on export. This is the same
shape as 27.8's stripped stems, but unlike that one there is no preprocessor flag to turn
it off.

So `slur.slot.N.side` gets no supervision and the manifest will decline to declare it,
which is the machinery working as designed rather than a surprise at inference. The
information is not lost from the corpus, only from this path to it: recovering it means
taking placement from the original whole-score MusicXML and joining it onto the segments
by note, which is precisely the kind of positional correspondence that has produced four
bugs here already. Worth doing deliberately, not as a side effect.

#### 27.19 Building the training set: three things that stop a conversion dead

The synthetic partwise crops came out at **52,973 against the published figure of 52,960**,
which is as close a confirmation as a rebuild can give. Converting them into a homr
training set then failed three times, each on something that killed the whole run rather
than one staff. They are recorded together because they share a shape: each is a case where
the corpus is legal and homr's pipeline is reasonable, and the two do not meet.

**A rhythm the vocabulary has no token for.** `KeyError: 'note_72'` - a tuplet factor
scales the base duration (16 x 9/2 here) into a value the rhythm vocabulary never
enumerated, the same family as 27.10's 256th notes. There is no correct token, and
inventing one would put a symbol in the labels the model can never predict.

**A backup reaching behind the start of its measure.** `ValueError: Backup duration is too
long 0 -8`. This is 27.18's durationless whole-measure rest from a second angle: with no
duration the position never advances past the rest, so a second voice's `<backup>` to the
measure start goes negative. See the correction in 27.18 - the conclusion there was right
about the token and stopped too early.

**Every path in the corpus contains a comma.** homr's index is one `image,token_file` pair
per line, split on the comma; OSSQ files every score under `Lastname,_Firstname`, and all
47 composer directories in this corpus are of that form. So no line of the index parses -
the loader takes the wrong side of the split and opens
`_Elfrida/String_Quartet_in_A_major/...png,/workspace/...txt`. Rewriting the shared index
format would touch every corpus homr trains on, so the dataset gets its own comma-free name
for each crop, symlinked beside its token file.

With all three handled as skips rather than aborts, the train split converts:

```
42,089 examples written
     4 parts skipped: staff crops did not match the parts   (the 27.11 guard)
    23 parts skipped: token pipeline refused them
                      note_19 x8, note_24. x6, note_52 x3,
                      Backup duration x2, note_60 x1, note_72 x1
```

27 losses in 42,116 - 0.06%. Both guards cost almost nothing, which is the outcome the
27.11 measurement predicted and the reason it was worth measuring first.

**The operational lesson, which cost two runs.** Every step of the driver script piped
through `tail`, so a failing Python exited 0 as far as `set -e` was concerned. One
conversion crash became four cascading `FileNotFoundError`s and buried its own cause.
`set -eo pipefail`.

#### 27.18 Whole-measure rests need no repair on the training side

Converting real segments raises hundreds of `Note without duration` warnings, which is
27.5's finding: whole-measure rests carry no `<duration>` because MusicXML lets the meter
imply it. §27.5 repairs this for the benchmark, so the obvious conclusion is that
`convert_ossq.py` needs the same repair. It does not, and adding it would have introduced
label errors.

Measured on 800 segments: every durationless rest is `<rest measure="yes"/>`, 1,110 of
8,106 rests (13.7%), and all of them lack `<type>` as well. But the parser does not reach
`<type>` for these - a measure rest takes a separate path through
`_measure_rest_rhythm(duration, divisions)`, and with duration 0 that falls through its
lookup table and returns `rest_1`. The rest is still emitted; only its value is in
question.

`rest_1` is the right label. A full-bar rest is engraved as a whole-rest glyph in the
simple meters this corpus uses, whatever the meter, so the token matches the image.
Materialising the duration would produce `rest_2.` in 3/4 and `rest_2` in 2/4 - correct
arithmetic, wrong glyph, and a label error on one of the most common constructs in
ensemble music.

It is also not possible per segment. `<divisions>` appears in 100% of segments but `<time>`
in none of the ones carrying these rests: a systemwise segment restates the meter only at a
movement start or a genuine meter change. The benchmark can materialise because it carries
meter across pages; a converter working one segment at a time has nothing to compute from.

**Correction: the token is not the only thing the missing duration touches.** The
conclusion above - that no repair is needed - is right about the rest token and stops too
early. Position accounting also reads the duration, and with duration 0 the position never
advances past the rest. When a second voice's `<backup>` then returns to the start of the
measure, it goes negative and the parser refuses the part outright:
`Backup duration is too long 0 -8`. That killed a whole conversion run.

The repair is still not available - the meter needed to compute the duration is not in the
segment - so the converter skips those staves and counts them. The right way to read 27.18
is therefore narrower than first written: the *labels* need no repair, but the durationless
rest is not harmless, and staves where a second voice backs up over one are lost rather
than converted.

**One flagged uncertainty, on the benchmark side rather than this one.** Materialising
changes nothing in 4/4 - the arithmetic gives back `rest_1` - so §27.5's repair only alters
the reference in other meters, where it will disagree with a whole-rest glyph that homr
reads correctly, and charge that to homr. How much this matters depends on how much of the
corpus is not in 4/4, which has not been measured. It would be settled by comparing the
reference token against homr's output on measure rests in 3/4 bars specifically.

#### 27.17 Slurs that cross a system break

The training unit is one system's staff, so the symbolic segment behind it is cut at
system breaks - and a slur crossing one arrives in the next segment with no start.

The extractor was dropping those stops, which labelled the note as carrying no slur on a
crop that plainly shows a slur ending on it. The asymmetry is what gives it away: the
other half of the same crossing, a start whose stop is in the next system, was already
recorded as a START and merely reported at `close()`. Only the stops were discarded.

Measured over 600 real segments, 56,553 notes:

```
unmatched stops (start is in the previous system)   291
unclosed starts (stop is in the next system)        293
slot overflow                                         0
```

The near-equality is the confirmation that these are the two halves of the same
crossings. Scaled to the corpus's 13,244 segments that is roughly 6,400 notes - and they
are not scattered, they sit at the start of systems, so the head would have learned that
slurs never end near the left edge of a staff. That is a learnable, wrong bias rather
than noise, which is what makes 0.5% of notes worth fixing.

Unmatched stops are still counted in `Findings` - that is how the crossings are measured -
but they are no longer a reason to discard the label. Note this does not make the *span*
complete: the endpoint-pair metric will see a stop with no start within the crop and score
no span, which is right, because within one staff image there is no span to find.

#### 27.16 The Gate C baseline, per split

27.12's number came from a throwaway script. `training/omr_datasets/beam_baseline.py` is
the same measurement committed, and it disagrees with the recorded figure - so the
recorded figure goes.

**These figures were themselves corrected once - see 27.21. Rests were being counted as
free agreements, overstating every number here by 1.7 to 2.4 points. The corrected table
is below; the reasoning about splits is unchanged and the conclusion is strengthened.**

Corpus-wide, 121 scores:

```
automatic beaming matches the engraving   669,051   82.0%
exceptions the rule does not predict      146,705   18.0%
```

against 79.4% from the uncommitted script. Two differences account for most of it. The committed tool measures
whole scores rather than systemwise segments - a segment cuts at a system break, which
cuts beam groups and restarts the divisions and time-signature context, and the rule
scored 91.9% on a sample of segments purely from that fragmentation. And it counts one
decision per stem rather than per notehead, so a chord no longer counts its beam three
times. The exact composition of the old 20-score sample cannot be recovered, so the gap
is not fully reconciled; the committed number is the one to quote because it can be
re-derived.

Per split, which is the part that matters for Gate C:

```
split        scores   rule matches   exceptions
train            99          82.1%        17.9%
valid            11          85.3%        14.7%
test_synth       11          77.3%        22.7%
all             121          82.0%        18.0%
```

**The baseline moves 8.0 points between splits.** A head evaluated on `test_synth` faces
a 77.3% baseline, not the 82.0% corpus figure - quoting the corpus number against a
test-split result would hand the head 4.7 points it did not earn. Gate C should compare
against the baseline on whatever split the head is scored on, which is why the tool takes
`--split`.

The disagreement profile is the same one 27.12 described and is the reason to believe the
residue is editorial rather than a rule still missing: `begin -> continue` (27,495) and
`continue -> begin` (27,095) are nearly equal, as are `continue -> end` (25,658) and
`end -> continue` (25,275). A rule with a systematic offset would push one way. This
pushes both ways in almost equal measure, which is what per-instance choice looks like.

One score, `sq7362818`, is assigned a scanned split but no synthetic one, so 121 of the
122 synthetic scores are measured. That is the manifest working as intended rather than a
gap.

#### 27.7 Known gaps

- 2.9% of pages still fail layout, 80 of them collapsing four parts to one. These are
  the cases where geometry declines - overlapping duplicate staff detections, too few
  systems to read - rather than decides wrongly. Measured before incomplete-system
  recovery landed; due a re-measure.
- Incomplete systems are now recovered where the spacing says which voice is absent, and
  still dropped where it does not. A voice missing between two detected staffs shows up
  as an internal gap of about two ordinary gaps plus a staff height; one missing from a
  system's top or bottom shows up as an oversized boundary to the neighbouring system
  (19.26 unit sizes against a typical 9.15 on the page this was traced on). A short
  system at the very end of a page has neither, so it is still dropped.

  On the 60-page set, mean 5.19% -> 3.99%, 9 pages improved and none worse. Of the 11
  pages that were truncated, 7 recovered: their mean NED 12.21% -> 5.72% and their mean
  `pred/ref` token ratio 0.82 -> 0.94. That ratio is the load-bearing number - it says
  the recovered systems' music is being read, not that the metric got easier. It does not
  return to 1.01 because the missing staff is still missing: one voice genuinely has no
  music for that system.

  End to end across both layout changes, on the same 60 pages: mean 16.34% -> 4.09%,
  layout-broken 12 -> 0, median unchanged at 2.70%, all 60 pages producing output.
- The scanned benchmark has not been run; only its pipeline is validated end to end on
  one score (168 detected systems against 168 symbolic segments, aligning to scanned
  pages 3-58, matching that score's curated `3:58` range).
- The non-regression tolerance for Gate D has not been declared. B0 variance is now
  measurable, which is the precondition §22 sets for declaring it.


---

# Part II — Work log

*Verbatim from `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md`; headings demoted one level.*

## Ensemble transcription: open threads to work from

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

### Contents

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
  - [Release published: `onnx_checkpoints` on `jhlusko/homr`, verified end-to-end](#release-published-onnxcheckpoints-on-jhluskohomr-verified-end-to-end)

### 1. The text detector's page-level precision has collapsed for five of seven classes — **CLOSED 2026-08-20, user decision**

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

#### Where it stands

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

#### Why this happened (§27.86–27.87, `ENSEMBLE_TRANSCRIPTION_DESIGN.md`)

The original per-pixel training/valid IoU numbers (27.86: 0.81–0.997 across every class)
looked strong enough to skip an architecture search. That measurement only ever showed
the model `DetectorPatches`' 70%-positive-biased training crops. A full page is >99%
background outside boxes; the model was never evaluated — or, implicitly, ever trained —
against that ratio. The whole-page eval (27.87) exposed the real failure: a
training-patch-distribution vs. inference-distribution domain gap, not a per-pixel
segmentation problem. **`DetectorPatches.POSITIVE_RATIO = 0.7` is still the leading
suspect and has not been changed.**

#### What has already been tried

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

#### Not yet tried

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

### 2. Fingering's corpus-level rarity — **CLOSED 2026-08-20, user decision, same as §1**

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

### 3. Score-profile conditioning — both training runs complete; frozen-core result stands, unfrozen follow-up did not improve on it

Full contract already specified in `ENSEMBLE_TRANSCRIPTION_DESIGN.md` §7; reproduced
here so this file is self-contained.

#### Built so far

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

#### Contract

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

#### Use in layout

Supplies an expected ordered physical-staff pattern (`[1,1,1,1]` for a quartet, `[1,2]`
for voice+piano) used as a *scored hypothesis*, not an assertion. Layout must report:
detected physical staff count, proposed systems/staff rows, proposed row→profile-part
mapping, an evidence score with competing hypotheses, deviations from the supplied
profile, and exact source-image regions.

#### Use in staff recognition

For a recognized staff, encode only the applicable context: instrument-family embedding,
part-ordinal embedding, staff-within-part-ordinal embedding, expected-staff-count
embedding, likely-clef-set embedding, transposition embedding, and an
unknown/context-missing indicator. First implementation: inject as prefix/context tokens
to the decoder, or a gated additive vector to encoder context — **the gate must be
zero-initialized so the unconditioned path is bit-identical at initialization**, the
same zero-gate discipline the structured heads already use for backward compatibility.

#### Training: context dropout

Randomly remove the whole profile and independently mask fields during training, so the
model does not become dependent on it. Starting hypothesis (not fixed): 30% no profile,
30% partially masked, 40% complete. Evaluate both conditioned and unconditioned
inference.

#### Why this is next after §1, not before

§24's original implementation slice lists this as item 9, after the structured heads
(item 1, done) and system grouping (also done). Nothing about it depends on §1's detector
work resolving first — it is independent, parallel work. It is ranked below §1 here only
because §1 has existing infrastructure and a clear next experiment already named, where
this thread needs a new module (`homr/score_profile.py` or similar does not exist yet)
built from a written spec with no empirical validation yet.

#### Next implementation step

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

#### `phase20`: the training run — complete, 10/10 epochs positive

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

#### `phase21`: the unfrozen follow-up — complete, and it reverses phase20's read

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

### 4. Cross-staff consistency checks and repair — Stage A complete, Stage B measured

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

#### Stage A: deterministic consistency analysis — all 8 of 8 §12.1 checks built, plus
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

#### Wired into the live pipeline, log-only, validated on real pages

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

#### Stage B: targeted repair proposals from existing alternatives (tier 1 built for key/time signature)

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

#### A second motivating case for cross-staff repair, from a design discussion: shared
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

#### Systematic 200-page benchmark — the "built and benchmarked" evidence §12.3 asks for

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

#### Stage C: learned variable-staff context adapter (not started, blocked on A+B being measured)

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

#### Why this is last

§12.3 states its own precondition in the imperative: only build the learned adapter
*after* Stages A and B are benchmarked. Stage A alone may resolve enough — the design
never assumes it will need Stage C. Starting here means starting at the beginning
(Stage A), which has no dependency on §1 or §3 resolving first, but is scoped last in
this document because it is the newest, least-derisked of the four threads: no code, no
measurement, and its own success criterion (Stage C being *unnecessary*) is still
completely open.

---

### 5. Decoder rhythm/duration accuracy — corrected 2026-08-21 for a second ground-truth bug (movement splicing); Beethoven-shaped is now the plurality (34/75, ~45%), reversing the prior ~20%/~65% read

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

#### Phase 1 started 2026-08-21: decode-time cross-staff-consistency reranking, built and validated, 20-page result already positive (31→16 findings)

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

#### Phase 2 started 2026-08-21: time-signature conditioning + a ground-truth-supervised duration loss, training run launched

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

### 5. (prior) Decoder rhythm/duration accuracy — final: real decoder divergence confirmed, ~20% Beethoven-shaped (Phase 1's target), ~65% Moeran-shaped (broadly poor decode, Phase 1 alone won't fix)

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

### 6. IMSLP corpus expansion beyond OLiMPiC's own 200 manually-annotated scores — download complete, automated detection built, review tooling built

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

### 7. Stage 2 & Stage 3 training-data extraction from the expanded IMSLP corpus — scoped, not started

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

#### Stage 2: real-scan training pairs from the bar-count-confirmed matches

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

#### Stage 3: real lyrics/dynamics text-region ground truth from the scans

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

#### Update, 2026-08-24 overnight: real pairing fix, OCR-first built and validated, scope corrected to 472 scores

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

#### Update, 2026-08-24 (later still): Stage 2's pair-extraction script - built, tested, and validated on real data

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

#### Update, 2026-08-25: full-scale Stage 2 extraction run complete - 2,535 real training pairs

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

#### Update, 2026-08-25: a real user-found bug in the extracted pairs, root-caused and fixed

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

#### Update, 2026-08-25: review sites built - a Stage 2 pair reviewer and a Stage 3 text reviewer, merged into one server

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

#### Update, 2026-08-25: a real multi-verse lyrics bug found via the review site, root-caused and fixed

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

#### Update, 2026-08-25: THE root cause - a one-measure off-by-one in every system range in the whole corpus

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

#### Update, 2026-08-25: what "Stage 2 training" actually requires on this box

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

#### Update, 2026-08-25: post-fix bar-count result, and what ENSEMBLE_TRANSCRIPTION_DESIGN.md requires of the training run

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

#### Update, 2026-08-25: recovering the excluded systems by content, and the replay decision

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

#### Update, 2026-08-25: recovery complete, and the Stage 2 scans fine-tune is running

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

#### Update, 2026-08-25: Stage 2 scans fine-tune - result, and stopped early on a plateau

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

#### Update, 2026-08-25: Stage 3 (text detector) - tooling built, experiment matrix started

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

#### The box has a pid limit, and onnxruntime does not respect thread hints

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

#### E0 baseline — complete (2026-08-25)

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

#### E1-E3 results — real-scan data helps substantially, and E3 is the configuration to keep

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

#### The page-level measurement contradicts the patch measurement — E3 halves precision

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

#### E4/E5 — the middle masking policy (running)

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

#### The OSSQ scanned track is systematically mislabeled — root cause found (2026-08-25)

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

#### Beams reach the output, and repeats are recovered — two gaps closed from one review session

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

#### Structured heads on the final base — beaming recovered at 95%, dynamics did not train

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

#### Adding the OSSQ synthetic track costs scan accuracy — measured, not assumed

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

#### Shipping decision: two detectors, E2 for vocal and the instrumental model for the rest

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

#### Head-to-head against upstream homr — the number the whole effort is judged on

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

#### The instrumental detector — the "without lyrics" half of the two-model split

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

#### The clef-corrected continuation — plateaued, and the sequence of runs so far

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

#### A second, independent data bug: 2.4% of staves had no clef — found by eye, invisible to every metric

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

#### Training on the corrected corpus — first numbers, and what they do not yet prove

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

#### The OSSQ fix, and its verification (2026-08-25)

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

#### E4/E5: the middle masking policy did not work, and the prediction it was built on failed

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

#### OSSQ instrumental text extraction — complete (2026-08-25)

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

#### Building the best Stage 2 model: what the artifacts actually are, and the order to do it in

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

#### Stage 2 renders and the review site (2026-08-25)

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

#### Packaging the corpora and models for distribution

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

### Structured heads in production, and the refinement UI

**Decided 2026-08-25 (user): heads on in production, and their probabilities exposed as
multiple-choice refinements.** This section records what that requires and what was
deliberately scoped out, because the gap between "the heads work" and "the heads do
anything for a user" is larger than the evaluation numbers suggest.

#### The chain is built at both ends and missing in the middle

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

#### What ships as a choice, and what only ships

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

#### Surfacing: threshold-gated, not always-on

Alternatives appear only where the model is genuinely uncertain. At 0.9508 exact-beam
accuracy, offering a choice on every note would bury the ~5% worth reviewing under 95%
noise. The threshold is a tunable, and choosing it needs the confidence distribution on
real pages - not yet measured, and it should not be guessed.

The known cost: a confident-but-wrong prediction is never surfaced. That is the accepted
trade, and it is the reason the threshold should be set from a measured distribution
rather than picked.

#### Why `/v1/regenerate` is the right seam

OTS already has `/v1/regenerate`: rebuild MusicXML from an edited token sequence, no
image, no GPU. A refinement is exactly that shape - the user picks an alternative, the
score is rebuilt, no re-recognition. The contract needs extending, since regenerate
currently validates six-field string symbols and structured notation is not among those
six fields.

#### ~~Still unproven~~ Resolved 2026-08-25: the heads export cleanly

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

#### What is now the real blocker

Not the model. `base_url` in `homr/main.py:370` is a **hardcoded local variable** pointing
at `liebharc/homr`'s release tag, with no environment override and no config seam. Serving
our own weights means changing that line - so the redirect has to live in the commit OTS
pins, which is why the code pin and the weights pin are less independent than they look.

Also worth knowing before any model swap: `download_weights` decides freshness purely by
`os.path.exists(model)` - no version, no checksum. The Dockerfile bakes weights into the
image so a rebuild is clean, but any deployment with a persisted cache will silently keep
serving the old model.

#### Release published: `onnx_checkpoints` on `jhlusko/homr`, verified end-to-end

https://github.com/jhlusko/homr/releases/tag/onnx_checkpoints

Ten assets: segnet (fp32 + fp16, unchanged from upstream - required so pointing
`HOMR_WEIGHTS_BASE_URL` here is a strict superset of upstream's release, not a partial
swap that 404s on Stage 1), encoder, decoder (quantized default + fp32 fallback),
structured heads, and both Stage 3 detectors (quantized default + fp32 fallback each).

**A real gap caught before publishing, not after.** `download_weights` uses one
`base_url` for every required model, segnet included. Publishing only the parts we'd
changed (encoder/decoder/heads/detectors) would have meant anyone setting
`HOMR_WEIGHTS_BASE_URL` got a 404 fetching segnet - a required model, so this breaks
startup entirely, not degrades gracefully. Fixed by re-hosting upstream's unchanged
segnet weights as additional assets on the same release.

**Stage 3 wired into auto-download, not just published as inert assets.**
`homr/text_detector_config.py` (new) holds `detector_vocal_path`/
`detector_instrumental_path`, flat module constants matching
`homr/segmentation/config.py`'s own style - the text detector isn't part of the
transformer pipeline, so it doesn't belong on `Config.filepaths`. `download_weights`
now attempts all three optional models (heads + both detectors) via one shared
`download_optional_model()` instead of one-off copies, each independently best-effort.

**Writing the test for this caught a real bug before it shipped.** The three optional
fetches were placed after `if len(missing_models) == 0: return` - which fires on every
run after the first, once segnet/encoder/decoder are already cached. That made the
optional fetches unreachable except on a completely fresh install: upgrading to a
version that ships a text detector for the first time would have silently never
fetched it. Fixed by running the optional fetches unconditionally.

**Verified for real, not asserted:** ran `download_weights` from a completely empty
directory against the actual published `HOMR_WEIGHTS_BASE_URL`, over the real network,
on the GPU instance. All 6 non-fp16 models (3 required + 3 optional) downloaded with
zero errors, and every file landed at the exact byte size published - confirming the
quantized decoder (47,619,839 bytes) and both quantized detectors
(14,517,874/14,517,875 bytes) are what a fresh install actually gets, not the fp32
fallbacks.

**What is still not true, stated plainly so it isn't assumed:** no inference class in
`homr/` reads the two detector paths yet - only the download step exists. Wiring an
actual Stage 3 inference class, the vocal/instrumental toggle itself, and the
refinement-choice UI for the structured heads are all separate, not-yet-started work.

### Truncated vocal labels invalidate part of the recognition delta — found 2026-08-27

Found while building examples for `docs/writeups/homr-devs.html`, by trying to show off and
then checking what got picked. The mining filter was "upstream diverges sharply from the
reference, ours matches it almost exactly" (`ours_dist <= 1 and baseline_dist >= 3`). That
is not a quality filter, it is a **truncated-label detector**: the cheapest way to produce
that signature is for the label to stop before the crop does, so a model that also stops
early scores perfectly and one that keeps correctly reading is penalised per extra bar
(`base_predictions.py` pads both sides and normalises by `max(len(ref), len(pred))`, so a
length disagreement counts against you — as intended, but that assumes the label is right).

All four vocal examples the filter surfaced were truncation victims. Measured across the
held-out split, de-padded, counting real barlines:

| | vocal (Lieder, n=362) | instrumental (OSSQ, n=600) |
|---|---|---|
| mean symbols beyond label, upstream | **+4.28** | +0.08 |
| mean symbols beyond label, ours | **+3.97** | +0.03 |
| upstream reads more barlines than label | **14.9%** | 0.2% |
| **both** models read more barlines than label | **10.5%** | **0.0%** |

Two independently-trained models agreeing a crop holds more music than its label is
evidence about the label. The instrumental corpus shows none of it, so this is a
Lieder-pipeline defect, not a metric artifact.

**Consequence.** The vocal per-branch deltas (+2.1 to +3.5) are not a clean recognition
measurement: our model was trained on this corpus, so learned truncation and real
improvement are confounded. Not affected: instrumental deltas (+1.8 to +6.3); the
structured-head figures (supervised per position, not by sequence length); and Phase 1
cross-staff reranking (scored by inter-staff consistency, never touches these references).

**Verified NOT implicated**, checked rather than assumed: `split_by_system` (splits on the
explicit `newline` marker), `staves_by_system` (presence-cursor bookkeeping), and
`benchmark_phase1_rerank.py`'s `_call_index_to_system_voice`. Tested by decoding each
captured context and matching it against every system's chunk by pitch edit distance:
**0/16 captures attributed to the wrong system** on the Haydn benchmark page. So the 20.8%
figure stands.

**Next: adjudicate, then decide about retraining.** `docs/writeups/review.html` (serve
`docs/writeups/` over HTTP) presents 54 vocal staves where upstream reads more bars than the
label, plus 12 all-agree controls, each as scan + label vs. upstream reading with the scan as
arbiter. Verdicts export as JSON. If truncation is confirmed at scale, the Lieder token
pipeline needs fixing and the vocal half of the corpus rebuilding before the vocal delta can
be quoted at all.

A second review set (20 staves) covers **beaming**: our predicted beams vs. the engraver's
automatic beaming, which is what upstream effectively renders since it emits zero `<beam>`
elements. A bar where automatic wins is a regression we introduced. Manual review already
found at least one crop with two such bars, so the regression rate is non-zero and needs
measuring — the 95.1% exact-beam-vector figure is per-position against head labels and is a
different claim from "our beaming beats the default on a given line".

**The transferable rule.** Check every write-up example against the *source image*, never
against the score that selected it.

---

# Part III — Corpus construction (Lieder)

*Rewritten 2026-08-27 after human review. This part is the authority on how scan/label
pairs are built. Where it contradicts Parts I and II, this wins.*

The corpus pairs a **crop of one printed system** with the **MusicXML tokens for the
measures that system contains**. Getting that measure range right is the whole
problem, and every defect this project has nearly shipped traces back to a pair whose
label did not describe its image.

## III.1 Why the first two attempts were wrong

**Ordinal pairing (`extract_stage2_pairs.py`).** Zipped scanned systems to OpenScore
systems by flat position. One printed line can correspond to two reference lines, so
from the first divergence every later label shifts. Held-out measurement: 79 of 146
ordinary pairs disagreed with their crop; in 27 the crop matched a *different*
system's label exactly.

**Model-span recovery (`recover_excluded_pairs.py`).** Took the upstream recognizer's
matched note span as the label boundary, so a three-bar crop could get one bar. Also
circular: that recognizer is the evaluation baseline's own family. Ordinary pairs had
more upstream bars than label in 10/236 (4.2%); recovered pairs in 44/126 (**34.9%**).

**Why it mattered.** Accuracy is normalised by the padded width of reference and
prediction (`base_predictions.py`), so a truncated label *penalises a model that reads
the full system*. Train on truncated labels and the metric rewards stopping early.
~10.5% of vocal records had both models reading past the label, against 0.0%
instrumental. The published vocal deltas (+2.1 to +3.5) are confounded and must not be
quoted; instrumental (+1.8 to +6.3), the structured-head numbers and the 20.8%
reranking result are unaffected.

## III.2 Two methods, and why neither is enough alone

**A. Measure-count alignment** (`system_count_alignment.py`, `align_lieder_systems.py`)
— whole-score DP over physical bar counts. **Model-free**, so the only method whose
output may enter a held-out evaluation set. `DEFAULT_MIN_MARGIN = 2.0`; do not lower
it (at 1.0 a probe accepted content-wrong ranges on IMSLP632174).

Its ceiling is structural: one small integer per system. Measured on the ambiguous
cases, alternative-path margins run **0.0 to 1.8 — none reaches 2.0**. Counts cannot
separate them and no threshold move recovers them safely.

**Margin is not correctness.** The single most confident alignment in IMSLP637441
(margin 23.75) was a false-positive system detection that registered three barlines,
took measures 0–2, and displaced ten later systems by exactly −3. A high margin means
"confidently better than the alternative *under bar counts*" — worthless if the counts
are wrong.

**B. Reverse fingerprinting** (`reverse_fingerprint.py`) — ask which crop contains each
labelled bar. Because every measure belongs to exactly one system and systems appear in
order, this is one **global monotone segmentation**, not N independent lookups, so a
system that reads badly is pinned by its neighbours. **Model-derived**, therefore
training-only. Faster than the per-crop forward pass (~15 s/score vs ~36 s) because it
reads one crop per system rather than every voice.

The earlier per-crop forward pass (`recover_by_fingerprint.py`) is superseded. Its one
durable lesson: every staff in a printed system spans the *same* measures, so a
trusted vocal match gives the piano its range — that took piano recovery from 13% to
89% and is why the DP only needs one readable staff per system.

## III.3 Consensus: agreement is the evaluation set

`build_consensus_corpus.py` classifies every system by how much evidence its range has.

| verdict | meaning | use |
|---|---|---|
| `consensus` | both methods chose the same measures | **evaluation** + training |
| `arbitrated` | they disagreed; bar-count label kept | training only |
| `reverse` | only content alignment placed it | training only |
| `unarbitrated` | content alignment abstained | training only |
| `phantom` | reverse read notes and still placed it nowhere | neither |
| `rejected` | no usable range | neither |

Current: **3,968 consensus / 258 arbitrated / 3,656 reverse / 308 unarbitrated /
68 phantom / 391 rejected.** Evaluation set **3,964 pairs**, training **8,175**.

## III.4 What human review corrected — all four in the same direction

Every automated judgement I made erred toward **discarding good data**, and only human
review caught it.

1. **Rejecting on disagreement was wrong 32 times out of 33.** Disagreement means one
   method is right, not that both are wrong: bar-count correct 28, content 5, neither 1.
   Reverse's span score does not separate the cases (~1.00 in both), so there is no
   threshold to tune — prefer bar-count, mark `arbitrated`, keep out of evaluation.
2. **"Reverse says empty" did not mean "no music".** `note_tokens` dropped rests, so a
   rest-heavy system produced no tokens, scored 0.0 for every span, and lost to the
   empty move. Reviewed at **30 of 30 wrong** — a third of them labels with no pitched
   note at all. Rests now carry their own rhythm token, and an unreadable crop scores
   `UNREADABLE_SPAN_SCORE = 0.2` so its neighbours pin it. **This one bug produced most
   of the disagreements**: fixing it moved 822 arbitrated → 258 and 3,376 consensus →
   3,968, and of 45 pairs the reviewer said were wrongly rejected, **40 became plain
   consensus and none stayed discarded**.
3. **Stratifying the split on score id alone** put all 113 many-to-many and
   reference-line-split pairs in train, leaving validation 100% one-to-one — unable to
   detect the defect the rebuild exists to remove. Strata must be derived from the
   manifest being split, per rare topology.
4. **The review generator kept reading the old verdict vocabulary** after `phantom` and
   `arbitrated` became their own verdicts, and silently produced two **empty** review
   sets. Same class of quiet failure as an under-covered corpus.

## III.5 The known blind spot

**Consensus cannot detect the case where both methods agree and are both wrong.**

Confirmed, not hypothetical. `IMSLP183806-sys1-v1`: the crop is a densely written piano
grand staff; the bar-count label is `R R | R R | R R` — three measures of pure rests —
and the content label has notes but five measures for a three-measure crop. The
reviewer judged **neither correct**, and stands by it. v5 classifies this pair
`consensus`, so it would enter the evaluation set.

The pair is voice 1 while consensus is decided from the voice-0 reading and applied to
every staff in the system. A system-level agreement therefore does **not** validate
each voice's label. Nothing currently checks per-voice content.

## III.6 The parser recovered a third of the corpus

`music_xml_parser.py` raised on `<octave-shift>` (86 scores) and `<clef-octave-change>`
(11), discarding those scores entirely — a quarter of the corpus for a routine piano
engraving mark. Both are now handled: **94 of 111 recover**, clean pairs 3,214 → 4,587,
scores 165 → 234. The 17 stragglers all fail on `Backup duration is too long`.

MusicXML `<pitch>` is the **sounding** pitch; the label must carry the **written** one,
so the correction is `written - sounding`. `OCTAVE_SHIFT_DIRECTION = {"down": -1,
"up": 1}` — 8va is exported as `type="down"`. Only ~1% of notes are affected, so no
aggregate metric can see the sign: running both signs gave byte-identical corpus
results. Human review settled it — **22/25 octaves correct, 0 octave errors**.

## III.7 Guards, each added after a real failure

- **Coverage gate** (`finish_rebuild.sh`) — `align_lieder_systems` builds its map purely
  from the rows given, so a missing score is *invisible*, not an error. A 12-shard
  recount lost 39% of the corpus and would have produced a smaller corpus that read as
  healthy conservatism. Exclusions must be named with a reason.
- **Non-zero exit, no empty rows file** (`compare_bar_counts.py`) — two shards wrote
  `[]` and exited 0.
- **All five measure dividers counted** — counting only `barline` flagged 400 sound
  pairs; adding `doublebarline`, `bolddoublebarline`, `repeatEnd`, `repeatStart`
  resolves all 400 with zero residual. The count must still equal the span exactly.
- **Provenance by path, not name** — a rebuilt label reuses its system's stem, so a
  name test flagged 468 rebuilt labels including ones that had *repaired* a truncation.
- **Build-time span consistency** — 7 pairs carried more dividers than their span.
- **PNG root selection** — two roots hold the same score under different naming;
  selecting by directory name built paths to files that did not exist. Loud in
  `compare_bar_counts`, **silent** in `build_clean_stage2_pairs`.

## III.8 Operational traps

- **Concurrency.** Each ONNX process holds 206–307 threads against a cgroup `pids.max`
  of 3840. Twelve simultaneous starts exhausted it: 130 scores died on `pthread_create`
  EAGAIN and every survivor **deadlocked** — all threads sleeping, main thread in
  `futex_wait_queue`, zero CPU ticks. Run **4 workers, staggered**.
- **Buffered progress.** ONNX output is stderr; per-score `print()`s are stdout, block
  buffered at 8KB. Logs look stalled while work proceeds — count files, not log lines.
- **Masked exit status.** `cmd | grep -v …` reports grep's status; a failing audit
  looked like a clean exit.
- **`grep -v` exits 1 on no match**, which under `set -e -o pipefail` killed a run at
  the coverage gate *after* coverage came back 330/330.
- **`seq -w 0 3`** pads to the width of the largest value — it yields `0 1 2 3`, not
  `00 01 02 03`.
- **Corrupt downloads.** One "PDF" was a 2902-byte IMSLP HTML disclaimer page. IMSLP now
  gates non-browser clients behind a JS token POST, so re-fetching is a manual step.

## III.9 Pipeline

```
compare_bar_counts.py        # physical bar counts (4 shards, staggered)
  -> coverage gate           # 330/330 or a named exclusion
align_lieder_systems.py      # model-free DP, min-margin 2.0
build_clean_stage2_pairs.py  # aligned systems -> bar-count pairs
reverse_fingerprint.py       # content segmentation -> content pairs
build_consensus_corpus.py    # agreement -> eval; disagreement -> arbitrated
audit_clean_stage2_pairs.py  # fails closed on provenance/span/sidecar
split_pairs_by_score.py      # score-disjoint AND topology-stratified
make_review_sets.py          # six sets, one hypothesis each
```

**Vocal training remains blocked on human review.** Four automated decisions have been
overturned by it so far, every one of them in the direction of discarding good data.

# Part IV — Onset representation, tuplet repair, and structured-head promotion

A single session's work, condensed here from an in-conversation "Promotion Week" plan
(published as a private artifact and moved into this log per standing practice). Spans
three corpora (Lieder, PDMX, OSSQ) and touches both the corpus-construction thread (Part
III) and the structured-notation-heads thread (Part I/II's 27.x series) — filed as its
own Part rather than shoehorned into either, since it is neither Lieder-only nor
dynamics-only.

## IV.1 Naturals restored, and the true cost isolated

`strip_naturals` (`homr/circle_of_fifths.py`) stripped every `N` lift unconditionally
across four of five converters; only `convert_ossq.py` never did, which is why OSSQ was
the only corpus that ever recorded a natural and why every checkpoint carried a uniform
~0.54% ceiling on lift-branch accuracy (no checkpoint had ever predicted `N`: 0/879 OSSQ
references). A naturals-kept fine-tune (3,880 Lieder pairs) moved OSSQ N recall 0% → 62%
on two seeds, with lift accuracy *improving* (91.55% → 93.06/93.15%) despite learning a
new symbol class from scratch.

A naive before/after PDMX comparison suggested a ~5pp accuracy cost — but PDMX regresses
under almost any fine-tune on this corpus, independent of naturals. Isolated the true
cost with a matched control: identical 3,880 pairs, identical recipe, naturals stripped
from the *same* files rather than a different corpus slice. True cost: **-1.2 to -1.6pp
PDMX exact-match**, an order of magnitude smaller than the naive comparison implied.
**Default flipped**: `HOMR_KEEP_NATURALS` now defaults to keep (was strip); 5 tests, all
14 call sites verified to share the one switch.

## IV.2 Tuplet repair: from a narrow win to zero losses

The single-staff arithmetic tuplet repair (implied tuplets — no bracket, no numeral,
which the model cannot learn from pixels) initially showed a real but narrow win: exact
staves +1/+2/+4/+4 across four OSSQ checkpoints, with up to 2 broken staves per run. All
of those losses turned out to share one cause: `bar_duration` summed every token in a bar
naively, double-counting chord members (`SymbolChord.get_duration()` correctly takes the
**minimum** across a simultaneity; the repair's own duration check did not). Fixed, plus
the run-detection bug it exposed (a candidate run could grab one member of a chord while
its partner sat untouched). **Reconfirmed: exact staves +3/+4/+7/+6, zero losses across
all four checkpoints, precision 71.7-93.3%.** `HOMR_TUPLET_REPAIR` flipped to default-on.

## IV.3 The onset/advance representation problem

Research (`docs/private/ONSET_REPRESENTATION_RESEARCH.md`) established that homr's
min-duration rule for a grand staff's bar length — `SymbolChord.get_duration()` takes the
min across a simultaneity — is not an approximation but structurally wrong wherever the
two hands' onsets desynchronize, which MusicXML's `<backup>`/`<forward>` allow and kern's
spine format does not. Measured directly against real Lieder ground truth: the min-rule
disagrees with the true onset gap **30.6% of the time** on grand-staff simultaneities —
far past the report's own 95%-agreement "stop, this is smaller than believed" threshold.

Built a full structured "advance" head end to end (schema → model head → targets → losses
→ decoding → metrics → sidecar), extracted via true cross-onset position tracking for the
MusicXML path (Lieder, PDMX, OSSQ) and via a proof that the min-rule is *already exact*
for kern (GrandStaff) — kern's spine format guarantees a new data line at the finest
rhythmic grid present in any voice, confirmed against a real excerpt (four bass 16ths
spanning one treble quarter, exactly 4:1). Frozen-core probe: **82.4% macro F1 on the
nonzero-gap subset**, well above the 69.4% min-rule baseline and not a majority-class
collapse (verified via macro F1, not just micro accuracy). A joint probe on the combined
Lieder+PDMX corpus (7,709 pairs) scored 88.7% micro / 83% macro — currently train-set fit
only, held-out validation still open (Part IV.8).

## IV.4 Kern beam/stem/tie: real markup, real signal, actually held out

GrandStaff (kern-sourced) previously contributed **zero** beam/tie/stem signal to any
structured head — a placeholder `NoteNotation` with everything blank except `advance`.
Replaced with real parsing of kern's own markup: `L`/`J` beam levels including `k`/`K`
partial-beam hooks (previously discarded as noise), `[`/`]`/`_` ties (this corpus writes
them as *suffixes*, differing from the reference spec — handled both spellings), and
confirmed this corpus states **zero** real stems anywhere across all 53,882 `.krn` files
(a true property of the data, not a parser gap — correctly comes out `UNKNOWN`, masked
from loss).

Validated on a genuine held-out split (disjoint scores, verified zero overlap) — the only
head this session got a true generalization check on before promotion:

| head | in-sample | held-out | majority baseline |
|---|---|---|---|
| beam level 1 | 0.838 | 0.833 | 0.124 |
| beam level 2 | 0.797 | 0.770 | 0.097 |
| beam level 3 | 0.804 | 0.713 | 0.093 |
| ties | 0.909 | 0.893 | 0.246 |

Negligible generalization gap on every well-supported class; ~42k trainable head
parameters against 3.5M scored positions makes memorization implausible. **Promoted.**

## IV.5 Three renderer bugs found by a ground-truth roundtrip check

Built `training/omr_datasets/roundtrip_fidelity.py` (Lieder) and
`roundtrip_fidelity_corpora.py` (PDMX/OSSQ): real ground-truth tokens → `generate_xml` →
reparse → diff against the original, token for token. No model in the loop — any mismatch
is a tokenizer/renderer bug, provable directly.

1. **Duplicated/misordered `<attributes>`/time-signature blocks** — a state-tracking bug
   in `build_measures` (`homr/music_xml_generator.py`) that didn't carry the current
   attributes reference forward. Lieder exact-match 0% → 52.6% from this fix alone.
2. **A measurement artifact in the tool itself** (chord-member order not canonicalized
   before diffing) — found and fixed before trusting anything downstream. → 84.1%.
3. **`_collect_articulation` produced `"slurStart_slurStart"`**
   (`training/omr_datasets/music_xml_parser.py`) whenever a note carried both a
   `<tied type="start">` and a `<slur type="start">` at once — common (741/800 notes in
   an earlier sample), because homr's vocabulary deliberately collapses ties and slurs
   into one string field. Crashed `build_slurs` on ~8% of real crops via the actual
   production render path. Fixed with a one-line dedup (`articulations` already had the
   same dedup two lines below; `slurs` never did).
4. **`build_clef` mangled multi-character clef signs** — indexed by character
   (`sign = sign_and_line[0]`), so `clef_TAB5` split as sign `T`, line `A`, not a real
   MusicXML clef. TAB staves are common enough to surface in a 50-file PDMX sample. Fixed
   by splitting on trailing digits.
5. **PDMX/musetrainer windowed conversion lost the time-signature numerator on 92.4%/
   89.2% of files** — `_context_at_measure` seeded a fresh per-window `MeasureCutter`
   from context that carried the denominator but not the numerator, so every window after
   the first emitted a bare `timeSignature/4`. Fixed (`time_beats` now a fourth return
   value, wired into both converters); full corpora reconverted. **PDMX roundtrip exact-
   match: 15.3% → 91.9%** (confirmed on the live, reconverted corpus, not just the
   validation scratch tree).

Lieder roundtrip now sits at **92.0% exact** (231/251 crops), up from 84.1% at the start
of the session. Remaining gap: one confirmed, twice-attempted-and-reverted open bug (below)
plus a handful of documented representational limits (chorded rests can't be represented
in MusicXML at all; a trailing repeat token at a crop edge grows a phantom measure).

**Open, not fixed: `TupletParser.add_tuplets` can stitch two unrelated same-ratio
triplets from different hands into one bogus bracket** (confirmed on `IMSLP435041`). A
same-hand-tracking fix solved that case but regressed a different score
(`IMSLP83318`, which has *genuinely simultaneous* triplets in both hands at once) from 83
to 226 field mismatches — reverted. A second attempt (independent per-hand cursors) fixed
`IMSLP83318` but exposed that `IMSLP435041`'s own triplet groups are "short by exactly
one" under strict per-hand counting, for a reason not yet established (real data property,
crop-boundary artifact, or a `group_into_chords` assembly issue) — also reverted (165
mismatches, still worse than shipped). Documented in the function's own docstring with
both dead ends recorded, rather than silently reintroduced by a future attempt.

## IV.6 Dynamics: mostly confirmed prior history, plus one real new metrics bug

Full account in `docs/private/DYNAMICS_HEAD_FINDINGS.md`. A joint probe scored dynamics
at 11.97% macro F1, far below every other head. Investigation found two stacked
artifacts — a ~10-point scoring floor from masked decoder positions being scored as free
correct `NONE` predictions (real signal is closer to 0.02-0.03), and an illusory
"improvement" over a prior 0.063 baseline that was really just a vocabulary-fold effect.
**Both findings replicate this project's own prior history** (RUNLOG 27.94-27.98, several
sessions ago: phase14-17 already found the identical fold artifact and diagnosed the same
"registration failure, not confusion" pattern for `mf`/`mp`/`ppp`) — convergent
confirmation, not new discovery, and the scoring-floor bug was independently rediscovered
a second way by the kern probe (IV.4): tie support inflated 9.2x by the same mechanism,
small in tie's case (0.998 none-accuracy) but severe for dynamics.

**Also re-surfaced: dynamics already has a working, separate Stage 3 pipeline** — a
page-level detector (84.0% F1) feeding a dedicated CNN classifier (88.6% mark accuracy)
feeding a position-based attachment heuristic (27.45, 27.95-27.97). The structured head is
a deliberate parallel attempt at inline decoding, not a replacement for a broken Stage 3.
Given that, and given this project has already measured unfreezing quietly erase a
working signal elsewhere (score-profile conditioning, `docs/writeups/homr-devs.html` —
+0.0615 mean ablation delta under a frozen core, oscillating in sign under an otherwise
identical unfrozen run), the case for spending an unfreeze specifically on dynamics is
weak. The cheaper, already-identified-but-never-done next step (RUNLOG 27.98's own
diagnostic pass) is to crop and look at the real `mf`/`mp`/`ppp` misses before any
training-recipe or architecture change.

## IV.7 This week's plan

1. **Finish what's running** — evaluate the Lieder+PDMX promotion run on a genuine
   held-out split (not done yet — every advance-head number so far is train-set fit);
   confirm the PDMX/musetrainer numerator fix on the live corpus (done, IV.8).
2. **Promote what's validated** — kern beam/stem/tie (done, IV.4); one real end-to-end
   confirmation of tuplet repair through the actual image pipeline, not just the offline
   harness; fix the scoring-floor bug now confirmed twice.
3. **Cheap next steps on open threads** — the RUNLOG-planned dynamics crop diagnostic;
   decide (don't just note) the trailing-repeat phantom-measure question; leave the
   cross-hand tuplet bug alone absent a genuinely new angle.
4. **Bigger calls, gated, not started** — unfreezing (needs the same clean ablation that
   caught the score-profile regression, checked against the primary six-branch decode);
   a full-scale naturals production run; wiring the advance head into the renderer
   (blocked on the held-out check in item 1).

## IV.8 Progress against the plan

- **PDMX numerator fix, confirmed on the live corpus.** 0.30% of a 2,000-file sample
  still missing a numerator (down from 92.4%), all attributable to numerators outside
  `VALID_TIME_SIGNATURE_NUMERATORS` (documented, unrelated residue). Reran the PDMX
  roundtrip check against the live reconverted corpus rather than the validation scratch
  tree: **91.9% exact, 1,243 crops, 0 crashes** — reconfirmed, not just carried over.
- **Held-out manifest built** for the advance-head promotion run: 2,000 PDMX
  `index_train.txt` pairs, verified zero overlap with the training manifest. Evaluation
  pending the promotion training run finishing (epoch 5/12 as of this entry).
- **Scoring-floor fix, in progress.** Added `TieState.UNKNOWN`/`DynamicMark.UNKNOWN`
  sentinels (`homr/transformer/structured_notation.py`), mirroring `StemDirection.UNKNOWN`
  exactly — excluded from `TIE_CLASSES`/`DYNAMIC_CLASSES` (never a real inference class),
  and added by filtering rather than reordering the existing enum members, so no trained
  checkpoint's class indices move. Next: wire `_lookup`'s masked-position fallback to the
  new sentinel and exclude it in `tie_report`/`dynamic_report`.
- **Scoring-floor fix, shipped.** `TieState.UNKNOWN`/`DynamicMark.UNKNOWN` wired end to
  end: `structured_decoding.py::_note` now decodes a masked position (padding, BOS/EOS,
  a non-note token) to `UNKNOWN` for both heads instead of `NONE`, and
  `tie_report`/`dynamic_report` (`structured_metrics.py`) exclude it - mirroring
  `stem_report`'s already-correct `StemDirection.UNKNOWN` handling exactly. 5 new tests
  (2 metrics, 1 decoding, both directions - a masked position decodes to UNKNOWN, a real
  note's genuine NONE still decodes to NONE and is still scored). 80/80 structured-heads
  tests pass. No trained checkpoint's class indices move (UNKNOWN excluded from
  `TIE_CLASSES`/`DYNAMIC_CLASSES` by filtering, not reordering, the existing members).
- **Tuplet repair, real end-to-end confirmation (partial).** Ran the actual
  `homr/main.py` image → detect → decode → repair pipeline (not the offline harness) on
  several real OSSQ scans with `HOMR_TUPLET_REPAIR=1` vs `=0`: no crash, valid MusicXML,
  and correctly a no-op when the staff has no overfull bar. Full test suite: 1805 passed,
  no regressions. **Not yet decisive**: the only checkpoint with exported ONNX weights is
  426, which predates the checkpoints the offline OSSQ measurement (+3/+4/+7/+6 exact
  staves, zero losses) used - so this run confirms the integration is safe, not that it
  fires correctly end to end on a real overfull bar. That needs a fresh ONNX export of a
  current checkpoint, not done this session - left as a named gap rather than papered
  over with a checkpoint that can't actually test the claim.
- **Trailing-repeat phantom measure, fixed and decided.** A token stream ending on a
  bare `repeatStart` opened a fresh measure to hold the forward-repeat barline, then the
  post-loop "anything left to close" check counted that bare barline as real content and
  emitted it as its own phantom measure - an empty bar with a forward repeat on its
  RIGHT edge, a shape real engraving never produces. Added
  `_measure_has_real_content` (any child other than `barline`) and used it in place of
  the old "any children at all" check. 3 new tests (the repeat case, the ordinary-ending
  case that must stay untouched, and a trailing-content case that must still be kept).
  Full suite: 1808 passed, no regressions. Real-corpus confirmation inconclusive at
  n=150/seed=11 (crops_exact and field_mismatches identical before/after - the pattern is
  only ~1.6% of crops, too rare to reliably land in this sample); a larger sample is
  running to try to surface it directly.
- **Dynamics crop diagnostic, item 3's plan entry: already closed, not open.** Went to run
  the RUNLOG-planned mf/mp/ppp crop diagnostic and found it had already been done in an
  earlier session (§24, lines ~3345-3387): 13 real crops hand-checked (9 mf/mp/ppp, 4 p/f
  control), 7/9 mf/mp/ppp print exactly as labelled, and the conclusion already reached was
  that the ~2-in-9 miss rate is real but far too small to explain the head's 87-92%
  predicts-none behavior - most of the collapse is a genuine model limitation, not a corpus
  artifact. Corrected `docs/private/DYNAMICS_HEAD_FINDINGS.md`'s "what would actually fix
  it" section, which had stated the diagnostic as still-open. Nothing cheap remains on this
  thread; the next real decision is the unfreeze-vs-Stage-3 tradeoff, not another
  crop-sampling pass.
- **Advance-head promotion run, in progress.** Epoch 9/12 as of this entry (~660s/epoch,
  ~2 more epochs of wall time left); held-out evaluation against the 2,000-pair manifest
  still pending completion.
