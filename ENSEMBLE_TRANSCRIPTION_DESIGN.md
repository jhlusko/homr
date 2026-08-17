# Generic ensemble transcription with structured notation heads

**Status:** accepted design; implementation not started

**Date:** 2026-08-15

**Initial corpus:** OpenScore String Quartet OMR (OSSQ-OMR)

**Future corpus:** OpenScore Lieder

## 1. Executive summary

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

## 2. Goals

### 2.1 Primary goals

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

### 2.2 Secondary goals

- Improve MusicXML round-trip fidelity through MuseScore.
- Make explicit notation independently measurable from pitch and rhythm.
- Capture confirmed and corrected beam, stem, slur, layout, and cross-staff choices
  as provenance-rich future training data.
- Make model capabilities discoverable so an older checkpoint can coexist with
  newer provider and review code.

## 3. Non-goals for the first experiments

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

## 4. Current architecture and relevant constraints

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

## 5. Design principles

### 5.1 Preserve the pretrained core

New semantics should be represented by new heads instead of expanding the rhythm
vocabulary. Changing the rhythm vocabulary would alter the most important existing
softmax and embedding matrices and would entangle notation fidelity with note/rest
sequence accuracy.

Checkpoint loading must explicitly distinguish expected new parameters from an
accidental mismatch. `strict=False` by itself is insufficient; loading should
validate an allowlist of missing new-head and adapter parameters and reject all
other missing or unexpected keys.

### 5.2 Factor independent musical dimensions

Avoid a combinatorial token such as
`slur2StartAbove_slur1Stop_beam2BackwardHook_stemDown`. Beam level, beam state,
stem direction, slur identity, slur event, and slur direction have distinct
semantics and class frequencies. They should have distinct targets and losses.

### 5.3 Treat metadata as optional evidence

Instrument and part information is often known before recognition. It is useful
for layout, clef priors, and repair, but it can be wrong or incomplete. Every
conditioning field therefore has an unknown value, training uses context dropout,
and layout rules can be overridden by visual evidence or a reviewer.

### 5.4 Prefer explicit refusal and review to silent repair

An output can be valid MusicXML and still be musically wrong. Structural repair
must report its evidence, alternatives, and any mutation. Ambiguous staff grouping,
measure mismatch, or spanning-notation damage should yield a review item or a
refusal rather than plausible-looking output.

### 5.5 Preserve exact source-image identity

All review geometry and training corrections must be tied to the exact normalized
page raster used for inference, its checksum, model revision, preprocessing
revision, and token vocabulary/head schema versions.

## 6. Target architecture

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

## 7. Optional score-profile conditioning

### 7.1 Contract

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

### 7.2 Use in layout

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

### 7.3 Use in staff recognition

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

### 7.4 Context dropout

During training, randomly remove the entire profile and independently mask fields.
The exact probabilities should be configured and recorded with the run. A sensible
starting point is:

- 30% of examples receive no profile;
- another 30% receive a partially masked profile;
- 40% receive the complete available profile.

This is a starting hypothesis, not a fixed constant. Evaluation must include both
conditioned and unconditioned inference.

## 8. Staff and system detection

### 8.1 Responsibility boundary

Four-staff system detection is not solely a U-Net task. Stage 1 provides the staff
and symbol masks; subsequent geometry creates physical staffs and groups them into
systems. Beam and slur heads do not require a segmentation change because the
semantic Transformer sees the dewarped source pixels.

### 8.2 Deterministic grouping before another neural model

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

### 8.3 Exceptions

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

### 8.4 Human correction

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

## 9. Structured beam and stem heads

### 9.1 Why MusicXML beam vectors are the canonical target

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

### 9.2 Stem direction

Add a separate actual-stem-direction head:

```text
NOT_APPLICABLE | UP | DOWN | NONE | DOUBLE
```

Missing or unreliable source labels use a dataset-side `UNKNOWN` sentinel that is
masked out of the loss; `UNKNOWN` is not an inference class. For ordinary flagged
notes, flag orientation follows `UP` or `DOWN`. For a beamed group, the head records
the visible stem direction independently of beam connectivity.

### 9.3 Output shape

The first model adds seven output projections to the shared decoder hidden state:

- six `beam_level_N` categorical projections;
- one `stem_direction` projection.

They are output-only in the first experiment. They do not contribute embeddings to
the next autoregressive step. This minimizes checkpoint disruption and provides a
clean ablation.

### 9.4 Masking and losses

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

### 9.5 Materializing automatic beams

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

## 10. Structured slur heads

### 10.1 Requirements

The representation must preserve:

- start and stop events;
- a stop and a new start on the same note;
- above/below placement;
- more than one concurrent slur;
- endpoint identity across intervening notes;
- system/page-boundary incompleteness;
- future line style without replacing the initial schema.

### 10.2 Slot representation

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

### 10.3 Canonical slot assignment

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

### 10.4 Checkpoint-compatible migration

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

### 10.5 Slur validation

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

## 11. Semantic data model and MusicXML generation

### 11.1 Structured symbol representation

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

### 11.2 Parser requirements

The MusicXML parser must preserve:

- all `<beam number>` elements and hook values;
- `<stem>` direction;
- every slur element, number, type, placement/orientation, and supported line
  style;
- voice, staff, chord, grace, and temporal identity required for canonicalization;
- measure and part provenance used for leakage-safe grouping.

Parsing should produce validation findings rather than globally dropping a file
for a rare unsupported marking.

### 11.3 Generator requirements

The MusicXML generator must:

- emit explicit beam elements at every applicable predicted level;
- emit stem direction when supported;
- pair structured slur starts and stops by canonical slot;
- emit above/below placement on slur starts;
- retain raw versus repaired notation provenance;
- remain able to generate output from legacy symbols/checkpoints;
- reject impossible combinations before storing output;
- pass a headless MuseScore load/render smoke test.

### 11.4 Capability manifest

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

## 12. Cross-staff context and repair

Cross-staff work is staged from least invasive to most invasive.

### 12.1 Stage A: deterministic consistency analysis

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

### 12.2 Stage B: targeted repair proposals from existing alternatives

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

### 12.3 Stage C: learned variable-staff context adapter

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

### 12.4 Why not a fixed four-staff decoder

A fixed quartet tensor would make the initial benchmark easy but would block or
complicate piano, Lieder, trios, orchestral reductions, missing staves, divisi, and
partial crops. A masked set/sequence of staff summaries gives the model quartet
context without encoding quartet as the architecture.

## 13. OSSQ dataset adapter

### 13.1 Constructed local assets

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

### 13.2 Corpus evidence for new heads

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

### 13.3 Training record

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

### 13.4 Exact synthetic versus edition-noisy scanned labels

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

### 13.5 Splits and leakage prevention

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

### 13.6 Dataset mixing

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

## 14. Training plan

### 14.1 Phase 0: reproduce and freeze baselines

Before architecture changes:

1. evaluate the pinned pretrained checkpoint on existing smoke/system tests;
2. evaluate page-local HOMR on the fixed OSSQ test scores;
3. record layout, staff, pitch, rhythm, slur, MusicXML, and runtime metrics;
4. archive manifests, model hashes, commands, and raw predictions;
5. verify the original training data conversion still reproduces a known baseline.

No fine-tuning result is meaningful without this frozen comparison.

### 14.2 Phase 1: label and round-trip validation

- Implement structured parser targets.
- Materialize automatic beams with a pinned MuseScore.
- Render source and materialized scores and verify visual equivalence.
- Canonicalize slur slots and report invalid/overflow spans.
- Confirm that generated structured MusicXML reloads in MuseScore.
- Reconcile OSSQ exclusions and system counts.
- Produce per-class support tables by split.
- Manually inspect examples of every beam state and secondary slur class.

### 14.3 Phase 2: new-head-only training

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

### 14.4 Phase 3: limited joint fine-tuning

- Unfreeze the last decoder layers and new heads first.
- Retain a general-data replay mixture.
- Use a lower learning rate for pretrained layers than for new projections.
- If needed, unfreeze the visual encoder last at an even lower rate.
- Train all existing semantic heads, not only the lift head.
- Retain checkpoint selection based on a multi-objective validation report rather
  than new-head accuracy alone.

The current fine-tuning path that freezes most of the model and trains only lift is
not suitable for this domain adaptation and needs a separate explicit mode.

### 14.5 Phase 4: score-profile conditioning

- Add profile embeddings behind a zero-initialized gate.
- Train with full/partial/no-context examples.
- Compare unconditioned inference with the original baseline.
- Measure conditioned gains by head and by instrument.
- Test deliberately incorrect profiles and ensure they reduce confidence or create
  review warnings rather than catastrophically overriding the image.

### 14.6 Phase 5: cross-staff work

Implement and measure in order:

1. deterministic consistency findings;
2. targeted top-k repair proposals with a human decision;
3. optional variable-staff learned context adapter.

Do not proceed to the learned adapter merely because GPU time is available. Proceed
when the error taxonomy shows remaining errors that truly require simultaneous
staff context.

### 14.7 Compute and run configuration

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

## 15. Evaluation

### 15.1 Existing-head non-regression

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

### 15.2 Beam metrics

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

### 15.3 Stem metrics

- up/down macro F1;
- exact accuracy on flagged notes;
- exact accuracy on beamed notes;
- beam-group stem consistency;
- performance by voice count and chord density.

### 15.4 Slur metrics

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

### 15.5 Layout and system metrics

- physical-staff recall and precision;
- exact systems per page;
- exact physical staffs per system;
- part/staff assignment accuracy;
- dropped, duplicated, merged, and split staff counts;
- correct handling of incomplete systems;
- conditioned versus unconditioned results;
- rate of pages requiring structural review.

### 15.6 Cross-staff and end-to-end metrics

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

### 15.7 Calibration and review utility

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

## 16. Human-in-the-loop contract

An external review system already supports confidence-driven token questions,
geometry-bound crops, cumulative corrections, regeneration, content signatures,
engine comparison, durable flags, and correction capture. HOMR should expose a
generic contract rather than depending on one client.

### 16.1 Head identifiers

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

### 16.2 Beam review

A beam correction applies a validated full per-note beam vector atomically, even if
the UI lets the reviewer change one level. This prevents an isolated per-level edit
from creating an impossible vector.

Show:

- a source crop wide enough to include neighboring notes;
- the recognized vector by rhythmic level;
- the top alternatives and confidence per uncertain level;
- stem direction;
- a warning if the proposed vector breaks group continuity.

### 16.3 Slur review

Show a system-width crop when possible, because a measure-local crop may omit an
endpoint. Present primary and secondary spans separately with above/below placement.
An endpoint correction must rerun pairing validation before MusicXML regeneration.

### 16.4 Structural review

Structural questions precede token questions. A stale layout invalidates staff
crops, token identities, attention hints, cross-staff analysis, and generated
MusicXML. Re-running only after token corrections would discard work or attach it to
the wrong staff.

### 16.5 Cross-staff review

For a consistency finding, show:

- the aligned source bands for all affected staves;
- each decoded measure;
- the precise invariant that failed;
- a bounded proposed token change, if any;
- resulting measure totals and downstream warnings;
- choices to accept, select another alternative, defer, or open an editor.

### 16.6 Training capture

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

## 17. Page boundaries and assembly

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

## 18. Lieder extension and lyrics

### 18.1 Reused architecture

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

### 18.2 Lyric stage

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

### 18.3 Reuse without pretending the lyric head is pretrained

The music encoder/decoder and notation heads are reused. The new lyric-region,
text-recognition, and alignment modules still require training or initialization
from an appropriate pretrained OCR/text model. This is not full-model training from
scratch: the stable music result and note anchors are inputs to an independently
versioned lyric capability.

Keep lyric capability optional. Quartet checkpoints and pages without lyrics should
not allocate lyric decoding work or emit empty lyrics as if they were confident
predictions.

## 19. Backward compatibility and deployment

### 19.1 Checkpoints

- Old checkpoints load into the unchanged legacy architecture.
- New architecture loading an old checkpoint initializes only allowlisted new heads
  and adapters.
- A checkpoint manifest declares exactly which heads are trained.
- An untrained new head is not exported as a supported capability.
- New checkpoints may continue to emit legacy slur output during migration.

### 19.2 Token/index formats

- Legacy token files remain readable.
- Structured token schema is versioned and preferably JSON-based.
- Dataset indexes include score grouping and provenance.
- Converters reject mixed schema versions unless explicitly migrated.

### 19.3 ONNX

ONNX encoder and decoder export must name outputs rather than depend on tuple
position. Dynamic-cache behavior remains unchanged for output-only heads.

The decoder inference wrapper should expose a dictionary of head logits and use the
checkpoint manifest to bind ONNX output names. This prevents adding one head from
silently shifting every downstream positional output.

### 19.4 Provider/API

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

## 20. Concrete implementation areas

Likely HOMR changes, grouped by responsibility:

### Semantic vocabulary and structures

- `homr/transformer/vocabulary.py`
- `training/transformer/training_vocabulary.py`
- new structured-symbol schema and migration helpers

### Parser and dataset adapters

- `training/omr_datasets/music_xml_parser.py`
- `training/omr_datasets/convert_lieder.py`
- new `training/omr_datasets/convert_ossq.py`
- MuseScore beam-materialization helper
- explicit split-manifest tooling

### Transformer model and inference

- `training/architecture/transformer/decoder.py`
- `training/architecture/transformer/tromr_arch.py`
- `training/transformer/data_loader.py`
- `training/transformer/metrics.py`
- `homr/transformer/decoder_inference.py`
- `homr/transformer/staff2score.py`
- `homr/transformer/configs.py`
- `training/onnx/convert.py`

### Training orchestration

- `training/transformer/train.py`
- `training/transformer/mix_datasets.py`
- checkpoint manifest/load validation
- explicit fine-tuning modes and parameter groups
- effective sampler-stat logging

### Layout and system context

- `homr/staff_detection.py`
- `homr/brace_dot_detection.py`
- `homr/staff_parsing.py`
- `homr/staff_position_save_load.py`
- new system-partition scoring module
- new score-profile module
- optional learned staff-context adapter

### MusicXML and validation

- `homr/music_xml_generator.py`
- structured beam/slur validator
- page-state summary and raw/repaired provenance

### Tests

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

## 21. Experiment matrix

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

## 22. Acceptance gates

### Gate A: data correctness

- No split leakage by score/work.
- Every retained synthetic example loads and renders.
- Beam materialization is visually equivalent on the controlled corpus.
- All label warnings and exclusions are accounted for.
- Representative examples of every supported class are manually inspected.
- Structured parser -> generator -> parser round trips preserve supported labels.

### Gate B: checkpoint compatibility

- Existing checkpoint loads with only expected new parameters absent.
- Frozen-core existing outputs remain identical.
- Old token files and old inference models remain supported.
- ONNX output naming is manifest-driven and covered by tests.

### Gate C: new-head usefulness

- Beam heads outperform automatic beaming on held-out visible exceptions, not only
  on common regular groups.
- Stem direction beats the source/layout heuristic baseline.
- Structured slur endpoint and direction metrics improve over the legacy head.
- Rare-class metrics include sufficient support or are explicitly inconclusive.

### Gate D: no material general regression

- Existing fixed smoke and system-level benchmarks remain within a predeclared
  non-regression tolerance.
- Single-staff and grand-staff inference remain valid without a score profile.
- Runtime and memory increases are measured and acceptable.

The numeric tolerance should be declared after reproducing B0 variance and before
examining held-out experiment results.

### Gate E: safe conditioned behavior

- Missing context reproduces the unconditioned path within the expected numerical
  tolerance.
- Correct context improves the target metrics.
- Incorrect context cannot silently force structurally impossible output.
- Profile deviations become evidence/review findings.

### Gate F: human-review value

- Structural corrections reliably invalidate and regenerate dependent output.
- Beam/slur questions show sufficient source context.
- Cross-staff proposals have measured precision and explain the invariant involved.
- Confirmations and corrections are captured with immutable provenance.
- Review reduces verified correction time or residual errors on representative
  scores.

## 23. Risks and mitigations

### Sparse secondary notation

Secondary slurs and deep beam levels are rare. Use factored heads, class-support
reporting, targeted sampling, and macro metrics. Do not claim success from micro
accuracy dominated by `NONE`.

### Synthetic-to-scan domain gap

Synthetic labels are exact but visually clean; scans are realistic but may have
edition-mismatched notation. Use strong visual augmentation, exact synthetic
supervision, per-head scanned masks, and reviewed scanned corrections.

### Catastrophic forgetting

Adding OSSQ only can specialize the model at the expense of other scores. Retain
general-data replay, separate pretrained/new parameter learning rates, staged
unfreezing, and fixed general regression tests.

### Context over-reliance

A model may learn that cello always means bass clef or that a supplied four-part
profile always means exactly four visible staves. Use unknown tokens, context
dropout, incorrect-context tests, soft layout scores, and explicit deviation output.

### Invalid beam/slur sequences

Independent per-token heads can produce locally likely but globally invalid spans.
Use sequence validators, group-level metrics, bounded review/repair, and optional
feedback embeddings only after measuring the output-only baseline.

### Token correction invalidation

Changing rhythm can change measure alignment; changing layout changes every token
identity. Content signatures, dependency-aware invalidation, and regeneration are
mandatory.

### Page-boundary ambiguity

Page-local models cannot see the missing half of a spanning slur or tie. Expose open
state and let assembly review it; do not invent endpoints.

### Capability drift

Provider, checkpoint, vocabularies, ONNX output order, parser, and generator can
drift independently. Named heads and immutable capability/version manifests are the
mitigation.

## 24. Recommended first implementation slice

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

### 24.1 Revised sequencing after B0 (2026-08-16)

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

### 24.2 Where the sequence actually stands (2026-08-16)

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
16  the scanned track        converted (27.34); training on all three corpora running
17  Gate C on the combined   the three-corpus script scores each domain but does not
    model                    dump predictions, so the crosstab and the arbiter sweep
                             have to be run separately afterwards
18  lieder sidecars          one call added; the corpus has not been downloaded
19  test_synth               held back deliberately - it is the one split that has not
                             been looked at, and it should stay that way until a
                             configuration is being reported rather than explored
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

## 25. Settled decisions and open measurements

### 25.1 Settled by this design

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

### 25.2 Must be measured before finalizing implementation constants

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

## 26. Implementation evidence and related designs

This document is self-contained, but the following local artifacts contain the
specific implementation evidence behind it:

### HOMR

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

### MuseScore

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

### OSSQ

- `../ossq-omr/scores/**`: original and cleaned score MusicXML plus constructed
  page/system artifacts.
- `../ossq-omr/excluded_segments_*`: published task-specific exclusions.
- `../string-quartet-omr-benchmark`: evaluation code and benchmark definitions.
- `../omr-data-preprocessor/omrdp/ossq`: construction and alignment pipeline.

### Human review and assembly

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

## 27. Reproduction record (2026-08-15/16)

Everything in this section was run and measured. It exists so the numbers quoted in
§24.1 can be re-derived rather than taken on trust, and so the next person does not
rediscover the environment problems.

### 27.1 What was built

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

### 27.2 Environment

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

### 27.3 Dataset construction

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

### 27.4 B0: the pinned checkpoint on OSSQ synthetic

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

### 27.5 Label support, and what it settles

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

### 27.6 Beam materialization is not needed here, and would be harmful

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

### 27.8 The stem head has no supervision from the segment labels

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

### 27.9 Notation has to survive the dataset files, not just the parser

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

### 27.10 homr's rhythm vocabulary stops at the 128th note

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

### 27.11 Training needs staff crops, which nobody has built

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

### 27.12 How much beaming is derivable without looking at the page

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

### 27.13 The scanned track, built

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

### 27.14 B0 on the scanned track, and the synthetic-to-scan gap

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

### 27.15 The frozen-core run, made runnable

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

### 27.25 Which corpora may carry notation labels

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

### 27.24 Ties are not slurs, and the labels could not tell them apart

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

### 27.44 The page is not only lyrics, and MuseScore types the rest for free

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

### 27.43 MuseScore earns the dependency: exact syllable boxes from the engraving it drew

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

### 27.42 The lyric stage is OCR plus a resolve, and the numbers say why

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

### 27.41 The voice comes back by arithmetic, and MuseScore is not needed

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

### 27.40 OLiMPiC's lyrics are recoverable, and it took looking to know it

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

### 27.39 A lyric track, and what OLiMPiC would have to be repaired to

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

### 27.38 The synthetic-to-scan gap is far worse for notation than for notes

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

### 27.37 Other corpora: what is worth preparing, and what each would cost

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

### 27.36 Both models on both splits: mixing was right, and quartets do not generalise

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

### 27.35 Mixing corpora: the predicted gain arrives, and so does a cost

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

### 27.34 The scanned track, converted

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

### 27.33 The scanned crop guard costs nothing either, and why I expected otherwise

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

### 27.32 The v2 retrain: nine heads, and the hooks move

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

### 27.31 PDMX, converted from source

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

### 27.30 Is OSSQ representative? For beams yes, for slurs not at all

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

### 27.29 The re-conversion, and two bugs that only a prediction would have caught

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

### 27.28 The stem head and the rule are complementary, not redundant

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

### 27.27 The stem head can be replaced by a rule over its own beam predictions

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

### 27.26 The converged run: three epochs was already most of the way

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

### 27.23 Stem direction is mostly a rule, and the head has not yet beaten it

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

### 27.22 Slur placement can be recovered, and the join is verifiable

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

### 27.21 Phase 2, the first frozen-core run, and Gate C

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

### 27.20 What the built labels actually contain

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

### 27.19 Building the training set: three things that stop a conversion dead

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

### 27.18 Whole-measure rests need no repair on the training side

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

### 27.17 Slurs that cross a system break

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

### 27.16 The Gate C baseline, per split

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

### 27.7 Known gaps

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
