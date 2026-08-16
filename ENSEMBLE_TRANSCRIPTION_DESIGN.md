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
code.

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

**The step**: run the synthetic partwise cropping (`yolo_detect_staves`,
`yolo_crop_staves`), then `convert_ossq.py`.

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
