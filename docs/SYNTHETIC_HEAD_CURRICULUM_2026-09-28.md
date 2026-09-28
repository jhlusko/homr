# Synthetic curriculum for structured heads — proposed 2026-09-28

**Status: generator committed as `ed3f368`; MuseScore pilot complete, not training-ready.**
The 256-family, two-font run produced 2,048 images with zero render failures. SHA-256
verification passed after copying the result locally. The duplicate-image audit found
160 identical-image pairs among beam variants across 48 motif families, despite
differing intended beam labels. This is a renderer/contrast defect to fix before any
candidate is promoted. The output stays quarantined (`training_ready=false`); no training
has started, and the bounded LR/epoch comparison remains unchanged. This proposal targets
heads over the frozen `public-rareNum-20260927` core, not another full-core adaptation.

Local run artifact: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/synthetic-head-pilot-musescore-20260928/`.

## Question and motivation

Can controlled visual coverage improve notation heads on real scores without causing
the source tradeoffs seen in earlier refits? The saved-prediction audit finds:

- Tie starts/stops are missed more often on every corpus when unchanged v7 heads move
  to the new core. PDMX tie errors and stem errors are broadly distributed across scores.
- Slur-span loss includes invented spans: PDMX false positives rise 616→874, while
  core-swap fixes/breaks of previously correct spans are 119/96. Endpoint recall alone
  is insufficient.
- Beam-3 forward/backward hooks are confused; source mixing trades OSSQ against PDMX.
  The capped square-root class weighting trial did not meet its hypothesis.

These motivate exact contrasts, hard negatives and realistic context, rather than a
large undirected synthetic corpus. Earlier synthetic continuation harmed core scan
recognition; freezing the core protects its weights, but does not guarantee the heads
will transfer to scans. Measure transfer directly.

The current heads are linear projections of decoder hidden states. More synthetic
examples cannot recover information those states fail to expose to a linear readout.
Test that capacity before paying for a full curriculum. A failed probe is a reason to
investigate representation/head design separately, not automatically unfreeze the core.

## Data construction and renderer qualification

Generate original short MusicXML motifs, then derive both tokens and v6 sidecars from
that **same authoritative XML** using the current parser and note-position mapping.
Parse before any rendering cleanup; operate on copies, because previous in-place
stripping destroyed stem labels. Store source XML, canonical family ID, generator seed,
renderer/font/options, crop geometry, token/sidecar/image hashes and rejection reasons.
Never infer synthetic ground truth from the model's output.

Use existing infrastructure:

- `training/omr_datasets/structured_notation_parser.py`: explicit beams, stems, notated
  ties, canonical slur slots and sides, with findings for ambiguous/invalid annotations.
- `training/omr_datasets/convert_lieder.py`: MuseScore 4.6.5 AppImage and music-font
  style-file pattern.
- `training/omr_datasets/convert_musetrainer.py`: font/scale choices and SVG rasterization.
  Reuse rendering pieces, not its token-only generation as authoritative head labels.
- `training/omr_datasets/render_stage2_tokens.py`: MuseScore batch rendering pattern.
  Token-only XML round trips need not preserve every structured field; qualify an
  original-XML render path instead.
- Existing structured dataset/target builders: keep BOS/symbol/EOS positions and masks
  consistent; no new positional schema.

Reproduce the quarantined pilot from the HOMR repository root with:

```sh
python -m training.omr_datasets.synthetic_head_curriculum \
  --out /path/to/new-output-dir \
  --musescore-bin /path/to/MuseScore-4.6.5.AppImage \
  --source-revision <committed-homr-revision>
```

The output directory must be new or empty. Candidate indexes remain unusable for
training until the review identifies no intent/render disagreements and a person
explicitly promotes a verified subset.

For each renderer, qualify explicit stem direction, beam hooks at all four trained
levels, tie versus slur, slur side, two simultaneous slur slots and cross-bar curves.
Check visual evidence using SVG objects/geometry where reliable and rendered review
examples. MusicXML round-trip agreement alone is not visual verification. Identical
pixels paired with different requested visual labels are a rejection signal. Record
renderer-inferred notation changes; derive effective labels from independently verified
rendered/exported notation or reject the case, never silently retain intent as truth.

Renderers may normalize unusual engraving. Start with one qualified renderer; use a
second only for categories it demonstrably preserves. Review coverage examples as
original XML intent / image / effective labels, labelled by measure and beat, with
implementation positions available only in details. No per-example preference voting
is needed: this review establishes label correctness.

## Targeted coverage

| Head | Contrasts and progression | Important negatives / constraints |
| --- | --- | --- |
| Stem direction | Same pitches/rhythm, forced up versus down; then chords, ledger lines, two voices | Rests/whole notes have no applicable stem; explicit absence differs from unknown |
| Ties | None/start/stop/start-and-stop; two notes, chains, bar crossings, chord members, voice contexts | Repeated pitches without a tie; same-pitch phrase slur; tie and slur coexisting; valid same-pitch/voice ties only |
| Slur event/span | Short and long spans, shared endpoints, nested/overlapping spans, two canonical slots | Untied/unslurred repetitions, ties alone, adjacent curves, unequal span lengths; measure complete endpoint pairs and inventions |
| Slur side | Above/below at both directions of stem and pitch contour | Side is supervised only when visibly specified; arbitrary XML slur numbers are not class labels |
| Beam levels 1–4 | Flags, begin/continue/end, forward/backward hooks; isolated groups then chords/tuplets/rest-spanning groups | Matched notes with opposite hook direction; valid duration/beam-depth combinations; rests remain NOT_APPLICABLE even within a rest-spanning group |
| Advance delta | Rhythmic/voice variety as an auxiliary task; later simultaneities and tuplets | Preserve canonical simultaneity-member masking; do not duplicate onset targets onto every chord member |

Dynamics stays excluded from this curriculum's trainable/promotion allowlist, following
the existing separate-detector policy. The other eleven configured heads are in scope.
Only genuinely supervised positions contribute loss. Explicit NONE is a real negative;
UNKNOWN/NOT_APPLICABLE remain masked according to each existing head's semantics.
Require actual optimizer updates for every declared active head, and manifests matching
the trainable allowlist. A tiny probe declares only the heads it actually fits.

Construct matched visual counterfactuals while preserving the base token sequence
where the vocabulary permits it (particularly stems/beams). For ties/slurs, record any
legacy-token difference rather than claiming identical conditioning. Include image-shuffle
or other suitable image-dependence controls to expose token-only shortcuts.

## Phase 0: small capacity and label pilot

Generate roughly **256 distinct motif families / 1,000–2,000 qualified renders** covering
the target contrasts. Balance by annotated events and independent musical contexts,
not just crop count. Many font/blur copies of one motif do not count as more families.
Keep all mutations/renders/degradations of a family in the same split; reserve distinct
families for generalization checks, and renderer/font holdouts where available.

On clean, unambiguous, deliberately balanced examples, test whether the frozen core plus
linear heads can fit the target distinctions. Provisional capacity bars, to freeze
before the pilot: >=95% training macro-F1 on supported balanced target classes and
>=90% complete-slur-span F1 on the simple-span subset. Report per-class support and
image-dependence controls; clean training memorization is not a real-data improvement.
If a category fails, first check visible labels/alignment, then representation/readout
capacity. Do not scale a failed category blindly. Head hidden states may be cached for
this fixed-image probe; changing pixels/augmentation requires recomputation.

## Proposed curriculum and bounded initial budget

Once label/capacity checks pass, start with this **provisional** schedule. Freeze counts
and ratios after a generator/rendering throughput pilot and before comparing arms.
Use draw quotas and per-head/event support, not an epoch tied to an arbitrary pool size.

| Stage | Synthetic examples | Synthetic / real draws | Updates, batch 8 |
| --- | --- | --- | ---: |
| 1. Clean contrasts | One concept, short staff, clear spacing, balanced target events | 75 / 25% | 1,000 |
| 2. Musical context | Chords, multiple voices, longer chains/spans, tuplets, varied clefs/metres | 50 / 50% | 2,000 |
| 3. Appearance transfer | Qualified font/scale/spacing changes, mild blur/ink variation/skew/compression/paper | 25 / 75% | 2,000 |
| 4. Real-data consolidation | Low synthetic replay retaining rare contrasts; mostly existing real corpus pool | 10 / 90% | 3,000 |

Total initial budget: **8,000 updates / 64,000 draws per arm**, including 20,400 synthetic
and 43,600 real draws for the curriculum. Sample real sources at a single prespecified
mix throughout; begin with the current internal training pool's source proportions,
and report each source independently. Do not simultaneously optimize source weighting.
Ensure adequate independent families for each rare class before selecting a larger
render count; forbid one or two families from dominating a rare class's exposure.

Degradations must preserve visibility and target geometry. No horizontal flip, destructive
curve erasure, or clipped endpoint relabelled as an ordinary absence. Cross-boundary
starts/stops stay annotated under the current parser semantics; closed-span metrics are
interpreted accordingly. Keep clean controls and severity records. Fit severity ranges
to **training** scans, not validation/test pages; retain existing deterministic RNG fix.

## Experiments that separate the hypotheses

Use one frozen public core, identical fresh head initialization, optimizer setting,
batch size, total updates and preprocessing implementation. The running LR/epoch trial
can supply an internal-selected setting; if none qualifies, use the declared .001
baseline and label the new experiment diagnostic. Do not import existing heads that
have seen the internal holdout.

1. **Real-only control:** equal updates, existing internal training pool.
2. **Unordered synthetic + real control:** exactly the curriculum's complete draw
   multiset, including image/degradation seeds and supervision masks, globally shuffled.
3. **Curriculum:** same multiset as arm 2, ordered by the stages above.

This separates benefit from adding synthetic examples (2 versus 1) from benefit of
ordering them (3 versus 2). Do not give the curriculum more updates or a different
aggregate distribution. Checkpoint at stage boundaries, saving optimizer/RNG state;
first verify resume and gradient/identity guards with a small smoke. Keep the core
bit-identical and the existing recognition outputs unchanged on fixed inputs.

Select on real **score-grouped internal validation**, retaining current split/provenance
and acknowledging it has already been inspected. Synthetic family/font/renderer
holdouts measure construction and transfer, not the release claim. Report source-wise
head metrics, tie starts/stops, hook directions, stem up/down, complete slur precision/
recall/F1, inventions, and exact beam vectors. Include ordinary/rest-spanning beam groups
separately when references permit; synthetic gains do not settle the existing runtime
beam arbitration decision.

Prespecify selection: no supported source/head loses >1 pp versus real-only, at least
one improves >=1 pp, and no concealed directional/false-positive regression in the
targeted diagnostics. A claim that ordering itself helps also requires improvement over
the matched unordered arm; otherwise prefer the simpler mixture. Freeze exact metric
support thresholds and targeted guardrails before training, not after seeing results.
One selected candidate then gets full-training confirmation against corrected old-core/v7
release gates. Published validation is read once for that confirmation, not every stage.
Repeat a promising result at another seed before claiming a robust curriculum advantage.

## Implementation order and estimates

1. Implement an original-XML generator, renderer qualification, provenance/family split,
   visual audit and label integrity checks. **MuseScore generator implemented; renderer
   fidelity and visible-label qualification remain open.**
2. Generate the pilot and measure rendering/feature extraction throughput. **Roughly
   1–3 CPU-hours plus 30–60 GPU-minutes**, to replace with measured estimates.
3. Freeze full draw schedule and the three-arm protocol. Initial comparison **roughly
   2–4 GPU-hours**, depending on crop length, loader throughput and evaluation time.
   More complete fitting and a seed repeat add separate time; do not imply a full
   release can be proved by the pilot.
4. If real-source gates pass, freeze full training/confirmation and final examples;
   otherwise retain the current candidate and report the failed hypothesis.

Generation can run on CPU independently after implementation is approved. Do not
interrupt or contaminate the current LR/epoch experiment. Preserve its environment;
use installed tools or qualify alternatives without rebuilding its venv. Actual renderer
availability and throughput must be checked before promising a full corpus ETA.

## Corrected renderer qualification (v4, 2026-09-28)

The v3 pilot had 160 identical-image contrast pairs in 48 beam families. MuseScore
ignored hook-direction overrides on equal-duration notes. V4 uses a short note before
or after a longer note within one beat to produce isolated level 2–4 hooks. These are
rhythm-changing contrasts, not identical-token examples. Level 1 uses a four-note group
versus two pairs; level-1 hooks remain an explicit coverage gap.

Every render now includes a MuseScore-exported MusicXML audit of note count, stems,
ties, beam levels and slur events/explicit sides. Unspecified exported slur sides are
reported and still require visual checking. Mismatches and identical-image contrasts
are excluded from candidate indexes. All outputs remain `training_ready=false`.
The gallery gives side-by-side examples with musical measure/beat descriptions.

Local bounded qualification (16 families, four variants, Leland and Bravura):
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/synthetic-head-v4-qualification-20260928/`.
Four generator/audit regression tests pass. A corrected 256-family pilot follows only
if this bounded run has no renderer-label differences or duplicate contrasts.
Frozen-core capacity testing follows visible-label qualification; no training is launched
on unqualified samples.

Bounded run completed: **128/128 renders, zero failures, zero renderer-label
mismatches, zero duplicate contrasts**, in 87.77 seconds locally with two workers.
Selected examples were visually inspected for stem direction, tie/slur spans and
beam groups; this is not an exhaustive approval of every sample. The corrected full
pilot is CPU generation only, in persistent tmux with a completion watcher. Its
local destination is
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/synthetic-head-pilot-musescore-v4-20260928/`.

## Owner qualification and capacity protocol (2026-09-28)

The owner reviewed the 128-image v4 gallery (all four motif kinds, beam depths 1–4,
both fonts) and approved the visible labels: “they all look good. proceed”. Approval
covers these generator/renderer templates for **synthetic capacity diagnostics only**.
Full-pilot examples must still pass every source-artifact checksum, renderer-label audit,
no-duplicate contrast check and family-split check. Original generator artifacts remain
unchanged/quarantined; a separate qualification manifest records the approved scope.

`training/transformer/synthetic_head_capacity.py` extracts the frozen
`public-rareNum-20260927` decoder states using clean fixed images (normal canvas and
normalization, no augmentation), authoritative token branches and existing shifted
structured targets. It fits fresh linear heads, excludes dynamics, preserves family
splits, and never adapts the core. Core tensor hashes and fixed-input recognition logits
must remain exactly unchanged. Feature caches carry data/core/source identities.

Capacity protocol is fixed before fitting: Adam .001 (the completed LR/epoch trial
selected no candidate), seed 20260928, 2,000 updates, 128 balanced positions per active
head per update, no focal loss or class weights. This per-class sampling is for capacity
only, not the later equal-budget three-arm real-transfer comparison. Report every class,
including absent classes, and train/valid/test macro-F1. Training capacity bar: .95
macro-F1 on heads with at least two represented classes, at least 20 positions and four
training families for every represented class; simple-slur complete-span F1 .90 with
at least 20 reference spans. Insufficient support is a gap, never a pass. No result
selects a real-data model or triggers full curriculum training automatically.

Image dependence: swap images within the same family/font and **identical token bytes**;
keep the original tokens and targets. Recompute decoder states, and report normal versus
swapped performance only at positions whose target labels differ between these token
twins. This isolates visual dependence for covered stem/grouping/slur-side contrasts.
Ties/hooks with different teacher-forcing tokens remain potentially token-confounded;
passing synthetic fit alone cannot establish their visual dependence or scan transfer.

Save resumable Adam/head/sampling-RNG state every 250 updates, with exact identity checks.
A 128-image smoke checks extraction, actual head gradients, report output and frozen-core
identity before the 2,048-image probe. The smoke has too few independent families to
meet the full capacity bars. Nine targeted generator/probe tests pass, including exact
uninterrupted-versus-resumed linear updates. Provisional full extraction/fitting ETA:
30–60 GPU-minutes, to replace with measured throughput.
