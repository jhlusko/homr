# Exact timing prediction targets and checkpoint migration — 2026-10-04

Four output-only timing projections and a separate checkpoint contract are implemented as an experiment. Current model forward, recognition targets, notation heads, ONNX graphs, runtime and deployed pins are unchanged. No training has run.

## Proposed targets

| Head | Meaning | Supervision |
|---|---|---|
| timing.elapsed | A note/rest's elapsed quarter-note duration | Timed pitched notes/rests with supported, consistent targets |
| timing.written | Written note type and dot count | Timed non-full-measure-rest events with a stated type |
| timing.ratio | Cumulative actual:normal sounding ratio | Consistent ordinary/tuplet events, including bounded tick quantization |
| timing.full_rest | Full-measure-rest identity | Rests only; pitched notes are masked |

The elapsed head describes printed members' sounding durations. Existing advance.delta describes the gap to the next simultaneity; those values can differ in polyphony. The ratio head describes cumulative time scaling, not the printed number that rareNum recognizes. A nested pair of 3:2 tuplets can have cumulative ratio 9:4 while printing two 3s. The four heads alone do not predict nested bracket boundaries or their complete engraving.

BOS/EOS, padding, control/grace symbols, missing metadata and unseen classes remain masked. Targets are laid out at full token positions and shifted once with the existing decoder alignment helper. No rare value is relabelled as a common duration or fabricated OTHER class. Per-head diagnostics retain missing/unseen/source-conflict support.

## Canonical inventory, with no validation class discovery

All 297 indexed canonical OSSQ/Lieder scores were read with reference hashes and recovered score-disjoint train/valid membership. Class lists come from training-score events only, not validation. These are canonical whole-score events, not actual crop training frequencies; some canonical windows cannot enter the existing vocabulary.

| Source/split | Scores | Timed visible pitched/rest events | Tick-quantized events | Unresolved duration conflicts |
|---|---:|---:|---:|---:|
| OSSQ train | 76 | 928,090 | 414 | 0 |
| OSSQ valid | 9 | 97,914 | 32 | 0 |
| Lieder train | 199 | 256,383 | 0 | 29 |
| Lieder valid | 13 | 18,269 | 0 | 0 |

Invisible, grace and unpitched events are excluded explicitly. Strict raw-tick inventory yields 52 elapsed, 21 written, 18 ratio and 2 full-rest classes. The proposed bounded quantization policy yields 51 elapsed, 21 written, 22 ratio and 2 full-rest classes.

### Why the bounded policy is preferable

Some apparently strange classes are source tick rounding of valid nested tuplets. In OSSQ sq8482283 part 2 physical measure 489 / printed measure 78, a 9:4 sixteenth has intended duration 1/9 quarter, but the source uses 53 ticks at divisions=480 (53/480 quarter). Separate nested 3:2 bracket elements corroborate the cumulative ratio. A similar case occurs in validation sq7354505 part 1 physical measure 1025 / printed measure 202.

The candidate policy uses exact written type/dots and the explicit cumulative ratio only when the intended tick count is non-integral and the stored integer ticks equal its floor or ceiling. It retains original source metadata and changes experimental targets only. Larger differences and missing/inconsistent ratios stay unresolved; elapsed and ratio supervision are masked in those cases while written/rest-identity labels can remain available. This classifies all 446 OSSQ conflicts as bounded quantization rather than teaching fractional tick artefacts. It is a source-arithmetic inference, not a new recognition result or a blanket MusicXML rewrite.

Lieder's 29 unresolved events are IMSLP622105 (3), IMSLP624193 (21), IMSLP632172 (5), all training scores. Located examples include eighth rests with elapsed 3/4 quarter and no time-modification in IMSLP624193 part 1 physical measure 9 / printed measure 8. These need targeted source/image investigation; do not infer a missing dot or duplet automatically.

### Held-out coverage remains explicit

The mixed-source training inventory covers all Lieder validation values. Its four 5:3 events gain support from OSSQ training; Lieder-only training would not cover that ratio/duration combination. OSSQ validation still has 15 unseen elapsed values (15/16 once, 2/11 eleven times, 79/16 three times), one unseen eighth-with-three-dots target, and 23 unseen cumulative ratio targets (9:2 twelve times, 11:8 eleven times). These stay masked/reported; validation was not used to add classes. Full-rest flag values are predefined yes/no semantics.

The existing v2 timing sidecars omit source divisions. The bounded policy cannot safely reconstruct an input tick grid from a reduced duration fraction. Its sequence target builder therefore refuses tick-quantized supervision without source-divisions metadata. [The subsequent v3 contract](EXACT_TIMING_SOURCE_GRID_2026-10-04.md) adds that provenance and qualifies serialization and normalized targets. Existing v2 sidecars still require rebuilding for this policy; they remain readable without invented source grids.

## Additive checkpoint preparation

`ExactTimingHeads` is an independent projection module over decoder hidden states, not an alteration of existing branch embeddings or projection shapes. A separate bundle binds the exact base checkpoint SHA-256, target contract (including class order/policy/provenance), hidden dimension and head state. Loading is strict; wrong base or reordered classes fail, and untrained bundles are refused for inference.

An initialized addon was prepared against the accepted September 18 core checksum 1b2855ce3facea9aa8b0017f6e30143abd9aa43569a2f9ce1ab46bf8bf818fa3. Hidden dimension 512 is read from the accepted v7 notation ONNX graph. It contains 49,248 parameters and outputs widths 51/21/22/2. The base file's checksum is identical before/after. Saved weights reload with weights_only=True and pass a zero-feature shape smoke; inference loading correctly refuses them. This is not an accepted-model decoding comparison or trained checkpoint promotion.

Twenty focused tests pass: exact duplet targets, rests-only flag masking, source conflict masking, bounded quantization, unseen classes, decoder alignment, core-weight/logit/RNG preservation in a small structural fixture, strict migration guards and head-only gradients. Existing codec/sidecar tests are included. Core-logit identity is a unit structural check; no real model feature or decoding gate ran here.

Bundles contain no optimizer state. An old joint optimizer is not implicitly migrated. A future auxiliary-head warm-up starts its own resumable optimizer; subsequent joint training must explicitly define/save parameter groups, optimizer/scheduler state, RNG, source sampling and target-contract hashes.

## Loss and experiment sequence

The loss API requires four explicit finite nonnegative weights. It does not append these heads to the old recognition + notation/12 objective or silently divide by a new head count. Tests use unit weights to verify mechanics; these are not a selected training recipe.

1. Source divisions and selected canonical-window serialization are qualified by [v3 follow-up](EXACT_TIMING_SOURCE_GRID_2026-10-04.md). Remaining: derive actual training-crop support against the frozen train-only manifest, investigate the 29 unresolved Lieder events, and track unsupported canonical crop windows separately. Before fitting, freeze two scoring views: original source ticks and notation-derived durations for the bounded quantization cases. Apply the same policy to accepted and candidate outputs; no blanket onset tolerance or post-hoc gate change.
2. Wire opt-in targets/head forward and resumable addon training. Begin with the accepted core frozen and only new heads trainable; do tiny-fit checks followed by real OSSQ/Lieder evaluation. Report rare duplets, nested ratios, long/full-measure rests and per-class support, not just majority accuracy.
3. Before a joint experiment, measure gradient norms and direction conflicts for pitch, rhythm, accidentals and each timing group at shared decoder/backbone layers; compare a small explicit weight set. Recognition remains the primary gate. Retain the agreed Lieder +0.25 percentage-point symbol-error ceiling and report pitch/rhythm/accidental changes and OSSQ results separately.
4. Design prediction-only XML reconciliation with consistency checks and explicit nested bracket treatment; qualify decoding on OSSQ/Lieder before activation, publication or website advantage claims. Reference sidecars must never supply production predictions.

Steps 1 and target/trainer wiring can be developed locally. GPU is needed for meaningful feature fitting, gradient probes and model qualification.

## Local evidence

Parent: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `timing-target-inventory-20261004/`: strict raw-tick train-only inventory, source-disagreement examples and exact original source snapshots; 74.5 seconds, EXIT=0.
- `timing-target-inventory-quantized-20261004/`: bounded policy, train-only target manifest, separate source/split support and unseen counts; 142.9 seconds, EXIT=0.
- `timing-addon-init-20261004/`: explicitly untrained heads, base/graph/contract/bundle checksums, shape/rejection report and tested source snapshots.

Both inventories use one CPU thread in persistent tmux with completion watchers. All jobs complete; remaining ETA zero. No public corpus/model/provider pins or deployment changed. Predicted advance remains disabled.

Tools: `training/architecture/transformer/exact_timing_targets.py`, `exact_timing_heads.py`; `training/onnx/inventory_exact_timing_targets.py`, `prepare_exact_timing_head_bundle.py`. Previous dataset work: `EXACT_TIMING_DATASET_INTEGRATION_2026-10-04.md`.
