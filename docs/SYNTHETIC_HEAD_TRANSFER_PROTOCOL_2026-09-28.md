# Fixed-pixel synthetic pilot transfer — protocol frozen 2026-09-28

This tests whether the qualified clean pilot transfers to real scores, and whether
ordering the same draws helps. It is a bounded precursor to the richer curriculum,
not a test of unimplemented multiple voices, tuplets, clefs/metres or degradation.
Prerequisite: the v4+v5 capacity report passes all eleven supported-head bars.
Flags/level-1 hooks remain outside the demonstrated synthetic coverage.

## Frozen data and controls

Use `public-rareNum-20260927` core (SHA-256
`89a97a2084743db22323969341c0bfb08c77f098c47ed3e40740df4f0b5709a0`),
fresh identical head initialization at seed 20260928, and eleven trained heads with
**dynamics excluded**. Freeze/eval the whole core, disable TF32 and cuDNN benchmarking,
use deterministic cuDNN. Adam .001, betas .9/.999, eps 1e-8, zero weight decay, no
AMP, scheduler, weighting or focal loss. Class-balanced capacity fitting does not carry
into this real-transfer study.

Use the existing score-grouped internal training index: 40,290 real rows
(31,496 OSSQ / 5,701 PDMX / 3,093 Lieder), index SHA-256
`f8c7f272b0fddd2397c5db169698a28e1ebaca82f6cd7a8a2baba5e7441202ca`.
Real internal validation is unchanged: 2,871 OSSQ / 609 PDMX / 429 Lieder staves;
no published validation. The core has seen these real scores, and this internal set
has been repeatedly inspected. This is head fitting/transfer, not untouched release
confirmation. Select synthetic TRAIN families only; supplements retain v4 family seeds
and splits. Do not move synthetic held-out families to fill missing targets.

Every image is its original crop, using normal canvas resizing and normalization,
**without stochastic augmentation**, in every arm and evaluation. Fixed pixels isolate
data/ordering and avoid testing new synthetic degradations before qualification. Real
augmentation and the richer proposed curriculum are separate follow-ups. A fall in real
held-out metrics remains a failed hypothesis regardless of synthetic fitting results.

## Matched 8,000-update schedules (batch 8)

| Stage | Qualified pilot pool | Updates | Synthetic draws | Real draws |
|---|---|---:|---:|---:|
| 1 | Simple notes/ties/slurs/beam groups | 1,000 | 6,000 | 2,000 |
| 2 | Existing chords, chains, cross-bar/overlapping curves, hooks/pairs | 2,000 | 8,000 | 8,000 |
| 3 | All existing concepts in qualified Bravura appearance | 2,000 | 4,000 | 12,000 |
| 4 | Complete qualified replay, mostly real | 3,000 | 2,400 | 21,600 |
| Total | | 8,000 | 20,400 | 43,600 |

Sample independent synthetic families in shuffled cycles, then one eligible render per
family. No rare motif gets most exposure by having more font/variant copies. Real
source ratios remain the internal training pool proportions, with exact per-stage
largest-remainder quotas and within-source shuffled cycles. No source-mix search.

Three arms: real-only (64,000 real draws, equal update budget); unordered mixture;
ordered curriculum. The latter two use the **identical complete draw multiset**,
including sample identity, stage-qualified font/context and pixel seed/mask policy;
only draw order changes. Archive the literal plan and all source/data hashes before
training. Use fresh heads, not the capacity-fitted heads or v7 weights. Train all eleven
heads jointly; do not cherry-pick different heads from different arms.

## Checks and scoring

Require actual gradients/support and changed weights for all declared trained heads,
22 active Adam states, finite gradients/weights, unchanged dynamics, bit-identical core
tensors and fixed-input recognition logits. Save Adam/head/Torch/CUDA RNG state every
200 updates and weights at 1,000/3,000/5,000/8,000 updates. Resume rejects changed
model/data/source/optimizer identities. GPU smoke compares uninterrupted 16 updates
versus 8 plus resume to 16, requiring bit-identical head tensors. Check pixels across
worker counts before full training. Measure throughput on the real-only control before
refining the total ETA.

Evaluate **final 8,000-update checkpoints only** for selection, source by source.
Keep exact prediction IDs, normalized pixel hashes and reference hashes matched across
arms; retain per-staff head reference/prediction vectors. Report beam macro-F1 1–4,
exact beam vectors, hooks F1, stem up/down accuracy and macro-F1, tie macro-F1,
slur complete-span precision/recall/F1 and inventions, side macro-F1 and nontrivial advance.
Mask UNKNOWN/NOT_APPLICABLE according to existing builders; NONE is a real negative.
Rest-spanning/ordinary-beam policy still needs its separate paired diagnostic and is not
settled by this pilot's beam score. Supplemental slot-2 sides are explicit, not inferred.

Freeze support >=100. A mixture candidate must have no supported source/metric loss
>1 percentage point versus real-only and at least one gain >=1 point. Directional
stem, tie-event and side class recalls and each supported hook direction at each level
also may not fall >1 point. Supported slur inventions may increase by no more than
0.01 false spans per reference span (one per 100). Low-support cells stay descriptive.
A claim of an ordering advantage requires the same gates versus unordered; otherwise
prefer an eligible unordered mixture. Neither eligible means stop, report the failed
hypothesis and preserve the baseline. No automatic search expansion or release.

One selected candidate requires another seed and separate full-published-training/
published-validation confirmation against corrected old-core/v7 release gates. This
pilot alone never changes Modal/provider pins, publishes models or updates public claims.

## Implementation and estimated runtime

`training/transformer/synthetic_head_transfer.py` creates the literal plan, trains,
saves resumable checkpoints, scores and applies the frozen comparisons. Schedule tests
verify equal multiset, budgets, family isolation, padding/masks and guardrail vetoes.
Start in persistent instance tmux with completion/failure watcher and local collection.
Initial estimate: roughly 1–2 GPU-hours for three arms plus evaluation; replace this
with measured smoke/control throughput. Existing training environment is retained.
