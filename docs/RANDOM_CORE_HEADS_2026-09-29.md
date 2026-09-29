# Fresh music-recognition training — 2026-09-29

Owner authorized training the recognition model and all notation heads from scratch, then explicitly allowed ImageNet pretraining. The selected initialization is **ImageNet ConvNeXt encoder + random music decoder and notation heads**. No HOMR/core checkpoint is loaded. This is distinct from the stopped public-core-initialized joint run; it does not replace its epoch-27 candidate or accepted runtime pins.

Reproducible runner: `training/transformer/train_joint_random.py`, code commit `d55f5d9`. Encoder pretraining is explicitly controlled by `--encoder-pretrained`; the default encoder behavior for existing runners is retained. Existing instance environments/drivers are preserved.

## Fixed recipe

- Pinned current corpora and 40,290-row training index, SHA-256 `f8c7f272b0fddd2397c5db169698a28e1ebaca82f6cd7a8a2baba5e7441202ca`. Expanded PDMX-v2 is still rendering; it is not added mid-run.
- Source sampling: OSSQ 60%, Lieder 25%, PDMX 15%, followed by uniform crop selection within source, with replacement. This is an explicit operational choice reflecting owner priorities, not an optimized or previously agreed ratio. Long scores can still have more exposure within a source; this is not score-balanced sampling.
- 160 epochs of 40,290 draws; batch 32, accumulation 1; 201,600 optimizer updates; seed 20260929. Epoch count describes draws, not guaranteed coverage of every crop.
- AdamW, peak encoder/decoder/head LR 3e-4, betas 0.9/0.999, epsilon 1e-8, weight decay 0.05, 5,000-update linear warmup, cosine decay, global gradient clipping 1.0. This schedule is a starting recipe for fresh initialization, not an empirically selected optimum.
- Six existing recognition branch losses/consistency retained; notation head-loss sum /12 retained to avoid introducing an untested weighting change. All 12 notation heads train, including experimental dynamics; dynamics remains prohibited as automatic production output.
- BF16 autocast, float32 weights, TF32 off, deterministic math SDPA; existing image augmentation. Per-branch recognition losses, individual notation losses/support, LR, aggregate losses and ETA logged every ten updates.
- Atomic optimizer/scheduler/RNG resume every 200 updates and each epoch; all epoch weights retained. Validation metrics separately for OSSQ, Lieder, PDMX every epoch. The inherited >3-pp rhythm guard is only a catastrophic deterioration check against initialization, not a promotion criterion for a randomly initialized model.

## Checks and follow-up

Before launch, perform finite-loss/gradient checks and compare continuous five-step training with three steps plus resume to five; require identical weights. Check source-specific evaluation runs on fresh weights. Persist training and watcher in separate tmux sessions. Initial duration is several days; refine after actual steady throughput, including epoch validation overhead. Completion time does not guarantee convergence.

Owner prioritizes OSSQ/Lieder; PDMX is diagnostic. Suggested epoch-27 promotion steps remain: untouched real-score evaluation against the strongest accepted stack, predeclared Lieder tolerance, divergent-case review, local candidate packaging and rollback. This longer experiment does not perform those steps or establish production readiness.

Local collected artifacts will live at `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/train-20260929/random-core-heads-20260929/`. Do not cite uncollected instance-only reports as durable evidence. No production pin, deploy, upload or published claims changed.
