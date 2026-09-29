> **2026-09-29 UTC update:** Joint training stopped by user request after epoch 27. The earlier 35-epoch continuation plan below is superseded. Scoring is complete. Owner prioritizes OSSQ/Lieder and considers epoch 27 a promotion candidate; see JOINT_TRAINING_NEXT_STEPS_2026-09-28.md for suggested follow-up. No production promotion has occurred.

# Joint public core and fresh notation heads — 2026-09-28

Owner requested a long integrated run and chose **public core + new heads trained
jointly**, not fully random initialization. Name: `public-joint-core-heads-20260928`.
This is a warm-start joint experiment, not training from scratch.

Initialize recognition from public-rareNum-20260927, SHA-256
`89a97a2084743db22323969341c0bfb08c77f098c47ed3e40740df4f0b5709a0`.
Discard any old structured projections and initialize all twelve fresh at seed
20260928. Train encoder, decoder and every head from the first update; no freeze
callback. Heads: beams levels 1–4, stem direction, tie state, dynamic mark,
advance delta, and event/side for each of two slur slots. Voice/profile context
is not an existing notation head in this inventory.

Use the audited real-only, score-grouped internal train index (40,290 staves:
OSSQ 31,496, PDMX 5,701, Lieder 3,093), SHA-256
`f8c7f272b0fddd2397c5db169698a28e1ebaca82f6cd7a8a2baba5e7441202ca`.
Keep the 2,871/609/429 internal validation staves out of fitting. Recognition has
previously seen some of these scores, so this is not unseen release evidence.
Do not add synthetic data to this long run: preceding mixtures failed gates;
the separately authorized low-dose diagnostic remains independent.

Recipe frozen before full fitting:

- 35 epochs maximum; batch 8, accumulate 4 (effective batch 32), 4 loader workers.
- AdamW: core LR 3e-5, new heads 1e-4, betas .9/.999, epsilon 1e-8, decay .05.
- 10% update warmup, then cosine decay; clip gradient norm at 1.
- Existing recognition objective plus structured CE sum divided by the fixed 12-head
  inventory. Missing targets stay masked; no focal/class weighting or forced labels.
- BF16, TF32 disabled, deterministic algorithms and math SDPA; autocast cache disabled.
- Teacher forcing, no scheduled sampling. Real image augmentation seeded by epoch
  and row; no profile-context changes.
- Decoder CE is flattened to select deterministic 1-D NLL; mask, label smoothing
  and objective retained. CPU loss/gradient parity tested.
- Atomic model/AdamW/scheduler/RNG checkpoints every 200 optimizer updates and each
  epoch; every epoch's weights retained. Resume rejects changed recipe/source/index.
- Initial and per-epoch sourcewise teacher-forced recognition loss/accuracy and
  notation metrics, including dynamics. These do not replace autoregressive scoring.
- Stop if any source's rhythm-token accuracy loses >3 percentage points against
  initial recognition for two successive epochs. Save the state and reason.

Before launch: continuous vs interrupted GPU smoke must give identical weights,
nonzero encoder/decoder gradients and changed weights, and support/gradients/weight
changes for all twelve heads. Check augmented training and the evaluation path too.
The initial GPU check exposed tiny differences with non-deterministic kernels;
strict determinism also revealed legacy 3-D NLL has no deterministic CUDA kernel.
Those attempts are preserved; neither was the long run.

Run and watcher must live in instance tmux. Capture code snapshot, source identities,
environment versions and logs. Local collector copies completed reports/weights,
checks SHA-256 and updates docs with local paths. No automatic promotion, export,
deployment or publication. Final evaluation must compare recognition and notation
against the existing accepted stack per corpus, including rare directional recalls
and invented spans, before any release claim.
