# Joint training interpretation and next experiments — 2026-09-28

## Observations, not promotion evidence

At completed epoch 11, the current public-core-plus-fresh-heads run has mostly stabilized
on beams and stems; tie, advance and slur-span metrics still improve. Between epochs 6 and
11, PDMX tie macro F1 rises 82.77 -> 89.73 (+6.96 pp), PDMX slur span F1 76.97 -> 81.30
(+4.33 pp), and Lieder slur span F1 75.08 -> 83.10 (+8.02 pp). Dynamics stays near 10–11%
macro F1 from epoch 1; slur side metrics wobble rather than steadily improve.

Recognition training loss falls, but recognition validation loss rises on every source.
At epoch 11, rhythm/pitch/accidental teacher-forced accuracy deltas vs the initial public
core are OSSQ -0.038/-0.023/-0.046 pp, PDMX -0.146/-0.178/-0.160 pp, and Lieder
-0.206/-0.182/-0.158 pp. Thus more fitting is not evidence of better recognition.
These are internal teacher-forced diagnostics with core-seen score identities; they are
not untouched autoregressive release evaluation.

The epoch-12 refresh reinforces the tradeoff: PDMX rhythm/pitch deltas are
-0.182/-0.224 pp and Lieder -0.269/-0.198 pp. Lieder slur-span F1 fluctuates
83.10 -> 81.38 from epochs 11 to 12, so improvements are not monotonic. Rare beam
levels and second-slot slur events/sides need separate support-aware checks; aggregate
slur metrics can hide the rare slot.

Continue the already-running 35-epoch recipe as an unchanged baseline with all epoch
weights retained. Select checkpoints based on recognition/notation tradeoffs rather than
assuming the last epoch is best. The current >3-pp rhythm guard is a catastrophe guard,
not an acceptable promotion margin. No automatic shipping/pin changes.

## Sequence after PDMX-v2 lands

1. **Validate and pin the new data.** Finish crop and package checks; verify all 46 reviewed
   exclusions, percussion/length gates, protected-ID disjointness, image/token/sidecar
   alignment, notation target support, source and track distributions, and crop exposure.
   Keep v1 available. Transfer verified v2 shards to the training instance; publish/upload
   and production pin changes are separate actions, not implicitly performed by the build.
2. **Freeze an experimental holdout before training on new scores.** Preserve the existing
   protected validation IDs and reserve an additional reproducibly selected, score-level
   stratified sample from newly introduced score IDs. Balance by ensemble size/length and
   retain IDs in a tracked protocol. Its crops must be excluded from every fitting arm.
   Avoid choosing it after inspecting model results. Use selection/development data for
   checkpoint/weight decisions and retain a separate final test slice.
3. **Audit the current objective.** Measure per-branch/head raw and weighted shared-core
   gradient norms and pairwise cosine agreement across sources; add actual individual
   head-loss logging to the next runner. Investigate the dynamics confusion distribution,
   positive target counts/class imbalance, masking and attachment alignment. Existing
   aggregate loss logs cannot recover individual head losses.
4. **Run bounded weight ablations on the current fixed dataset.** Same public starting
   core, seeded fresh heads, crop order, augmentation, optimizer and update budgets. Compare
   current lambda=1 (already-running baseline) against notation multipliers 0.5 and 0.25;
   separately test an explicit pitch/rhythm/accidental-priority arm with normalized legacy
   group scale. Prespecify recognition noninferiority margins and head gains before looking
   at outcomes. An approximately 7,500-update screen is about six current-data epochs;
   preliminary measured rate suggests ~2 GPU-hours per new arm, roughly 4 hours for the two
   notation-weight arms and ~2 more for a priority arm. These are estimates, not launched jobs.
5. **Test PDMX-v2 as a separate data change.** Compare v1/v2 using the selected objective,
   equal optimizer updates and explicit multi-source sampling, not concatenation or equal
   epoch counts. Initially preserve source shares to isolate within-PDMX improvements;
   evaluate alternate source balance separately. Control source-score/ensemble exposure so
   long single-track scores cannot dominate through crop count. Protect OSSQ/Lieder scan
   performance and report new ensemble strata separately. Do not combine new loss weights
   and new data into one otherwise unidentifiable comparison.
6. **Run the next long joint candidate and full-stack evaluation.** Only after bounded
   trials support a recipe. Use resumable optimizer/RNG checkpoints, per-head losses,
   source-specific validation and tighter recognition selection gates. Compare autoregressive
   pitch/rhythm/accidental errors and notation policies against the public core with accepted
   real-only heads, earlier joint checkpoints and upstream HOMR on identical pages/staves.
   Regenerate examples/review and any public advantage claims from these final results.

The current uniform crop sampler is approximately OSSQ 78.2%, PDMX 14.2%, Lieder 7.7%.
Blindly appending the much larger PDMX-v2 pool would change source exposure and schedule
length; "35 epochs" would no longer mean a comparable compute budget. Choose source
probabilities and optimizer-update budget explicitly.

Source-independent corpus improvements do not depend on the current joint run passing.
The three-route lyrics/direction/dynamics plan is documented separately in
`TEXT_LYRICS_DYNAMICS_PLAN_2026-09-28.md`; dynamics remains review-only at inference.
