# Dynamics head: why it doesn't work yet, and what would actually fix it

Investigated 2026-08-28 after a joint frozen-core structured-heads probe (advance, beam,
stem, tie, slur, dynamics all trained together) scored dynamics at **11.97% macro F1** —
far below every other head in the same run (beam 86-98%, stems 64%, ties 73%, slur spans
73%, slur sides 81%, advance 82-83%). This note exists so the next person who reaches for
class reweighting on this head first reads why that alone won't move it, and so the
decision to unfreeze the core (or add dynamics to the token vocabulary) has the evidence
already assembled.

**Correction, added after this doc was first written: most of this was already known.**
`RUNLOG.md` phases 27.94-27.98 (multiple prior sessions) already built and measured this
exact head across four training runs - plain cross-entropy (macro F1 0.068), focal loss
(0.063, already ruling out class-imbalance-as-hard/easy), vocabulary collapse (0.120, with
RUNLOG's own contemporaneous caveat that the gain is "mostly from removing zero-support
classes out of the average rather than a clean win on every kept mark" - the identical
illusory-improvement finding this doc reports below, independently rederived rather than
novel), and a diagnostic pass that already concluded "it is not confusion between marks,
it is the head mostly failing to register that a mark is present at all" - the same
conclusion this doc's "root cause" section reaches from a different angle. This doc's
genuinely new contributions are the scoring-floor artifact (below) and the tie-inflation
cross-check; treat everything else here as convergent confirmation of RUNLOG's history,
not a fresh discovery, and read RUNLOG 27.94-27.98 for the fuller record (including
`dynamics_placement.py`'s OSSQ-segment data-loss bug, unrelated to anything below).

**Also established (RUNLOG 27.45, 27.94-27.97) and important for what to do next:
dynamics already has a working, SEPARATE Stage 3 pipeline** - a page-level bounding-box
detector (84.0% whole-page F1) feeding a dedicated small CNN classifier
(`DynamicsCNN`, 88.6% mark accuracy) feeding a position-based attachment heuristic
(3.35% of 142,243 real notes correctly carry an attached mark). That pipeline works today,
independent of anything below. The structured-notation head exists as a deliberate,
separate attempt to get dynamics decoded *inline* (same pass as beam/tie/slur, one
coherent `NoteNotation` instead of a result reconciled from a separate detector chain) -
not because Stage 3 doesn't work, but because inline decoding was judged worth attempting
on its own terms (27.94). See "What would actually fix it," revised below: given Stage 3
already solves the practical problem, unfreezing specifically to rescue the structured
head is a harder case to make than it looked before this correction.

## The headline number is worse than it looks, and also better documented than it looks

**Two independent effects are stacked inside "11.97%," and they point in opposite
directions.**

### It's ~83% a scoring-floor artifact, not model behaviour

`decode_reference`/`decode_predictions` (`training/architecture/transformer/structured_decoding.py`)
map every masked position — padding, BOS/EOS, any non-note token — to each head's
`absent` value before scoring. For `DYNAMIC_HEAD` (and separately `TIE_HEAD`) that value
is `DynamicMark.NONE`, and `dynamic_report`/`PerClassReport` treat `NONE` as a **real,
scoreable class** (correctly, for the positions that were actually asked — see
`_fill_note`'s comment: "NONE is a real prediction... not an absence of one"). The
consequence: every masked position becomes a free, trivially-correct `NONE` prediction
counted in the metric.

Measured directly: `none` support in the eval was **2,721,251**, against only **166,256**
positions the model was actually asked about (the real note count) — a 94% inflation.
Since `none` scores ~0.999 regardless of what the model does, this alone creates a hard
macro-F1 floor of **~1/10 = 0.0999** for a 10-class vocabulary, no matter how bad the head
is. So of the 0.1197: **~0.0999 is the floor, ~0.028 is real signal.**

The same artifact inflates the tie head's `none` support (2,714,877), floor ~0.25 there —
but ties still land at start 0.557 / stop 0.690 / start_and_stop 0.661, comfortably above
their floor. That contrast is the evidence dynamics is broken in a different way than
ties, not just a smaller version of the same thing.

**This floor is a real metrics bug, separate from the dynamics finding below, and is not
fixed.** `decode_reference`'s own docstring ("a masked position cannot be scored as a
correct prediction") is currently false for exactly `DYNAMIC_HEAD` and `TIE_HEAD`. Fixing
it needs a real target-vs-absent distinction threaded through to the metrics, and it would
retroactively change every published tie number — a design decision, not a safe one-liner.
Flagged here so nobody free-rides on the current floor when interpreting either head's
score in the future.

**Independently confirmed from a second angle** by the kern beam/stem/tie probe run the
same day: its tie support came back as *exactly* `607 x n_sequences` (every decoder
position — padding, BOS/EOS, barlines, clefs — not just supervised ones), a 9.2x
inflation over the true target count, traced to the same `_lookup`/`decode_reference`
mechanism. Because tie's real `none` accuracy is already ~0.998, the distortion there is
small (corrected macro ~0.908 vs reported 0.909 in-sample) - it does not change that
head's promotion verdict. But it confirms the mechanism precisely and shows it will not
stay small for every head: dynamics' `none` accuracy is nowhere near tie's, so the same
inflation mechanism does much more damage there (the ~0.0999 floor derived independently
above). One shared bug, two very different-sized consequences depending on how skewed the
head's true class distribution already is.

### The apparent improvement over the prior baseline is also an artifact

An earlier probe (`phase16`, scored on OSSQ with a full ~18-class dynamics vocabulary)
measured macro F1 **0.063**. Refolding that same per-class F1 data onto today's smaller
`TRAINED_DYNAMIC_MARKS` 10-class vocabulary (the fold `trained_dynamic_mark` performs)
gives **~0.114** — statistically identical to the 0.1197 measured here. **The vocabulary
fold moved the metric, not the model.** Per-class, this probe is slightly *worse* than
phase16 on the marks phase16 partially learned (`p` 0.037 vs 0.040, `ff` 0.019 vs 0.053,
`sf` 0.000 vs 0.005).

### What's left after both corrections

Once the floor and the fold-artifact are subtracted out, the dynamics head's real,
attributable contribution is **~0.02-0.03 macro F1** — i.e. it has learned essentially
nothing beyond the majority class. Confusion counts over 400 real staves confirm this
directly: the head predicted `NONE` for every non-none ground-truth position except one
correct `p`:

```
p              -> none:42, p:1
pp             -> none:38
f              -> none:35
other-dynamics -> none:31
mf             -> none:23
```

Ground truth in the same corpus slice (4,543 files, 169,870 annotated notes) has real
support for every trained class — `none` 98.22%, then `p` 989, `f` 558, `pp` 473, `mf` 409,
`other` 214, `sf` 123, `ff` 114, `ppp` 43, `mp` 21. **This is not a tiny-support artifact**
— `p` and `f` have hundreds of examples and still score near zero.

## What was ruled out

Checked and cleared, so nobody re-derives these:

- **Source coverage**: `<dynamics>` only ever appears under `<direction-type>` in the
  corpus (389/389 in a 300-file sample), never under `<notations>`. `_direction_dynamic`/
  `NotationExtractor.handle_direction` already cover 100% of it. No analogue to the
  `<tied>`/`<slur>` string-collapse bug found elsewhere this session.
- **Masking**: `_fill_note` (`structured_targets.py`) supervises `DYNAMIC_HEAD`
  unconditionally, exactly mirroring how `TIE_HEAD` is supervised — `NONE` is a real
  target, not skipped. No masking bug.
- **Attachment**: label counts (3,019 non-none marks / 4,543 sidecars) line up with the
  eval's demand (2,965 / 4,488). A visual spot-check (`IMSLP112763-sys1-v0.png`) confirmed
  a label (`other-dynamics` then `pp`) matches real "più f" / "pp" markings at the correct
  notes in the image. Not a crop bug, not a misattachment bug.

## Root cause

Two things, and only one of them is fixable by reweighting:

1. **No class reweighting was used in this probe.** The training log shows no
   `--class-weights` and no focal gamma for this run — plain cross-entropy against a
   98.2%-majority class does exactly what it's supposed to (loss fell 0.1415 -> 0.0838
   over 12 epochs, tracking the prior, not the signal).
2. **The frozen core has never been trained to represent dynamics at all.** The dynamics
   head is one `nn.Linear` (`structured_heads.py:92`) reading a hidden state produced by a
   core whose own training objective — the six-branch token vocabulary
   (`homr/transformer/vocabulary.py`) — **has dynamics tokens commented out** (lines
   92-99). Every other head in this probe predicts something with real footprint in what
   the core already decodes: rhythm tokens -> beam levels and advance; `slurStart`/
   `slurStop` -> tie and slur; pitch height -> stem. Dynamics is the only head whose
   target the core was never asked to notice. A linear readout has nothing to read.

Reweighting (lever 1) can move the number a little — phase16 measured single-digit
macro-F1 gains on `p`/`f`/`ff` from focal loss / class weights. It cannot fix lever 2: no
amount of reweighting teaches a frozen, dynamics-blind representation to see a `<direction>`
element it was never trained to attend to.

## What would actually fix it (revised)

**Do the cheap diagnostic RUNLOG already identified as the next step before touching
architecture at all.** RUNLOG's diagnostic pass found `mf`/`mp`/`ppp` specifically behave
like an "unmarked" detector (87-92% predicted `NONE`) rather than a misclassifier, and
proposed - but never did - looking at the actual visual crops at those positions: mark
size, position within the crop relative to the note, rendering artifacts. That is strictly
cheaper than any of the below and was the planned next move before this thread stalled.

**Given Stage 3 already solves the practical problem, weigh whether the inline structured
head is worth further investment at all before reaching for unfreezing specifically.**
Unfreezing has a demonstrated cost in this exact codebase - score-profile conditioning
went from a clean +0.0615 mean ablation delta (positive in all 10 epochs) under a frozen
core to a delta that "oscillated in sign and ran an order of magnitude below the frozen
run's smallest value" under an otherwise-identical unfrozen run (`docs/writeups/homr-devs.html`,
"Four things that didn't work" #1). That is not hypothetical risk; it is a working
mechanism this project already watched an unfreeze erase. If unfreezing is attempted for
dynamics anyway, it must be gated on the same kind of clean ablation that caught THAT
regression - identical run, only the freeze policy changed, checked against the PRIMARY
six-branch decode's own benchmark, not just the dynamics number.

If, after the diagnostic and the Stage-3-vs-inline tradeoff are both weighed, inline
dynamics is still worth pursuing:

1. Unfreeze the core (fully, or at least enough of it that gradient from the dynamics head
   reaches representations built from the image, not just a frozen readout) — or add
   dynamics marks to the token vocabulary directly (uncomment/build out the commented-out
   entries in `homr/transformer/vocabulary.py:92-99`) so the core's own training objective
   has a reason to encode them.
2. Pass real reweighting for this head once gradient can reach something worth
   reweighting: `--class-weights` and/or `--focal-gamma-head dynamic.mark=<value>` in
   `training/transformer/train_structured_heads.py` — phase16's own numbers are the
   starting point for tuning gamma.
3. Before declaring victory, fix or route around the scoring-floor bug above (this
   document's own biggest caveat) — otherwise a real gain will read as a much smaller
   delta than it is, the same way this run's real ~0.02-0.03 read as "11.97%."
4. Judge the result against the corrected baseline this document establishes: **~0.02-0.03
   real macro F1, not 0.1197 and not phase16's 0.063** — both of those numbers are
   contaminated by artifacts documented above, not honest baselines to beat.

## Sources

Joint probe: `/workspace/b0/lieder-rebuild/advance_probe/eval_report.json` (GPU instance),
checkpoint `pytorch_model_463-69f4db00f98dbb1649cfe2d4f1ee130e8fcb08d1.pth`, corpus
`stage2_clean_advance_probe_manifest.txt` (4,488 examples). Per-class breakdown and
confusion dump: `/workspace/b0/lieder-rebuild/advance_probe/dyn_eval2.log` and
`dyn_predictions.jsonl` (400-staff sample). Prior baseline: `/workspace/b0/phase16.log`
line 24. Ground-truth distribution: computed directly from the 4,543 corpus sidecars.
