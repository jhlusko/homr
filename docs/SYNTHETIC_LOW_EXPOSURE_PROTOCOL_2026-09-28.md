# Coverage-preserving low-exposure diagnostic — 2026-09-28

Frozen before fitting: the preceding fixed-pixel pilot failed its gates. It used
43,600 real draws in each mixture versus 64,000 in real-only. This experiment
reduces synthetic dose while retaining that original 64,000-real-draw multiset.
It changes dose and real coverage together; it cannot separately attribute cause.

Three fresh arms use 8,400 updates, batch eight, identical initialization and the
unchanged synthetic_head_transfer runner, optimizer, frozen public core, pixels,
internal validation and original regression/invention gates:

- Real-only: original 64,000 draws followed by 3,200 proportionally sampled real draws.
- Unordered: original 64,000 real draws plus 3,200 qualified synthetic draws, shuffled.
- Curriculum: preserves original real draw order, interleaves 1,600 synthetic draws
  among the first 16,000 real draws, 800 among the next 16,000, and 800 among the
  final 32,000. All stages use the complete qualified training-family/font pool.

Mixtures contain 4.76% synthetic draws. Their literal multiset is identical.
The control matches compute and exceeds mixture real exposure by 5%; any gain is
therefore not due to more real examples or more optimizer updates in the mixture.
The prior 8,000-update real-only control is an additional descriptive reference.

Do not select a candidate unless every prespecified per-source guard passes.
If neither passes, stop training and inspect labels/domain mismatch; do not sweep
ratios until one appears favorable. One seed and inspected internal validation
remain diagnostic, not release evidence. No deployment or publication.
