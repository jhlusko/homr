"""Refuse a detector checkpoint whose own training history says a class never learned.

The released `non-lyric-text` detector was measured at **0% precision and 0% recall for
`Tempo`** at full-page box level, and at 0.2% precision overall. That failure was not
discovered by the evaluation - it is written in the history file sitting beside the
weights, and had been since the run finished:

    epoch  loss   Tempo  Fingering  Dynamic  StaffText  Expression  | valid Tempo
       10  0.085  0.024      0.009    0.937      0.911       0.928  |      0.000

`Tempo` at 0.000 validation IoU and `Fingering` at 0.001, while every other class reached
0.87-0.94 and the loss fell steadily the whole way. **Nothing read that file before the
checkpoint was pinned into `pins.py` and shipped.**

The gate is deliberately crude: a floor on per-class IoU, applied to the last epoch's
validation figures (training figures when a run had no validation split). It is not a
quality bar - a class at 0.3 is bad and passes - it exists to catch the case where a class
was never learned at all, which is the case that shipped.

**Run it before pinning anything.** `--json` makes it usable from a release script; the
exit status is 1 when the gate fails, so it can gate a pipeline without parsing output.

**Passing this gate does not mean a checkpoint is usable.** Measured at full-page box
level over the same 8 pages, with `min_area=200`:

    e4 (released, REFUSED here)   630 predicted   2.7% precision   85% recall
    e0 (PASSES, best history)     894 predicted   0.0% precision    0% recall
    e5 (PASSES)                   158 predicted   6.3% precision   50% recall

`e0` scores 0.875-0.99 validation IoU on every class and recovers **not one correct box**.
Per-class patch IoU does not predict page-level detection - which is what the roadmap
means by "patch IoU is not admissible; it hid this three times", now with a number on it.
So this gate is a floor on one failure mode, not a release criterion: it catches a class
that never learned, and says nothing about whether the ones that did are any use.

*Known limitation.* `Fingering` reads exactly 0.875 in five of six runs, `StaffText` 0.827
in three, `Expression` 0.806 in two - identical figures across independent runs mean very
small validation support, so a 0.000 may be one missed instance rather than a class never
learned. The history does not record per-class support; until it does, a refusal is a
reason to look, not proof on its own. For the released `non-lyric-text` the refusal is
independently confirmed: 0% precision and recall for `Tempo` over 264 boxes on 299 pages.
"""

# flake8: noqa: T201

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

#: A class below this never learned. Chosen an order of magnitude above the failures it
#: has to catch (0.000 and 0.001) and well below every class that did learn in the same
#: runs (0.78 at worst), so it separates the two without pretending to be a quality bar.
FLOOR = 0.05

#: Classes a run may legitimately have no data for. `detector_data`'s own docstring notes
#: MeasureNumber is derivable rather than detected; a run that excludes it is not broken.
OPTIONAL = frozenset({"MeasureNumber"})


@dataclass
class Verdict:
    checkpoint: str
    epochs: int
    source: str
    failed: dict[str, float]
    passed: dict[str, float]
    missing: list[str]

    @property
    def ok(self) -> bool:
        return not self.failed and not self.missing


def inspect(history_path: Path, floor: float = FLOOR) -> Verdict:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    history = payload.get("history") or []
    if not history:
        raise SystemExit(f"{history_path}: no epochs recorded")
    last = history[-1]

    # Validation when the run had a split, training otherwise - and which one is reported,
    # because a gate that silently falls back to training figures is a gate that passes a
    # model which memorised its patches.
    scores = last.get("valid")
    source = "validation"
    if not scores:
        scores = {k: v for k, v in last.items() if isinstance(v, float) and k != "loss"}
        source = "training (no validation split in this run)"

    classes = [c for c in payload.get("classes", []) if c != "background"]
    failed: dict[str, float] = {}
    passed: dict[str, float] = {}
    missing: list[str] = []
    for name in classes:
        if name in OPTIONAL and name not in scores:
            continue
        if name not in scores:
            missing.append(name)
            continue
        value = float(scores[name])
        (failed if value < floor else passed)[name] = value
    return Verdict(str(history_path), len(history), source, failed, passed, missing)


def describe(verdict: Verdict, floor: float) -> str:
    lines = [
        f"{verdict.checkpoint}",
        f"  {verdict.epochs} epochs, scored on {verdict.source}, floor {floor:.2f}",
    ]
    for name, value in sorted(verdict.failed.items(), key=lambda kv: kv[1]):
        lines.append(f"  FAIL  {name:<16}{value:.3f}   never learned")
    for name in verdict.missing:
        lines.append(f"  FAIL  {name:<16}    -    no score recorded")
    for name, value in sorted(verdict.passed.items(), key=lambda kv: -kv[1]):
        lines.append(f"  ok    {name:<16}{value:.3f}")
    lines.append("  PASS" if verdict.ok else "  REFUSED - do not pin this checkpoint")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("history", type=Path, nargs="+", help="train_detector.py --out files")
    parser.add_argument("--floor", type=float, default=FLOOR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    verdicts = [inspect(path, args.floor) for path in args.history]
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "checkpoint": v.checkpoint,
                        "ok": v.ok,
                        "epochs": v.epochs,
                        "source": v.source,
                        "failed": v.failed,
                        "missing": v.missing,
                        "passed": v.passed,
                    }
                    for v in verdicts
                ],
                indent=1,
            )
        )
    else:
        for verdict in verdicts:
            print(describe(verdict, args.floor))
            print()
    sys.exit(0 if all(v.ok for v in verdicts) else 1)


if __name__ == "__main__":
    main()
