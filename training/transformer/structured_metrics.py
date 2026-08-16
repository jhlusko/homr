"""
Metrics for the beam, stem and slur heads.

Accuracy over every position is close to meaningless here. Most notes carry no beams, most
notes are not slur endpoints, and a head that answered NONE everywhere would score well on
all three tasks while having learned nothing. So each measure below is restricted to the
positions where its question is actually asked, and the rare classes are reported
separately rather than averaged away.

Three shapes of measure, for three different failure modes:

  per-class F1        catches a head that has collapsed onto the majority class - macro
                      across classes, so a class with a few hundred examples counts as
                      much as one with hundreds of thousands.
  exact vector match  catches a head that is right about each level independently and
                      wrong about the note. A beam vector is one engraving decision;
                      getting three levels of four right does not render.
  endpoint pairing    catches a slur head that emits plausible starts and stops which do
                      not join up. A span is the unit that means something, not its ends.

The beam figures also come with the automatic-beaming baseline alongside, because the
question Gate C asks is not whether a head is accurate but whether it beats what a rule
already gives for free.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
)


@dataclass
class ClassMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def support(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class PerClassReport:
    classes: dict[str, ClassMetrics] = field(default_factory=dict)

    def observe(self, predicted: str, actual: str) -> None:
        for name in (predicted, actual):
            self.classes.setdefault(name, ClassMetrics())
        if predicted == actual:
            self.classes[actual].true_positive += 1
        else:
            self.classes[predicted].false_positive += 1
            self.classes[actual].false_negative += 1

    @property
    def macro_f1(self) -> float:
        """Averaged over classes that actually occur, so a rare class counts fully.

        Classes with no support are left out rather than scored 0, which would let the
        figure depend on how many classes the vocabulary happens to define.
        """
        present = [m for m in self.classes.values() if m.support]
        return sum(m.f1 for m in present) / len(present) if present else 0.0

    @property
    def micro_accuracy(self) -> float:
        correct = sum(m.true_positive for m in self.classes.values())
        total = sum(m.support for m in self.classes.values())
        return correct / total if total else 0.0

    def describe(self) -> str:
        rows = sorted(self.classes.items(), key=lambda kv: -kv[1].support)
        listed = "  ".join(f"{name}:F1={m.f1:.3f}(n={m.support})" for name, m in rows if m.support)
        return f"macro F1 {self.macro_f1:.3f}  micro {self.micro_accuracy:.3f}  {listed}"


def beam_level_report(
    predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation], level: int
) -> PerClassReport:
    """One level's states, over the notes whose duration can carry that level.

    Notes the level does not apply to are excluded rather than counted as correct
    NOT_APPLICABLE, which the rhythm token already determines.
    """
    report = PerClassReport()
    for left, right in zip(predicted, actual, strict=True):
        expected = right.beam_levels[level - 1]
        if expected == BeamLevelState.NOT_APPLICABLE:
            continue
        report.observe(str(left.beam_levels[level - 1]), str(expected))
    return report


def exact_vector_accuracy(
    predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation], levels: int
) -> tuple[int, int]:
    """(notes whose whole beam vector matches, notes where either side beams anything).

    A beam vector is one engraving decision. Three levels right out of four does not
    render, so partial credit would flatter a head that never gets a note wholly right.
    """
    matching = comparable = 0
    for left, right in zip(predicted, actual, strict=True):
        first = tuple(left.beam_levels[:levels])
        second = tuple(right.beam_levels[:levels])
        if all(s == BeamLevelState.NOT_APPLICABLE for s in first + second):
            continue
        comparable += 1
        matching += first == second
    return matching, comparable


def hook_report(
    predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation], levels: int
) -> ClassMetrics:
    """Hooks specifically, pooled across levels.

    They are rare, they are what MuseScore's BeamMode discards, and they are the clearest
    case of information only the image carries - so they get their own figure rather than
    disappearing into a macro average.
    """
    hooks = {BeamLevelState.FORWARD_HOOK, BeamLevelState.BACKWARD_HOOK}
    metrics = ClassMetrics()
    for left, right in zip(predicted, actual, strict=True):
        for level in range(levels):
            got, want = left.beam_levels[level], right.beam_levels[level]
            if got == want and want in hooks:
                metrics.true_positive += 1
                continue
            # Not an elif chain: a backward hook where the engraving has a forward one is
            # both a missed hook and an invented one, and charging only the miss would
            # hide a head that has the right idea and the wrong direction.
            if want in hooks:
                metrics.false_negative += 1
            if got in hooks:
                metrics.false_positive += 1
    return metrics


def stem_report(
    predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation]
) -> PerClassReport:
    """Stem direction, over notes whose reference states one.

    UNKNOWN marks a source that does not say, so scoring it would measure the dataset
    rather than the model.
    """
    report = PerClassReport()
    for left, right in zip(predicted, actual, strict=True):
        if right.stem == StemDirection.UNKNOWN:
            continue
        report.observe(str(left.stem), str(right.stem))
    return report


def tie_report(
    predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation]
) -> PerClassReport:
    """Tie state over every note, including the ones with no tie.

    NONE is scored here, unlike UNSPECIFIED for a slur side, because it is a real
    prediction rather than a silent source: the engraving says plainly whether a note is
    tied. That makes the classes extremely unbalanced - roughly 67,000 tie events against
    a million notes - which is exactly what the macro average is for.
    """
    report = PerClassReport()
    for left, right in zip(predicted, actual, strict=True):
        report.observe(str(left.tie), str(right.tie))
    return report


def slur_side_report(
    predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation], slots: int
) -> PerClassReport:
    """Which way a slur bends, over the endpoints whose reference states a direction.

    UNSPECIFIED is excluded rather than scored. Roughly half the slurs in both corpora
    state no placement (27.30), and that is a silent source rather than a third direction -
    scoring it would measure how often the engraver bothered rather than how well the head
    reads the page.

    Only endpoints count. A note in the middle of a span has no side to read off the
    image, so including the NONE positions would drown the measure in free correct answers
    the same way scoring every position would.
    """
    report = PerClassReport()
    for left, right in zip(predicted, actual, strict=True):
        for slot in range(slots):
            event, side = right.slurs[slot]
            if event == SlurEvent.NONE or side == SlurSide.UNSPECIFIED:
                continue
            report.observe(str(left.slurs[slot][1]), str(side))
    return report


def slur_endpoint_pairs(notation: Sequence[NoteNotation], slot: int) -> set[tuple[int, int]]:
    """(start, stop) index pairs for the spans one slot actually closes.

    Endpoints alone are the wrong unit: a head can emit a plausible start and a plausible
    stop that do not belong to each other and score well on both. A span is what means
    something, so a start with no stop contributes nothing here and shows up as a miss.
    """
    pairs: set[tuple[int, int]] = set()
    open_at: int | None = None
    for index, note in enumerate(notation):
        event = note.slurs[slot - 1][0]
        if event in (SlurEvent.START, SlurEvent.START_AND_STOP):
            if event == SlurEvent.START_AND_STOP and open_at is not None:
                pairs.add((open_at, index))
            open_at = index
        elif event == SlurEvent.STOP and open_at is not None:
            pairs.add((open_at, index))
            open_at = None
    return pairs


def slur_span_report(
    predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation], slots: int
) -> ClassMetrics:
    """Complete spans matched by both endpoints, pooled across slots."""
    metrics = ClassMetrics()
    for slot in range(1, slots + 1):
        got = slur_endpoint_pairs(predicted, slot)
        want = slur_endpoint_pairs(actual, slot)
        metrics.true_positive += len(got & want)
        metrics.false_positive += len(got - want)
        metrics.false_negative += len(want - got)
    return metrics


def _merge(into: ClassMetrics, other: ClassMetrics) -> None:
    into.true_positive += other.true_positive
    into.false_positive += other.false_positive
    into.false_negative += other.false_negative


@dataclass
class Evaluation:
    """Every measure, accumulated across a whole evaluation set.

    Sequences are added one at a time because the sequence-level measures - exact beam
    vectors, slur spans - only mean anything within one staff. Pooling positions from
    different staves first would let a slur opened on one close on another.
    """

    beam_levels: int
    slur_slots: int
    per_level: dict[int, PerClassReport] = field(default_factory=dict)
    hooks: ClassMetrics = field(default_factory=ClassMetrics)
    stems: PerClassReport = field(default_factory=PerClassReport)
    slur_spans: ClassMetrics = field(default_factory=ClassMetrics)
    slur_sides: PerClassReport = field(default_factory=PerClassReport)
    ties: PerClassReport = field(default_factory=PerClassReport)
    vectors_matching: int = 0
    vectors_total: int = 0
    sequences: int = 0

    def observe(self, predicted: Sequence[NoteNotation], actual: Sequence[NoteNotation]) -> None:
        self.sequences += 1
        for level in range(1, self.beam_levels + 1):
            report = self.per_level.setdefault(level, PerClassReport())
            for name, metrics in beam_level_report(predicted, actual, level).classes.items():
                _merge(report.classes.setdefault(name, ClassMetrics()), metrics)

        matching, total = exact_vector_accuracy(predicted, actual, self.beam_levels)
        self.vectors_matching += matching
        self.vectors_total += total

        _merge(self.hooks, hook_report(predicted, actual, self.beam_levels))
        _merge(self.slur_spans, slur_span_report(predicted, actual, self.slur_slots))
        for name, metrics in slur_side_report(predicted, actual, self.slur_slots).classes.items():
            _merge(self.slur_sides.classes.setdefault(name, ClassMetrics()), metrics)
        for name, metrics in tie_report(predicted, actual).classes.items():
            _merge(self.ties.classes.setdefault(name, ClassMetrics()), metrics)
        for name, metrics in stem_report(predicted, actual).classes.items():
            _merge(self.stems.classes.setdefault(name, ClassMetrics()), metrics)

    @property
    def exact_vector_rate(self) -> float:
        return self.vectors_matching / self.vectors_total if self.vectors_total else 0.0

    @property
    def stem_direction_accuracy(self) -> float:
        """Accuracy over notes that actually have a stem.

        The plain micro figure includes NOT_APPLICABLE - rests, whole notes - which the
        head gets right almost for free and which the rhythm token already determines.
        `stem_baseline.py` scores only notes with a stated direction, so this is the
        figure that can be set against it; the micro number cannot.
        """
        directional = ("up", "down")
        correct = sum(
            self.stems.classes[name].true_positive
            for name in directional
            if name in self.stems.classes
        )
        total = sum(
            self.stems.classes[name].support for name in directional if name in self.stems.classes
        )
        return correct / total if total else 0.0

    def describe(self) -> str:
        lines = [f"{self.sequences:,} sequence(s)", ""]
        for level in sorted(self.per_level):
            report = self.per_level[level]
            if report.classes:
                lines.append(f"beam level {level}: {report.describe()}")
        lines.append(
            f"exact beam vector: {self.exact_vector_rate:.3f} "
            f"({self.vectors_matching:,}/{self.vectors_total:,})"
        )
        lines.append(
            f"hooks: F1={self.hooks.f1:.3f} P={self.hooks.precision:.3f} "
            f"R={self.hooks.recall:.3f} (n={self.hooks.support:,})"
        )
        lines.append(f"stems: {self.stems.describe()}")
        lines.append(
            f"stem direction only (up/down, comparable to stem_baseline): "
            f"{self.stem_direction_accuracy:.3f}"
        )
        lines.append(
            f"slur spans: F1={self.slur_spans.f1:.3f} P={self.slur_spans.precision:.3f} "
            f"R={self.slur_spans.recall:.3f} (n={self.slur_spans.support:,})"
        )
        # Only where something predicted a direction. Before the placement recovery of
        # 27.22 this head had no targets at all, and a zeroed row would read as a head
        # scoring nothing rather than a capability that does not exist.
        if any(metrics.support for metrics in self.slur_sides.classes.values()):
            lines.append(f"slur sides (above/below): {self.slur_sides.describe()}")
        # A tie head that has never been trained predicts NONE everywhere and would score
        # a flattering micro against a corpus that is mostly untied, so this is reported
        # only when something actually predicted a tie.
        if any(name != "none" and metrics.support for name, metrics in self.ties.classes.items()):
            lines.append(f"ties: {self.ties.describe()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "sequences": self.sequences,
            "beam_levels": {
                str(level): {
                    "macro_f1": report.macro_f1,
                    "micro_accuracy": report.micro_accuracy,
                    "support": sum(m.support for m in report.classes.values()),
                }
                for level, report in sorted(self.per_level.items())
            },
            "exact_beam_vector": {
                "rate": self.exact_vector_rate,
                "matching": self.vectors_matching,
                "total": self.vectors_total,
            },
            "hooks": {"f1": self.hooks.f1, "support": self.hooks.support},
            "stems": {
                "macro_f1": self.stems.macro_f1,
                "micro": self.stems.micro_accuracy,
                "direction_only": self.stem_direction_accuracy,
            },
            "slur_spans": {"f1": self.slur_spans.f1, "support": self.slur_spans.support},
            "ties": {
                "macro_f1": self.ties.macro_f1,
                "support": sum(
                    m.support for name, m in self.ties.classes.items() if name != "none"
                ),
            },
            "slur_sides": {
                "macro_f1": self.slur_sides.macro_f1,
                "support": sum(m.support for m in self.slur_sides.classes.values()),
            },
        }
