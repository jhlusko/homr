"""Diagnostic-only tuplet removal constrained by predicted timing context.

No reference labels are consulted. This is not wired into default inference.
"""

from collections import Counter
from fractions import Fraction

from homr.transformer import vocabulary as v


def _material(chord):
    return [s for s in chord if s.rhythm.startswith(("note_", "rest_"))]


def guarded_tuplets(chords):
    measures = v._group_into_measures(chords)
    if len(measures) < 3:
        return chords
    # A grand staff or unequal chord durations do not have a reliable single clock.
    positions = {s.position for c in chords for s in _material(c)}
    if (
        not positions <= {"upper", "lower"}
        or len(positions) > 1
        or any(len({s.get_duration().fraction for s in _material(c)}) > 1 for c in chords)
    ):
        return chords
    contexts = []
    meter = None
    pending = None
    region = 0
    for measure in measures:
        seen_material = False
        ambiguous = False
        for chord in measure:
            for symbol in chord:
                rhythm = symbol.rhythm
                if rhythm.startswith(v.TIME_SIGNATURE_BEATS_PREFIX):
                    meter = None
                    pending = int(rhythm.removeprefix(v.TIME_SIGNATURE_BEATS_PREFIX))
                    region += 1
                    ambiguous |= seen_material
                elif rhythm.startswith("timeSignature/"):
                    denominator = int(rhythm.split("/", 1)[1])
                    meter = Fraction(pending, denominator) if pending and denominator > 0 else None
                    pending = None
                    region += 1
                    ambiguous |= seen_material
                seen_material |= rhythm.startswith(("note_", "rest_"))
        closed = bool(
            measure
            and measure[-1]
            and ("barline" in measure[-1][0].rhythm or "repeat" in measure[-1][0].rhythm)
        )
        contexts.append((region, meter, ambiguous, closed))
    # Estimate only from closed interior bars without any predicted tuplets.
    samples = {}
    for index, (measure, context) in enumerate(zip(measures, contexts)):
        scope, _, ambiguous, closed = context
        if index in (0, len(measures) - 1) or ambiguous or not closed:
            continue
        if any(s.is_tuplet() for c in measure for s in _material(c)):
            continue
        duration = v._get_duration_of_measure(measure)
        if duration > 0:
            samples.setdefault(scope, Counter())[duration] += 1
    inferred = {}
    for scope, counts in samples.items():
        ranked = counts.most_common()
        target, support = ranked[0]
        if support >= 3 and support * 3 >= sum(counts.values()) * 2:
            inferred[scope] = target
    result = []
    for index, (measure, context) in enumerate(zip(measures, contexts)):
        scope, meter, ambiguous, closed = context
        target = meter if meter is not None else inferred.get(scope)
        if index in (0, len(measures) - 1) or ambiguous or not closed or target is None:
            result.append(measure)
            continue
        current = v._get_duration_of_measure(measure)
        if current <= 0 or current >= target:
            result.append(measure)
            continue
        rewritten = v._remove_tuplets(measure)
        result.append(rewritten if v._get_duration_of_measure(rewritten) == target else measure)
    return v._flatten_measures(result)
