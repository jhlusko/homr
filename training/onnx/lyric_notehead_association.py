"""Conservative diagnostic association; no count equality or native labels required."""

import math


def associate_x(decoded, detected, *, max_distance, ambiguity_margin):
    """Return mutual nearest pairs with explicit unresolved notes.

    Coordinates must already share page units. Caller supplies distance gates;
    they are experimental, not calibrated production defaults. Equal-x chords
    and near ties are deliberately unresolved. Never retry after removing a
    matched point: that would let a missing head cascade into shifted matches.
    """
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be positive and finite")
    if not math.isfinite(ambiguity_margin) or ambiguity_margin <= 0:
        raise ValueError("ambiguity_margin must be positive and finite")
    if any(not math.isfinite(x) for x in [*decoded, *detected]):
        raise ValueError("Coordinates must be finite")

    def nearest(x, candidates):
        distances = sorted((abs(x - value), index) for index, value in enumerate(candidates))
        if not distances or distances[0][0] > max_distance:
            return None
        if len(distances) > 1 and distances[1][0] - distances[0][0] < ambiguity_margin:
            return None
        return distances[0][1]

    forward = [nearest(x, detected) for x in decoded]
    reverse = [nearest(x, decoded) for x in detected]
    pairs = [(i, j) for i, j in enumerate(forward) if j is not None and reverse[j] == i]
    return {
        "pairs": pairs,
        "unresolved_decoded": [i for i in range(len(decoded)) if i not in {p[0] for p in pairs}],
        "unresolved_detected": [j for j in range(len(detected)) if j not in {p[1] for p in pairs}],
    }
