"""Conservative diagnostic association; no count equality or native labels required."""

import math
import re


def pitch_staff_position(pitch, clef):
    """Diatonic half-line units: bottom staff line is 1; unknown clefs fail closed."""
    note = re.fullmatch(r"([A-G])(?:[#b]*)(-?\d+)", pitch)
    sign = re.fullmatch(r"clef_([GFC])([1-5])", clef)
    if note is None or sign is None:
        raise ValueError("Unsupported pitch or clef")
    steps = "CDEFGAB"
    reference = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}
    step, octave = reference[sign[1]]
    bottom = 7 * octave + steps.index(step) - 2 * (int(sign[2]) - 1)
    return 7 * int(note[2]) + steps.index(note[1]) - bottom + 1


def reject_pitch_conflicts(result, decoded_positions, detected_positions):
    """Reject conflicting pairs without rematching to another nearby head."""
    pairs = []
    rejected = []
    for i, j in result["pairs"]:
        if decoded_positions[i] == detected_positions[j]:
            pairs.append((i, j))
        else:
            rejected.append((i, j))
    return {
        "pairs": pairs,
        "pitch_conflicts": rejected,
        "unresolved_decoded": sorted([*result["unresolved_decoded"], *(i for i, _ in rejected)]),
        "unresolved_detected": sorted([*result["unresolved_detected"], *(j for _, j in rejected)]),
    }


def forced_sequence_pairs(decoded_positions, detected_positions):
    """Pairs present in every maximum ordered equal-pitch alignment.

    No coordinates or native labels enter this sequence check. Insertions,
    deletions and repeated pitches can create multiple optimal alignments;
    only pairs common to all are returned. Caller must establish monophonic
    chronological order: equal-x chords/grandstaffs are not supported.
    """
    n, m = len(decoded_positions), len(detected_positions)
    prefix = [[0] * (m + 1) for _ in range(n + 1)]
    suffix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            prefix[i + 1][j + 1] = max(
                prefix[i][j + 1],
                prefix[i + 1][j],
                prefix[i][j] + int(decoded_positions[i] == detected_positions[j]),
            )
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            suffix[i][j] = max(
                suffix[i + 1][j],
                suffix[i][j + 1],
                suffix[i + 1][j + 1] + int(decoded_positions[i] == detected_positions[j]),
            )
    best = prefix[n][m]
    forced = []
    for i in range(n):
        possible = [
            j
            for j in range(m)
            if decoded_positions[i] == detected_positions[j]
            and prefix[i][j] + 1 + suffix[i + 1][j + 1] == best
        ]
        can_skip = any(prefix[i][j] + suffix[i + 1][j] == best for j in range(m + 1))
        if len(possible) == 1 and not can_skip:
            forced.append((i, possible[0]))
    return forced


def reject_sequence_ambiguity(result, decoded_positions, detected_positions):
    forced = set(forced_sequence_pairs(decoded_positions, detected_positions))
    rejected = [(i, j) for i, j in result["pairs"] if (i, j) not in forced]
    return {
        **result,
        "pairs": [(i, j) for i, j in result["pairs"] if (i, j) in forced],
        "sequence_rejected": rejected,
        "unresolved_decoded": sorted([*result["unresolved_decoded"], *(i for i, _ in rejected)]),
        "unresolved_detected": sorted([*result["unresolved_detected"], *(j for _, j in rejected)]),
    }


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
