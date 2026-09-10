"""The page shapes a wider ensemble range actually introduces.

Uniform pages of 1-8 staves already hold. These are the ones that break the assumption
the module rests on - that the gap inside a system is smaller than the gap between them.
"""
from unittest.mock import MagicMock

from homr.system_grouping import find_system_grouping

UNIT, HEIGHT = 10.0, 45.0


def build(gap_pattern: list[float]):
    """Staves whose successive gaps are exactly `gap_pattern`, in unit sizes."""
    spans, y = [(0.0, HEIGHT)], HEIGHT
    for gap in gap_pattern:
        y += gap * UNIT
        spans.append((y, y + HEIGHT))
        y += HEIGHT
    out = []
    for lo, hi in spans:
        s = MagicMock()
        s.min_y, s.max_y, s.average_unit_size = lo, hi, UNIT
        out.append(s)
    return out


def report(name, gaps, expect):
    result = find_system_grouping(build(gaps), set())
    if result is None:
        got = "declined"
    else:
        got = f"{[len(g) for g in result.best.groups]} confident={result.confident}"
    print(f"{name:<46} {got:<34} expected {expect}")


# 1. An orchestral system: staves grouped by family, so a gap INSIDE the system (9) is
#    as large as the gap between systems (12). This is the assumption under strain.
report("orchestral, family gaps 5/5/9 inside, 12 between",
       [5, 5, 9, 5, 5, 12, 5, 5, 9, 5, 5, 12, 5, 5, 9, 5, 5],
       "3 systems of 6")

# 2. Above the configured cap of 8 staves per system.
report("12 staves per system, 3 systems",
       ([5] * 11 + [9]) * 2 + [5] * 11,
       "3 systems of 12")

# 3. Systems of different sizes, which orchestral engraving does when instruments rest.
report("mixed sizes 4 / 3 / 4",
       [5, 5, 5, 9, 5, 5, 9, 5, 5, 5],
       "declines or 4/3/4")

# 4. A duo where the two staves sit closer than a grand staff's.
report("duo, tight 3 inside, 10 between",
       [3, 10, 3, 10, 3, 10, 3],
       "4 systems of 2")

# 5. Lead sheet: one staff per system, evenly spaced - no multi-staff evidence at all.
report("lead sheet, 1 staff per system",
       [9, 9, 9, 9, 9],
       "declined")
