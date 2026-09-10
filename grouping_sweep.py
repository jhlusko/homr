"""Does system grouping hold across ensemble sizes, or only the ones it was built for?

It was written for grand staffs, adapted for quartets, and PDMX brings lead sheets, piano
and duos. This sweeps every plausible page shape rather than trusting the hand-picked
cases: staves per system 1..8, systems per page 2..6, with an internal gap of 5 units and
a system gap of 9 - the proportions measured on real quartet pages.
"""
from unittest.mock import MagicMock

from homr.system_grouping import find_system_grouping

UNIT = 10.0
HEIGHT = 45.0


def page(staves_per_system: int, systems: int, internal=5.0, between=9.0):
    spans, y = [], 0.0
    for s in range(systems):
        for k in range(staves_per_system):
            spans.append((y, y + HEIGHT))
            y += HEIGHT + (internal if k < staves_per_system - 1 else between) * UNIT
    out = []
    for lo, hi in spans:
        staff = MagicMock()
        staff.min_y, staff.max_y, staff.average_unit_size = lo, hi, UNIT
        out.append(staff)
    return out


print(f"{'staves':>7} {'systems':>8}  {'result':<28} verdict")
print("-" * 62)
for n in range(1, 9):
    for systems in (2, 3, 4, 6):
        staffs = page(n, systems)
        result = find_system_grouping(staffs, [])
        if result is None:
            got, ok = "declined", (systems < 3 or n == 1)
        else:
            sizes = {len(s) for s in result.best.groups}
            got = f"{len(result.best.groups)} systems of {sorted(sizes)}"
            ok = sizes == {n} and len(result.best.groups) == systems
        flag = "ok" if ok else "<-- CHECK"
        print(f"{n:>7} {systems:>8}  {got:<28} {flag}")
