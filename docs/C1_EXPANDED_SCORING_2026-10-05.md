# Expanded lyric detector scoring pass — 2026-10-05

The eight frozen calibration pages completed the same pinned local OCR and e2 detector pass used for the two-page pilot. Full-page RapidOCR produced 594 text regions in 54 seconds total. The e2 pass took 8 minutes 2 seconds and produced 4–200-pixel component counts of:

| Score | e2 area 4 | e2 area 200 |
|---|---:|---:|
| IMSLP89287 | 40 | 29 |
| IMSLP89290 | 47 | 35 |
| IMSLP617260 | 49 | 36 |
| IMSLP89796 | 52 | 41 |
| IMSLP435041 | 38 | 27 |
| IMSLP286659 | 41 | 39 |
| IMSLP83318 | 44 | 35 |
| IMSLP83314 | 26 | 20 |

The existing raster staff-gap heuristic was applied to the area-200/10% intact selection as a diagnostic. It found usable five-line groups on only four of eight pages: IMSLP89287 (2), IMSLP617260 (6), IMSLP435041 (8), IMSLP83318 (9), and IMSLP83314 (8). IMSLP89290 (0), IMSLP89796 (1), and IMSLP286659 (3) fail the minimum two-group precondition. No production filter was applied and no lyric accuracy score was claimed because visible rows have not been independently transcribed.

This is the expected failure control: the two-page geometry result does not generalize to pages with different staff density, engraving or crop geometry. The next selector must use identified system/staff geometry from the OMR path, or retain full-page OCR while using a learned/validated semantic filter. It must not silently drop these pages.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-layout-expansion-20261005/` (`pages.json`, `full-ocr/`, `e2-region-ocr/`, `intact-gate/`, and geometry failure output). All compute is complete.
