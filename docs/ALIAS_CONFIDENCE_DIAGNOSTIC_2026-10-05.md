# Cached alias-confidence diagnostic — 2026-10-05

All eight policy XML arms replay exactly with provenance now saved at every pitched endpoint. No new recognition or XML mutation. The deletion hypothesis is restricted to complete phrase spans exactly duplicating both exported sounding and engraved ties. At both endpoints, both structured phrase slots must be NONE, raw event choices must confidently be NONE, and tie choices must confidently own the relevant START/STOP (or START_AND_STOP). A single threshold applies to all six confidence checks. Reference categories are used only after deciding, to score deletions.

| Threshold | Accepted tie-only aliases removed / 38 | Joint tie-only aliases removed / 44 | Joint correct phrases removed / 1 |
|---|---|---|---|
| 0.80 | 5 | 30 | 1 |
| 0.90 | 4 | 21 | 0 |
| 0.95 | 0 | 10 | 0 |
| 0.98 | 0 | 3 | 0 |

The0.90 diagnostic removes21 joint aliases without removing the single known Franz phrase;0.80 removes30 but loses that phrase. Accepted-head confidence differs markedly, so this is not a universal calibrated threshold. The grid was evaluated after earlier calibration error review and is exploratory. One preserved counterexample does not establish recall or safety on broader true phrases.

No shipping rule or XML edit was added. Next implement an opt-in cached export experiment that removes only the owned start/stop pair and verifies exact recognition, ties, unaffected phrases, multiplicity and boundaries. Validate independent true-phrase controls and broader pages before choosing a threshold; do not require this improvement to block otherwise qualified release work.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/alias-confidence-grid-20261005/report.json` (all choices, confidences and scored outcomes); exact replay provenance in sibling `slur-all-endpoints-20261005/report.json`. Executed sources are preserved.
