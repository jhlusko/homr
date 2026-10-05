# Conditional tie and phrase span audit — 2026-10-05

Four frozen real pages, two saved policy outputs per page. Numbered phrase endpoints are paired within each part; sounding ties and engraved ties are paired separately by voice, staff and pitch. Stops are processed before starts on tie-chain notes, regardless of XML tag order. No inference, exporter or model changes.

Only complete spans whose endpoints are unique timing/pitch anchors common to source and both predictions enter the comparison. This excludes recognition failures, duplicate unisons, unsupported continuation tags, ambiguous pairing and page-boundary spans. Pairing follows XML note order. Cross-voice/staff tie ownership is not inferred. These are conditional source-XML span counts, not complete scan engraving accuracy. Source and predicted orphan/unclosed endpoints are reported separately; they can include valid page-boundary events.

| Page | Family | Source spans | Accepted matched / missing / extra | Joint matched / missing / extra |
|---|---|---|---|---|
| IMSLP154086 | phrase | 29 | 20 / 9 / 9 | 23 / 6 / 6 |
| IMSLP154086 | sounding_tie | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| IMSLP89797 | phrase | 33 | 13 / 20 / 43 | 16 / 17 / 39 |
| IMSLP89797 | sounding_tie | 24 | 11 / 13 / 0 | 14 / 10 / 1 |
| sq7358579 | phrase | 42 | 32 / 10 / 32 | 34 / 8 / 30 |
| sq7358579 | sounding_tie | 30 | 26 / 4 / 0 | 30 / 0 / 0 |
| sq9146376 | phrase | 43 | 43 / 0 / 7 | 43 / 0 / 7 |
| sq9146376 | sounding_tie | 7 | 7 / 0 / 0 | 7 / 0 / 0 |

Engraved-tie counts equal sounding-tie counts in all eight arms. Joint Glazunov matches all 30 supported ties versus 26 for accepted; joint Franz matches 14/24 versus 11/24, with one extra tie. Joint phrase matching improves on Chaminade, Franz and Glazunov; Haydn is unchanged. Both models retain substantial extra phrase spans, consistent with the legacy tie/phrase endpoint alias. A side-only fallback cannot resolve this ownership problem.

Next: identify extra phrase spans whose endpoint pair exactly equals a sounding tie, retain simultaneous true tie/phrase controls, and prepare a reversible exporter experiment on cached data. Do not delete every tie-associated phrase or change defaults based on these four pages.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/notation-span-audit-20261005-v2/report.json` and `executed-source/`. The preliminary unsorted-chain run in sibling `notation-span-audit-20261005/` is superseded. Five pairing tests cover tie chains, simultaneous events, ambiguity, page boundaries and missing shared anchors.
