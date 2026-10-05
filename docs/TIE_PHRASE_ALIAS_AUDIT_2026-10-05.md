# Exact tie / phrase duplicate audit — 2026-10-05

Cached source-assisted diagnostic, not an inference rule. Complete predicted phrase spans were compared with BOTH sounding and engraved predicted ties on unique timing/pitch anchors common to source and both model arms. The source then independently classified each duplicate. Pairing limitations from the conditional span audit remain.

| Page | Accepted: source tie only | Joint: source tie only | Joint: source phrase only |
|---|---|---|---|
| IMSLP154086 | 0 | 0 | 0 |
| IMSLP89797 | 8 | 8 | 1 |
| sq7358579 | 23 | 29 | 0 |
| sq9146376 | 7 | 7 | 0 |

There are38 accepted and44 joint exact duplicates that the source represents as ties only. They explain a substantial subset of the extra phrase spans. Joint also contains one Franz duplicate that is a source-correct phrase and an extra predicted tie. Blindly deleting all duplicate phrases would remove that correct phrase. The source is unavailable at inference, so source-assisted classification cannot become a shipping filter.

No complete duplicate in this bounded support has both source tie and phrase with exactly the same endpoints. This does not contradict the earlier seven simultaneous starts: a phrase can start on a tied note and end later than the tie. Start-only tie suppression remains unsafe. Tests explicitly protect a hypothetical exact simultaneous tie/phrase pair.

Next: recover provenance at both endpoints of the38/44 source tie-only aliases and the one harmful Franz counterexample, then evaluate an opt-in suppression hypothesis using structured phrase event confidence and complete span ownership. Maintain true simultaneous starts, true phrases without source ties, boundary spans and recognition controls. Ground truth may score, but must not select runtime deletions. No runtime or default changes made.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/tie-phrase-alias-audit-20261005/report.json` plus `executed-source/`. Full endpoint keys and physical measure indices are recorded for every duplicate.
