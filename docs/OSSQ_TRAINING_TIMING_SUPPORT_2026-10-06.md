# Unchanged OSSQ training timing support — 2026-10-06

Audited all 716 Borodin quartet training crops available in the verified local
public shard. Qualified 238 unchanged crops, containing 5,348 timed events,
126 triplet events and 186 rests (166 ordinary, 20 full-measure). Zero missing
payloads, unseen frozen classes or duration conflicts in the accepted subset.
Elapsed targets: 5,348; written and ratio targets: 5,328 each; rest targets: 186.
Full-measure rests intentionally lack written/ratio supervision.

Final report, qualified bundles and local training index:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-unchanged-training-timing-20261006-v2/`.
The earlier sibling without `-v2` is an incomplete failed attempt, not a packet
to consume. The final index references existing local public images and copied
qualified labels; it is not a new published corpus index.

Each accepted crop belongs to the frozen train index and its score is absent
from validation. Its public/replayed token bytes match exactly, its current
pixels equal the historical pixels, and its archived segment has one unique
accidental-aware musical window shared by every original source part. All four
parts must pass timing comparison before any crop from that system is accepted.
Canonical divisions/duration ticks are restored into the timing sidecar;
public token and v6 notation bytes stay unchanged. All 238 bundles were reloaded
and their hashes and training-index count verified. Eleven focused tests pass,
including cached-window equivalence/ambiguity and exclusion of changed/valid
Lieder examples from training support.

This proves label/source-window pairing and historical/public pixel identity.
It does not independently certify historical scan boundaries or the whole OSSQ
corpus. The earlier 12-crop corrected packet is deliberately excluded from these
unchanged-crop counts, preventing overlap/double counting.

478 rejected crops remain recorded:

| Rejection | Crops |
| --- | ---: |
| Source/segment extraction requires explicit timed duration and divisions | 338 |
| No unique accidental-aware all-part source window | 107 |
| Historical/public pixels differ | 13 |
| Public/replayed token bytes differ | 12 |
| Missing divisions value | 8 |

The first inspected duration rejection is an archived note with no duration,
not a detected recognition error. These were not silently filled or dropped.
Counts are per crop; one failed part currently rejects its whole system.

## Combined eligible training support

Combined with the existing 3,409 **unchanged training** Lieder crops; changed
Lieder crops and every validation crop excluded. Both reports use the same
frozen mixed-source tick-quantized target manifest (no class rediscovery).
The two corpora contribute 133,468 timed-event occurrences across 3,647 crops.
Per-head supervision counts differ because source conflicts and full rests
mask some Lieder targets.

| Head | Eligible occurrences | Classes supported / frozen |
| --- | ---: | ---: |
| Elapsed duration | 133,454 | 23 / 51 |
| Written duration | 132,314 | 15 / 21 |
| Tuplet ratio | 132,300 | 6 / 22 |
| Full-measure rest | 10,325 | 2 / 2 |

Exact unsupported class names are in `report.json`. Borodin adds ordinary and
triplet/rest examples, but does not expand Lieder's supported-class set. This
supports a common-class timing experiment, not rare-class readiness or promotion.

Next: recover omitted segment durations/divisions from uniquely paired canonical
source notes, with an explicit restoration receipt and unchanged-token checks;
retain ambiguous windows as rejected. Then wire frozen-core timing-head feature
extraction and resumable training, evaluating primarily on held-out OSSQ/Lieder.
Keep frozen class dimensions stable, report unsupported classes, and assess
recognition/export behavior before any runtime activation. No new GPU job,
publication, deployment, corpus pin or model pin in this work.
