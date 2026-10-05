# Slur endpoint ownership audit — 2026-10-05

All eight cached page/model arms reproduce their saved policy XML exactly. All 155 unspecified slur-start elements (66 accepted, 89 joint) originate in core endpoints without a compatible structured start slot. Known sides attached to event NONE or STOP cannot safely supply a start side.

The legacy core slur field combines tied-note and phrase-slur endpoints; structured heads distinguish them. Source categories for these unspecified starts: 75 tie-only, 19 phrase-only, 7 simultaneous tie and phrase, 45 neither source start, and 9 unmatched/ambiguous anchors. These are source-XML comparisons, not scan-adjudicated hallucination counts. Joint Glazunov's eleven conditional unknown starts comprise five phrase-only and six simultaneous tie/phrase starts. Blanket deletion of tie-associated endpoints would lose legitimate phrases; borrowing every non-start slot's side could attach a side to the wrong event.

Unspecified placement permits automatic engraving; it does not itself prove a visually incorrect slur. No runtime fallback or deletion was introduced. Next: audit independent tie and phrase span ownership, including simultaneous events and boundary-spanning controls, before proposing an exporter change.

Local evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/slur-placement-provenance-20261005-v2/` (`report.json`, `source-categories.json`, executed sources).
