# C1 attachment interface audit — 2026-10-05

Native ground truth is ready: each lyric entry carries source part, physical measure, note ordinal, voice/staff, onset, duration, verse number, text and syllabic segments. The OTS-HOMR service currently exposes detector boxes and OCR/fusion metadata, but no predicted lyric-to-note attachment record. HOMR itself does not yet emit lyrics or attachment in its MusicXML output.

Therefore a text-coverage score can run now, but a verse/attachment accuracy score cannot be honestly computed from the current service output. The existing nearest-x resolver is a baseline for rendered lyric crops, not a score of the deployed detector/OCR path.

Required next interface, kept off by default:

```json
{
  "page_id": "...",
  "detector_box_id": "...",
  "text": "...",
  "verse": "1",
  "source_part": "P1",
  "physical_measure": 4,
  "source_note_ordinal": 12,
  "onset_quarters": "3/2",
  "confidence": 0.0
}
```

The attachment evaluator should compare predicted `(source_part, physical measure, note ordinal/onset, verse)` tuples against native targets, report detector/OCR misses separately from attachment errors, and preserve melisma, multiple-verse, multiple-staff and page-boundary skips. It must not infer a correct attachment from text similarity alone. Native note ordinals and onsets restart within each measure, so the physical measure is mandatory.

No runtime schema or service behavior was changed. This audit prevents the release sequence from treating detector boxes or ordered OCR character coverage as attachment evidence.

The schema-only contract is implemented in `OurTextScores/services/ots-homr-modal/lyric_attachment.py` with fail-closed validation tests. It is intentionally not imported by the scanner engine until a producer and attachment policy are qualified.

The first diagnostic producer is now `lyric_attachment_policy.py`. Its nearest-x baseline requires explicit staff bands and note anchors, refuses unmatched boxes, and preserves duplicates. It is a measurable baseline only; it is not wired into the service.

## Producer geometry audit and evaluator — 2026-10-05

`EncodedSymbol.coordinates` exists and is populated from decoder attention. The staff parser
uses it to draw debugging marks on the transformed staff crop. These coordinates are explicitly
documented as imprecise; they are not already page-level notehead anchors. Export requires a
validated transformation back to page coordinates, preservation through reranking/repair,
and linkage between predicted XML notes and native source ownership. Source ownership cannot
be assumed from a predicted note's index when recognition inserts or deletes notes.

The contract now requires `physical_measure`. The OTS diagnostic evaluator
`lyric_attachment_evaluation.py` accepts independently linked `(page, box)` references and
reports detector misses, unresolved detected boxes, duplicates, literal text correctness,
ownership correctness and joint correctness. It compares rational onsets exactly. Unlinked
predictions are reported without treating partial annotations as false positives. No measured
real-scan attachment accuracy is claimed: certified box-to-note links and a producer remain
necessary. Fourteen focused contract/policy/evaluator tests pass.

The policy's `staff_top`/`staff_bottom` currently mean the eligible **lyric-row assignment
band**, not the five staff lines; callers must supply the below-staff region deliberately.
Multi-verse assignment must be certified separately, including bilingual pages where native
verse order differs from vertical order. Do not infer verse IDs from row height alone.
