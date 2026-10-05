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
  "source_note_ordinal": 12,
  "onset_quarters": "3/2",
  "confidence": 0.0
}
```

The attachment evaluator should compare predicted `(source_part, note ordinal/onset, verse)` tuples against native targets, report detector/OCR misses separately from attachment errors, and preserve melisma, multiple-verse, multiple-staff and page-boundary skips. It must not infer a correct attachment from text similarity alone.

No runtime schema or service behavior was changed. This audit prevents the release sequence from treating detector boxes or ordered OCR character coverage as attachment evidence.

The schema-only contract is implemented in `OurTextScores/services/ots-homr-modal/lyric_attachment.py` with fail-closed validation tests. It is intentionally not imported by the scanner engine until a producer and attachment policy are qualified.

The first diagnostic producer is now `lyric_attachment_policy.py`. Its nearest-x baseline requires explicit staff bands and note anchors, refuses unmatched boxes, and preserves duplicates. It is a measurable baseline only; it is not wired into the service.
