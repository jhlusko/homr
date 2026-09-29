# Lyrics, direction text and dynamics plan — 2026-09-28

The owner agreed to a region-detection/fusion architecture with specialized dynamics
recognition. This document separates the agreed target from currently implemented code.

## Agreed architecture

| Logical component | Responsibility | Starting model / status |
|---|---|---|
| Music recognition core and notation heads | Notes, rhythm and structural notation | Public core plus joint heads under evaluation |
| Lyrics detector | Locate lyric regions when the user requests lyrics | Released e2 / calibrated lyrics-only-v1 route |
| Direction detector | Locate non-lyric written directions | Released e4; collapsed DirectionText successor not promoted |
| Dynamics detector | Locate musical dynamic glyph regions | Reuse e4's Dynamic channel initially; separate logical identity |
| Fusion | Class allowlists, calibrated overlap handling, provenance and ambiguity handling | Local fusion-v1 box-review implementation; disabled |

Three logical detector routes do not necessarily mean three physical model passes:
while direction text and dynamics both use e4, reuse one inference result and route its
classes independently. A separate checkpoint/pass is needed when either route adopts a
specialized model. Measure its full-page incremental cost before promotion.

Tempo, StaffText and Expression share the DirectionText semantic family. The collapsed
training experiment already exists, but its candidate failed the release gate. Current
e4 still exposes its legacy classes. Never change training channel order independently of
runtime CLASS_ORDER; checkpoint metadata and calibration must identify the exact order.

## Recognition and musical attachment

Detection answers where a region is. It does not establish what it says or where it belongs
in the score. These subsequent stages have distinct quality requirements:

- Lyrics: OCR the syllables/text, retain verse identity, and associate syllables with the
  correct staff, notes and musical positions. The checkbox remains gated on C1 read/attach
  quality and C2 final-path latency.
- Direction text: read the words, preserve their text, and place them at the appropriate
  staff/measure/beat. Semantic interpretation of tempo/expression is a later decision.
- Dynamics: recognize p/pp/mp/mf/f/ff etc. with a music-glyph-specific classifier or
  recognizer rather than assuming generic prose OCR is sufficient. Associate the glyph
  with the correct staff and musical position, then emit the appropriate MusicXML marking.
  Written directions such as crescendo/diminuendo remain DirectionText; graphical hairpins
  require a separate notation/geometry path.

Dynamics initially remain review-only. The joint model's dynamic.mark head is experimental,
not an automatic output authority. It remains prohibited in the current service policy.
Do not route its present approximately 10–11% macro-F1 output into released MusicXML.
Training inclusion is separate from inference permission; audit its flat behavior before
choosing its role in the next training objective.

## Required wiring and gates

1. Add the dynamics detector identity to pins, fusion class allowlists, calibration metadata,
   provenance and tests. Route only Dynamic from this identity; prevent duplicate candidates
   when e4 supplies both direction and dynamics. Retain lyrics-only filtering for e2.
2. Keep thresholds/calibrated confidence specific to checkpoint and class. Raw pixel
   confidences from different detectors cannot decide cross-model conflicts.
3. Fail closed for explicitly requested fusion if a required model/calibration is missing.
   Ambiguous or uncalibrated candidates remain review items.
4. Define source-disjoint gates for region recall/false positives, recognized glyph/text,
   and correct note/staff/verse attachment. Dynamics strength confusion matters separately
   from box localization. Do not collapse these into a single score.
5. Measure production-path latency including core decoding, detector passes, OCR/recognition,
   fusion, attachment and cold/HTTP overhead within the 280-second per-page timeout.
6. Promote lyrics/directions/dynamics independently after their gates pass; deployment and
   production pin changes remain separate decisions.

## Current implementation and evidence

`OurTextScores/services/ots-homr-modal/fusion_policy.py` currently accepts only the lyrics
and non-lyric-text detector identities. Dynamic candidates are review-only, and
`dynamic.mark` is prohibited. The third identity/specialized glyph reader is pending.
`fusion_v1.py` explicitly produces box decisions, not text attachment to MusicXML.

The fixed warm paired latency run passed 22/22 pages: median 114.604 s, conservative p95
195.091 s, maximum 249.589 s. This leaves only 30.411 s on its slowest page for unmeasured
final-path overhead, so that result alone does not authorize release.

The master's decision 13 (2026-09-25) already approved separate dynamics ownership. B4's
later test failure means e4 remains released. This plan preserves that result rather than
assuming the DirectionText candidate was promoted.
