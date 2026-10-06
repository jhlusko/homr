# Vocal part and complete-page context diagnostic — 2026-10-05

Runtime follow-up now retains the final piano grandstaff and exports the two-bar
vocal introduction. Both parts reach measure8; all28 vocal event identities are
preserved with corrected page numbering, and69 piano notes are restored.
The same32/42 lyric targets now score directly against runtime XML, without the
diagnostic offset or extra routing-only staff boundaries. See
`docs/C1_RUNTIME_INTRODUCTION_2026-10-05.md` for the paired regression checks.
The diagnostic results below remain preserved with their original scope.

Chaminade's complete page now recovers **32/42 exact native lyric targets**, versus
zero under the first-voice assumption. Across the four follow-up pages, the same
corrected reference coverage gives **50/145 → 82/145 (56.6%)**; bounded row
recovery adds one more target, **83/145 (57.2%)**. The other three pages' baseline
attachments are exactly unchanged. All 34 new Chaminade attachments have exact
native lyric-note identities; 32 have literal text matches with evaluation-only
verse mapping. This is post-validation calibration, not a new held-out result.

| Page | Old context | Complete context | With row recovery | Reference targets |
| --- | ---: | ---: | ---: | ---: |
| Chaminade, Les rêves | 0 | 32 | 32 | 42 |
| Fauré, Après un rêve | 5 | 5 | 6 | 48 |
| Warlock, Lusty Juventus | 25 | 25 | 25 | 29 |
| Hardelot, In 1822 | 20 | 20 | 20 | 26 |

## Three separate causes and controls

**Part identity:** the piano is first in the runtime's voice ordering on Chaminade,
and the vocal part is second. The opt-in diagnostic identifies a text-supported
five-line voice, refusing grandstaffs. If multiple candidates exist, exactly one
must have row support across two or more prepared staffs; competing recurring
candidates remain unresolved. This is structural evidence, not certified semantic
lyric classification. No reference pitch, text, verse or voice enters selection.

**Introduction context:** all traced pitched notes in the first two measures of
the piano part belong to a grand staff wholly above the first vocal staff. That
contiguous prefix supplies the vocal part's page-relative +2 measure context.
Missing provenance, gaps or contradictory prefix evidence fail closed. This
offset is inferred from cached geometry and XML provenance, not searched against
the native score. Only diagnostic event metadata receives the offset: original
MusicXML is unchanged and replay remains exact.

**Missing physical boundary:** segmentation detects eight physical staffs, but
the decode cache retains six. Its log explicitly says it removes the last staff
to fit a repeating two-staff layout; that last prepared grand staff comprises the
two missing physical piano staffs. With no lower boundary, the last vocal staff's
lyric rows are unbounded and previously unresolved. A segmentation-only pass
retains the two staffs as routing boundaries. They receive negative diagnostic
IDs, no decoded voice and no invented note ownership. Every existing physical
staff must reproduce uniquely before missing boundaries can be added. This
restores 11 additional literal targets beyond the 21 obtained by part/prefix
context alone, without another core decode.

Native coverage was corrected separately: the scan has a two-measure piano-only
introduction, a three-measure first vocal system and a three-measure final vocal
system, covering physical measures1–8. The old window ended at5 and omitted12
lyric targets. Scan/native inspection and phrase anchors establish42 targets;
the review was not blinded to model diagnostics. Inference page metadata is
unchanged. Do not compare 50/133 directly to82/145 without disclosing this correction.

## Regression and limitations

All eight previous calibration pages retain exactly the same prediction and
unresolved-syllable lists with the refined selector. The initial unique-only
selector abstained on Tanzlied im Mai: OCR reads one piano pedal-release star as
`米`. Recurring vocal-row support resolves that lone competing region without
word/language filtering. The initial abstention report is preserved. Two voices
with recurring text support still fail closed; directions within lyric rows remain
an explicit semantic limitation (`a tempo` control).

The runtime has not been fixed: its original XML still lacks the dropped final
piano staff and the vocal introduction measure context. Diagnostic metadata repair
does not recover those undecoded piano notes or authorize attaching corrected
measure labels to the unchanged XML. Next repair system grouping for a leading
piano-only system, preserving all detected staffs and correct voice identity;
then preserve absent-vocal measure context in exported parts. Qualify that change
against paired real Lieder/OSSQ page outputs and these introduction controls before
runtime lyric integration. Keep C1 semantic/verse gates and fusion disabled.

## Local evidence

All main evidence is under:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-fixed-rule-validation-20261005/`.

- `complete-context-comparison.json`: equal-reference-coverage comparison.
- `score-complete-context-v3/` and `score-complete-context-recovered-v3/`: canonical
  source-free diagnostic predictions, provenance and native metrics.
- `score-baseline-complete-reference/`: unchanged producer at corrected denominator.
- `chaminade-complete-page-certificate.json` and
  `inputs/evaluation-pages-complete-context.json`: reference correction evidence.
- `raw-physical-staffs/`: segmentation-only captures, 44.0 seconds, one thread;
  completion receipt and source/model hashes.
- `decode/page-0.log`: explicit periodic-layout trimming, unchanged decode cache.
- `complete-context-source-snapshot/`: diagnostic executed source.

Regression audit:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-vocal-context-regression-audit-v2-20261005/report.json`.
Earlier partial/unique-only reports remain preserved, including the failed batch
and per-page abstention audit. No earlier validation receipt is relabelled held-out.

Twenty-two focused tests and scoped lint pass. Tests cover voice ambiguity,
recurring support, traced introduction counts, missing provenance, gapped/non-piano
prefixes, physical-boundary correspondence, routing-only ownership, literal native
scoring and exporter identity. All jobs finished, ETA0; no runtime/pin/deploy change.
