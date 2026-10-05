# DirectionText artifact recovery and C1 page linkage — 2026-10-05

## DirectionText names, identities and eligibility

The split-system amendment in
[the September25 gate](DETECTOR_REAL_PAGE_RELEASE_GATE_2026-09-25.md) proposed
**B4's calibrated epoch4 for DirectionText**, with the **older released experiment e4
for Dynamic**. “Experiment e4” and “B4 epoch4” are different models. The amendment
did not promote B4 without its direction-only test. That single test subsequently
failed; the current fusion pin still identifies the older seven-class experiment.

| Model | Local identity | Role/status |
| --- | --- | --- |
| Legacy seven-class experiment e4, August30 archive | SHA `fcb8ca529fa355b0427112bda1c4edd95b2d6cc6c369be0c32f3313987a4ec82` | Current `jhlusko/non-lyric-text` pin; selected loss classes Lyrics/Dynamic; not a working DirectionText model |
| Five-class DirectionText B4, September25 epoch4 | SHA `64a26c08b7501e755ab6a0fe8e53a8b1a66303088245000cc3093260fa10667e` | Real supervised DirectionText candidate, recovered locally; direction-only release gate failed |

B4's full weights, class-order sidecar, recipe, calibrated selection report and existing
direction-only decision survive locally. Weight bytes exactly match the archived test
identity. The sidecar is `Dynamic,Fingering,DirectionText,MeasureNumber,Lyrics`;
CPU state inspection confirms six output channels including background. The pinned
provider currently instantiates eight channels for seven legacy classes. Using B4 would
require explicit per-model channel metadata, independent physical identity, routing and
latency qualification. It cannot be substituted by renaming e4 or deleting legacy labels.

Existing direction-only decision: `gate_pass=false`, `direction_only=true`.
The unchanged selection threshold was `DirectionText ≥0.7220876812934877`:

- OSSQ recall **221/421**, above the105-match floor.
- OSSQ predictions **788**, above the367 cap: failed.
- Lieder recall **8/11**, below the11-match floor: failed.
- Lieder predictions55, below the87 cap; synthetic learning floor passed.

This audits the already-materialized September25 reports, not another inference or
test-annotation read. No thresholds were retuned, no test split re-read and no pin
changed. The earlier “no eligible newer direction pin” conclusion is confirmed, while
**a trained DirectionText artifact does exist**. A new detector experiment requires
a new written criterion; existing failed test evidence must not be bypassed by fusion
calibration. The selected September19 adaptation parent is another distinct artifact,
not the current provider pin.

Local base:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.
Recovered B4 files: `detector-b4-full/weights/epoch-4.pth{,.classes.json}` and
`recipe.json`, `detector-b4-calib/epoch-4.json`,
`detector-b4-test-read/decision_direction_only.json`.
Identity/hash receipt: `c1-page-linkage-20261005/direction-artifact-audit.json`.
Historical single-read explanation: `context-20260925/results/b4_test_read_result.md`.

## Bounded C1 source-to-scan page linkage

Two pages were selected from the existing **calibration partition only**, without OCR
text or recognition predictions. Selection uses minimum pixel SHA among57 eligible
originals≤8MP with exact independent page-name correspondence, complete contiguous
boundary-certified systems and at least ten native lyric entries; the second page has
a distinct score and at least two native verse numbers. No text annotation from
`IMSLP154086`, `IMSLP89797` or held-out test pages was inspected or fit.

| Original scan | Native physical measures | Visible systems | Native lyric targets |
| --- | --- | --- | ---: |
| Chaminade, IMSLP154060 p001, “Auprès de ma mie / Beside my Darling” | 1–11 | 3,4,4 measures | 42 |
| Franz, IMSLP89794 p001, Op.1 No.4, “Nun holt mir eine Kanne Wein” | 1–13, including opening pickup | 3,3,4,3 measures | 86 |

The independent source-count alignment is hash-bound, marks
`model_predictions_used=false`, and each selected system is aligned, count-exact and
boundary-certified. Visual original-page review confirms title/edition identity,
vocal-over-piano part order, complete page coverage, bar counts, and bilingual phrase
anchors. Native parts contain equally counted64/64 and38/38 physical measures.
The linker re-parses the exact MXL and requires the stored lyric targets to reproduce
the native source, verifies original bytes/RGB hashes and calibration membership,
checks each visible anchor's source part/note/measure/verse/text, and refuses missing,
ambiguous or gapped windows.

Result: **two pages, seven systems,128 native lyric entries and eight visible anchors**.
Each target carries page/system linkage and exact native note ownership, but explicitly
`note_pixel_attachment_certified=false`. This establishes page/system context;
it is not exhaustive syllable transcription, note pixel alignment, OCR accuracy or
attachment accuracy. No detector/OCR inference was needed, no runtime output changed.

## Concrete verse and transcription controls

**Verse number is not necessarily physical row order.** Chaminade's scan prints English
above French, while its native verse2 is English and verse1 French. At physical measure6,
the same A4 source note owns upper-row English “If” and lower-row French “Si”. Native
source numbering must be mapped explicitly if the product uses top-to-bottom row IDs.
Franz's scan has German above English and native verse1/verse2 in that same order:
the opening pickup owns “Nun” / “Go”. These paired examples protect against silently
assigning every upper row to native verse1.

**Raw native lyric text is not automatically literal scan transcription.** Chaminade's
physical measure8 target stores a leading hyphen/non-breaking-space in `- mong`, while
the visible syllable is “mong”. Preserve the native target and record this distinction;
do not count an OCR exact-match failure from that raw string without a frozen textual
normalization/adjudication policy. Punctuation, elisions, extensions and language/verse
identity should remain explicit rather than silently flattening all text into one line.

Franz also provides a pickup control: source physical measure1 has an eighth-note
pickup, so source onset0 is not evidence that the syllable occupies beat1 of a complete
3/4 bar. Its fourth system starts with two-and-a-half quarters of rests before “Am” /
“The” at source physical measure11. Both are source timing anchors, not estimated pixels.

## Reproducibility, evidence and next steps

Committed local tool:
`OurTextScores/services/ots-homr-modal/link_c1_lyric_pages.py`.
The new tests check zero-based/exclusive window-to-physical-measure boundaries,
missing/ambiguous/gapped systems and refusal of test-partition pages before image or
native-source access. Together with source-preparation tests, **six focused tests pass**;
Ruff/Black pass. No environment rebuild or package installation was needed.

Evidence under the local base above:

- `c1-page-linkage-20261005/selection.json`, `selection-policy.json`, `visual-review.json`.
- `c1-page-linkage-20261005/linked/{pages.json,targets.jsonl,receipt.json}` and exact
  executed source matching the receipt's hash.
- Original Chaminade scan:
  `/home/jhlusko/workspace/homr-artifacts/lieder-v8/_build/pages/IMSLP154060/IMSLP154060-p001.png`.
- Original Franz scan:
  `/home/jhlusko/workspace/homr-artifacts/datasets/lieder_vocal_text/pages/IMSLP89794/IMSLP89794-p001.png`.

Next use these windows to prepare separately adjudicated syllable/row transcription and
note-attachment references, preserving their calibration role and explicit native/printed
verse mapping. Do not infer note pixel labels from a boundary-certified measure count.
The training-membership and partially labelled-region limitations from
[the preceding fusion report](TEXT_FUSION_QUALIFICATION_2026-10-05.md) remain.
DirectionText requires its own eligible checkpoint/criterion before three-route activation;
lyric reference work can continue independently. No compute remains running, ETA0;
no model promotion, training, upload, pin/flag change, push or deployment.
