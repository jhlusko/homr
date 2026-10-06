# OSSQ crop-pixel review and source context

Review: http://localhost:8765/ossq-pixel-review-20261006/

All 150 validation and 13 Borodin training pixel differences are displayed,
grouped into 67 quartet systems. Each comparison names the part and dimensions;
clicking a crop opens full resolution. Expanders show the archived quartet
system, original scan page and canonical source rendered with MuseScore.
All rendered PDF pages are included; renderer system breaks are not scan
boundary evidence. No individual accept/reject decisions are required.

The question is whether the changed crop loses music or changes staff ownership.
Scale and boundary shifts alone are not sufficient to reject it. Nine inspected
comparisons (large dimension shifts, same-size differences, Borodin examples)
show the same passages; existing clipping occurs in both builds. This sample
does not certify all 163 crops or resolve their original source ownership.

## Source-window resolution

Original source measure numbers restart across movements. The initial numeric
resolver rendered only nine windows and rejected 58. A musical-content resolver
checks visible note type, pitch step/octave/alteration, dots, chord/rest and grace
status in every part and requires one shared contiguous physical window. Sixty
windows match uniquely. Four exact repeated passages resolve using neighbouring
symbolic systems; their paths/hashes are retained. This yields 64 source previews
with zero rendering failures. Source window matching is source-only provenance,
not independent scan adjudication or exhaustive notation/timing fidelity.

Three systems (six changed crops in this viewer) remain unresolved because their
archived segments disagree with the original source on pitch alterations:

| Scan system | Source discrepancy | Affected parts |
| --- | --- | --- |
| `sq10414906:0008:0001` | Original B-flat; segment B-natural | Violin I and viola |
| `sq10675759:0027:0003` | Original G-sharp; segment G-natural | Cello |
| `sq7354505:0047:0001` | Original C-natural; segment C-sharp | Violin II |

These are source-versus-segment discrepancies, not evidence that the pixel
changes caused an error. Some notes cross ties/system boundaries. Neither
representation has been declared correct against the scan. The viewer flags
these systems explicitly and supplies crop/scan context without attaching a
misleading supposedly matching source render. Keep them outside qualified
timing-label replay until adjudicated.

The earlier Borodin whole-score signature check did not include pitch alteration.
The 716-label replay checked consistency with published segment-derived labels,
not complete accidental agreement with original whole-score MusicXML. Therefore
those earlier passing checks do not override this stricter discrepancy.

## Evidence and verification

Final viewer and frozen asset manifest:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-pixel-review-20261006-v3/`

Source diagnostics and representative comparison image:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-mapping-recovery-20261005/`

Includes `source-accidental-discrepancies.json`, `representative-pixel-comparisons.png`
and all three build logs. Earlier v1/v2 viewers preserve the rejected/ambiguous
source context checks. `manifest.json` records source and segment hashes,
physical windows, neighbour evidence, asset hashes and unresolved cases. The
final HTML hash and source-diagnostics hash are bound there as well.

Generator: `python -m training.onnx.build_ossq_pixel_review --recovery ...
--archive ... --out ... --musescore ... --workers 4`. Refuses existing output
directories and changed crop evidence. Four focused tests pass: physical index
versus number, ambiguous numbering, accidental-aware source matching and exact
neighbour disambiguation. Scoped lint/diff checks pass. Every one of 518 linked
images returned HTTP 200 with nonempty content. Build jobs have finished; the
local server remains in persistent tmux `tuplet-review-server`.

Next obtain an overall crop review, adjudicate the three accidental disagreements
against scans and complete source/scan ownership qualification. No recognition
rerun, training, public corpus/model pin or runtime change was made.

## Review accepted and accidental values adjudicated

User accepted the overall pixel comparisons: “They are nearly identical and I
can't imagine the extra pixel will make a difference.” Bound receipt
`user-pixel-review-receipt.json` covers all 163 crop differences, not a user
verdict on source accidentals or exhaustive annotation accuracy. The pixel
difference gate is closed; no additional per-crop review is required.

Codex subsequently inspected high-resolution original scans and preceding
systems/pages. The scans support the original source in all three discrepancies:

- Haydn: the one-flat key signature supplies B-flat for the violin/viola notes;
  no natural cancels it. Three disputed notes.
- Borodin: cello G-sharp continues the preceding system's tied G-sharp. One note.
- d’Indy: violin II explicitly marks C-natural at the new system and ties it to
  the following eighth note. Two notes.

Original and cleaned source preserve these values. This locates the discrepancy
after cleaning, in the symbolic segmentation/round trip rather than crop pixels.
It does not yet isolate the precise faulty stage. Haydn's published crop tokens
also lack the scanned one-flat key; the two other cases involve tie/accidental
context across boundaries. Corrected local labels must preserve that context,
not blindly edit pitch strings without updating related representations.

`source-accidental-adjudication-receipt.json` in the recovery directory binds
original/cleaned source hashes and the scan-context images. Reviewer is explicitly
Codex; the user's approval is recorded separately. `accidental-scan-context/`
contains five high-resolution system views, including Borodin's preceding system
and d’Indy's preceding page. The final viewer displays the accepted pixel review
and source follow-up, with its updated HTML hash in `review-followup-receipt.json`;
the prior HTML is preserved. Six note values are adjudicated, not every note in
the corpus. Next prepare a bounded local corrected replay of those systems,
verify key/tie context and sidecar alignment, then complete timing-label support.
Public corpus/model/provider pins remain unchanged.
