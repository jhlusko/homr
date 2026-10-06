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
