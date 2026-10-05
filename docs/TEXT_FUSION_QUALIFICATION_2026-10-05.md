# Text-fusion qualification — 2026-10-05

## Scope and conclusion

This continues the authorized local fusion track alongside page-reference certification.
It does not promote a detector, fit a production calibration, inspect the held-out test
partition, or enable the checkbox/fusion flags. Detector-region quality, transcription,
verse ordering and musical attachment remain separate checks.

The current direction route has a more fundamental limitation than component filtering:
its pinned e4 checkpoint was trained with loss on Lyrics and Dynamic only. Its non-lyric
channels do not establish a working direction detector. Today's real-page diagnostic
reproduces that known failure. Calibration alone cannot create a missing capability.

## Immutable detector identities and historical decisions

`OurTextScores/services/ots-homr-modal/pins.py` and `detector_routes.py` currently assign:

| Route | Physical checkpoint | Exact identity |
| --- | --- | --- |
| Lyrics | e2 | `jhlusko/lyrics` @ `1bf2201d2eead54f78d79e2beba502b204614a92`; SHA `7ff21020d5fefe5b774af71dd50610d1b05011f040c4b2c9c58f249b43f39804` |
| Directions | e4 | `jhlusko/non-lyric-text` @ `9073adb53d901c90419f63b0f7e3fb4372ed18cc`; SHA `fcb8ca529fa355b0427112bda1c4edd95b2d6cc6c369be0c32f3313987a4ec82` |
| Dynamics, review-only | same e4 physical pass | Identical checkpoint; Dynamic has one logical owner |

Both hashes exactly match `homr-artifacts/instance-2026-08-30/b0/detector_e{2,4}.pth`.
The preserved e4 launcher is `homr-artifacts/instance-2026-08-30/b0/prep_blank_and_train.sh`,
SHA `3b3f2209a57174fe9fffcd4acb69a10cdfb5491608a4f7e4f9122389ac16fed4`.
Its e4 command specifies `--loss-classes Lyrics,Dynamic`; `stage3_e4.log` confirms
“loss: Dice over Lyrics,Dynamic”. e2 uses the corresponding background-outside-mask
experiment in `run_exp.sh`, SHA `94131a8f7c0123e52c0aa2f1c453b291ada67fa16193b3a4db7149f9fe9859d0`.
These models retain seven legacy channels; this is not the five-class DirectionText model.

The September19 DirectionText adaptation parent (`a03cd631…`) is a different checkpoint.
Its subsequent 60-real-page fine-tune was rejected. The September24 synthetic retrain
and September25 B4 test failure did not authorize replacement of e4. The existing
[real-page gate](DETECTOR_REAL_PAGE_RELEASE_GATE_2026-09-25.md) already records zero
folded direction matches for e4 in B3's selection and test roles. No newer eligible
direction pin was found in the release sequence. None was substituted here.

## Component-area diagnostic

The committed helper `OurTextScores/services/ots-homr-modal/sweep_fusion_components.py`
uses the production tiled probability-mask path and computes each physical mask once
per page. It sweeps connected foreground-pixel areas 4, 10, 25, 50, 100 and 200, using
original page pixels, exact weight hashes, one CPU thread and two physical passes.
The area is actual component foreground area, not bounding-box area or an IoU threshold.

The first two pages are the unchanged October1 calibration-only smoke inputs. Area4
exactly reproduces all three routes' candidate counts despite reducing CPU threads from
two to one. At area200:

| Diagnostic, two pages | Area4 | Area200 |
| --- | ---: | ---: |
| Direction candidates | 1,153 | 84 |
| Dynamic candidates | 499 | 29 |
| Lyric candidates | 36 | 25 |
| Known OSSQ dynamic labels covered at IoU≥0.5 | 20/25 | 20/25 |
| Known Lieder lyric labels covered at IoU≥0.5 | 8/13 | 8/13 |
| Known OSSQ direction labels covered at IoU≥0.5 | 0/2 | 0/2 |
| Verified no-lyrics OSSQ candidates | 6 | 5 |

These are known-positive *coverage* diagnostics (best same-family box per label), not
the release gate's greedy one-to-one recall or calibrated precision. Partial annotations
do not turn unmatched candidates into negatives. Only explicitly complete, empty
semantic families supply negative candidates. No threshold is accepted from two pages.

Both OSSQ direction rectangles were inspected on the original scan: coordinates correctly
enclose “Andante con moto. M.M.” with metronome marking and “tranquillo”. Their only
overlapping admissible direction candidate is a Tempo box `[3,0,1567,2221]`, almost the
entire page, confidence0.86225. The two IoUs are0.00715 and0.00266. Covering all text
with one page-spanning box is not successful detection; this is not an annotation shift.
Raising the minimum component area cannot remove a large erroneous region.

Four further calibration-only pages were selected before their inference using minimum
pixel SHA per corpus/family, labelled originals≤8MP, distinct scores, at least two
direction labels or ten lyric/dynamic labels: `sq7295726`, `IMSLP184205`, `IMSLP154289`,
`sq9010547`. The held-out test index was not used. Their selection policy and results
are retained locally. Both jobs completed EXIT0; their watchers completed. Six pages,
twelve physical inference passes,832.9seconds of measured page processing, one CPU
thread; no fusion compute remains running (ETA0).

| Corpus/route, six-page diagnostic | Area4 candidates | Area200 candidates | Labels covered at IoU≥0.5, both areas |
| --- | ---: | ---: | ---: |
| OSSQ directions | 2,953 | 204 | 0/22 |
| OSSQ dynamics | 1,720 | 98 | 40/72 |
| OSSQ lyrics, verified empty | 18 | 14 | No positives; all these candidates are verified negatives |
| Lieder directions | 3,711 | 493 | 0/2 |
| Lieder dynamics | 2,890 | 133 | 4/4 |
| Lieder lyrics | 101 | 79 | 22/43 |

All six tested areas preserve the observed IoU≥0.5 coverage counts, but a universal
area200 cutoff is not automatically safe: OSSQ dynamic weak overlaps at IoU≥0.1 fall
60→48/72, and at IoU≥0.3 fall49→46/72. Area50 preserves both of those counts while
reducing OSSQ dynamic candidates1,720→428. Lieder lyric coverage at all three diagnostic
IoUs is unchanged up to200. For a subsequent broader calibration-only pilot, test
class-specific area200 for Lyrics and50 for Dynamic, keeping the original area4 control;
do not impose200 on every class or interpret preserved high-IoU counts as preserved
glyph/context recall. These are proposed pilot settings, not frozen production policy.
Directions remain0/24 across both corpora at every tested area.

## Training exclusion and C1 reference audit

The archived e2/e4 patch indexes survive: 29,808 training rows, 3,464 validation rows
for e2, and29,808 training rows for e4. They contain only numerical patch filenames,
without score identifiers. Original source-order indexes in the archived `scan_split_bg`
and `detector_split_v4` directories are absent. Source order is needed to map
`000000_00.png` back to its parent score. Consequently detector-training disjointness
is still unverified; the current score-disjoint calibration/test preparation is not proof
that either partition is independent of detector training. Preserve this distinction.

An audit of121 existing calibration OCR-score-matched annotation files finds7,754
entries with text, including6,243 lyric entries with verse identifiers. None has an
explicit note, measure or beat identity. These OCR-selected labels are useful positive
region evidence, not an independent OCR transcription test or a note-attachment reference.
Do not treat the OCR confidence or matching selection as a C1 accuracy result.

The new `prepare_c1_lyric_references.py` prepares source-owned lyric references without
using OCR or predictions. It resolves exact OpenScore/IMSLP identities and hashes native
MXL/XML, preserves verse, syllabic/elision/extension segments, part, physical and printed
measure, note ordinal, explicit voice/staff, pitch and exact rational onset/duration.
It handles backups, forwards and chords; unsupported source layout/missing divisions
fail rather than guess. All124 calibration Lieder scores resolve uniquely:26,662 lyric
entries, including1,459 extension markings and four verse numbers. Two lyrics attached
to source rests are preserved and flagged. These source-only references do not certify
scan linkage or C1 accuracy. Canonical output is `c1-source-lyrics-20261005/` below.
All source-ownership keys are unique, and onsets are non-negative. The sources omit
explicit staff elements on these lyric notes; `staff` remains null rather than injecting
an inference assignment. Source part and explicit voice ownership are retained.

Next certify scan linkage for these native references. Recover detector
source-order training indexes if available; otherwise label the existing partition's
training independence as unknown. Resolve the direction detector capability/pin plan
before attempting broad three-route fusion calibration. Lyrics/dynamics can be evaluated
separately; dynamics stays review-only.

## Local evidence and verification

Base: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`.

- `text-fusion-sweep-20261005/`: exact input/runtime/model receipt, per-area diagnostics,
  executed source, training-exclusion audit, C1 reference audit and direction-box audit.
- `text-fusion-sweep-expanded-20261005/`: four additional calibration pages and source receipt.
- `c1-source-lyrics-20261005/`:26,662 source-owned targets,124 native source/XML hashes,
  source-catalogue and input-index receipts, executed preparation source; no scan projection.
- `text-fusion-sweep-20261005-launch/`: immutable selected-page index/policy, logs, exit
  statuses and watcher receipts; persistent tmux execution uses one CPU thread.

Five focused tests cover partial unmatched regions, complete empty lyric truth, exclusive
Dynamic ownership, backup/chord onsets, verse/elision/extension ownership, and rejection
of missing divisions or unsupported XML layout. Runtime policies, physical pins,
manifests and deployment are unchanged. Another51 existing fusion/preparation/manifest
tests pass; Ruff/Black pass for all four new files. Code is committed locally in
OurTextScores as `d0c30c4`, not pushed. Exact executed-source snapshots match the run
receipts; later formatting and the explicit unsupported-layout guard are retained in
the committed source without rewriting the completed evidence.
