# Real-source cached pitch-dedup ablation — 2026-10-04

Preserving simultaneous same-pitch events improves aggregate recognition on both OSSQ and Lieder for both retained stacks. It also preserves some unwanted repetitions. This is evidence for a bounded cleanup correction, not approval to disable deduplication globally or promote a model.

The audit pairs all4,211 raw crop predictions:3,911 OSSQ crops from9 scores and300 Lieder crops from13 scores, accepted September18 core/September24 v7 heads versus public-initialized joint epoch27. Exact manifest coverage, paired pixel/reference/sidecar hashes, and all33,688 stored recognition scores pass. Reference labels remain unchanged. These are existing model-selected validation labels, not an untouched test.

Each prediction is restored independently for raw, current standard cleanup, and standard cleanup with only pitch deduplication disabled. Other cleanup steps stay enabled. Metrics use the original scorer's chord ordering, reference denominators and edit distance. The runtime defaults and original reports are unchanged.

|Source|Stack|Symbol errors, standard → preserve|Pitch errors, standard → preserve|Rhythm errors, standard → preserve|Accidental errors, standard → preserve|
|---|---|---:|---:|---:|---:|
|OSSQ|Accepted|4,185 →4,174|287 →274|3,541 →3,529|635 →622|
|OSSQ|Joint27|2,773 →2,758|229 →212|2,270 →2,255|459 →442|
|Lieder|Accepted|380 →340|193 →148|231 →189|163 →119|
|Lieder|Joint27|327 →287|192 →144|205 →166|149 →102|

Symbol/rhythm denominators: OSSQ102,219, Lieder11,255. Pitch/accidental denominators: OSSQ83,776, Lieder10,239. Symbol-error improvements are0.011/0.015 percentage points on OSSQ and0.355/0.355 on Lieder (accepted/joint27). Do not confuse these cleanup deltas with the earlier model-versus-model raw comparison.

Pitch deduplication changes15 accepted and10 joint OSSQ crops, and26 accepted and23 joint Lieder crops. Symbol-error outcomes across those74 crop/model pairs:60 improved,7 worsened,7 unchanged. Joint27 has no OSSQ regression in these metrics; it has one Lieder symbol/rhythm regression. Accepted has4 OSSQ and2 Lieder symbol regressions. Changed-crop details preserve both cleaned streams and reference labels for inspection.

The discarded collisions include identical legacy note fields and same-pitch events with differing duration or accidental. Some preserved events are rests or malformed note/rest combinations, rather than legitimate unison notes. Harmful examples include accepted repeated notes on sq7354505 and sq8075304; repeated E4 in Lieder IMSLP61117; and joint27 IMSLP44341 with repeated rests and a rest carrying C4/sharp fields. These classifications describe predicted token collisions, not independent source-XML voice verification.

## Important control: other cleanup remains harmful here

Raw → standard → preserve symbol errors are OSSQ accepted4,067 →4,185 →4,174; OSSQ joint27 2,580 →2,773 →2,758; Lieder accepted333 →380 →340; Lieder joint27 286 →327 →287. Preserving unisons does not recover all errors introduced by other cleanup. The audit runs cleanup on standalone crops; production runs it on concatenated physical-voice streams. Missing crop context can affect tuplet and lower-staff cleanup, so these numbers are not full-page production scores or evidence that all cleanup should be removed.

Next isolate the remaining cleanup stages on these same caches, then verify the useful change on complete voice/page traces, including valid unisons, different durations/accidentals, excess notes/rests and cross-staff cases. Preserve XML voice qualification as a separate gate. The existing wrong dictionary key in the longest-duration branch also warrants a controlled test; it is not silently fixed by this audit.

## Local evidence and reproducibility

Report and all changed cases:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/cached-pitch-dedup-20261004/`

Inputs:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-qualification-20261001/`

Tool: `training/transformer/audit_cached_pitch_dedup.py --run <input-root> --out <new-output-directory>`. Use the existing HOMR environment with the audit-python overlay and OMP/OPENBLAS/MKL threads1; no model load, downloads or recognition rerun. The report hashes both cache files, both identity receipts, manifest and relevant source. The exact executed tool is retained in `executed-source/`; the final tool only splits a long diagnostic string for linting. `collision-audit.json` is a supplementary classification of changed predicted chords.

Completed locally on one CPU thread. Seven focused tests pass; Ruff and Black pass. ETA0, no job remains running. No inference-default change, promotion, push, upload or deployment.
