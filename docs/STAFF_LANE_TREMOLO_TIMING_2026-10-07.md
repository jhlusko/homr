# Staff-lane timing investigation — 2026-10-07

Latest context experiment: 93 timing edits on four additional selected scores,
zero native-measure regressions; seven cached and one single-note controls unchanged.
See [TREMOLO_PEER_CONTEXT_2026-10-07.md](TREMOLO_PEER_CONTEXT_2026-10-07.md).

Original fixed-policy broader qualification: all 13 pages completed with zero
edits; safe abstention only. Follow-up context gains are linked above. See
[TREMOLO_SCORE_VALIDATION_2026-10-07.md](TREMOLO_SCORE_VALIDATION_2026-10-07.md).

The apparent “duplet” gains in this timing probe are **two-note tremolo timing**.
All 135 validation targets carrying ratio 2:1 belong to tremolo-bearing co-onset
groups: 133 in Schubert Der Zwerg and two in IMSLP557088. Unmarked chord members
inherit their group's ornament association; inspecting only each note's articulation
would miss 65 Schubert targets. This validation set supplies no ordinary-duplet
qualification evidence.

The native source for Der Zwerg measures 70–71 explicitly writes half-note types,
shorter consumed durations, time-modification 2:1 and two-note tremolo start/stop
marks, without tuplet brackets. This agrees with the [MusicXML tremolo specification](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/tremolo/),
which calls for 2:1 time modification on double-note tremolos. Numerical class
accuracy remains valid; our earlier musical interpretation of that class was wrong.

## Comparison on the complete accepted-stack page

Der Zwerg IMSLP16369 p004, certified native source window [64,87), three physical
staves, 477 reference events. Use previously saved accepted v7 export objects;
reproduce the accepted XML structure before proposing changes. References enter
only after both complete XML arms and source-free proposals are frozen.

| Policy | Changed notes / bars | Rhythm errors / 477 | Strict measure/onset event errors / 477 |
|---|---:|---:|---:|
| Accepted export | 0 / 0 | 123 (25.79%) | 155 (32.49%) |
| All timing heads must agree | 0 / 0 | 123 (25.79%) | 155 (32.49%) |
| Ratio + written heads, elapsed advisory, paired tremolo context | 16 / 2 | 107 (22.43%) | 138 (28.93%) |
| Legacy paired-tremolo rule, no timing-head decisions | 103 / 13 | 20 (4.19%) | 34 (7.13%) |

Pitch remains 3/465 errors (0.65%); accidentals remain 1/465 (0.22%). No native
measure regresses in rhythm or strict-onset event edits in the two successful
policies. Every untouched native measure retains identical events/locations.
Global edit distances may align across bars; per-native-measure audits are retained
separately. These are diagnostics on one already-inspected page, not independent
promotion evidence or overall corpus accuracy.

The head policy improves source measures 70 and 71. Their strict-onset errors fall
from 10 to 1 and from 8 to 0 respectively. It changes two targeted notes' XML voice
IDs from 2 to 1 as shortened durations remove overlap. The rule changes seven
such targeted voice IDs. Every untargeted voice assignment, written type/dots,
pitch, staff, tremolo mark, beam, tie and slur is preserved. Existing triplet ratios
are preserved; only selected consumed durations change, by exactly one half.

## Meter, chords, voices and brackets

Correction to earlier status: the decoded **piano** part has an explicit 4/4
signature. The vocal part lacks a declaration, but its other complete bars supply
independent four-quarter witnesses (at least three, at least 75% agreement).
Candidate measure is excluded from those votes; peer and piano predicted bar
counts must agree. No ground-truth meter enters proposal logic.

Count an equal-duration chord once as an onset. Reject mixed-duration chord groups,
rests mixed into multi-member groups, grace groups, changing/missing target meter,
incomplete pairs and lane lengths that do not fit independently supported meter.
Require a tremolo-bearing member at each corrected onset. The rule uses legacy
rhythm/articulation and meter context; timing predictions do not affect it.

After regeneration, require all note intervals to stay within the bar, no overlap
within a rhythmic voice, and a single corrected staff voice spanning the complete
bar. Six provisional head-policy bars reduce to two after these timeline checks.
Changing duration can legitimately remove erroneous overlap, so targeted XML voice
normalization is separately recorded; untargeted voice changes remain prohibited.

An initial generic ratio candidate failed protection: the exporter copied an upper
staff's group-level tuplet bracket to ordinary bass notes. The isolated research
renderer now preserves each untargeted note's original bracket mark. Tremolo mode
suppresses all newly introduced tuplet brackets on targeted notes, retains the 2:1
time modification, and preserves existing tremolo notation. No shipping runtime
file changed. Tremolo stroke count/placement errors already present in the accepted
output remain outside this experiment.

## Local artifacts and reproducibility

Script: `training/onnx/probe_staff_lane_timing.py`.
- Default: require all timing heads to agree.
- `--elapsed-advisory --tremolo-only`: head/context diagnostic.
- `--legacy-tremolo-rule --tremolo-only`: legacy-rule comparator, no timing-head decisions.

Input: trusted local `final-export-input.pkl` under
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-positive-pages-20261007/IMSLP16369/observed/`.
No model inference, GPU, provider change or deployment occurs.

Final head policy:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-tremolo-context-20261007/`.
Final rule:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-tremolo-rule-20261007-v2/`.
Each contains frozen XML, planning/proposal receipts, report and per-native-measure
audit. Earlier candidates and the failed bracket/invariant audit remain archived.
Comparison: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-tremolo-policy-comparison-20261007.json`.
Reference semantic audit: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/timing-tremolo-target-semantic-audit-20261007-v2.json`,
reproducible with `training/onnx/audit_timing_ratio_semantics.py`.

## Next priorities

1. Qualify the **paired-tremolo export rule** on broader OSSQ/Lieder positive and
   negative cases, including single-note tremolos, mixed-duration chords, grace
   notes, overlapping voices, changing meters and ordinary tuplets. Preserve the
   strict default; the current result does not authorize promotion.
2. Build an isolated runtime patch behind a flag only after the rule's scope and
   bracket behavior have appropriate controls. Qualify untouched notation and
   note timing, with ground truth as evaluator, then inspect concrete rendered
   examples before enabling it.
3. Correct timing-head evaluation categories: ordinary duplets need independently
   identified examples, not tremolo-associated 2:1 labels. This result supports an
   export correction more strongly than a new timing-model release.
4. Preserve the separately confirmed five full-rest metadata gains; broader
   qualification is still needed. C1 semantic lyric routing remains separate.

All cached experiments completed. No training or scoring ETA remains.

Validation: sixteen focused tests pass, including chord counting, independent meter,
head disagreement, exact lane fit, voice overlap, bracket isolation, tremolo-only
notation and independence of the legacy rule from missing/adversarial head values.
Scoped formatting and lint pass.
