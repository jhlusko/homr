# Expanded native lyric attachment diagnostic — 2026-10-05

Latest canonical result is v6: **311/480 (64.8%)**, with evaluation-only verse
mapping. The v4 baseline and v5 follow-up below remain historical evidence.
See the chord-localization follow-up at the end for remaining limitations.
The subsequent frozen error audit partitions the 169 misses into 100 absent
attachments at decoded identities, 60 text differences, five absent decoded
identities and four ambiguous native targets. See
`docs/C1_NATIVE_LYRIC_ERROR_AUDIT_2026-10-05.md` for caveats and priorities.

The eight expanded calibration pages recover 300 of 480 native lyric targets
(62.5%) with literal text at the exact exported note's measure, onset, written pitch,
alteration and duration, under an evaluation-only page-global verse permutation.
This is a genuine native-output attachment diagnostic, not the earlier ordered
character proxy. It is still conditional on the first runtime voice being the
known vocal part, and it does not qualify production C1 or fusion.

| Score | Native targets | Predicted attachments | At native lyric-note identity | Exact joint recovery |
| --- | ---: | ---: | ---: | ---: |
| IMSLP89287 — Jagdlied | 81 | 55 | 52 | 44 (54.3%) |
| IMSLP89290 — Schlummerlied | 90 | 55 | 55 | 49 (54.4%) |
| IMSLP617260 — In Welschland | 68 | 60 | 55 | 52 (76.5%) |
| IMSLP89796 — Tanzlied im Mai | 88 | 72 | 72 | 62 (70.5%) |
| IMSLP435041 — It was a Lover and his Lass | 33 | 33 | 33 | 31 (93.9%) |
| IMSLP286659 — He doesn't love me | 49 | 42 | 41 | 40 (81.6%) |
| IMSLP83318 — Rose d'Amour | 38 | 34 | 30 | 15 (39.5%) |
| IMSLP83314 — À une Fleur | 33 | 19 | 19 | 7 (21.2%) |
| Total | 480 | 370 | 357 | 300 (62.5%) |

The displayed song titles are source context, not classifier inputs. Source hashes
and previously aligned physical measure windows were checked. The continuation
pages retain their nonzero native measure offsets; the scorer never shifts notes
to a nearby reference to repair an onset, measure or accidental error.

## What was produced and what was evaluated

Inference receives only eight calibration scan IDs, paths and image hashes. Native
sources, target text and measure windows live in a separate evaluation manifest.
Pinned candidate core/head graphs are the same public-joint epoch-27 graphs used
by the previous pilot. This diagnostic policy keeps stem/slur-side arbitration off,
so it is not a complete final release-stack comparison or production latency test.

The producer reuses frozen area-200/10%-overlap lyric selection and positioned OCR.
It routes physical rows, applies same-row seed recovery, interpolates notehead
staff positions, and uses mutual-nearest/pitch/sequence guards. For each syllable it
first selects the nearest head among *all* heads. If that head has no accepted
decoded-note mapping, the syllable remains unresolved; it cannot jump to the next
available recognized note. Native verse stays null in predictions.

The evaluator chooses one injective physical-row-to-native-verse mapping per page,
using ground truth only after inference. It cannot choose a new mapping per note or
system. Literal joint recovery ignoring verse gives the same 300 targets here.
This supports the grouping diagnostic but does not measure literal native verse-ID
accuracy. Repeated predictions are capped at native target multiplicity. Two native
note keys in Schlummerlied represent distinct source notes with identical timing,
pitch and duration: their four lyric targets are retained in the denominator but
excluded from matched ownership claims as ambiguous.

There are 134 unresolved positioned syllable candidates: 66 physical-row failures,
34 unsupported/nonvocal-staff cases, and 34 nearest heads without an accepted
decoded note. These include non-lyric candidate text; they are not 134 certified
missed lyrics. Likewise, the 13 emitted predictions without an exact native
lyric-note identity mix recognition, timing and attachment errors; they are not
independently certified detector false positives. Independent OCR-box ownership,
detector misses and native verse-ID accuracy remain null, rather than invented.

## Export provenance correction

All eight scans decoded successfully in a persistent one-thread tmux session,
433.9 seconds (7 minutes 14 seconds). A second persistent watcher scored them
automatically when the completion receipt appeared.

The initial scorer refused both Bizet pages because cached final-note order/duration
does not equal exported XML order/duration. This was an evaluator provenance limit:
Rose d'Amour exports six sixteenths as triplet durations (1/4 to 1/6 quarter units),
and À une Fleur reorders two chord pairs. It was not evidence of a missing model
decode or a reason to alter the score.

The cached exporter replay restores the page's recorded post-decode policy, tags
each restored token outside musical identity, and traces actual emitted XML note
objects. It must reproduce the original entire XML exactly before returning links.
All eight replay identically. No note is matched using native pitch or text, and
no model decode is repeated. This restores Bizet export linkage; vocal chord
ownership remains explicitly unsupported in two À une Fleur systems. Equal-x
head ambiguity blocks one opening system on each of Jagdlied and Schlummerlied.
These four blocked systems remain in the evidence, not silently removed.

## Local evidence and next priorities

Canonical result:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-attachment-score-v4-20261005/report.json`.
The same directory has per-score predictions, unresolved candidates, system
diagnostics and exporter traces. Earlier partial/v1/v2/v3 reports are preserved.

Inputs:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-attachment-inputs-20261005/`
(`inference-pages.json`, `evaluation-pages.json`, hash receipt).
Decode/cache/logs/source snapshot:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-attachment-decode-20261005/`.
Watcher receipt:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-attachment-score-20261005-watch.json`.
The watcher completed the original score; the canonical v4 adds traced export
provenance, native-anchor ambiguity handling and a separate verse-free diagnostic.

Thirty focused tests pass across geometry, attachment scoring and exporter tracing;
scoped lint passes. Tests enforce duplicate caps, fixed page-global verse mapping,
timing/accidental errors remaining errors, ambiguous native anchors staying
unscored, chord-sort token identity, and refusal of a nonidentical exporter replay.

Next prioritize localized extra/equal-x-head ambiguity handling in the two dense
Franz openings, then the two Bizet vocal chord systems. Preserve uncertain notes
instead of dropping entire otherwise usable systems where safe. Review remaining
literal-text/nearest-note discrepancies against these native targets before any
detector/model changes. Keep semantic controls such as `a tempo` and native verse
grouping explicit. All compute and watcher jobs finished, ETA 0; fusion stays off.
## Localized equal-x ambiguity follow-up

Canonical v5 preserves every v4 attachment and increases exact joint recovery
to **310/480 (64.6%)**: Jagdlied 46/81 and Schlummerlied 57/90; the other six
pages are unchanged. It emits 389 attachments, 374 on exact native lyric-note
identities. The 19 additions include 10 exact literal targets, 17 native-note
matches and two without native lyric-note identity. This is diagnostic evidence,
not a promotion gate or a detector false-positive measurement.

The two Franz opening systems now retain reliable later matches. All permutations
of each exact-x head group are checked, and only forced sequence pairs common to
every ordering survive. Tied heads themselves remain unresolved and remain in
the full nearest-syllable competition; no syllable jumps past a removed head.
Enumeration fails closed above 64 orders. No reference pitch/text enters matching.
The two decoded Bizet vocal-chord systems remain unsupported. Evaluation-only
verse mapping and all earlier native identity limitations still apply.

Local result:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-attachment-score-v5-20261005/report.json`.
The v4 result below is preserved. Next inspect and localize Bizet chord uncertainty,
then review remaining native-referenced attachment/text discrepancies. All jobs
finished, ETA 0; runtime fusion remains disabled.

## Localized decoded chord uncertainty

Canonical v6 intersects forced sequence matches across all within-chord decoded
orders and all exact-x head orders, capped at 64 combined permutations. Chord
members themselves are never accepted. All heads and decoded attention points
remain in geometric competition; there is no retry after removing uncertain
notes, and native labels never enter prediction.

Every v5 attachment survives. À une Fleur rises from 7/33 to 8/33 exact targets:
three additional attachments all match exact native lyric-note identities, but
only one has correct literal text. The other seven pages are unchanged. Total
recovery is **311/480 (64.8%)**, 392 attachments, 377 exact native lyric-note
matches. Native verse IDs remain unpredicted; the reported recovery uses the same
evaluation-only page-global mapping. This is calibration, not a promotion gate.

Its second vocal system retains nine of eleven note associations, with two chord
notes unresolved. The fourth retains none of thirteen: ten decoded chord notes
are explicitly withheld and the remaining three are not invariant/geometrically
accepted. Thus whole-system rejection is replaced by local uncertainty, but the
chord-dense system remains unrecovered. Next audit literal text and unresolved
nearest-head cases against native targets; do not assume more permissive chord
matching or another detector will fix them.

Local report:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-expanded-attachment-score-v6-20261005/report.json`.
Twenty-one focused association/native-scoring/export-trace tests pass, including
chord order reversal, tied heads, repeated-note ambiguity and enumeration limits.
All jobs finished, ETA 0; runtime fusion remains disabled.
