# Bounded replay of scan-adjudicated OSSQ accidentals

The three previously unresolved quartet systems now have isolated corrected
local labels. No public corpus, runtime or model pin changed.

Replay covers all four parts of Haydn `sq10414906:0008:0001`, Borodin
`sq10675759:0027:0003` and d’Indy `sq7354505:0047:0001`: 12 crops, 209 annotated
symbols. It changes exactly six accidental fields and inserts the missing
one-flat key signature in the four Haydn crop labels. Rhythm, pitch position,
event count, published articulation/slur fields and note positions remain
unchanged. All published v6 notation sidecars are copied byte-identically;
stems, ties, beams, dynamics and other head targets are preserved. All 12 labels
load, attach their sidecars and encode into decoder branches successfully.

Inputs are hash-bound original/cleaned source and scan-context evidence from
the prior adjudication. Source segments are cloned and changed only at reviewed
pitch alterations and Haydn's initial key. The writer's before/after differential
is applied to published tokens, preserving their other fields. Baseline timed
tokens must match the public labels; rhythm/pitch/event-count changes are refused.
Chord alterations are refused pending a separate serialization-order audit.
The final run's original inputs still match their recorded hashes.

After correcting the six alterations, every part of each system matches a
unique original-source musical window including accidental values:

| System | Physical source measures | Split |
| --- | --- | --- |
| Haydn `0008:0001` | 78–81 | Validation |
| Borodin `0027:0003` | 902–912 | Training |
| d’Indy `0047:0001` | 1028–1034 | Validation |

Physical counts span movement boundaries; printed measure numbers can restart.
There are 88 annotated training symbols and 121 validation symbols. Validation
labels remain validation controls; they are not added to training. This is a
bounded accidental/key repair, not a complete corpus audit or timing-label policy
migration. The original source/crop pixel review scope remains unchanged.

## Evidence and reproduction

Final local replay:

`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-accidental-replay-20261006-v3/`

`receipt.json` binds public and corrected label hashes and sidecar checks.
`source-window-receipt.json` records unique physical windows, split support and
unchanged public inputs. Corrected MusicXML, final labels, byte-identical notation
sidecars and separate parser baseline/patched diagnostics are preserved. Earlier
partial run directories remain diagnostic only; v3 is the completed result.

Reproduce with `python -m training.onnx.replay_ossq_accidental_repairs`, supplying
`--recovery`, `--review`, `--public-valid` and a new `--out`. The existing pinned
validation and Borodin datasets supply public labels/sidecars. The tool refuses
existing output and changed adjudication evidence; partial runs without the
completion receipt must not be consumed.

Two new differential-repair tests plus four source-window tests pass. They cover
preserving published slur/articulation/position fields and refusing changed
rhythm, pitch, event count or baseline labels. Scoped lint and diff checks pass.
No GPU or inference rerun was required. All replay jobs finished.

Next fold the verified source windows and this explicit overlay into the local
timing-label qualification packet, measure eligible train/validation class
support separately, and prepare the resumable timing-head experiment. Additional
public-data corrections require a separate versioned build; this replay does not
overwrite pinned v2 or establish that all other OSSQ systems are correct.
