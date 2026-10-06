# Bounded OSSQ repaired timing packet — 2026-10-06

The 12 locally corrected crops now have canonical exact-timing sidecars, with
the corrected token bytes and public notation sidecars preserved. Nine focused
tests pass. No published corpus, runtime pin or training default changed.

Final evidence:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/ossq-repaired-timing-packet-20261006-v2/report.json`.
The adjacent token/notation/timing bundles are the qualified local packet.

The binder rechecks public and corrected token hashes, original source hashes,
and unique accidental-aware musical windows across all four parts. It compares
every visible source/segment note's timing in order. Archived segments rescaled
divisions and duration ticks; canonical source grids are restored. Two d’Indy
tuplet starts lost display-only `bracket=no` and `show-number=none`; canonical
attributes are restored too. Any other semantic timing difference, ambiguous
source-grid mapping or serialized identity mismatch stops the build.

| Split | Crops | Timed events | Elapsed/written/ratio eligible | Rest-head eligible |
| --- | ---: | ---: | ---: | ---: |
| Train (Borodin) | 4 | 88 | 88 each | 0 |
| Validation (Haydn, d’Indy) | 8 | 121 | 121 each | 19 |

There are zero missing timing payloads, masked duration conflicts or unseen
classes against the existing frozen mixed-source, tick-quantized training
manifest. Classes were not rediscovered from these crops or validation.
The training tranche supports quarter, half and eighth notes, all ratio 1:1.
Validation includes six triplet events and 19 non-full-measure rests. Thus this
packet does **not** establish training support for tuplets or the rest head.
The previous 88/121 annotated-symbol counts happen to equal timed-event counts
here; that equivalence was measured, not assumed.

Next: extend source-window certification to unchanged OSSQ crops with training
tuplets and rests, then combine their actual crop support with the existing
Lieder inventory before preparing resumable timing-head training. This bounded
label qualification is not itself a model-promotion result or an unconditional
release gate. The accepted pixel review remains closed.
