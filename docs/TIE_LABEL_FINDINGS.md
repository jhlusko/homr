# Tie-label investigation findings, 2026-09-13

The v7 sidecar ordering fix is incomplete. It follows `sort_token_chords`, but the
token writer subsequently sorts each chord again in `_chord_to_str`, using
`_symbol_to_sortable`. V7 therefore attaches notation to the wrong notes whenever
those two orders differ. This explains why it repairs the documented F4 chord and
nevertheless makes aggregate tie agreement worse.

The remedy tested here is to apply **both sorts**, preserving their stable ordering.
It was tested in a separate artifact corpus; production code, checkpoints, vendored
runtime, frozen alignment, and existing v6/v7 corpora were not modified.

## Independent source comparison

`training/omr_datasets/source_tie_label_audit.py` reads the frozen alignment,
`_build/ground_truth/SCORE_ID.json`, `_build/mxl-tree.json`, and the archived MXL
directly with the standard library. It does **not** use HOMR's MusicXML parser,
sidecar voice/onset labels, tie pairing rules, or the proposed order to match notes.

The matching key is positional measure index, staff, written pitch letter/octave,
and rhythm. A key must identify exactly one source note and one token note.
Repeated/ambiguous keys remain unscored. Rests are included in sidecar indexing,
but the agreement table concerns pitched notes. Grace notes, unsupported rhythms,
and entire staff measures touched by an octave shift are excluded from matching.
Octave-shift state is tracked from the beginning of the source, before the crop.
The comparison reads graphical `<notations><tied>` start/stop states directly.

All **4,187 crops** were examined. Of 145,654 visible source pitched notes,
71,045 had conservative unique matches. Of 8,538 source notes carrying a tie state,
6,220 were matched. These are selected-note agreement figures, not unbiased
estimates of whole-corpus or scan accuracy.

| Measure, on identical matched notes | v6 | v7 | Both-sort experiment |
| --- | ---: | ---: | ---: |
| Exact state on 6,220 source tied notes | 5,709 (91.8%) | 4,974 (80.0%) | 6,220 (100%) |
| Tie incorrectly attached to an untied source note | 615 | 1,070 | 0 |
| Any tie-state disagreement among 71,045 matched notes | 1,126 | 2,316 | 0 |
| Source start endpoints missed | 268 | 623 | 0 |
| Source stop endpoints missed | 223 | 621 | 0 |

The unscored source population comprises 67,708 ambiguous notes and 6,901 unmatched
notes (including unsupported cases). Among source tied notes, 2,089 were ambiguous
and 229 unmatched. A `start_and_stop` is one tied note but two endpoints.

During development, failing to account for octave shifts produced five apparent
residual disagreements: a raw sounding pitch could match a *different* written
note at that octave. The final matcher excludes affected staff measures entirely.
Those five cases were matching errors, not established extraction defects.

## Controlled ordering experiment

`training/omr_datasets/tie_order_probe.py` regenerates source tokens with the current
parser and the builder's measure slicing/natural handling. Before accepting a crop,
it requires:

1. Every serialized note/rest row, including all six fields, matches the existing
   token file in order. Non-note numerator suppression is irrelevant to this check.
2. Every v7 sidecar record is reproduced exactly by the first sort alone.

Both checks passed for **all 4,187 crops**, with no skipped or mismatched crops.
The experiment then applies `_symbol_to_sortable` within each first-sorted chord,
writes those notation records beside unchanged token text, and submits the result
to the independent source comparison above.

Of 158,102 note/rest records, 59,571 positions receive different notation records;
3,716 receive a different tie state. This is a permutation defect affecting all
sidecar fields, including voice and onset, not just ties.

The original four ordering tests all still pass on the defective writer. In
particular, the purported grand-staff test puts the hands in *separate* token
simultaneities, so it misses the cross-staff permutation. Another test derives its
expected order from the first sort itself. A production regression test needs a
single simultaneity containing mixed rhythms and both staves, with distinct
notation identities checked after an actual write/read/attach cycle.

## Visual checks

### IMSLP10416-sys6-v1: the handoff's own example establishes both outcomes

I inspected the [scan crop](../../homr-artifacts/lieder-v7/pairs/IMSLP10416-sys6-v1.png)
and its aligned source, `lc5946872.mxl`, part 1, positional measures `[27,31)`.
The upper staff clearly connects C5 across the first barline. The final lower-staff
measure has the F4 tie identified in the earlier investigation; its lower arc
connects different pitches and is a slur.

Indices below are zero-based among note/rest sidecar entries:

| Visible/source endpoint | Token index | v6 | v7 | Both sorts |
| --- | ---: | --- | --- | --- |
| C5 half-note start, upper, measure 27 | 7 | start | none | start |
| C5 quarter-note stop, upper, measure 28 | 13 | stop | none | stop |
| F4 half-note start, lower, measure 30 | 40 | none | start | start |
| F4 quarter-note stop, lower, measure 30 | 46 | none | stop | stop |

V7 puts the C5 start onto D3 at index 9 and the C5 stop onto A-flat4 at index 14.
At the C5 start's simultaneity, the first sort orders the source symbols as
`A4 eighth (upper), D3 half (lower), C5 half (upper)`, while the token writer emits
`C5 half (upper), A4 eighth (upper), D3 half (lower)`.
This is a direct explanation of a visible wrong label, not an inference from counts.

### IMSLP112763-sys0-v1: independent piano example

The [scan](../../homr-artifacts/lieder-v7/pairs/IMSLP112763-sys0-v1.png) visibly ties
the low B1 half note to B1 later in each of its first three measures. Raw source
notes likewise carry tie starts. V7 omits the starts from the B1 half notes at
indices 1, 31, and 61. The both-sort result agrees with the source for these notes.

### Residual audit flags are not all ordering errors

In [IMSLP154110-sys3-v0](../../homr-artifacts/lieder-v7/pairs/IMSLP154110-sys3-v0.png),
the first measure contains small E5–F-sharp5 grace pairs leading to principal E5
notes, with short arcs under the figures. The source explicitly records a tie
start on the grace E5 and a stop on the principal E5, across the grace F-sharp5.
The internal audit calls the start impossible because it checks the next onset.
This case needs grace-aware notation interpretation, not sidecar reordering.

In [IMSLP154070-sys4-v1](../../homr-artifacts/lieder-v7/pairs/IMSLP154070-sys4-v1.png),
the first lower-staff figure is B-flat3–G4–B-flat3 under an arc; the second is
B-flat3–A4–B-flat3. The source encodes the same-pitch endpoints as `<tied>` even
though the intervening notes have ordinary, nonzero durations in the same voice.
The scan shows the arcs but does not by itself establish sustained sound across
those intervening attacks. A source tie/slur interpretation issue is plausible;
the corrected sidecar faithfully reproduces the source here.

## What the internal audit does and does not establish

On the first 1,500 lexically sorted crops, with 2,344 tie-bearing note records:

| Existing audit count | v6 | v7 | Both sorts |
| --- | ---: | ---: | ---: |
| Starts whose pitch is absent at the next onset | 177 | 375 | 46 |
| Stops classified as orphaned | 273 | 483 | 101 |

These raw counts are reproducible with the current audit; historical percentages
in the handoff use other groupings/denominators and should not be substituted here.
The 46 residual starts have not all been individually classified.

Additional audit limitations visible in the code:

- Pitch comparison ignores accidentals, so equal letter/octave is not sufficient
  to establish equal sounding pitch.
- A stop at the first note of its own voice can be called orphaned unless it is
  literally the first note of the entire crop. Conversely, any earlier same-pitch
  start can satisfy a stop without enforcing adjacency or consuming that start.
- Slur slot pairing uses one global open state per slot, although extraction
  allocates slots independently per source voice.
- Token endpoints are hoisted and deduplicated per staff/simultaneity. Counting
  per-note sidecar endpoints is not an equivalent representation. The 198
  token/sidecar count disagreements remain unchanged by either ordering, which is
  expected for a permutation.

Thus this investigation resolves the ordering regression, not the entire quality
of the source's tie/slur annotation or the scanner model's accuracy.

## Artifacts, reproduction, and next implementation

Results live in `../homr-artifacts/tie-label-investigation-20260913/` relative to
the HOMR repository: `source-comparison.json` contains per-crop counts and every
disagreement; `lieder-order-probe/order-probe.json` records reproduction counts and
example permutations; `lieder-order-probe/pairs` contains experimental tokens and
sidecars only. The experiment was initially generated under `/tmp/lieder-order-probe`
and moved to that persistent artifact directory after evaluation; the JSON retains
the original input path as run provenance.

From `homr`, using a new, nonexistent experiment output directory:

```bash
.venv/bin/python -m training.omr_datasets.tie_order_probe \
  --alignment ../lieder-omr-data/provenance/alignment_v4_boundary_safe.json \
  --build ../homr-artifacts/lieder-v7/_build \
  --corpus ../homr-artifacts/lieder-v7/pairs \
  --out /tmp/new-tie-order-experiment

.venv/bin/python -m training.omr_datasets.source_tie_label_audit \
  --alignment ../lieder-omr-data/provenance/alignment_v4_boundary_safe.json \
  --build ../homr-artifacts/lieder-v7/_build \
  --corpus ../homr-artifacts/lieder-v6/pairs \
  --corpus ../homr-artifacts/lieder-v7/pairs \
  --corpus /tmp/new-tie-order-experiment/pairs \
  --out /tmp/new-tie-source-comparison.json
```

Validation: three new independent-matcher tests and the four existing ordering
tests pass (7 total). The reusable experiment CLI was also exercised on one crop.
All-corpus reproduction and source agreement are the substantive validation.

The next production change should share the **complete serialization order**
between the token and sidecar writers, without changing token bytes or the model
vocabulary. Add mixed-staff/mixed-rhythm round-trip regressions, then re-pin the
vendored runtime and build a new corpus version. Do not overwrite v6/v7 or retrain
from the experimental directory as though it were a complete rebuilt dataset.
