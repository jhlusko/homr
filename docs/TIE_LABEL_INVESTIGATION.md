# Tie and slur labels in the Lieder corpus: an open investigation

**Status: unresolved.** One defect confirmed and fixed, one fix whose effect is not
understood, three hypotheses tested and refuted. Written for someone picking this up cold.

Last worked: 2026-09-13. Commits `0937b63` … `b72284f` on `homr` main.

---

## 1. The problem

The corpus's own tie labels frequently cannot be drawn. A tie joins two notations of **one
pitch** - that is what distinguishes it from a slur, and it is a necessary condition, not a
preference. A `start` whose pitch does not recur in the next simultaneity of its own voice
is not a tie the corpus recorded; it is a label that can never find a partner.

Measured by `training/omr_datasets/reference_label_audit.py` over 1,500 crops:

```
                             Lieder (scanned)   OSSQ (corrected)
tie endpoints impossible           17.7%              1.0%
slur stops nothing opened           9.8%              0.2%
```

**OSSQ is effectively clean; Lieder is not.** Lieder is the corpus the scan models train
on. OSSQ is a string quartet (one line per staff); Lieder is voice-over-piano with a grand
staff, and 22-34% of its systems carry a staff with more than one voice.

This matters twice over:

- **Scoring.** Any slur or tie accuracy measured against this reference is capped by it.
  RUNLOG VI.9's "80.08% of predicted spans match the engraved reference" is *not* a clean
  measure of model error and should not be quoted until this is settled.
- **Training.** A head trained where one tie start in ten has no possible partner is being
  taught noise. The tie head's "earns its keep" verdict (macro-F1 .844 against a 15.9%
  derivable baseline) rests on OSSQ, which is clean, and says nothing about what the
  Lieder-trained scan model learned.

---

## 2. Where everything is

| thing | path |
| --- | --- |
| the audit | `training/omr_datasets/reference_label_audit.py` |
| tie extraction | `training/omr_datasets/structured_notation_parser.py` (`_tie`, `NotationExtractor`) |
| sidecar read/write | `training/omr_datasets/notation_sidecar.py` |
| token file writer | `training/transformer/training_vocabulary.py` (`token_lines_to_str`) |
| chord sorting | `homr/transformer/vocabulary.py` (`sort_token_chords`) |
| the shipped repair pass | `homr/tie_repair.py` |
| corpus builder | `lieder-omr-data/tools/build_lieder_v4.py` (separate repo) |

Corpora built during this investigation, all from the same detection run:

```
homr-artifacts/lieder-v5/pairs   schema v5   voice only
homr-artifacts/lieder-v6/pairs   schema v6   voice + onset index
homr-artifacts/lieder-v7/pairs   schema v6   + the sidecar ordering fix
```

Each holds 4,187 pairs. `_build` (pages, systems, ground truth) lives in `lieder-v6` and
`lieder-v7`; detection takes ~5 hours and should be reused via `--skip-detection`.

### Rebuilding

```bash
cd ~/workspace/lieder-omr-data
python3 tools/vendor_runtime.py --refresh --source ~/workspace/homr   # re-pin the runtime
mkdir -p ~/workspace/homr-artifacts/lieder-vN
cp -r ~/workspace/homr-artifacts/lieder-v7/_build ~/workspace/homr-artifacts/lieder-vN/
~/workspace/homr/.venv/bin/python tools/build_lieder_v4.py \
  --source sources/lieder-all-source --out ~/workspace/homr-artifacts/lieder-vN \
  --skip-detection
```

Takes a few minutes with detection reused. **The vendored runtime is a copy of `homr`;
changes to `homr` do not reach the build until you re-pin.** That cost one wasted rebuild.

### Auditing

```bash
cd ~/workspace/homr
PYTHONPATH=. .venv/bin/python -m training.omr_datasets.reference_label_audit \
  --corpus ~/workspace/homr-artifacts/lieder-v7/pairs --limit 1500
```

---

## 3. What was tried, in order

### 3.1 Hypothesis: the representation has no voice — **refuted**

A tie joins one pitch *within one voice*. The token format carried `position` (upper or
lower) and nothing else, so two voices on a staff were indistinguishable. Splitting 55
systems by whether any staff carried two voices gave 0.0% impossible in monophonic against
8.8% in polyphonic, which looked decisive.

`VoiceClass` was added (`87ff3d2`), sidecar schema v5, extraction renumbering part-global
MusicXML voices to 1..N per staff. The corpus was rebuilt.

**Result: no change.** With voices recorded and used by the audit, polyphonic files still
showed 27.6% impossible ties against 5.7% monophonic. The 55-system estimate was also
wrong in both arms - it rested on 73 tie endpoints.

### 3.2 Hypothesis: adjacency is unrecoverable without onsets — **refuted**

A token line is a simultaneity across *all* voices: the converter merges whatever sounds
together onto one line, so a line can hold voices 1, 2 and 3 at once and a voice's
successive notes may share a line or sit several lines apart. Filtering by voice cannot fix
that, because "the next chord" is itself a cross-voice notion.

`NoteNotation.onset_index` was added (`24775db`), sidecar schema v6: which simultaneity of
its own voice a note belongs to, counted per `(staff, voice)`, advanced only by a note
without `<chord/>` so chord members share an index. The corpus was rebuilt.

**Result: no change.** 27.6% → 27.4%.

### 3.3 Hypothesis: chord members are reordered, scrambling per-note labels — **partly right**

Reading the actual cases (rather than the counts) showed a tie start on E♭4 whose pitch
never recurs, while F4 *in the same chord* ties correctly a quarter later. Comparing the
crop's tokens against the source measures the alignment says it covers:

```
IMSLP10416 measure 30, lower staff
  source        E♭4   F4(chord)@tied-start
  token file    F4    E♭4                     ← sorted
  v6 sidecar    tie start on the 2nd entry → E♭4     WRONG
```

`token_lines_to_str` puts every chord through `sort_token_chords`, which ends
`return [sorted(chord) for chord in chords]`. `write_sidecar` wrote notation in the order
the caller passed - source order - and `attach_sidecar` pairs the two **by position**.

Fixed in `b72284f`: the sidecar is now written in `sort_token_chords` order.
`tests/test_sidecar_ordering.py` asserts a tie stays on its own notehead; two of its tests
fail on `e271e42` and pass on `b72284f`.

**This is a real defect and the fix is right on the traced case.** v7 puts the tie on F4
(index 40) and its stop on F4 (index 46), matching the source; v6 had both on the
neighbouring noteheads.

**But the aggregate got worse, and that is not understood:**

```
              one voice        polyphonic
v6   ties     5.7% impossible  27.4%
v7   ties    12.0% impossible  51.7%
```

### 3.4 Confirmed and fixed: ties on rests

`_tie` read `<tied>` from any note element including rests. A rest is silence; nothing
sustains into the next note. 25 of the impossible labels were exactly this. Fixed in
`e271e42`.

---

## 4. The composition of the failures

From `lieder-v6` (before the ordering fix), reading every impossible tie start:

```
joins its pitch (fine)                              934
IMPOSSIBLE: pitch absent from the next simultaneity  152
partner past the crop                               107   (legitimate - next system)
IMPOSSIBLE: tie start on a rest                       25   (fixed in e271e42)
```

Only the last was explained at that point. The ordering fix addresses an unknown share of
the 152.

---

## 5. The open question, and the test that settles it

**Why did correcting the sidecar order make the audit worse?**

Two readings, and they are distinguishable:

1. **The metric was flattered by the scrambling.** A tie start landing on the wrong chord
   member can *accidentally* satisfy the constraint when that member's pitch happens to
   recur. Correcting the alignment would then expose genuinely unpairable labels that were
   previously hidden - so the number rising is consistent with the fix being right.
2. **The fix broke something else.** Possible if `write_sidecar`'s ordering does not in
   fact match what the token file holds for some input shape.

### The decisive test

**Stop comparing the corpus to itself.** Every number above is internal consistency; the
audit asks whether a label can pair within the crop, not whether it matches the page.

For a sample of crops, compare against the **source MusicXML** the alignment names:

1. Read `provenance/alignment_v4_boundary_safe.json` (in `lieder-omr-data`) for
   `scores[SCORE_ID]["systems"]`, find the entry whose `scan_index` matches the crop's
   `sysN`, and take `start_measure` / `end_measure`.
2. Load the source `.mxl` via `_build/mxl-tree.json` keyed by `lieder_key` from
   `_build/ground_truth/SCORE_ID.json`; the crop's `-vN` suffix is the part index.
3. For every `<tied type="start">` the source records in that measure range, find the
   corresponding notehead in the crop's tokens (same staff, same pitch, same position in
   the voice) and check whether the sidecar records the tie **on that note**.
4. Report agreement for `lieder-v6` and `lieder-v7` separately.

Whichever corpus agrees with the source more often is the better one, and the answer does
not depend on the audit's own notion of pairability. A worked example of steps 1-2 is in
this session's transcript; the alignment entry for `IMSLP10416` `sys6` is
`start_measure 27, end_measure 31`, part index 1, source `lc5946872.mxl`.

Only after that is settled is it worth asking whether the residue is the alignment stage
dropping notes, or the OpenScore labels themselves.

---

## 6. Traps

- **Re-pin the vendored runtime** before every rebuild, or the build runs old code
  (`tools/vendor_runtime.py --refresh`, then `--verify`).
- **The audit is internal-consistency only.** It cannot tell a wrong label from a label on
  a note the crop does not contain.
- **Crops are loosely framed vertically**: they routinely include a cut-off staff from the
  adjacent part and page furniture (footers, publisher lines). Horizontal alignment
  (measure counts) checked out in every case examined. This does not corrupt the labels but
  it does put unlabelled ink in the training image.
- **Counting artifacts is not checking them.** An earlier detection run wrote 215 system
  files, every one of them `{"pages": {}}`, and it was taken as success because the file
  count was right.
- **`sorted()` on `EncodedSymbol`** compares the six token fields. Two chord members with
  identical fields sort stably, so source order survives there; the reordering only bites
  when the fields differ.

## 7. Lessons from how this went

Three hypotheses were formed from correlations and two of them were shipped as
representation changes before being tested against ground truth. Each was plausible, each
explained the polyphonic/monophonic split, and neither moved the number. The one real
defect was found by **opening the image, reading the bar, and diffing the crop's tokens
against the source measures** - which cost one afternoon and could have come first.

The pattern to avoid: measuring a corpus against itself, finding a gap, and inventing a
mechanism for the gap. The reference has to enter the comparison.
