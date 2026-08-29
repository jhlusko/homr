# Onset representation: prior art, tradeoffs, and options for the grand-staff bar-duration blocker

Research note. No code was changed. Measurements below were taken read-only against the
local rebuilt Lieder corpus (`~/workspace/homr-artifacts/corpora/lieder_scanned_pairs`,
3,715 token files) with a throwaway script in `/tmp`.

---

## 1. Executive summary

homr's token stream encodes music at what the OLiMPiC paper calls the **homophonic**
tier - "chords allowed, but all simultaneous notes have the same length" - and applies it
to **pianoform** music, which that same typology places two tiers higher. The gap between
those two tiers is the whole blocker.

Concretely: a simultaneity (`&`-joined chord group) carries no statement of how much time
passes before the next one. The renderer and every audit tool recover that by taking the
**minimum duration in the group**. That rule is exact for a single staff, and it is a
guess whenever the two hands of a grand staff are rhythmically independent - which,
measured on the current corpus, is **25.5% of all grand-staff simultaneity groups**, and
**46.4%** of the groups where both hands actually sound together.

The consequence is not a decoder bug and cannot be fixed downstream: bar duration on a
grand staff is *unrecoverable from the label itself*. That is why
`audit_label_consistency.py` refuses every duration-dependent check on a grand staff, why
the overfull-pair rule was found unsound, and why `cross_staff_consistency`'s three
strongest checks (`check_measure_counts`, `check_measure_durations`,
`check_barline_positions`) are dark over 45% of the training corpus.

**Recommendation: add an "advance" (onset-delta) head as a structured, output-only head
(Option A).** It is the only option that fixes the representation without touching the
rhythm vocabulary, without invalidating checkpoint 426, and without re-cutting a single
image. Its training target is *already computed and then discarded* by the existing
tokenizer, so the expensive half of the migration - deriving ground-truth onsets - is
already paid for. Estimated cost: one build session plus one frozen-core probe run, with
a decisive go/no-go before any full-model commitment.

---

## 2. What the current representation actually is

(Recovered and re-verified; this is the internals section from the interrupted run.)

### 2.1 The token line

`homr/transformer/vocabulary.py` defines six parallel branches, decoded by six heads at
each autoregressive step:

| branch | size | role |
|---|---|---|
| rhythm | 279 | durations, rests, clefs, key/time signatures, barlines, `chord`, control tokens |
| pitch | 72 | C0-B9 plus `.`/`_` |
| lift | 7 | accidentals |
| articulation | 61 | combined articulation strings |
| slur | 5 | slur start/stop |
| **position** | **3** | `.`, `upper`, `lower` |

`position` is the *only* thing that distinguishes the two staves of a grand staff. There
is no voice branch, no staff-change token, no time-shift token, and no onset field.

A simultaneity is written on one line with `&` separators; `&` is shorthand for the
rhythm token `chord`. So `note_4 G4 _ _ _ upper & note_2 C3 _ _ _ lower` is one group
holding a quarter in the right hand and a half in the left.

### 2.2 Where onsets are created, and where they are thrown away

`training/omr_datasets/music_xml_parser.py` is the MusicXML tokenizer, and it is fully
onset-aware. `MeasureBuilder` maintains `self.current_position` in MusicXML divisions,
`append_position_change()` implements `<backup>`/`<forward>`, and every symbol is wrapped
as `EncodedSymbolWithPos(position, symbol, insert_before)` with `sort_order() =
position * 2 - (1 if insert_before else 0)`.

`training/omr_datasets/staff_merging.py::merge_upper_and_lower_staff` then does this:

```python
positions[symbol.sort_order()].append(symbol.symbol)
...
for key in sorted(positions):
    result.extend(create_chord_over_two_staffs(positions[key]))
```

It buckets by exact onset, emits the buckets **in onset order**, and then **discards the
keys**. The ordering survives into the token file; the *gaps between onsets* do not.
That single line is the representational loss this whole document is about, and it is
also why the fix is cheap: the information exists one function call upstream of where it
is dropped.

The kern/music21 path (`training/omr_datasets/music21_kern_parser.py::_convert_m21_staff_group`)
does the same thing with `by_offset: dict[float, list[object]]` and `for off in
sorted(by_offset)`. Same loss, same availability.

### 2.3 How the renderer guesses the gap back

`homr/music_xml_generator.py`:

```python
def get_duration(self) -> Fraction:          # SymbolChord
    notes_rests = [s.get_duration().fraction for s in self.symbols
                   if s.rhythm.startswith(("note", "rest"))]
    return min(notes_rests) if notes_rests else Fraction(0)
```

and in `build_measures`:

```python
staff_positions = group.into_positions()
for pos_no, staff_pos in enumerate(staff_positions):
    chord_duration = (group.get_duration()
                      if pos_no == len(staff_positions) - 1 else Fraction(0))
```

`build_note_chord` is genuinely careful within a group - `_group_notes` buckets by
duration, emits `<backup>` between buckets, and finishes with
`backup(max(by_duration) - chord_duration)` - so *within* one simultaneity the two hands
render correctly. The cursor then advances by exactly `min`. `rebalance_measure_voices`
afterwards replays `<backup>`/`<forward>` to assign non-overlapping MusicXML voices per
staff, so the output XML is well-formed.

The identical rule appears in `vocabulary.py::_get_duration_of_measure`, which drives
metre inference (`find_division_and_time_signature_nominator`), the modal-bar check
behind `distrust_stated`, and `_fix_over_eager_tuplets`.

### 2.4 Why min-advance is not merely approximate but structurally wrong

The rule is exact iff, for every group, `min(durations in group)` equals the gap to the
next onset. That holds when the onset sets of the two staves are *nested*. It fails as
soon as they are not:

> upper: quarter, quarter. lower: dotted-quarter, eighth.
> True onsets: 0, 1/4, 3/8. Groups: `{q, q.}` then `{q}` then `{8th}`.
> min-advance gives 1/4 + 1/4 + 1/8 = **5/8**. The bar is 1.

There is no post-hoc rule that recovers the 1/8 - the label never stated it. Splitting
the stream by `position` does not help either, because the duration was attributed to
exactly one of the two halves (see the `chord_duration ... else Fraction(0)` above), and
symbols with no position (barlines, key/time signatures) belong to neither.

`training/omr_datasets/audit_label_consistency.py` states this outright and quarantines
the affected checks:

```python
DURATION_DEPENDENT = (check_measure_counts, check_measure_durations, check_barline_positions)
SYMBOL_ONLY        = (check_key_signatures, check_time_signatures, check_shared_motifs)
```

### 2.5 Fresh measurement of the blocker (this session, on ground-truth labels)

Over 3,715 rebuilt Lieder token files (1,551 grand-staff, 2,164 single-staff):

| measurement | grand staff | single staff |
|---|---|---|
| simultaneity groups holding **more than one distinct duration** | **25.5%** (9,692 / 37,952) | n/a |
| ...restricted to groups where both hands sound at once | **46.4%** (8,992 / 19,390) | n/a |
| bars whose min-rule duration disagrees with their own file's modal bar | **10.3%** (631 / 6,134) | **2.4%** (193 / 8,028) |
| naive per-staff split disagrees on bar length | **14.1%** (843 / 5,982) | n/a |
| decoder tokens per crop, median / p95 / max | 97 / 195 / 335 | 20 / 30 / 55 |

Two things to read off this. First, the 4.3x deviation ratio is measured on **ground
truth**, not on decodes, so it is a property of the representation. Second, the earlier
"median bar 13/16 against 3/4" figure quoted in `audit_label_consistency.py` no longer
reproduces on the rebuilt corpus - the median is now 3/4 on both - so that specific
number should be treated as stale and replaced with the deviation-rate figures above.
The blocker is real but its headline statistic changed when the corpus was rebuilt.

### 2.6 Why it matters even though OSSQ is 0% grand staff

`BENCHMARKS.md` removed grand staves from training and measured the loss on three
corpora: OSSQ -1.9 to -2.1pp, PDMX -3.16pp (CI -4.34 to -2.01), Lieder -15.2 to -16.1pp.
The grand-staff half of the corpus is load-bearing training signal *even for the
single-staff benchmark*. And the same file records that **structural errors - losing the
bar grid - are the one error class that no corpus change has ever moved**, and concludes
"that points at alignment or architecture, not at data quantity or label quality". A
representation that cannot state a bar's duration on 45% of its training data is a
direct, named candidate for that.

---

## 3. Survey of prior art

### 3.1 The typology everyone is implicitly working in

The OLiMPiC/ICDAR-2024 paper (Mayer et al., *Practical End-to-End Optical Music
Recognition for Pianoform Music*) sets out four tiers: (a) monophonic - a plain sequence;
(b) homophonic - chords allowed, **all simultaneous notes the same length**, workable
even under CTC; (c) polyphonic - linearization becomes necessary; (d) pianoform - staff
interaction plus the highest object density, "the final frontier".

homr's encoding is tier (b) with a `position` flag bolted on, used on tier (d) material.
Naming that is the single most useful thing this survey produced: the min-advance rule is
not an implementation shortcut, it is the defining assumption of tier (b).

### 3.2 Linearized MusicXML (LMX) - explicit time travel

LMX linearizes MusicXML by depth-first traversal, keeping MusicXML's own element order as
token order. Relevant design decisions:

- **Onset is implicit in ordering plus duration** - same as homr - *but* LMX keeps
  `<backup>` and `<forward>` as first-class tokens, so the clock can move arbitrarily.
  Backup is emitted as a jump to the start of the measure before each new voice; forward
  fills gaps where a voice starts or stops mid-measure.
- Backup/forward have no `<type>`, so LMX encodes their duration as a **sum of type
  tokens** (`forward half forward quarter forward eighth`) - a neat trick for expressing
  arbitrary rationals over a small vocabulary.
- `staff:1` / `staff:2` and `voice:1..8` are **state-change** tokens, reset at measure
  start and at every backup, not repeated per note.
- Voices are laid out **voice-wise within a measure**, not onset-wise. The paper notes
  explicitly that "**kern orders notes onsetwise, not voice-wise" - the two families made
  opposite choices.
- Vocabulary: **224 tokens** total, 182 observed in OLiMPiC. Dataset: 17,945 systems,
  4,107,597 tokens in the train split = **~274 LMX tokens per grand-staff system**.
- Results: Zeus (CNN + BiLSTM) reaches 11.3% SER / 13.7% full TEDn on synthetic, and
  18.4% full TEDn on scanned with augmentation. About 4% of MusicXML nodes are dropped by
  the encoding (dynamics, barline styles, pedal).

The relevant lesson: LMX pays roughly **2.8x homr's token count** per grand staff (274 vs
our measured 97 median) to buy exact onsets and full MusicXML round-tripping.

### 3.3 Humdrum **kern / bekern / ekern - parallel spines

The GrandStaff dataset and the Sheet Music Transformer line (Ríos-Vila et al.) encode
pianoform music as **kern, where each staff is a spine and a line of the file is read
left-to-right for simultaneities, then top-to-bottom. Onsets are therefore explicit by
construction: every spine advances in lockstep because null tokens (`.`) hold a spine's
place. Tokenization variants:

- raw kern: >20,000 unique tokens, training instability;
- **bekern** (basic kern): decomposed into minimal semantic units, ~133-215 tokens;
- **ekern**: split by graphical meaning.

Both bekern and ekern beat raw kern. SMTNeXt reports 5.6% CER / 6.9% SER / 12.9% LER on
FP-GrandStaff synthetic; 14.1% SER on Mozarteum and 25.8% on Polish Digital Scores after
fine-tuning.

The known limitation, quoted by the OLiMPiC authors as their reason for not using it:
**kern "cannot represent certain situations, such as a voice changing staves in the
middle of a beamed group" - which is exactly the piano-specific case homr would most want
covered. Also, spine-parallel encodings pay a null-token tax proportional to how often
the staves *don't* align, which is the 46.4% figure above.

### 3.4 MIDI-domain tokenizations - the parallel-field precedent

Not OMR, but the closest structural analogue to homr's six-head decoder:

- **REMI** (Huang & Yang 2020): explicit `Bar` and `Position` (1/16 grid) tokens make
  onset absolute rather than cumulative. Robust to a single duration error - the error
  cannot propagate past the next `Position`.
- **Compound Word** (Hsiao et al. 2021): groups a note's fields into one "compound word"
  with **field-specific parallel heads and embedding pooling**. This is architecturally
  the same shape as homr's rhythm/pitch/lift/articulation/slur/position heads.
- **Octuple** (MusicBERT, Zeng et al. 2021): every note carries eight fields, including
  **bar** and **position** as their own fields. Onset is stated per note, absolutely.

The transferable lesson: in every one of these, when onset had to be recovered reliably,
the answer was **to give onset a field of its own**, not to infer it from durations. And
Compound Word/Octuple show that adding a field to a parallel-head model is a cheap,
well-trodden move rather than a redesign.

### 3.5 Others considered and set aside

- **LilyPond**: rejected by the OLiMPiC authors as "a programming language rather than a
  data format" (e.g. it can force an accidental with `!` but cannot state an explicitly
  absent one).
- **MEI**: mature and expressive, but heavier than MusicXML and less well served by
  MuseScore, which is homr's whole ground-truth toolchain.
- **ABC**: monophonic-first; multi-voice piano support is an extension, not a core
  design.
- **Agnostic vs. semantic encodings** (Calvo-Zaragoza et al.): the agnostic family
  encodes glyph-plus-staff-position and defers musical meaning entirely. homr's `position`
  branch is a vestigial agnostic feature inside an otherwise semantic vocabulary, which is
  part of why it does not carry enough information to reconstruct time.
- **TrOMR** (homr's own architectural ancestor): single-staff by design; the grand-staff
  support in homr is an extension of it, and the min-advance rule is where that extension
  ran out.

### 3.6 What the survey converges on

Every system that handles pianoform music correctly makes onset **explicit** - either by
spine-parallel null tokens (**kern), by time-travel tokens (MusicXML/LMX), or by an onset
field per note (Octuple/CP). No system in the survey recovers onset from a min-duration
rule. homr is alone in tier (b) among tier (d) systems.

---

## 4. Tradeoff analysis

The axes that actually discriminate between the options here:

1. **Expressivity** - can it state a bar where the hands are rhythmically independent?
   Can it state a voice changing staff mid-beam?
2. **Sequence length** - grand-staff crops already run to 335 decoder tokens against
   `max_seq_len = 608`. An encoding that doubles length breaks the cap on the dense tail
   and forces a re-export of the ONNX graphs.
3. **Checkpoint compatibility** - the project's established rule (`§5.1`, and the
   `TIME_SIGNATURE_BEATS_PREFIX` comment) is that new semantics get new heads or
   *appended* tokens, never renumbered ones, because renumbering silently invalidates
   every trained embedding.
4. **Corpus re-conversion cost** - seven converters (`convert_lieder`,
   `convert_grandstaff`, `convert_olimpic`, `convert_ossq`, `convert_pdmx`,
   `convert_primus`, `convert_musetrainer`) plus the Lieder scan pipeline. Anything that
   changes the *token line format* touches all of them and every corpus artefact.
5. **Target availability** - can the training target be derived from what the converters
   already parse, or does it need new musical analysis? (Decisive here: see §2.2.)
6. **Downstream blast radius** - 19 files reference `get_duration` / `group_into_chords` /
   `measure_durations`, including `music_xml_generator.py`, `cross_staff_consistency.py`,
   `tuplet_repair.py`, `staff_parsing.py`, and six audit tools.
7. **Measurability** - can the change be evaluated without a full training run? The
   project has an explicit discipline here (frozen-core probe before unfrozen run, real
   benchmark rather than loss alone) and options that cannot be probed cheaply should be
   ranked down for that reason alone.

Two cross-cutting facts constrain everything:

- **OSSQ cannot measure this.** OSSQ is 0% grand staff. Any fix here must be evaluated on
  Lieder, OLiMPiC, and PDMX, and the headline OSSQ number will not move. Anyone reporting
  this work must say so up front or it will read as a null result.
- **The corpus's sub-1% label defects have all been null.** `BENCHMARKS.md` records four
  correct corpus fixes in a row that moved nothing, against a 0.23-0.69pp noise floor.
  A defect touching 25.5% of grand-staff groups is two orders of magnitude larger than
  those, which is the argument that this one is different - but it also means the bar for
  "we measured an improvement" needs seeds, not a single run.

---

## 5. Ranked options

### Option A - Advance (onset-delta) head, as a structured output-only head. **Recommended.**

Add one non-autoregressive head that predicts, for each simultaneity group, **how much
time passes before the next group** - a classification over a small duration vocabulary
(the kern duration set plus a `same-onset` class), not a change to any existing branch.

- **Fixes**: the min-advance rule is replaced by a stated advance. Bar duration becomes
  computable on a grand staff; `DURATION_DEPENDENT` checks can be re-enabled; the
  overfull rule becomes sound; metre inference stops being fed guessed bars.
- **Expressivity**: full onset independence between hands. Does *not* by itself express a
  voice changing staff mid-beam (that needs a voice branch), but that case is rare and is
  not what is blocking us.
- **Sequence length**: **zero growth.** The head is a projection of the existing hidden
  state at existing decode steps.
- **Checkpoint compatibility**: total. `configs.py` already documents that the structured
  heads "export as their own small graph rather than as extra decoder outputs... which
  also means a deployment can drop this file and keep working exactly as it did."
  Checkpoint 426 loads unchanged; when the head is absent, the renderer falls back to
  min-advance and behaves exactly as today.
- **Target availability**: **free.** `merge_upper_and_lower_staff` already computes exact
  onsets and sorts by them; the advance target is the difference between consecutive keys,
  normalized by `divisions`. `_convert_m21_staff_group` has the same via `by_offset`.
- **Precedent**: beam, stem, slur, tie, and dynamic heads were all built this way, with
  `structured_heads.py`, `structured_targets.py`, `structured_losses.py`,
  `structured_decoding.py`, `structured_metrics.py`, `notation_sidecar.py` (versioned) and
  `capability_manifest.py` already in place, plus
  `tromr_arch.freeze_core_for_structured_heads()` for the probe.
- **Risks**: (i) the advance is a *group-level* property being predicted at a *token-level*
  position - the target must be attached to a canonical member of the group (last symbol
  of the group is the natural choice, matching where the renderer consumes it) and masked
  elsewhere; (ii) class imbalance - the overwhelming majority of advances equal the
  min-duration, so the head can score 90%+ by learning the existing rule and teach us
  nothing. **The metric must be accuracy on the 25.5% subset where the group is
  ambiguous**, reported separately, exactly as the dynamics head's per-label accuracy was
  (the project has hit the majority-class trap twice already).

### Option B - Explicit backup/forward tokens appended to the rhythm vocabulary (LMX-style)

Append `advance_<dur>` / `backup_<dur>` tokens to the **end** of `build_rhythm()`, the way
`timeSignatureBeats_*` was appended, and emit them between simultaneities.

- **Fixes**: same as A, plus arbitrary time travel, so it can also express a voice
  starting or ending mid-measure and (with a voice branch) a staff change mid-beam.
- **Sequence length**: grows by roughly one token per non-nested onset - on the measured
  distribution, ~+25-35% on grand staves, pushing the 335-token tail toward ~450 against a
  608 cap. Tight but survivable; a re-export would be prudent.
- **Checkpoint compatibility**: good but not free. Appending leaves existing embeddings
  valid, but the output softmax grows and old checkpoints will never emit the new tokens,
  so *this needs real training* to be worth anything - there is no cheap frozen probe.
- **Corpus re-conversion**: all seven converters and every corpus artefact must be
  regenerated, because the token *line sequence* changes.
- **Blast radius**: every consumer of `group_into_chords` must learn to skip or consume
  the new tokens, including six audit tools and `validation/ned_score.py` (which would
  otherwise start scoring the new tokens as content and make old and new checkpoints
  incomparable).
- **Verdict**: the right *eventual* shape, and the natural escalation if A's head cannot
  learn the ambiguous cases. Ranked second only because it costs a full retrain and a
  full corpus rebuild to find that out.

### Option C - Decode the two staves separately

Either cut the grand staff into two crops, or condition the decode on a staff index and
run it twice.

- **Fixes**: eliminates interleaving entirely; each stream is tier (b) again, where the
  min rule is exact.
- **Costs**: loses the cross-staff alignment the model currently gets for free from seeing
  both staves in one crop - and the whole `cross_staff_*` track exists precisely because
  independent decoding causes disagreement. Requires re-cutting the image corpus.
  OLiMPiC's unit *is* a grand staff, and `convert_ossq` explicitly refuses grand staves
  for the mirror-image reason. It also cannot represent cross-staff beams or a hand
  crossing the divide - which is the main thing piano notation does.
- **Verdict**: trades a solvable representation problem for an unsolvable alignment one.
  Not recommended.

### Option D - Adopt LMX wholesale

Replace the vocabulary with LMX and become directly comparable to the OLiMPiC literature.

- **Fixes**: everything, with published baselines and a published metric (TEDn) to score
  against.
- **Costs**: a rewrite. New tokenizer for seven converters, new renderer, ~2.8x sequence
  length (274 median tokens/system vs our 97, against a 608 cap), the six-head parallel
  architecture becomes pointless (LMX is single-stream), **every checkpoint invalidated**,
  and the entire benchmark history becomes non-comparable. TEDn also costs 30-120s per
  system to evaluate.
- **Verdict**: the strongest destination if the project were starting today, and worth
  keeping on the table as a long-horizon direction. Not a migration anyone should attempt
  as a fix for one defect.

### Option E - Post-hoc onset reconstruction at render time

Keep the representation; at render time, solve for a set of advances consistent with the
measure's stated or inferred time signature.

- **Fixes**: nothing at the label level. The training data stays wrong, so the model keeps
  learning tier (b) semantics, and the audit tools still cannot validate a grand staff
  (they would be validating their own solver).
- **Value**: it is nonetheless the cheapest thing on this list and a legitimate **stopgap
  for output quality only** - it would make rendered MusicXML from a grand staff line up
  in a bar more often. Its real use is as a **measurement harness**: run the solver
  against converter-side true onsets and you get, for free, the exact distribution of how
  wrong the min rule is, per bar, which is the Phase 0 evidence Option A needs.
- **Verdict**: do it as Phase 0 instrumentation, not as the fix.

### Option F - Status quo

Keep the exclusions. Grand staves remain unauditable, `DURATION_DEPENDENT` stays dark
over 45% of training data, and the overfull-style rules keep being rediscovered as unsound.
Listed for completeness and as the baseline every option above must beat.

**Ranking: A > B > E (as instrumentation) > D > C > F.**

---

## 6. Recommendation and migration cost

**Do Option A, staged, with Option E's solver built first purely as measurement, and
Option B held as the named escalation.**

### Phase 0 - measure it properly (cheap, no training)
Instrument the converters to emit, alongside each token file, the true advance sequence
they already compute. Then report, over the whole corpus: how often min-advance equals
truth; the distribution of the error where it does not; and the per-bar duration error.
This session's numbers (25.5% / 46.4% / 10.3% vs 2.4%) are a lower bound derived from
token files *after* onsets were discarded - the converter-side measurement is the real
one, and it also produces the training target as a by-product. Cost: hours, no GPU.
**Gate:** if the true-onset measurement shows min-advance is exact more than ~95% of the
time on grand staves, stop - the blocker is smaller than believed and the exclusions can
be replaced by a narrow guard instead.

### Phase 1 - build the head
- `structured_notation.py`: new `Advance` field on `NoteNotation` (or a small
  group-level record), defaulted so existing sidecars keep decoding, following the
  precedent `tie` and `dynamic` both set.
- `structured_targets.py`: derive the target from the converter's onset keys; mask every
  token that is not the group's canonical carrier.
- `structured_heads.py` / `structured_losses.py` / `structured_decoding.py` /
  `structured_metrics.py`: one head, one loss term, one decode path, **and a metric split
  into "ambiguous groups" vs "unambiguous groups"**.
- `notation_sidecar.py`: bump v3 -> v4. `capability_manifest.py`: declare the head.
- Regenerate sidecars for the grand-staff corpora. No image work, no re-cutting.

### Phase 2 - frozen-core probe
`freeze_core_for_structured_heads()`, train only the new head. This is the project's
established attributability step and it structurally cannot regress checkpoint 426.
**Gate:** accuracy on the 25.5% ambiguous subset must beat the min-advance baseline by a
margin larger than the seed spread. If the head only reproduces the min rule, Option A is
dead and Option B is the escalation - the information may simply not be recoverable
without spending sequence length on it.

### Phase 3 - consume it
- `music_xml_generator.py`: use the predicted advance when the head is present, fall back
  to `min` when it is not. One code path, flag-guarded.
- `audit_label_consistency.py`: re-enable `DURATION_DEPENDENT` on grand staves when the
  true advance is available from the converter (this works on *labels* and needs no
  model at all - it is available the moment Phase 0 lands).
- `cross_staff_consistency.py`: `check_measure_durations` and `check_barline_positions`
  become live on grand staves.
- Revisit the overfull rule now that its arithmetic is valid.

### Phase 4 - measure on the right benchmarks
Lieder, OLiMPiC scanned, and PDMX - **not** OSSQ, and say so in the write-up. Multiple
seeds, given the documented 0.23-0.69pp noise floor.

### Cost summary

| item | cost |
|---|---|
| Phase 0 instrumentation + measurement | hours, no GPU |
| Phase 1 build (9 files, all with direct precedent) | ~1 session |
| Sidecar regeneration for grand-staff corpora | ~1 conversion pass, no image work |
| Phase 2 frozen-core probe | 1 short GPU run |
| Phase 3 consumption + re-enabling audits | ~1 session, flag-guarded |
| Phase 4 multi-seed benchmark on 3 corpora | ~ a few GPU-hours |
| **Checkpoints invalidated** | **none** |
| **Images re-cut** | **none** |
| **Rhythm vocabulary changed** | **none** |

Compare Option B: everything above, plus a full corpus rebuild across seven converters,
plus a full retrain, plus an ONNX re-export, plus a break in `ned_score` comparability.
Compare Option D: a rewrite.

---

## 7. What would change this recommendation

- **Phase 0 says min-advance is nearly always right.** Then the exclusions are
  over-broad, and the fix is a narrow guard plus a corrected audit, not a head.
- **The Phase 2 probe shows the head just learns the min rule.** Then the onset genuinely
  needs its own sequence position and Option B is the answer, at Option B's price.
- **A decision to chase literature comparability** (TEDn against OLiMPiC/SMT baselines)
  outweighs incremental accuracy. Then Option D stops being a rewrite-for-one-defect and
  becomes a deliberate re-platforming, and should be planned as such rather than reached
  by drift.
- **Grand staves get dropped from the corpus.** They will not - `BENCHMARKS.md` measured
  -1.9 to -3.2pp across three corpora for removing them - but if that ever reverses, the
  blocker becomes moot.

## 8. Open questions

1. Does the advance target need to be a group-level record rather than a note-level
   field? The structured-heads machinery is note-keyed; an advance belongs to the
   *transition between* groups. Attaching it to the group's last symbol is the least
   invasive answer but should be sanity-checked against how `align_to_decoder_output`
   handles chord members.
2. What advance-duration vocabulary? The kern set covers notated durations, but a true
   gap can be any rational (a triplet against a dotted figure). LMX's answer - a *sum* of
   type tokens - suggests either allowing a short sequence of advance classes or accepting
   a quantization grid and measuring the residual in Phase 0.
3. Should a voice branch follow? Without one, a hand crossing staves mid-beam remains
   unrepresentable. Phase 0 should count how often that actually occurs in the Lieder and
   OLiMPiC repertoire before anyone designs for it.
4. `validation/ned_score.py` currently scores the six branches. If the advance head lands,
   does NED stay six-branch (comparable to history) or become seven-branch (correct)? It
   should stay six for comparability, with the advance reported as its own metric.

---

## Sources

- [Practical End-to-End Optical Music Recognition for Pianoform Music (Mayer et al., ICDAR 2024)](https://arxiv.org/abs/2403.13763) — [PDF](http://ufal.mff.cuni.cz/biblio/attachments/2024-mayer-m6741385360515472301.pdf)
- [ufal/olimpic-icdar24 — LMX implementation, vocabulary, TEDn](https://github.com/ufal/olimpic-icdar24) — [linearized-musicxml.md spec](https://raw.githubusercontent.com/ufal/olimpic-icdar24/master/docs/linearized-musicxml.md)
- [GrandStaff-LMX dataset](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5423)
- [Sheet Music Transformer++: End-to-End Full-Page OMR for Pianoform Sheet Music](https://arxiv.org/html/2405.12105v4)
- [End-to-end optical music recognition for pianoform sheet music (IJDAR)](https://link.springer.com/article/10.1007/s10032-023-00432-z)
- [MidiTok — Octuple, Compound Word, REMI tokenizations](https://miditok.readthedocs.io/en/latest/tokenizations.html)
- Local repository evidence: `homr/transformer/vocabulary.py`, `homr/music_xml_generator.py`, `training/omr_datasets/music_xml_parser.py`, `training/omr_datasets/staff_merging.py`, `training/omr_datasets/audit_label_consistency.py`, `homr/cross_staff_consistency.py`, `BENCHMARKS.md`, `CORPUS_CHANGELOG.md`, `DECODER_RHYTHM_ACCURACY_DESIGN.md`.
