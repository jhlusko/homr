# The OTS-HOMR pipeline: what it does, what ships, and what does not

**Status:** authoritative. Supersedes the write-ups listed under
[Superseded documents](#superseded-documents), which are retained as published-artifact
copies and are no longer maintained.
**Last verified against deployed code:** 2026-09-10.

Everything below distinguishes three different claims, because conflating them is what
made the earlier write-ups misleading:

- **Measured** — a held-out number with support and an interval.
- **Built** — code exists and is tested.
- **Shipping** — it runs in the deployed provider on a real scan.

A feature can be measured and built and still not ship. Several are.

---

## 1. Deployment topology (read this first)

The `ots-homr` engine is **not** this repository at HEAD. It is a frozen, SHA-pinned
source archive plus patches:

| piece | value |
| --- | --- |
| source archive | `services/ots-homr-modal/source/homr-997b70-source.tar.gz` |
| archive SHA-256 | `819eca26f26e977e2a376d03c8aa4ddaf1f631db69a87ccc9d1ad943f75b4de9` |
| homr commit | `997b70a88c9f945b632f4129b036b2cd0972b2a3` |
| runtime patch | `patches/homr-997-runtime.patch` (~1,200 lines) |
| review patch | `patches/homr-structured-review.patch` |
| extra module | `homr_runtime/tuplet_repair.py` → `/opt/homr-tuplet-repair.py` |
| service revision | `ots-ots-homr-provider-v4` |

**The pinned commit `997b70a8` does not exist in this repository**, and is not
recoverable by `git fetch`. It was rebased or force-pushed away. The pinning itself is
correct — production is reproducible from the archive — but you currently cannot diff
production against source history, which is how the write-ups drifted out of date
without anyone noticing.

Only two modules exist here but not in the archive: `tuplet_repair.py` (shipped
separately, see above) and `text_detector_config.py`.

## 2. Feature status

| feature | built | ships | why |
| --- | --- | --- | --- |
| Arm A scan-adapted core | yes | **yes** | §3 |
| Rare-numerator continuation | yes | **yes** | §3.1 — the older write-ups say otherwise and are wrong |
| Phase 1 cross-staff rhythm rerank | yes | **yes** | §4.1 |
| Tuplet repair | yes | **yes** | via runtime patch, not the archive |
| Structured heads: beam, advance, stem, tie, slur | yes | **yes** | §5 |
| Structured head: dynamics | yes | **no** | macro-F1 .105 (935) — withheld, §5 |
| Cross-staff tier-1 repair (key/time/articulation) | yes | **yes, new** | §4.2 — was log-only until 2026-09-10 |
| Cross-staff position-divergence repair | proposer only | **no** | no applier exists, §4.2 |
| Stage C staff-context two-pass decode | yes | **no** | weights not pinned or shipped, §4.3 |
| Score profiles (clef/layout checks) | yes | **no** | no channel to supply one, §6 |
| Stage 3 text: lyrics / dynamics / measure numbers | yes | partial | one detector per run, §7 |
| Stage 3 text: tempo, staff text, expression, fingering | detector only | **no** | not ready for always-on output |

## 3. Recognition: Arm A

Arm A is a six-epoch continuation of the common scan-adapted checkpoint, seed 42. It
retains OSSQ and v4 Lieder scans and uses PDMX-only replay.

Against pinned upstream `pytorch_model_426`, free-running, paired bootstrap over staves:

| held-out set | staves | 426 | Arm A | Δ (95% CI) |
| --- | --- | --- | --- | --- |
| OSSQ scans | 792 | 91.49 | 95.05 | +3.56pp (+2.46, +4.63) |
| PDMX held-out | 3,349 | 83.97 | 88.08 | +4.11pp (+3.19, +5.09) |
| Lieder v4 holdout | 300 | 87.23 | 94.54 | +7.30pp (+4.38, +10.37) |

Per-branch on OSSQ: pitch 90.41→94.55, rhythm 87.50→93.11, lift 90.00→95.50,
articulation 92.65→94.93, slur 93.15→95.30, position 95.25→96.91.

**Caveat that is easy to drop and must not be.** 426 predates the time-signature
numerator and naturals vocabulary. Comparisons suppress `timeSignatureBeats_` on both
sides, but 426 has no `N` lift class, and a missing early class shifts later positional
tokens. This is end-to-end evidence of pipeline difference, not attribution of gain to
training data.

Attribution arm (numerator-neutral): Arm A retains the scan result while recovering
PDMX (OSSQ −0.09pp ns, PDMX +1.10pp, Lieder +0.13pp ns against `scans_v4`). Arm B, which
removed scanned OSSQ entirely, lost 3.89pp on OSSQ — the real scanned crops supply the
OSSQ gain, not the replay change.

### 3.1 Rare numerators

The deployed provider is the **rareNum** bundle: `modal_app.py` describes itself as the
"private rareNum OTS-HOMR provider", the idempotency fingerprint carries
`preprocessingRevision: 'ots-ots-homr-rare-num-v1'`, and `onnx_core.py` exposes
`load_rare_num_sessions`.

`homr-best-model.html` §5 states the rare-numerator continuation is *not* the default.
**That is stale.** It ships. The caveat behind that sentence remains true and is why it
is called out here: the two-seed targeted experiment repairs 5/x and 12/x metre
recognition, and seed 7 carried a small significant Lieder cost.

## 4. Cross-staff work

Three distinct mechanisms, routinely confused with each other.

### 4.1 Phase 1 rhythm rerank — ships, on by default

`enable_phase1_rerank=True`, active whenever `selected_staff < 0`, which is what the
service passes. It forks the narrowest-margin **rhythm** decisions into full alternate
decodes and picks the candidate whose cumulative barline positions best agree with
sibling staves.

Measured: 428→339 structural findings across 200 pages (−20.8%); 81/899 systems
improved; zero pages worse.

It is gated to systems that already show a `check_barline_positions` or
`check_measure_durations` finding, because forking is a full extra decode per staff.

**It does not vary the time-signature branch.** A staff that decodes a different metre
from its siblings is not something this can fix, and this was the source of a real
production confusion.

### 4.2 Tier-1 repair — ships as of 2026-09-10 (new)

`cross_staff_consistency` has always detected `time_signature_mismatch`,
`key_signature_mismatch`, measure-count and dangling-slur disagreements, and
`cross_staff_repair` has always built majority-correction proposals with matching
`apply_*` functions. Nothing ever called the appliers. Findings and proposals were
written to a log and the page shipped uncorrected.

`Config.cross_staff_repair` (env `HOMR_CROSS_STAFF_REPAIR`, default on) now applies, in
`_apply_cross_staff_repairs`:

- opening **key** and **time** signature majority corrections (`propose_repairs`);
- motif-corroborated **articulation** corrections;
- **carried-forward key signatures** (insertions, applied back-to-front so positions
  stay valid).

Guards, all pre-existing in the proposers and deliberately kept:

- a system needs **at least three** staves with a decoded result;
- `propose_majority_correction` returns nothing on a tie, or with fewer than two staves
  stating a signature at all;
- `apply_proposal` refuses a proposal built against a staff that has since changed;
- a minority staff is only ever moved **onto a value another staff actually read** —
  nothing is invented;
- any exception leaves the system exactly as decoded.

`propose_majority_position_corrections` has **no applier** and remains diagnostic.

**This shipped without the 200-page benchmark `enable_phase1_rerank` carries**, at the
maintainer's explicit instruction. Polymetric and polytonal music is real; the
three-staff minimum and strict-majority rule are what stand between this and damaging
it. If a regression appears, `HOMR_CROSS_STAFF_REPAIR=0` restores log-only behaviour.

### 4.3 Stage C staff-context decode — does not ship

`enable_staff_context` pools each present voice's hidden states, runs the trained
`StaffContextTransformer` across the system's voices, and re-decodes every voice with
its own context vector. It is the most general of the three mechanisms — it attends
across voices rather than reranking rhythm alone.

Off by default and **blocked**, not merely unbenchmarked: `staff_context_weights` is
required when it is on, the `phase24-staff-context-weights` release is not present in
this repository, not in `pins.py`, and not in the service image. Enabling it needs the
artifact downloaded, SHA-pinned, added to the model volume, and threaded through
`ProcessingConfig`.

## 5. Structured heads

Frozen-core projections predicting beam levels, hooks, ties, advance delta, stem
direction, slur event/side, and dynamics. Held-out support from the mixed
GrandStaff/Lieder/PDMX run:

| head | metric | support |
| --- | --- | --- |
| ties | macro-F1 .844 | 16,693 |
| stem up/down | F1 .811 | 217,631 |
| slur span | F1 .772 | 3,400 |
| slur side | macro-F1 .723 | 3,121 |
| advance (nontrivial) | macro-F1 .751 | 166,624 |
| **dynamics** | **macro-F1 .105** | **935** |

The provider hardcodes `['beam', 'advance', 'stem', 'tie', 'slur']`. Dynamics is
withheld on the .105 result and is prohibited at the provider boundary; see the text
fusion design for the fail-closed rule.

## 6. Score profiles — built, no channel

`check_clefs_against_profile` and the §7.2 layout-deviation report only run when a
`ScoreProfile` is supplied. The service constructs `ProcessingConfig` positionally and
leaves `score_profile=None`, and the provider option surface is a single
`textDetector` field validated by `Object.keys(value).length === 1`. There is no way to
pass a profile even if you had one.

For a string quartet — where the parts and their clefs are known before recognition
starts — this is the most obviously applicable dropped capability. Design in
`docs/design/SCORE_PROFILE_CHANNEL.md`.

## 7. Stage 3 page text

On 307 held-out pages: lyrics 66.7/94.1/78.1 F1 (3,555 boxes), dynamics 75.9/93.9/84.0
(429), measure numbers 75.5/94.9/84.1 (78). Scan supervision moved lyric patch IoU from
.581 synthetic-only to .966 — patch IoU is **not** a release metric; this project has
twice seen respectable patch IoU reverse at full-page box evaluation.

`textDetector` selects exactly one of `lyrics | non-lyric-text | none`, so lyrics and
non-lyric classes cannot both run in a single pass. Fusing them is designed in
`docs/private/SCANNER_OTS_HOMR_TEXT_FUSION_DESIGN_2026-09-10.md` (OurTextScores).

## 8. Data: how the corpus was built

The original OSSQ pipeline paired scan crops with symbolic material by page and system
index, across editions whose pagination differs. Guards passed — a record existed, part
counts matched, tokens parsed — and the semantic pair was still wrong.

900-staff audit across 9 held-out scores:

| | before | after |
| --- | --- | --- |
| staves with >50-point collapse | 56.7% | 7.9% |
| mean scanned accuracy | 46.8 | 89.9 |
| median per-staff drop | 76.7 | 0.0 |

The diagnostic that settled it was qualitative: a model reading matched the scan exactly
while disagreeing with its assigned target. That is a label join failure, not a model
failure.

Rebuild: start from whole-score MusicXML, recover staff-local streams positionally,
align by visual bar-count/geometry, independently align by reverse fingerprinting from
recognition readings, and admit to evaluation **only where the two agree**. Reverse
matching is more accurate and is firewalled out of evaluation anyway, because filtering
an evaluation set by agreement with the model measures the model against itself.

| verdict | pairs | used for |
| --- | --- | --- |
| consensus | 3,968 | evaluation + training |
| arbitrated | 258 | training only |
| reverse | 3,656 | training only |
| unarbitrated | 308 | training only |
| phantom | 68 | neither |
| rejected | 391 | neither |

A parser bug that raised on `<octave-shift>` and `<clef-octave-change>` was discarding
94 of 111 scores whole: 3,214 → 4,587 clean pairs, 165 → 234 scores. The correction is
`written − sounding` and 8va exports as `type="down"`; ~1% of notes are affected, so no
aggregate metric can see the sign — running it both ways gave byte-identical corpus
statistics. Human review settled it at 22/25 octaves correct, zero octave errors.

### 8.1 Known blind spot

Consensus cannot catch the case where both methods agree and both are wrong.
`IMSLP183806-sys1-v1` is confirmed: bar-count reads three measures of rests, content
reads five measures for a three-measure crop, the reviewer judged neither correct, and
the pipeline classifies the pair `consensus`. The mechanism is specific — consensus is
decided from the voice-0 reading and applied to every staff in the system, so a
system-level agreement does not validate each voice's label. **Nothing currently checks
per-voice content.** Until it does, the evaluation set carries an unmeasured error floor
of this kind.

### 8.2 Open problem: the implied-tuplet discard rule

The builder discards any pair containing a bar that overflows its metre: 417 pairs
across 144 scores, ~10% of the clean corpus. Quarantining rather than deleting them
showed the rule is doing two opposite jobs and one of them wrongly.

Of 546 offending bars: 161 tuplet-shaped, 80 exactly 2× or 3× the bar (a missing
barline — correctly rejected), 76 a second metre in the crop, 229 unexplained. So
"recover the 417" would have been the wrong instruction.

Worse, the test is unsound for 89% of them. 371 of 417 are grand staves; the kept corpus
is 44.9% grand staff. Discard rate is 2.0% for single staves and **16.5% for grand
staves — 8.4×**. `group_into_chords` takes the *minimum* duration across a chord, so on
a grand staff a bar where the hands play different rhythms is neither their sum nor
either hand's length. The corpus audit already refuses to run duration-dependent checks
on grand staves for exactly this reason; the discard rule calls the same arithmetic with
no such guard.

## 9. Hard-won rules

- A benchmark built from labels you rebuilt yourself measures your labels as much as
  your model. Checkpoint 448 scored +0.88pp on rebuilt Lieder labels (ns, and more
  staves got worse than better) while an untouched corpus showed **−1.23pp**. It was not
  shipped. An earlier write-up reported a regression as +3%.
- Do not add synthetic replay to a scan-adapted model by default.
- Do not trust patch IoU as page quality.
- Do not trust unit tests to prove export — the first beam implementation predicted
  beams and discarded them during MusicXML export.
- Invalidate duration caches after tuplet rewrites.
- Do not report rare-class macro scores without support.
- Never recompute an evaluation target with a model whose score will be reported
  against it.
- An automated pipeline that errs conservatively looks healthy from the inside: all four
  automated judgements a human reviewer overturned were wrong in the same direction —
  discarding good data.

## Superseded documents

Retained because they are local copies of published artifacts, and because the galleries
they embed still work. Not maintained; where they disagree with this document, this
document is correct.

| file | superseded by |
| --- | --- |
| `docs/writeups/homr-devs.html` | §2, §4, §5, §7 |
| `docs/writeups/homr-best-model.html` | §3 — its §5 rare-numerator claim is stale |
| `docs/writeups/homr-dataset.html` | §8 |
| `docs/writeups/train2-writeup.html` | §8, §9 |
| `docs/writeups/omr-findings.html` | §3 |

`ENSEMBLE_TRANSCRIPTION_DESIGN.md` and `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` remain the
design record. This document is the state of the pipeline; where they describe intent
and this describes what ships, both can be true at once.

To view the write-ups with their live compare embeds, serve the directory over HTTP:
`python3 -m http.server 8123` from `docs/writeups/`.
