# Lieder corpus: what changed, when, and what it cost

Written because a bisect was about to test the wrong variable. The plan was to strip the
model-derived half of the corpus and see whether OSSQ recovered toward 447's 94.03 —
but **447 was itself trained on a corpus that was 50% model-derived**, so that arm
tests a recipe 447 never had, and a null result would have proved nothing.

Every corpus version, what produced it, and what it scored. Independent = OSSQ, 792
staves, never relabelled by this project. Ours = the Lieder held-out split.

## Lineage

| index | built | pairs | composition | checkpoint | independent | ours |
|---|---|---|---|---|---|---|
| `imslp_train_index.txt` | 08-25 08:06 | 3,353 | 3,353 `stage2_pairs_out` | — (pre-rebuild) | — | — |
| `imslp_train_index_v6.txt` | 08-27 14:02 | 7,273 | 3,608 `clean_v2` + 3,665 `reverse_v2` | **447** | **94.03** | 91.81 |
| `imslp_train_index_v7.txt` | 08-27 21:24 | 7,016 | 3,327 `clean_v5` + 3,689 `reverse_v3` | **448** | 92.80 | 92.69 |
| `imslp_train_index_v8.txt` | 08-28 01:14 | 6,989 | 3,548 `clean_v5` + 3,441 `reverse_v3` | **449** | 92.43 | 94.53 |

Baseline `426` (five original corpora, no Lieder scans): independent 91.70, ours 83.25.

**The trend is monotonic and opposite on the two axes.** Independent 94.03 → 92.80 →
92.43, back toward the 91.70 baseline. Ours 91.81 → 92.69 → 94.53. Every corpus change
since 447 has been rewarded by our own metric and punished by the only corpus that can
see us.

## What landed between 447 and 448

Three commits, all on 08-27, between v6 (14:02) and v7 (21:24).

| commit | change | effect on the corpus |
|---|---|---|
| `7933f20` 15:10 | Represent time signature numerators; act on review findings | vocabulary 260 → 279 rhythm tokens; labels gain `timeSignatureBeats_*` |
| `43b9db2` 16:18 | Measure the content gap, **exclude implied tuplets** and `IMSLP405017` | **−417 pairs** quarantined as "overfull"; one score excluded entirely |
| `9407e64` 16:46 | Stop `MeasureCutter` dropping the numerator | 77 of 416 labels regained a stated numerator |

Plus the pair sets themselves were rebuilt: `clean_v2` → `clean_v5`, `reverse_v2` →
`reverse_v3`.

## Ranked suspects, and what is already ruled out

**Ruled out — do not spend a run on these:**

- *Model-derived (`reverse`) pairs.* 447, 448 and 449 are all ~50% model-derived. This
  cannot explain a difference between them.
- *The arbitration rule.* Directly ablated: 449 reverted it to bar-count labels and
  OSSQ got **worse** (92.80 → 92.43), not better.
- *Training schedule.* 451 reran v8 at 2 epochs instead of 12: OSSQ 91.83, below 449's
  92.43. Fewer epochs did not preserve general performance, it just trained less.
- *The metre vocabulary, at inference.* Stripping the numerator slots from predictions
  recovered 0.03pp of a 1.23pp gap.

**Live suspects, in order:**

1. **The overfull exclusion (`43b9db2`).** It removed 417 pairs that 447 *had*. We have
   since shown the rule is unsound for the pairs it removes: **371 of the 417 are grand
   staves**, where `group_into_chords` takes the *minimum* duration across a chord, so
   a bar where the hands differ is neither their sum nor either hand's length — the
   corpus audit refuses to run any duration check on a grand staff for exactly this
   reason, and the discard rule calls that arithmetic anyway. Grand staves are
   discarded at 16.5% against 2.0% for single staves, 8.4x the rate. This is a change
   we identified as wrong and never reverted.
2. **The `clean_v2` → `clean_v5` rebuild.** Of the systems both corpora contain, 24%
   changed: 12.3% got a different bar count, 11.8% different content at the same bar
   count. 100 of 364 changed labels are *exactly the old label of a neighbouring
   system*, with negative offsets outnumbering positive 3:1 — phantom system detections
   displacing everything after them. 10 scores show a constant-offset run to the end of
   the score; 247 of 3,548 clean pairs (7%) belong to them.
3. **The metre numerator tokens in training.** Exonerated at inference, not in
   training: the embedding and output matrices grew by 19 rows, and 6.0% of the labels
   that state a numerator contradict their own bars.
4. **`reverse_v2` → `reverse_v3`.** Never examined. Same size (3,665 → 3,689 in v7),
   but rebuilt with the rest fix and the widened `MAX_SPAN`.

## The structural problem behind suspect 2

Consensus is the rebuild's central safeguard: a label is evaluation-admissible only
where two independent methods agree. **They are not independent.** Bar-count alignment
and reverse fingerprinting both consume the same `detect_imslp_systems` output — they
cut the same crops from the same boxes. A phantom system is inherited by both, both
agree, and the pair is stamped `consensus`.

Consensus validates *which measures a crop gets*. It cannot validate *the crop set*.
Every guard in the pipeline sits downstream of the one thing that is wrong.

## Rule for choosing the next arm

Vary one thing that **447 actually had** and 448/449 do not, or vice versa. Check this
table first. The arm launched before this document existed — v8 with the model-derived
half removed — fails that test, and its result should be read as "what does clean-only
do", not as "did the pseudo-labels cost us 447".
