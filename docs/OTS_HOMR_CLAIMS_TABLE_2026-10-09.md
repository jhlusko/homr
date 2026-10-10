# OTS-HOMR review-assisted lyrics and staff text: claims table (draft, 2026-10-09)

A draft for the owner's review, not an approval to publish and not a statement about the hosted service (OTS-HOMR is not
live in production; production is about 200 commits behind). Every number says what was counted, who judged it, and
whether it is **held-out** or **development** evidence. Extends `LYRIC_RELEASE_READINESS_2026-10-08.md`, whose "do not
advertise" list still applies.

**Evidence classes.** `H` = scored on pages no earlier step had touched, rules fixed beforehand. `D` = development: the
rule, threshold or idea was designed or tuned while looking at (some of) these pages, so the figure is an upper bound.
`O` = owner-judged. `A` = scored automatically against the native OpenScore source (a scan's printed marks can differ
from the encoded score, which understates availability). "Unsure" always counts as wrong.

## A. Claims the evidence supports

| # | Claim (suggested wording) | Number and denominator | Class | Where it comes from | Caveat that must travel with it |
|---|---|---|---|---|---|
| 1 | Nothing is added to your score until you accept it, and your original scan result is never overwritten. | Behavioural: the copy writer refuses stale, duplicate, occupied or non-lyric changes; the saved score is a derivative; one mixed end-to-end smoke test and a browser flow test pass. | tests | backend/frontend suites, `scanner-lyric-checkbox-flow.spec.cjs`, review-flow verification 2026-10-09 | Decisions are kept in the browser (not across devices); later token-level note corrections regenerate the score and drop lyrics. |
| 2 | Lyric regions are found with a calibrated confidence: at the default setting at least about 92 in 100 suggested regions on vocal pages contain lyric text (that is the lowest accepted confidence band; the accepted set pooled is higher). | Calibrated 0.922 for the lowest accepted band, raw confidence 0.7-0.8 (90% interval 0.86-0.96; the 0.8 and 0.9 bands are 0.926 and 0.987), from 218 pages / 89 scores and 150 owner-judged unmatched boxes; held-out 143 pages: 1,944 accepted vocal boxes, 53.8% match OCR-confirmed positives (a lower bound), projected 95.3%. | O+H (bounds) | `fusion-v1-fit-report.json`, `fusion-v1-heldout-evaluation.json`, `fusion-v1-extraction-requalification.json` | The accept line (raw 0.7) was an owner choice recorded as post-hoc. Held-out precision is a projection from the judged rate, not a new judgement. Instrumental pages: 0.8 accepted boxes per page (12 on 15 pages). |
| 3 | Suggested lyric text is usually a real lyric. | OCR + gate, 3,757 rows / 89 scores: 87.6% of proposed rows lyric-like; 96.7% of lyric-like rows are proposed; 94.0% of tokens in lyric rows read exactly. Clean strata (never in detector training or its pool): 91.5% / 98.2% on 556 rows; 85.5% / 97.1% on 549 rows. | A+D | `lyric-ocr-qualification.json` | The gate and detectors have partial exposure to these scores (strata are reported separately); "lyric-like" is a text test, not a correctness-of-placement test. |
| 4 | About 9 in 10 proposed syllable-to-note placements are right, and you confirm each one. | 109 of 120 correct (90.8%; 90% interval 85.6-94.3) on 10 fresh Lieder pages, rules not tuned on the sample. An earlier sample: 124 of 132 (93.9%), but its far-from-notes rule was chosen after seeing it. | O (H for the first) | `lyric-anchor-fresh-evaluation-v2.json` | Clean-score vocal pages only; syllable ownership across verses and exact note ownership are not claimed. |
| 5 | Most printed written directions and dynamic marks that are offered are read correctly. | Confirmation set, 101 judgeable rows: 88 right place (87%, 90% interval 81-92). Words 47 of 50 (94%) plus 3 right place with wrong text; OCR dynamics 34 of 42 (81%); classifier dynamics 7 of 9. Earlier set (before the filters): 138 of 170 (81%). | O+A (H for the confirmation set, filters fitted on the earlier set) | direction placement judgements (2026-10-09), `direction-confirm-20261009` | Pages were not chosen for cleanliness but are Lieder only; the figure covers rows with a proposed note; 15 offered rows had none then (2 after the alignment fix). |
| 6 | Marks the scanner can read but cannot attach to a note are still offered; you choose the note. | 29 of 31 such rows judged real and read correctly (94%; 90% interval 82-98). | O | `direction-row-judgements (1).json` | The note is not judged here (you choose it). 31 more rows to review on 27 pages. |
| 7 | The review shows a large share of the written directions and dynamics a score contains. | Native directions available as a row somewhere on the page: 169 of 251 (67%) and 121 of 218 (55%) on two sets (pooled 290 of 469, 62%). | A+D | `ground-truth-score.json`, `aligned-score.json` | Development: the unplaced rows and the alignment fix were designed on these pages. "Available" is not "placed correctly", and the native source encodes marks a scan may not print. About 38% never reach a reviewer. |
| 8 | Alignment of recognised notes to the score file is exact for almost all parts, and withheld otherwise. | 167 of 171 parts (97.7%) on 83 real pages; four parts withheld (one-rest count differences); never guessed. | A | `alignment-crash-audit-20261009/report.json` | Lieder scanner pages only. |
| 9 | Pages are processed in about a minute; a title page fails clearly. | Live GPU provider, direct HTTP, scanner-prepared pages: warm p50 58 s, p90 85 s, max 136 s (Lieder p50 27 s; OSSQ quartets p50 64 s); cold start 56-65 s; 3 simultaneous requests finish in 85-124 s; a no-staff page returns "no staff lines detected" in 4 s. All 31 notation requests succeeded. | H (single session) | `latency-run-20261009/results.jsonl` | 32 requests, one session, two cold starts; one container, so batches queue; not a load test; excludes the scanner's own queueing, storage and PDF rendering. |
| 10 | The provider and its models are pinned and checked. | The scanner refuses a provider whose detector, calibration, OCR, gate, dynamics-classifier or row-selection identity differs from its pin; the calibration is bound to the confidence extraction it was fitted on (requalified: every band moved at most 0.01, accept line unchanged). | tests | `scanner-review-text-identity`, `confidence_extraction.py`, runbook 4a | Local and test-provider evidence; the hosted service is not live. |

## B. Claims not to make

| Do not say | Why |
|---|---|
| "Reads all lyrics" / any recall or 100% recognition figure for lyrics. | The 95% conditional lyric-retention measurement (readiness doc) is still unclosed; counts of suggestions are not recall. |
| "Automatically places lyrics, directions or dynamics." | Placement is a proposal; roughly one in ten (lyrics) to one in eight (directions and dynamics) is wrong, and every one needs a reviewer. |
| "Reads dynamics" without qualification. | OCR reads p and pp well and f, ff, mf poorly; the classifier adds a few f marks (precision 7 of 9, recall low); p, pp, ppp are deliberately left to OCR. Hairpins are not read at all. |
| "Finds pedal markings, tempo words in any language." | Vocabulary is Italian, German and French terms listed in code; "Ped." depends on OCR; German tempo phrases were missed. |
| Any precision or recall for instrumental, orchestral or non-Lieder scores. | Evidence is Lieder (voice and piano); instrumental pages were used for latency and negative controls only. |
| "Handles every page" / failure-free. | About 5% of audited pages are not scoreable (cover, text-only, white-on-black notation: 4 of 83); two inverted pages need an offline fix that is not shipped. |
| Anything that treats the development figures (7, parts of 2-5) as untouched test results. | Rules and filters were fitted after seeing judgements; a confirming fresh set exists only for rows 5 and part of 2. |
| "Zero lyric suggestions on instrumental pages." | True for the 25-page replay after backend filtering (readiness doc); the 17 instrumental pages in the latency run returned 35 OCR lyric rows on 10 pages that were not classified as proposed or withheld. Check before claiming. |
| "In production", "available", "live". | OTS-HOMR is not deployed to production. |

## C. Draft public copy (built only from section A)

> **Lyric and staff text review (preview).** After a scan, review the lyrics, written directions (cresc., rit., dolce...)
> and dynamic marks (p, mf...) found on the page, beside the original. Correct the text, choose the note each one goes
> on, and add what you accept to your score: nothing is added until you accept it. Suggestions can miss marks, misread
> text, or confuse printed instructions with sung words, and about one placement in ten is wrong, so your review matters.
> Marks the scanner cannot attach to a note are still offered; you choose where they go. Tested on vocal scores with piano
> (OpenScore Lieder); other kinds of music have not been evaluated.

## D. To close before publishing

1. A fresh, untouched set for the direction and dynamics filters and the unplaced rows (the only confirming set for the
   filters is the confirmation set; recall, rows 7, is development).
2. The 95% conditional lyric-retention measurement (readiness doc) or an explicit decision to drop that claim.
3. Classify the 35 OCR lyric rows seen on instrumental pages (proposed or withheld) before any "no false lyrics on
   instrumental scores" statement.
4. A latency note refreshed on the frozen candidate if anything in the pinned digest changes (any edit to the ten
   provider modules or the extraction files changes it).
5. Product decision: public wording for "Lyric and staff text review", the Lieder-only scope, and whether examples show
   the unplaced-row flow.
6. Deployment approval, which also needs production brought current (about 200 commits) and the provider pins set on the
   production scanner.

## Evidence index

Under `homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/`: `latency-run-20261009/`,
`alignment-crash-audit-20261009/report.json`, `direction-final-20261009/` and `direction-confirm-20261009/` (judgements,
keys, native scoring), `fusion-lyrics-calibration-corrected-20261009/` and `fusion-lyrics-test-corrected-20261009/`
(requalification), `lyric-prepared-full25-20261008/` (replay). In OTS: `services/ots-homr-modal/calibration/`
(`fusion-v1*.json`, `lyric-ocr-qualification.json`, `lyric-anchor-fresh-evaluation-v2.json`), `docs/SCANNER_OTS_HOMR_MODAL_RUNBOOK.md`.
Dated narrative: homr `RUNLOG.md` entries of 2026-10-08 and 2026-10-09.
