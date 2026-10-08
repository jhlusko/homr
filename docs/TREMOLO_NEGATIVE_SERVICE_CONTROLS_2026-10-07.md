# Fresh corrected-candidate negative controls — 2026-10-07

All five predetermined real-scan controls completed fresh image inference through
HTTP, with distinct core-only and accepted-heads workers, explicit timing off/on.
All ten candidate comparisons are byte-identical XML; decoded tokens, written-note
inventories and measure counts are unchanged. Native source audits show no measure
regressions. These historically exposed pages test safe abstention, not independent
corpus accuracy. Newer object caches were not used.

Controls: Lieder IMSLP89026 p002 (changing meter),617260 p006 (single-note tremolo /
grace),154289 p001 (grace/triplets),558609 p006 (ordinary chords); OSSQsq8071278 p001
(single-note tremolo). Selection was frozen before inference. Source-category
coverage does not prove every feature was correctly decoded; five runtime guard
unit tests separately exercise explicit decoded conditions.

| Corpus controls | Rhythm errors, off = on | Pitch errors, off = on | Accidental errors, off = on |
|---|---:|---:|---:|
| Lieder, four pages | 39 / 1377 | 27 / 1301 | 7 / 1301 |
| OSSQ, one page | 21 / 455 | 3 / 383 | 1 / 383 |

Values hold for each distinct inference candidate; do not double support by adding
core and heads over the same pages. These are baseline residual errors, unaffected
by the rule. Positive development-page results remain separately Lieder30→10/357,
OSSQ46→2/380 for each candidate; no wider test-corpus improvement claim is added.

Local HTTP wall times off→on:95.24→101.55s,138.83→150.85s,65.67→71.98s,
95.63→117.59s,229.46→212.12s in the page order above. Arms ran concurrently;
times include both worker candidates, exclude OCR/text fusion and use CPU.
They do not establish GPU latency or precise policy overhead/280-second margin.

Artifacts:
`/home/jhlusko/workspace/homr-artifacts/local-tremolo-negative-pages-20261007/`.
Frozen pages/selection/candidate manifest, both arms' responses/XML/logs, comparison,
per-page native audits, aggregate native-score-report, watcher and audit-completion
receipts retained. Audit exit0,10/10 unchanged, distinct head routing passed.
All jobs/watchers finished.

## Parallel release work

A separate executable Modal candidate recipe is prepared in OTS:
`services/ots-homr-modal/modal_tremolo_candidate.py`, with isolated on/off app/revision
names and model cache, verified source/service inputs and explicit worker masks.
It uses existing private fp16 pins and is marked unqualified against the experimental
fp32 evidence. It was tested offline using fake Modal resources; never built/deployed.
The experimental fp32 build specification remains separate. Shipping entrypoint,
model pins, production configuration and backend environment remain unchanged.

C1 semantic data preparation created241 ground-truth-positive regions across100
Lieder scores:45 directions +196 lyrics. Preserve24 earlier score exclusions;
score-disjoint internal training/calibration73/27 scores (178/63 regions), not a
formal untouched test. Missing title/credit and OSSQ non-lyric coverage is explicit;
unannotated regions are not negatives. All45 direction boxes lack source text,
scan-only OCR now completed all241 crops:45 directions nonempty,163 lyrics nonempty
and33 lyrics empty (16.84%), zero exceptions. All53 prior results retained and188
remaining crops processed in four one-thread workers. No source transcription or
roles entered OCR. Empty lyrics remain in evaluation; OCR confidence does not prove
transcription accuracy. No classifier trained or semantic threshold chosen.
Review: http://localhost:8765/c1-semantic-gt-packet-20261007/ .
Packet receipts: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/c1-semantic-gt-packet-20261007/`.

OCR receipt: `ocr-scan-only-sharded/receipt.json` beneath the packet above.

Next: address data coverage for C1 and run a separate abstaining semantic
pilot with score-disjoint evaluation and actual false-attachment measurement.
The prepared fp16 provider still needs model-loaded GPU/OCR/text-path accuracy and
latency qualification; automatic lyric fusion remains disabled. E3 examples use
settled candidates after these gates. No timing-head/beam promotion is implied.

## 2026-10-07 — Private fp16 positive and negative CPU qualification complete

Fresh private-fp16 negative controls finished actual exit0: fivepages × two genuine
plain/heads candidates, explicit off/on, all10 XML comparisons byte-identical.
No token/written-note/measure changes, zero native measure regressions, correct
worker routing. Same aggregate residual errors for each candidate: Lieder rhythm
39/1377,pitch27/1301,accidental7/1301; OSSQ21/455,3/383,1/383. No duplicated support
by adding the two candidates over identical pages. Controls historically exposed,
not independent corpus accuracy. Exact private graph hashes verified before run.

Negative CPU HTTP seconds off→on in fixed page order:26.30→37.26,34.08→45.53,
19.63→22.88,29.81→34.81,60.03→68.09. Both workers; OCR/detectors/fusion off.
Single sequential runs cannot isolate precise overhead or establish production
GPU280-second margin. Prior private positive results remain Lieder30→10/357 and
OSSQ46→2/380 with unchanged pitch/accidentals and zero measure regressions.
The private recognition graph compatibility gate now has direct local evidence;
full GPU/calibrated text-path accuracy and latency gates remain open.

Final comparison/native-score-report/watch-status/run-exit receipts:
`/home/jhlusko/workspace/homr-artifacts/local-tremolo-private-fp16-negatives-20261007/`.
All compute jobs and their watchers finished; only review serving persists.

Next priorities:1) assemble and qualify the complete calibrated OCR/detector/text
path, with automatic lyric attachment disabled pending C1;2) broaden C1 annotated
non-lyric training coverage (current score-grouped text sweep found no useful
zero-observed-direction-error policy);3) actual model-loaded GPU/provider build
and280-second latency check;4) settle E3 examples against qualified candidates.
Keep deployment/publication separate from these local preparations. No model pin,
backend environment, production deployment or push changed. No running-job ETA.
