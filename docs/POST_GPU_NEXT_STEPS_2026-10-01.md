# Local follow-up after the 2026-09-29 random-start experiment

## Current local follow-up — 2026-10-01

### 2026-10-03 — reference advance A/B complete

Reference advance reduces moved-onset events 470 → 412 on OSSQ and 15,855 → 5,252 on Lieder, on common inventory-preserved measures. No newly lost correct inventories. There are 39 Lieder exact-count regressions; the count does not measure distance. A grace-cursor probe removes four genuinely worse cases in two scores. A separate synthetic probe exposes zero-duration rounding when advance changes XML divisions. See `REFERENCE_ADVANCE_ABLATION_2026-10-03.md`. All jobs complete on one CPU thread; no remaining ETA.

Next: narrowly tested printed-duration division allocation and future-build grace cursor corrections; repeat reference qualification before evaluating predicted advance. No model promotion or published pin change. Text-fusion/C1 and new full-page references remain open.

### 2026-10-03 — OSSQ/Lieder timing census complete

No PDMX was read. All 39,137 OSSQ and 3,867 published-index Lieder labels were available. Among canonical references for indexed score identities (85 OSSQ / 212 Lieder), the current formula loses full-rest timing on 356 / 132 events and non-integral reciprocal-tuplet timing on 2,031 / 46 consistent-source events. These categories affect 21 / 22 score identities and are 0.233% / 0.065% of printed timed events; full canonical score events are not interchangeable with retained crop exposures. Relevant historical OSSQ parser functions match current source. See `REAL_TIMING_LABEL_CENSUS_2026-10-03.md` for precise definitions, larger-collection counts and local artifacts.

Whole-score reference roundtrips complete in 809.8 seconds on one CPU thread. Published-score success coverage is 75/85 OSSQ and 212/212 Lieder; ten OSSQ exports reject detachedLegato. With advance output disabled, onset changes occur in inventory-preserved measures in 30/75 successful OSSQ scores and 160/212 Lieder scores. These fixed-staff/measure diagnostics are not voice-aware equivalence or model error rates. Eleven tests pass; both tmux jobs complete, no remaining ETA. Next: bound a ground-truth advance A/B to locate lost grouping timing; inspect representative real-source rests/duplets and make narrowly tested future-build representation corrections before changing corpus pins or training. Text-fusion/C1 remain independent work.

### 2026-10-03 — cached timing ablation complete

Exact XML replay passes for both stacks. Late tuplet/beam/tie/slur-crossing repairs do not change pitch/rhythm edits on Lorelei. The canonical-reference label roundtrip reproduces the five shared vocal duration errors: six-quarter whole-bar rests collapse to `rest_1`, and 2:3 eighth-note duplets round to `note_5`. Piano ordering changes also show that ordered edits can penalize unchanged event inventory. No runtime or published label policy changed. See `PAGE_TIMING_LABEL_AUDIT_2026-10-03.md` for local evidence, limitations and the `rest_0` duration-unit warning.

Next: measure these representation problems across frozen labels and canonical references, establish historical parser provenance, then implement a narrowly tested future-build correction and strengthen onset/meter-aware page scoring. Do not enable a blanket sole-rest rewrite or retrain before that audit. All jobs complete on one CPU thread; six tests pass, no remaining ETA. Full-page untouched references/OSSQ alignment and text-fusion/C1 still remain open.

### 2026-10-03 — reference audit and paired original-page diagnostic complete

See `FULL_PAGE_DIAGNOSTIC_2026-10-03.md` for the current result and local reproduction paths. The reference audit finds no verified untouched locally scoreable pages: four certified Lieder pages are from Lorelei, explicitly previously bar-count evaluated; OSSQ scan alignment is missing. A deterministically selected Lorelei page was scored with matched accepted/candidate float32 exports. Candidate pitch edit error improves 1.955% → 1.397%, while rhythm/event edits worsen 3.599% → 4.370%. Accidental projection differences are deletions, not substitutions. This one-page diagnostic does not apply the crop-token Lieder regression gate or justify promotion.

Both stacks complete in about 118 CPU seconds on one thread. All tmux jobs are complete. Next local task: cached-symbol/XML-only ablations of shared 12/8 whole-bar-rest and fractional-duration errors, retaining original results. Then certify genuinely new full-page references and OSSQ scan alignment. Detector-fusion calibration/C1 lyrics remain parallel release work; no new training or production change.

### 2026-10-03 — integrated runtime correction qualified locally

New decoder exports explicitly support empty continuation context and carry `homr.decoder.empty_continuation_context=1`. The runtime selects that behavior for greedy, margin and prefix decoding; existing unmarked graphs retain the legacy path. Actual runtime qualification, without the diagnostic monkeypatch, passes all 22 strict crop parity checks and all 12 head projection checks. Margin/prefix/fork checks pass with a constant 1,280-token cross-attention cache across 616 steps. Eight focused unit tests pass.

Repeated full-page functional smoke passes: Lieder 116.2 CPU seconds, OSSQ 108.3, both with nonempty well-formed XML and zero automatic dynamics. One CPU thread; all tmux jobs completed, `FOLLOWUP_EXIT=0`. Evidence and hash-verified tested source are at `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-integrated-runtime-20261003/`. Shipping graphs and provider pins were not changed. Local source snapshots include pre-existing uncommitted pipeline work, so the correction commit is not by itself a clean-checkout reproduction of every tested policy.

Next: audit a full-page ground-truth evaluation manifest against training and model-selection score identities, then compare complete accepted/candidate pipelines. Continue detector-fusion area/recall calibration and C1 lyrics independently. Untouched full-page accuracy, GPU/HTTP timing, final example regeneration and promotion remain open; no additional training is justified by the resolved export mismatch.

### 2026-10-03 — corrected diagnostic passes strict export parity

The ONNX one-token continuation workaround changes cross-attention cache semantics. Eager decoding sends empty continuation context, but the exported graph rejected its zero-length head-split reshape. An isolated graph correction makes the eight-head dimension explicit and permits literal zero dimensions; a diagnostic runtime supplies empty context. All 22 pilot crops now match eager recognition and complete serialized notation exactly, with all encoder, cached-eager and head-projection checks passing. No tolerance was relaxed.

Evidence: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/joint-context-full-parity-20261003/`. Reproduce with `training/onnx/audit_joint_context_cache.py --repair-empty-reshape --full-parity` and the source/export/output directories recorded in the release sequence. Job complete on one CPU thread. Next: integrate the correction with graph capability metadata so older graphs remain compatible, cover greedy/margin/fork paths, then repeat full-page smoke. The shipping runtime is unchanged; the original page smoke does not qualify the corrected graph. New-score accuracy, GPU/HTTP latency and text-fusion/C1 remain open.

### 2026-10-02 — export parity is a release blocker

The epoch-27 public-joint checkpoint has isolated ONNX exports and a bounded parity audit. Encoder and head logits are numerically close, but greedy recognition differs on one of 22 pilot crops at a low-margin rhythm decision. Keep the accepted stack as the runtime baseline and do not publish candidate advantages as deployment results until the cause is understood or the release gate explicitly accepts this behavior. Full-page functional smoke completed on one Lieder and one OSSQ page; it is a validity check, not an accuracy benchmark. Both produced valid MusicXML and no automatic dynamics: Lieder 148.7 seconds CPU-only, OSSQ 124.8 seconds CPU-only. This closes basic export/pipeline wiring, but not strict parity, full-page accuracy, GPU latency, detector fusion, or C1 lyric attachment.

The training-held-out real-source comparison completed all 8,422 decodes without failures. Candidate symbol error improved by 1.455 percentage points on OSSQ and 0.418 points on Lieder, passing the accepted +0.25-point Lieder regression margin. The first 3,501 accepted-model decodes used two CPU threads; remaining decodes used one. This uses previously evaluated validation scores and raw eager crop decoding, not untouched/full-page qualification. See `JOINT_HOLDOUT_QUALIFICATION_2026-10-01.md` for exact local paths, limitations and next gates. No training or GPU job launched; no model promoted.

## Archived experiment state

The ImageNet-backed random music decoder + joint notation-head run is paused at **epoch 81 / update 102,060**. It did not complete the planned 160 epochs. Epoch 80 is the last complete three-source validation. No model or production pin was promoted, and instance destruction has not been confirmed by the owner.

Prioritized preservation completed at 2026-10-01 04:01 UTC: **29,024 files / 21.23 GB logical size** and code/environment records passed SHA-256 checks. The owner explicitly chose a smaller salvage over waiting for all intermediate checkpoints. **125 missing intermediate model/optimizer files / 44.58 GB** were omitted, along with redundant supplementary core-training states/context. Some other epochs were already local and retained. Published corpora and rebuildable environments/tools were not copied.

Local archive: `/home/jhlusko/workspace/homr-artifacts/instance-salvage-20261001/`.
Read `SAFE_TO_DESTROY.md`, `PRIORITY_VERIFIED.json`, `priority-manifest.json` and `intentionally-not-pulled.json` for the precise scope. The original full inventory is not a claim that every inventoried file was copied.

Retained checkpoints include epochs **44, 73, 80, 81** and the full epoch-81 resume with model, optimizer, scheduler and RNG. Resume:
`/home/jhlusko/workspace/homr-artifacts/instance-salvage-20261001/files/runs/random-core-heads-pdmx-v2-audited-20260930/full/resume.pt`.
SHA-256: `f2cfffbddd3780472a402e901dbd9b5475f1f1c36eaf9ec8cf8adaa9f76a9a29`.
Archived files may share hard links with existing immutable artifacts; copy a checkpoint to a working directory before modifying it.

## What actual decoding found

Both epochs decoded all **3,909 paired crops** successfully: 2,871 OSSQ, 429 Lieder, 609 PDMX. The reference is **public-rareNum-20260927 core + the 8,400-update real-only frozen-core heads**. It is not necessarily the strongest accepted stack or the deployed Modal model. This was float32/math-SDPA eager greedy decoding, without runtime repair policies.

| Symbol edit error; lower is better | Frozen reference | Random-start epoch 73 | Random-start epoch 80 |
|---|---:|---:|---:|
| OSSQ | 2.372% | 2.681% | 2.538% |
| Lieder | 3.542% | 5.784% | 5.738% |
| PDMX | 2.865% | 8.702% | 8.493% |

Epoch 80 is slightly better overall than 73, but OSSQ pitch and accidental errors still worsen versus the reference, with paired score-bootstrap intervals above zero. Lieder symbol/pitch/accidental regressions also have intervals above zero. OSSQ overall symbol/rhythm differences have intervals crossing zero.

There are useful notation gains: on the same correctly recognized notes, epoch 80 improves stem-direction accuracy by approximately 6.3 percentage points on OSSQ and 20.6 on Lieder, and beam-vector accuracy by 6.9 and 20.3 respectively. These conditional accuracies cannot be substituted for recognition quality or compared directly with teacher-forced macro-F1. Wrong or missing notes must count in the end-to-end assessment.

Reports and complete predictions are local:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/train-20261001/random-core-decode-epochs73-80/epoch-{73,80}/comparison.md` (and `.json`, `part-*/{frozen,joint}.jsonl`).

The public-core-initialized **joint epoch-27 candidate** is a different experiment, not the poorly performing frozen-backbone E1 run or the random-start run. Its cached decoding is at:
`/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/train-20260929/joint-vs-frozen-20260929/decoded/`.
It had symbol errors OSSQ 1.862%, Lieder 3.728%, PDMX 3.222% on this same benchmark. These are reference points; recompute matched comparisons rather than subtracting separate confidence intervals.

The benchmark contains some core-seen score identities and is an internal diagnostic, not an untouched real-score/full-page release gate. These results establish that neither random-start candidate should be promoted now. They do not prove that the remaining 79 planned epochs could never improve it.

## Completed local comparison and review — 2026-10-01

Steps 1–3 below are complete. Four-way identity matching and all 62,544 stored recognition metrics pass recomputation. The 17-case MuseScore gallery includes original pixel-verified crops, readable crop-local measure/beat labels, and explicit warnings for reconstruction or engraving changes. It requires no individual accept/reject decisions.

Report: `/home/jhlusko/workspace/homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/cached-decoding-audit-20261001/comparison.md`.
Audit: same directory, `error-audit.md`; gallery: `review/index.html`.
Browser: http://localhost:8765/cached-decoding-audit-20261001/review/.

Public-initialized joint **2026-09-29 epoch 27** has better recognition and end-to-end stem/beam coverage than random73/80 on all sources. It is the preferred joint candidate to qualify against the accepted full stack, not a newly promoted model. Frozen/public27/random73/random80 symbol errors are OSSQ 2.372/1.862/2.681/2.538%, Lieder 3.542/3.728/5.784/5.738%, PDMX 2.865/3.222/8.702/8.493%.

Only eight independent OSSQ score identities underlie the 2,871 crops. Large failures are concentrated, but Lieder/PDMX still degrade after removing their worst 5% (a diagnostic, not an evaluation filter). Random80 emits 149 invalid-pitch note symbols on PDMX versus 32 frozen and 15 public27. Invalid combinations motivate a bounded decoding-constraint experiment but do not explain the whole real-source gap. EOS causes were not logged, and no library/optimizer cause has been demonstrated.

Next: resolve export parity, then qualify the full public27 stack on untouched real-score/full-page inputs; for the random branch, test decoder consistency and termination logging before spending on more training. The random run remains paused at epoch81; the separate local real-source crop comparison is complete. See `error-audit.md` for limits and the detailed experiment recommendation.

## Original sequence, with steps 1–3 now completed

1. **Build a paired comparison from cached predictions — no GPU.** Include the frozen reference, public-initialized joint epoch 27, random-start epochs 73 and 80. Verify crop IDs, pixel/reference/sidecar hashes and scoring settings before matching. Report source-specific symbol, pitch, rhythm and accidental errors, score-clustered intervals, common-note head metrics and correctly-recognized-and-notated GT-note coverage. Prioritize OSSQ/Lieder; retain PDMX as a separate diagnostic.
2. **Audit the decoding regressions locally — no GPU for the initial audit.** Rank changes by score, separate widespread small errors from a few catastrophic crops, and inspect deletions/insertions, repeated passages, early termination, output/reference lengths, pitch/octave errors and accidental/key-signature handling. Check training/validation versus inference image preparation, including the Lieder pre-resize cap, canvas geometry, vocabulary order, checkpoint configuration and archived/runtime code differences. The teacher-forced/free-running gap motivates this audit; its cause is not established.
3. **Generate a focused review gallery — no GPU for stored predictions.** Show original crops beside the reference and divergent model outputs, emphasizing real-source regressions and notation gains. Use measure/beat/pitch labels. Use MuseScore for reconstruction and audit its output; keep original images authoritative and do not infer encoded stem/beam differences solely from renderer output. The review should explain representative failures, not require hundreds of individual accept/reject decisions.
4. **Choose a bounded follow-up only after the audit.** Keep the existing recognition baseline and accepted runtime policies in place. If the audit finds a code/protocol defect, fix it and re-decode the affected retained checkpoints before drawing training conclusions. If the regression is model behavior, select a targeted experiment on the public-initialized stack or a limited resume of the random-start run, with actual decoded OSSQ/Lieder recognition gates. Do not automatically spend another ~15 GPU-hours or transplant heads between incompatible core feature representations.
5. **Return to release qualification after selecting a complete stack.** Measure untouched real-score/full-page performance with the selected core, heads, runtime policies and detector fusion. Then regenerate final side-by-side examples (E3), finalize pins/calibration/provenance and update the public OTS-HOMR model description to claims supported by those results. C1 lyric-recognition/attachment quality and the lyrics/directions/dynamics fusion plan remain separate release work; random-start scoring does not close them.

No GPU is needed for steps 1–3. Random-start training/decoding is inactive. The comparison, initial error audit, gallery and full raw crop comparison are complete; export/runtime qualification remains open.
