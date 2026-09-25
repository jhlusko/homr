# Handoff: detector retrain and GPU work queue

Written 2026-09-18 for an agent with no prior context. Read this first, then
`docs/TIE_LABEL_INVESTIGATION.md` and the roadmap at
`~/workspace/OurTextScores/docs/private/OTS_HOMR_PUBLIC_RELEASE_ROADMAP.md` (gitignored,
on disk only; §0c, §0d and §3 are the relevant sections).


> **Status, 2026-09-25: the retrain and real-scan evaluation are done.** Everything below
> describes the plan as of 2026-09-18, and **§1's instance no longer exists** — the work
> ran on `ssh -p 40097 root@88.207.87.60` (A100-SXM4-40GB).
>
> Result, on 1,569 held-out **synthetic** renders at IoU 0.5 — MuseScore re-renders of
> per-system MusicXML, **no real scans**: `Tempo` recall **0% → 83.9%**, overall
> recall **93.3%** — but `Tempo` precision **10.2%** (7,831 predicted against 949 real) and
> `StaffText` 30.0%. Two of this handoff's three *synthetic* criteria are met; "cut the
> over-prediction" is not. `detector_release_gate.py` refuses it, on classes with almost
> no validation data (`Fingering` 2 boxes, `Lyrics` and `Expression` none).
> **On real scans, this checkpoint is rejected:** `Tempo` recall is 52/264 (19.7%)
> on 299 OSSQ pages; after correctly folding `Tempo`/`StaffText`/`Expression`/
> `SystemText` into `DirectionText` and re-matching boxes, recall is 236/888 (26.6%)
> on OSSQ and 13/22 (59.1%) on Lieder. The selected 09-19 adaptation parent scored
> 19.9% OSSQ and 86.4% Lieder. Real OSSQ `Dynamic` recall fell from released `e4`'s
> 64.8% to 24.4%. Per-model class order was committed as `223f012` and checked on
> the actual five- and seven-class weights. The real-page recall criterion and
> score-disjoint selection/test manifest are frozen in
> `docs/DETECTOR_REAL_PAGE_RELEASE_GATE_2026-09-25.md`. The next step is baseline
> scoring on both roles before retraining; see the accepted release sequence.
>
> The corrected sampler alone would not have got there. Three data defects had to be fixed
> first: boxes split from `_cleaned.musicxml` contained no `<direction>` text at all (re-split
> from raw MusicXML); every rendered page read as solid black because MuseScore's PNG keeps
> ink in the alpha channel (already fixed in the uncommitted `musescore_boxes.py` edit here —
> commit it, or it will be rediscovered again); and `detector_split.py` split by system, not
> score, which would have leaked (fixed to 104/18 scores).
>
> Details: the roadmap's §3, `homr-artifacts/gpu-roadmap-20260919/STATUS.md`, and `RUNLOG.md`.
> Weights and evaluation:
> `homr-artifacts/gpu-roadmap-20260919/instance-results/workspace/train-20260924/detector-retrain/run1/`.

---

## 1. The instance - set up, not yet used

```
ssh -p 20092 root@38.255.16.69 -L 8080:localhost:8080
A100-PCIE-40GB, 96 cores, 754 GB RAM (vast.ai, unprivileged container)
/workspace   267 GB VOLUME - survives recycle. Put everything here.
/            41 GB overlay - OS only, lost on recycle.
```

State at handoff:

| path / session | state |
| --- | --- |
| `/workspace/repo/homr` | **shallow** clone of `jhlusko/homr` at `c441fc6` - done |
| `/workspace/data/ossq-source` | OSSQ source, 1.5 GB - done |
| `/workspace/data/lieder-all-source` | Lieder v4 private archive, 4.1 GB - done |
| `/usr/local/bin/mscore` | MuseScore **3.6.2**, AppImage extracted to `/workspace/squashfs-root` - done |
| tmux `setup` | was still installing CUDA torch into `/workspace/venv`, then the OCR stack. Log: `/workspace/setup.log`, ends `SETUP_DONE` when finished |

**Verify before any long job:**

```bash
grep SETUP_DONE /workspace/setup.log
/workspace/venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True
xvfb-run -a mscore --version                                                        # 3.6.2
```

The `mscore` symlink points into `/workspace`, but the symlink itself lives on the overlay
at `/usr/local/bin` - recreate it after a recycle. `xvfb` and friends are apt-installed on
the overlay too.

---

## 2. The job: retrain the `non-lyric-text` detector

### Why - the diagnosis, which took four corrections to get right

The released detector (`jhlusko/non-lyric-text` @ `9073adb5` = local `detector_e4.pth`)
has **0% recall for `Tempo`** over 264 known boxes at full-page box level. That is the
finding. Things that are **not** the cause, all measured:

- **Not preprocessing.** The checkpoint carries its own ImageNet normalisation, applied in
  `CamVidModel.forward`; training and inference both feed `/255.0`.
- **Not box recovery.** `min_area` was 4 (fixed to 200, `f443cff`); filtering does not
  rescue it.
- **Not a decision threshold.** A probability floor swept to 0.999 does not rescue it - the
  model is confidently wrong, not uncertain.
- **Not "pick a better checkpoint".** Of six runs, `e0` has the best patch IoU (0.875-0.99
  every class) and **0% page recall**. Patch IoU does not predict page behaviour.

**The cause is the training sampler.** `training/ocr/detector_patches.POSITIVE_RATIO = 0.7`
centres 70% of training patches on a box. A sliding window over a real page meets text in
**14.0%** of 320x320 patches (measured over 2,400 patches; recorded as
`MEASURED_SLIDING_WINDOW_RATIO = 0.14`, `c441fc6`). The model learned that text is
everywhere: the released checkpoint labels ~31% of every page as text.

### Read recall, not precision

The evaluation ground truth `homr-artifacts/ground_truth/ossq_boxes` (299 pages) is
**OCR-confirmed text only**. A detector that finds a mark the OCR missed is scored as a false
positive. Its own docstring (`training/ocr/ossq_box_ground_truth.py`): *"recall is
trustworthy; precision is a lower bound, not a measurement."* Every precision figure in the
roadmap is a lower bound. Compare detectors against each other on the same pages rather
than quoting absolutes.

Current standings, 8 pages, `min_area=200`:

| checkpoint | recall (trustworthy) | precision (lower bound) |
| --- | ---: | ---: |
| `e4` released | **85.0%** | 2.7% |
| `e5` | 50.0% | 6.3% |
| `e0` | 0.0% | 0.0% |

### Success criterion

**Beat `e4`'s 85% overall recall, get `Tempo` recall off 0%, and cut the over-prediction**
(fraction of page pixels called text). Not "beat 0.2% precision".

### The plan

1. **Confirm inputs.** `training/omr_datasets/musescore_boxes.py --scores <dir>` needs a
   directory of `.musicxml` systems and renders them with MuseScore to get SVG boxes and a
   PNG. Check `/workspace/data/ossq-source` actually contains those before a long render.
   The local laptop has no mask corpus: the previous instance's masks and patch bank
   (`/workspace/b0/patch_bank_v4`) were lost because that instance had no volume.
2. **Render -> masks -> patch bank.** Use MuseScore 3.6.2 only - the boxes come from its
   SVG, and a different version moves the layout.
3. **Train with `--positive-ratio 0.14`** (the flag exists; the default was deliberately
   left at 0.7 so old histories stay comparable) plus whole-page negatives.
4. **Add the Lieder labels.** `homr-artifacts/ground_truth/lieder_boxes` on the laptop:
   127 pages, 99 hand-labelled `Tempo` boxes, 48 deliberate negatives (title pages). Same
   `.boxes.json` shape as `ossq_boxes`. Copy them up.
5. **Validate on pages, not patches**, with `training/ocr/detector_box_eval.py`. That is the
   whole lesson of `e0`.
6. **Gate the result** with `training/ocr/detector_release_gate.py <history.json>`. It
   refuses a checkpoint with any class below 0.05 validation IoU. It is a floor on one
   failure mode, **not** a release criterion - `e0` passes it and is useless. It also cannot
   see per-class validation support (many classes sit on identical values across runs,
   e.g. `Fingering` 0.875 in five of six - tiny support).

---

## 3. The rest of the GPU queue (same machine)

In rough priority:

1. **Evaluate `lyrics` at box level.** It is `detector_e2`, trained through the same
   sampler, with `MeasureNumber` at 0.006 in its own history, and it ships. Never measured
   at page level. Quick on this GPU.
2. **Re-take the structured heads' accuracies against `lieder-v8`.** Every Lieder-measured
   slur/tie figure was scored against a corpus that scrambled notation across chord members
   (see `docs/TIE_LABEL_INVESTIGATION.md`). `lieder-v8` agrees with the source on 100% of
   matched notes; use it, never v6/v7.
3. **Re-take in-place head-vs-rule comparisons on Lieder material.** The stem rule
   (+3.33pp) and beam repair were measured on OSSQ only. The slur-side rule was 94.8% on
   OSSQ and **51.3% on Lieder** and is now off by default (`81cd48e`).
4. **Retrain the scan models on `lieder-v8`** - agreed with the user as "eventually". The
   data changed, not the recipe.

`lieder-v8` lives on the laptop at `~/workspace/homr-artifacts/lieder-v8` (4.8 GB). To
rebuild it on the instance instead: `lieder-omr-data` repo, `tools/build_lieder_v4.py
--source <lieder-all-source> --out <dir>` with the vendored runtime pinned via
`tools/vendor_runtime.py --refresh`. Detection takes ~5 hours on CPU.

---

## 4. Traps that cost time this session

- **Kill by PID, never `pkill -f` over ssh.** The pattern matches the ssh command line
  itself and kills your own shell. Happened twice.
- **`python -u` under nohup/tmux**, or progress lines sit in the buffer and a healthy job
  looks silent. `detector_box_eval` prints every 20 pages.
- **Full-page evaluation is ~8 hours on CPU for 299 pages.** Do it on the GPU.
- **GitHub is slow from this host (~13 KB/s); R2 is fast.** Use `git clone --depth 1`.
- **Count is not correctness.** A detection run once wrote 215 system files that were all
  `{"pages": {}}`; it was reported as done because the file count was right. Open one.
- **Measure the corpus against the source, not against itself.** Three wrong hypotheses
  about the tie labels came from internal-consistency numbers; the defect was found by
  diffing a crop's tokens against the source measures.
- **Local laptop venv** is `~/workspace/homr/.venv`; CPU torch was installed into it this
  session. One test fails there before and after this work
  (`test_base_predictions.py::...reads_filepaths_checkpoint`); ignore it.

## 5. Where the user is

- OurTextScores: **commit, never push** (its pre-push hook tears down the local Docker stack).
  homr and lieder-omr-data push normally.
- `lieder-omr-data` commit `bb69492` still carries 178 MB of ONNX weights in history; they
  are untracked since `99ca69c`. Removing them needs a force-push - the user's call.
- The user labelled the Lieder boxes by hand and prefers not to do mechanical labelling;
  derive where you can, but see `training/ocr/derive_tempo_boxes.py` for three geometric
  approaches that failed on first pages.

## 2026-09-19: unified DirectionText real-page fine-tuning

Tempo, StaffText, Expression, and SystemText now map to DirectionText. Two four-epoch
unified runs completed under `/workspace/retrain-direction-20260919`. Select the 0.14
checkpoint as the adaptation parent (SHA256
`a03cd63177353360faf7c8a377e061d5130c3eeeab10775fd2c980a08096520c`). It beats 0.7 on
real-page DirectionText and Dynamic recall; neither is approved for release. Fingering
is explicitly excluded from the learning-floor gate and reported separately; both
parent histories pass the corrected gate. This is not a full-page quality gate.

A new run is launched at `/workspace/finetune-direction-20260919`, using Supervisor
programs `detector-finetune-pipeline` and `detector-finetune-watchdog`. SSH remains
`ssh -p 20092 root@38.255.16.69`. Local scripts are in
`../homr-artifacts/detector-finetune-20260919/scripts/`.

- Four epochs, AdamW 2e-5, seed 20260919, 16,384 sampled patches/epoch, batch 32.
- Half of draws from 60 real training pages (77 DirectionText boxes); half from
  46,800 cached synthetic patches. Real Dynamics/Lyrics labels are only in held-out
  scores and are NOT moved into training. This is DirectionText adaptation, not a
  comprehensive real-data fix for all classes.
- Preserve the existing score-disjoint split. Exclude 38 sparsely annotated empty
  real pages instead of treating missing annotation as proof of negative content.
  On annotated real pages, unlabelled ink stays ignored.
- Real annotation geometry: no out-of-bounds boxes; a visual tempo-box spot-check
  aligned. Scan dimensions vary greatly; lyric GT mixes phrases and syllables, so
  low strict box recall needs further scale/granularity investigation.
- Epoch selection uses full-page Lieder DirectionText F1 lower bound and synthetic
  macro F1 (DirectionText/Dynamic/Lyrics), rejects >5pp synthetic Dynamic/Lyrics recall
  regression, and includes the original parent as fallback. Existing validation
  cohorts were already inspected; do not call them a fresh untouched test set.
- Final selected checkpoint receives all-cohort page evaluation. Results:
  `results/selection.json`, `results/selected-full.json`, `results/gate.json`.
- Progress: `results/pipeline-status.json`, `results/watchdog.json`, `logs/pipeline.log`.
  Watchdog checks every 30s and allows at most two restarts; training resumes from an
  atomic epoch checkpoint. A stage longer than 20 minutes is flagged, not killed.
