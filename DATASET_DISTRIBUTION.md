# Distributing the corpora and models

This describes what we ship, how it is laid out, and how someone who has just downloaded
it gets a number out of it without asking us anything. It is the packaging counterpart to
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md`, which records *why* each artifact exists.

## The three things that stop a corpus being turnkey

All three pass a naive file count, which is what makes them worth writing down. Every one
of them was hit for real in this project.

1. **The staff crops are symlinks, not pixels.** `convert_ossq.py` calls `link_image`, so
   a corpus directory on the build machine contains links into that machine's scratch
   space. `rsync -a` copies the links. The result: 38,421 files arrive, the file count is
   exactly right, and every single one is dangling. The fix is `rsync -aL` (or `cp -L`,
   or `tar -h`) - dereference at copy time. **`-a` alone is the wrong flag for shipping
   these corpora**, and it fails silently.

2. **Index files carry build-machine absolute paths.** Rows look like
   `/workspace/b0/phase7bar/train/sq7383977_0003_0001_1.png,...`. That directory does not
   exist on the downloader's disk, and no amount of correct copying changes it.

3. **Box ground-truth JSON references pages by absolute path.** Same failure, one level
   deeper, and easier to miss because nothing reads it until an evaluation runs.

## The packager

`training/omr_datasets/package_dataset.py` fixes all three and then checks its own work.

```bash
python3 -m training.omr_datasets.package_dataset --root <corpus-dir>            # fix
python3 -m training.omr_datasets.package_dataset --root <corpus-dir> --verify-only
```

It materialises every symlink, rewrites index rows and ground-truth page references to
paths that resolve, writes `MANIFEST.json`, and exits non-zero if anything is still
unportable.

Two design points are load-bearing:

**Paths become relative to the index that names them, not to the dataset root.** These
indexes describe the build machine's layout while the files ship laid out beside their
index, so there is no common prefix to strip - a plain `relative_to()` finds nothing to
do and leaves every row absolute. Resolution order is: already-relative-and-resolves,
then a file of that name beside the index, then genuinely-under-root. It also means each
corpus directory can be moved on its own without breaking.

**A path it cannot resolve is left absolute rather than guessed.** `verify()` then reports
it. This is deliberate: a plausible-looking path that resolves to the *wrong* file is the
exact silent failure this whole module exists to prevent, and is strictly worse than a
loud one.

**The manifest excludes itself from its own digest.** Otherwise the checksum we publish is
computed on a tree without a manifest and checked against a tree with one, and never
matches - a checksum nobody can reproduce is worse than shipping none.

`verify()` re-reads from disk rather than trusting what the rewrite returned. The whole
category of bug here is a packaging step that reports success over a broken dataset.

## What ships

### Corpora

| directory | contents | built by |
|---|---|---|
| `corpora/ossq_scanned_corrected/` | 34,510 OSSQ scanned staves, pagination- and clef-corrected | `convert_ossq.py` |
| `corpora/lieder_scanned_pairs/` | OpenScore Lieder scanned staff pairs | `convert_lieder.py` + IMSLP system detection |

Both are `image,tokens` indexes with a `.notation.json` sidecar per staff carrying the
structured-head labels (beams, stems, slurs, ties, dynamics).

**These corpora are corrected in two ways that matter and that no upstream copy has.**
Both corrections are documented in full in `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md`:

- **Pagination.** Joining scanned crops to `musicxml/unaligned` by `(page, system)` looks
  right - the segment counts match - but the two paginations are produced independently,
  and 56.7% of scanned staves were silently mislabelled. Now 7.9%, corroborated by 52
  independent human verdicts with zero mismatches.
- **Clefs.** MuseScore's round-trip drops the clef from continuation systems, and because
  the tokens are *absolute pitches* (`B3`, not "third line"), a missing clef is invisible
  to every accuracy metric in the project. It only surfaces when something reconstructs
  the staff. 837 -> 7 affected staves (scanned), 1,012 -> 8 (synthetic).

That second one is the reason to prefer this copy over regenerating from source: the
defect does not show up in any metric you would think to check.

### Models

| path | what it is | headline number |
|---|---|---|
| `models/base/scans_clef_best.pth` | best Stage 2 base transformer | `eval_accuracy` **0.96906** |
| `models/base/heads_clef.pth` + `heads_manifest.json` | structured heads, frozen-core trained on the above | `exact_beam_vector` **0.9508** |
| `models/detector/detector_e2.pth` | text detector, **vocal** ("with lyrics") | 88.5% F1 Lyrics+Dynamic pooled |
| `models/detector/detector_instr_bg.pth` | text detector, **instrumental** | Dynamic 43.1% P / 75.7% R |
| `models/ocr/crnn*.pth` | text recognisers (3 input heights) | - |
| `models/baseline/pytorch_model_426-*.pth` | pinned upstream homr checkpoint | the comparison baseline |
| `models/experiments/` | `lieder_only_model.pth`, `scans_synthetic_w1.pth` | kept so the negative results reproduce |

`heads_clef.pth` must be paired with `scans_clef_best.pth` - it holds 22 head tensors
trained with the other 326 core parameters frozen, and means nothing on a different core.

The two detectors are **not** interchangeable, and the choice between them is a
correctness component rather than a UI preference. `detector_e2` receives gradient only
on `Lyrics` and `Dynamic`; its other five classes predict everywhere. `detector_instr_bg`
is the reverse and emits tens of thousands of spurious `Lyrics` boxes on any vocal page.
Running a score through the wrong one does not degrade gracefully.

## Reproducing the headline numbers

Base model against the pinned upstream checkpoint, identical loader and metric:

```bash
python3 -m training.transformer.base_predictions \
  --checkpoint models/base/scans_clef_best.pth \
  --index corpora/ossq_scanned_corrected/train/index.txt \
  --out ours.jsonl
```

Structured heads:

```bash
python3 -m training.transformer.evaluate_structured_heads \
  --checkpoint models/base/scans_clef_best.pth \
  --weights   models/base/heads_clef.pth \
  --manifest  models/base/heads_manifest.json \
  --index     corpora/ossq_scanned_corrected/train/index.txt
```

Detectors, scored against a patch bank. **The banks are not shipped** - the four of them
come to 3.2 GB and are exactly reproducible, so they are a command rather than a
download. `extract_patch_bank.py` derives each image's seed by hashing `(seed,
image_index)` instead of advancing one shared RNG, specifically so that a worker pool -
which claims images in a different order every run - still produces an identical bank:

The masking policy is chosen when the **masks** are built, not when the bank is
extracted - `scan_text_masks.py --background-outside`. This matters more than the flag
placement suggests: a bank inherits whatever policy its masks were written under, and
nothing downstream records or checks it. Scoring two models trained under different
policies against one bank is meaningless, and no tooling stops you.

```bash
python3 -m training.ocr.scan_text_masks \
  --matches ground_truth/ossq_boxes/ \
  --pngs    datasets/ossq_instrumental_text/pages/ \
  --out     masks/ --background-outside

python3 -m training.ocr.extract_patch_bank \
  --index masks/index.txt --out bank/ --seed 0

python3 -m training.ocr.eval_detector \
  --checkpoint models/detector/detector_e2.pth models/detector/detector_instr_bg.pth \
  --index bank/index.txt
```

**Page-level evaluation is the one that decides anything.** It needs the box ground truth
in `ground_truth/ossq_boxes/`, which is shipped because it is the output of an OCR pass
over the scans and is not cheaply regenerable:

```bash
python3 -m training.ocr.detector_box_eval \
  --weights models/detector/detector_instr_bg.pth \
  --boxes   ground_truth/ossq_boxes/ \
  --index   <detector_split valid_index.txt>
```

**Read patch IoU as a training-progress signal only, never as a selection criterion.**
Twice in this project the patch table ranked a model that the page-level metric then
reversed - E3 led on patches and came last on page F1, and the first instrumental
detector scored 0.545-0.853 valid IoU on precisely the classes where it turned out to
emit 4,378 boxes per page. Selection needs `detector_box_eval.py` at page level.

## Known limits

State these wherever the dataset is published; they are the difference between a
reproducible release and one that generates confused issues.

- **The instrumental detector is not shippable on its rare classes.** Expression (9.7%)
  and Tempo (13.2%) precision mean most predicted boxes are wrong. Dynamic is the only
  class carrying its weight.
- **`StaffText` and `Expression` are dead on vocal pages**, not merely degraded - E2's
  loss does not cover them. This was an accepted trade, not an oversight.
- **The OSSQ synthetic track did not help** at any weight tried (w=1.0 and w=0.4 both
  score below the baseline's 0.96906). The weights are shipped so the negative result
  reproduces; do not treat that track as an improvement.
- **Macro metrics over the ultra-rare classes are noise.** Beam level 4 has support 8;
  several dynamics classes have fewer than 13 examples. The 0.5972 and 0.1030 figures
  should not be read as measurements.
- **Absolute accuracy is not comparable across the two domains.** Lieder and OSSQ scans
  differ in difficulty; only the within-domain deltas mean anything.

## Licensing

OpenScore Lieder and OpenScore String Quartets are CC0. IMSLP page scans carry their own
per-scan terms and are **not** blanket-redistributable - check before including page
images rather than crops. The derived token files and `.notation.json` sidecars are ours
to license.
