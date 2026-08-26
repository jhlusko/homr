"""Draw every training patch once, to disk, instead of re-deriving them each epoch.

`DetectorPatches` reads a full page to produce a patch. Measured on this corpus, that
is not a small overhead: pages are 4138x2928, `cv2.imread` costs ~6.2s and
`box_centres_by_class` ~3.8s, so roughly ten seconds of work stands behind each image's
patches. `ImageBlockSampler` plus the dataset's one-slot cache already amortise that
across an image's eight draws rather than paying it eight times - but it is still paid
again on every epoch, and again for every experiment in the matrix. Four experiments of
ten epochs re-decode the same corpus forty times.

Nothing about those patches changes between runs, so this writes them out once. The
draws come from `DetectorPatches` itself rather than a reimplementation, so the sampling
semantics - the positive ratio, the per-class centre choice, the jitter - are the same
ones the live dataset uses and stay the same by construction if that logic changes.

Two things follow from the patches being on disk as small files, both of which matter
more than the raw speedup:

1. **Batches become properly shuffled.** The block sampler exists only to make the
   decode cache hit; its cost is that a batch is drawn from one or two pages, so the
   batch is not close to i.i.d. Patch files are tiny, so a plain `shuffle=True` is
   affordable and every batch mixes pages freely.
2. **The four experiments become comparable by construction.** E0-E3 read the same
   bank, so a difference between them cannot come from having drawn different patches.

The seed for each image is derived from the run seed and the image's index, so an
image's patches do not depend on how work was distributed across workers.
"""

# flake8: noqa: T201

import argparse
import hashlib
import multiprocessing
from pathlib import Path

import cv2

from training.ocr.detector_patches import PATCH_SIZE, DetectorPatches, Sample, read_index


def seed_for_image(seed: int, image_index: int) -> int:
    """A per-image seed that does not depend on worker scheduling.

    Deriving it by hashing rather than by advancing one shared RNG is what makes the
    bank reproducible: with a pool, the order images are claimed in varies run to run,
    so anything positional would give a different corpus each time.
    """
    digest = hashlib.sha256(f"{seed}:{image_index}".encode()).hexdigest()
    return int(digest[:16], 16)


def _draw_one(task: tuple[int, Sample, int, float, int, Path]) -> list[tuple[str, str]]:
    image_index, sample, patches_per_image, positive_ratio, seed, out_dir = task
    # One thread per worker, not one per core. OpenCV otherwise sizes its pool from the
    # core count inside every pool member at once, which both oversubscribes the CPU and
    # multiplies the process's thread count - the same failure mode that took this box
    # past its 3840 pid limit when the OCR shards each opened ~572 threads.
    cv2.setNumThreads(1)
    dataset = DetectorPatches(
        [sample],
        patches_per_image=patches_per_image,
        positive_ratio=positive_ratio,
        seed=seed_for_image(seed, image_index),
    )
    written: list[tuple[str, str]] = []
    for k in range(patches_per_image):
        image_patch, mask_patch = dataset[k]
        stem = f"{image_index:06d}_{k:02d}"
        image_path = out_dir / f"{stem}.png"
        mask_path = out_dir / f"{stem}.mask.png"
        cv2.imwrite(str(image_path), image_patch)
        cv2.imwrite(str(mask_path), mask_patch)
        written.append((str(image_path), str(mask_path)))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--positive-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    samples = read_index(args.index)
    patch_dir = args.out / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (i, sample, args.patches_per_image, args.positive_ratio, args.seed, patch_dir)
        for i, sample in enumerate(samples)
    ]

    written: list[tuple[str, str]] = []
    done = 0
    with multiprocessing.Pool(args.workers) as pool:
        for result in pool.imap_unordered(_draw_one, tasks, chunksize=1):
            written.extend(result)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} images", flush=True)

    # Sorted so the index does not record the order the pool happened to finish in -
    # two runs of the same seed should produce byte-identical index files.
    written.sort()
    index_path = args.out / "index.txt"
    index_path.write_text(
        "\n".join(f"{image},{mask}" for image, mask in written) + "\n", encoding="utf-8"
    )
    print(f"{len(written):,} patches from {len(samples):,} images -> {index_path}")
    print(f"  patch size {PATCH_SIZE}, positive ratio {args.positive_ratio}, seed {args.seed}")


if __name__ == "__main__":
    main()
