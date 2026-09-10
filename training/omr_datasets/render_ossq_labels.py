"""Render OSSQ staff labels back into notation, so a reviewer compares pictures.

`ossq_pair_review_server.py` asks one question: does this scanned crop show the music its
label claims? Answering it from a pitch token sequence means reading
`D4 F3 F3 F3 G3 ...` against a photograph of a staff, which is slow, error-prone and
unpleasant enough that it degrades the review itself. Rendering the label puts the same
question as two pictures side by side.

This is `render_stage2_tokens.py` pointed at a different corpus: same
`generate_xml` -> `mscore -j` path, same batching, and deliberately the same
`render_batch`, so the MuseScore invocation and its failure handling live in one place.
What differs is the input - an index of `image,tokens` rows rather than a directory of
token files - because the review queue is defined by an index and only the staves under
review are worth the render time.

Batching needs a fallback, not just a smaller size. `mscore -j` under `check=True`
discards the whole batch when any one file in it fails, so a bad staff costs its
batch-mates too. `render_stage2_tokens.py` reduced the batch from 200 to 20, which
helped - but this tool renders in *review* order, worst-scoring staves first, which
deliberately clusters the problematic ones at the front: measured, 60 of the first 120
were lost that way while each rendered perfectly well on its own. So a failed batch is
retried one file at a time, and only the files that genuinely fail are lost.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

import cv2
import numpy as np

from training.omr_datasets.render_stage2_tokens import render_batch

#: Small on purpose - see the module docstring.
BATCH_SIZE = 20

#: Pixels of white kept around the trimmed content.
MARGIN = 12

#: At or above this grey level a pixel counts as page, not ink.
PAPER = 245


def trim(image: np.ndarray, margin: int = MARGIN, paper: int = PAPER) -> np.ndarray:
    """The rendered staff, with MuseScore's page margins cut away.

    MuseScore renders a full page: a staff occupying a few percent of an A4 sheet,
    the rest white. Shown beside a tightly-cropped photograph of a staff, the notes
    come out perhaps a twentieth the size of the ones they are being compared against,
    which defeats the purpose of showing pictures at all. Trimming to the inked
    bounding box makes the two images comparable at a glance.

    An all-white image is returned unchanged rather than collapsing to zero size.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inked = np.argwhere(grey < paper)
    if inked.size == 0:
        return image
    top, left = inked.min(axis=0)
    bottom, right = inked.max(axis=0)
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(grey.shape[0], bottom + margin + 1)
    right = min(grey.shape[1], right + margin + 1)
    return image[top:bottom, left:right]


def trim_in_place(path: Path) -> bool:
    image = cv2.imread(str(path))
    if image is None:
        return False
    trimmed = trim(image)
    if trimmed.shape == image.shape:
        return False
    cv2.imwrite(str(path), trimmed)
    return True


def token_paths(index_path: Path, limit: int = 0) -> list[Path]:
    paths = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        paths.append(Path(line.split(",")[1]))
    return paths[:limit] if limit else paths


def pending(paths: list[Path], out_dir: Path) -> list[Path]:
    """Only what is not already rendered, so a re-run resumes rather than repeats."""
    return [p for p in paths if not (out_dir / f"{p.stem}.png").is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="0 = every row")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    paths = token_paths(args.index, args.limit)
    todo = pending(paths, args.out)
    print(f"{len(paths)} staves, {len(todo)} not yet rendered", flush=True)

    done = failed = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        try:
            render_batch(batch, args.scratch, args.out, title=False)
        except Exception:  # noqa: BLE001 - one bad batch must not end the run
            # Retry singly: the batch failed because *some* file in it did, and the
            # rest render perfectly well on their own.
            recovered = 0
            for one in batch:
                try:
                    render_batch([one], args.scratch, args.out, title=False)
                    recovered += 1
                except Exception:  # noqa: BLE001, PERF203
                    failed += 1
            print(f"batch at {start} failed; {recovered}/{len(batch)} recovered singly",
                  flush=True)
        for p in batch:
            rendered = args.out / f"{p.stem}.png"
            if rendered.is_file():
                trim_in_place(rendered)
                done += 1
        print(f"  {start + len(batch)}/{len(todo)} ({done} rendered, {failed} failed)", flush=True)

    print(f"{done:,} rendered, {failed} failed -> {args.out}")


if __name__ == "__main__":
    main()
