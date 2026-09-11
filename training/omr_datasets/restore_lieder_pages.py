"""Re-render the Lieder page images that were dropped as regenerable.

`instance-2026-08-28/README.md` left 18.5GB of rendered IMSLP pages behind on the grounds
that they were "regenerable from the PDFs, and the largest thing on the box by far". That
was true, and this is the regeneration -- needed again because the scanned-syllable corpus
for the recogniser is cut from those pages, and only a fraction of the lyric-bearing
scores still have them.

The PDFs come from the SHA-pinned `lieder-all-source` archive, so the input is fixed and
checkable rather than re-fetched from IMSLP.

**The renderer, the resolution and the filenames are all verified, not assumed.** Rendering
IMSLP10416 page 1 from the archive with poppler's `pdftoppm -r 300 -gray` reproduces the
surviving page byte-for-byte in shape and content -- 2480x3509, mean absolute pixel
difference 0.0/255 -- and the PDF's page order matches the filename numbering with no
offset for title pages. That check earns its keep: a resolution or numbering that merely
looked right would attach every label in the corpus to the wrong pixels, and nothing
downstream would raise a word about it. Use the same renderer; a different one may agree on
size while disagreeing on antialiasing.
"""

# flake8: noqa: T201

import argparse
import subprocess
import sys
from pathlib import Path

#: The resolution the surviving pages were rendered at; see the module docstring.
DPI = 300


def render_score(pdf: Path, out_dir: Path, dpi: int = DPI) -> int:
    """Render every page of `pdf` as `<score>-pNNN.png`, returning the page count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # pdftoppm appends "-NNN.png" to the prefix and zero-pads to the page count's width,
    # so a >99-page score would give "-p0100" against our fixed 3 digits. Render to a
    # scratch prefix and rename, rather than trusting its padding to match ours.
    prefix = out_dir / f".{pdf.stem}-tmp"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf), str(prefix)],
        check=True,
        capture_output=True,
    )
    rendered = sorted(out_dir.glob(f".{pdf.stem}-tmp-*.png"))
    for page in rendered:
        number = int(page.stem.rsplit("-", 1)[1])
        page.rename(out_dir / f"{pdf.stem}-p{number:03d}.png")
    return len(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdfs", type=Path, required=True,
                        help="The archive's pdfs/ directory of <score>.pdf.")
    parser.add_argument("--pages", type=Path, required=True,
                        help="lieder_vocal_text/pages, one directory per score.")
    parser.add_argument("--labels", type=Path,
                        help="If given, only restore scores that have a label file here.")
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-render scores that already have pages.")
    args = parser.parse_args()

    pdfs = sorted(args.pdfs.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs under {args.pdfs}")

    wanted = None
    if args.labels:
        wanted = {path.stem for path in args.labels.glob("*.json")}

    rendered = present = unlabelled = 0
    failed: list[str] = []
    for index, pdf in enumerate(pdfs, start=1):
        if wanted is not None and pdf.stem not in wanted:
            unlabelled += 1
            continue
        out_dir = args.pages / pdf.stem
        if not args.overwrite and out_dir.is_dir() and any(out_dir.glob("*.png")):
            present += 1
            continue
        try:
            count = render_score(pdf, out_dir, args.dpi)
        except subprocess.CalledProcessError as error:
            failed.append(pdf.stem)
            print(f"[{index}/{len(pdfs)}] {pdf.stem}: FAILED "
                  f"{error.stderr.decode(errors='replace').strip()[:200]}", flush=True)
            continue
        rendered += count
        print(f"[{index}/{len(pdfs)}] {pdf.stem}: {count} page(s)", flush=True)

    print(f"\n{rendered:,} pages rendered; {present} score(s) already had pages, "
          f"{unlabelled} unlabelled, {len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
