"""Render a matched Lieder piece's ground-truth score to page images, so a human can
directly see whether a `targeted_review_candidates.py` flag is a real detection
problem or a bad source/page match - not just trust the bar-count numbers.

Renders through MuseScore itself (`mscore`, already installed on this box), not a
from-scratch layout: this needs to look like the actual engraved score, which only
MuseScore's own renderer can reliably produce from a `.mscx` file. `xvfb-run` is
required - `mscore`'s own `-platform offscreen`/`QT_QPA_PLATFORM=offscreen` do not
work on this box (checked directly: both still try to open a real X display and
fail; `xvfb-run -a mscore ... -o out.pdf` does not).
"""

# flake8: noqa: T201

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pypdfium2 as pdfium

from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mscx,
    load_lieder_file_tree,
    load_lieder_scores,
    match_single_piece_scores,
)


def render_piece_to_pages(mscx_bytes: bytes, out_dir: Path, dpi: int = 150) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mscx_path = Path(tmp) / "score.mscx"
        mscx_path.write_bytes(mscx_bytes)
        pdf_path = Path(tmp) / "score.pdf"
        subprocess.run(  # noqa: S603
            ["xvfb-run", "-a", "mscore", str(mscx_path), "-o", str(pdf_path)],
            check=True, capture_output=True, timeout=120,
        )
        pdf = pdfium.PdfDocument(str(pdf_path))
        paths = []
        try:
            scale = dpi / 72.0
            for index, page in enumerate(pdf, start=1):
                bitmap = page.render(scale=scale)
                page_path = out_dir / f"page{index:03d}.png"
                bitmap.to_pil().save(page_path)
                paths.append(page_path)
        finally:
            pdf.close()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--candidates", type=Path, required=True,
        help="targeted_review_candidates.py's --out file - only these scores' "
        "pieces get rendered, not the whole matched corpus.",
    )
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output dir, one subdir per score.")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    score_ids = sorted({c["score_id"] for c in candidates})
    print(f"{len(score_ids)} unique score(s) to render")

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    matched = match_single_piece_scores(lieder, score_ids)

    ok = skipped = failed = 0
    for score_id in score_ids:
        out_dir = args.out / score_id
        if out_dir.exists() and any(out_dir.glob("page*.png")):
            skipped += 1
            continue
        match = matched.get(score_id)
        if match is None:
            print(f"{score_id}: no single-piece Lieder match, skipping")
            continue
        key, entry = match
        try:
            mscx_bytes = fetch_mscx(entry, key, file_tree)
            pages = render_piece_to_pages(mscx_bytes, out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"{score_id}: FAILED ({e})")
            failed += 1
            continue
        print(f"{score_id}: rendered {len(pages)} page(s)")
        ok += 1

    print(f"{ok} rendered, {skipped} already cached, {failed} failed")


if __name__ == "__main__":
    main()
