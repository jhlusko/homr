"""
The correct way to find OSSQ-OMR ground truth and page-to-score measure alignment -
built after discovering that every previous script in this investigation
(ossq_measure_length_audit.py, deep_barline_audit.py, deep_barline_audit_broad.py) read
`<page>.musicxml` as ground truth, which is actually `homr.main`'s own prior output
written to that exact path (confirmed via its `<software>homr</software>` metadata).
See OSSQ_GROUND_TRUTH_ERRORS.md's retraction and DECODER_RHYTHM_ACCURACY_DESIGN.md §7.1
for the full account.

Real ground truth lives at each piece's own top level: `scores/<composer>/<piece>/
sq<id>.musicxml`. The page-local-to-absolute measure mapping comes from the corpus's own
`metadata/{scanned,synthetic,unaligned}/{systemwise/,}sq<id>:<page>:<system>.yaml` -
`measure_start`/`measure_end`, corpus-provided, not reconstructed.
"""
import re
from pathlib import Path

import yaml


def piece_dir(image_path: Path) -> Path:
    """images/{scanned,synthetic}/original/<page>.png -> the piece's own directory."""
    return image_path.parents[3]


def score_and_page(image_path: Path) -> tuple[str, str]:
    """'sq10675759:0024.png' -> ('sq10675759', '0024')."""
    stem = image_path.stem
    score_id, page_str = stem.split(":")
    return score_id, page_str


def real_ground_truth_path(image_path: Path) -> Path | None:
    gt = piece_dir(image_path) / f"{score_and_page(image_path)[0]}.musicxml"
    return gt if gt.exists() else None


def measure_start_for_system(image_path: Path, system_index: int) -> int | None:
    """`system_index` is HOMR's own 0-based system index on this page. Returns the
    corpus's own `measure_start` for that system (the score-absolute number of its
    first measure), or None if no metadata is found for this page/system."""
    score_id, page_str = score_and_page(image_path)
    system_num = system_index + 1  # corpus's own system_idx is 1-based
    candidates = [
        piece_dir(image_path) / "metadata" / "scanned" / "systemwise"
        / f"{score_id}:{page_str}:{system_num:04d}.yaml",
        piece_dir(image_path) / "metadata" / "synthetic" / "systemwise"
        / f"{score_id}:{page_str}:{system_num:04d}.yaml",
        piece_dir(image_path) / "metadata" / "unaligned"
        / f"{score_id}:{page_str}:{system_num:04d}.yaml",
    ]
    for path in candidates:
        if path.exists():
            data = yaml.safe_load(path.read_text())
            return int(data["measure_start"])
    return None
