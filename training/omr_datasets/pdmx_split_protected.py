"""Split the PDMX index by score, reserving every archived evaluation score.

pdmx_split.py assigns scores to validation at random (seed 0). That was how the current
holdout was drawn, and re-running it after a reconversion would draw a *different* one:
scores the archived evaluations measured would land in training, and the numbers this
project has already published would no longer refer to held-out data.

pdmx_protected_scores.json is a snapshot of that original validation set - 632 scores,
3,349 rows. Every one of them goes to validation here, and nothing else does. A window is
kept only if its image, tokens and sidecar all exist, so a half-written conversion cannot
leak a row with no ground truth into either side.

Re-rendered windows are not byte-identical to the originals - Verovio's engraving and the
converter's filters have both moved - so this preserves *score membership*, not pixels.
"""

import argparse
import hashlib
import json
from pathlib import Path

script_location = Path(__file__).resolve().parent
PROTECTED = script_location / "pdmx_protected_scores.json"


def score_of(row: str) -> str:
    """The PDMX hash names the score; -v<n>-w<n> names the render and the window."""
    return Path(row.split(",", 1)[0]).name.rsplit("-v", 1)[0]


def split(index: Path, root: Path, protected: Path = PROTECTED) -> dict:
    held = set(json.loads(protected.read_text())["scores"])
    rows: dict[str, list[str]] = {"train": [], "valid": []}
    scores: dict[str, set[str]] = {"train": set(), "valid": set()}
    incomplete = []
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        image, tokens = line.rsplit(",", 1)
        missing = [
            name
            for name in (image, tokens, tokens + ".notation.json")
            if not (root / name).is_file()
        ]
        if missing:
            incomplete.append({"row": line, "missing": missing})
            continue
        score = score_of(line)
        side = "valid" if score in held else "train"
        rows[side].append(line)
        scores[side].add(score)

    if scores["train"] & held:
        raise AssertionError(f"{len(scores['train'] & held)} protected scores landed in train")
    if not rows["train"] or not rows["valid"]:
        counts = {k: len(v) for k, v in rows.items()}
        raise AssertionError(f"empty side: {counts}")

    for side, lines in rows.items():
        (index.parent / f"index_{side}.txt").write_text(
            "".join(line + "\n" for line in sorted(lines)), encoding="utf-8"
        )

    return {
        "windows": {k: len(v) for k, v in rows.items()},
        "scores": {k: len(v) for k, v in scores.items()},
        "protected_scores": len(held),
        "protected_scores_not_converted": sorted(held - scores["valid"]),
        "incomplete_windows_excluded": len(incomplete),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "policy": "reserve archived PDMX evaluation score IDs; no random reassignment",
        "limitation": "re-rendered windows are not byte-identical; score membership is preserved",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="datasets/pdmx/index.txt")
    parser.add_argument(
        "--root",
        type=Path,
        default=script_location.parent.parent,
        help="directory the index rows are relative to (default: the git root)",
    )
    parser.add_argument("--audit", type=Path, help="write the full report here as JSON")
    args = parser.parse_args()

    report = split(args.index, args.root)
    if args.audit:
        args.audit.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "protected_scores_not_converted"}
    print(json.dumps(summary, indent=2))
    not_converted = report["protected_scores_not_converted"]
    if not_converted:
        print(f"WARNING: {len(not_converted)} protected scores did not convert")


if __name__ == "__main__":
    main()
