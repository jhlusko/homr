"""Fail closed when a rebuilt Lieder manifest violates its provenance contract."""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

from training.omr_datasets.notation_sidecar import sidecar_path
from training.omr_datasets.stage2_pair_review_server import parse_stem
from training.transformer.training_vocabulary import read_tokens


#: Every glyph that closes a measure, not just the plain one.  Counting only
#: "barline" undercounted 400 of 3189 rebuilt pairs on 2026-08-27 - each of them
#: ends on a repeat, double, or bold-double barline - and reported them as span
#: mismatches.  Adding these four resolves all 400 with zero residual; the count
#: must still equal the aligned span exactly, so this corrects the counter rather
#: than relaxing the contract.
MEASURE_DIVIDERS = frozenset(
    {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
)


def manifest_rows(path: Path) -> list[tuple[Path, Path]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            image, tokens = line.split(",", 1)
            rows.append((Path(image), Path(tokens)))
    return rows


def audit(manifest: Path, alignment_path: Path, recovered_manifest: Path) -> dict:
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment.get("model_predictions_used") is not False:
        raise ValueError("alignment does not certify model_predictions_used=false")
    rows = manifest_rows(manifest)
    recovered_rows = manifest_rows(recovered_manifest)
    # Provenance is about which FILES a row points at, not what they are called.
    # A rebuilt pair for a system that previously had a recovered label necessarily
    # reuses that system's stem, so a stem-name test flagged 468 rows on 2026-08-27 -
    # including ones where the rebuild had demonstrably repaired a truncated label
    # (IMSLP10416-sys7-v0: 2 barlines and a mid-system stop, rebuilt to 4 plus a
    # closing bolddoublebarline).  That test failed the rebuild for succeeding.
    recovered_files = {path.resolve() for pair in recovered_rows for path in pair}
    recovered_stems = {image.stem for image, _ in recovered_rows}
    problems = []
    seen = set()
    for image, tokens in rows:
        stem = image.stem
        if stem in seen:
            problems.append({"stem": stem, "problem": "duplicate manifest row"})
        seen.add(stem)
        if image.resolve() in recovered_files or tokens.resolve() in recovered_files:
            problems.append({"stem": stem, "problem": "historical recovered pair leaked in"})
        if not image.is_file() or not tokens.is_file():
            problems.append({"stem": stem, "problem": "missing image or tokens"})
            continue
        parsed = parse_stem(stem)
        if parsed is None:
            problems.append({"stem": stem, "problem": "unparseable stem"})
            continue
        score_id, system, _voice = parsed
        score = alignment.get("scores", {}).get(score_id)
        item = next(
            (entry for entry in score.get("systems", []) if entry["system"] == system),
            None,
        ) if score else None
        if item is None or item.get("status") != "aligned":
            problems.append({"stem": stem, "problem": "not backed by an aligned system"})
            continue
        expected = item["end_measure"] - item["start_measure"]
        actual = sum(symbol.rhythm in MEASURE_DIVIDERS for symbol in read_tokens(str(tokens)))
        if actual != expected:
            problems.append(
                {"stem": stem, "problem": "bar count differs from aligned span",
                 "expected": expected, "actual": actual}
            )
        if not sidecar_path(tokens).is_file():
            problems.append({"stem": stem, "problem": "missing notation sidecar"})
    return {
        "manifest": str(manifest),
        "pairs": len(rows),
        "unique_stems": len(seen),
        # Rows that literally point at a recovered file - must be zero.
        "recovered_overlap": sum(
            1
            for image, tokens in rows
            if image.resolve() in recovered_files or tokens.resolve() in recovered_files
        ),
        # Systems the rebuild re-labelled that previously had a recovered label.
        # Informational: this is the rebuild doing its job, not a violation.
        "rebuilt_over_recovered": len(seen & recovered_stems),
        "problems": problems,
        "passed": not problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--recovered-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.alignment, args.recovered_manifest)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{report['pairs']} pairs; recovered overlap {report['recovered_overlap']}; "
        f"rebuilt over recovered {report['rebuilt_over_recovered']}; "
        f"problems {len(report['problems'])}"
    )
    if not report["passed"]:
        raise SystemExit("clean-pair audit failed")


if __name__ == "__main__":
    main()
