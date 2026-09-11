"""Build a deterministic, class-balanced replay index for rare metre numerators.

Generic replay leaves rare metres to chance: a 1,300-row PDMX draw showed the model only
about 20 examples of numerator 12 and 35 of numerator 5.  This tool selects existing
training rows that contain explicitly requested ``timeSignatureBeats_N`` tokens.  It
does not create labels or touch a validation index.

Example (on the GPU instance)::

    python -m training.omr_datasets.build_numerator_replay \
      --source pdmx=datasets/pdmx/index_train.txt \
      --source grandstaff=datasets/grandstaff/index.txt \
      --target 5=400 --target 12=400 \
      --output /workspace/b0/lieder-rebuild/rare_numerator_replay_index.txt

The adjacent ``.metadata.json`` records the input indexes, seed, requested coverage and
actual selected coverage so a training run remains reproducible after corpora change.
"""

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from homr.transformer.vocabulary import TIME_SIGNATURE_BEATS_PREFIX


@dataclass(frozen=True)
class Candidate:
    """One original index row and the requested numerator classes it supplies."""

    row: str
    classes: frozenset[int]


def _name_path(value: str) -> tuple[str, Path]:
    name, sep, raw_path = value.partition("=")
    if not sep or not name or not raw_path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(raw_path)


def _target(value: str) -> tuple[int, int]:
    raw_class, sep, raw_count = value.partition("=")
    if not sep or not raw_class.isdigit() or not raw_count.isdigit() or int(raw_count) <= 0:
        raise argparse.ArgumentTypeError(f"expected NUMERATOR=COUNT, got {value!r}")
    return int(raw_class), int(raw_count)


def _token_path(row: str, root: Path) -> Path:
    token_path = Path(row.rsplit(",", maxsplit=1)[-1])
    return token_path if token_path.is_absolute() else root / token_path


def _classes_in_token_file(path: Path, targets: set[int]) -> frozenset[int]:
    result = set()
    for token in path.read_text(encoding="utf-8").split():
        if not token.startswith(TIME_SIGNATURE_BEATS_PREFIX):
            continue
        raw = token.removeprefix(TIME_SIGNATURE_BEATS_PREFIX)
        if raw.isdigit() and int(raw) in targets:
            result.add(int(raw))
    return frozenset(result)


def collect_candidates(
    sources: dict[str, Path], targets: set[int], root: Path
) -> tuple[dict[int, list[Candidate]], int]:
    """Read source indexes, resolving their relative token paths from ``root``."""
    by_class = {target: [] for target in targets}
    missing = 0
    for source_name, index_path in sources.items():
        resolved_index = index_path if index_path.is_absolute() else root / index_path
        if not resolved_index.is_file():
            raise FileNotFoundError(f"source {source_name!r} index not found: {resolved_index}")
        for row in resolved_index.read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            token_path = _token_path(row, root)
            if not token_path.is_file():
                missing += 1
                continue
            candidate = Candidate(row, _classes_in_token_file(token_path, targets))
            for target in candidate.classes:
                by_class[target].append(candidate)
    return by_class, missing


def select_candidates(
    by_class: dict[int, list[Candidate]], targets: dict[int, int], seed: int
) -> list[Candidate]:
    """Select requested coverage without duplicating an index row in the output."""
    rng = random.Random(seed)
    selected: list[Candidate] = []
    selected_rows: set[str] = set()
    supplied: Counter[int] = Counter()
    for target in sorted(targets):
        candidates = list(by_class[target])
        rng.shuffle(candidates)
        for candidate in candidates:
            if supplied[target] >= targets[target]:
                break
            if candidate.row in selected_rows:
                # A row selected for another numerator still supplies this one, but it
                # must never be written twice and thereby become accidental oversampling.
                supplied.update(candidate.classes)
                continue
            selected.append(candidate)
            selected_rows.add(candidate.row)
            supplied.update(candidate.classes)
        if supplied[target] < targets[target]:
            raise ValueError(
                f"numerator {target}: requested {targets[target]} rows, "
                f"but only {supplied[target]} unique rows are available"
            )
    return selected


def coverage(candidates: list[Candidate]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for candidate in candidates:
        counts.update(candidate.classes)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", action="append", required=True, type=_name_path, metavar="NAME=INDEX"
    )
    parser.add_argument(
        "--target", action="append", required=True, type=_target, metavar="NUMERATOR=COUNT"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Base for relative source/index token paths (default: current directory).",
    )
    args = parser.parse_args()
    sources = dict(args.source)
    targets = dict(args.target)
    root = args.root.resolve()
    by_class, missing = collect_candidates(sources, set(targets), root)
    selected = select_candidates(by_class, targets, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(c.row for c in selected) + "\n", encoding="utf-8")
    metadata = {
        "sources": {name: str(path) for name, path in sources.items()},
        "targets": targets,
        "seed": args.seed,
        "root": str(root),
        "available_rows_by_numerator": {key: len(value) for key, value in by_class.items()},
        "selected_rows": len(selected),
        "selected_coverage": coverage(selected),
        "missing_token_files": missing,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(selected)} rows to {args.output}")  # noqa: T201
    print(
        f"coverage: {metadata['selected_coverage']}; missing token files: {missing}"
    )  # noqa: T201


if __name__ == "__main__":
    main()
