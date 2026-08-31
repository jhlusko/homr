from pathlib import Path

import pytest

from training.omr_datasets.build_numerator_replay import (
    collect_candidates,
    coverage,
    select_candidates,
)


def _write_index(tmp_path: Path, rows: dict[str, str]) -> Path:
    lines = []
    for name, text in rows.items():
        token_path = tmp_path / f"{name}.tokens"
        token_path.write_text(text, encoding="utf-8")
        lines.append(f"image/{name}.png,{token_path.relative_to(tmp_path)}")
    index = tmp_path / "index.txt"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def test_selects_requested_numerators_without_duplicate_rows(tmp_path: Path) -> None:
    index = _write_index(
        tmp_path,
        {
            "five_a": "timeSignatureBeats_5 timeSignature/4",
            "five_b": "timeSignatureBeats_5 timeSignature/8",
            "twelve": "timeSignatureBeats_12 timeSignature/8",
            "both": "timeSignatureBeats_5 timeSignatureBeats_12",
        },
    )
    candidates, missing = collect_candidates({"test": index}, {5, 12}, tmp_path)
    selected = select_candidates(candidates, {5: 2, 12: 2}, seed=42)

    assert missing == 0
    assert len({candidate.row for candidate in selected}) == len(selected)
    assert coverage(selected)[5] >= 2
    assert coverage(selected)[12] >= 2


def test_refuses_an_undersupplied_target(tmp_path: Path) -> None:
    index = _write_index(tmp_path, {"five": "timeSignatureBeats_5"})
    candidates, _ = collect_candidates({"test": index}, {5, 12}, tmp_path)

    with pytest.raises(ValueError, match="numerator 12"):
        select_candidates(candidates, {5: 1, 12: 1}, seed=42)
