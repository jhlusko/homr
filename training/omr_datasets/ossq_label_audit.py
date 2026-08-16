# flake8: noqa: T201

"""
Per-class support for the notation the new heads would predict, broken down by split.

The design leaves the head configuration open until this is measured: whether level-5
and level-6 beam heads have enough examples to be learned or should start
deterministic/unsupported, and whether six slur slots are the right cap or the rare ones
should be masked and reported as overflow. Building the heads first and counting after
means rebuilding them, so this runs first.

It reads the *original* MusicXML, not the cleaned copy: the cleaning pass flattens slur
numbering to 1 and drops placement, which are two of the three things being counted here.

Counts are of what the source actually writes. MusicXML has no explicit "this note is
flagged, not beamed" marker - an eighth note with no <beam> is either automatically
beamed or deliberately unbeamed, which is the ambiguity beam materialization exists to
resolve - so the totals here are a lower bound on beam supervision, and will move once
materialization runs. Stem and slur counts are exact.

Usage:
    python -m training.omr_datasets.ossq_label_audit --dataset-root ../ossq-omr
"""

import argparse
import collections
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from training.omr_datasets.ossq_splits import load_split_manifest

#: A score's symbolic content is the same whichever track renders it, so labels are
#: grouped by the coarse split rather than per track.
_COARSE = {"train": "train", "valid": "valid", "test_synth": "test", "test_scanned": "test"}


@dataclass
class LabelCounts:
    notes: int = 0
    beam_states: collections.Counter[str] = field(default_factory=collections.Counter)
    beam_levels: collections.Counter[int] = field(default_factory=collections.Counter)
    stems: collections.Counter[str] = field(default_factory=collections.Counter)
    slur_slots: collections.Counter[int] = field(default_factory=collections.Counter)
    slur_placements: collections.Counter[str] = field(default_factory=collections.Counter)

    def add(self, other: "LabelCounts") -> None:
        self.notes += other.notes
        self.beam_states.update(other.beam_states)
        self.beam_levels.update(other.beam_levels)
        self.stems.update(other.stems)
        self.slur_slots.update(other.slur_slots)
        self.slur_placements.update(other.slur_placements)


def count_score(path: Path) -> LabelCounts:
    counts = LabelCounts()
    root = ET.parse(path).getroot()  # noqa: S314
    for note in root.iter("note"):
        counts.notes += 1
        for beam in note.findall("beam"):
            level = beam.get("number")
            if level is not None and level.isdigit():
                counts.beam_levels[int(level)] += 1
            counts.beam_states[(beam.text or "").strip()] += 1
        stem = note.find("stem")
        if stem is not None:
            counts.stems[(stem.text or "").strip()] += 1
        for slur in note.iter("slur"):
            number = slur.get("number")
            if number is not None and number.isdigit():
                counts.slur_slots[int(number)] += 1
            counts.slur_placements[slur.get("placement") or slur.get("orientation") or "none"] += 1
    return counts


def audit(dataset_root: Path) -> dict[str, LabelCounts]:
    """Count notation per split. Returns split name -> counts, plus 'all'."""
    manifest = load_split_manifest()
    manifest.check_no_leakage()
    by_split: dict[str, LabelCounts] = collections.defaultdict(LabelCounts)
    for score_id, tracks in manifest.scores.items():
        split = next(
            (_COARSE[tracks[t]] for t in ("synthetic", "scanned") if t in tracks),
            None,
        )
        if split is None:
            continue
        path = dataset_root / "scores" / manifest.paths[score_id] / f"{score_id}.musicxml"
        if not path.is_file():
            continue
        counts = count_score(path)
        by_split[split].add(counts)
        by_split["all"].add(counts)
    return dict(by_split)


def _table(title: str, per_split: dict[str, collections.Counter], keys: list) -> str:
    splits = ["train", "valid", "test", "all"]
    width = max(len(str(k)) for k in keys) if keys else 5
    lines = [title, "  " + "class".ljust(width) + "".join(f"{s:>12}" for s in splits)]
    for key in keys:
        row = "".join(f"{per_split.get(s, collections.Counter())[key]:>12,}" for s in splits)
        lines.append("  " + str(key).ljust(width) + row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class notation support for OSSQ by split.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2].parent / "ossq-omr",
        help="Path to an ossq-omr checkout (default: ../ossq-omr next to this repo).",
    )
    args = parser.parse_args()
    results = audit(args.dataset_root)
    if not results:
        raise SystemExit(f"No scores found under {args.dataset_root}/scores.")

    def counters(field_name: str) -> dict[str, collections.Counter]:
        return {split: getattr(counts, field_name) for split, counts in results.items()}

    def observed(field_name: str) -> list:
        return sorted({key for counts in results.values() for key in getattr(counts, field_name)})

    totals = "  ".join(
        f"{split}={results.get(split, LabelCounts()).notes:,}"
        for split in ("train", "valid", "test", "all")
    )
    print(f"notes: {totals}")
    for title, field_name, keys in (
        ("beam level", "beam_levels", list(range(1, 7))),
        ("beam state", "beam_states", observed("beam_states")),
        ("stem", "stems", observed("stems")),
        ("slur slot", "slur_slots", list(range(1, 7))),
        ("slur placement", "slur_placements", observed("slur_placements")),
    ):
        print()
        print(_table(title, counters(field_name), keys))


if __name__ == "__main__":
    main()
