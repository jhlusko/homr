"""Build a review set showing where two checkpoints disagree, as rendered scores.

An aggregate delta says a checkpoint is better or worse; it never says *how*. A small
average gain hiding a handful of large regressions is a different result from a
uniform small gain, and only the second is safe to ship.

The two predictions are reconstructed into MusicXML and handed to the same
checkpoint-compare view the other review sets use, with the scan crop above as the
arbiter. Reading engraved music is what a musician can actually adjudicate; a wall of
token strings is not, however precisely it encodes the same disagreement.

Staves are ordered by net change with the regressions first, because those are what a
reviewer needs to look at.
"""

# flake8: noqa: T201

import argparse
import html
import json
import shutil
from pathlib import Path

PAD = "\x00missing"
#: The branches a reader can actually adjudicate from a scan.  Rhythm and pitch carry
#: the disagreement; the rest are shown in the totals but not diffed position by
#: position, because a colour-coded wall of six streams is unreadable.
DIFF_BRANCHES = ("rhythm", "pitch")
ALL_BRANCHES = ("rhythm", "pitch", "lift", "articulation", "slur", "position")


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["tokens"]] = record
    return out


def accuracy(record: dict, branches=ALL_BRANCHES) -> tuple[int, int]:
    hit = total = 0
    for branch in branches:
        want = record[f"{branch}_reference"]
        got = record[f"{branch}_predicted"]
        hit += sum(1 for w, g in zip(want, got) if w == g)
        total += len(want)
    return hit, total


def classify(want: str, old: str, new: str) -> str:
    old_ok, new_ok = old == want, new == want
    if old_ok and new_ok:
        return "agree"
    if new_ok:
        return "gain"
    if old_ok:
        return "regression"
    return "both-wrong"


def show(token: str) -> str:
    """A padded position is an absence, not a token; say so rather than printing NULs."""
    if token.startswith(PAD):
        return "—"
    return token


def diff_rows(old: dict, new: dict) -> list[dict]:
    rows = []
    for branch in DIFF_BRANCHES:
        want = old[f"{branch}_reference"]
        a = old[f"{branch}_predicted"]
        b = new[f"{branch}_predicted"]
        cells = [
            {"want": show(w), "old": show(x), "new": show(y), "cls": classify(w, x, y)}
            for w, x, y in zip(want, a, b)
        ]
        rows.append({"branch": branch, "cells": cells})
    return rows




def symbols_from(record: dict, side: str) -> list:
    """Rebuild EncodedSymbols from one side of a scored record.

    `base_predictions` stores each decoder branch as its own parallel array, which is
    exactly the six fields an EncodedSymbol carries - so the symbol stream can be
    reassembled without re-running the model. Padded positions are dropped: they mark
    an absence, and emitting them would put a nonsense symbol into the engraving.
    """
    from homr.transformer.vocabulary import EncodedSymbol

    key = "reference" if side == "reference" else "predicted"
    columns = {b: record[f"{b}_{key}"] for b in ALL_BRANCHES}
    out = []
    for index in range(len(columns["rhythm"])):
        values = {b: columns[b][index] for b in ALL_BRANCHES}
        if any(str(v).startswith(PAD) for v in values.values()):
            continue
        out.append(
            EncodedSymbol(
                values["rhythm"], values["pitch"], values["lift"],
                values["articulation"], values["slur"], values["position"],
            )
        )
    return out


def write_xml(symbols: list, destination: Path) -> int:
    import xml.etree.ElementTree as ET

    from homr.music_xml_generator import XmlGeneratorArguments, generate_xml

    xml = generate_xml(XmlGeneratorArguments(None, None, None), [symbols], "")
    ET.ElementTree(xml).write(destination, encoding="unicode", xml_declaration=True)
    return sum(1 for s in symbols if s.rhythm.startswith("barline"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", type=Path, required=True, help="baseline predictions .jsonl")
    parser.add_argument("--new", type=Path, required=True, help="candidate predictions .jsonl")
    parser.add_argument("--index", type=Path, required=True, help="image,tokens index")
    parser.add_argument("--out", type=Path, required=True, help="review set directory")
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Items to emit, worst regressions first. 0 = every differing stave.",
    )
    args = parser.parse_args()

    old, new = load_jsonl(args.old), load_jsonl(args.new)
    images = {}
    for line in args.index.read_text(encoding="utf-8").splitlines():
        if line.strip():
            image, tokens = line.split(",", 1)
            images[tokens.strip()] = Path(image.strip())

    crops = args.out / "crops"
    scores = args.out / "scores"
    crops.mkdir(parents=True, exist_ok=True)
    scores.mkdir(parents=True, exist_ok=True)

    improved = regressed = unchanged = 0
    candidates = []
    for key in sorted(set(old) & set(new)):
        hit_old, total = accuracy(old[key])
        hit_new, _ = accuracy(new[key])
        if hit_new > hit_old:
            improved += 1
        elif hit_new < hit_old:
            regressed += 1
        else:
            unchanged += 1
        rows = diff_rows(old[key], new[key])
        gains = sum(1 for r in rows for c in r["cells"] if c["cls"] == "gain")
        regs = sum(1 for r in rows for c in r["cells"] if c["cls"] == "regression")
        if not (gains or regs):
            continue
        candidates.append({
            "key": key,
            "delta": hit_new / max(total, 1) - hit_old / max(total, 1),
            "old_accuracy": round(hit_old / max(total, 1), 4),
            "new_accuracy": round(hit_new / max(total, 1), 4),
            "gains": gains,
            "regressions": regs,
        })

    candidates.sort(key=lambda c: c["delta"])
    if args.limit:
        candidates = candidates[: args.limit]

    manifest = []
    for item in candidates:
        key = item["key"]
        stem = Path(key).stem
        source = images.get(key)
        if not source or not source.is_file():
            continue
        shutil.copy2(source, crops / f"{stem}.png")
        left_bars = write_xml(symbols_from(old[key], "predicted"), scores / f"{stem}__left.musicxml")
        right_bars = write_xml(symbols_from(new[key], "predicted"), scores / f"{stem}__right.musicxml")
        write_xml(symbols_from(old[key], "reference"), scores / f"{stem}__reference.musicxml")
        parsed = stem.rsplit("-sys", 1)
        manifest.append({
            "id": stem,
            "score_id": parsed[0],
            "system": int(parsed[1].split("-v")[0]) if len(parsed) > 1 else 0,
            "voice": int(stem.rsplit("-v", 1)[1]) if "-v" in stem else 0,
            "left_bars": left_bars,
            "right_bars": right_bars,
            "has_right": True,
            "delta": round(item["delta"], 4),
            "old_accuracy": item["old_accuracy"],
            "new_accuracy": item["new_accuracy"],
            "gains": item["gains"],
            "regressions": item["regressions"],
        })

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(set(old) & set(new))} staves: {improved} improved, {regressed} regressed, "
          f"{unchanged} unchanged")
    print(f"wrote {len(manifest)} items -> {args.out}")


if __name__ == "__main__":
    main()
