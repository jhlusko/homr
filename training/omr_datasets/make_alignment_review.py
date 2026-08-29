"""Create a deterministic, stratified review set for rebuilt Lieder labels."""

# flake8: noqa: T201

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.omr_datasets.stage2_pair_review_server import parse_stem
from training.transformer.training_vocabulary import read_tokens


def topology_by_system(score_report: dict) -> dict[int, str]:
    result = {}
    for move in score_report.get("moves", []):
        if move["kind"] != "match":
            continue
        scan_size = move["scan_end"] - move["scan_start"]
        source_size = move["source_end"] - move["source_start"]
        if scan_size > 1 and source_size > 1:
            topology = "many-to-many"
        elif scan_size > 1:
            topology = "reference-line-split"
        elif source_size > 1:
            topology = "reference-lines-merged"
        else:
            topology = "one-to-one"
        for system in range(move["scan_start"], move["scan_end"]):
            result[system] = topology
    return result


def _stable_key(stem: str) -> bytes:
    return hashlib.sha256(("alignment-review-v1:" + stem).encode("utf-8")).digest()


def stratified_sample(items: list[dict], limit: int) -> list[dict]:
    """Round-robin across topology and confidence bands, deterministically."""
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        band = "near-threshold" if item["margin"] < 5 else "high-margin"
        strata[(item["kind"], band)].append(item)
    for values in strata.values():
        values.sort(key=lambda item: _stable_key(item["id"]))

    selected = []
    keys = sorted(strata)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < limit:
                selected.append(strata[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _manifest_map(paths: list[Path]) -> dict[str, tuple[Path, Path]]:
    result = {}
    for manifest in paths:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image, tokens = line.split(",", 1)
            result[Path(image).stem] = (Path(image), Path(tokens))
    return result


def _write_xml(tokens: Path, destination: Path) -> int:
    symbols = read_tokens(str(tokens))
    # The six legacy token fields intentionally omit structured notation.  Review
    # engraving must use the optional sidecar when it exists: it preserves beam levels,
    # stem direction, slur span slots/placement, ties, dynamics, and advance metadata.
    # Without it, a perfectly valid pair of simultaneous upper/lower slurs is rebuilt
    # through the generator's necessarily ambiguous fallback stack and can look crossed.
    attach_sidecar(tokens, symbols)
    xml = generate_xml(XmlGeneratorArguments(None, None, None), [symbols], "")
    ET.ElementTree(xml).write(destination, encoding="unicode", xml_declaration=True)
    return sum(symbol.rhythm == "barline" for symbol in symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, nargs="+", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", default="validation-candidate")
    parser.add_argument(
        "--statuses", nargs="+", default=["aligned"],
        help="Alignment statuses eligible for review (default: aligned). Quarantined "
        "statuses are reviewable only when an explicitly separate label manifest is supplied.",
    )
    args = parser.parse_args()

    clean = _manifest_map([args.manifest])
    old = _manifest_map(args.old_manifest)
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    candidates = []
    for stem in sorted(clean):
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        score_id, system, voice = parsed
        report = alignment["scores"].get(score_id)
        if report is None:
            continue
        system_item = next(
            (item for item in report["systems"] if item["system"] == system), None
        )
        if system_item is None or system_item["status"] not in args.statuses:
            continue
        topologies = topology_by_system(report)
        candidates.append(
            {
                "id": stem,
                "kind": topologies.get(system, "one-to-one"),
                "margin": float(system_item["margin"]),
                "score_id": score_id,
                "system": system,
                "voice": voice,
                "split": args.split,
                "alignment_status": system_item["status"],
                "has_old_label": stem in old,
            }
        )

    chosen = stratified_sample(candidates, args.limit)
    crops = args.out / "crops"
    scores = args.out / "scores"
    crops.mkdir(parents=True, exist_ok=True)
    scores.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in chosen:
        stem = item["id"]
        image, tokens = clean[stem]
        shutil.copy2(image, crops / f"{stem}.png")
        corrected_bars = _write_xml(tokens, scores / f"{stem}__corrected.musicxml")
        if stem in old:
            old_bars = _write_xml(old[stem][1], scores / f"{stem}__old.musicxml")
        else:
            # Keep the compare view functional while making the absence explicit
            # in metadata; judgment is always about the corrected left pane.
            shutil.copy2(
                scores / f"{stem}__corrected.musicxml",
                scores / f"{stem}__old.musicxml",
            )
            old_bars = None
        manifest.append(
            {**item, "corrected_bars": corrected_bars, "old_bars": old_bars}
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(candidates)} eligible candidates; wrote {len(manifest)} stratified items")


if __name__ == "__main__":
    main()
