"""Build corrected Lieder staff pairs from explicit whole-score alignments.

Only systems marked ``aligned`` by :mod:`align_lieder_systems` are emitted.  The
historical model-recovered manifest is copied to an explicit quarantine manifest
for auditability; none of its pairs can enter the clean output by accident.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from homr.circle_of_fifths import strip_naturals
from training.omr_datasets.convert_lieder import (
    contains_only_supported_clefs,
    is_grandstaff,
)
from training.omr_datasets.extract_stage2_pairs import (
    flat_detected_systems,
    group_staff_boxes_into_voices,
)
from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl,
    load_lieder_file_tree,
    load_lieder_mxl_tree,
    load_lieder_scores,
    match_single_piece_scores,
)
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.notation_sidecar import write_sidecar
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.omr_datasets.audit_clean_stage2_pairs import MEASURE_DIVIDERS
from training.omr_datasets.audit_label_consistency import overfull_bars
from training.omr_datasets.system_count_alignment import aligned_ranges
from training.transformer.training_vocabulary import calc_ratio_of_tuplets, token_lines_to_str

Image.MAX_IMAGE_PIXELS = None


def pick_png_root(roots: list[Path], detected: list[dict]) -> Path | None:
    """The PNG root that actually holds every page these detected systems name.

    Two roots can each contain a directory for the same score under different file
    naming (`IMSLP621830-001-000.png` vs `IMSLP621830-p002.png`), so selecting by
    directory name alone can pick one whose files do not exist.
    """
    pages = {system["page_image"] for system in detected}
    if not pages:
        return None
    return next((root for root in roots if all((root / name).exists() for name in pages)), None)


def quarantine_recovered(source: Path, destination: Path, report_path: Path | None) -> int:
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if report_path:
        report_path.write_text(
            json.dumps(
                {
                    "reason": "model-derived partial-span labels; excluded from training and evaluation",
                    "recoverable_files_deleted": False,
                    "pairs": len(lines),
                    "source_manifest": str(source),
                    "quarantine_manifest": str(destination),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--mxl-tree-cache", type=Path)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--pngs", type=Path, required=True, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overfull-out", type=Path,
                        help="Where to quarantine pairs with an overfull bar.")
    parser.add_argument("--overfull-manifest", type=Path)
    parser.add_argument("--recovered-manifest", type=Path)
    parser.add_argument("--quarantine-manifest", type=Path)
    parser.add_argument("--quarantine-report", type=Path)
    args = parser.parse_args()

    if bool(args.recovered_manifest) != bool(args.quarantine_manifest):
        parser.error("--recovered-manifest and --quarantine-manifest must be supplied together")

    alignment_doc = json.loads(args.alignment.read_text(encoding="utf-8"))
    if alignment_doc.get("model_predictions_used") is not False:
        raise SystemExit("alignment provenance is missing or uses model predictions")

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)
    score_ids = sorted(alignment_doc["scores"])
    matched = match_single_piece_scores(lieder, score_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    audit = []
    manifest_lines = []
    inconsistent = 0
    overfull_skipped = 0
    overfull_lines: list[str] = []
    overfull_detail: list[dict] = []

    for score_id in score_ids:
        ranges = aligned_ranges(alignment_doc["scores"][score_id])
        if not ranges or score_id not in matched:
            continue
        systems_path = args.systems / f"{score_id}.yaml"
        if not systems_path.exists():
            audit.append({"score_id": score_id, "status": "missing inputs"})
            continue

        key, entry = matched[score_id]
        try:
            musicxml = unzip_mxl(fetch_mxl(entry, key, mxl_tree or file_tree))
            voices = music_xml_string_to_tokens(musicxml.decode("utf-8"))
            grandstaff_flags = [is_grandstaff(voice) for voice in voices]
            systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
            detected = flat_detected_systems(systems_doc)
        except Exception as exc:  # noqa: BLE001
            audit.append({"score_id": score_id, "status": "prepare failed", "error": str(exc)})
            continue

        # Pick the root that actually holds this score's pages, not merely one with a
        # directory of that name: IMSLP621830 and IMSLP622484 exist under both roots
        # with different file naming, and choosing by directory name alone builds
        # paths to files that are not there - here that is silent, since a missing
        # page just `continue`s and the pairs vanish without a word.
        pngs_dir = pick_png_root(args.pngs, detected)
        if pngs_dir is None:
            audit.append({"score_id": score_id, "status": "missing inputs"})
            continue

        page_cache: dict[str, Image.Image] = {}
        written = 0
        for position, (start, end) in sorted(ranges.items()):
            if position >= len(detected):
                continue
            system = detected[position]
            groups = group_staff_boxes_into_voices(system.get("staffBoxes", []), grandstaff_flags)
            if groups is None:
                continue
            page_name = system["page_image"]
            if page_name not in page_cache:
                page_path = pngs_dir / page_name
                if not page_path.exists():
                    continue
                page_cache[page_name] = Image.open(page_path).convert("L")
            page = page_cache[page_name]

            for voice_index, box in enumerate(groups):
                measures = slice_voice_measures(voices[voice_index], start, end)
                if not measures or calc_ratio_of_tuplets(measures) > 0.2:
                    continue
                if not contains_only_supported_clefs(measures):
                    continue
                cleaned = strip_naturals(measures)
                # A pair whose own label carries a different number of measure
                # dividers than its assigned span is internally inconsistent - the
                # exact class of defect this rebuild exists to exclude.  Seven of
                # 3189 on 2026-08-27 did (all with MORE dividers than the span, and
                # none from adjacent `:||:` glyph pairs, so they are real
                # disagreements rather than a counting artefact).  Drop them here so
                # the corpus is consistent by construction and the audit stays a
                # strict independent check rather than being relaxed to accommodate.
                # An implied tuplet - engraved with no bracket and no numeral - is
                # recorded by neither the transcription nor the model, so the bar comes
                # out longer than the staff's prevailing one. Training on it teaches the
                # model to write overfull bars. 45 across 25 scores; reviewed examples
                # ran 1.062 and 1.125 whole notes against a 4/4 bar.
                overfull = overfull_bars(cleaned)
                if overfull:
                    # Quarantined, not discarded. 417 pairs is ~10% of the corpus and
                    # the loss is asymmetric: implied tuplets are concentrated in the
                    # densest, most interesting writing, so dropping them silently
                    # trains the model on the easy half of the repertoire. Written to
                    # their own directory and manifest so they can be reviewed and, if
                    # a representation is found, recovered.
                    stem = f"{score_id}-sys{position}-v{voice_index}"
                    if args.overfull_out:
                        args.overfull_out.mkdir(parents=True, exist_ok=True)
                        image_path = args.overfull_out / f"{stem}.png"
                        tokens_path = args.overfull_out / f"{stem}.tokens"
                        crop = page.crop(
                            (box["left"], box["top"],
                             box["left"] + box["width"], box["top"] + box["height"])
                        )
                        crop.save(image_path)
                        tokens_path.write_text(token_lines_to_str(cleaned), encoding="utf-8")
                        write_sidecar(tokens_path, cleaned)
                        overfull_lines.append(f"{image_path},{tokens_path}")
                        overfull_detail.append({"stem": stem, "bars": overfull,
                                                "measures": end - start})
                    overfull_skipped += 1
                    continue
                divider_count = sum(
                    1 for line in token_lines_to_str(cleaned).splitlines()
                    if line.split() and line.split()[0] in MEASURE_DIVIDERS
                )
                if divider_count != end - start:
                    inconsistent += 1
                    continue
                crop = page.crop(
                    (
                        box["left"],
                        box["top"],
                        box["left"] + box["width"],
                        box["top"] + box["height"],
                    )
                )
                stem = f"{score_id}-sys{position}-v{voice_index}"
                image_path = args.out / f"{stem}.png"
                tokens_path = args.out / f"{stem}.tokens"
                crop.save(image_path)
                tokens_path.write_text(token_lines_to_str(cleaned), encoding="utf-8")
                write_sidecar(tokens_path, cleaned)
                manifest_lines.append(f"{image_path},{tokens_path}")
                written += 1
        audit.append(
            {
                "score_id": score_id,
                "status": "built",
                "aligned_systems": len(ranges),
                "pairs": written,
            }
        )
        print(f"{score_id}: {written} clean pair(s) from {len(ranges)} aligned systems")

    args.manifest.write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8"
    )
    quarantine_count = 0
    if args.recovered_manifest and args.quarantine_manifest:
        quarantine_count = quarantine_recovered(
            args.recovered_manifest, args.quarantine_manifest, args.quarantine_report
        )
    args.report.write_text(
        json.dumps(
            {
                "alignment": str(args.alignment),
                "model_predictions_used": False,
                "pairs": len(manifest_lines),
                "span_inconsistent_pairs_skipped": inconsistent,
                "overfull_bar_pairs_skipped": overfull_skipped,
                "overfull_detail": overfull_detail,
                "quarantined_recovered_pairs": quarantine_count,
                "scores": audit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(manifest_lines)} clean pairs written")
    print(f"{inconsistent} pair(s) skipped: divider count disagreed with aligned span")
    print(f"{overfull_skipped} pair(s) skipped: overfull bar (implied tuplet)")
    if args.overfull_manifest:
        args.overfull_manifest.write_text(
            "\n".join(overfull_lines) + ("\n" if overfull_lines else ""), encoding="utf-8"
        )
        print(f"  quarantined to {args.overfull_manifest}")
    print(f"{quarantine_count} historical recovered pairs quarantined (files preserved)")


if __name__ == "__main__":
    main()
