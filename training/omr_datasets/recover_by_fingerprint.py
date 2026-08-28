"""Recover the Lieder systems that measure-count alignment could not place, by
fingerprinting what homr reads off each crop against the piece's real MusicXML.

Why this exists: :mod:`align_lieder_systems` has exactly one signal per system -
a measure count, one small integer.  On the 2026-08-27 rebuild that placed 2595
of 7610 systems and left 2478 skipped, 2147 ambiguous and 390 count-mismatched.
The ambiguous ones are not near-misses: their alternative-path margins run from
0.0 to a maximum of 1.8, so *none* of them reaches the 2.0 threshold, and no
threshold move recovers them without also accepting content-wrong ranges (margin
1.0 did exactly that on IMSLP632174).  Counts cannot separate these; content can.

This pass therefore reuses :mod:`fingerprint_measures`' method - run homr's own
Stage 2 model on the crop, flatten to a pitch sequence, and align that against the
same piece's real MusicXML note stream with `difflib` - and emits a pair only where
the match clears both coverage and similarity thresholds.

**Provenance: these labels are model-derived.**  That is the property
`align_lieder_systems` certifies it does *not* have (`model_predictions_used:
false`), because `recover_excluded_pairs.py` produced circular labels that way:
its spans came from the same upstream model family used as the evaluation
baseline.  Output here is written to its own directory and its own manifest,
tagged `provenance: fingerprint`, and must never enter a held-out evaluation set.
Training on it is a deliberate pseudo-labelling choice; evaluating against it is
not a choice at all, it is the bug this whole rebuild exists to remove.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from homr.circle_of_fifths import strip_naturals
from training.omr_datasets.build_clean_stage2_pairs import pick_png_root
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
from training.omr_datasets.fingerprint_measures import (
    MIN_COVERAGE,
    MIN_SIMILARITY,
    align_to_ground_truth,
    note_tokens,
)
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.notation_sidecar import write_sidecar
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.transformer.training_vocabulary import max_tuplet_ratio, calc_ratio_of_tuplets, token_lines_to_str

Image.MAX_IMAGE_PIXELS = None

#: Statuses worth attempting.  "aligned" is excluded by definition - those already
#: have a model-free label and must keep it.  A skipped system with no detected
#: measures at all is an illustration/ornament box, not music, so there is nothing
#: to fingerprint.
RECOVERABLE_STATUSES = frozenset({"ambiguous", "count_mismatch", "skipped"})


def recoverable_systems(score_report: dict) -> list[dict]:
    """The systems in one score's alignment report worth fingerprinting."""
    out = []
    for item in score_report.get("systems", []):
        if item.get("status") not in RECOVERABLE_STATUSES:
            continue
        if item.get("status") == "skipped" and not item.get("detected_measures"):
            continue
        out.append(item)
    return out


def ground_truth_stream(voice: list) -> tuple[list[str], list[int]]:
    """`(flat_note_tokens, measure_index_per_token)` for one voice of the real
    MusicXML - the stream a crop's own reading is aligned against."""
    flat: list[str] = []
    owner: list[int] = []
    for measure_index, measure in enumerate(voice):
        for token in note_tokens(measure):
            flat.append(token)
            owner.append(measure_index)
    return flat, owner


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
    parser.add_argument("--clean-manifest", type=Path, help="Refuse to duplicate these stems.")
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    parser.add_argument("--limit-scores", type=int)
    parser.add_argument("--score-ids", type=Path, help="Subset, one id per line (for sharding).")
    args = parser.parse_args()

    alignment_doc = json.loads(args.alignment.read_text(encoding="utf-8"))
    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)
    score_ids = sorted(alignment_doc["scores"])
    if args.score_ids:
        wanted = {line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()}
        score_ids = [sid for sid in score_ids if sid in wanted]
    if args.limit_scores:
        score_ids = score_ids[: args.limit_scores]
    matched = match_single_piece_scores(lieder, score_ids)

    already = set()
    if args.clean_manifest and args.clean_manifest.exists():
        for line in args.clean_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                already.add(Path(line.split(",", 1)[0]).stem)

    from homr.transformer.configs import Config
    from homr.transformer.staff2score import Staff2Score

    model = Staff2Score(Config())
    from training.omr_datasets.fingerprint_measures import predict_crop

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    records: list[dict] = []
    counts = {
        "attempted": 0, "trusted": 0, "written": 0, "rejected": 0,
        "no_notes": 0, "disagreement": 0,
    }

    for score_id in score_ids:
        report = alignment_doc["scores"].get(score_id) or {}
        targets = recoverable_systems(report)
        if not targets or score_id not in matched:
            continue
        systems_path = args.systems / f"{score_id}.yaml"
        if not systems_path.exists():
            continue

        key, entry = matched[score_id]
        try:
            musicxml = unzip_mxl(fetch_mxl(entry, key, mxl_tree or file_tree))
            voices = music_xml_string_to_tokens(musicxml.decode("utf-8"))
            grandstaff_flags = [is_grandstaff(voice) for voice in voices]
            systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
            detected = flat_detected_systems(systems_doc)
        except Exception as exc:  # noqa: BLE001
            print(f"{score_id}: prepare failed ({exc})")
            continue

        pngs_dir = pick_png_root(args.pngs, detected)
        if pngs_dir is None:
            print(f"{score_id}: no PNG root holds this score's pages")
            continue

        streams = [ground_truth_stream(voice) for voice in voices]
        page_cache: dict[str, Image.Image] = {}
        written = 0

        for item in targets:
            position = item["system"]
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

            # Phase 1: fingerprint every voice of this system.
            probes = []
            for voice_index, box in enumerate(groups):
                stem = f"{score_id}-sys{position}-v{voice_index}"
                if stem in already or voice_index >= len(streams):
                    continue
                crop = page.crop(
                    (
                        box["left"],
                        box["top"],
                        box["left"] + box["width"],
                        box["top"] + box["height"],
                    )
                )
                counts["attempted"] += 1
                probe_path = args.out / f"{stem}.probe.png"
                crop.save(probe_path)
                try:
                    predicted = predict_crop(model, probe_path)
                except Exception as exc:  # noqa: BLE001
                    probe_path.unlink(missing_ok=True)
                    print(f"{stem}: predict failed ({exc})")
                    continue
                probe_path.unlink(missing_ok=True)

                crop_tokens = note_tokens(predicted)
                gt_tokens, gt_owner = streams[voice_index]
                match = align_to_ground_truth(crop_tokens, gt_tokens, gt_owner)
                trusted = bool(
                    match
                    and match["coverage"] >= args.min_coverage
                    and match["similarity"] >= args.min_similarity
                )
                if match is None:
                    counts["no_notes"] += 1
                elif trusted:
                    counts["trusted"] += 1
                probes.append(
                    {
                        "stem": stem,
                        "voice": voice_index,
                        "crop": crop,
                        "match": match,
                        "trusted": trusted,
                        "notes": len(crop_tokens),
                    }
                )

            # Phase 2: one measure range for the whole system.
            #
            # Every staff in a printed system is vertically aligned and therefore
            # spans the SAME measures.  The vocal staff fingerprints cleanly (87%
            # written, median similarity 1.00 on the 20-score probe) while the piano
            # grand staff rarely does (13%, median 0.05) - its two hands interleave,
            # so homr's reading of the crop is not in the MusicXML voice's order and
            # sequence alignment cannot lock on.  Rather than discard the piano, take
            # the best-evidenced trusted match in the system and apply its range to
            # every staff.  The signal comes from the staff that can be read; the
            # geometry does the rest.
            best = max(
                (p for p in probes if p["trusted"]),
                key=lambda p: (p["match"]["similarity"], p["match"]["coverage"]),
                default=None,
            )
            if best is None:
                for p in probes:
                    counts["rejected"] += 1
                    records.append(
                        {
                            "stem": p["stem"], "score_id": score_id, "system": position,
                            "voice": p["voice"], "prior_status": item.get("status"),
                            "prior_margin": item.get("margin"), "predicted_notes": p["notes"],
                            "match": p["match"], "outcome": "no trusted match in system",
                        }
                    )
                continue

            start, end = best["match"]["start_measure"], best["match"]["end_measure"]
            for p in probes:
                own = p["match"] if p["trusted"] else None
                # A staff that read cleanly on its own but landed somewhere else is a
                # genuine disagreement - record it rather than silently overriding.
                disagrees = bool(
                    own and (own["start_measure"], own["end_measure"]) != (start, end)
                )
                record = {
                    "stem": p["stem"], "score_id": score_id, "system": position,
                    "voice": p["voice"], "prior_status": item.get("status"),
                    "prior_margin": item.get("margin"), "predicted_notes": p["notes"],
                    "match": p["match"],
                    "range_source": "own" if p is best else f"system voice v{best['voice']}",
                    "sibling_disagreement": disagrees,
                }
                if disagrees:
                    counts["disagreement"] += 1
                    records.append({**record, "outcome": "sibling disagreement, not written"})
                    continue

                measures = slice_voice_measures(voices[p["voice"]], start, end)
                if not measures or calc_ratio_of_tuplets(measures) > max_tuplet_ratio():
                    records.append({**record, "outcome": "unusable measure slice"})
                    continue
                if not contains_only_supported_clefs(measures):
                    records.append({**record, "outcome": "unsupported clef"})
                    continue
                cleaned = strip_naturals(measures)

                image_path = args.out / f"{p['stem']}.png"
                tokens_path = args.out / f"{p['stem']}.tokens"
                p["crop"].save(image_path)
                tokens_path.write_text(token_lines_to_str(cleaned), encoding="utf-8")
                write_sidecar(tokens_path, cleaned)
                manifest_lines.append(f"{image_path},{tokens_path}")
                records.append({**record, "outcome": "written", "range": [start, end]})
                counts["written"] += 1
                written += 1

        print(f"{score_id}: {written} recovered pair(s) from {len(targets)} unaligned systems")

    args.manifest.write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(
            {
                "provenance": "fingerprint",
                "model_predictions_used": True,
                "warning": (
                    "Model-derived labels. Safe to train on as pseudo-labels; NEVER "
                    "admissible in a held-out evaluation set - that is the circularity "
                    "the Lieder rebuild exists to remove."
                ),
                "alignment": str(args.alignment),
                "min_coverage": args.min_coverage,
                "min_similarity": args.min_similarity,
                "counts": counts,
                "pairs": len(manifest_lines),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"attempted {counts['attempted']}, trusted {counts['trusted']}, "
        f"written {counts['written']}, rejected {counts['rejected']}, "
        f"no-notes {counts['no_notes']}, disagreements {counts['disagreement']}"
    )


if __name__ == "__main__":
    main()
