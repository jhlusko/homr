"""Stage 2 pair extraction: real-scan crop <-> MusicXML token-sequence training
pairs from the IMSLP corpus, scoped in `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §7.

Everything the existing transformer training data (`convert_lieder.py`) is built
from is MuseScore's own rendered SVG output - clean engraving, exact symbol
positions for free. The model has likely never trained on a real historical scan's
varied engraving, ink density, page skew or scan artifacts. This module builds the
same (crop image, token sequence) pair shape from real IMSLP scans instead, reusing
`convert_lieder.py`'s own `MeasureCutter` (the stateful per-voice measure-popping
cutter that already handles clef/key/time-signature continuity across slice
boundaries) and `music_xml_parser.py` (the same MusicXML->token parser, needing
only the correct measure range sliced out per system rather than a new parser).

Eligibility for a system (per §7's own written rule): either an automated exact
bar-count match (`compare_bar_counts.py`'s own output), or a human `/compare`
"match"/"different_layout" judgment on that system's own scan page - a
"different_layout" judgment still confirms the crop/piece/starting-measure are
right, only the *page grouping* differs, which doesn't matter once pairing is
per-system via the flat position fix `compare_bar_counts.py` already applies.

Physical staff-box grouping: homr's own training examples are per staff-GROUP -
either a single voice staff, or a piano grand staff (treble+bass, drawn as one
brace-connected unit) - not per whole multi-staff system and not per single
physical staff line. Confirmed by reading `homr/main.py`'s own detection
pipeline (not assumed): `find_braces_brackets_and_grand_staff_lines` already
merges a piano's two physical staff lines into one `MultiStaff` entry *before*
`detect_imslp_systems.py` ever persists `staffBoxes` - so each saved staff box
already *is* one training-example unit (one XML voice's worth), not one raw
staff line. Verified empirically too: a random sample of 15 scores' first 3
systems each never showed 3 staff boxes for a piano+voice (1 vocal + 1
grand-staff) system - always 1 or 2, i.e. already one box per voice. An earlier
version of this module wrongly assumed the opposite (that a grandstaff voice
needed two consecutive raw staff boxes merged, by analogy with
`convert_lieder.py`'s own `merge_voice_with_next_one`, which really does operate
on two separate SVG staff areas) - that assumption does not hold for homr's own
real-scan detector output and was corrected before this module was trusted at
scale. `group_staff_boxes_into_voices` below is therefore a straight positional
zip: `staffBoxes[i]` pairs with XML voice `i`, in order - relying on the same
assumption `convert_lieder.py` already makes for its own synthetic pipeline that
MusicXML part order matches the printed top-to-bottom staff order (true by
convention for Lieder's piano+voice pieces - voice above, piano grand staff
below).

Every system in a piece - eligible or not - must still advance every voice's
`MeasureCutter` by its own measure count to keep the cutters correctly positioned
for the next system, mirroring `convert_xml_and_svg_file`'s own "remove the
measures from the cutter" discard path for a page/XML count mismatch. Ineligible
systems are walked past, not skipped outright.
"""

# flake8: noqa: T201

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

from homr.circle_of_fifths import strip_naturals
from training.omr_datasets.convert_lieder import (
    MeasureCutter,
    contains_only_supported_clefs,
    is_grandstaff,
)
from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl,
    load_lieder_file_tree,
    load_lieder_mxl_tree,
    load_lieder_scores,
    match_single_piece_scores,
)
from training.omr_datasets.music_xml_parser import Measure, music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.notation_sidecar import write_sidecar
from training.transformer.training_vocabulary import calc_ratio_of_tuplets, token_lines_to_str


# See recover_excluded_pairs.py: PIL's ~179M-pixel "decompression bomb" guard
# rejects some of our own full-resolution IMSLP page scans by raising, which would
# cost the whole score here rather than the one page. These are our own files.
Image.MAX_IMAGE_PIXELS = None


def eligible_system_positions(
    rows: list[dict], review: dict[str, dict]
) -> dict[str, set[int]]:
    """`{score_id: {system_position, ...}}` - the flat (whole-piece, not per-page)
    positions eligible for extraction, per this module's own eligibility rule.

    `rows` is `compare_bar_counts.py --rows-out`'s own output: one row per detected
    scan system, in the same page/system order `system_position` was assigned in, so
    a row's index within its own score's row group *is* that system's flat position -
    no separate position field needed.
    """
    by_score: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_score[row["score_id"]].append(row)

    confirmed_pages: dict[str, set[int]] = defaultdict(set)
    for entry in review.values():
        if entry["judgment"] in ("match", "different_layout"):
            confirmed_pages[entry["score_id"]].add(entry["page_index"])

    result: dict[str, set[int]] = {}
    for score_id, score_rows in by_score.items():
        eligible = set()
        for position, row in enumerate(score_rows):
            exact_match = row["detected"] == row["ground_truth"]
            reviewed_ok = row["page_index"] in confirmed_pages.get(score_id, set())
            if exact_match or reviewed_ok:
                eligible.add(position)
        result[score_id] = eligible
    return result


def flat_measure_ranges(ground_truth_pages: list[list[int]]) -> list[tuple[int, int]]:
    """`(start, end)` measure-index range (half-open) per *system*, cumulative across
    the whole piece - `fetch_lieder_ground_truth.py`'s own per-page list of per-system
    measure counts, flattened, same 0-based page/system order convention every other
    flat index in this corpus already uses. One entry per system, unlike
    `ocr_first_text_ground_truth.py`'s `page_measure_ranges`, which is per page.
    """
    ranges = []
    cursor = 0
    for page in ground_truth_pages:
        for count in page:
            ranges.append((cursor, cursor + count))
            cursor += count
    return ranges


def flat_detected_systems(systems_doc: dict) -> list[dict]:
    """Flattens a `detect_imslp_systems.py` yaml doc's systems into one page/system-
    order list, each entry carrying its own page's image filename alongside its
    `boundingBox`/`staffBoxes` - so a flat index lines up with `flat_measure_ranges`'s
    own flat index for the same score, the same pairing `compare_bar_counts.py`'s own
    `compare_one_score` already relies on.
    """
    result = []
    for page_key in sorted(systems_doc["pages"]):
        page = systems_doc["pages"][page_key]
        for system in page["systems"]:
            result.append({**system, "page_image": page["image"]})
    return result


def group_staff_boxes_into_voices(
    staff_boxes: list[dict], voice_is_grandstaff: list[bool]
) -> list[dict] | None:
    """One crop box per XML voice, in voice order - a straight positional pairing,
    since homr's own detector already merges a grand staff's two physical lines into
    one saved staff box before `detect_imslp_systems.py` ever persists `staffBoxes`
    (see this module's own docstring for how that was confirmed, not assumed).
    `voice_is_grandstaff` is accepted for the caller's own bookkeeping/documentation
    even though it plays no role in the grouping itself.

    Returns `None` (a skip signal, not an exception - the same "structure doesn't
    match, skip this one rather than guess" discipline `convert_xml_and_svg_file`
    already applies to its own SVG/XML measure-count mismatch) when the number of
    physical staff boxes doesn't match the piece's own voice count - most often
    because homr's own detector missed or merged a staff differently on this
    particular system, a real, expected source of loss at this stage.
    """
    if len(staff_boxes) != len(voice_is_grandstaff):
        return None
    return list(staff_boxes)


def extract_score_pairs(
    score_id: str,
    key: str,
    entry: dict,
    file_tree: dict,
    mxl_tree: dict | None,
    ground_truth: dict,
    systems_doc: dict,
    eligible: set[int],
    pngs_dir: Path,
    out_dir: Path,
) -> list[str]:
    """Extracts every eligible system's (crop, token-sequence) pair for one score,
    writing files into `out_dir` and returning the manifest lines
    `data_loader.py` already expects (`"image_path,tokens_path\\n"`, paths relative
    to whatever base the caller passes in - `main()` makes them git-root-relative).
    """
    mxl_bytes = fetch_mxl(entry, key, mxl_tree or file_tree)
    musicxml_bytes = unzip_mxl(mxl_bytes)
    voices: list[list[Measure]] = music_xml_string_to_tokens(musicxml_bytes.decode("utf-8"))
    cutters = [MeasureCutter(voice) for voice in voices]
    voice_is_grandstaff = [is_grandstaff(voice) for voice in voices]

    ranges = flat_measure_ranges(ground_truth["pages"])
    detected = flat_detected_systems(systems_doc)

    written: list[str] = []
    page_image_cache: dict[str, Image.Image] = {}
    for position, (start, end) in enumerate(ranges):
        if position >= len(detected):
            break
        count = end - start
        # Every voice must advance by this system's own measure count regardless of
        # eligibility, to stay correctly positioned for the next system.
        per_voice_measures = [cutter.extract_measures(count) for cutter in cutters]

        if position not in eligible:
            continue

        system = detected[position]
        groups = group_staff_boxes_into_voices(
            system.get("staffBoxes", []), voice_is_grandstaff
        )
        if groups is None:
            continue

        page_image_name = system["page_image"]
        if page_image_name not in page_image_cache:
            page_path = pngs_dir / page_image_name
            if not page_path.exists():
                continue
            page_image_cache[page_image_name] = Image.open(page_path).convert("L")
        page_image = page_image_cache[page_image_name]

        for voice_idx, (box, selected_measures) in enumerate(
            zip(groups, per_voice_measures, strict=True)
        ):
            if calc_ratio_of_tuplets(selected_measures) > 0.2:
                continue
            if not contains_only_supported_clefs(selected_measures):
                continue
            cleaned = strip_naturals(selected_measures)
            crop = page_image.crop(
                (box["left"], box["top"], box["left"] + box["width"], box["top"] + box["height"])
            )
            stem = f"{score_id}-sys{position}-v{voice_idx}"
            image_path = out_dir / f"{stem}.png"
            tokens_path = out_dir / f"{stem}.tokens"
            crop.save(image_path)
            tokens_content = token_lines_to_str(cleaned)
            tokens_path.write_text(tokens_content, encoding="utf-8")
            write_sidecar(str(tokens_path), cleaned)
            written.append(f"{image_path},{tokens_path}\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--rows", type=Path, required=True, help="compare_bar_counts.py --rows-out.")
    parser.add_argument("--review", type=Path, required=True, help="imslp_match_review.json.")
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--mxl-tree-cache", type=Path)
    parser.add_argument(
        "--ground-truth", type=Path, required=True,
        help="fetch_lieder_ground_truth.py's --out dir.",
    )
    parser.add_argument("--systems", type=Path, required=True, help="imslp_systems_with_staff_boxes dir.")
    parser.add_argument(
        "--pngs", type=Path, required=True, nargs="+",
        help="One or more imslp_pngs dirs (e.g. both imslp_pngs and imslp_pngs_new - the "
        "355-score and 121-OLiMPiC corpora keep separate png dirs even though "
        "imslp_systems_with_staff_boxes is shared). Resolved per score by whichever "
        "dir actually has that score's own subdirectory.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output dir for crops/tokens.")
    parser.add_argument("--manifest", type=Path, required=True, help="Output manifest (image,tokens csv).")
    parser.add_argument("--score-ids", type=Path, help="Optional subset, one id per line.")
    args = parser.parse_args()

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    eligible_by_score = eligible_system_positions(rows, review)

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)

    score_ids = sorted(eligible_by_score)
    if args.score_ids:
        wanted = {line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()}
        score_ids = [s for s in score_ids if s in wanted]
    matched = match_single_piece_scores(lieder, score_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    ok = skipped = failed = 0
    with open(args.manifest, "a", encoding="utf-8") as manifest:
        for score_id, (key, entry) in matched.items():
            eligible = eligible_by_score.get(score_id, set())
            if not eligible:
                skipped += 1
                continue
            gt_path = args.ground_truth / f"{score_id}.json"
            systems_path = args.systems / f"{score_id}.yaml"
            pngs_dir = next((d for d in args.pngs if (d / score_id).is_dir()), None)
            if not gt_path.exists() or not systems_path.exists() or pngs_dir is None:
                print(f"{score_id}: missing ground truth, detected systems, or pngs, skipping")
                skipped += 1
                continue
            try:
                ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
                systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
                lines = extract_score_pairs(
                    score_id, key, entry, file_tree, mxl_tree, ground_truth,
                    systems_doc, eligible, pngs_dir, args.out,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{score_id}: FAILED ({e})")
                failed += 1
                continue
            manifest.writelines(lines)
            manifest.flush()
            total_pairs += len(lines)
            print(f"{score_id}: {len(lines)} pair(s) extracted ({len(eligible)} eligible systems)")
            ok += 1

    print(f"{ok} scores processed, {skipped} skipped, {failed} failed, {total_pairs} total pairs")


if __name__ == "__main__":
    main()
