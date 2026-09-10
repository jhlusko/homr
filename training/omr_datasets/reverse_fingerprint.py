"""Assign every labelled measure to the system crop that contains it.

Forward fingerprinting (:mod:`recover_by_fingerprint`) asks, per crop, "where in
the label does this crop sit?" - and answers each crop independently.  A crop
whose OMR reading is poor simply fails, and nothing else can rescue it: on the
2026-08-27 20-score probe the piano grand staff matched at median similarity 0.05
because its two hands interleave and the reading is not in the label's order.

Running it in reverse removes that isolation.  Every labelled measure belongs to
exactly one system, and systems appear in reading order, so the assignment is not
N independent lookups but one *global, monotone* segmentation: cut the score's
ordered measure sequence into consecutive spans, one per system, in order.  A
system that reads badly is then pinned by its neighbours rather than dropped -
the measures on either side are already claimed, so what is left is its span
whether or not its own reading could prove it.

This also yields a completeness statement the forward pass cannot make: every
labelled bar is accounted for, or explicitly is not.

Provenance is identical to the forward pass - the crop readings come from homr's
own Stage 2 model, so these labels are model-derived and inadmissible in a
held-out evaluation set.  See :mod:`recover_by_fingerprint`.
"""

# flake8: noqa: T201

import difflib
from dataclasses import dataclass

#: A single printed system rarely holds more than this many measures; bounding the
#: span keeps the DP at O(systems x measures x MAX_SPAN) instead of quadratic in
#: measures, and rejects degenerate segmentations that hand one system half a piece.
MAX_SPAN = 16

#: A system may legitimately be assigned nothing - a narrow illustration/ornament
#: detection is not music.  Charge a small constant so the DP prefers to place a
#: system when the evidence is anywhere near adequate, but can still skip one.
EMPTY_SPAN_SCORE = 0.15


@dataclass(frozen=True)
class Assignment:
    system: int
    start_measure: int
    end_measure: int
    score: float


#: What a span scores when the crop yielded no tokens at all.  Slightly above
#: EMPTY_SPAN_SCORE so the DP prefers to *place* an unreadable system rather than
#: empty it, letting its neighbours pin the boundaries - which is the whole point of
#: segmenting globally.  Scoring it zero made the empty move always win, so
#: neighbour-pinning only ever worked for systems that were already readable.
UNREADABLE_SPAN_SCORE = 0.2


def span_score(crop_tokens: list[str], gt_tokens: list[str], start: int, end: int,
               owner: list[int]) -> float:
    """How well this crop's reading explains the label's measures ``[start, end)``.

    Uses `difflib`'s similarity on the flat note sequence rather than a per-measure
    comparison, for the same reason the forward pass does: the crop's own barline
    reading may be wrong, and anything keyed to measure boundaries would inherit
    that error.  An empty candidate span scores zero, never a free win.
    """
    lo = _first_index(owner, start)
    hi = _first_index(owner, end)
    window = gt_tokens[lo:hi]
    if not window:
        return 0.0
    if not crop_tokens:
        # Nothing was read off this crop.  That is an abstention, not evidence of
        # emptiness, so let the neighbours decide where it sits.
        return UNREADABLE_SPAN_SCORE
    matcher = difflib.SequenceMatcher(None, window, crop_tokens, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks())
    # Penalise both directions: a span that leaves much of the crop unexplained is
    # too short, one much longer than the crop's own content is too long.
    return matched / max(len(window), len(crop_tokens))


def _first_index(owner: list[int], measure: int) -> int:
    lo, hi = 0, len(owner)
    while lo < hi:
        mid = (lo + hi) // 2
        if owner[mid] < measure:
            lo = mid + 1
        else:
            hi = mid
    return lo


def assign_measures_to_systems(
    system_tokens: list[list[str]],
    gt_tokens: list[str],
    owner: list[int],
    n_measures: int,
    max_span: int = MAX_SPAN,
) -> list[Assignment]:
    """Cut ``n_measures`` measures into one consecutive span per system, in order.

    Returns one :class:`Assignment` per system.  Spans never overlap and never run
    backwards; a system assigned nothing gets ``start == end``.
    """
    n_systems = len(system_tokens)
    if n_systems == 0:
        return []

    neg = float("-inf")
    # best[s][m] = best total score for systems [0, s) covering measures [0, m)
    best = [[neg] * (n_measures + 1) for _ in range(n_systems + 1)]
    back = [[0] * (n_measures + 1) for _ in range(n_systems + 1)]
    best[0][0] = 0.0

    for s in range(n_systems):
        tokens = system_tokens[s]
        for m in range(n_measures + 1):
            if best[s][m] == neg:
                continue
            base = best[s][m]
            # The system takes nothing.
            if base + EMPTY_SPAN_SCORE > best[s + 1][m]:
                best[s + 1][m] = base + EMPTY_SPAN_SCORE
                back[s + 1][m] = m
            limit = min(n_measures, m + max_span)
            for end in range(m + 1, limit + 1):
                total = base + span_score(tokens, gt_tokens, m, end, owner)
                if total > best[s + 1][end]:
                    best[s + 1][end] = total
                    back[s + 1][end] = m

    # Every labelled measure must be claimed: finish at n_measures.
    end_m = n_measures
    if best[n_systems][end_m] == neg:
        end_m = max(
            (m for m in range(n_measures + 1) if best[n_systems][m] != neg),
            default=0,
        )

    out: list[Assignment] = []
    m = end_m
    for s in range(n_systems, 0, -1):
        start = back[s][m]
        out.append(
            Assignment(
                system=s - 1,
                start_measure=start,
                end_measure=m,
                score=span_score(system_tokens[s - 1], gt_tokens, start, m, owner),
            )
        )
        m = start
    out.reverse()
    return out


def unclaimed_measures(assignments: list[Assignment], n_measures: int) -> list[int]:
    """Labelled measures no system was given - the completeness statement."""
    claimed = set()
    for a in assignments:
        claimed.update(range(a.start_measure, a.end_measure))
    return [m for m in range(n_measures) if m not in claimed]


def main() -> None:  # noqa: C901
    import argparse
    import json
    from pathlib import Path

    import yaml
    from PIL import Image

    from homr.circle_of_fifths import strip_naturals
    from training.omr_datasets.build_clean_stage2_pairs import pick_png_root
    from training.omr_datasets.convert_lieder import contains_only_supported_clefs, is_grandstaff
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
    from training.omr_datasets.fingerprint_measures import note_tokens, predict_crop
    from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
    from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
    from training.omr_datasets.notation_sidecar import write_sidecar
    from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
    from training.transformer.training_vocabulary import calc_ratio_of_tuplets, token_lines_to_str

    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alignment", type=Path, required=True, help="For the score list.")
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--mxl-tree-cache", type=Path)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--pngs", type=Path, required=True, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, help="Reuse and update crop readings here.")
    parser.add_argument("--score-ids", type=Path)
    parser.add_argument("--limit-scores", type=int)
    parser.add_argument(
        "--min-mean-score", type=float, default=0.45,
        help="Reject a score's whole segmentation below this mean span score - a "
        "global solution is only as trustworthy as its overall fit.",
    )
    args = parser.parse_args()

    alignment_doc = json.loads(args.alignment.read_text(encoding="utf-8"))
    score_ids = sorted(alignment_doc["scores"])
    if args.score_ids:
        wanted = {x.strip() for x in args.score_ids.read_text().splitlines() if x.strip()}
        score_ids = [s for s in score_ids if s in wanted]
    if args.limit_scores:
        score_ids = score_ids[: args.limit_scores]

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)
    matched = match_single_piece_scores(lieder, score_ids)

    cache: dict[str, list[str]] = {}
    if args.prediction_cache and args.prediction_cache.exists():
        cache = json.loads(args.prediction_cache.read_text(encoding="utf-8"))
        print(f"loaded {len(cache)} cached crop readings")

    from homr.transformer.configs import Config
    from homr.transformer.staff2score import Staff2Score

    model = Staff2Score(Config())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    records: list[dict] = []
    stats = {"scores": 0, "accepted": 0, "rejected": 0, "pairs": 0, "unclaimed_measures": 0}

    for score_id in score_ids:
        if score_id not in matched:
            continue
        systems_path = args.systems / f"{score_id}.yaml"
        if not systems_path.exists():
            continue
        key, entry = matched[score_id]
        try:
            musicxml = unzip_mxl(fetch_mxl(entry, key, mxl_tree or file_tree))
            voices = music_xml_string_to_tokens(musicxml.decode("utf-8"))
            grandstaff_flags = [is_grandstaff(v) for v in voices]
            systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
            detected = flat_detected_systems(systems_doc)
        except Exception as exc:  # noqa: BLE001
            print(f"{score_id}: prepare failed ({exc})")
            continue
        pngs_dir = pick_png_root(args.pngs, detected)
        if pngs_dir is None or not voices:
            continue

        gt_tokens, owner = [], []
        for index, measure in enumerate(voices[0]):
            for token in note_tokens(measure, include_rests=True):
                gt_tokens.append(token)
                owner.append(index)
        n_measures = len(voices[0])
        if not gt_tokens:
            continue

        stats["scores"] += 1
        page_cache: dict[str, Image.Image] = {}
        readings: list[list[str]] = []
        boxes_per_system: list[list | None] = []

        for position, system in enumerate(detected):
            groups = group_staff_boxes_into_voices(system.get("staffBoxes", []), grandstaff_flags)
            boxes_per_system.append(groups)
            stem = f"{score_id}-sys{position}-v0"
            if stem in cache:
                readings.append(cache[stem])
                continue
            if groups is None:
                readings.append([])
                continue
            page_name = system["page_image"]
            if page_name not in page_cache:
                page_path = pngs_dir / page_name
                if not page_path.exists():
                    readings.append([])
                    continue
                page_cache[page_name] = Image.open(page_path).convert("L")
            box = groups[0]
            crop = page_cache[page_name].crop(
                (box["left"], box["top"], box["left"] + box["width"], box["top"] + box["height"])
            )
            probe = args.out / f"{stem}.probe.png"
            crop.save(probe)
            try:
                tokens = note_tokens(predict_crop(model, probe), include_rests=True)
            except Exception:  # noqa: BLE001
                tokens = []
            probe.unlink(missing_ok=True)
            cache[stem] = tokens
            readings.append(tokens)

        assignments = assign_measures_to_systems(readings, gt_tokens, owner, n_measures)
        placed = [a for a in assignments if a.end_measure > a.start_measure]
        mean_score = sum(a.score for a in placed) / len(placed) if placed else 0.0
        missing = unclaimed_measures(assignments, n_measures)
        accepted = mean_score >= args.min_mean_score and not missing
        records.append(
            {
                "score_id": score_id,
                "systems": len(detected),
                "measures": n_measures,
                "placed": len(placed),
                "mean_span_score": round(mean_score, 3),
                "unclaimed_measures": missing,
                "accepted": accepted,
                "assignments": [
                    {"system": a.system, "start_measure": a.start_measure,
                     "end_measure": a.end_measure, "score": round(a.score, 3)}
                    for a in assignments
                ],
            }
        )
        if not accepted:
            stats["rejected"] += 1
            stats["unclaimed_measures"] += len(missing)
            print(f"{score_id}: REJECTED mean={mean_score:.2f} unclaimed={len(missing)}")
            continue
        stats["accepted"] += 1

        written = 0
        for a in placed:
            groups = boxes_per_system[a.system]
            if groups is None:
                continue
            system = detected[a.system]
            page_name = system["page_image"]
            if page_name not in page_cache:
                page_path = pngs_dir / page_name
                if not page_path.exists():
                    continue
                page_cache[page_name] = Image.open(page_path).convert("L")
            page = page_cache[page_name]
            for voice_index, box in enumerate(groups):
                if voice_index >= len(voices):
                    continue
                measures = slice_voice_measures(
                    voices[voice_index], a.start_measure, a.end_measure
                )
                if not measures or calc_ratio_of_tuplets(measures) > 0.2:
                    continue
                if not contains_only_supported_clefs(measures):
                    continue
                cleaned = strip_naturals(measures)
                stem = f"{score_id}-sys{a.system}-v{voice_index}"
                image_path = args.out / f"{stem}.png"
                tokens_path = args.out / f"{stem}.tokens"
                page.crop(
                    (box["left"], box["top"], box["left"] + box["width"],
                     box["top"] + box["height"])
                ).save(image_path)
                tokens_path.write_text(token_lines_to_str(cleaned), encoding="utf-8")
                write_sidecar(tokens_path, cleaned)
                manifest_lines.append(f"{image_path},{tokens_path}")
                written += 1
        stats["pairs"] += written
        print(f"{score_id}: {written} pair(s), mean span score {mean_score:.2f}")

    if args.prediction_cache:
        args.prediction_cache.write_text(json.dumps(cache), encoding="utf-8")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(
            {
                "provenance": "reverse-fingerprint",
                "model_predictions_used": True,
                "warning": (
                    "Model-derived labels. Trainable as pseudo-labels; NEVER admissible "
                    "in a held-out evaluation set."
                ),
                "min_mean_score": args.min_mean_score,
                "stats": stats,
                "scores": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
