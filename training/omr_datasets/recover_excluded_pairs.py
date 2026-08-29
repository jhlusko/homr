"""Recovering the systems `extract_stage2_pairs.py` throws away.

Eligibility there is "this system's detected bar-line count equals ground truth's
own measure count for it" (or a human `/compare` judgment covers its page). That
rejects ~47% of detected systems - 2,169 of 4,612 - and the rejection is on a
*proxy*: a bar-line miscount says the counter and the layout disagree, not that the
crop is unusable. If the crop's actual content can be located in the piece, the
pair is fine regardless of what counting believed.

So: run homr's own Stage 2 transformer over each excluded crop, align what it reads
against the piece's real MusicXML note stream, and take the measure range the
content actually lands in. Same machinery as `fingerprint_measures.py`, used here
to *recover* a range rather than to audit one.

**The one thing that must be different from a plain fingerprint.** Global alignment
finds *a* true match, not necessarily *the* one: strophic Lieder repeat whole
passages, and a later verse aligns just as well to the first occurrence - visible
in the fingerprint audit as large negative offsets (-15, -20, -45). Recovering a
range from that would confidently produce a wrong pair, which is worse than the
exclusion it replaces. Every alignment here is therefore **windowed around the
position the system is expected at**, so a repeat elsewhere in the piece is not a
candidate at all. The expected position comes from ground truth's own cumulative
system ranges: unreliable enough that we are re-deriving the range, but reliable
enough to say roughly where in the piece we are.
"""

# flake8: noqa: T201

import argparse
import copy
import json
import re
from pathlib import Path

import yaml
from PIL import Image

from homr.circle_of_fifths import strip_naturals
from training.omr_datasets.convert_lieder import (
    MeasureCutter,
    contains_only_supported_clefs,
    is_grandstaff,
)
from training.omr_datasets.extract_stage2_pairs import (
    eligible_system_positions,
    flat_detected_systems,
    flat_measure_ranges,
    group_staff_boxes_into_voices,
)
from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl,
    load_lieder_file_tree,
    load_lieder_mxl_tree,
    load_lieder_scores,
    match_single_piece_scores,
)
from training.omr_datasets.fingerprint_measures import align_to_ground_truth, note_tokens
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.notation_sidecar import write_sidecar
from training.transformer.training_vocabulary import max_tuplet_ratio, calc_ratio_of_tuplets, token_lines_to_str

#: How far either side of the expected position an alignment may land, in measures.
#: Wide enough to absorb real drift from a miscounted neighbour, narrow enough that
#: a repeat of the same passage elsewhere in the piece is never in the window.
WINDOW_MEASURES = 8

# PIL refuses images past ~179M pixels as a "decompression bomb". These are our own
# IMSLP page scans, some of which are genuinely that large at full resolution - the
# guard is protecting against hostile uploads, which this is not. Left at the
# default it does not merely skip the page: it raises, and an earlier run of this
# tool was killed outright at 105 of 266 scores by exactly that, mid-corpus, with
# no summary written. Raising the ceiling is the fix for the legitimate case; the
# try/except around page loading below is the fix for everything else.
Image.MAX_IMAGE_PIXELS = None


def window_bounds(gt_owner: list[int], lo_measure: int, hi_measure: int) -> tuple[int, int]:
    """The `[a, b)` slice of a flat note stream covering measures
    `[lo_measure, hi_measure]` inclusive - the search window for a windowed
    alignment. Returns an empty slice (`a == b`) when no note falls in range,
    which the caller treats as "nothing to align against here"."""
    a = None
    b = 0
    for index, measure_index in enumerate(gt_owner):
        if lo_measure <= measure_index <= hi_measure:
            if a is None:
                a = index
            b = index + 1
    return (a, b) if a is not None else (0, 0)


def align_near_expected(
    crop_tokens: list[str],
    gt_tokens: list[str],
    gt_owner: list[int],
    expected_start: int,
    window_measures: int = WINDOW_MEASURES,
) -> dict | None:
    """`align_to_ground_truth` restricted to a window around `expected_start`, so a
    repeated passage elsewhere in the piece cannot win the match. Measure indices
    come back absolute, because the window slices `gt_owner` (which holds absolute
    indices) rather than renumbering it.

    The window is **widened in stages** rather than opened to its full size at
    once, and the first trusted alignment wins. That ordering is the tie-break:
    within a single wide window, two equally perfect matches of a repeated passage
    are separated only by `difflib`'s own preference for the earliest matching
    block, which silently biases recovery *backwards* in the piece. Found in real
    data - a Die Forelle system expected at measure 45 recovered to 41 because the
    strophic repeat at 41 matched exactly as well and came first. Trying narrow
    windows first means the nearest candidate is the only candidate, and a distant
    repeat is reachable only when nothing closer explains the crop at all.
    """
    for width in _window_ladder(window_measures):
        lo = max(0, expected_start - width)
        # The upper bound has to clear the *end* of the crop, not just its start,
        # or a multi-measure system near the top of the window is truncated and
        # matches worse than it should.
        hi = expected_start + width + _estimated_measures(crop_tokens)
        a, b = window_bounds(gt_owner, lo, hi)
        if a == b:
            continue
        result = align_to_ground_truth(crop_tokens, gt_tokens[a:b], gt_owner[a:b])
        if result is not None and result["trusted"]:
            return result
    return None


def _window_ladder(max_width: int) -> list[int]:
    """Widths to try, narrowest first, ending at `max_width`."""
    ladder = [w for w in (1, 2, 4, 8, 16) if w < max_width]
    return [*ladder, max_width]


def _estimated_measures(crop_tokens: list[str]) -> int:
    """A rough upper bound on how many measures a crop spans, from its note count.
    Only used to keep the window's far edge past the end of the crop; being
    generous here costs nothing but a slightly wider search."""
    return max(1, len(crop_tokens) // 2)


def slice_voice_measures(
    voice: list, start: int, end: int, always_include_time: bool = False
) -> list:
    """The token sequence for measures `[start, end)` of one voice, with the
    clef/key/time-signature continuity `MeasureCutter` exists to maintain.

    A fresh cutter is walked from the beginning and the measures before `start` are
    extracted and discarded rather than skipped: the cutter's whole job is to carry
    running clef/key/time state forward, so jumping straight to `start` would
    produce a slice with the wrong (or missing) preamble. `voice` is deep-copied
    because the cutter pops from it.

    `always_include_time` is forwarded to the FINAL `extract_measures` call only - the
    discarded preamble never needs one, and forwarding it there too would do nothing but
    cost cycles. Default False preserves every existing caller's behaviour exactly; pass
    True to match what `music_xml_generator.generate_xml` itself does on a fresh render
    (it always draws a courtesy time signature - `build_add_time_direction`), so a slice
    meant to be compared against a render is comparing like with like rather than being
    penalised for a source that simply did not restate the metre here.
    """
    if start < 0 or end > len(voice) or end <= start:
        return []
    cutter = MeasureCutter(copy.deepcopy(voice))
    if start:
        cutter.extract_measures(start)
    return cutter.extract_measures(end - start, always_include_time=always_include_time)


_LOGGED_SCORE_RE = re.compile(r"^(IMSLP[0-9]+): (?:recovered \d+ pair|FAILED)", re.MULTILINE)


def already_logged(log_text: str) -> set[str]:
    """Score ids a previous run reported on - either a recovery count or a
    failure. Used to resume after a kill without redoing work or appending the
    same manifest lines twice."""
    return set(_LOGGED_SCORE_RE.findall(log_text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--mxl-tree-cache", type=Path)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--pngs", type=Path, required=True, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, help="Per-system recovery outcomes as JSON, for auditing."
    )
    parser.add_argument("--score-ids", type=Path)
    parser.add_argument(
        "--skip-logged", type=Path,
        help="A previous run's log; scores it already reported are skipped. Lets a "
        "killed run resume without redoing work or duplicating manifest lines.",
    )
    parser.add_argument("--window", type=int, default=WINDOW_MEASURES)
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from homr.transformer.staff2score import Staff2Score

    from homr.staff_parsing import add_image_into_tr_omr_canvas

    model = Staff2Score(Config())

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    eligible_by_score = eligible_system_positions(rows, review)

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)

    score_ids = sorted(eligible_by_score)
    if args.score_ids:
        wanted = {s.strip() for s in args.score_ids.read_text().splitlines() if s.strip()}
        score_ids = [s for s in score_ids if s in wanted]
    if args.skip_logged and args.skip_logged.exists():
        done = already_logged(args.skip_logged.read_text(encoding="utf-8", errors="replace"))
        before = len(score_ids)
        score_ids = [s for s in score_ids if s not in done]
        print(f"resuming: skipping {before - len(score_ids)} score(s) already logged")

    matched = match_single_piece_scores(lieder, score_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []
    recovered = untrusted = skipped = 0

    with open(args.manifest, "a", encoding="utf-8") as manifest:
        for score_id, (key, entry) in matched.items():
            gt_path = args.ground_truth / f"{score_id}.json"
            systems_path = args.systems / f"{score_id}.yaml"
            pngs_dir = next((d for d in args.pngs if (d / score_id).is_dir()), None)
            if not gt_path.exists() or not systems_path.exists() or pngs_dir is None:
                continue
            try:
                ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
                systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
                musicxml = unzip_mxl(fetch_mxl(entry, key, mxl_tree or file_tree))
                voices = music_xml_string_to_tokens(musicxml.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                print(f"{score_id}: FAILED preparing ({e})")
                continue

            grandstaff_flags = [is_grandstaff(v) for v in voices]
            gt_streams = []
            for voice in voices:
                tokens: list[str] = []
                owner: list[int] = []
                for measure_index, measure in enumerate(voice):
                    for token in note_tokens(measure):
                        tokens.append(token)
                        owner.append(measure_index)
                gt_streams.append((tokens, owner))

            ranges = flat_measure_ranges(ground_truth["pages"])
            detected = flat_detected_systems(systems_doc)
            eligible = eligible_by_score.get(score_id, set())
            page_cache: dict[str, Image.Image] = {}
            score_recovered = 0

            for position, system in enumerate(detected):
                if position in eligible or position >= len(ranges):
                    continue
                groups = group_staff_boxes_into_voices(
                    system.get("staffBoxes", []), grandstaff_flags
                )
                if groups is None:
                    skipped += 1
                    continue
                image_name = system["page_image"]
                if image_name not in page_cache:
                    page_path = pngs_dir / image_name
                    if not page_path.exists():
                        skipped += 1
                        continue
                    try:
                        page_cache[image_name] = Image.open(page_path).convert("L")
                    except Exception as e:  # noqa: BLE001
                        # One unreadable or absurdly large page must cost that page,
                        # not the rest of the corpus.
                        print(f"{score_id}: page {image_name} unreadable ({e})", flush=True)
                        skipped += 1
                        continue
                page_image = page_cache[image_name]
                expected_start = ranges[position][0]

                for voice_index, box in enumerate(groups):
                    if voice_index >= len(voices):
                        continue
                    crop = page_image.crop(
                        (
                            box["left"],
                            box["top"],
                            box["left"] + box["width"],
                            box["top"] + box["height"],
                        )
                    )
                    try:
                        import numpy as np

                        predicted = model.predict(
                            add_image_into_tr_omr_canvas(np.array(crop))
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"{score_id}-sys{position}-v{voice_index}: PREDICT FAILED ({e})")
                        continue

                    gt_tokens, gt_owner = gt_streams[voice_index]
                    alignment = align_near_expected(
                        note_tokens(predicted), gt_tokens, gt_owner, expected_start, args.window
                    )
                    record = {
                        "score_id": score_id,
                        "system": position,
                        "voice": voice_index,
                        "expected_start": expected_start,
                        "alignment": alignment,
                    }
                    report.append(record)
                    if alignment is None or not alignment["trusted"]:
                        untrusted += 1
                        continue

                    measures = slice_voice_measures(
                        voices[voice_index],
                        alignment["start_measure"],
                        alignment["end_measure"],
                    )
                    if not measures:
                        untrusted += 1
                        continue
                    if calc_ratio_of_tuplets(measures) > max_tuplet_ratio():
                        continue
                    if not contains_only_supported_clefs(measures):
                        continue

                    stem = f"{score_id}-sys{position}-v{voice_index}"
                    image_path = args.out / f"{stem}.png"
                    tokens_path = args.out / f"{stem}.tokens"
                    crop.save(image_path)
                    cleaned = strip_naturals(measures)
                    tokens_path.write_text(token_lines_to_str(cleaned), encoding="utf-8")
                    write_sidecar(str(tokens_path), cleaned)
                    manifest.write(f"{image_path},{tokens_path}\n")
                    recovered += 1
                    score_recovered += 1

            manifest.flush()
            # Always mark the score, even at zero, so a resume can tell "processed,
            # found nothing" from "never reached". A first run of this was killed
            # partway through and the log alone could not distinguish the two.
            print(f"{score_id}: recovered {score_recovered} pair(s)", flush=True)
            if args.report:
                # Written every score rather than once at the end: the same killed
                # run lost its entire audit trail because the report was a single
                # write after the loop.
                args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{recovered} recovered, {untrusted} not trusted, {skipped} skipped")


if __name__ == "__main__":
    main()
