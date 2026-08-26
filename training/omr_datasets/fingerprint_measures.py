"""Recovering each crop's *true* measure range by fingerprinting what homr itself
reads off the crop and aligning that against the piece's own real MusicXML.

Why this exists: `extract_stage2_pairs.py`'s eligibility rule ("this system's
detected bar-line count equals ground truth's own measure count for that system")
is a *local, per-system* check, but the measure cutter it feeds is a *stateful,
sequential* device - one system whose true measure count disagrees anywhere in a
piece silently shifts every later system's content, even where each later system's
own local bar-count check still passes. Real, confirmed example (IMSLP148200):
position 0 disagrees (4 detected vs 2 ground truth), and position 3's crop then
visibly contains position 4's music ("Have you felt the...") while its token file
carries position 3's own assigned range ("smutched it?"). Bar counts alone cannot
see this; content can.

The approach: run homr's own Stage 2 transformer (`Staff2Score`, the same one
production uses) on each crop, flatten its prediction to a note-token sequence,
and align that sequence against the same piece's real MusicXML note sequence.
Whatever ground-truth measure range the aligned region lands in *is* that crop's
true measure range - regardless of what bar counting believed.

Alignment is deliberately done on the **flat note sequence, not per measure**:
homr's own bar-line reading can be wrong too (that is half of what produced the
drift in the first place), so anything that assumes the crop's measure boundaries
agree with ground truth's would inherit the very error being corrected. A flat
sequence alignment (`difflib.SequenceMatcher`'s matching blocks) tolerates missing
or spurious barlines, OMR pitch errors, and whole-system shifts with one mechanism.
"""

# flake8: noqa: T201

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from homr.transformer.vocabulary import EncodedSymbol

#: A matched region must cover at least this fraction of the crop's own notes
#: before its recovered measure range is trusted - below this the alignment is
#: reported but flagged, rather than silently treated as a correction.
MIN_COVERAGE = 0.5

#: ...and match at least this well within the region it did cover. Historical
#: engraving plus real OMR error means an exact match is never the bar; this is
#: calibrated to accept "clearly the same passage, read imperfectly" and reject
#: "a different passage that happens to share a few pitches".
MIN_SIMILARITY = 0.55


def note_tokens(symbols: list[EncodedSymbol]) -> list[str]:
    """The flat, alignable note sequence for one staff's worth of symbols -
    pitch-with-accidental per note, in order, rests and barlines dropped.

    Pitch (not rhythm) carries the alignment: melodic contour is far more
    distinctive between passages than duration patterns are, and homr's pitch
    reading is the more reliable of the two on real scans. Rests are dropped
    because a rest-only stretch fingerprints identically everywhere in a piece
    and would only add false matches.
    """
    tokens = []
    for symbol in symbols:
        if not symbol.rhythm.startswith("note"):
            continue
        if symbol.pitch in ("", ".", "_"):
            continue
        lift = symbol.lift if symbol.lift in ("#", "b") else ""
        tokens.append(f"{symbol.pitch}{lift}")
    return tokens


def measure_note_tokens(measures: list[list[EncodedSymbol]]) -> tuple[list[str], list[int]]:
    """`(flat_tokens, measure_index_per_token)` for a piece's own ground-truth
    measures - the flat sequence to align against, plus the back-pointer that
    turns a matched position range back into a measure range."""
    flat: list[str] = []
    owner: list[int] = []
    for measure_index, measure in enumerate(measures):
        for token in note_tokens(measure):
            flat.append(token)
            owner.append(measure_index)
    return flat, owner


def align_to_ground_truth(
    crop_tokens: list[str], gt_tokens: list[str], gt_owner: list[int]
) -> dict | None:
    """Where in the ground-truth note stream this crop's own notes actually sit.

    Returns `{"start_measure", "end_measure", "coverage", "similarity",
    "trusted"}` - `end_measure` exclusive, matching every other measure range in
    this corpus - or `None` when the crop has no usable notes at all.

    `coverage` is the fraction of the crop's own notes that landed inside the
    matched region, `similarity` the match quality within it; `trusted` is the
    conjunction of both clearing their thresholds. A caller deciding whether to
    *correct* a range should require `trusted`; a caller merely reporting
    disagreement can use the raw numbers.
    """
    if not crop_tokens or not gt_tokens:
        return None

    matcher = difflib.SequenceMatcher(None, gt_tokens, crop_tokens, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    # The matched region spans from the first to the last aligned block - the
    # gaps between them are the crop's own OMR errors, which belong inside the
    # region rather than truncating it.
    gt_start = blocks[0].a
    gt_end = blocks[-1].a + blocks[-1].size
    matched_notes = sum(b.size for b in blocks)

    coverage = matched_notes / len(crop_tokens)
    span = max(gt_end - gt_start, 1)
    similarity = matched_notes / span

    start_measure = gt_owner[gt_start]
    end_measure = gt_owner[min(gt_end, len(gt_owner)) - 1] + 1

    return {
        "start_measure": start_measure,
        "end_measure": end_measure,
        "coverage": coverage,
        "similarity": similarity,
        "trusted": coverage >= MIN_COVERAGE and similarity >= MIN_SIMILARITY,
    }


_STEM_RE = re.compile(r"^(?P<score_id>.+)-sys(?P<system>\d+)-v(?P<voice>\d+)$")


def parse_stem(stem: str) -> tuple[str, int, int] | None:
    match = _STEM_RE.match(stem)
    if not match:
        return None
    return match["score_id"], int(match["system"]), int(match["voice"])


def predict_crop(model: object, image_path: Path) -> list[EncodedSymbol]:
    """homr's own Stage 2 reading of one crop - the same `Staff2Score.predict`
    production uses, on the same `add_image_into_tr_omr_canvas`-prepared input
    `data_loader.py` feeds it during training (minus the training-only
    distortion, which exists to augment, not to preprocess)."""
    from homr.staff_parsing import add_image_into_tr_omr_canvas

    image = np.array(Image.open(image_path).convert("L"))
    return model.predict(add_image_into_tr_omr_canvas(image))  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--crops", type=Path, required=True,
        help="Directory of crop pngs to fingerprint (extract_stage2_pairs.py's --out).",
    )
    parser.add_argument(
        "--ground-truth-tokens", type=Path, required=True,
        help="Per-score JSON of ground-truth per-measure note tokens "
        "(build_ground_truth_tokens.py's --out).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where per-crop results go.")
    parser.add_argument("--limit", type=int, help="Only do the first N crops (a quick run).")
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from homr.transformer.staff2score import Staff2Score

    model = Staff2Score(Config())

    crops = sorted(args.crops.glob("*.png"))
    if args.limit:
        crops = crops[: args.limit]
    print(f"{len(crops)} crops to fingerprint")

    gt_cache: dict[str, dict] = {}
    results = []
    for index, crop_path in enumerate(crops):
        parsed = parse_stem(crop_path.stem)
        if parsed is None:
            continue
        score_id, system, voice = parsed
        if score_id not in gt_cache:
            gt_path = args.ground_truth_tokens / f"{score_id}.json"
            gt_cache[score_id] = (
                json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
            )
        gt_doc = gt_cache[score_id]
        voices = gt_doc.get("voices") or []
        if voice >= len(voices):
            continue

        gt_measures_tokens = voices[voice]
        gt_tokens: list[str] = []
        gt_owner: list[int] = []
        for measure_index, measure_tokens in enumerate(gt_measures_tokens):
            for token in measure_tokens:
                gt_tokens.append(token)
                gt_owner.append(measure_index)

        try:
            predicted = predict_crop(model, crop_path)
        except Exception as e:  # noqa: BLE001
            print(f"{crop_path.stem}: PREDICT FAILED ({e})")
            continue

        crop_tokens = note_tokens(predicted)
        alignment = align_to_ground_truth(crop_tokens, gt_tokens, gt_owner)
        results.append(
            {
                "stem": crop_path.stem,
                "score_id": score_id,
                "system": system,
                "voice": voice,
                "predicted_notes": len(crop_tokens),
                "alignment": alignment,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(crops)} done")
            args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    trusted = sum(1 for r in results if r["alignment"] and r["alignment"]["trusted"])
    print(f"{len(results)} fingerprinted, {trusted} with a trusted alignment")


if __name__ == "__main__":
    main()
