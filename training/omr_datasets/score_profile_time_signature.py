"""
Real per-sample expected time signature for OSSQ training examples
(`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.3's refinement: condition
`ProfileContextEmbedding` on the expected time signature explicitly, sourced from
ground truth at training time - not the naive "hope the model reads its own decoded
token history" version already ruled out by this session's forced-prefix experiments).

Reuses `ossq_ground_truth.py`'s movement-aware measure resolution - built earlier this
session to fix a real ground-truth splicing bug (multi-movement pieces restart
`<measure number="...">` at 1 per movement) - rather than a second, less-tested lookup:
the exact same "which movement, which flat measure index" problem this module needs to
solve for training data was already solved, and validated, for review/audit tooling.
"""
import re
from pathlib import Path

from training.omr_datasets.ossq_ground_truth import (
    fragment_path,
    measure_start_for_system,
    movement_index_for_system,
    parse_ground_truth,
    resolve_flat_measure_range,
)
from training.omr_datasets.score_profile_pairing import _find_score_musicxml

_FULL_STEM_PATTERN = re.compile(r"^(?P<score_id>.+)_(?P<page>\d+)_(?P<system>\d+)_(?P<part>\d+)$")


def parse_ossq_stem_full(stem: str) -> tuple[str, str, int, int] | None:
    """`(score_id, page_str, 0-based system_index, 0-based part_index)`, or `None` if
    `stem` doesn't match OSSQ's naming convention at all (`convert_ossq.py`'s
    `f"{score_id}_{page}_{system}_{part_index + 1}"`, page/system 1-based zero-padded
    strings, part 1-based). Deliberately a separate function from
    `score_profile_pairing.parse_ossq_stem` rather than changing that one's return
    shape - this needs page/system too, that one's existing callers don't."""
    match = _FULL_STEM_PATTERN.match(stem)
    if match is None:
        return None
    return (
        match.group("score_id"),
        match.group("page"),
        int(match.group("system")) - 1,
        int(match.group("part")) - 1,
    )


def _latest_time_signature(measures: list) -> str:
    """The last `<time>` declared across `measures`, in document order - time
    signature, like `<divisions>`, is typically only re-stated in MusicXML on *change*
    and inherited otherwise, the same "seed by walking forward" pattern this session's
    other MusicXML-reading fixes (`content_verify_agrees.py`'s divisions bug, the
    spot-check script's own version of the same bug) already established as necessary
    here. "" if none of `measures` declares one."""
    current = ""
    for measure in measures:
        time_el = measure.find("attributes/time")
        if time_el is None:
            continue
        beats = time_el.find("beats")
        beat_type = time_el.find("beat-type")
        if beats is not None and beat_type is not None and beats.text and beat_type.text:
            current = f"{beats.text.strip()}/{beat_type.text.strip()}"
    return current


def _time_signature_from_fragment(frag_path: Path, part_index: int) -> str:
    """Fast path: `split_ground_truth_by_system.py`'s pre-extracted fragment already
    covers exactly this system's own measure_start..measure_end range with attributes
    carried forward, so the time signature at the *first* measure of the target part
    is exactly what's in effect at the start of this system - no movement resolution,
    no matching-by-number, no walking a multi-MB whole-score file needed. "" if the
    fragment doesn't have this many parts (falls through to the slow path)."""
    tree = parse_ground_truth(str(frag_path))
    parts = tree.getroot().findall(".//part")
    if part_index >= len(parts):
        return ""
    measures = parts[part_index].findall("measure")
    if not measures:
        return ""
    return _latest_time_signature(measures[:1])


def time_signature_for_sample(dataset_root: str, stem: str) -> str:
    """The real ground-truth time signature in effect at this training sample's actual
    measure range - "" (unknown) whenever any step of the resolution chain can't be
    completed (non-OSSQ stem, unresolvable score, no corpus alignment metadata for this
    page/system, an ambiguous multi-movement mapping), never a guess.

    Tries the fast, pre-split fragment first (`split_ground_truth_by_system.py`) -
    falls through to the original whole-score resolution only when no fragment exists
    yet (the corpus hasn't been split, or this specific page/system was skipped during
    splitting) - built after `phase22`'s first training run stalled at 0% GPU
    utilization: a per-sample lookup against a multi-MB whole-score file does not scale
    to corpus-wide training, even cached (see `ossq_ground_truth.py`'s own
    `parse_ground_truth`/`_systemwise_entries` docstrings for the measured numbers).

    A placeholder image path is built purely to reuse `ossq_ground_truth.py`'s
    existing path-arithmetic functions (`piece_dir`/`score_and_page` only ever
    manipulate the path string - they never read the image file itself, only metadata
    files relative to it) - the file need not exist on disk for this lookup.
    """
    parsed = parse_ossq_stem_full(stem)
    if parsed is None:
        return ""
    score_id, page_str, system_index, part_index = parsed

    score_path = _find_score_musicxml(dataset_root, score_id)
    if score_path is None:
        return ""
    piece_dir = score_path.parent

    frag = fragment_path(piece_dir, int(page_str), system_index + 1)
    if frag.exists():
        from_fragment = _time_signature_from_fragment(frag, part_index)
        if from_fragment:
            return from_fragment

    placeholder_image = (
        piece_dir / "images" / "synthetic" / "original" / f"{score_id}:{page_str}.png"
    )

    local_start = measure_start_for_system(placeholder_image, system_index)
    movement_index = movement_index_for_system(placeholder_image, system_index)
    if local_start is None or movement_index is None:
        return ""

    flat_range = resolve_flat_measure_range(
        score_path, movement_index, part_index, local_start, local_start
    )
    if flat_range is None:
        return ""

    tree = parse_ground_truth(str(score_path))
    parts = tree.getroot().findall(".//part")
    if part_index >= len(parts):
        return ""
    measures = parts[part_index].findall("measure")
    return _latest_time_signature(measures[: flat_range[0] + 1])
