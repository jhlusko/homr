"""
Cross-staff coherence loss, ground-truth side (`DECODER_RHYTHM_ACCURACY_DESIGN.md`
§7.3's loss brainstorm, item 2): a real, cheaper alternative to §4 Stage C's learned
adapter worth trying first. Ground truth for sibling staves is available at *training*
time even though each staff decodes independently at inference - this module resolves,
for one training sample, the per-measure cumulative duration its whole *system* (every
part, not just this one) agrees on, so a loss can penalize this sample's own decode for
diverging from what its siblings actually contain, not just from its own label.

Reuses the same fragment-splitting infrastructure §7.3's time-signature sourcing
already built (`ossq_ground_truth.py`'s `fragment_path`/`parse_ground_truth`) rather
than a second whole-score lookup - a system's pre-split fragment already carries every
part, already movement-disambiguated, at the exact measure range this sample's system
spans.

Takes the *median* across parts at each measure index, not any single part's value -
`ossq_measure_length_audit.py`'s own corpus audit found 999 real measures where ground-
truth parts genuinely disagree on length (a labeling defect, not a legitimate
irregularity), so using one arbitrary part's value as the shared target would
propagate that noise into every sibling's training signal. This is the same
robustness idiom `homr.cross_staff_consistency.check_measure_durations` and
`propose_majority_position_corrections` already use for the equivalent cross-staff
problem at inference/audit time.
"""
import statistics
from fractions import Fraction
from pathlib import Path

from training.omr_datasets.ossq_ground_truth import fragment_path, parse_ground_truth
from training.omr_datasets.ossq_measure_length_audit import measure_length_by_part
from training.omr_datasets.score_profile_pairing import _find_score_musicxml
from training.omr_datasets.score_profile_time_signature import parse_ossq_stem_full

#: A generous upper bound on measures within one physical system - real systems in
#: this corpus are a handful of measures at most (this is one *line* of a score, not
#: a whole piece); the loss gracefully ignores anything beyond this rather than
#: crashing, so a still-more-generous corpus is degraded, not broken, by a rare outlier.
MAX_COHERENCE_MEASURES = 32


def system_measure_curve(dataset_root: str, stem: str) -> list[float] | None:
    """Whole-note-relative cumulative duration at each measure boundary of this
    sample's system, taking the median across every part in the system's fragment.

    `None` whenever any step of the resolution chain can't be completed (non-OSSQ
    stem, unresolvable score, no pre-split fragment for this page/system, a fragment
    with no measures at all) - never a guess, the same discipline
    `time_signature_for_sample` uses. A caller with `None` should train this sample
    without the coherence signal, not with a fabricated one.
    """
    parsed = parse_ossq_stem_full(stem)
    if parsed is None:
        return None
    score_id, page_str, system_index, _part_index = parsed

    score_path = _find_score_musicxml(dataset_root, score_id)
    if score_path is None:
        return None
    piece_dir_path = score_path.parent

    frag = fragment_path(piece_dir_path, int(page_str), system_index + 1)
    if not frag.exists():
        return None

    tree = parse_ground_truth(str(frag))
    parts = tree.getroot().findall(".//part")
    if not parts:
        return None

    return _median_cumulative_curve(parts)


def _median_cumulative_curve(parts: list) -> list[float] | None:
    """Whole-note cumulative duration at each measure index, 0-based, taking the
    median across every part that has a measure at that index - a system with parts
    of differing measure counts (an edge case, not the common one) still contributes
    a target for every index at least one part reaches; an index no part reaches at
    all cannot happen since the loop is bounded by the longest part.
    """
    divisions = [{"current": 1} for _ in parts]
    measures_by_part = [part.findall("measure") for part in parts]
    max_len = max((len(measures) for measures in measures_by_part), default=0)
    if max_len == 0:
        return None

    curve: list[float] = []
    cumulative = Fraction(0)
    for i in range(max_len):
        lengths = [
            measure_length_by_part(measures[i], divisions[part_index])
            for part_index, measures in enumerate(measures_by_part)
            if i < len(measures)
        ]
        if not lengths:
            continue
        median_length = Fraction(statistics.median(lengths))
        cumulative += median_length / 4  # quarter notes -> whole notes
        curve.append(float(cumulative))
    return curve if curve else None
