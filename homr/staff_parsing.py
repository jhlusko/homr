import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from homr import constants

if TYPE_CHECKING:
    from training.architecture.transformer.staff_context import StaffContextTransformer
from homr.cross_staff_consistency import (
    analyze_system,
    check_barline_positions,
    check_measure_durations,
    check_page_staff_counts,
    check_part_order,
    staves_by_system,
)
from homr.cross_staff_repair import (
    propose_carry_forward_key_signature,
    propose_majority_position_corrections,
    propose_motif_articulation_corrections,
    propose_repairs,
)
from homr.cross_staff_rerank import fork_candidates_from_margins, rerank_staff_candidates
from homr.debug import Debug
from homr.image_utils import crop_image_and_return_new_top
from homr.model import MultiStaff, Staff
from homr.score_profile import ScoreProfile
from homr.score_profile_layout import propose_part_assignment, staff_to_part_by_system
from homr.simple_logging import eprint
from homr.staff_dewarping import StaffDewarping, dewarp_staff_image
from homr.staff_parsing_tromr import parse_staff_tromr, parse_staff_tromr_greedy_with_margins
from homr.staff_regions import StaffRegions
from homr.system_grouping import (
    SystemPartition,
    assign_voice_slots,
    find_system_grouping,
    report_grouping,
)
from homr.transformer.configs import Config, default_config
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
from homr.type_definitions import NDArray


def _flatten_staffs(staffs: list[MultiStaff]) -> list[Staff]:
    return [s for multi_staff in staffs for s in multi_staff.staffs]


def _regroup_by_period(
    flat_staffs: list[Staff], period: int, front_trim: int, back_trim: int
) -> list[MultiStaff]:
    core = flat_staffs[front_trim : len(flat_staffs) - back_trim]
    return [MultiStaff(core[i : i + period], []) for i in range(0, len(core), period)]


def _find_periodic_core(flat_staffs: list[Staff]) -> tuple[int, int, int] | None:
    """
    Find a repeating sequence of staff layouts among individual staffs, e.g.
    a solo staff followed by a piano grand staff (2 staffs), repeated for
    every system in a vocal score with piano accompaniment.

    We work on the flattened sequence of raw staffs rather than on the
    MultiStaff rows produced upstream, because that upstream grouping is
    itself only a heuristic (staffs sharing a bar line or clef get merged
    into one row) and can be inconsistent across a page: the same kind of
    solo-staff-plus-grand-staff pair might end up pre-merged into one row for
    one system and left as two separate rows for another, purely because of
    how cleanly a bar line lined up. Searching row-by-row would then see two
    different "shapes" for what is structurally the same repeating pattern.
    Working on individual staffs sidesteps that inconsistency entirely.

    A system right at the start or end of the page can break the pattern on
    its own without invalidating it: an introduction or coda system with a
    genuinely different layout, or simply the most poorly detected staff on
    the page. We therefore allow trimming up to one period's worth of staffs
    from either edge before requiring the remainder to tile exactly. We
    never trim from the middle of the page: a mismatch there is a detection
    problem to fix upstream, not something to paper over here.

    Returns (period, front_trim, back_trim) for the smallest total trim and,
    among ties, the smallest period -- so an already-uniform page (period 1,
    no trim) is always preferred when it fits, and we never discard more of
    the page than necessary. Returns None if no repeating core of at least
    two full cycles can be found.
    """
    layout = [s.is_grandstaff for s in flat_staffs]
    n = len(layout)
    best: tuple[int, int, int, int] | None = None
    for period in range(1, n // 2 + 1):
        for front_trim in range(period + 1):
            for back_trim in range(period + 1):
                core = layout[front_trim : n - back_trim]
                if len(core) < 2 * period or len(core) % period != 0:
                    continue
                rows = [tuple(core[i : i + period]) for i in range(0, len(core), period)]
                if not all(row == rows[0] for row in rows):
                    continue
                candidate = (front_trim + back_trim, period, front_trim, back_trim)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
    if best is None:
        return None
    _, period, front_trim, back_trim = best
    return period, front_trim, back_trim


def _adjacent_connected_pairs(
    flat_staffs: list[Staff], staffs: list[MultiStaff]
) -> set[tuple[int, int]]:
    """Adjacent flat-staff index pairs that the bracket/barline detector put in one row."""
    position = {id(staff): index for index, staff in enumerate(flat_staffs)}
    pairs: set[tuple[int, int]] = set()
    for multi_staff in staffs:
        indices = sorted(position[id(staff)] for staff in multi_staff.staffs)
        for first, second in zip(indices, indices[1:], strict=False):
            if second == first + 1:
                pairs.add((first, second))
    return pairs


@dataclass(frozen=True)
class SystemPlan:
    """Systems, plus which voice each of their staffs belongs to.

    Kept alongside the MultiStaff list rather than inside it because a system can be
    missing a voice: `slots[system][position]` is the voice index of that system's
    position-th staff, which is the identity mapping for a complete system and skips
    over the absent voice for a short one.
    """

    systems: list[MultiStaff]
    slots: list[tuple[int, ...]]

    @property
    def voices(self) -> int:
        return max((max(slots) + 1 for slots in self.slots if slots), default=0)

    def staff_for_voice(self, system: int, voice: int) -> Staff | None:
        slots = self.slots[system]
        if voice not in slots:
            return None
        return self.systems[system].staffs[slots.index(voice)]

    @staticmethod
    def dense(systems: list[MultiStaff]) -> "SystemPlan":
        """Every system complete, so the nth staff is the nth voice."""
        return SystemPlan(systems, [tuple(range(len(s.staffs))) for s in systems])


def _group_by_geometry(flat_staffs: list[Staff], staffs: list[MultiStaff]) -> SystemPlan | None:
    """Regroup the page from staff spacing, or None to leave the decision alone.

    A system short of a staff is not dropped when its spacing says which voice is
    missing. Detection missing one staff out of an otherwise complete system is common,
    and dropping the system costs every voice's music for it, where reading its staffs
    into the right voice slots costs only the absent voice's. A system whose slots cannot
    be pinned down is still dropped: guessing would read every one of its staffs into the
    wrong voice, which is worse than losing it.
    """
    result = find_system_grouping(flat_staffs, _adjacent_connected_pairs(flat_staffs, staffs))
    if result is None:
        return None
    report_grouping(result)
    if not result.confident:
        return None

    size = result.best.staves_per_system
    assignments = assign_voice_slots(flat_staffs, result.best)
    systems: list[MultiStaff] = []
    slots: list[tuple[int, ...]] = []
    recovered = dropped = 0
    for group, assigned in zip(result.best.groups, assignments, strict=True):
        if assigned is None:
            dropped += 1
            continue
        if len(group) < size:
            recovered += 1
        systems.append(MultiStaff([flat_staffs[index] for index in group], []))
        slots.append(assigned)
    if recovered:
        eprint(
            f"Recovered {recovered} incomplete system(s): placed their staffs into voice"
            f" slots from the spacing rather than dropping the system"
        )
    if dropped:
        eprint(
            f"Ignoring {dropped} incomplete system(s) of the {len(result.best.groups)} on this"
            f" page: fewer than the {size} staffs the rest of the page repeats, and the"
            " spacing does not say which voice is missing"
        )
    if not systems:
        return None
    return SystemPlan(systems, slots)


def _plan_systems(staffs: list[MultiStaff]) -> SystemPlan:
    """
    If every system already has the same number of *more than one* staff, trust that
    directly rather than re-deriving it via _find_periodic_core. That function's signature
    is each flat staff's is_grandstaff flag, which is a fine way to tell "solo staff" from
    "piano grand staff" apart when the two are pre-merged inconsistently across the page
    (see its own docstring) - but it carries zero information when a page has N genuinely
    independent, same-type staffs per system and none of them are a grand staff (e.g. a
    string quartet): the flattened signature is then a constant sequence, which trivially -
    and wrongly - satisfies period=1, collapsing all N voices into one. Checking uniformity
    upfront on the untouched, already-correct per-system grouping sidesteps that degenerate
    case entirely.

    Restricted to row length > 1: a page where every row is already a single raw staff
    (nothing grouped yet, e.g. a solo-plus-piano page where no bar line happened to
    pre-merge any pair) is *also* uniform by this same measure, but there _find_periodic_
    core is exactly what's needed to discover the real, larger repeating pattern from
    scratch - that's the case this function was originally written for, and it is never
    already uniform at a row length above 1.
    """
    row_lengths = {len(multi_staff.staffs) for multi_staff in staffs}
    if len(row_lengths) == 1 and next(iter(row_lengths)) > 1:
        return SystemPlan.dense(staffs)
    flat_staffs = _flatten_staffs(staffs)
    # Ask page geometry before the periodic signature. Once the rows disagree, that
    # signature is unreliable in both directions on the same score: on one page it reads
    # a constant is_grandstaff sequence and settles on period 1, collapsing every voice
    # into one part; on the next, a single staff wrongly flagged as a grand staff makes
    # period 2 tile after trimming three staffs away, which is just as wrong and loses
    # music besides. Geometry is measuring the thing that actually defines a system, so
    # it goes first - and because it declines when the gaps do not separate, the periodic
    # path below still handles every page it was written for.
    geometric = _group_by_geometry(flat_staffs, staffs)
    if geometric is not None:
        return geometric
    core = _find_periodic_core(flat_staffs)
    if core is not None:
        period, front_trim, back_trim = core
        if front_trim > 0:
            eprint(
                f"Removing the first {front_trim} staff(s), as they don't fit "
                "the staff layout the rest of the page repeats"
            )
        if back_trim > 0:
            eprint(
                f"Removing the last {back_trim} staff(s), as they don't fit "
                "the staff layout the rest of the page repeats"
            )
        if period > 1:
            eprint(
                "Systems repeat every",
                period,
                "staffs with a different layout each time, combining them into one row",
            )
        return SystemPlan.dense(_regroup_by_period(flat_staffs, period, front_trim, back_trim))
    result: list[MultiStaff] = []
    for staff in staffs:
        result.extend(staff.break_apart())
    return SystemPlan.dense(sorted(result, key=lambda s: s.staffs[0].min_y))


def _ensure_same_number_of_staffs(staffs: list[MultiStaff]) -> list[MultiStaff]:
    """Back-compatible view of _plan_systems for callers that only want the systems."""
    return _plan_systems(staffs).systems


def _get_number_of_voices(staffs: list[MultiStaff]) -> int:
    return len(staffs[0].staffs)


tr_omr_max_height = default_config.max_height
tr_omr_max_width = default_config.max_width


def get_tr_omr_canvas_size(
    image_shape: tuple[int, ...], margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    tr_omr_ratio = float(tr_omr_max_height_with_margin) / tr_omr_max_width
    height, width = image_shape[:2]

    # Calculate the new size such that it fits exactly into the
    # tr_omr_max_height and tr_omr_max_width
    # while maintaining the aspect ratio of height and width.

    if height / width > tr_omr_ratio:
        # The height is the limiting factor.
        new_shape = [
            int(width / height * tr_omr_max_height_with_margin),
            tr_omr_max_height_with_margin,
        ]
    else:
        # The width is the limiting factor.
        new_shape = [tr_omr_max_width, int(height / width * tr_omr_max_width)]
    return np.array(new_shape)


def center_image_on_canvas(
    image: NDArray, canvas_size: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    is_grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

    resized = cv2.resize(image, canvas_size)  # type: ignore

    if is_grayscale:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width),
            255,
            dtype=np.uint8,
        )
    else:
        new_image = np.full(
            (tr_omr_max_height, tr_omr_max_width, 3),
            255,
            dtype=np.uint8,
        )

    x_offset = 0
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    y_offset = (tr_omr_max_height_with_margin - resized.shape[0]) // 2 + margin_top

    new_image[
        y_offset : y_offset + resized.shape[0],
        x_offset : x_offset + resized.shape[1],
    ] = resized

    return new_image


def add_image_into_tr_omr_canvas(image: NDArray) -> NDArray:
    new_shape = get_tr_omr_canvas_size(image.shape)
    new_image = center_image_on_canvas(image, new_shape)
    return new_image


def remove_black_contours_at_edges_of_image(gray: NDArray, unit_size: float) -> NDArray:
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold = constants.black_spot_removal_threshold(unit_size)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < threshold or h < threshold:
            continue
        is_at_edge_of_image = x == 0 or y == 0 or x + w == gray.shape[1] or y + h == gray.shape[0]
        if not is_at_edge_of_image:
            continue
        average_gray_intensity = 127
        is_mostly_dark = np.mean(thresh[y : y + h, x : x + w]) < average_gray_intensity
        if is_mostly_dark:
            continue
        gray[y : y + h, x : x + w] = 255
    return gray


def _calculate_region(staff: Staff, regions: StaffRegions) -> NDArray:
    x_min = staff.min_x - 2 * staff.average_unit_size
    x_max = staff.max_x + 2 * staff.average_unit_size
    y_min = max(
        staff.min_y - 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_above(staff.min_y),
    )
    y_max = min(
        staff.max_y + 4 * staff.average_unit_size,
        regions.get_start_of_closest_staff_below(staff.max_y),
    )
    return np.array([int(x_min), int(y_min), int(x_max), int(y_max)])


def prepare_staff_image(
    debug: Debug, index: int, staff: Staff, staff_image: NDArray, regions: StaffRegions
) -> tuple[NDArray, Staff]:
    region = _calculate_region(staff, regions)
    image_dimensions = get_tr_omr_canvas_size(
        (int(region[3] - region[1]), int(region[2] - region[0]))
    )
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    staff_image = cv2.resize(
        staff_image,
        (int(staff_image.shape[1] * scaling_factor), int(staff_image.shape[0] * scaling_factor)),
    )
    region = np.round(region * scaling_factor)
    eprint("Dewarping staff", index)
    region_step1 = np.array(region) + np.array([-10, -50, 10, 50])
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step1)
    region_step2 = np.array(region) - np.array([*top_left, *top_left])
    top_left = top_left / scaling_factor
    staff = _dewarp_staff(staff, None, top_left, scaling_factor)
    dewarp = dewarp_staff_image(staff_image, staff, index, debug)
    staff_image = dewarp.dewarp(staff_image)
    staff_image, top_left = crop_image_and_return_new_top(staff_image, *region_step2)
    scaling_factor = 1

    eprint("Dewarping staff", index, "done")

    staff_image = remove_black_contours_at_edges_of_image(staff_image, staff.average_unit_size)
    staff_image = center_image_on_canvas(staff_image, image_dimensions)
    debug.write_image_with_fixed_suffix(f"_staff-{index}_input.jpg", staff_image)
    if debug.debug:
        transformed_staff = _dewarp_staff(staff, dewarp, top_left, scaling_factor)
        transformed_staff_image = staff_image.copy()
        for symbol in transformed_staff.symbols:
            center = symbol.center
            cv2.circle(transformed_staff_image, (int(center[0]), int(center[1])), 5, (0, 0, 255))
            cv2.putText(
                transformed_staff_image,
                type(symbol).__name__,
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )
        debug.write_image_with_fixed_suffix(
            f"_staff-{index}_debug_annotated.jpg", transformed_staff_image
        )
    return staff_image, staff


def _dewarp_staff(
    staff: Staff, dewarp: StaffDewarping | None, region: NDArray, scaling: float
) -> Staff:
    """
    Applies the same transformation on the staff coordinates as we did on the image.
    """

    def transform_coordinates(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        x -= region[0]
        y -= region[1]
        if dewarp is not None:
            x, y = dewarp.dewarp_point((x, y))
        x = x * scaling
        y = y * scaling
        return x, y

    return staff.transform_coordinates(transform_coordinates)


def parse_staff_image(
    debug: Debug, index: int, staff: Staff, image: NDArray, regions: StaffRegions, config: Config
) -> list[EncodedSymbol]:
    staff_image, transformed_staff = prepare_staff_image(
        debug, index, staff, image, regions=regions
    )
    eprint("Running TrOmr inference on staff image", index)
    result = parse_staff_tromr(staff_image=staff_image, staff=transformed_staff, config=config)
    if debug.debug:
        result_image = staff_image.copy()
        for i, symbol in enumerate(result):
            center = symbol.coordinates
            if center is None or symbol.rhythm.startswith("chord"):
                continue
            if math.isnan(center[0]) or math.isnan(center[1]):
                continue
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(result_image, center_int, 5, color=(0, 0, 255), thickness=2)
            cv2.putText(
                result_image,
                str(i) + ": " + symbol.rhythm,
                (center_int[0], center_int[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )

        debug.write_image_with_fixed_suffix(f"_staff-{index}_output.jpg", result_image)
    return result


def parse_staff_image_greedy_with_margins(
    debug: Debug,
    index: int,
    staff: Staff,
    image: NDArray,
    regions: StaffRegions,
    config: Config,
    staff_context_emb: NDArray | None = None,
) -> tuple[
    list[EncodedSymbol], list[EncodedSymbol], list[tuple[int, float]], object, object, NDArray
]:
    """Phase 1 (`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2) counterpart to
    `parse_staff_image`: the cheap first pass - one decode, same cost as
    `parse_staff_image` itself - that `parse_staffs` uses to decide, per system, whether
    the expensive forking pass is worth paying for at all. Returns
    `(filtered_greedy, raw_greedy, margins, context, decoder, hidden_states)`; see
    `parse_staff_tromr_greedy_with_margins` for what each carries and why forking needs
    the raw (unfiltered) sequence specifically. The debug-image side effect draws from
    `filtered_greedy`, identical to what `parse_staff_image` alone would have produced,
    so debug output does not depend on whether this system ends up reranked.

    `staff_context_emb`, given, makes this call §4/§7.4 Stage C's *second* pass instead
    of the first - see `parse_staff_tromr_greedy_with_margins`."""
    staff_image, transformed_staff = prepare_staff_image(
        debug, index, staff, image, regions=regions
    )
    eprint("Running TrOmr inference on staff image", index)
    filtered_greedy, raw_greedy, margins, context, decoder, hidden_states = (
        parse_staff_tromr_greedy_with_margins(
            staff_image=staff_image,
            staff=transformed_staff,
            config=config,
            staff_context_emb=staff_context_emb,
        )
    )
    if debug.debug:
        result_image = staff_image.copy()
        for i, symbol in enumerate(filtered_greedy):
            center = symbol.coordinates
            if center is None or symbol.rhythm.startswith("chord"):
                continue
            if math.isnan(center[0]) or math.isnan(center[1]):
                continue
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(result_image, center_int, 5, color=(0, 0, 255), thickness=2)
            cv2.putText(
                result_image,
                str(i) + ": " + symbol.rhythm,
                (center_int[0], center_int[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
            )

        debug.write_image_with_fixed_suffix(f"_staff-{index}_output.jpg", result_image)
    return filtered_greedy, raw_greedy, margins, context, decoder, hidden_states


#: §4/§7.4 Stage C: loaded once per weights path, not once per `parse_staffs` call -
#: the same reasoning `homr.staff_parsing_tromr`'s own `inference` global caching uses
#: for the encoder/decoder, just keyed by path since (unlike the ONNX models) more
#: than one trained StaffContextTransformer checkpoint could plausibly be compared in
#: one process.
_staff_context_cache: dict[str, "StaffContextTransformer"] = {}


def _get_staff_context_module(weights_path: str, dim: int) -> "StaffContextTransformer":
    if weights_path not in _staff_context_cache:
        from homr.staff_context_decode import load_staff_context

        _staff_context_cache[weights_path] = load_staff_context(weights_path, dim)
    return _staff_context_cache[weights_path]


def parse_staffs(
    debug: Debug,
    staffs: list[MultiStaff],
    image: NDArray,
    config: Config,
    selected_staff: int = -1,
    score_profile: ScoreProfile | None = None,
    enable_phase1_rerank: bool = True,
    phase1_max_forks: int = 3,
    enable_staff_context: bool = False,
    staff_context_weights: str | None = None,
) -> list[list[EncodedSymbol]]:
    """
    Dewarps each staff and then runs it through an algorithm which extracts
    the rhythm and pitch information.

    `enable_phase1_rerank` (on by default - `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2,
    justified by a 200-page benchmark showing a 20.8% reduction in cross-staff Stage A
    findings with zero regressions, then a ground-truth spot-check where every
    resolvable corrected measure matched real ground truth exactly) reranks a system's
    staves against cross-staff cumulative barline agreement before committing to a
    final decode - real content correction, not the log-only diagnostics Stage A/B
    report elsewhere.

    Two-pass and gated *by design*, not by accident: every staff's greedy decode costs
    the same either way (`generate_with_rhythm_margins` only adds bookkeeping over a
    plain decode), but forking an alternate candidate is a full extra decode pass each -
    paying that for every staff on every page would cost several times a normal decode
    even on the majority of systems the 200-page benchmark found had nothing to fix.
    So: decode every staff's greedy result first (cheap), check Stage A's own
    `check_barline_positions`/`check_measure_durations` against the greedy decode per
    system, and only pay for forking + reranking on a system that already shows a
    finding there - exactly the population the benchmark and spot-check actually
    measured, not a broader, unvalidated "always fork everything" behavior.

    `enable_staff_context` (§4/§7.4 Stage C, off by default - not yet benchmarked the
    way `enable_phase1_rerank` has been) reuses the same per-system raw greedy decode
    and hidden states Phase 1 already collects: pools each present voice's hidden
    states, runs the trained `StaffContextTransformer` (`staff_context_weights`,
    required when this is on - e.g. this project's own `phase24-staff-context-weights`
    release) across the system's voices, then decodes every voice a second time with
    its own context vector. Applied independently of Phase 1 - it runs on the *original*
    raw greedy decode's hidden states regardless of whether Phase 1's rerank changed
    that system's result, and its own second-pass decode is what `decoded` ends up
    holding for that system if both are enabled (last write wins; the two mechanisms
    were built and tested separately, not as a combined pipeline). Systems with fewer
    than 2 present voices are skipped - the module's own reasoning for why a lone staff
    has nothing to attend across.

    Disabled automatically whenever `selected_staff` restricts processing to one staff
    (that debug mode leaves most systems missing most voices, the same reason
    `_report_cross_staff_findings` is skipped there too) - set to `False` explicitly to
    compare against the pre-Phase-1 decode for any other reason.
    """
    plan = _plan_systems(staffs)
    # For simplicity we call every staff in a multi staff a voice,
    # even if it's part of a grand staff.
    number_of_voices = plan.voices
    do_rerank = (enable_phase1_rerank or enable_staff_context) and selected_staff < 0
    i = 0
    regions = StaffRegions(plan.systems)

    def systems_for_voice(voice: int) -> list[int]:
        # A system can be missing this voice: detection came up a staff short and the
        # spacing said which one. That voice simply has no music from that system,
        # which is a gap in one part rather than the whole system's music going missing.
        return [
            system for system in range(len(plan.systems)) if plan.staff_for_voice(system, voice) is not None
        ]

    decoded: dict[tuple[int, int], list[EncodedSymbol]] = {}
    # Only populated when do_rerank: the raw materials a system's staves need for the
    # (possibly skipped) expensive forking pass.
    raw_by_system: dict[int, dict[int, tuple]] = {}

    for voice in range(number_of_voices):
        for staff_index, system in enumerate(systems_for_voice(voice)):
            staff = plan.staff_for_voice(system, voice)
            assert staff is not None  # systems_for_voice already filtered on this
            if selected_staff >= 0 and staff_index != selected_staff:
                eprint("Ignoring staff due to selected_staff argument", i)
                i += 1
                continue
            if do_rerank:
                filtered, raw, margins, context, decoder, hidden_states = (
                    parse_staff_image_greedy_with_margins(debug, i, staff, image, regions, config)
                )
                decoded[(voice, system)] = filtered
                raw_by_system.setdefault(system, {})[voice] = (
                    staff, raw, margins, context, decoder, hidden_states,
                )
            else:
                decoded[(voice, system)] = parse_staff_image(debug, i, staff, image, regions, config)
            i += 1

    if enable_phase1_rerank and do_rerank:
        # staff_index within a system = rank among present voices, ascending - the
        # same convention staves_by_system/analyze_system already use, so a system's
        # greedy (pre-fork) decode lines up correctly against Stage A's own indexing
        # for this cheap pre-check.
        for system_index, voice_raw in raw_by_system.items():
            present_voices = sorted(voice_raw)
            greedy_staves = [decoded[(voice, system_index)] for voice in present_voices]
            has_finding = bool(
                check_barline_positions(greedy_staves) or check_measure_durations(greedy_staves)
            )
            if len(present_voices) < 3 or not has_finding:
                continue  # nothing Stage A already flags here - not worth forking

            candidates_by_staff = {}
            for staff_index, voice in enumerate(present_voices):
                staff, raw_greedy, margins, context, decoder, _hidden_states = voice_raw[voice]
                forks = fork_candidates_from_margins(
                    decoder,
                    raw_greedy,
                    margins,
                    max_forks=phase1_max_forks,
                    seq_len=config.max_seq_len,
                    eos_token=config.eos_token,
                    context=context,
                )
                # Forking operated on the *raw* (unfiltered) sequence to keep step
                # indices aligned with the model's own numbering - apply the same
                # grandstaff/lower filter parse_staff_tromr already applies to a plain
                # decode, to every candidate, so reranking compares like with like.
                candidates_by_staff[staff_index] = [
                    candidate if staff.is_grandstaff else [r for r in candidate if r.position != "lower"]
                    for candidate in forks
                ]

            reranked = rerank_staff_candidates(candidates_by_staff)
            for staff_index, voice in enumerate(present_voices):
                decoded[(voice, system_index)] = reranked[staff_index]

    if enable_staff_context and do_rerank:
        if not staff_context_weights:
            raise ValueError("enable_staff_context requires staff_context_weights")
        import torch

        from homr.staff_context_decode import pool_hidden

        staff_context_module = _get_staff_context_module(staff_context_weights, config.decoder_dim)
        for system_index, voice_raw in raw_by_system.items():
            present_voices = sorted(voice_raw)
            if len(present_voices) < 2:
                continue  # nothing for cross-staff attention to attend across

            pooled = np.stack([pool_hidden(voice_raw[voice][5]) for voice in present_voices])
            with torch.no_grad():
                mask = torch.ones(1, len(present_voices), dtype=torch.bool)
                context_vec = staff_context_module(
                    torch.from_numpy(pooled).float().unsqueeze(0), mask
                )
            context_np = context_vec.squeeze(0).numpy()
            fp16 = getattr(voice_raw[present_voices[0]][4], "fp16", False)
            context_np = context_np.astype(np.float16 if fp16 else np.float32)

            for slot, voice in enumerate(present_voices):
                staff = voice_raw[voice][0]
                filtered2, *_rest = parse_staff_image_greedy_with_margins(
                    debug, i, staff, image, regions, config,
                    staff_context_emb=context_np[slot : slot + 1],
                )
                decoded[(voice, system_index)] = filtered2
                i += 1

    voices = []
    for voice in range(number_of_voices):
        result_for_voice = []
        for staff_index, system in enumerate(systems_for_voice(voice)):
            if selected_staff >= 0 and staff_index != selected_staff:
                continue
            result_staff = decoded.get((voice, system))
            if not result_staff:
                eprint("Skipping empty staff")
                continue
            result_for_voice.extend([*result_staff, EncodedSymbol("newline")])
        voices.append(remove_duplicated_symbols(result_for_voice))

    if selected_staff < 0:
        # Only meaningful for a normal run: selected_staff restricts processing to one
        # staff, so most voices are deliberately absent from most systems rather than
        # genuinely missing, and the presence map below would not describe what
        # findings_by_page actually received.
        _report_cross_staff_findings(plan, voices, score_profile)
    return voices


def _report_cross_staff_findings(
    plan: SystemPlan, voices: list[list[EncodedSymbol]], score_profile: ScoreProfile | None
) -> None:
    """Stage A (design §12.1, plus a later shared-motif addition and the page-wide staff-
    count check), Stage B tier 1 (design §12.2, key/time signature and, since this
    session, motif-corroborated articulation), and §7.2's profile-layout deviations:
    log deterministic cross-staff disagreements and, where one exists, the majority-
    correction proposal for it - without altering anything `voices` carries forward. The
    clef-vs-profile check and the layout deviation report both only fire when the caller
    supplied a score profile; every other check runs regardless. Proposals are logged
    only, never applied - the same "review question, not an automatic correction" §12.2
    states for Stage B generally; nothing here changes `voices` or what `parse_staffs`
    returns.

    `propose_part_assignment`'s deviations (a system's detected staff count not matching
    what the profile expects) are the first real consumer of that function - previously
    built and tested but never called from anywhere, per the next-steps doc's own
    tracking. Distinct from `check_page_staff_counts`, which compares a system against
    the rest of the *page*: this compares a system against what the *profile* declared,
    which can disagree even when every system on the page agrees with each other (a
    profile that is simply wrong about how many parts this piece has).

    Stage A run end to end against two real OSSQ pages (`sq7313978:0001`,
    `sq8823783:0061`) on a GPU instance: no exception, `homr.main` wrote MusicXML
    normally in both cases, and it surfaced real findings (a key-signature restatement
    that only one of four parts carried into its second system; a time-signature and
    two key-signature mismatches on the second page) - genuine per-voice decode
    disagreements, not this module misreading its own input. The tier-1 proposal logic
    added alongside it has not yet had the same real-page validation. Still guarded
    regardless: a diagnostic that can find nothing new about a page's music must never
    be the reason a page fails to transcribe, so a bug here is logged and swallowed
    rather than allowed to propagate past what is otherwise a log-only addition.
    """
    try:
        presence = [
            [plan.staff_for_voice(system, voice) is not None for voice in range(len(voices))]
            for system in range(len(plan.systems))
        ]
        staff_to_part = (
            staff_to_part_by_system(score_profile, presence) if score_profile is not None else None
        )
        if score_profile is not None:
            for system_index, slots in enumerate(plan.slots):
                # propose_part_assignment (§7.2) wants one SystemPartition describing
                # every system at once, with a single page-wide staves_per_system - a
                # shape SystemPlan does not use, since its own systems can vary in size
                # (an incomplete system recovered by _group_by_geometry, or the dense
                # fallback path). Calling it once per system, with a single-system
                # partition built from that system's own slot count, sidesteps the
                # mismatch without needing propose_part_assignment itself to change:
                # each call answers exactly "does the profile expect this many staves
                # here", which is the only thing this diagnostic needs.
                partition = SystemPartition(
                    staves_per_system=len(slots),
                    groups=(tuple(range(len(slots))),),
                    separation=0.0,
                    broken_connections=0,
                )
                assignment = propose_part_assignment(score_profile, partition, [slots])[0]
                for deviation in assignment.deviations:
                    eprint(f"System {system_index}: profile layout - {deviation}")
        for finding in check_page_staff_counts(presence):
            eprint(f"Page: {finding.message}")
        if staff_to_part is not None:
            for finding in check_part_order(staff_to_part):
                eprint(f"Page: {finding.message}")
        for system_index, staves in enumerate(staves_by_system(voices, presence)):
            part_map = staff_to_part[system_index] if staff_to_part else None
            for finding in analyze_system(staves, part_map):
                eprint(f"System {system_index}: {finding.message}")
            for proposal in propose_repairs(staves):
                eprint(
                    f"System {system_index}: repair proposal - {proposal.reason} "
                    f"(staff {proposal.staff_index}, position {proposal.position}: "
                    f"{proposal.current_rhythm!r} -> {proposal.proposed_rhythm!r})"
                )
            for articulation_proposal in propose_motif_articulation_corrections(staves):
                eprint(
                    f"System {system_index}: repair proposal - "
                    f"{articulation_proposal.reason} "
                    f"(staff {articulation_proposal.staff_index}, "
                    f"position {articulation_proposal.position}: "
                    f"{articulation_proposal.current_articulation!r} -> "
                    f"{articulation_proposal.proposed_articulation!r})"
                )
            for insertion_proposal in propose_carry_forward_key_signature(staves):
                eprint(
                    f"System {system_index}: repair proposal - "
                    f"{insertion_proposal.reason} "
                    f"(staff {insertion_proposal.staff_index}, "
                    f"insert before position {insertion_proposal.position}: "
                    f"{insertion_proposal.inserted_rhythm!r})"
                )
            for position_proposal in propose_majority_position_corrections(staves):
                eprint(
                    f"System {system_index}: repair proposal - "
                    f"{position_proposal.reason} "
                    f"(staff {position_proposal.staff_index}, "
                    f"measure {position_proposal.measure_index}, "
                    f"offset {position_proposal.offset})"
                )
    except Exception as error:  # noqa: BLE001
        eprint(f"Cross-staff consistency check failed, skipping: {error}")
