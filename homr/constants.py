number_of_lines_on_a_staff = 5

max_number_of_ledger_lines = 4


def tolerance_for_staff_line_detection(unit_size: float) -> float:
    return unit_size / 3


def max_line_gap_size(unit_size: float) -> float:
    return 5 * unit_size


def is_short_line(unit_size: float) -> float:
    return unit_size / 5


def is_short_connected_line(unit_size: float) -> float:
    return 2 * unit_size


def min_height_for_brace_rough(unit_size: float) -> float:
    return 2 * unit_size


def max_width_for_brace_rough(unit_size: float) -> float:
    return 3 * unit_size


def min_height_for_brace(unit_size: float) -> float:
    return 4 * unit_size


def tolerance_for_touching_clefs(unit_size: float) -> int:
    return int(round(unit_size * 2))


def tolerance_for_staff_at_any_point(unit_size: float) -> int:
    return 0


def tolerance_note_grouping(unit_size: float) -> float:
    return 1 * unit_size


def bar_line_max_width(unit_size: float) -> float:
    return 2 * unit_size


def bar_line_min_height(unit_size: float) -> float:
    return 3 * unit_size


def black_spot_removal_threshold(unit_size: float) -> float:
    return 2 * unit_size


staff_line_segment_x_tolerance = 10

# We don't have to worried about mis-detections,
# because if not all staffs group the same way then we break the staffs up again
minimum_connections_to_form_combined_staff = 1

duration_of_quarter = 16

image_noise_limit = 50

staff_position_tolerance = 50

max_angle_for_lines_to_be_parallel = 10


NOTEHEAD_SIZE_RATIO = 1.285714  # width/height

grandstaff_x_distance_threshold_factor = 5
grandstaff_y_overlap_threshold_factor = 0.5

# A brace/bracket candidate blob can end up merged with unrelated ink that
# happens to touch it during preprocessing (e.g. a neighboring staff's clef).
# Such contamination is reliably thinner than the brace itself, so we keep
# only the vertical range where the blob is at least this fraction as wide
# as its own widest row, which recovers the true brace span regardless of
# how much extra ink got attached to it.
brace_core_width_ratio = 0.5

# prepare_brace_dot_image's morphological dilation (homr/brace_dot_detection.py) uses a
# 5px-wide kernel to bridge small gaps in brace/bracket ink into single blobs. A whole-system
# bracket that crosses a staff line's own ink can occasionally break into two contours there:
# the main bracket (correctly wide) plus a small, sparse leftover fragment too thin to fully
# dilate back up to kernel width. That leftover is small enough to fit entirely within a
# single staff pair's span, so it can still score well in _score_brace_with_staff_pair despite
# being noise, not a real brace - this is an absolute pixel width tied to the kernel itself,
# not to staff unit size, since it is about the morphology op, not the music engraving.
# 5 (not 4): observed leftover fragments were 2-4px wide, while every genuine candidate
# (barline connectors, real braces) observed across smb/polish-scores/testdata was >=5px -
# exactly the dilation kernel's own width.
min_width_for_brace_dot_candidate = 5


# --- Deterministic page-level system grouping (homr/system_grouping.py) ---

# How much larger the mean gap between systems must be than the mean gap inside one,
# in staff unit sizes, before the geometric partition is trusted over the caller's
# existing behaviour. Measured on a string quartet page whose bracket detection had gone
# inconsistent: internal gaps 3.6-6.7, system gaps 8.7-9.1, separation 3.1. A page of
# genuinely independent single staves has one population of gaps and separates at ~0, so
# this sits well below the real signal and well above the noise floor.
min_system_gap_separation = 2.0

# Every cut must also clear the *median* internal gap by this factor. The mean-based
# separation above can stay positive while individual boundaries sit in the wrong place;
# this is what pins them. Median rather than max because a staff missed by detection
# leaves one double-width gap inside a system (14.8 unit sizes on the page above, against
# an 8.7 smallest cut) and a max-based test would reject the correct partition outright.
min_cut_to_internal_gap_ratio = 1.2

# Cutting where the bracket/barline detector saw a connection costs this much score, in
# unit sizes. Deliberately comparable to min_system_gap_separation: bracket evidence
# should be able to overturn a marginal geometric win but not a decisive one, since that
# detector's disagreements are the reason this module exists.
broken_connection_penalty = 2.0
max_broken_connections_for_grouping = 0

# Geometry can only separate two populations of gaps once the page shows several systems.
# Below this the page is one or two systems, where there is no repetition to read and the
# bracket detector's own grouping is the better evidence.
min_systems_for_geometric_grouping = 3

# An upper bound on staves per system, to keep the search finite on pages with many
# staves. Comfortably above a string quartet (4) or a voice-plus-piano system (3), and
# above the largest ensembles this pipeline currently targets.
max_staves_per_system = 8

# Two staffs closer together than this, in unit sizes, are not two staffs: a negative gap
# means they overlap vertically, which happens when one staff line gets detected twice
# (e.g. once as its left half and once as its right). The small negative tolerance
# absorbs dewarping jitter at a genuine boundary without admitting a real overlap - the
# observed duplicate overlapped by 4.4 unit sizes, two orders of magnitude past this.
min_gap_for_distinct_staffs = -0.5
