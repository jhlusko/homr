"""
The correct way to find OSSQ-OMR ground truth and page-to-score measure alignment -
built after discovering that every previous script in this investigation
(ossq_measure_length_audit.py, deep_barline_audit.py, deep_barline_audit_broad.py) read
`<page>.musicxml` as ground truth, which is actually `homr.main`'s own prior output
written to that exact path (confirmed via its `<software>homr</software>` metadata).
See OSSQ_GROUND_TRUTH_ERRORS.md's retraction and DECODER_RHYTHM_ACCURACY_DESIGN.md §7.1
for the full account.

Real ground truth lives at each piece's own top level: `scores/<composer>/<piece>/
sq<id>.musicxml`. The page-local-to-absolute measure mapping comes from the corpus's own
`metadata/{scanned,synthetic,unaligned}/{systemwise/,}sq<id>:<page>:<system>.yaml` -
`measure_start`/`measure_end`, corpus-provided, not reconstructed.

SECOND BUG, found and fixed here: multi-movement pieces (e.g. string quartets with 3-4
movements) concatenate all movements into one ground-truth file, and each movement
*restarts* MusicXML `<measure number="...">` at 1 (or 0, for a pickup measure) - and the
corpus's own `measure_start`/`measure_end` metadata restarts the same way at each
movement boundary (confirmed by inspecting a full page sequence: a system's
`measure_start`/`measure_end` decreasing relative to the previous system's always lines
up exactly with a `<measure number="1">` reset in the ground truth file). A naive
`m.get("number") == str(target)` match against the *whole* ground-truth file is
therefore unsafe - the same number string can occur once per movement, and either
silently picks the wrong movement's measure (if code takes the first match) or splices
unrelated movements together (if code keeps every match). `movement_index_for_system` +
`resolve_flat_measure_range` fix this by first counting movement resets in the corpus's
own metadata sequence to find which movement a page/system belongs to, then matching by
number only within that movement's own slice of measures - where numbers are unique.
"""
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import yaml


def piece_dir(image_path: Path) -> Path:
    """images/{scanned,synthetic}/original/<page>.png -> the piece's own directory."""
    return image_path.parents[3]


def score_and_page(image_path: Path) -> tuple[str, str]:
    """'sq10675759:0024.png' -> ('sq10675759', '0024')."""
    stem = image_path.stem
    score_id, page_str = stem.split(":")
    return score_id, page_str


def real_ground_truth_path(image_path: Path) -> Path | None:
    gt = piece_dir(image_path) / f"{score_and_page(image_path)[0]}.musicxml"
    return gt if gt.exists() else None


def fragment_path(piece_dir_path: Path, page: int, system_num: int) -> Path:
    """Where `split_ground_truth_by_system.py`'s pre-extracted, movement-disambiguated
    per-system ground-truth window lives (or would live) for this piece - `system_num`
    is 1-based, matching the corpus's own systemwise metadata convention. Shared by the
    splitter (writer) and `score_profile_time_signature.py` (reader) so both agree on
    the one naming convention without either depending on the other's module."""
    return piece_dir_path / "metadata" / "systemwise_ground_truth" / f"{page:04d}:{system_num:04d}.musicxml"


def measure_start_for_system(image_path: Path, system_index: int) -> int | None:
    """`system_index` is HOMR's own 0-based system index on this page. Returns the
    corpus's own `measure_start` for that system (the *movement-local* number of its
    first measure - see module docstring), or None if no metadata is found for this
    page/system."""
    score_id, page_str = score_and_page(image_path)
    system_num = system_index + 1  # corpus's own system_idx is 1-based
    candidates = [
        piece_dir(image_path) / "metadata" / "scanned" / "systemwise"
        / f"{score_id}:{page_str}:{system_num:04d}.yaml",
        piece_dir(image_path) / "metadata" / "synthetic" / "systemwise"
        / f"{score_id}:{page_str}:{system_num:04d}.yaml",
        piece_dir(image_path) / "metadata" / "unaligned"
        / f"{score_id}:{page_str}:{system_num:04d}.yaml",
    ]
    for path in candidates:
        if path.exists():
            data = yaml.safe_load(path.read_text())
            raw = data["measure_start"]
            try:
                return int(raw)
            except (TypeError, ValueError):
                # Some corpus metadata carries a non-numeric placeholder here (e.g.
                # "X2") - presumably marking an alignment the corpus itself is unsure
                # of. Treat exactly like no metadata rather than guessing at it.
                continue
    return None


def _systemwise_entries(image_path: Path) -> list[tuple[int, int, int, int]]:
    """All (page, system, measure_start, measure_end) triples found in the *aligned*
    metadata (scanned/systemwise, then synthetic/systemwise) for this piece, sorted in
    page order. Used only to detect movement boundaries via resets in this sequence.

    Deliberately excludes `unaligned` here, even though `measure_start_for_system` uses
    it as a last-resort per-lookup fallback: for at least one piece in this corpus
    (Wolf op.posth. String Quartet), `unaligned` contains a spurious page 1 whose
    measure_start/measure_end duplicate page 2's - a page numbering that doesn't
    participate in the real aligned sequence. Folding it into a piece-wide sequence
    used for reset-counting produces false movement-boundary detections.

    A thin, uncached wrapper over `_systemwise_entries_cached` - this function's own
    result depends only on the *piece* (every page of the same piece computes the
    identical list), but its argument is a page-specific `image_path`, which would
    make a naive `lru_cache` on this signature never hit for two different pages of
    the same piece. Found the hard way: this glob-and-parse-every-yaml-in-the-piece
    scan, run uncached once per training sample (`score_profile_time_signature.py`'s
    `time_signature_for_sample`, called from `data_loader.py`), took ~2 seconds per
    successful lookup on a real multi-hundred-page piece - `phase22`'s first training
    run stalled on epoch 1 for this exact reason before being caught and fixed."""
    score_id, _ = score_and_page(image_path)
    return _systemwise_entries_cached(str(piece_dir(image_path)), score_id)


@lru_cache(maxsize=None)
def _systemwise_entries_cached(piece_dir_str: str, score_id: str) -> list[tuple[int, int, int, int]]:
    base = Path(piece_dir_str) / "metadata"
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for sub in ("scanned", "synthetic"):
        folder = base / sub / "systemwise"
        if not folder.exists():
            continue
        for path in folder.glob(f"{score_id}:*.yaml"):
            m = re.match(rf"{re.escape(score_id)}:(\d+):(\d+)\.yaml", path.name)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)))
            if key in seen:
                continue  # a higher-priority subfolder already supplied this one
            data = yaml.safe_load(path.read_text())
            try:
                start, end = int(data["measure_start"]), int(data["measure_end"])
            except (TypeError, ValueError):
                continue
            seen[key] = (start, end)
    return sorted((page, system, start, end) for (page, system), (start, end) in seen.items())


def movement_index_for_system(image_path: Path, system_index: int) -> int | None:
    """Which 0-based movement (in page order) this page/system falls in, found by
    counting resets in the corpus's own measure_start/measure_end sequence across the
    whole piece (see module docstring for why this reset lines up with the ground
    truth file's own per-movement measure numbering).

    A page/system with no aligned (scanned/synthetic) metadata of its own - e.g. an
    opening page whose only metadata lives in `unaligned` - still has one unambiguous
    case: before the very first aligned entry, no movement transition could possibly
    have happened yet, so it must be movement 0. (A tempting further heuristic -
    "if the nearest aligned entries immediately before and after agree on movement
    index, nothing could have changed in between" - turns out unsound in this corpus:
    aligned coverage can have long gaps with no entries at all for many consecutive
    pages, e.g. this piece has none for pages 25-35, and a gap that size can easily
    hide a real movement transition invisibly. Tried and reverted after it produced a
    wrong answer for exactly that case.) Anywhere else is genuinely ambiguous and
    returns None rather than guessing - the same discipline `measure_start_for_system`
    applies to non-numeric placeholders."""
    score_id, page_str = score_and_page(image_path)
    target_page, target_system = int(page_str), system_index + 1
    target_key = (target_page, target_system)
    entries = _systemwise_entries(image_path)
    if not entries:
        # No aligned metadata anywhere in this piece (confirmed to happen: some pieces
        # carry only `unaligned` metadata, no `scanned`/`synthetic` folder at all) -
        # "before the first aligned entry" has no meaning when there is no aligned
        # entry to be before, so this is fully unknown, not movement 0.
        return None

    movement = 0
    prev_end: int | None = None
    any_before = False
    for page, system, start, end in entries:
        if prev_end is not None and start < prev_end:
            movement += 1
        prev_end = end
        if (page, system) == target_key:
            return movement
        if (page, system) < target_key:
            any_before = True

    return 0 if not any_before else None


def _movement_boundaries(measures: list) -> list[int]:
    """Flat 0-based indices into `measures` where a new movement begins (index 0
    always included), detected as any point where the integer `number` attribute
    decreases relative to the last successfully-parsed one."""
    bounds = [0]
    prev: int | None = None
    for i, m in enumerate(measures):
        try:
            n = int(m.get("number"))
        except (TypeError, ValueError):
            continue
        if prev is not None and n < prev:
            bounds.append(i)
        prev = n
    return bounds


@lru_cache(maxsize=None)
def parse_ground_truth(gt_path_str: str) -> ET.ElementTree:
    """A real ground-truth file cached by path - some of this corpus's whole-score
    MusicXML files are several MB (multi-movement string quartets can run into the
    thousands of measures across four parts), and `resolve_flat_measure_range`/
    `time_signature_for_sample` are both called once per training *sample*, not once
    per piece - many samples share the same file. Found the hard way: on top of
    `_systemwise_entries`' own uncached-metadata-scan cost (fixed separately, see its
    own docstring), each of these two call sites re-parsing the same multi-MB file
    independently left `phase22`'s first training run at ~600ms per lookup even after
    that fix - parsing is read-only everywhere this cache is used, so sharing one
    parsed tree across both call sites (and across every sample from the same piece)
    is safe."""
    return ET.parse(gt_path_str)


def resolve_flat_measure_range(
    gt_path: Path, movement_index: int, part_index: int, local_start: int, local_end: int
) -> tuple[int, int] | None:
    """Global 0-based [start, end] flat measure indices (inclusive) into this part's
    full <measure> list, for movement-local measure numbers local_start..local_end
    within the given 0-based movement. Matching by number attribute is only safe
    *within* one movement's own slice, since numbers restart per movement - matching
    against the whole file either silently picks the wrong movement's measure or
    (if keeping every match) splices unrelated movements together."""
    tree = parse_ground_truth(str(gt_path))
    parts = tree.getroot().findall(".//part")
    if part_index >= len(parts):
        return None
    measures = parts[part_index].findall("measure")
    bounds = _movement_boundaries(measures)
    if movement_index >= len(bounds):
        return None
    lo = bounds[movement_index]
    hi = bounds[movement_index + 1] if movement_index + 1 < len(bounds) else len(measures)
    target_numbers = {str(n) for n in range(local_start, local_end + 1)}
    matches = [lo + i for i, m in enumerate(measures[lo:hi]) if m.get("number") in target_numbers]
    if not matches:
        return None
    return min(matches), max(matches)


def extract_ground_truth_window(
    gt_path: Path, movement_index: int, measure_start: int, measure_end: int, out_path: Path
) -> bool:
    """Writes a small, self-contained MusicXML window covering just
    `measure_start`..`measure_end` (movement-local) of every part in `gt_path`, with
    each part's last-declared clef/key/time/divisions carried forward from earlier in
    the *same movement* - so the fragment renders or parses correctly standalone
    without needing the whole score. Returns `False` (writes nothing) if no part has
    any measure in range.

    Generalized from `build_review_assets.py`'s original `extract_gt_window` (this
    session's corpus-review webpage) into a reusable library function - the same
    movement-aware extraction is also what `split_ground_truth_by_system.py` needs to
    run corpus-wide, once per (piece, page, system) instead of once per review entry.

    Deliberately parses `gt_path` fresh here rather than reusing `parse_ground_truth`'s
    shared cache: this function *mutates* the tree (removes out-of-range `<measure>`
    elements) to build the window, and mutating the cached tree would corrupt it for
    every other lookup against the same piece. Called once per (piece, page, system)
    during preprocessing, not once per training sample - an independent parse here
    costs nothing at the scale this function actually runs at.
    """
    tree = ET.parse(gt_path)
    root = tree.getroot()
    parts = root.findall(".//part")
    any_kept = False
    for part_index, part in enumerate(parts):
        measures = part.findall("measure")
        flat_range = resolve_flat_measure_range(
            gt_path, movement_index, part_index, measure_start, measure_end
        )
        if flat_range is None:
            continue
        lo, hi = flat_range  # inclusive flat indices

        # Carry forward clef/key/time/divisions from earlier in *this same movement*
        # only (a movement's own first measure always redeclares them, so searching
        # past the movement's own start would risk carrying the wrong movement's
        # attributes forward instead).
        #
        # Tracked per *child element* (divisions/key/time/clef), not "does <attributes>
        # exist at all" - MusicXML only restates a child when it changes, so a measure
        # can carry a real <attributes> with a <time> change but no <divisions> (or vice
        # versa). Skipping the whole carry-forward step whenever *any* <attributes>
        # element already exists silently leaves divisions at kern_to_symbol_duration's
        # (and measure_length_by_part's) implicit default of 1 for every part whose
        # window happens to open on an attribute change that isn't the one it needs -
        # found the hard way: system_measure_curve computed measure lengths inflated by
        # exactly the piece's real (uncarried) divisions value for any part whose window
        # opened this way, on real corpus data, before phase23 could even finish epoch 1.
        bounds = _movement_boundaries(measures)
        movement_start_idx = bounds[movement_index] if movement_index < len(bounds) else 0
        carried_children: dict[str, ET.Element] = {}
        for m in measures[movement_start_idx:lo]:
            attrs = m.find("attributes")
            if attrs is None:
                continue
            for child in attrs:
                carried_children[child.tag] = child

        keep = set(range(lo, hi + 1))
        for i, m in enumerate(list(measures)):
            if i not in keep:
                part.remove(m)
            else:
                any_kept = True
        remaining = part.findall("measure")
        if remaining and carried_children:
            first = remaining[0]
            first_attrs = first.find("attributes")
            if first_attrs is None:
                first_attrs = ET.Element("attributes")
                first.insert(0, first_attrs)
            present_tags = {child.tag for child in first_attrs}
            for tag, child in carried_children.items():
                if tag not in present_tags:
                    first_attrs.append(child)
    if not any_kept:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    return True
