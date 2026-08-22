"""
Pair an OSSQ training example with the real `ScoreProfile` extracted from its source
score, for §7.3 (design §7.3, `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §3).

`convert_ossq.py`'s own per-part tokenisation scratch file (`extract_part`) strips
instrument identity entirely - every part becomes a generic "Part 1" with no
`<score-instrument>`, since that scratch file exists only to tokenise one part in
isolation. The *original* whole-score file `convert_ossq.py` already locates for
slur/dynamics placement (`work / f"{score_id}.musicxml"`) still has the real
`<part-list>`, so this reads from there instead, using the part index
`convert_ossq.py`'s own crop naming already carries
(`<score_id>_<page>_<system>_<part>`, 1-based, matching `CROP_NAME` there) to pick
which `ScorePart` in that profile a given training example belongs to.

Scoped to OSSQ specifically - other corpora feeding decoder training (mbox-derived,
lieder, ...) have their own naming and provenance and would need their own pairing
logic, not attempted here. A stem that does not match OSSQ's convention is simply "no
profile available for this sample," not an error - decoder training mixes corpora
routinely (`mix_datasets.py`), and most samples in a mixed batch will not be OSSQ.
"""

import re
from functools import lru_cache
from pathlib import Path

from homr.score_profile import ScorePart, ScoreProfile
from training.omr_datasets.score_profile_extraction import extract_score_profile_from_file

_STEM_PATTERN = re.compile(r"^(?P<score_id>.+)_(?P<page>\d+)_(?P<system>\d+)_(?P<part>\d+)$")


def parse_ossq_stem(stem: str) -> tuple[str, int] | None:
    """`(score_id, 1-based part index)` from an OSSQ training example's filename stem
    (`convert_ossq.py`'s `stem = f"{score_id}_{page}_{system}_{part_index + 1}"`), or
    `None` if it does not match that convention at all.
    """
    match = _STEM_PATTERN.match(stem)
    if match is None:
        return None
    return match.group("score_id"), int(match.group("part"))


@lru_cache(maxsize=None)
def _find_score_musicxml(dataset_root: str, score_id: str) -> Path | None:
    """The one whole-score MusicXML `convert_ossq.py` itself locates as
    `work / f"{score_id}.musicxml"` - cached, since every part of every system in one
    score shares the same source file, and searching a multi-thousand-directory corpus
    once per training example would be needless repeated work.
    """
    matches = list(Path(dataset_root).glob(f"scores/*/*/{score_id}.musicxml"))
    return matches[0] if matches else None


@lru_cache(maxsize=None)
def _profile_for_score(dataset_root: str, score_id: str) -> ScoreProfile | None:
    path = _find_score_musicxml(dataset_root, score_id)
    if path is None:
        return None
    return extract_score_profile_from_file(str(path))


def profile_and_part_for_sample(
    dataset_root: str, stem: str
) -> tuple[ScoreProfile, ScorePart] | None:
    """The real `(ScoreProfile, ScorePart)` for one OSSQ training example, or `None`
    when it cannot be resolved - a non-OSSQ stem, a score whose whole-score MusicXML is
    missing, or a part index the extracted profile does not have that many parts for.
    The scratch tokenisation and the original document disagreeing on part count would
    itself be worth investigating separately, but silently returning "no profile" here
    is the same "unknown is valid, not an error" discipline the rest of this design
    uses - not a guess at which part it might have meant.
    """
    parsed = parse_ossq_stem(stem)
    if parsed is None:
        return None
    score_id, part_index = parsed
    profile = _profile_for_score(dataset_root, score_id)
    if profile is None:
        return None
    if not 1 <= part_index <= len(profile.parts):
        return None
    return profile, profile.parts[part_index - 1]
