"""
The optional, document-scoped score profile a caller may supply: which parts a score is
expected to contain.

This is a hint, not a constraint. `homr` reads a page without one today, and every field
here is designed so an absent, partial, or wrong profile degrades gracefully rather than
breaking transcription - `stableId` is scoped to one submitted job, not a universal
instrument registry, and nothing downstream may treat `likelyClefs` or a transposition as
a hard pitch rule (real music legitimately violates its own instrumentation's usual
range). The profile earns its keep by turning "four staves, unlabelled" into "these four
staves are probably Violin I/II, Viola, Cello, in that order" - a prior a human reviewer
can see and reject, not an assertion the pipeline enforces silently.

Kept schema-versioned and dependency-light (dataclasses, stdlib only) the same way
`homr.transformer.capability_manifest` is, since both are contracts a caller outside this
package may serialize, store, and hand back later - a version mismatch has to be a
refusal, not a guess at a shape that may have changed.
"""

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "homr.score-profile.v1"


class ScoreProfileSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ScorePart:
    """One expected part of a document. Every field but `stableId` is optional evidence,
    not a requirement - an instrument family homr has never seen, or an unknown clef, are
    both valid rather than errors, since the caller may know less than the schema can
    name."""

    #: Scoped to the submitted document/job, not a universal instrument identifier - two
    #: different jobs may reuse "violin-1" for unrelated scores.
    stable_id: str
    display_name: str = ""
    instrument_family: str = ""
    #: Physical staff lines this part occupies - 1 for a violin, 2 for a piano grand
    #: staff. Drives how many consecutive detected staves one part's worth of music
    #: should span; see `score_profile_layout.py`.
    expected_staff_count: int = 1
    likely_clefs: tuple[str, ...] = ()
    transposition_semitones: int = 0
    lyrics_expected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stableId": self.stable_id,
            "displayName": self.display_name,
            "instrumentFamily": self.instrument_family,
            "expectedStaffCount": self.expected_staff_count,
            "likelyClefs": list(self.likely_clefs),
            "transpositionSemitones": self.transposition_semitones,
            "lyricsExpected": self.lyrics_expected,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ScorePart":
        if "stableId" not in data:
            raise ScoreProfileSchemaError("a score part requires stableId")
        return ScorePart(
            stable_id=data["stableId"],
            display_name=data.get("displayName", ""),
            instrument_family=data.get("instrumentFamily", ""),
            expected_staff_count=data.get("expectedStaffCount", 1),
            likely_clefs=tuple(data.get("likelyClefs", ())),
            transposition_semitones=data.get("transpositionSemitones", 0),
            lyrics_expected=data.get("lyricsExpected", False),
        )


@dataclass(frozen=True)
class ScoreProfile:
    """An ordered set of expected parts, top to bottom the way a score prints them."""

    parts: tuple[ScorePart, ...] = ()
    schema_version: str = SCHEMA_VERSION

    @property
    def total_staff_count(self) -> int:
        """Physical staves the whole profile expects, summed across parts - the number a
        detected system's staff count is compared against before any per-part mapping is
        attempted."""
        return sum(part.expected_staff_count for part in self.parts)

    @property
    def expected_staff_pattern(self) -> tuple[str, ...]:
        """One `stableId` per expected physical staff, in top-to-bottom order - a piano
        part with `expectedStaffCount=2` contributes its `stableId` twice, adjacently, so
        this tuple's length always equals `total_staff_count` and index i is directly
        comparable to the ith physical staff of a matching system."""
        pattern: list[str] = []
        for part in self.parts:
            pattern.extend([part.stable_id] * max(part.expected_staff_count, 0))
        return tuple(pattern)

    def part_by_id(self, stable_id: str) -> ScorePart | None:
        return next((part for part in self.parts if part.stable_id == stable_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "parts": [part.to_dict() for part in self.parts],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ScoreProfile":
        schema = data.get("schemaVersion")
        if schema != SCHEMA_VERSION:
            raise ScoreProfileSchemaError(f"unsupported score profile schema {schema!r}")
        return ScoreProfile(
            parts=tuple(ScorePart.from_dict(part) for part in data.get("parts", ())),
            schema_version=schema,
        )


#: A string quartet: four single-staff parts, top to bottom. Not a special case in code -
#: any caller may build the same shape - but named here because it is this design's
#: running example and the corpus (OSSQ) every structured head so far has trained on.
STRING_QUARTET = ScoreProfile(
    parts=(
        ScorePart("violin-1", "Violin I", "strings.violin", likely_clefs=("G2",)),
        ScorePart("violin-2", "Violin II", "strings.violin", likely_clefs=("G2",)),
        ScorePart("viola", "Viola", "strings.viola", likely_clefs=("C3", "G2")),
        ScorePart("cello", "Cello", "strings.cello", likely_clefs=("F4", "C4", "G2")),
    )
)
