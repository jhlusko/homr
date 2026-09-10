"""
Score-level train/validation/test membership for OSSQ-OMR.

The design requires that every crop, system, page and movement of one score lands in a
single split, and that the split be an explicit manifest under version control whose
hash is recorded with each run - never something a data loader improvises at sample
level. `ossq_split_manifest.json` is that manifest.

It is the split published with the OSSQ-OMR benchmark, adopted verbatim rather than
invented here, so anything measured on these scores stays comparable with the paper's
numbers. Deriving our own would have silently forfeited that while looking equally
reasonable.

Four splits, keyed by score and by image track:

    train         100 scores, both tracks    13,480 segments
    valid          11 scores, both tracks     1,610 segments
    test_synth     11 scores, synthetic only    945 segments
    test_scanned   10 scores, scanned only      959 segments

train, valid and the test sets share no score. The two test sets do overlap by ten
scores, which is not leakage: those are the same held-out works seen once per track, so
a synthetic and a scanned reading of one score never end up on opposite sides of the
train/test line.
"""

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SPLIT_NAMES = ("train", "valid", "test_synth", "test_scanned")
TRACKS = ("synthetic", "scanned")

#: Splits whose scores must never share a score with each other.
_DISJOINT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("train",),
    ("valid",),
    ("test_synth", "test_scanned"),
)

_MANIFEST_PATH = Path(__file__).with_name("ossq_split_manifest.json")


@dataclass(frozen=True)
class SplitManifest:
    scores: dict[str, dict[str, str]]
    #: score id -> "<Composer>/<Work>", relative to the dataset's scores/ directory.
    paths: dict[str, str]
    segment_counts: dict[str, int]
    source: dict[str, str]
    #: sha256 of the manifest file, to stamp into run metadata so a result can be tied
    #: back to the exact split that produced it.
    digest: str

    def split_for(self, score_id: str, track: str) -> str | None:
        """Which split this score belongs to for this track, or None if it has none.

        A score can be absent from a track: the ten scores in test_scanned have no
        synthetic test entry, and scores without a usable scan have no scanned entry at
        all.
        """
        return self.scores.get(score_id, {}).get(track)

    def scores_in(self, split: str, track: str | None = None) -> set[str]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
        return {
            score
            for score, tracks in self.scores.items()
            if any(
                assigned == split and (track is None or name == track)
                for name, assigned in tracks.items()
            )
        }

    def check_no_leakage(self) -> None:
        """Raise if any score appears in two groups that must stay disjoint."""
        members = {
            group: set().union(*(self.scores_in(split) for split in group))
            for group in _DISJOINT_GROUPS
        }
        for index, first in enumerate(_DISJOINT_GROUPS):
            for second in _DISJOINT_GROUPS[index + 1 :]:
                shared = members[first] & members[second]
                if shared:
                    raise ValueError(
                        f"scores in both {'/'.join(first)} and {'/'.join(second)}: "
                        f"{sorted(shared)}"
                    )


@lru_cache(maxsize=1)
def load_split_manifest(path: Path | None = None) -> SplitManifest:
    manifest_path = path or _MANIFEST_PATH
    raw = manifest_path.read_bytes()
    data = json.loads(raw)
    schema = data.get("schemaVersion")
    if schema != "homr.ossq-split.v1":
        raise ValueError(f"unsupported split manifest schema {schema!r}")
    return SplitManifest(
        scores={score: body["tracks"] for score, body in data["scores"].items()},
        paths={score: body["path"] for score, body in data["scores"].items()},
        segment_counts=data["segmentCounts"],
        source=data["source"],
        digest=hashlib.sha256(raw).hexdigest(),
    )
