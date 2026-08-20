"""
§7.2/§7.3's score-profile context embedding (design §7, `ENSEMBLE_TRANSCRIPTION_NEXT_
STEPS.md` §3): instrument-family, part-ordinal, staff-within-part-ordinal, expected-
staff-count, likely-clef-set, and transposition, combined into one additive vector for
`ScoreTransformerWrapper`'s decoder input - plus an explicit missing-context case,
since a caller (or a training sample) with no profile at all is the common case, not
the exception, and needs its own learnable representation rather than silence.

**Zero-initialized gate**, per §7.2's own stated requirement: "the gate must be
zero-initialized so the unconditioned path is bit-identical at initialization." Every
sub-embedding here can be arbitrarily initialized - they need real gradients to learn
anything - it is only the *gate* that must start at zero, so their combined
contribution is exactly zero at the start of training regardless of what the
sub-embeddings compute. This is why a checkpoint that never receives profile context at
inference (a caller with nothing to supply) reproduces the pretrained model's existing
behavior exactly, not approximately: `embed_one(None, ...)` returns a genuine zero
before training moves the gate, not a small-but-nonzero vector that happens to be near
zero.

Deliberately knows nothing about where a `ProfileContext` comes from - `ScoreProfile`/
`ScorePart` (live inference), `training.omr_datasets.score_profile_pairing` (OSSQ
training data), or any future corpus's own pairing all build a `ProfileContext` and
hand it to this module; it has no per-corpus logic of its own.
"""

from dataclasses import dataclass

import torch
from torch import nn

from homr.score_profile import ScorePart

#: Bounded, explicit vocabularies - an unenumerable free-text field (`instrument_family`
#: is a MusicXML sound-ID string) needs a fixed index space to embed into; anything not
#: in the table falls into a shared "other/unknown" bucket (index 0) rather than growing
#: the table at training time, the same way a rare token would.
INSTRUMENT_FAMILIES = (
    "strings.violin", "strings.viola", "strings.cello", "strings.contrabass",
    "strings.harp", "keyboard.piano", "keyboard.organ", "keyboard.harpsichord",
    "voice.vocals", "wind.flutes.flute", "wind.flutes.piccolo", "wind.reed.oboe",
    "wind.reed.clarinet", "wind.reed.bassoon", "wind.reed.saxophone",
    "brass.french-horn", "brass.trumpet", "brass.trombone", "brass.tuba",
    "drum.timpani", "pluck.guitar",
)
CLEFS = ("G2", "F4", "C1", "C2", "C3", "C4", "C5", "TAB5")

MAX_PART_ORDINAL = 8  # clipped, not truncated - a 9th part still gets the 8th bucket
MAX_STAFF_WITHIN_PART = 4
MAX_STAFF_COUNT = 4
MIN_TRANSPOSITION = -24
MAX_TRANSPOSITION = 24


def _bucket_index(value: str, vocabulary: tuple[str, ...]) -> int:
    """1-based index into `vocabulary`, or 0 for "not in the table" - 0 is reserved so
    "unknown/other" is a real, learnable embedding row, not an out-of-range crash."""
    try:
        return vocabulary.index(value) + 1
    except ValueError:
        return 0


@dataclass(frozen=True)
class ProfileContext:
    """One staff's worth of resolved profile context - what `ProfileContextEmbedding`
    actually consumes, already reduced from a `(ScoreProfile, ScorePart)` pair plus the
    caller's own knowledge of which physical staff of that part this is. Built by the
    caller, not by this module.
    """

    instrument_family: str
    #: 0-based position of this part within its profile.
    part_ordinal: int
    #: 0-based - which physical staff of a multi-staff part (0 for every single-staff
    #: part, which is most of them).
    staff_within_part: int
    expected_staff_count: int
    likely_clefs: tuple[str, ...]
    transposition_semitones: int

    @staticmethod
    def from_score_part(
        part: ScorePart, part_ordinal: int, staff_within_part: int = 0
    ) -> "ProfileContext":
        return ProfileContext(
            instrument_family=part.instrument_family,
            part_ordinal=part_ordinal,
            staff_within_part=staff_within_part,
            expected_staff_count=part.expected_staff_count,
            likely_clefs=part.likely_clefs,
            transposition_semitones=part.transposition_semitones,
        )


class ProfileContextEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.instrument_family_emb = nn.Embedding(len(INSTRUMENT_FAMILIES) + 1, dim)
        self.part_ordinal_emb = nn.Embedding(MAX_PART_ORDINAL + 1, dim)
        self.staff_within_part_emb = nn.Embedding(MAX_STAFF_WITHIN_PART + 1, dim)
        self.staff_count_emb = nn.Embedding(MAX_STAFF_COUNT + 1, dim)
        self.clef_emb = nn.Embedding(len(CLEFS) + 1, dim)
        self.transposition_emb = nn.Embedding(MAX_TRANSPOSITION - MIN_TRANSPOSITION + 1, dim)
        # The explicit "there is no profile context for this staff at all" case - a
        # dedicated, trainable vector rather than a zero one, so the model can
        # eventually distinguish "confirmed no profile" from "gate not yet open."
        self.missing_emb = nn.Parameter(torch.zeros(dim))
        # Zero-initialized per §7.2: the unconditioned path must be bit-identical at
        # initialization, regardless of what the sub-embeddings above compute.
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.instrument_family_emb.weight, std=0.02)
        nn.init.normal_(self.part_ordinal_emb.weight, std=0.02)
        nn.init.normal_(self.staff_within_part_emb.weight, std=0.02)
        nn.init.normal_(self.staff_count_emb.weight, std=0.02)
        nn.init.normal_(self.clef_emb.weight, std=0.02)
        nn.init.normal_(self.transposition_emb.weight, std=0.02)
        nn.init.normal_(self.missing_emb, std=0.02)

    def _clef_set_vector(self, clefs: tuple[str, ...], device: torch.device) -> torch.Tensor:
        """Mean of the set's per-clef embeddings, not a sum - `likely_clefs` is
        unordered and variable-length (a cello's three plausible clefs are not "three
        times as much clef" as a violin's one), so a mean keeps the vector's scale
        comparable regardless of how many clefs a part lists.
        """
        indices = torch.tensor(
            [_bucket_index(clef, CLEFS) for clef in clefs] or [0],
            dtype=torch.long,
            device=device,
        )
        return self.clef_emb(indices).mean(dim=0)

    def embed_one(self, context: "ProfileContext | None", device: torch.device) -> torch.Tensor:
        """The gated context vector for one staff - `None` when nothing is known about
        it, in which case only `missing_emb` (still gated) contributes.
        """
        if context is None:
            return self.gate * self.missing_emb

        family_index = torch.tensor(
            _bucket_index(context.instrument_family, INSTRUMENT_FAMILIES), device=device
        )
        part_ordinal_index = torch.tensor(
            min(max(context.part_ordinal, 0), MAX_PART_ORDINAL), device=device
        )
        staff_within_index = torch.tensor(
            min(max(context.staff_within_part, 0), MAX_STAFF_WITHIN_PART), device=device
        )
        staff_count_index = torch.tensor(
            min(max(context.expected_staff_count, 0), MAX_STAFF_COUNT), device=device
        )
        transposition_index = torch.tensor(
            min(max(context.transposition_semitones, MIN_TRANSPOSITION), MAX_TRANSPOSITION)
            - MIN_TRANSPOSITION,
            device=device,
        )
        total = (
            self.instrument_family_emb(family_index)
            + self.part_ordinal_emb(part_ordinal_index)
            + self.staff_within_part_emb(staff_within_index)
            + self.staff_count_emb(staff_count_index)
            + self._clef_set_vector(context.likely_clefs, device)
            + self.transposition_emb(transposition_index)
        )
        return self.gate * total

    def forward(self, contexts: list["ProfileContext | None"]) -> torch.Tensor:
        """One additive vector per sequence in the batch, shape `(batch, dim)` - meant
        to broadcast-add into `ScoreTransformerWrapper`'s `(batch, seq, dim)` token
        embedding sum at every position, since profile context does not vary within one
        staff's decode.
        """
        device = self.gate.device
        return torch.stack([self.embed_one(context, device) for context in contexts], dim=0)
