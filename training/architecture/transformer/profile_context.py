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

import random
from dataclasses import dataclass, replace

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
#: `likely_clefs` is a variable-length set; a fixed-shape tensor needs a cap - 3 covers
#: essentially every real case (a cello's three plausible clefs, F4/C4/G2, is exactly
#: the widest this design's own examples ever use). Extra clefs beyond the cap are
#: dropped, not an error - the same "priors, not constraints" spirit the rest of this
#: schema uses.
MAX_CLEF_SLOTS = 3


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


#: §7.4's starting hypothesis (not fixed): 30% no profile, 30% partially masked, 40%
#: complete.
DEFAULT_NO_PROFILE_PROB = 0.3
DEFAULT_PARTIAL_MASK_PROB = 0.3


def apply_context_dropout(
    context: "ProfileContext | None",
    rng: random.Random,
    no_profile_prob: float = DEFAULT_NO_PROFILE_PROB,
    partial_mask_prob: float = DEFAULT_PARTIAL_MASK_PROB,
) -> "ProfileContext | None":
    """§7.4's training-time context dropout - so the model does not become dependent on
    profile context always being present, since most real callers will not supply one.
    A single roll chooses between three outcomes, in that order: drop to no profile at
    all, partially mask, or leave the real context untouched. Nothing to drop from an
    already-missing context - a `None` sample is not "dropped to `None`," it already
    carries no signal.

    **"Partially masked" is scoped to `instrument_family` specifically, not every field
    independently** - a real, named simplification, not an oversight. That field is the
    one genuinely optional piece of context a real caller is most likely to omit while
    still knowing structural facts (staff count, position within the system) from
    layout alone, and it is the only field with a pre-existing "unknown" sentinel (an
    empty string) that does not also double as a legitimate real value the way
    `part_ordinal == 0` or `transposition_semitones == 0` both do. Masking the integer
    fields independently would need those fields to gain their own explicit "unknown"
    states first (a third bucket beyond "real value" and "out-of-vocabulary," which
    `ProfileContextEmbedding` does not currently have for anything but
    `instrument_family` and `likely_clefs`) - a real design question, not attempted
    here.
    """
    if context is None:
        return None
    roll = rng.random()
    if roll < no_profile_prob:
        return None
    if roll < no_profile_prob + partial_mask_prob:
        return replace(context, instrument_family="")
    return context


def context_to_batch_fields(context: "ProfileContext | None") -> dict[str, "int | torch.Tensor"]:
    """Plain, default-collatable representation of one sample's profile context - small
    ints and one fixed-length tensor, no dataclass or `None` - for `DataLoader.
    __getitem__` to emit directly into a batch dict that HuggingFace `Trainer`'s default
    collator (`train.py` has no custom `data_collator`) can stack without special-
    casing.

    `profile_clef_indices` is a `torch.Tensor`, not a plain Python list, and this is
    load-bearing, not stylistic: PyTorch's default collate treats a per-sample Python
    `list` as a sequence to recurse into (zipping position-by-position across the
    batch into several `(batch,)` tensors), not as one fixed-size unit to stack - so a
    plain list here would silently arrive at `forward_from_batch` as a `list` of
    tensors instead of one `(batch, MAX_CLEF_SLOTS)` tensor and crash `nn.Embedding`
    ("must be Tensor, not list"). A per-sample tensor is what collates into the single
    stacked tensor `forward_from_batch` actually expects - caught by running the real
    training script end to end, not by any of this module's own unit tests, since none
    of them exercise PyTorch's default collate function at all.

    `profile_clef_count` alongside the padded `profile_clef_indices` is what lets
    `ProfileContextEmbedding.forward_from_batch` mask out the padding before averaging -
    without it, a padded slot (index 0, the same "unknown/other" bucket a real
    unrecognised clef would also land on) would dilute the mean by a fake "unknown clef"
    that was never actually in `likely_clefs`, which `embed_one`'s own unpadded mean
    does not do. Both entry points must agree on one context's meaning; see
    `test_profile_context.py`'s `TestBatchAndListAgree` for the property this exists to
    hold.
    """
    if context is None:
        return {
            "profile_present": 0,
            "profile_family_index": 0,
            "profile_part_ordinal_index": 0,
            "profile_staff_within_part_index": 0,
            "profile_staff_count_index": 0,
            "profile_clef_indices": torch.zeros(MAX_CLEF_SLOTS, dtype=torch.long),
            "profile_clef_count": 0,
            "profile_transposition_index": 0,
        }
    clefs = context.likely_clefs[:MAX_CLEF_SLOTS]
    clef_indices = [_bucket_index(clef, CLEFS) for clef in clefs]
    clef_indices += [0] * (MAX_CLEF_SLOTS - len(clef_indices))
    return {
        "profile_present": 1,
        "profile_family_index": _bucket_index(context.instrument_family, INSTRUMENT_FAMILIES),
        "profile_part_ordinal_index": min(max(context.part_ordinal, 0), MAX_PART_ORDINAL),
        "profile_staff_within_part_index": min(
            max(context.staff_within_part, 0), MAX_STAFF_WITHIN_PART
        ),
        "profile_staff_count_index": min(max(context.expected_staff_count, 0), MAX_STAFF_COUNT),
        "profile_clef_indices": torch.tensor(clef_indices, dtype=torch.long),
        "profile_clef_count": len(clefs),
        "profile_transposition_index": min(
            max(context.transposition_semitones, MIN_TRANSPOSITION), MAX_TRANSPOSITION
        )
        - MIN_TRANSPOSITION,
    }


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

    def forward_from_batch(
        self,
        profile_present: torch.Tensor,
        profile_family_index: torch.Tensor,
        profile_part_ordinal_index: torch.Tensor,
        profile_staff_within_part_index: torch.Tensor,
        profile_staff_count_index: torch.Tensor,
        profile_clef_indices: torch.Tensor,
        profile_clef_count: torch.Tensor,
        profile_transposition_index: torch.Tensor,
    ) -> torch.Tensor:
        """The training-facing entry point: every argument is a batched tensor exactly
        as `context_to_batch_fields` produces per-sample and the default collator
        stacks - `(batch,)` for every field except `profile_clef_indices`, which is
        `(batch, MAX_CLEF_SLOTS)`. Fully vectorized (no Python loop over the batch),
        unlike `forward`/`embed_one`, which exist for a single direct caller (live
        inference, a unit test) rather than a training batch.

        Must agree with `forward`/`embed_one` for the same logical context - see
        `test_profile_context.py`'s `TestBatchAndListAgree`. The one place that needs
        care to keep that true: a padded clef slot holds index 0, the same "unknown/
        other" bucket a real unrecognised clef would also land on, so it is masked out
        of the mean using `profile_clef_count` - except when a sample states no clefs
        at all (`profile_clef_count == 0`), where every slot is padding and the
        unmasked mean over three identical "unknown" rows already equals what
        `embed_one`'s own `likely_clefs == ()` fallback computes, so no special case is
        needed there beyond leaving the mask as all-ones.
        """
        device = self.gate.device
        clef_vectors = self.clef_emb(profile_clef_indices)  # (batch, slots, dim)
        slot_positions = torch.arange(MAX_CLEF_SLOTS, device=device).unsqueeze(0)  # (1, slots)
        real_slot_mask = (slot_positions < profile_clef_count.unsqueeze(1)).float()
        mask = torch.where(
            profile_clef_count.unsqueeze(1) > 0, real_slot_mask, torch.ones_like(real_slot_mask)
        )
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        clef_mean = (clef_vectors * mask.unsqueeze(-1)).sum(dim=1) / denom

        total = (
            self.instrument_family_emb(profile_family_index)
            + self.part_ordinal_emb(profile_part_ordinal_index)
            + self.staff_within_part_emb(profile_staff_within_part_index)
            + self.staff_count_emb(profile_staff_count_index)
            + clef_mean
            + self.transposition_emb(profile_transposition_index)
        )
        present = profile_present.to(total.dtype).unsqueeze(-1)  # (batch, 1)
        combined = present * total + (1 - present) * self.missing_emb.unsqueeze(0)
        return self.gate * combined
