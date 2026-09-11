"""
§4/§7.4 Stage C: batches training samples by *system* (every part of one system
together), not i.i.d. shuffled single staves - the one thing no mechanism before
Stage C in this document ever needed, since `StaffContextTransformer`'s whole point
is attending across a system's own staves.

Reuses the existing per-sample `DataLoader.__getitem__` (`training/transformer/
data_loader.py`) unchanged for each part in a group - a system batch is just several
of those samples padded and stacked along a new leading "staff" dimension, not a
second, parallel loading path. Every per-sample field is already padded to a fixed
`default_config.max_seq_len` by `to_decoder_branches` (required for the *existing*
i.i.d. per-sample batching to work at all), so stacking across staves needs no
further variable-length handling of its own.
"""

from pathlib import Path
from typing import Any

import torch

from training.architecture.transformer.staff_context import MAX_STAVES_PER_SYSTEM
from training.omr_datasets.score_profile_time_signature import parse_ossq_stem_full
from training.transformer.data_loader import DataLoader as PerStaffDataLoader


def group_by_system(corpus_list: list[str]) -> dict[tuple[str, str, int], list[int]]:
    """Maps `(score_id, page, system_index)` -> the `corpus_list` indices of every
    part belonging to that system. Only OSSQ-stemmed entries can be grouped this way
    (the stem must carry `<score>_<page>_<system>_<part>`) - a non-OSSQ or malformed
    stem is silently excluded from every group rather than force-fit into one, since
    it has no real sibling relationship to group by - the same "unknown is never a
    guess" discipline `time_signature_for_sample`/`system_measure_curve` already use.
    """
    groups: dict[tuple[str, str, int], list[int]] = {}
    for idx, entry in enumerate(corpus_list):
        _, tokens_path = entry.strip().split(",")
        stem = Path(tokens_path).stem
        parsed = parse_ossq_stem_full(stem)
        if parsed is None:
            continue
        score_id, page_str, system_index, _part_index = parsed
        key = (score_id, page_str, system_index)
        groups.setdefault(key, []).append(idx)
    return groups


def part_indices(corpus_list: list[str], indices: list[int]) -> list[int]:
    """The part index of each entry, in the order `indices` gives them."""
    found = []
    for idx in indices:
        parsed = parse_ossq_stem_full(Path(corpus_list[idx].strip().split(",")[1]).stem)
        if parsed is not None:
            found.append(parsed[3])
    return found


def has_complete_parts(corpus_list: list[str], indices: list[int]) -> bool:
    """Whether a system's part numbering runs 0..n-1 with no gap.

    A gap is proof that a staff is missing, because the parts of one system are numbered
    consecutively when the corpus is written. It is proof in one direction only: a quartet
    that lost its *last* part still numbers 0,1,2 and looks whole, so this catches the
    holes it can see rather than claiming to catch all of them.

    Worth catching because the holes are not random. Of the 10 such systems in OSSQ, 7 are
    missing part 0 and two more are missing parts 0-1: the top voices, which are the ones
    that leave the staff and pick up ledger lines. A module built to learn how a system's
    staves relate would be reading those relationships off a system with its melody gone.
    """
    found = sorted(part_indices(corpus_list, indices))
    return found == list(range(len(found)))


class SystemBatchDataset:
    """One "sample" here is one whole system: every part's own per-staff sample,
    padded to `pad_width` and stacked along a new leading "staff" dimension, plus a
    `staff_mask` (`True` = a real staff, `False` = padding) - the exact shape
    `StaffContextTransformer.forward` expects.

    **`pad_width` defaults to `MAX_STAVES_PER_SYSTEM`, but a caller that knows its
    corpus should say so.** Every padded slot is a full-size zero image that still
    costs a forward pass through the frozen encoder, so the padding is pure waste
    proportional to the gap between the bound and the corpus. Measured on OSSQ: 8,621
    of 8,631 systems have exactly 4 staves and none has more, so the default bound of
    12 spends two thirds of every batch encoding zeros - enough to exhaust a 40GB card
    at the default batch size. Note this changes only the padding, never
    `staff_position_emb`, which stays `MAX_STAVES_PER_SYSTEM` wide so a module trained
    on narrow systems still loads and runs for any N <= that bound.

    Systems with only one real part are excluded by default (`min_staves=2`):
    `StaffContextTransformer` degenerately self-attends for a single staff and still
    runs correctly (see its own tests), but a module whose entire purpose is
    cross-staff context has nothing to learn from a system with no real siblings -
    excluded here so a caller does not have to filter every batch by hand.
    """

    def __init__(
        self,
        per_staff: PerStaffDataLoader,
        groups: list[list[int]],
        min_staves: int = 2,
        pad_width: int = MAX_STAVES_PER_SYSTEM,
    ) -> None:
        if not 1 <= pad_width <= MAX_STAVES_PER_SYSTEM:
            raise ValueError(
                f"pad_width must be between 1 and {MAX_STAVES_PER_SYSTEM}, got {pad_width}"
            )
        self.per_staff = per_staff
        self.groups = [g for g in groups if len(g) >= min_staves]
        self.pad_width = pad_width

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        indices = self.groups[idx]
        samples = [self.per_staff[i] for i in indices]

        # A system with more real parts than the fixed width supports (rare - a
        # large divisi or full-orchestra reduction) keeps the first `pad_width`
        # rather than crash the whole training example over one oversized system.
        samples = samples[: self.pad_width]
        n_real = len(samples)
        n_pad = self.pad_width - n_real

        keys = samples[0].keys()
        stacked: dict[str, Any] = {}
        for key in keys:
            real_values = [s[key] for s in samples]
            if n_pad > 0:
                pad_value = torch.zeros_like(torch.as_tensor(real_values[0]))
                real_values = [torch.as_tensor(v) for v in real_values] + [pad_value] * n_pad
            else:
                real_values = [torch.as_tensor(v) for v in real_values]
            stacked[key] = torch.stack(real_values, dim=0)

        stacked["staff_mask"] = torch.tensor([True] * n_real + [False] * n_pad, dtype=torch.bool)
        return stacked


def build_system_batches(
    corpus_list: list[str],
    per_staff: PerStaffDataLoader,
    min_staves: int = 2,
    pad_width: int = MAX_STAVES_PER_SYSTEM,
    unfiltered: list[str] | None = None,
) -> SystemBatchDataset:
    """`unfiltered` is the index as it stood before `_filter_valid_samples` ran.

    Give it whenever you have it. That filter drops *individual staves* - too many
    ledger lines, or a token sequence over `max_seq_len` - and grouping what survives
    silently produces a system with a hole in it, which is a different thing from a
    system that genuinely has fewer parts. Stage C would then learn cross-staff
    context from an incomplete picture and have no way to know.

    Measured on OSSQ, where every system is a string quartet: 10 of 8,631 systems lose
    a staff this way, and 7 of those lose part 1. That bias is the reason to drop them
    rather than tolerate them - the filter removes high parts, because violin I is what
    accumulates ledger lines, so the incomplete systems are systematically missing the
    top voice whose relationship to the others this module exists to learn.
    """
    grouped = {
        key: value
        for key, value in group_by_system(corpus_list).items()
        if has_complete_parts(corpus_list, value)
    }
    if unfiltered is not None:
        expected = {key: len(value) for key, value in group_by_system(unfiltered).items()}
        grouped = {
            key: value
            for key, value in grouped.items()
            if len(value) >= expected.get(key, len(value))
        }
    return SystemBatchDataset(
        per_staff, list(grouped.values()), min_staves=min_staves, pad_width=pad_width
    )


def widest_system(corpus_list: list[str], min_staves: int = 2) -> int:
    """The real staff count to pad to, so a caller need not guess or hard-code it.

    Returns 0 when nothing groups, which a caller should treat as "no systems", not as
    a pad width - the same "unknown is never a guess" rule `group_by_system` follows.
    """
    sizes = [len(g) for g in group_by_system(corpus_list).values() if len(g) >= min_staves]
    return min(max(sizes, default=0), MAX_STAVES_PER_SYSTEM)
