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


class SystemBatchDataset:
    """One "sample" here is one whole system: every part's own per-staff sample,
    padded to `MAX_STAVES_PER_SYSTEM` and stacked along a new leading "staff"
    dimension, plus a `staff_mask` (`True` = a real staff, `False` = padding) - the
    exact shape `StaffContextTransformer.forward` expects.

    Systems with only one real part are excluded by default (`min_staves=2`):
    `StaffContextTransformer` degenerately self-attends for a single staff and still
    runs correctly (see its own tests), but a module whose entire purpose is
    cross-staff context has nothing to learn from a system with no real siblings -
    excluded here so a caller does not have to filter every batch by hand.
    """

    def __init__(
        self, per_staff: PerStaffDataLoader, groups: list[list[int]], min_staves: int = 2
    ) -> None:
        self.per_staff = per_staff
        self.groups = [g for g in groups if len(g) >= min_staves]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        indices = self.groups[idx]
        samples = [self.per_staff[i] for i in indices]

        # A system with more real parts than the fixed width supports (rare - a
        # large divisi or full-orchestra reduction) keeps the first
        # MAX_STAVES_PER_SYSTEM rather than crash the whole training example over
        # one oversized system.
        samples = samples[:MAX_STAVES_PER_SYSTEM]
        n_real = len(samples)
        n_pad = MAX_STAVES_PER_SYSTEM - n_real

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
    corpus_list: list[str], per_staff: PerStaffDataLoader, min_staves: int = 2
) -> SystemBatchDataset:
    groups = list(group_by_system(corpus_list).values())
    return SystemBatchDataset(per_staff, groups, min_staves=min_staves)
