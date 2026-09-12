"""Does Stage C's loss improvement show up as token accuracy?

Every Stage C number so far is a **loss** delta: +0.0184 on the validation ablation, 1.66%
of the without-context loss. That is the quantity the training objective optimises, and it
is not the quantity anyone cares about. A model can lower cross-entropy by becoming better
calibrated on tokens it already gets right, and this is a *second decode pass per staff*
against a 280-second hard timeout, so what it buys has to be stated in transcription terms
before it can be weighed.

This reports **teacher-forced token accuracy**, per branch, with and without the staff
context vector, over the same held-out systems and the same first-pass hidden state - the
identical ablation the loss figure comes from, counting argmax hits instead of summing
cross-entropy.

**Teacher-forced is not what production does, and the gap matters.** Production decodes
free-running, so an early mistake shifts everything after it; teacher forcing hands the
model the correct prefix at every step and measures one decision at a time. That makes
this an *upper bound* on the benefit and, more importantly, a different statistic - a
cross-staff context vector is most plausibly useful exactly where free-running decoding
has started to drift, which is the case teacher forcing removes. Treat a positive result
here as necessary and not sufficient, and a null result as decisive against.

The logits already come back from the decoder, and the targets are the inputs shifted by
one and masked the same way the loss masks them, so nothing in the model or the training
path is touched to obtain this.
"""

# flake8: noqa: T201

import argparse
import statistics
from pathlib import Path

import torch

#: The order `ScoreDecoder` packs its logits in.
BRANCHES = ("rhythm", "pitch", "lift", "position", "articulation", "slur")

#: What the decoder masks padded targets with.
IGNORE_INDEX = -100


def _targets(flat: dict, device: str) -> dict[str, torch.Tensor]:
    """The same targets the loss uses: inputs shifted by one, padding masked out.

    `flat` is the *flattened* batch - `(batch * staves, sequence)`. The unflattened batch
    is `(batch, staves, sequence)`, where slicing `[:, 1:]` drops a staff rather than
    shifting the sequence, which produces tensors that differ by one in the wrong
    dimension and an error far from its cause.
    """
    mask = flat["mask"].to(device)
    sequence_length = flat["rhythms"].shape[1]
    if mask.shape[1] == sequence_length:
        mask = mask[:, :-1]
    mask = mask.bool()
    out = {}
    for name, key in zip(
        BRANCHES, ("rhythms", "pitchs", "lifts", "positions", "articulations", "slurs"), strict=True
    ):
        target = flat[key].to(device)[:, 1:]
        out[name] = target.masked_fill(~mask, IGNORE_INDEX)
    return out


def _hits(logits: tuple, targets: dict[str, torch.Tensor]) -> dict[str, tuple[int, int]]:
    counts = {}
    for branch, branch_logits in zip(BRANCHES, logits, strict=True):
        target = targets[branch]
        predicted = branch_logits.argmax(dim=-1)
        scored = target != IGNORE_INDEX
        counts[branch] = (
            int((predicted.eq(target) & scored).sum().item()),
            int(scored.sum().item()),
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.transformer.train_staff_context import (
        build_batches,
        flatten_staff_dim,
        load_pinned,
        two_pass_forward,
    )

    config = Config()
    config.enable_staff_context = True
    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"weights do not fit this model: {sorted(unexpected)[:3]}")
    gate = float(state["decoder.staff_context.gate"].item())
    print(f"loaded {len(state)} Stage C tensors, gate {gate:+.6f}")
    model.to(args.device)
    model.eval()

    batches, count = build_batches(
        args.index, config, args.batch_size, args.workers, shuffle=False, validation=True
    )
    print(f"{count} validation system(s)\n")

    totals = {arm: {b: [0, 0] for b in BRANCHES} for arm in ("with", "without")}
    with torch.no_grad():
        for raw in batches:
            outputs = two_pass_forward(model, raw, args.device, sampling_prob=1.0)
            flat, _, _ = flatten_staff_dim({k: v.to(args.device) for k, v in raw.items()})
            targets = _targets(flat, args.device)
            for arm, key in (("without", "first_pass"), ("with", "second_pass")):
                for branch, (hit, total) in _hits(outputs[key]["logits"], targets).items():
                    totals[arm][branch][0] += hit
                    totals[arm][branch][1] += total

    print(f"{'branch':<14} {'without':>9} {'with':>9} {'delta':>9}   {'positions':>10}")
    deltas = []
    for branch in BRANCHES:
        hit_without, total = totals["without"][branch]
        hit_with, _ = totals["with"][branch]
        if not total:
            continue
        without = hit_without / total
        with_context = hit_with / total
        deltas.append(with_context - without)
        print(
            f"{branch:<14} {without:>8.2%} {with_context:>9.2%} "
            f"{with_context - without:>+9.2%}   {total:>10,}"
        )

    pooled_total = sum(totals["without"][b][1] for b in BRANCHES)
    pooled_without = sum(totals["without"][b][0] for b in BRANCHES) / max(pooled_total, 1)
    pooled_with = sum(totals["with"][b][0] for b in BRANCHES) / max(pooled_total, 1)
    print(
        f"\n{'pooled':<14} {pooled_without:>8.2%} {pooled_with:>9.2%} "
        f"{pooled_with - pooled_without:>+9.2%}   {pooled_total:>10,}"
    )
    print(f"{'macro':<14} {'':>8} {'':>9} {statistics.mean(deltas):>+9.2%}")
    print(
        "\nTeacher-forced: the model is handed the correct prefix at every step, so this\n"
        "is an upper bound on the benefit and not what free-running decoding would show."
    )


if __name__ == "__main__":
    main()
