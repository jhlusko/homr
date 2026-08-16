"""
Score trained notation heads against a held-out split.

This is what makes Gate C answerable. The training run reports a falling loss, which says
nothing about whether the heads beat the rule they have to beat - and 27.12 puts that rule
near four fifths of the corpus's beaming, so a head can look well-trained and still be
worthless.

Two things this refuses to do, both of which would produce a better-looking number than
the model deserves. It scores only positions the targets supervise, so a head is never
credited for a masked position or for agreeing that a quarter note carries no beam. And it
declares only heads the manifest says were trained: an untrained projection still emits an
argmax, and scoring it would report a number for a capability that does not exist.

The beam figures should be read against `training/omr_datasets/beam_baseline.py` on the
same split, not against zero.
"""

# flake8: noqa: T201

import argparse
import json
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from homr.transformer.capability_manifest import CapabilityManifest
from homr.transformer.structured_notation import BeamLevelState, NoteNotation
from training.architecture.transformer.structured_decoding import (
    decode_predictions,
    decode_reference,
)
from training.architecture.transformer.structured_targets import align_to_decoder_output
from training.transformer.structured_metrics import Evaluation
from training.transformer.train_structured_heads import _target_names, build_batches


@torch.no_grad()
def evaluate(
    model: Any,
    batches: Any,
    head_targets: list[str],
    beam_levels: int,
    slur_slots: int,
    device: str = "cpu",
    sink: Callable[[Sequence[NoteNotation], Sequence[NoteNotation]], None] | None = None,
) -> Evaluation:
    """Run the set and accumulate every measure."""
    model.eval()
    evaluation = Evaluation(beam_levels=beam_levels, slur_slots=slur_slots)

    for raw in batches:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw.items()}
        targets = {name: batch[name] for name in head_targets if name in batch}
        if not targets:
            continue
        targets = align_to_decoder_output(targets)
        outputs = model(**{k: v for k, v in batch.items() if k not in head_targets})
        logits = outputs["structured_logits"]
        if logits is None:
            raise ValueError("model has no structured heads - enable them in the config")

        predicted = decode_predictions(logits, targets, beam_levels, slur_slots)
        reference = decode_reference(targets, beam_levels, slur_slots)
        # One staff at a time: the sequence-level measures only mean anything within a
        # staff, and pooling first would let a slur opened on one close on another.
        for got, want in zip(predicted, reference, strict=True):
            evaluation.observe(got, want)
            if sink is not None:
                sink(got, want)
    return evaluation


def dump_predictions(
    batches: Any, beam_levels: int, handle: Any
) -> Callable[[Sequence[NoteNotation], Sequence[NoteNotation]], None]:
    """A sink that writes one JSON record per staff, named by its token file.

    Aggregates cannot answer the question Gate C actually turns on - whether the head is
    right where the *rule* is wrong - because that needs the two lined up note by note.
    This writes the per-staff vectors so a separate pass can join them against the rule
    without the evaluation having to know anything about MusicXML.

    Identity comes from counting along the index, which is only valid because the
    evaluation loader is built unshuffled. Names are written rather than implied so the
    join can be checked instead of assumed.
    """
    entries = batches.dataset.inner.corpus_list
    position = 0

    def write(
        predicted: Sequence[NoteNotation], reference: Sequence[NoteNotation]
    ) -> None:
        nonlocal position
        if position >= len(entries):
            return
        supervised = [
            (index, note)
            for index, note in enumerate(reference)
            if any(
                state != BeamLevelState.NOT_APPLICABLE for state in note.beam_levels[:beam_levels]
            )
        ]
        record = {
            "tokens": entries[position]["tokens"],
            "positions": [index for index, _ in supervised],
            "reference": [
                [str(s) for s in note.beam_levels[:beam_levels]] for _, note in supervised
            ],
            "predicted": [
                [str(s) for s in predicted[index].beam_levels[:beam_levels]]
                for index, _ in supervised
            ],
        }
        handle.write(json.dumps(record) + "\n")
        position += 1

    return write


def trained_heads(manifest_path: Path | None, available: list[str]) -> list[str]:
    """The heads worth scoring: what the manifest declares, intersected with what exists.

    Without a manifest every head is scored, which is right for a fresh run and wrong for
    a checkpoint whose manifest says only some heads were trained - hence the warning
    rather than a silent choice either way.
    """
    if manifest_path is None:
        print("no manifest given: scoring every head, trained or not")
        return available
    manifest = CapabilityManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    declared = [name for name in available if manifest.supports(name)]
    skipped = sorted(set(available) - set(declared))
    if skipped:
        print(f"not declared by the manifest, so not scored: {', '.join(skipped)}")
    return declared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="Dataset index.txt.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned core .pth.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained head weights.")
    parser.add_argument("--manifest", type=Path, help="Capability manifest from the run.")
    parser.add_argument("--out", type=Path, help="Write the report as JSON.")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Write per-staff beam vectors as JSONL, for rule_vs_head.py to join.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.transformer.train_structured_heads import load_pinned

    config = Config()
    config.enable_structured_heads = True

    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    heads = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.decoder.structured_heads.load_state_dict(heads)
    model.to(args.device)

    names = trained_heads(args.manifest, _target_names(config))
    # Unshuffled: the prediction dump names examples by counting along the index.
    batches, examples = build_batches(
        args.index, config, args.batch_size, args.workers, shuffle=False
    )
    print(f"{examples} example(s) from {args.index}")

    with (
        args.predictions.open("w", encoding="utf-8") if args.predictions else nullcontext()
    ) as handle:
        sink = (
            dump_predictions(batches, config.structured_beam_levels, handle)
            if args.predictions
            else None
        )
        evaluation = evaluate(
            model,
            batches,
            names,
            config.structured_beam_levels,
            config.structured_slur_slots,
            args.device,
            sink,
        )
    print(evaluation.describe())
    if args.predictions:
        print(f"predictions: {args.predictions}")
    if args.out:
        args.out.write_text(json.dumps(evaluation.to_dict(), indent=2), encoding="utf-8")
        print(f"report: {args.out}")


if __name__ == "__main__":
    main()
