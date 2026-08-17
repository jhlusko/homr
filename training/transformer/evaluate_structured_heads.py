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
from training.architecture.transformer.structured_heads import STEM_HEAD
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
    sink: Callable[..., None] | None = None,
    all_targets: Sequence[str] | None = None,
) -> Evaluation:
    """Run the set and accumulate every measure.

    `head_targets` is what gets scored; `all_targets` is every key the loader may attach.
    They differ whenever the manifest declares fewer heads than the dataset labels - an
    untrained head still has target tensors in the batch - and the difference matters
    because every non-target key is forwarded to the model as a keyword argument. Scoring
    seven heads while the batch carries nine would hand the decoder `slur.slot.1.side` and
    it would refuse it.
    """
    strip = set(all_targets if all_targets is not None else head_targets)
    model.eval()
    evaluation = Evaluation(beam_levels=beam_levels, slur_slots=slur_slots)

    for raw in batches:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw.items()}
        targets = {name: batch[name] for name in head_targets if name in batch}
        if not targets:
            continue
        targets = align_to_decoder_output(targets)
        outputs = model(**{k: v for k, v in batch.items() if k not in strip})
        logits = outputs["structured_logits"]
        if logits is None:
            raise ValueError("model has no structured heads - enable them in the config")

        predicted = decode_predictions(logits, targets, beam_levels, slur_slots)
        reference = decode_reference(targets, beam_levels, slur_slots)
        # How sure the stem head is, per position. An arbiter needs a reason to prefer one
        # source over the other, and the head's own softmax is the only signal it carries.
        stem_confidence = (
            torch.softmax(logits[STEM_HEAD], dim=-1).max(dim=-1).values.tolist()
            if STEM_HEAD in logits
            else None
        )
        # One staff at a time: the sequence-level measures only mean anything within a
        # staff, and pooling first would let a slur opened on one close on another.
        for row, (got, want) in enumerate(zip(predicted, reference, strict=True)):
            evaluation.observe(got, want)
            if sink is not None:
                sink(got, want, stem_confidence[row] if stem_confidence else None)
    return evaluation


def dump_predictions(batches: Any, beam_levels: int, handle: Any) -> Callable[..., None]:
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
        predicted: Sequence[NoteNotation],
        reference: Sequence[NoteNotation],
        stem_confidence: list[float] | None = None,
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
        # Stems are supervised on a different set of notes from beams - a quarter note
        # has a stem and no beam - so they are their own parallel list rather than folded
        # into the beam rows.
        directional = ("up", "down")
        stems = [
            (index, str(note.stem), str(predicted[index].stem))
            for index, note in enumerate(reference)
            if str(note.stem) in directional
        ]
        record = {
            "tokens": entries[position]["tokens"],
            "stem_reference": [actual for _, actual, _ in stems],
            "stem_predicted": [got for _, _, got in stems],
            "stem_confidence": (
                [round(stem_confidence[index], 4) for index, _, _ in stems]
                if stem_confidence
                else []
            ),
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


def load_head_weights(module: Any, state: dict, declared: Sequence[str]) -> None:
    """Load head weights, allowing only heads this run will not score to be absent.

    The architecture grows - a tie head arrived after some weights were saved - so strict
    loading refuses a checkpoint that is merely older. But loading loosely is worse: an
    absent head keeps its random initialisation and still emits an argmax, which is a
    confident prediction from a projection that never saw a gradient.

    The manifest already says which heads a run trained, and those are the only ones that
    will be scored. So a missing weight is fine exactly when the manifest does not declare
    its head, and an error otherwise - the same allowlist rule the core checkpoint uses.
    """
    missing, unexpected = module.load_state_dict(state, strict=False)
    scored = {name.split(".")[0] for name in declared}
    # Parameter names are "<head>.weight"; slur heads live under slur_event/slur_side.
    prefixes = {
        "beam": "beam.level.",
        "stem": "stem.direction",
        "tie": "tie.state",
        "slur_event": "slur.slot.",
        "slur_side": "slur.slot.",
    }
    refused = []
    for key in missing:
        group = key.split(".")[0]
        head_prefix = prefixes.get(group, group)
        if any(name.startswith(head_prefix) for name in declared):
            refused.append(key)
    if refused:
        raise ValueError(
            f"weights are missing heads this run will score: {sorted(refused)}. "
            "The checkpoint and the manifest disagree about what was trained."
        )
    if missing:
        print(f"not in these weights, and not declared, so not scored: {sorted(missing)}")
    if unexpected:
        print(f"in these weights but not in the model: {sorted(unexpected)}")


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
    load_head_weights(model.decoder.structured_heads, heads, names)
    model.to(args.device)

    names = trained_heads(args.manifest, _target_names(config))
    # Unshuffled so the prediction dump can name examples by counting along the index,
    # and in validation mode so the images are not distorted before being scored.
    batches, examples = build_batches(
        args.index, config, args.batch_size, args.workers, shuffle=False, validation=True
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
            all_targets=_target_names(config),
        )
    print(evaluation.describe())
    if args.predictions:
        print(f"predictions: {args.predictions}")
    if args.out:
        args.out.write_text(json.dumps(evaluation.to_dict(), indent=2), encoding="utf-8")
        print(f"report: {args.out}")


if __name__ == "__main__":
    main()
