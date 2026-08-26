"""Per-staff base-model predictions, in the shape `domain_gap.py` already reads.

`domain_gap.py` answers the question that decides what to do about the synthetic-to-scan
gap - is it *spread* (every staff a little worse, so the domain is genuinely harder and
augmentation or more data is the lever) or *concentrated* (most staves fine, a minority
collapsed, so the crops are misaligned and no amount of data will touch it). It answers
it by scoring each staff against its own synthetic twin, which works because the two
tracks write the same token filenames into different directories.

What it cannot currently read is the **base** model's own branches.
`evaluate_structured_heads.py` writes beam, stem, slur and dynamics vectors, because it
exists to evaluate the structured heads; there is no `pitch_reference` anywhere. That
matters now because the per-branch accuracies on `stage2_scans_best` put pitch at 0.857
against 0.914 for the next worst branch and 0.966 for the best - so pitch is where the
base's remaining error actually lives, and pitch is also the branch a misaligned
crop-to-part pairing would damage most while leaving rhythm and position largely intact.

So this writes the same record shape with the base's branches in it, and
`domain_gap.py --field pitch` then works unchanged. Reusing that tool rather than
re-deriving its comparison is deliberate: the spread-vs-concentrated logic and its
reporting are the parts already validated, and the only thing missing was a field.

**On length mismatch.** Head vectors are supervised per position, so reference and
prediction are the same length by construction and `domain_gap` zips them. A base
prediction is free-running and can emit a different number of symbols from the
reference. Zipping would then silently score only the overlap and divide by it, which
*flatters* exactly the failure this is looking for - a wholesale divergence would score
as a short, mostly-correct sequence. Both sides are therefore padded to the longer
length with a sentinel that cannot equal anything, so the accuracy is normalised by
`max(len(reference), len(prediction))` and a length disagreement counts against the
staff, as it should.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

#: Cannot equal any real token, so a padded position is always a miss.
PAD = "\x00missing"

#: The base's own branches, as `EncodedSymbol` attribute names.
BRANCHES = ("pitch", "rhythm", "lift", "articulation", "slur", "position")


def padded(reference: list[str], predicted: list[str]) -> tuple[list[str], list[str]]:
    width = max(len(reference), len(predicted))
    return (
        reference + [PAD] * (width - len(reference)),
        predicted + [PAD + "p"] * (width - len(predicted)),
    )


def branch_values(symbols: list, branch: str) -> list[str]:
    return [str(getattr(s, branch)) for s in symbols if not s.is_control_symbol()]


def record_for(tokens_path: Path, reference: list, predicted: list) -> dict:
    record: dict = {"tokens": str(tokens_path)}
    for branch in BRANCHES:
        want, got = padded(branch_values(reference, branch), branch_values(predicted, branch))
        record[f"{branch}_reference"] = want
        record[f"{branch}_predicted"] = got
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, required=True, help="image,tokens index.txt")
    parser.add_argument("--out", type=Path, required=True, help="predictions .jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = every row")
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Base weights to score with. Overrides the pinned checkpoint rather than "
             "replacing it on disk: the pinned file is what production and every other "
             "run loads, and swapping it to measure one thing would silently change "
             "all of them.",
    )
    args = parser.parse_args()

    import cv2

    from homr.transformer.configs import Config
    from homr.type_definitions import NDArray  # noqa: F401

    # NOT `homr.transformer.staff2score`. There are two classes of this name, and only
    # this one loads `config.filepaths.checkpoint`; the homr-side class is the ONNX
    # inference path and ignores it entirely. Scoring two different checkpoints through
    # that one returns byte-identical numbers for both - which is exactly what happened,
    # and reads as "these models are equivalent" rather than "neither was loaded".
    from training.architecture.transformer.staff2score import Staff2Score
    from training.omr_datasets.fingerprint_measures import predict_crop
    from training.transformer.training_vocabulary import read_tokens

    config = Config()
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise SystemExit(f"no such checkpoint: {args.checkpoint}")
        config.filepaths.checkpoint = str(args.checkpoint)
    print(f"scoring with {config.filepaths.checkpoint}", flush=True)
    model = Staff2Score(config)
    rows = [
        line.strip().split(",")
        for line in args.index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    written = failed = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for image_path, tokens_path in rows:
            try:
                reference = read_tokens(tokens_path)
                predicted = predict_crop(model, Path(image_path))
            except Exception as e:  # noqa: BLE001 - one bad crop must not end the run
                failed += 1
                if failed <= 5:
                    print(f"FAILED {image_path}: {e}", flush=True)
                continue
            handle.write(json.dumps(record_for(Path(tokens_path), reference, predicted)) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"  {written}/{len(rows)}", flush=True)

    print(f"{written:,} staves written, {failed} failed -> {args.out}")
    _ = cv2  # imported for the side effect of failing early if unavailable


if __name__ == "__main__":
    main()
