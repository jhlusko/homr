"""`base_predictions.py`, scored through the ONNX runtime path instead of PyTorch.

E1 (`OTS_HOMR_RELEASE_SEQUENCE_2026-09-25.md`) compares a freshly trained core against
the released one, and the released core exists only as a pinned ONNX encoder/decoder
pair - no `.pth` for it exists anywhere, laptop included
(`/workspace/context/E1_SCORING_INPUTS.md`). `base_predictions.py`'s own docstring
warns against the trap this exists to avoid: `homr.transformer.staff2score.Staff2Score`
(the ONNX path) ignores `config.filepaths.checkpoint` entirely, so pointing it at two
different checkpoints under that name would score the same graph twice and report two
models as identical. This script takes the encoder/decoder ONNX paths directly instead,
so scoring the released pair and an E1 export can never silently collapse onto one
model.

Writes the identical record shape `base_predictions.py` does, so `domain_gap.py` and
`compare_checkpoints.py` read either one unchanged - a released-vs-E1 comparison and a
PyTorch-checkpoint comparison are the same downstream analysis either way.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

from training.transformer.base_predictions import record_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, required=True, help="image,tokens index.txt")
    parser.add_argument("--out", type=Path, required=True, help="predictions .jsonl")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = every row")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA/CoreML if available.")
    args = parser.parse_args()

    import cv2

    from homr.transformer.configs import Config
    from homr.transformer.staff2score import Staff2Score
    from homr.type_definitions import NDArray  # noqa: F401
    from training.omr_datasets.fingerprint_measures import predict_crop
    from training.transformer.training_vocabulary import read_tokens

    for path in (args.encoder, args.decoder):
        if not path.is_file():
            raise SystemExit(f"no such ONNX file: {path}")

    config = Config()
    config.use_gpu_inference = args.gpu
    config.use_coreml_encoder = False
    # Both attributes, fp16 and not: the encoder/decoder loaders read whichever one
    # their code path picks (fp16 only when `use_gpu_inference`/CoreML succeeds), and
    # the graph's own declared input dtype - not the filename - decides fp16 handling
    # from there (`encoder_inference.Encoder.__init__`'s own comment on this).
    config.filepaths.encoder_path = str(args.encoder)
    config.filepaths.encoder_path_fp16 = str(args.encoder)
    config.filepaths.decoder_path = str(args.decoder)
    config.filepaths.decoder_path_fp16 = str(args.decoder)
    print(f"scoring with encoder={args.encoder} decoder={args.decoder} gpu={args.gpu}", flush=True)
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
