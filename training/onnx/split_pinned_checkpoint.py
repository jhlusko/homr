"""Split a full `TrOMR` checkpoint into the two files `convert.py` expects.

`convert_encoder`/`convert_decoder` in `training/onnx/convert.py` each load their weights
from a hardcoded relative filename (`encoder_weights.pt`, `decoder_weights.pt`) in the
current directory - upstream's own convention, and untouched here since both functions
are also how the pinned upstream checkpoint gets exported. Our trained checkpoints
(`scans_clef_best.pth`) are a single combined `TrOMR` state dict instead, keyed
`encoder.*` and `decoder.net.*`. This is the missing step between the two: split one file
into the pair the exporters already know how to consume.

`decoder.net.*` rather than `decoder.*`: `convert_decoder` builds and loads only
`get_score_wrapper(config)`, the inner `ScoreTransformerWrapper` - not the outer
`ScoreDecoder`, which is what `TrOMR.decoder` actually is. The one key outside that prefix
(`decoder.note_mask`) is a registered buffer `ScoreDecoder` derives from config rather
than a learned tensor, so it has nothing to load into `ScoreTransformerWrapper` and no
value to lose by not carrying it over.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from homr.simple_logging import eprint

ENCODER_PREFIX = "encoder."
DECODER_PREFIX = "decoder.net."


def split_checkpoint(state: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """A full `TrOMR` state dict, as the two sub-state-dicts its parts expect.

    Keys outside both prefixes (or under `decoder.` but not `decoder.net.`) are silently
    dropped: they are the outer `ScoreDecoder`'s own attributes, not tensors either
    exported module can load.
    """
    encoder = {
        key[len(ENCODER_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(ENCODER_PREFIX)
    }
    decoder = {
        key[len(DECODER_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(DECODER_PREFIX)
    }
    return encoder, decoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="A full TrOMR .pth.")
    parser.add_argument(
        "--out", type=Path, default=Path("."),
        help="Directory to write encoder_weights.pt/decoder_weights.pt into "
        "(convert_encoder/convert_decoder expect them in the current directory).",
    )
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    encoder, decoder = split_checkpoint(state)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(encoder, args.out / "encoder_weights.pt")
    torch.save(decoder, args.out / "decoder_weights.pt")
    eprint(
        f"wrote {len(encoder)} encoder tensor(s) and {len(decoder)} decoder tensor(s) "
        f"from {len(state)} total"
    )


if __name__ == "__main__":
    main()
