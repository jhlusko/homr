"""Passive research timing observations on an exactly matched accepted page replay."""

import argparse
import json
from pathlib import Path

from training.onnx.qualify_joint_export import setup, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe-models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    identity = json.loads((args.baseline / "identity.json").read_text())
    root = Path(__file__).resolve().parents[2]
    for name, digest in identity["code_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Accepted runtime changed: " + name)
    setup()
    import numpy as np
    import torch

    from homr import music_xml_generator as generator
    from homr.transformer.decoder_inference import ScoreDecoder
    from training.architecture.transformer.exact_timing_heads import load_timing_bundle
    from training.onnx import qualify_joint_pages as worker
    from training.onnx.qualify_free_timing_probe import serialize

    spec = json.loads(args.probe_models.read_text())
    checkpoint = spec["heads"]["epoch6"]
    if sha(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Probe checkpoint changed")
    bundle = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if bundle["trainingStatus"] != "probe-trained":
        raise ValueError("Require research probe")
    head = load_timing_bundle(
        bundle, spec["target_manifest"], spec["core_sha256"], allow_untrained=True
    ).eval()
    head.requires_grad_(False)
    original_attach = ScoreDecoder._attach_structured_notation
    observed = 0

    def attach(self, symbol, hidden):
        nonlocal observed
        original_attach(self, symbol, hidden)
        with torch.inference_mode():
            logits = head(torch.from_numpy(np.asarray(hidden[:, -1:, :], dtype=np.float32)))
        symbol.probe_timing = {
            key: spec["target_manifest"]["classes"][key][int(value.argmax(-1).item())]
            for key, value in logits.items()
        }
        observed += 1

    ScoreDecoder._attach_structured_notation = attach
    original_generate = generator.generate_xml

    def capture(arguments, voices, title):
        write(
            args.out / "final-voices-timing.json",
            [
                [
                    dict(serialize(symbol), timing=getattr(symbol, "probe_timing", None))
                    for symbol in voice
                ]
                for voice in voices
            ],
        )
        return original_generate(arguments, voices, title)

    generator.generate_xml = capture
    worker.setup = lambda: None
    ossq = identity["page"]["source"] == "ossq"
    worker.POLICY.update(
        HOMR_STEM_ARBITRATION="ossq-v7" if ossq else "1",
        HOMR_SLUR_SIDE="ossq-rule" if ossq else "0",
    )
    args.out.mkdir(parents=True)
    worker.process(
        identity["page"], {identity["arm"]: identity["models"]}, identity["arm"], args.out
    )
    matches = sha(args.out / "page.musicxml") == sha(args.baseline / "page.musicxml")
    voices = json.loads((args.out / "final-voices-timing.json").read_text())
    notes = [s for voice in voices for s in voice if s["rhythm"].startswith(("note", "rest"))]
    write(
        args.out / "probe-receipt.json",
        dict(
            accepted_xml_exact_match=matches,
            observed_decoder_symbols=observed,
            final_note_rest_symbols=len(notes),
            final_note_rest_with_timing=sum(s["timing"] is not None for s in notes),
            probe_checkpoint=checkpoint,
            baseline=str(args.baseline),
            source_sha256=sha(Path(__file__)),
            reference_opened=False,
            scope="Passive complete-page observation; no candidate policy or accuracy qualification yet",
        ),
    )
    if not matches:
        raise ValueError("Accepted XML did not reproduce exactly")


if __name__ == "__main__":
    main()
