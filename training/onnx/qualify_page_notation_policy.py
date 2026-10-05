"""Matched page diagnostic with explicit corpus-scoped notation policy."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.onnx.qualify_joint_export import setup, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Worker output must be new")
    identity = json.loads((args.baseline / "identity.json").read_text())
    root = Path(__file__).resolve().parents[2]
    for name, digest in identity["code_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Baseline runtime changed: " + name)
    models = json.loads(args.models.read_text())
    if models[identity["arm"]] != identity["models"]:
        raise ValueError("Models differ from matched baseline")
    setup()
    from homr import main as runtime
    from training.onnx import qualify_joint_pages as worker
    from training.transformer.qualify_joint_holdout import serialize

    worker.setup = lambda: None
    ossq = identity["page"]["source"] == "ossq"
    worker.POLICY.update(
        HOMR_STEM_ARBITRATION="ossq-v7" if ossq else "1",
        HOMR_SLUR_SIDE="ossq-rule" if ossq else "0",
    )
    original = runtime.parse_staffs

    def capture(*pos, **kw):
        voices = original(*pos, **kw)
        write(
            args.out / "pre-policy-with-confidence.json",
            [
                [
                    dict(
                        serialize(s),
                        structured_choices=[asdict(c) for c in s.structured_choices or ()],
                    )
                    for s in v
                ]
                for v in voices
            ],
        )
        return voices

    runtime.parse_staffs = capture
    args.out.mkdir(parents=True)
    write(
        args.out / "policy-experiment.json",
        {
            "baseline": str(args.baseline),
            "baseline_identity_sha256": sha(args.baseline / "identity.json"),
            "wrapper_sha256": sha(Path(__file__)),
            "policy": worker.POLICY,
            "candidate_ossq_threshold_is_provisional": ossq and identity["arm"] == "candidate",
            "scope": (
                "v7 threshold is calibrated for accepted v7 heads; applying it to joint heads "
                "is a diagnostic, not promotion"
            ),
        },
    )
    worker.process(identity["page"], models, identity["arm"], args.out)


if __name__ == "__main__":
    main()
