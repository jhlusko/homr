"""Build a live local dashboard over immutable support and updating probe logs."""

import argparse
import json
import shutil
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "lieder-support", "ossq-support", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads((args.run / "dataset.json").read_text())
    lieder = json.loads(args.lieder_support.read_text())
    ossq = json.loads(args.ossq_support.read_text())
    if lieder["target_manifest_sha256"] != ossq["target_manifest_sha256"]:
        raise ValueError("Support uses different frozen classes")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "data").symlink_to(args.run.resolve(), target_is_directory=True)
    template = Path(__file__).with_suffix(".html")
    shutil.copyfile(template, args.out / "index.html")
    write(
        args.out / "training-support.json",
        dict(
            classes=dataset["target_manifest"]["classes"],
            sources=dict(
                lieder=lieder["groups"]["train/identical"]["heads"],
                ossq=ossq["ossq_train"]["heads"],
            ),
            scope="Original crop support, before source-balanced resampling. Training only.",
        ),
    )
    write(
        args.out / "receipt.json",
        dict(
            run=str(args.run.resolve()),
            dataset_sha256=sha(args.run / "dataset.json"),
            input_sha256={
                str(p): sha(p)
                for p in (args.lieder_support, args.ossq_support, template, Path(__file__))
            },
            limitation="Displays teacher-forced probe metrics; no free-decoding or release claim.",
        ),
    )


if __name__ == "__main__":
    main()
