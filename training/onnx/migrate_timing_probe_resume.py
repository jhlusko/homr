"""Explicitly import an interrupted timing probe after a hash-bound trainer fix.

Only the trainer-code identity may change. Weights, AdamW state, epoch, next
batch, running losses and scientific configuration are retained byte-for-byte
as tensor values. Never edits the original interrupted run.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from training.onnx.qualify_joint_export import sha, write
from training.transformer.train_cached_timing_heads import atomic_save


def migrate(parent, old_source, out, reason):
    state = torch.load(parent / "latest.pth", map_location="cpu", weights_only=True)
    previous = state["identity"]
    if sha(old_source) != previous["code_sha256"]:
        raise ValueError("Old trainer source does not match checkpoint identity")
    if json.loads((parent / "identity.json").read_text()) != previous:
        raise ValueError("Parent identity differs from checkpoint")
    current_source = Path(__file__).parents[1] / "transformer/train_cached_timing_heads.py"
    replacement = {**previous, "code_sha256": sha(current_source)}
    if replacement == previous or not reason.strip():
        raise ValueError("Require an actual code change and explicit reason")
    out.mkdir(parents=True, exist_ok=False)
    state["identity"] = replacement
    write(out / "identity.json", replacement)
    atomic_save(out / "latest.pth", state)
    copied = {}
    for name in ("initial-validation.json", "history.json"):
        if (parent / name).exists():
            shutil.copyfile(parent / name, out / name)
            copied[name] = sha(parent / name)
    write(
        out / "resume-migration.json",
        dict(
            parent=str(parent.resolve()),
            parent_checkpoint_sha256=sha(parent / "latest.pth"),
            old_source=str(old_source.resolve()),
            old_source_sha256=sha(old_source),
            current_source_sha256=sha(current_source),
            reason=reason,
            changed_identity_fields=["code_sha256"],
            copied_files_sha256=copied,
            preserved_epoch=state["epoch"],
            preserved_next_batch=state["next_batch"],
            code_sha256=sha(Path(__file__)),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    migrate(args.parent, args.old_source, args.out, args.reason)


if __name__ == "__main__":
    main()
