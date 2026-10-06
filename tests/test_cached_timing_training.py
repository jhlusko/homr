import argparse
import json

import pytest
import torch

from training.architecture.transformer.exact_timing_heads import load_timing_bundle
from training.architecture.transformer.exact_timing_targets import HEADS
from training.onnx.qualify_joint_export import sha, write
from training.transformer.train_cached_timing_heads import source_balanced_schedule, train


def test_balanced_schedule_excludes_validation_and_is_deterministic():
    rows = [
        dict(source="ossq", split="train"),
        *[dict(source="lieder", split="train") for _ in range(3)],
        dict(source="ossq", split="valid"),
    ]
    schedule = source_balanced_schedule(rows, 0, 1)
    assert schedule == source_balanced_schedule(rows, 0, 1)
    assert len(schedule) == 3
    assert all(
        len(batch) == 2 and {rows[i]["source"] for i in batch} == {"ossq", "lieder"}
        for batch in schedule
    )
    assert all(4 not in batch for batch in schedule)


def fixture(tmp_path):
    manifest = dict(
        schemaVersion="homr.exact-timing-targets.v1", classes={h: ["a", "b"] for h in HEADS}
    )
    dataset = tmp_path / "dataset.json"
    write(dataset, dict(target_manifest=manifest, base_sha256="a" * 64))
    records = []
    generator = torch.Generator().manual_seed(42)
    for source in ("ossq", "lieder"):
        for index, split in enumerate(["train", "train", "train", "valid"]):
            row = dict(id=f"{source}/{index}", source=source, split=split)
            path = tmp_path / (source + str(index) + ".pt")
            torch.save(
                dict(
                    hidden=torch.randn(3, 4, generator=generator),
                    targets={h: torch.tensor([0, 1, 0]) for h in HEADS},
                ),
                path,
            )
            records.append(dict(path=str(path), sha256=sha(path), row=row))
    features = tmp_path / "features.json"
    write(
        features, dict(identity=dict(dataset_sha256=sha(dataset)), records=records, complete=True)
    )
    return (
        argparse.Namespace(
            dataset=dataset,
            features=features,
            out=tmp_path / "continuous",
            threads=1,
            lr=1e-3,
            crops_per_source=1,
            epochs=2,
            allow_partial=False,
            max_updates=0,
        ),
        manifest,
    )


def test_resume_matches_continuous_optimizer_steps_and_allows_epoch_extension(tmp_path):
    args, manifest = fixture(tmp_path)
    train(args)
    continuous = torch.load(args.out / "latest.pth", weights_only=True)
    args.out = tmp_path / "resumed"
    args.max_updates = 1
    train(args)
    interrupted = torch.load(args.out / "latest.pth", weights_only=True)
    assert interrupted["epoch"] == 0 and interrupted["next_batch"] == 1
    assert interrupted["optimizer"]["state"]
    args.max_updates = 0
    train(args)
    resumed = torch.load(args.out / "latest.pth", weights_only=True)
    assert resumed["epoch"] == 2 and resumed["next_batch"] == 0
    for name, value in continuous["bundle"]["heads"].items():
        torch.testing.assert_close(value, resumed["bundle"]["heads"][name], rtol=0, atol=0)
    assert continuous["history"] == resumed["history"]
    with pytest.raises(ValueError, match="cannot be used for inference"):
        load_timing_bundle(resumed["bundle"], manifest, "a" * 64)
    args.epochs = 3
    train(args)
    assert torch.load(args.out / "latest.pth", weights_only=True)["epoch"] == 3
    args.lr = 1e-2
    with pytest.raises(ValueError, match="Resume identity changed"):
        train(args)


def test_changed_cache_tensor_refused(tmp_path):
    args, _ = fixture(tmp_path)
    records = json.loads(args.features.read_text())["records"]
    torch.save(dict(hidden=torch.zeros(3, 4)), records[0]["path"])
    with pytest.raises(ValueError, match="Cached features changed"):
        train(args)
