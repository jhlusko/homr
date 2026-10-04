"""Prepare an explicitly untrained timing addon without changing a base checkpoint."""
import argparse
import json
from pathlib import Path

import onnx
import torch

from training.architecture.transformer.exact_timing_heads import initialize_timing_bundle, load_timing_bundle
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--target-manifest', type=Path, required=True)
    parser.add_argument('--feature-onnx', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261004)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'heads-initialized-20261004.pth'
    if path.exists():
        raise ValueError('Refusing to overwrite head bundle')
    graph = onnx.load(str(args.feature_onnx), load_external_data=False)
    if len(graph.graph.input) != 1:
        raise ValueError('Expected one hidden-state input on the notation-head graph')
    dim = graph.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    if dim <= 0:
        raise ValueError('Hidden-state dimension must be explicit')
    manifest = json.loads(args.target_manifest.read_text())
    base_hash = sha(args.base_checkpoint)
    bundle = initialize_timing_bundle(dim, manifest, base_hash, seed=args.seed)
    torch.save(bundle, path)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    heads = load_timing_bundle(restored, manifest, base_hash, allow_untrained=True)
    with torch.no_grad():
        logits = heads(torch.zeros(1, 1, dim))
    try:
        load_timing_bundle(restored, manifest, base_hash)
    except ValueError as error:
        inference_rejection = str(error)
    else:
        raise RuntimeError('Untrained head bundle was accepted for inference')
    if sha(args.base_checkpoint) != base_hash:
        raise ValueError('Base checkpoint changed during addon preparation')
    code_root = Path(__file__).resolve().parents[2]
    files = [str(Path(__file__).resolve().relative_to(code_root)),
        'training/architecture/transformer/exact_timing_heads.py',
        'training/architecture/transformer/exact_timing_targets.py']
    write(args.out/'report.json', dict(base_checkpoint=str(args.base_checkpoint),
        base_checkpoint_sha256=base_hash, feature_onnx_sha256=sha(args.feature_onnx),
        target_manifest_sha256=sha(args.target_manifest), head_bundle_sha256=sha(path),
        hidden_dim=dim, parameters=sum(p.numel() for p in heads.parameters()),
        output_shapes={h: list(x.shape) for h, x in logits.items()},
        training_status='initialized', inference_rejection=inference_rejection,
        code_sha256={f: sha(code_root/f) for f in files},
        limitation='Separate untrained addon; base file unchanged and inference refused. '
            'Hidden dimension read from existing notation ONNX; zero-feature shape smoke only. '
            'No real feature extraction, optimizer resume migration, model training, or recognition gate.'))


if __name__ == '__main__':
    main()
