"""Experimental additive timing projections and explicit, separate checkpoint bundles.

Not wired into model forward, ONNX, runtime arbitration or current training scripts.
Core and existing notation checkpoints retain their state keys and dimensions.
"""
import hashlib
import json
import math

import torch
from torch import nn

from training.architecture.transformer.exact_timing_targets import HEADS
from training.architecture.transformer.structured_losses import structured_loss


def contract_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ExactTimingHeads(nn.Module):
    def __init__(self, dim, manifest):
        super().__init__()
        if dim <= 0 or manifest['schemaVersion'] != 'homr.exact-timing-targets.v1':
            raise ValueError('Unsupported timing target contract/dimension')
        classes = manifest['classes']
        if set(classes) != set(HEADS) or any(not classes[h] or len(set(classes[h])) != len(classes[h]) for h in HEADS):
            raise ValueError('Invalid timing class lists')
        self.projections = nn.ModuleDict({h.replace('.', '_'): nn.Linear(dim, len(classes[h])) for h in HEADS})

    def forward(self, hidden):
        return {h: self.projections[h.replace('.', '_')](hidden) for h in HEADS}


def timing_loss(logits, targets, *, weights):
    # Explicit weights are required; adding these heads must not silently
    # change the old recognition + notation/12 joint objective.
    if set(weights) != set(HEADS) or any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Provide four explicit finite, nonnegative timing loss weights')
    return structured_loss(logits, targets, weights=weights)


def initialize_timing_bundle(dim, manifest, base_checkpoint_sha256, *, seed):
    if len(base_checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in base_checkpoint_sha256):
        raise ValueError('Requires the exact base checkpoint SHA-256')
    # Initializing additive heads must not perturb an existing run's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        heads = ExactTimingHeads(dim, manifest)
    return dict(schemaVersion='homr.exact-timing-checkpoint.v1', hiddenDim=dim,
        baseCheckpointSha256=base_checkpoint_sha256, targetContractSha256=contract_digest(manifest),
        trainingStatus='initialized', seed=seed, heads=heads.state_dict())


def load_timing_bundle(bundle, manifest, base_checkpoint_sha256, *, allow_untrained=False):
    if bundle['schemaVersion'] != 'homr.exact-timing-checkpoint.v1':
        raise ValueError('Unsupported timing checkpoint schema')
    if bundle['baseCheckpointSha256'] != base_checkpoint_sha256:
        raise ValueError('Timing checkpoint base hash mismatch')
    if bundle['targetContractSha256'] != contract_digest(manifest):
        raise ValueError('Timing target class order/contract mismatch')
    if bundle['trainingStatus'] != 'trained' and not allow_untrained:
        raise ValueError('Untrained timing heads cannot be used for inference')
    with torch.random.fork_rng(devices=[]):
        heads = ExactTimingHeads(bundle['hiddenDim'], manifest)
    heads.load_state_dict(bundle['heads'], strict=True)
    return heads
