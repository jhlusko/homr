"""Does the published checkpoint load into a model that has the structured heads?

The allowlist in load_checkpoint only tolerates the new head parameters being absent.
If anything else fails to line up this says so now, rather than after the dataset build.
"""
import torch
from homr.transformer.configs import Config
from training.architecture.transformer.tromr_arch import TrOMR
from training.transformer.train_structured_heads import load_pinned, structured_parameters

config = Config()
config.enable_structured_heads = True
model = TrOMR(config)
load_pinned(model, config.filepaths.checkpoint)
trainable = model.freeze_core_for_structured_heads()
params = structured_parameters(model)
frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
print(f"trainable tensors: {len(trainable)}  optimisable: {len(params)}  frozen: {len(frozen)}")
print(f"trainable params: {sum(p.numel() for p in params):,}")
print("heads:", sorted({n.split('.')[2] for n in trainable}))
