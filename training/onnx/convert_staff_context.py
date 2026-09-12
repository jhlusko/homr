"""Export the Stage C `StaffContextTransformer` as its own ONNX graph.

The decoder graph already takes `staff_context_emb` as an input (`convert.py`), so the
piece missing at runtime is the module that *produces* that vector: the one that attends
across a system's staves. Exporting it separately is what the two-pass decode needs -
decode every staff once, pool each one's hidden state, run this graph over the system,
then decode again with each staff's context vector added to its input embedding.

**The parity check is the point of this script, not the export.** A graph that loads and
runs proves nothing: the failure this project has actually hit is an export that emits
plausible numbers computed from the wrong thing, and unit tests did not catch it (the
first beam implementation predicted beams correctly and discarded them on the way out).
So the conversion is only reported as successful when the ONNX output matches torch's on
the same inputs, including a padded system, which is where a mask convention gets
silently inverted.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

import numpy as np
import torch

from training.architecture.transformer.staff_context import (
    MAX_STAVES_PER_SYSTEM,
    StaffContextTransformer,
)

#: Tolerance for the torch/ONNX comparison. Tight on purpose: these are the same
#: operations in the same order, so anything above float noise means the graph is not
#: computing what the module computes.
ATOL = 1e-5


#: Below this the module is a no-op: its whole output is multiplied by the gate, which
#: starts at zero and has to be trained off it. The first Stage C run ended holding a
#: checkpoint at 1.1e-5 after a late divergence, which is where this check comes from.
MINIMUM_USEFUL_GATE = 1e-4


def load_module(weights: Path, dim: int) -> StaffContextTransformer:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    prefix = "decoder.staff_context."
    stripped = {
        name[len(prefix) :]: tensor for name, tensor in state.items() if name.startswith(prefix)
    }
    if not stripped:
        raise SystemExit(f"No decoder.staff_context.* tensors in {weights}")
    module = StaffContextTransformer(dim=dim)
    module.load_state_dict(stripped)
    module.eval()
    gate = float(stripped["gate"].item())
    print(f"loaded {len(stripped)} tensors, gate {gate:+.6f}")
    return module


def export(module: StaffContextTransformer, dim: int, path_out: Path) -> None:
    staff_hidden = torch.randn(1, MAX_STAVES_PER_SYSTEM, dim)
    mask = torch.ones(1, MAX_STAVES_PER_SYSTEM, dtype=torch.bool)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (staff_hidden, mask),
        str(path_out),
        input_names=["staff_hidden", "staff_mask"],
        output_names=["staff_context"],
        # Both axes vary: a page's systems differ in staff count, and the caller batches
        # systems together.
        dynamic_axes={
            "staff_hidden": {0: "batch", 1: "staves"},
            "staff_mask": {0: "batch", 1: "staves"},
            "staff_context": {0: "batch", 1: "staves"},
        },
        opset_version=17,
    )
    print(f"exported {path_out} ({path_out.stat().st_size / 1e6:.2f} MB)")


def check_parity(module: StaffContextTransformer, dim: int, path_out: Path) -> bool:
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path_out), providers=["CPUExecutionProvider"])
    torch.manual_seed(0)
    cases = {
        "full 4-staff system": (4, 4),
        "padded to 12, 4 real": (MAX_STAVES_PER_SYSTEM, 4),
        # A single staff exercises the degenerate self-attention path the module's own
        # docstring calls out, and a fully padded row exercises the nan_to_num guard.
        "single staff": (1, 1),
        "fully padded row": (MAX_STAVES_PER_SYSTEM, 0),
    }
    worst = 0.0
    for label, (width, real) in cases.items():
        staff_hidden = torch.randn(1, width, dim)
        mask = torch.zeros(1, width, dtype=torch.bool)
        mask[0, :real] = True
        with torch.no_grad():
            expected = module(staff_hidden, mask).numpy()
        actual = session.run(
            ["staff_context"],
            {"staff_hidden": staff_hidden.numpy(), "staff_mask": mask.numpy()},
        )[0]
        difference = float(np.abs(expected - actual).max())
        worst = max(worst, difference)
        print(f"  {label:<22} max |torch - onnx| = {difference:.3e}")
    print(f"worst difference {worst:.3e} against tolerance {ATOL:.0e}")
    return worst <= ATOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="Trained .pt.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the .onnx.")
    parser.add_argument(
        "--dim", type=int, default=512, help="Decoder dim the module was trained at."
    )
    parser.add_argument(
        "--allow-degenerate-gate",
        action="store_true",
        help="Export even when the gate is ~0 and the parity check is therefore vacuous.",
    )
    args = parser.parse_args()

    module = load_module(args.weights, args.dim)
    gate = abs(float(module.gate.item()))
    if gate < MINIMUM_USEFUL_GATE and not args.allow_degenerate_gate:
        raise SystemExit(
            f"Refusing to export: the gate is {gate:.2e}, so this module contributes\n"
            "nothing, and the parity check below cannot tell you otherwise - it would\n"
            "compare zeros against zeros and pass. Point --weights at an epoch whose\n"
            "ablation delta was positive, or pass --allow-degenerate-gate to export it\n"
            "anyway for inspection."
        )
    export(module, args.dim, args.out)
    if not check_parity(module, args.dim, args.out):
        raise SystemExit("PARITY FAILED - the exported graph does not match the module")
    print("parity OK")
    # Both files or neither: torch puts the weights in a sidecar, and the .onnx alone is
    # 13KB of graph that loads and produces nothing. Copying one file is the obvious
    # mistake, and it fails at inference rather than at load.
    sidecar = args.out.with_suffix(args.out.suffix + ".data")
    if sidecar.exists():
        print(
            f"NOTE: weights are in {sidecar.name} "
            f"({sidecar.stat().st_size / 1e6:.1f} MB).\n"
            f"      Ship it alongside {args.out.name}; the graph alone is not a model."
        )


if __name__ == "__main__":
    main()
