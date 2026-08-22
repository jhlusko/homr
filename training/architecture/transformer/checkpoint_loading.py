"""
Loading a pretrained checkpoint into a model that has grown new heads.

`strict=False` is not enough on its own, and this is the module that says why. It accepts
every missing and unexpected key silently, so it cannot tell "the new beam heads are
absent, as expected, and will be trained" from "the decoder weights did not load because
a layer was renamed". The second failure produces a model that runs, trains, and is
quietly worse than the checkpoint it was supposed to start from.

So loading takes an allowlist. Parameters the new architecture adds may be missing from
an older checkpoint; anything else missing, and anything in the checkpoint the model has
no place for, is an error.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class LoadReport:
    """What loading actually did, so a run can record it rather than assume it."""

    #: Allowlisted parameters absent from the checkpoint - the new heads, on a first run.
    initialized: tuple[str, ...]
    loaded: int

    def describe(self) -> str:
        if not self.initialized:
            return f"loaded {self.loaded} parameters, nothing left to initialize"
        prefixes = sorted(
            {
                name.split(".")[0] + "." + name.split(".")[1]
                for name in self.initialized
                if "." in name
            }
        )
        return (
            f"loaded {self.loaded} parameters; initialized {len(self.initialized)} new ones"
            f" under {', '.join(prefixes) or 'the model root'}"
        )


class CheckpointMismatch(RuntimeError):
    pass


def load_checkpoint(
    model: nn.Module,
    state: dict,
    expected_new_prefixes: Iterable[str] = (),
) -> LoadReport:
    """Load `state` into `model`, allowing only the expected new parameters to be absent.

    expected_new_prefixes names the parameter subtrees the new architecture adds, e.g.
    "decoder.structured_heads". A missing parameter under one of those is initialized and
    reported; a missing parameter anywhere else, or a checkpoint key the model has no
    place for, raises.
    """
    prefixes = tuple(expected_new_prefixes)
    incompatible = model.load_state_dict(state, strict=False)

    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise CheckpointMismatch(
            f"checkpoint has {len(unexpected)} parameter(s) the model has no place for, "
            f"e.g. {unexpected[:3]} - the architecture and the checkpoint disagree"
        )

    missing = list(incompatible.missing_keys)
    unexplained = [
        name for name in missing if not any(name.startswith(prefix) for prefix in prefixes)
    ]
    if unexplained:
        raise CheckpointMismatch(
            f"{len(unexplained)} parameter(s) missing from the checkpoint that are not new "
            f"heads, e.g. {unexplained[:3]} - expected new parameters under {prefixes or '()'}"
        )

    return LoadReport(initialized=tuple(missing), loaded=len(state))
