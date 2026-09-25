"""The ordered labels belonging to each text-detector checkpoint.

Channel count alone cannot identify labels: a six-channel model has five foreground
classes, but it does not say which class occupies channel three. New checkpoints carry
an adjacent ``.classes.json`` file. The two previously pinned ONNX files predate that
contract and have an explicit legacy mapping here.
"""

import json
from pathlib import Path

from homr import text_detector_config


LEGACY_CLASS_ORDER = (
    "Dynamic",
    "Fingering",
    "Expression",
    "Tempo",
    "MeasureNumber",
    "StaffText",
    "Lyrics",
)
DIRECTION_CLASS_ORDER = (
    "Dynamic",
    "Fingering",
    "DirectionText",
    "MeasureNumber",
    "Lyrics",
)


def class_order_path(model_path: str | Path) -> Path:
    return Path(f"{model_path}.classes.json")


def validate_class_order(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("detector class_order must be a nonempty ordered list")
    order = tuple(value)
    if any(
        not isinstance(label, str) or not label or label == "background" for label in order
    ):
        raise ValueError("detector class_order must contain foreground label names only")
    if len(set(order)) != len(order):
        raise ValueError("detector class_order has duplicate labels")
    return order


def read_class_order(
    model_path: str | Path, explicit: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Read the checkpoint's labels, or the explicit mapping for a pinned old ONNX.

    An explicit override is for evaluating an older unaccompanied PyTorch checkpoint.
    It must agree with a sidecar when one exists. Unknown models without either source
    fail closed instead of inheriting whatever training scheme is currently imported.
    """
    sidecar = class_order_path(model_path)
    from_file = None
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        from_file = validate_class_order(payload.get("class_order"))
    if explicit is not None:
        chosen = validate_class_order(explicit)
        if from_file is not None and chosen != from_file:
            raise ValueError(f"class order disagrees with {sidecar}")
        return chosen
    if from_file is not None:
        return from_file
    pinned = {
        Path(text_detector_config.detector_vocal_path).resolve(),
        Path(text_detector_config.detector_instrumental_path).resolve(),
    }
    if Path(model_path).resolve() in pinned:
        return LEGACY_CLASS_ORDER
    raise ValueError(
        f"{model_path}: missing {sidecar.name}; specify the checkpoint's class order"
    )


def write_class_order(model_path: str | Path, class_order: tuple[str, ...]) -> Path:
    """Write the label sidecar beside weights or an exported ONNX graph."""
    order = validate_class_order(class_order)
    destination = class_order_path(model_path)
    destination.write_text(
        json.dumps({"class_order": list(order)}, indent=2) + "\n", encoding="utf-8"
    )
    return destination
