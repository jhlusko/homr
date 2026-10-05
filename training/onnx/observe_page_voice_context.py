"""Observe actual page decoding and pre-cleanup voice context without applying a new policy."""

import argparse
import json
from pathlib import Path

from PIL import Image

from training.onnx.qualify_joint_export import sha, write
from training.onnx.qualify_joint_pages import process
from training.transformer.qualify_joint_holdout import serialize


def observe(image, models_path, out, max_pixels=10_000_000):
    if out.exists():
        raise FileExistsError("Refusing to replace page observation")
    with Image.open(image) as pixels:
        dimensions = list(pixels.size)
    if dimensions[0] * dimensions[1] > max_pixels:
        raise ValueError("Original page exceeds chosen diagnostic pixel budget")
    models = json.loads(models_path.read_text())
    from homr import staff_parsing

    original = {
        name: getattr(staff_parsing, name)
        for name in (
            "parse_staff_image",
            "parse_staff_image_greedy_with_margins",
            "remove_duplicated_symbols",
            "rerank_staff_candidates",
        )
    }
    staff_calls, cleanup_calls, rerank_calls = [], [], []

    def capture_staff(name):
        def wrapped(*args, **kwargs):
            result = original[name](*args, **kwargs)
            symbols = result if name == "parse_staff_image" else result[0]
            staff = args[2] if len(args) > 2 else kwargs["staff"]
            call_id = len(staff_calls)
            for symbol in symbols:
                symbol.audit_staff_call = call_id
            staff_calls.append(
                {
                    "call": call_id,
                    "decoder_method": name,
                    "bounds_in_runtime_image": [
                        float(staff.min_x),
                        float(staff.min_y),
                        float(staff.max_x),
                        float(staff.max_y),
                    ],
                    "grandstaff": bool(staff.is_grandstaff),
                    "symbols": [serialize(symbol) for symbol in symbols],
                }
            )
            return result

        return wrapped

    def capture_cleanup(symbols, *args, **kwargs):
        before = [serialize(s) for s in symbols]
        result = original["remove_duplicated_symbols"](symbols, *args, **kwargs)
        cleanup_calls.append(
            {
                "call": len(cleanup_calls),
                "before": before,
                "after": [serialize(s) for s in result],
                "staff_origins_before": [getattr(s, "audit_staff_call", None) for s in symbols],
            }
        )
        return result

    def capture_rerank(candidates):
        result = original["rerank_staff_candidates"](candidates)
        rerank_calls.append(
            {
                "candidates": {
                    str(k): [[serialize(s) for s in candidate] for candidate in values]
                    for k, values in candidates.items()
                },
                "selected": {str(k): [serialize(s) for s in v] for k, v in result.items()},
            }
        )
        return result

    staff_parsing.rerank_staff_candidates = capture_rerank
    staff_parsing.parse_staff_image = capture_staff("parse_staff_image")
    staff_parsing.parse_staff_image_greedy_with_margins = capture_staff(
        "parse_staff_image_greedy_with_margins"
    )
    staff_parsing.remove_duplicated_symbols = capture_cleanup
    page = {
        "source": "ossq",
        "image": str(image),
        "image_sha256": sha(image),
        "dimensions": dimensions,
        "pixel_budget": max_pixels,
        "scope": (
            "Observed actual production-code-path context; no native page accuracy certificate"
        ),
    }
    try:
        process(page, models, "accepted", out)
    finally:
        for name, function in original.items():
            setattr(staff_parsing, name, function)
        if out.exists():
            write(out / "staff-calls.json", staff_calls)
            write(out / "cleanup-calls.json", cleanup_calls)
            write(out / "rerank-calls.json", rerank_calls)
    result = json.loads((out / "result.json").read_text())
    if result["status"] != "ok":
        raise RuntimeError("Page observation failed; inspect result.json")
    root = Path(__file__).resolve().parents[2]
    write(
        out / "observation-receipt.json",
        {
            "page": page,
            "tool_sha256": sha(__file__),
            "staff_calls": len(staff_calls),
            "voice_cleanup_calls": len(cleanup_calls),
            "staff_calls_sha256": sha(out / "staff-calls.json"),
            "cleanup_calls_sha256": sha(out / "cleanup-calls.json"),
            "source_sha256": {
                name: sha(root / name)
                for name in [
                    "homr/staff_parsing.py",
                    "homr/transformer/staff2score.py",
                    "homr/transformer/vocabulary.py",
                    "training/onnx/qualify_joint_pages.py",
                ]
            },
            "scope": (
                "Observational single accepted-stack run. No uninstrumented-repeat parity claim, "
                "no reference injected, no guard activated, no full-page accuracy claim."
            ),
        },
    )
    print(
        json.dumps({"staff_calls": len(staff_calls), "voice_cleanup_calls": len(cleanup_calls)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "models", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=10_000_000)
    args = parser.parse_args()
    observe(args.image.resolve(), args.models.resolve(), args.out.resolve(), args.max_pixels)
