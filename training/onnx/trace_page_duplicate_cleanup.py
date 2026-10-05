"""Capture duplicate-cleanup inputs without changing page inference behavior."""

import argparse
import json
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write
from training.onnx.qualify_joint_pages import process
from training.transformer.qualify_joint_holdout import serialize


def run(*, pages_path, models_path, original, index, arm, out):
    if out.exists():
        raise FileExistsError("Refusing to replace cleanup trace")
    pages = json.loads(pages_path.read_text())
    models = json.loads(models_path.read_text())
    page = pages[index]
    original_identity = json.loads((original / "identity.json").read_text())
    if original_identity["page"] != page or original_identity["models"] != models[arm]:
        raise ValueError("Different page/model identities")
    root = Path(__file__).resolve().parents[2]
    for name, digest in original_identity["code_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Runtime changed since original page: " + name)
    from homr import staff_parsing

    clean = staff_parsing.remove_duplicated_symbols
    calls = []

    def capture(symbols, *args, **kwargs):
        before = [serialize(s) for s in symbols]
        result = clean(symbols, *args, **kwargs)
        calls.append({"before": before, "after": [serialize(s) for s in result]})
        return result

    staff_parsing.remove_duplicated_symbols = capture
    try:
        process(page, models, arm, out)
    finally:
        staff_parsing.remove_duplicated_symbols = clean
        if out.exists():
            write(out / "cleanup-calls.json", calls)
    result = json.loads((out / "result.json").read_text())
    old = json.loads((original / "result.json").read_text())
    exact_xml = result["xml_sha256"] == old["xml_sha256"]
    exact_cached = sha(out / "decoded-staves.json") == sha(original / "decoded-staves.json")
    write(
        out / "trace-receipt.json",
        {
            "tool_sha256": sha(__file__),
            "original_identity_sha256": sha(original / "identity.json"),
            "cleanup_calls_sha256": sha(out / "cleanup-calls.json"),
            "cleanup_source_sha256": sha(root / "homr/transformer/vocabulary.py"),
            "staff_parsing_sha256": sha(root / "homr/staff_parsing.py"),
            "exact_original_xml": exact_xml,
            "exact_original_cache": exact_cached,
            "scope": "Observed cleanup inputs; no policy changes or restored recognition claim",
        },
    )
    if not exact_xml or not exact_cached:
        raise ValueError("Instrumented output differs from original; trace is unqualified")
    print(
        json.dumps(
            {
                "arm": arm,
                "calls": len(calls),
                "exact_xml": exact_xml,
                "exact_cache": exact_cached,
                "seconds": result["seconds"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--page-index", type=int, required=True)
    parser.add_argument("--arm", choices=("accepted", "candidate"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(
        pages_path=args.pages.resolve(),
        models_path=args.models.resolve(),
        original=args.original.resolve(),
        index=args.page_index,
        arm=args.arm,
        out=args.out.resolve(),
    )
