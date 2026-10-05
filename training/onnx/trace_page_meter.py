"""Record meter provenance without changing a page's decoding or repairs."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.onnx.qualify_joint_export import setup, sha, write


def meters(symbols):
    return [
        {"index": i, "rhythm": s.rhythm}
        for i, s in enumerate(symbols)
        if s.rhythm.startswith("timeSignature")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    identity = json.loads((args.baseline / "identity.json").read_text())
    if args.out.exists():
        raise ValueError("Trace output must be new")
    root = Path(__file__).resolve().parents[2]
    for name, digest in identity["code_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Baseline runtime changed: " + name)
    setup()
    from homr import staff_parsing as parsing
    from training.onnx import qualify_joint_pages as page_runner

    # setup() already configured PyTorch inter-op threads in this process.
    page_runner.setup = lambda: None

    events = []
    greedy = parsing.parse_staff_image_greedy_with_margins
    plain = parsing.parse_staff_image
    repairs = parsing._apply_cross_staff_repairs
    deduplicate = parsing.remove_duplicated_symbols

    def trace_greedy(*pos, **kw):
        result = greedy(*pos, **kw)
        events.append(
            {
                "stage": "greedy",
                "staff_call": pos[1],
                "filtered_meter": meters(result[0]),
                "raw_meter": meters(result[1]),
            }
        )
        return result

    def trace_plain(*pos, **kw):
        result = plain(*pos, **kw)
        events.append({"stage": "plain", "staff_call": pos[1], "meter": meters(result)})
        return result

    def snapshot(decoded):
        return [
            {"voice": voice, "system": system, "meter": meters(symbols)}
            for (voice, system), symbols in sorted(decoded.items())
        ]

    def trace_repairs(plan, decoded, voices):
        events.append(
            {"stage": "before_cross_staff_repairs", "rows": snapshot(decoded), "slots": plan.slots}
        )
        result = repairs(plan, decoded, voices)
        events.append({"stage": "after_cross_staff_repairs", "rows": snapshot(decoded)})
        return result

    def trace_deduplicate(symbols):
        before = meters(symbols)
        result = deduplicate(symbols)
        events.append({"stage": "voice_cleanup", "before": before, "after": meters(result)})
        return result

    parsing.parse_staff_image_greedy_with_margins = trace_greedy
    parsing.parse_staff_image = trace_plain
    parsing._apply_cross_staff_repairs = trace_repairs
    parsing.remove_duplicated_symbols = trace_deduplicate
    models = json.loads(args.models.read_text())
    if models[identity["arm"]] != identity["models"]:
        raise ValueError("Model identity differs")
    try:
        page_runner.process(identity["page"], models, identity["arm"], args.out)
    finally:
        if args.out.exists():
            write(args.out / "meter-trace.json", events)
    xml_equal = ET.tostring(ET.parse(args.baseline / "page.musicxml").getroot()) == ET.tostring(
        ET.parse(args.out / "page.musicxml").getroot()
    )
    cache_equal = json.loads((args.baseline / "decoded-staves.json").read_text()) == json.loads(
        (args.out / "decoded-staves.json").read_text()
    )
    write(
        args.out / "trace-receipt.json",
        {
            "baseline": str(args.baseline),
            "baseline_identity_sha256": sha(args.baseline / "identity.json"),
            "trace_source_sha256": sha(Path(__file__)),
            "exact_xml": xml_equal,
            "exact_cached_symbols": cache_equal,
            "meter_reference_injected": False,
        },
    )
    if not xml_equal or not cache_equal:
        raise ValueError("Instrumentation changed output; attribution invalid")


if __name__ == "__main__":
    main()
