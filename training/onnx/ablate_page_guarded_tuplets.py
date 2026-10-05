"""Replay observed complete-page voices with the diagnostic tuplet context guard."""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from training.onnx.qualify_joint_export import setup, sha, write
from training.onnx.score_full_page_musicxml import compare, staff_events
from training.onnx.score_paired_page_diagnostic import read_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to replace ablation")
    setup()
    from homr import music_xml_generator as generator
    from homr.beam_repair import repair_beams
    from homr.slur_crossing import repair_crossings
    from homr.tie_repair import repair_ties
    from homr.transformer.vocabulary import remove_duplicated_symbols
    from homr.tuplet_repair import repair_symbols
    from training.transformer.audit_guarded_tuplet_cleanup import VARIANTS, policy
    from training.transformer.review_cached_joint_decoding import restore

    comparison = json.loads((args.run / "comparison.json").read_text())
    page = comparison["page"]
    if sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Reference changed")
    expected = staff_events(
        read_reference(page["reference"]), page["start_measure"], page["end_measure"]
    )
    args.out.mkdir(parents=True)
    code_root = Path(__file__).resolve().parents[2]
    source_names = [
        "training/onnx/ablate_page_guarded_tuplets.py",
        "training/transformer/audit_guarded_tuplet_cleanup.py",
        "training/transformer/guarded_tuplet_cleanup.py",
        "homr/transformer/vocabulary.py",
        "homr/staff_parsing.py",
        "homr/music_xml_generator.py",
        "homr/beam_repair.py",
        "homr/tie_repair.py",
        "homr/slur_crossing.py",
        "homr/tuplet_repair.py",
        "training/transformer/review_cached_joint_decoding.py",
    ]
    sources = {name: sha(code_root / name) for name in source_names}
    arms = {}
    for arm in ("accepted", "candidate"):
        directory = args.trace / arm
        receipt = json.loads((directory / "trace-receipt.json").read_text())
        if (
            sources["homr/transformer/vocabulary.py"] != receipt["cleanup_source_sha256"]
            or sources["homr/staff_parsing.py"] != receipt["staff_parsing_sha256"]
        ):
            raise ValueError("Cleanup implementation changed since trace")
        traces = directory / "cleanup-calls.json"
        if not receipt["exact_original_xml"] or not receipt["exact_original_cache"]:
            raise ValueError("Instrumented trace not qualified")
        if sha(traces) != receipt["cleanup_calls_sha256"]:
            raise ValueError("Cleanup trace changed")
        identity = json.loads((args.run / arm / "identity.json").read_text())
        os.environ.update(identity["policy"])
        calls = json.loads(traces.read_text())
        variants = {}
        for name in VARIANTS:
            voices = []
            for call in calls:
                with policy(name):
                    voice = remove_duplicated_symbols(restore(call["before"]))
                voice, _ = repair_symbols(voice)
                repair_beams(voice)
                repair_ties(voice)
                repair_crossings(voice)
                voices.append(voice)
            root = generator.generate_xml(generator.XmlGeneratorArguments(), voices, "")
            metrics = compare(expected, staff_events(root))
            original = args.run / arm / "page.musicxml"
            if sha(original) != comparison["arms"][arm]["xml_sha256"]:
                raise ValueError("Original prediction changed")
            if name == "standard" and (
                ET.tostring(root) != ET.tostring(ET.parse(original).getroot())
                or metrics != comparison["arms"][arm]["metrics"]
            ):
                raise ValueError("Cleanup baseline does not replay original")
            target = args.out / arm / (name + ".musicxml")
            target.parent.mkdir(exist_ok=True)
            ET.ElementTree(root).write(target, encoding="unicode")
            variants[name] = {"metrics": metrics, "xml_sha256": sha(target)}
        arms[arm] = {
            "variants": variants,
            "trace_receipt_sha256": sha(directory / "trace-receipt.json"),
        }
    report = {
        "page": page,
        "arms": arms,
        "threads": 1,
        "tool_sha256": sha(__file__),
        "code_sha256": sources,
        "limitations": (
            "Diagnostic on one previously evaluated page. No model rerun in ablation. "
            "Preserved event multiplicity does not certify source voice identity. "
            "Stage ablations are not qualified for default inference."
        ),
    }
    write(args.out / "report.json", report)
    print(json.dumps(arms, indent=2))


if __name__ == "__main__":
    main()
