"""Paired old/new meter guards on hash-verified OSSQ and Lieder sources."""

import argparse
import ast
import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from homr import music_xml_generator as generator
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write


def load_old_guard(path):
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "stated_numerator_contradicts_bars"
    )
    namespace = dict(vars(generator))
    exec(  # noqa: S102 - compile only the audited local function snapshot
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["stated_numerator_contradicts_bars"]


def without_times(root):
    root = copy.deepcopy(root)
    for attributes in root.findall("./part/measure/attributes"):
        for signature in attributes.findall("time"):
            attributes.remove(signature)
    return hashlib.sha256(ET.tostring(root)).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inventory", type=Path, required=True)
    cli.add_argument("--old-generator", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "report.json").exists():
        raise ValueError("Refusing overwrite")
    old_guard = load_old_guard(args.old_generator)
    new_guard = generator.stated_numerator_contradicts_bars
    results = []
    start = time.monotonic()
    rows = list(map(json.loads, args.inventory.read_text().splitlines()))
    for index, row in enumerate(rows, 1):
        path = Path(row["reference"])
        if not sha(path) == row["reference_sha256"]:
            raise ValueError("Qualification invariant failed")
        root = read_mxl(path) if path.suffix == ".mxl" else ET.parse(path).getroot()
        voices = music_xml_string_to_tokens(ET.tostring(root, encoding="unicode"))
        arms = {}
        xmls = {}
        for name, guard in [("old", old_guard), ("new", new_guard)]:
            try:
                with (
                    patch.object(generator, "stated_numerator_contradicts_bars", guard),
                    patch.object(generator, "structured_export_enabled", return_value=False),
                ):
                    xml = generator.generate_xml(
                        generator.XmlGeneratorArguments(),
                        copy.deepcopy([[s for m in voice for s in m] for voice in voices]),
                        "Meter qualification",
                    )
                xmls[name] = xml
                arms[name] = {
                    "xml_sha256": hashlib.sha256(ET.tostring(xml)).hexdigest(),
                    "without_time_sha256": without_times(xml),
                    "times": [
                        ET.tostring(t, encoding="unicode")
                        for t in xml.findall("./part/measure/attributes/time")
                    ],
                }
            except Exception as error:
                arms[name] = {"error": f"{type(error).__name__}: {error}"}
        changed = arms["old"] != arms["new"]
        non_time_changed = arms["old"].get("without_time_sha256") != arms["new"].get(
            "without_time_sha256"
        )
        if changed:
            for name, xml in xmls.items():
                ET.ElementTree(xml).write(
                    args.out / (row["score"] + "-" + name + ".musicxml"), encoding="utf-8"
                )
        result = {
            "source": row["source"],
            "score": row["score"],
            "reference": str(path),
            "changed": changed,
            "non_time_changed": non_time_changed,
            "arms": arms,
            "source_times": [
                ET.tostring(t, encoding="unicode")
                for t in root.findall("./part/measure/attributes/time")
            ],
        }
        results.append(result)
        write(args.out / (row["score"] + ".json"), result)
        print(
            f"{index}/{len(rows)} {row['source']} {row['score']} "
            f"changed={changed} non_time={non_time_changed}",
            flush=True,
        )
    write(
        args.out / "report.json",
        {
            "scores": results,
            "seconds": time.monotonic() - start,
            "threads": 1,
            "inventory_sha256": sha(args.inventory),
            "old_generator_sha256": sha(args.old_generator),
            "tool_sha256": sha(Path(__file__)),
            "limitation": (
                "Paired runtime meter guard, legacy export with structured fields disabled; "
                "no predictions or corpus changes. Changed declarations and any non-time "
                "XML differences require source-specific qualification."
            ),
        },
    )


if __name__ == "__main__":
    main()
