"""
Corrected version of deep_barline_audit.py, fixing the bug documented in that file's
own RETRACTED docstring: ground truth now comes from the piece's real top-level
MusicXML (`ossq_ground_truth.real_ground_truth_path`), and the page-local measure_index
a `majority_position_correction` proposal names is converted to an absolute score
measure number using the corpus's own `measure_start` metadata
(`ossq_ground_truth.measure_start_for_system`) - not self-summed barline counts, which
depended on HOMR's own (possibly wrong) decode of every earlier system on the page.

Same three buckets as before: ground_truth_disagrees (known corpus defect),
ground_truth_agrees (candidate: could be a genuine HOMR decode error - now actually
trustworthy), no_ground_truth (no real ground truth file, or no corpus measure-mapping
metadata for this page/system).
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")
sys.path.insert(0, "/workspace/b0/homr/training/omr_datasets")

from homr.cross_staff_consistency import _cumulative_barline_positions, staves_by_system
from homr.cross_staff_repair import propose_majority_position_corrections
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.staff_parsing import _plan_systems, parse_staffs
from homr.transformer.configs import Config as TransformerConfig
from ossq_ground_truth import measure_start_for_system, real_ground_truth_path
from ossq_measure_length_audit import measure_length_by_part
import xml.etree.ElementTree as ET


def analyze_page(image_path: str) -> list[dict]:
    config = ProcessingConfig(False, False, False, False, -1, True, True, False, False, None)
    multi_staffs, image, debug, title_future, _ = detect_staffs_in_image(image_path, config)
    transformer_config = TransformerConfig()
    voices = parse_staffs(debug, multi_staffs, image, config=transformer_config)
    plan = _plan_systems(multi_staffs)
    presence = [
        [plan.staff_for_voice(system, voice) is not None for voice in range(len(voices))]
        for system in range(len(plan.systems))
    ]
    systems = list(staves_by_system(voices, presence))

    results = []
    for system_index, staves in enumerate(systems):
        for proposal in propose_majority_position_corrections(staves):
            results.append(
                {
                    "system_index": system_index,
                    "staff_index": proposal.staff_index,
                    "measure_index_in_system": proposal.measure_index,
                    "offset": str(proposal.offset),
                    "corroborating_staves": proposal.corroborating_staves,
                }
            )
    return results


def check_ground_truth(gt_path: Path, measure_number: int) -> dict | None:
    try:
        tree = ET.parse(gt_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    parts = root.findall(".//part")
    if len(parts) < 2:
        return None
    per_part_divisions = {p.get("id"): {"current": 1} for p in parts}
    lengths = {}
    for p in parts:
        measures = [m for m in p.findall("measure") if m.get("number") == str(measure_number)]
        if not measures:
            return None
        lengths[p.get("id")] = str(
            measure_length_by_part(measures[0], per_part_divisions[p.get("id")])
        )
    distinct = set(lengths.values())
    return {"per_part_lengths": lengths, "agrees": len(distinct) == 1}


def main() -> None:
    sample_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    images = [line.strip() for line in sample_path.read_text().splitlines() if line.strip()]

    all_results = []
    for idx, image in enumerate(images, 1):
        image_path = Path(image)
        try:
            proposals = analyze_page(image)
        except Exception as e:  # noqa: BLE001
            print(f"[{idx}/{len(images)}] {image_path.name}: ERROR - {e}", flush=True)
            traceback.print_exc()
            continue
        if not proposals:
            print(f"[{idx}/{len(images)}] {image_path.name}: no proposals", flush=True)
            continue

        gt_path = real_ground_truth_path(image_path)
        for proposal in proposals:
            entry = {"image": image, **proposal}
            measure_start = measure_start_for_system(image_path, proposal["system_index"])
            if gt_path is None or measure_start is None:
                entry["absolute_measure_number"] = None
                entry["ground_truth_check"] = None
            else:
                absolute = measure_start + proposal["measure_index_in_system"]
                entry["absolute_measure_number"] = absolute
                entry["ground_truth_check"] = check_ground_truth(gt_path, absolute)
            all_results.append(entry)
        print(
            f"[{idx}/{len(images)}] {image_path.name}: {len(proposals)} proposal(s)",
            flush=True,
        )
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    disagrees = sum(
        1 for r in all_results if r["ground_truth_check"] and not r["ground_truth_check"]["agrees"]
    )
    agrees = sum(
        1 for r in all_results if r["ground_truth_check"] and r["ground_truth_check"]["agrees"]
    )
    no_gt = sum(1 for r in all_results if r["ground_truth_check"] is None)
    print("\n===== SUMMARY (corrected: real ground truth + corpus measure_start) =====")
    print(f"total majority_position_correction proposals: {len(all_results)}")
    print(f"  ground truth disagrees (known corpus defect by the invariant): {disagrees}")
    print(f"  ground truth agrees (candidate: real decode error): {agrees}")
    print(f"  no ground truth / no measure-mapping metadata available: {no_gt}")


if __name__ == "__main__":
    main()
