"""
Overnight Phase 0 follow-up: for each sample page, run HOMR's own pipeline in-process
(not via subprocess+regex scraping) to get exact per-system barline counts on every
system - agreeing or not - so a majority_position_correction proposal's measure_index
(local to its system) can be converted to an absolute ground-truth measure number by
summing barline counts of every earlier system on the page.

For each such proposal, looks up the page's ground-truth MusicXML (sitting right next
to the training image) and checks whether that absolute measure's per-part durations
actually agree once normalized by each part's own <divisions> - the same check
ossq_measure_length_audit.py does, but targeted at exactly the measures HOMR's own
decode flagged, not a blind corpus sweep.

Three buckets result:
  - ground_truth_disagrees: the measure is already a known-bad one by the invariant
    (two parts disagree on length) - a corpus defect, no visual check needed for this
    level of confidence.
  - ground_truth_agrees: the totals check out - could be a genuine HOMR decode error,
    or a hidden content-substitution defect like the Borodin case (which this
    duration-only check cannot see). These need a human/visual follow-up, listed
    separately as candidates.
  - no_ground_truth: no <name>.musicxml sits next to this image, or the absolute
    measure number falls outside what the ground truth has (e.g. HOMR mis-detected
    the number of systems and hoos double-counted its own measures).
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")

from homr.cross_staff_consistency import _cumulative_barline_positions, staves_by_system
from homr.cross_staff_repair import propose_majority_position_corrections
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.staff_parsing import _plan_systems, parse_staffs
from homr.transformer.configs import Config as TransformerConfig
from training.omr_datasets.ossq_measure_length_audit import measure_length_by_part
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
    barlines_before_this_system = 0
    for system_index, staves in enumerate(systems):
        counts = [len(_cumulative_barline_positions(s)) for s in staves]
        majority_count = max(set(counts), key=counts.count) if counts else 0

        for proposal in propose_majority_position_corrections(staves):
            results.append(
                {
                    "system_index": system_index,
                    "staff_index": proposal.staff_index,
                    "measure_index_in_system": proposal.measure_index,
                    "absolute_measure_number": barlines_before_this_system
                    + proposal.measure_index
                    + 1,
                    "offset": str(proposal.offset),
                    "corroborating_staves": proposal.corroborating_staves,
                }
            )
        barlines_before_this_system += majority_count
    return results


def ground_truth_path(image_path: str) -> Path | None:
    p = Path(image_path)
    gt = p.with_suffix(".musicxml")
    return gt if gt.exists() else None


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
        lengths[p.get("id")] = str(measure_length_by_part(measures[0], per_part_divisions[p.get("id")]))
    distinct = set(lengths.values())
    return {"per_part_lengths": lengths, "agrees": len(distinct) == 1}


def main() -> None:
    sample_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    images = [line.strip() for line in sample_path.read_text().splitlines() if line.strip()]

    all_results = []
    for idx, image in enumerate(images, 1):
        try:
            proposals = analyze_page(image)
        except Exception as e:  # noqa: BLE001
            print(f"[{idx}/{len(images)}] {Path(image).name}: ERROR - {e}", flush=True)
            traceback.print_exc()
            continue
        if not proposals:
            print(f"[{idx}/{len(images)}] {Path(image).name}: no proposals", flush=True)
            continue

        gt_path = ground_truth_path(image)
        for proposal in proposals:
            entry = {"image": image, **proposal}
            if gt_path is not None:
                gt_check = check_ground_truth(gt_path, proposal["absolute_measure_number"])
                entry["ground_truth_check"] = gt_check
            else:
                entry["ground_truth_check"] = None
            all_results.append(entry)
        print(
            f"[{idx}/{len(images)}] {Path(image).name}: {len(proposals)} proposal(s)",
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
    print("\n===== SUMMARY =====")
    print(f"total majority_position_correction proposals: {len(all_results)}")
    print(f"  ground truth disagrees (known corpus defect by the invariant): {disagrees}")
    print(f"  ground truth agrees (candidate: real decode error, or hidden content defect): {agrees}")
    print(f"  no ground truth available / measure out of range: {no_gt}")


if __name__ == "__main__":
    main()
