"""
RETRACTED - do not use as-is, for the same reason as deep_barline_audit.py.
`ground_truth_path()` resolves to `<image>.with_suffix(".musicxml")`, which is exactly
where `homr.main` writes its own decode output - every "ground truth" this script
checked was HOMR's own prior output, not real ground truth (which lives at
`scores/<composer>/<piece>/sq<id>.musicxml` instead). Produced a fully invalid
"509/517 corpus noise, 1 candidate" result in this exact form once already; see
`OSSQ_GROUND_TRUTH_ERRORS.md`'s retraction and `DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.1.
Needs the same fix as deep_barline_audit.py before reuse.

Original docstring, describing what the (currently broken) comparison was meant to do:

Broader follow-up to deep_barline_audit.py: that script only checked measures where
propose_majority_position_corrections actually fires (a 3+ staff majority AND a
constant offset) - a narrow, high-confidence slice that turned out to be 91/91 corpus
noise. This script checks every barline_position_mismatch disagreement
check_barline_positions itself would flag, regardless of majority size or whether the
offset stays constant afterward - the "messier" cases (chaotic multi-staff
disagreement, 2-staff disagreements below the 3-staff corroboration bar) the narrow
check doesn't reach, explicitly named as the remaining open question in
ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md §5 and DECODER_RHYTHM_ACCURACY_DESIGN.md §7.1.

For every system with any barline-position disagreement (2+ staves with barlines,
truncated sequences not all equal), takes the most common truncated sequence as the
reference and, for every staff that diverges from it, finds the *first* index where it
differs and converts that to an absolute ground-truth measure number the same way
deep_barline_audit.py does (summing barline counts of every earlier system). Reports
whether that measure's ground truth agrees across parts, and separately whether this
specific divergence would have cleared propose_majority_position_corrections' own bar
(constant offset, 3+ agreeing majority) - so the narrow and broad results stay directly
comparable.
"""
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")

from homr.cross_staff_consistency import _cumulative_barline_positions, staves_by_system
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
        positions = {i: _cumulative_barline_positions(s) for i, s in enumerate(staves)}
        with_barlines = {i: p for i, p in positions.items() if p}
        counts = [len(p) for p in with_barlines.values()]
        majority_count = max(set(counts), key=counts.count) if counts else 0

        if len(with_barlines) >= 2:
            shortest = min(len(p) for p in with_barlines.values())
            if shortest > 0:
                truncated = {i: tuple(p[:shortest]) for i, p in with_barlines.items()}
                seq_counts = Counter(truncated.values())
                majority_seq, majority_n = seq_counts.most_common(1)[0]
                tied = list(seq_counts.values()).count(majority_n) > 1
                if len(set(truncated.values())) > 1:
                    for staff_index, seq in truncated.items():
                        if seq == majority_seq:
                            continue
                        divergence_index = next(
                            i for i in range(shortest) if seq[i] != majority_seq[i]
                        )
                        offsets = {
                            majority_seq[i] - seq[i] for i in range(divergence_index, shortest)
                        }
                        constant_offset = len(offsets) == 1
                        stage_b_eligible = (
                            majority_n >= 3 and not tied and constant_offset
                        )
                        results.append(
                            {
                                "system_index": system_index,
                                "staff_index": staff_index,
                                "divergence_index_in_system": divergence_index,
                                "absolute_measure_number": barlines_before_this_system
                                + divergence_index
                                + 1,
                                "majority_size": majority_n,
                                "majority_tied": tied,
                                "constant_offset": constant_offset,
                                "stage_b_eligible": stage_b_eligible,
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
        try:
            divergences = analyze_page(image)
        except Exception as e:  # noqa: BLE001
            print(f"[{idx}/{len(images)}] {Path(image).name}: ERROR - {e}", flush=True)
            traceback.print_exc()
            continue
        if not divergences:
            print(f"[{idx}/{len(images)}] {Path(image).name}: no divergences", flush=True)
            continue

        gt_path = ground_truth_path(image)
        for divergence in divergences:
            entry = {"image": image, **divergence}
            entry["ground_truth_check"] = (
                check_ground_truth(gt_path, divergence["absolute_measure_number"])
                if gt_path is not None
                else None
            )
            all_results.append(entry)
        print(
            f"[{idx}/{len(images)}] {Path(image).name}: {len(divergences)} divergence(s)",
            flush=True,
        )
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    def bucket(pred):
        return sum(1 for r in all_results if pred(r))

    print("\n===== SUMMARY (all barline-position divergences, not just Stage-B-eligible) =====")
    print(f"total divergences: {len(all_results)}")
    print(f"  stage_b_eligible (3+ majority, constant offset): {bucket(lambda r: r['stage_b_eligible'])}")
    print(f"  NOT stage_b_eligible (the messier, previously-unmeasured cases): {bucket(lambda r: not r['stage_b_eligible'])}")
    print()
    for label, pred in [
        ("all divergences", lambda r: True),
        ("stage_b_eligible only", lambda r: r["stage_b_eligible"]),
        ("NOT stage_b_eligible only", lambda r: not r["stage_b_eligible"]),
    ]:
        disagrees = bucket(lambda r, pred=pred: pred(r) and r["ground_truth_check"] and not r["ground_truth_check"]["agrees"])
        agrees = bucket(lambda r, pred=pred: pred(r) and r["ground_truth_check"] and r["ground_truth_check"]["agrees"])
        no_gt = bucket(lambda r, pred=pred: pred(r) and r["ground_truth_check"] is None)
        total = bucket(pred)
        print(f"{label} (n={total}): gt_disagrees={disagrees}, gt_agrees={agrees}, no_gt={no_gt}")


if __name__ == "__main__":
    main()
