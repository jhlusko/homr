"""Audit saved lyric anchors against source-free notehead geometry; no accuracy labels."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from homr.segmentation import inference_segnet as seg
from homr.segmentation.config import segnet_path_onnx
from training.onnx.lyric_notehead_association import forced_pairs_with_x_ties, pitch_staff_position


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_clef(staves, staff):
    clef = None
    ordered = sorted(
        (s for s in staves if s["partIndex"] == staff["partIndex"]),
        key=lambda s: (s["systemIndex"], s["index"]),
    )
    for current in ordered:
        seen_note = False
        for token in current["tokens"]:
            if token[0].startswith("clef_"):
                if current["index"] == staff["index"] and seen_note:
                    raise ValueError("Clef changes within audited staff")
                clef = token[0]
            elif token[0].startswith("note_"):
                seen_note = True
        if current["index"] == staff["index"]:
            if clef is None:
                raise ValueError("No preceding decoded clef")
            return clef
    raise ValueError("Audited staff missing from decoded part")


def staff_positions(mask, heads):
    records = []
    for head in heads:
        x, y = head["center"]
        lo, hi = max(0, int(x) - 40), min(mask.shape[1], int(x) + 40)
        projection = np.count_nonzero(mask[:, lo:hi], axis=1)
        groups = []
        for z in np.where(projection > 20)[0]:
            if not groups or z > groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(int(z))
        lines = [float(np.average(g, weights=projection[g])) for g in groups]
        candidates = []
        for start in range(len(lines) - 4):
            five = lines[start : start + 5]
            gaps = np.diff(five)
            if min(gaps) > 3 and max(gaps) / min(gaps) < 1.3:
                spacing = float(np.median(gaps))
                if five[0] - 3 * spacing <= y <= five[-1] + 3 * spacing:
                    candidates.append(five)
        if len(candidates) != 1:
            # Missing or competing staff geometry is never guessed.
            raise ValueError("Missing or ambiguous local five-line staff")
        five = candidates[0]
        spacing = float(np.median(np.diff(five)))
        position_float = 1 + (five[-1] - y) / (spacing / 2)
        if abs(position_float - round(position_float)) > 0.35:
            raise ValueError("Head too far from a diatonic position")
        records.append(dict(head, staffPosition=round(position_float), staffLines=five))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    model = seg.Segnet.__new__(seg.Segnet)
    model.fp16 = False
    model.model = ort.InferenceSession(
        segnet_path_onnx, sess_options=options, providers=["CPUExecutionProvider"]
    )
    model.io_binding = model.model.io_binding()
    seg._segnet_inference = model
    keyfile = (
        args.base / "guarded-tuplet-20261004/review/lyric-anchor-judgement-2-20261008-key.json"
    )
    keys = json.loads(keyfile.read_text())
    pages = dict(
        line.split(" ", 1)
        for line in (
            (args.base / "lyric-fresh2-upload-audit-20261008/fresh2-pages.txt")
            .read_text()
            .splitlines()
        )
    )
    groups = {}
    for key in keys:
        groups.setdefault((key["score"], key["staffIndex"]), []).append(key)
    records, inputs = [], {str(keyfile): digest(keyfile)}
    started = time.monotonic()
    for number, ((score, staff_index), targets) in enumerate(sorted(groups.items()), 1):
        saved = args.base / "lyric-fresh2-upload-audit-20261008/saved-fresh2-inputs"
        reviewfile = saved / f"review-{score}.json"
        review = json.loads(reviewfile.read_text())
        page = Path(pages[score])
        inputs[str(reviewfile)] = digest(reviewfile)
        inputs[str(page)] = digest(page)
        image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
        scale = review["raster"]["w"] / image.shape[1]
        image = cv2.resize(image, (review["raster"]["w"], review["raster"]["h"]))
        inventory_file = saved / f"inventory-{score}.json"
        inputs[str(inventory_file)] = digest(inventory_file)
        inventory = json.loads(inventory_file.read_text())
        staff = next(s for s in review["staves"] if s["index"] == staff_index)
        top = max(0, staff["region"][1] - 10)
        bottom = min(image.shape[0], staff["region"][3] + 10)
        crop = image[top:bottom, :]
        prefix = f"{score}-staff{staff_index}"
        result = {"score": score, "staffIndex": staff_index, "targets": [], "status": "unresolved"}
        try:
            staff_mask, _, _, head_mask, _ = seg.inference(
                crop, use_gpu_inference=False, batch_size=4, step_size=160, win_size=320
            )
            _, _, stats, centers = cv2.connectedComponentsWithStats(head_mask)
            heads = [
                {"center": c.tolist(), "bbox": s[:4].tolist(), "area": int(s[4])}
                for s, c in zip(stats[1:], centers[1:])
                if s[4] >= 120 and 10 <= s[2] <= 30 and 10 <= s[3] <= 24
            ]
            heads.sort(key=lambda h: h["center"][0])
            detected = staff_positions(staff_mask, heads)
            notes = [(i, t) for i, t in enumerate(staff["tokens"]) if t[0].startswith("note_")]
            clef = active_clef(review["staves"], staff)
            positions = [pitch_staff_position(t[1], clef) for _, t in notes]
            pairs = dict(
                forced_pairs_with_x_ties(
                    positions,
                    [h["staffPosition"] for h in detected],
                    [h["center"][0] for h in detected],
                )
            )
            result.update(
                status="sequence_checked",
                decodedCount=len(notes),
                detectedCount=len(detected),
                forcedPairCount=len(pairs),
            )
            annotated = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            for h in detected:
                x, y = h["center"]
                cv2.circle(annotated, (round(x), round(y)), 8, (0, 150, 0), 2)
            note_indexes = [i for i, _ in notes]
            for key in targets:
                # Frozen sample x coordinates are in original page units, not raster units.
                row = next(r for r in inventory["ocrRows"] if r["id"] == key["row"])
                token = row["anchor"]["syllables"][key["syllable"]]["noteTokenIndex"]
                note_index = note_indexes.index(token)
                detected_index = pairs.get(note_index)
                target = {
                    "id": key["id"],
                    "text": key["text"],
                    "attentionX": key["noteX"] * scale,
                    "lyricX": key["x"] * scale,
                    "humanCorrectness": "provisional; original markers imperfect",
                    "status": "unresolved_sequence",
                }
                if detected_index is not None:
                    head = detected[detected_index]
                    target.update(
                        status="forced_sequence_match",
                        detectedX=head["center"][0],
                        detectedY=head["center"][1] + top,
                        displacementPixels=abs(head["center"][0] - target["attentionX"]),
                    )
                result["targets"].append(target)
            cv2.imwrite(str(args.out / f"{prefix}.png"), annotated)
            result["image"] = f"{prefix}.png"
            result["detectedHeads"] = detected
        except ValueError as error:
            result["reason"] = str(error)
            result["targets"] = [
                {"id": k["id"], "text": k["text"], "status": "unresolved_geometry"} for k in targets
            ]
        records.append(result)
        progress = {
            "completed": number,
            "total": len(groups),
            "elapsedSeconds": time.monotonic() - started,
        }
        (args.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(
            json.dumps(
                dict(progress, score=score, staffIndex=staff_index, status=result["status"])
            ),
            flush=True,
        )
    report = {
        "scope": (
            "Development geometry audit only. Earlier human judgements are provisional; "
            "no accuracy, recall or promotion gate. "
            "Fixed component/line thresholds are unqualified."
        ),
        "modelSha256": digest(Path(segnet_path_onnx)),
        "sourceSha256": digest(Path(__file__)),
        "inputSha256": inputs,
        "automaticAttachment": False,
        "records": records,
        "elapsedSeconds": time.monotonic() - started,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
