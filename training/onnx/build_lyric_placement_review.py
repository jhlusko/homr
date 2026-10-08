"""Show cached source-free lyric/note proposals; no ground-truth verse mapping."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from training.onnx.lyric_staff_pitch_geometry import interpolate_notehead_positions
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((args.predictions / "report.json").read_text())
    for path, digest in previous["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Cached input changed")
    pages = json.loads(args.pages.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    inventory = []
    for index, page in enumerate(pages):
        prediction_path = args.predictions / (page["score"] + "-predictions.json")
        result = json.loads(prediction_path.read_text())
        if result["native_input_used"] or not result["export_trace"]["exact_xml"]:
            raise ValueError("Require source-free exact export provenance")
        folder = args.decode / f"page-{index}"
        identity = json.loads((folder / "identity.json").read_text())
        image = Path(identity["page"]["image"])
        if sha(image) != identity["page"]["image_sha256"]:
            raise ValueError("Page image changed")
        transforms = json.loads((folder / "staff-transforms.json").read_text())
        grids = {t["staff_call"]: t for t in transforms}
        with Image.open(image) as source:
            for number, prediction in enumerate(result["predictions"]):
                matches = [
                    t
                    for t in result["attachment_trace"]
                    if t.get("outcome") == "attached"
                    and t.get("selected_row_index") == prediction["selected_row_index"]
                    and t.get("syllable_index") == prediction["syllable_index"]
                ]
                if len(matches) != 1:
                    raise ValueError("Nonunique attachment provenance")
                trace = matches[0]
                transform = grids[trace["staff_call"]]
                head = interpolate_notehead_positions(transform)[trace["nearest_head_index"]]
                ratio = transform["original_to_processed_ratio"]
                nx, ny = head["center"][0] * ratio[0], head["center"][1] * ratio[1]
                lx, ly = trace["center"]
                left = max(0, int(min(lx, nx) - 220))
                right = min(source.width, int(max(lx, nx) + 220))
                top = max(0, int(min(ly, ny) - 90))
                bottom = min(source.height, int(max(ly, ny) + 70))
                crop = source.crop((left, top, right, bottom)).convert("RGB")
                draw = ImageDraw.Draw(crop)
                draw.ellipse(
                    (nx - left - 12, ny - top - 12, nx - left + 12, ny - top + 12),
                    outline="#1766ce",
                    width=3,
                )
                draw.ellipse(
                    (lx - left - 12, ly - top - 12, lx - left + 12, ly - top + 12),
                    outline="#ca6c10",
                    width=3,
                )
                name = f'{page["score"]}-{number}.jpg'
                crop.save(args.out / name, quality=90)
                note = prediction["note"]
                row = prediction["physical_row_order"]
                inventory.append(
                    {
                        "id": f'{page["score"]}:{number}',
                        "pageSha256": sha(image),
                        "musicXmlSha256": sha(folder / "page.musicxml"),
                        "score": page["score"],
                        "text": prediction["text"],
                        "physicalRowOrder": row,
                        "generatedNote": note,
                        "image": name,
                        "image_sha256": sha(args.out / name),
                        "predictions_sha256": sha(prediction_path),
                        "verseConfirmed": False,
                        "manualTextReviewed": False,
                    }
                )
    template = Path(__file__).with_name("lyric_placement_review.html")
    (args.out / "index.html").write_bytes(template.read_bytes())
    write(args.out / "inventory.json", inventory)
    write(
        args.out / "receipt.json",
        {
            "cases": len(inventory),
            "source_sha256": sha(__file__),
            "template_sha256": sha(template),
            "inventory_sha256": sha(args.out / "inventory.json"),
            "original_score_report_sha256": sha(args.predictions / "report.json"),
            "ground_truth_verse_mapping_used": False,
            "musicxml_modified": False,
        },
    )
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print("Published", len(inventory), "cached placement proposals")


if __name__ == "__main__":
    main()
