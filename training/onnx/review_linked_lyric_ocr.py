"""Build a local scan/OCR inspection page without inventing lyric-note alignment."""

import argparse
import json
import shutil
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Review output must be new")
    pages = json.loads(args.pages.read_text())
    targets = [json.loads(line) for line in args.targets.read_text().splitlines()]
    records = []
    args.out.mkdir(parents=True)
    for page in pages:
        result = json.loads((args.ocr / (page["score"] + ".json")).read_text())
        if page["partition"] != "calibration" or result["image_sha256"] != sha(page["image"]):
            raise ValueError("OCR/source image identity differs")
        image_name = page["score"] + ".png"
        shutil.copyfile(page["image"], args.out / image_name)
        records.append(
            {
                "page": page,
                "image": image_name,
                "ocr": result["all_page_text"],
                "targets": [r for r in targets if r["score"] == page["score"]],
            }
        )
    payload = json.dumps(records).replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html")
    html = template.read_text()
    (args.out / "index.html").write_text(html.replace("PAYLOAD", payload))
    write(
        args.out / "receipt.json",
        {
            "pages_sha256": sha(args.pages),
            "targets_sha256": sha(args.targets),
            "ocr_receipt_sha256": sha(args.ocr / "receipt.json"),
            "review_source_sha256": sha(Path(__file__)),
            "template_sha256": sha(template),
            "note_attachment_certified": False,
        },
    )


if __name__ == "__main__":
    main()
