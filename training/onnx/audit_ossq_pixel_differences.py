"""Compare decoded historical/public crops without assuming PNG byte equality."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def compare_pixels(row):
    for path_key, hash_key in (
        ("pixels", "pixels_sha256"),
        ("public_pixels", "public_pixels_sha256"),
    ):
        digest = hashlib.sha256(Path(row[path_key]).read_bytes()).hexdigest()
        if digest != row[hash_key]:
            raise ValueError(f"Candidate pixels changed: {row[path_key]}")
    with Image.open(row["pixels"]) as image:
        old = np.asarray(image.convert("RGB"))
    with Image.open(row["public_pixels"]) as image:
        new = np.asarray(image.convert("RGB"))
    same_shape = old.shape == new.shape
    result = dict(
        label=row["label"],
        backup=row["pixels"],
        public=row["public_pixels"],
        backup_sha256=row["pixels_sha256"],
        public_sha256=row["public_pixels_sha256"],
        backup_shape=list(old.shape),
        public_shape=list(new.shape),
        decoded_pixels_equal=bool(same_shape and np.array_equal(old, new)),
    )
    if same_shape:
        result["different_pixels"] = int(np.any(old != new, axis=-1).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    candidates = json.loads(args.candidates.read_text())
    rows = [
        row
        for row in candidates["crops"]
        if "public_pixels_match" in row and not row["public_pixels_match"]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(compare_pixels, rows))
    report = dict(
        byte_differences=len(cases),
        decoded_equal=sum(row["decoded_pixels_equal"] for row in cases),
        dimension_differences=sum(row["backup_shape"] != row["public_shape"] for row in cases),
        certified_crop_mappings=False,
        cases=cases,
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))


if __name__ == "__main__":
    main()
