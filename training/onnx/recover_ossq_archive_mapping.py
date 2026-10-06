"""Audit historical crop mappings against public indexes; do not certify them."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--indexes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--public-corpus", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    folders = {
        p.stem: p.parent
        for p in args.archive.glob("scores/*/*/sq*.musicxml")
        if p.stem.startswith("sq") and p.stem[2:].isdigit()
    }
    rows = []
    for split in ("train", "valid"):
        for line in (args.indexes / f"index_{split}.txt").read_text().splitlines():
            image, label = line.split(",")
            score, page, system, part = Path(label).stem.split("_")
            rows.append((split, image, label, score, int(page), int(system), int(part)))

    @lru_cache(maxsize=None)
    def load_metadata(path):
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        return yaml.load(path.read_text(), Loader=loader), sha(path)

    def check(row):
        split, image, label, score, page, system, part = row
        folder = folders[score]
        system_id = f"{score}:{page:04d}:{system:04d}"
        metadata = folder / "metadata/scanned/systemwise" / (system_id + ".yaml")
        pixels = folder / "images/scanned/partwise" / f"{system_id}:{part}.png"
        result = dict(
            split=split,
            label=label,
            image=image,
            score=score,
            system_id=system_id,
            part=part,
            metadata_present=metadata.is_file(),
            pixels_present=pixels.is_file(),
        )
        if metadata.is_file():
            data, metadata_hash = load_metadata(metadata)
            result.update(
                metadata=str(metadata),
                metadata_sha256=metadata_hash,
                measure_numbers=data.get("measure_numbers"),
                metadata_identity_matches=(
                    data.get("score_id") == score
                    and data.get("page_idx") == page
                    and data.get("system_idx") == system
                    and 1 <= part <= data.get("number_of_parts", 0)
                ),
            )
        if pixels.is_file():
            digest = sha(pixels)
            result.update(
                pixels=str(pixels),
                pixels_sha256=digest,
            )
            public = args.public_corpus / image if args.public_corpus else None
            if public and public.is_file():
                public_hash = sha(public)
                result.update(
                    public_pixels=str(public),
                    public_pixels_sha256=public_hash,
                    public_pixels_match=public_hash == digest,
                )
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, rows))
    summary = {}
    for split in ("train", "valid"):
        selected = [row for row in results if row["split"] == split]
        summary[split] = dict(
            windows=len(selected),
            **{
                key: sum(bool(row.get(key)) for row in selected)
                for key in (
                    "metadata_present",
                    "metadata_identity_matches",
                    "pixels_present",
                    "public_pixels_match",
                )
            },
        )
    report = dict(
        summary=summary,
        missing_metadata_by_score=dict(
            Counter(row["score"] for row in results if not row["metadata_present"])
        ),
        index_sha256={
            split: sha(args.indexes / f"index_{split}.txt") for split in ("train", "valid")
        },
        certified_crop_mappings=False,
        scope="Historical metadata candidates. Pixel identity, source physical "
        "measure ownership and independent boundary qualification remain separate gates.",
        crops=results,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
