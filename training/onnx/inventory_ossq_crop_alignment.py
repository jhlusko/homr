"""Inventory local OSSQ crops against reviewed page boundaries, without guessing joins."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write

NAME = re.compile(r"(sq\d+)_(\d+)_(\d+)_([1-4])\.txt$")


def inventory(corpus, ossq, certified):
    receipt = json.loads((certified / "receipt.json").read_text())
    page_path = certified / "pages.json"
    if sha(page_path) != receipt["pages_sha256"]:
        raise ValueError("Certified page manifest changed")
    pages = json.loads(page_path.read_text())
    reviewed = {}
    for page in pages:
        if page["source"] != "ossq":
            continue
        for key in ("image", "reference"):
            if sha(page[key]) != page[key + "_sha256"]:
                raise ValueError("Certified input changed")
        page_number = int(Path(page["image"]).stem.split(":")[-1])
        reviewed[(page["score"], page_number)] = page
    references = {
        p.stem: p for p in ossq.glob("scores/*/*/sq*.musicxml") if re.fullmatch(r"sq\d+", p.stem)
    }
    scores, counts, matched, indexes = {}, Counter(), [], {}
    for split in ("train", "valid"):
        index_path = corpus / split / "index.txt"
        index_rows = [line.split(",") for line in index_path.read_text().splitlines() if line]
        if any(len(row) != 2 for row in index_rows):
            raise ValueError("Malformed corpus index")
        indexed_tokens = [row[1] for row in index_rows]
        if len(set(indexed_tokens)) != len(indexed_tokens):
            raise ValueError("Duplicate corpus index labels")
        local_tokens = {p.name for p in (corpus / split).glob("sq*.txt")}
        if set(indexed_tokens) != local_tokens:
            raise ValueError("Index and local token inventory disagree")
        if any(not (corpus / split / name).is_file() for row in index_rows for name in row):
            raise ValueError("Missing indexed crop file")
        indexes[split] = dict(path=str(index_path), sha256=sha(index_path), rows=len(index_rows))
        for token in sorted((corpus / split).glob("*.txt")):
            if token.name == "index.txt":
                continue
            match = NAME.fullmatch(token.name)
            if match is None:
                raise ValueError(f"Unrecognized crop name: {token}")
            score, page, system, part = match.groups()
            row = scores.setdefault(
                score, {"splits": Counter(), "missing_images": 0, "reviewed_page_candidates": 0}
            )
            row["splits"][split] += 1
            counts[split + "_crops"] += 1
            if not token.with_suffix(".png").is_file():
                row["missing_images"] += 1
            certified_page = reviewed.get((score, int(page)))
            if certified_page is not None:
                row["reviewed_page_candidates"] += 1
                matched.append(
                    dict(
                        token=str(token),
                        score=score,
                        page=int(page),
                        system=int(system),
                        part=int(part),
                        split=split,
                        scope="Filename join only; crop pixels and part identity unverified",
                    )
                )
    for score, row in scores.items():
        source = references.get(score)
        row["reference"] = str(source) if source else None
        row["reference_sha256"] = sha(source) if source else None
        if source:
            folder = source.parent
            csv = folder / (score + "_scanned.csv")
            row["scan_csv"] = str(csv) if csv.is_file() else None
            row["scan_csv_sha256"] = sha(csv) if csv.is_file() else None
            row["scan_csv_nonempty"] = bool(csv.is_file() and csv.read_text().strip())
            row["scanned_system_metadata"] = len(
                list((folder / "metadata/scanned/systemwise").glob("*.yaml"))
            )
            row["unaligned_system_metadata"] = len(
                list((folder / "metadata/unaligned").glob("*.yaml"))
            )
        else:
            row.update(
                scanned_system_metadata=0, unaligned_system_metadata=0, scan_csv_nonempty=False
            )
    return dict(
        corpus=str(corpus),
        certified_manifest_sha256=sha(page_path),
        corpus_manifest_sha256=(
            sha(corpus / "MANIFEST.json") if (corpus / "MANIFEST.json").is_file() else None
        ),
        counts=dict(
            counts,
            scores=len(scores),
            reviewed_ossq_pages=len(reviewed),
            reviewed_page_crop_candidates=len(matched),
            scores_with_scanned_system_metadata=sum(
                r["scanned_system_metadata"] > 0 for r in scores.values()
            ),
            scores_with_nonempty_scan_csv=sum(r["scan_csv_nonempty"] for r in scores.values()),
        ),
        scores=scores,
        indexes=indexes,
        matched_crop_candidates=matched,
        reviewed_scores_absent_from_corpus=sorted({s for s, _ in reviewed} - scores.keys()),
        timing_training_support_qualified=False,
        scope="Local legacy corpus inventory; directory splits, not a current-model training index. "
        "Unaligned metadata and scan CSV presence do not certify crop/source identity. "
        "Reviewed evaluation pages must not be relabeled as training support.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "ossq", "certified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to replace inventory")
    result = inventory(args.corpus.resolve(), args.ossq.resolve(), args.certified.resolve())
    result["tool_sha256"] = sha(Path(__file__))
    write(args.out, result)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
