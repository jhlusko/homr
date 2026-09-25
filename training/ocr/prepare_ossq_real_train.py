"""Build score-audited real OSSQ detector masks from archived OCR-confirmed matches.

The match records name an old /workspace/b0 scan root. The byte-identical scans now
live under /workspace/ossq-build/ossq-omr/scores; only this prefix is remapped.
Train scores come from the 09-19 audit, while both its heldouts and the frozen B3
selection/test scores are excluded before any image is read.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from training.ocr.detector_masks import CLASS_INDEX
from training.ocr.scan_text_masks import BLANK_THRESHOLD, IGNORE

OLD_ROOT = Path("/workspace/b0/ossq-omr/scores")
KIND_CLASS = {
    "dynamic": "Dynamic",
    "tempo": "DirectionText",
    "stafftext": "DirectionText",
    "expression": "DirectionText",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed_scores(audit: dict, split: dict) -> set[str]:
    frozen = {
        score
        for corpus in split["corpora"].values()
        for role in corpus["roles"].values()
        for score in role["scores"]
    }
    train = set(audit["train_scores"])
    earlier_heldout = set(audit["heldout_scores"])
    if train & (frozen | earlier_heldout):
        raise ValueError("audited train scores overlap a held-out cohort")
    return train


def source_page(old_name: str, scan_root: Path, score: str) -> Path:
    old = Path(old_name)
    try:
        relative = old.relative_to(OLD_ROOT)
    except ValueError as exc:
        raise ValueError(f"unexpected source root: {old}") from exc
    if ".." in relative.parts or not relative.name.startswith(f"{score}:"):
        raise ValueError(f"source page does not belong to {score}: {old}")
    page = scan_root / relative
    if not page.is_file():
        raise FileNotFoundError(page)
    return page


def build_score(task: tuple[Path, Path, Path]) -> dict:
    doc_path, scan_root, mask_root = task
    cv2.setNumThreads(1)
    doc = json.loads(doc_path.read_text())
    score = doc["score_id"]
    if score != doc_path.stem:
        raise ValueError(f"score mismatch: {doc_path}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    skipped = Counter()
    for match in doc["matches"]:
        if match["kind"] in KIND_CLASS:
            page_id = Path(match["page_image"]).name.split(":", 1)[0]
            if page_id != score:
                # Three OCR match groups in sq7070781 point at other scores,
                # including one held-out score. Never infer permission from the
                # containing JSON filename when the image says otherwise.
                skipped["cross_score_matches"] += 1
            else:
                grouped[match["page_image"]].append(match)
    rows = []
    labels = Counter()
    for old_name, matches in sorted(grouped.items()):
        try:
            image_path = source_page(old_name, scan_root, score)
        except FileNotFoundError:
            # Original OCR export included preview *_teaser.png files absent
            # from the pinned full-resolution scan tree. Record their loss.
            skipped["missing_page_matches"] += len(matches)
            skipped["missing_pages"] += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"unreadable image: {image_path}")
        height, width = image.shape
        mask = np.where(image >= BLANK_THRESHOLD, 0, IGNORE).astype(np.uint8)
        for match in matches:
            label = KIND_CLASS[match["kind"]]
            box = match["box"]
            left = max(0, min(width, int(box["left"])))
            top = max(0, min(height, int(box["top"])))
            right = max(0, min(width, int(box["left"] + box["width"])))
            bottom = max(0, min(height, int(box["top"] + box["height"])))
            if right <= left or bottom <= top:
                continue
            mask[top:bottom, left:right] = CLASS_INDEX[label]
            labels[label] += 1
        destination = mask_root / score / f"{image_path.stem}.mask.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), mask):
            raise OSError(f"cannot write mask: {destination}")
        rows.append(f"{image_path},{destination}")
    return {"score": score, "pages": len(rows), "boxes": dict(labels),
            "skipped": dict(skipped), "index_rows": rows}


def prepare(args: argparse.Namespace) -> dict:
    audit = json.loads(args.page_audit.read_text())
    split = json.loads(args.split_manifest.read_text())
    train = allowed_scores(audit, split)
    docs = sorted(args.matches.glob("sq*.json"))
    selected = [doc for doc in docs if doc.stem in train]
    if not selected:
        raise ValueError("no approved OSSQ training score has a match file")
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(doc, args.scan_root, args.out / "masks") for doc in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(build_score, tasks))
    index_rows = sorted(row for result in results for row in result["index_rows"])
    (args.out / "index.txt").write_text("\n".join(index_rows) + "\n")
    totals = Counter()
    skipped = Counter()
    for result in results:
        totals.update(result["boxes"])
        skipped.update(result["skipped"])
    summary = {
        "score_documents": len(selected),
        "scores_with_pages": sum(result["pages"] > 0 for result in results),
        "pages": len(index_rows), "boxes": dict(totals),
        "skipped": dict(skipped),
        "selected_scores": sorted(doc.stem for doc in selected),
        "matches_sha256": {doc.name: sha256(doc) for doc in selected},
        "page_audit_sha256": sha256(args.page_audit),
        "split_manifest_sha256": sha256(args.split_manifest),
        "scan_root": str(args.scan_root),
        "policy": "OCR-confirmed boxes; blank paper background; unlabeled ink ignore",
        "class_order": list(CLASS_INDEX),
    }
    (args.out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--page-audit", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    print(json.dumps(prepare(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
