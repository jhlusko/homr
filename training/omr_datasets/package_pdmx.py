"""Package the converted PDMX corpus into verifiable shards for distribution.

The corpus is ~50K windows in three files each (image, tokens, sidecar). Sharding is not
about surviving a dropped connection - fetch_source.py already resumes a single archive
with a .part file and a Range header, which is how the 774MB Lieder snapshot is fetched
today. Two things a monolith cannot do:

* **Fetch one split.** The validation shard is 142MB against 1.94GB, and evaluating a
  released checkpoint needs nothing else.
* **Localise a bad download.** A digest mismatch invalidates one 280MB shard rather than
  the whole corpus.

Shards never straddle a score. convert_pdmx cuts each score into overlapping windows, so
two windows of one score share an engraving and often the same bars; keeping a score whole
means a shard is a self-contained unit and a partial download can never be mistaken for a
clean subset of the split.

Train and validation are packaged separately. Evaluating a released checkpoint needs only
the validation shards, which is a fraction of the bytes.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from training.omr_datasets.pdmx_split_protected import score_of

#: How a row's filename names the score it came from, per corpus. The default splits on the
#: last `-v`, which is right for PDMX (`<hash>-v<n>-w<n>`) and wrong for anything else:
#: Lieder names windows `IMSLP<id>-sys<n>-v<n>`, so the default returns `IMSLP10416-sys0` -
#: the *system*, not the source. That turns "a shard never straddles a score" into "never
#: straddles a system", which is a different and much weaker promise, and it would let two
#: systems of one scan land on opposite sides of a split that believed itself disjoint.
SCORE_PATTERNS = {"pdmx": None, "lieder": r"^(IMSLP\d+)"}

MEMBER_SUFFIXES = (".jpg", ".tokens", ".tokens.notation.json")


def _members(row: str) -> list[str]:
    image, tokens = row.rsplit(",", 1)
    return [image, tokens, tokens + ".notation.json"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _score_key(pattern: str | None):
    """The function that maps a row to the score it belongs to."""
    if pattern is None:
        return score_of
    compiled = re.compile(pattern)

    def key(row: str) -> str:
        name = Path(row.split(",", 1)[0]).name
        match = compiled.search(name)
        if not match:
            raise ValueError(f"score pattern {pattern!r} does not match {name!r}")
        return match.group(1)

    return key


def _shard_scores(
    rows: list[str], target_bytes: int, root: Path, pattern: str | None = None
) -> list[list[str]]:
    key = _score_key(pattern)
    by_score: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_score[key(row)].append(row)

    shards: list[list[str]] = []
    current: list[str] = []
    size = 0
    for score in sorted(by_score):
        score_rows = by_score[score]
        score_size = sum(
            (root / name).stat().st_size for row in score_rows for name in _members(row)
        )
        if current and size + score_size > target_bytes:
            shards.append(current)
            current, size = [], 0
        current.extend(score_rows)
        size += score_size
    if current:
        shards.append(current)
    return shards


def _write_shard(rows: list[str], root: Path, out: Path, compress: str, level: int = 10) -> int:
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)
    try:
        with tarfile.open(tar_path, "w") as tar:
            for row in rows:
                for name in _members(row):
                    tar.add(root / name, arcname=name)
        if compress == "zstd":
            subprocess.run(  # noqa: S603
                ["pzstd", "-q", f"-{level}", "-f", "-o", str(out), str(tar_path)],
                check=True,
            )
        else:
            out.write_bytes(tar_path.read_bytes())
    finally:
        tar_path.unlink(missing_ok=True)
    return out.stat().st_size


def package(
    root: Path,
    index_dir: Path,
    dest: Path,
    *,
    target_mb: int,
    compress: str,
    level: int = 10,
    pattern: str | None = None,
) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    shard_dir = dest / "shards"
    shard_dir.mkdir(exist_ok=True)
    suffix = ".tar.zst" if compress == "zstd" else ".tar"

    manifest: dict = {"shards": [], "splits": {}}
    for split in ("train", "valid"):
        index = index_dir / f"index_{split}.txt"
        rows = [line for line in index.read_text().splitlines() if line.strip()]
        groups = _shard_scores(rows, target_mb * 1024 * 1024, root, pattern)
        key = _score_key(pattern)
        manifest["splits"][split] = {
            "windows": len(rows),
            "scores": len({key(r) for r in rows}),
            "shards": len(groups),
        }
        for n, group in enumerate(groups):
            name = f"{split}-{n:04d}{suffix}"
            out = shard_dir / name
            size = _write_shard(group, root, out, compress, level)
            manifest["shards"].append(
                {
                    "name": name,
                    "split": split,
                    "windows": len(group),
                    "scores": len({key(r) for r in group}),
                    "bytes": size,
                    "sha256": _sha256(out),
                }
            )
            print(f"{name}  {len(group):6d} windows  {size / 1e6:8.1f} MB", flush=True)
        (dest / f"index_{split}.txt").write_text(
            "".join(r + "\n" for r in rows), encoding="utf-8"
        )

    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dest / "SHA256SUMS").write_text(
        "".join(f"{s['sha256']}  shards/{s['name']}\n" for s in manifest["shards"]),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="index rows are relative to this")
    parser.add_argument(
        "--index-dir", type=Path, required=True, help="holds index_{train,valid}.txt"
    )
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--target-mb", type=int, default=500, help="approximate shard size")
    parser.add_argument("--compress", choices=("zstd", "none"), default="zstd")
    parser.add_argument("--level", type=int, default=10, help="zstd level")
    parser.add_argument(
        "--corpus",
        choices=sorted(SCORE_PATTERNS),
        default="pdmx",
        help="how filenames name their score; 'pdmx' splits on the last -v, 'lieder' takes "
        "the IMSLP id. Getting this wrong groups by system and silently weakens the "
        "no-straddling guarantee.",
    )
    args = parser.parse_args()

    manifest = package(
        args.root,
        args.index_dir,
        args.dest,
        target_mb=args.target_mb,
        compress=args.compress,
        level=args.level,
        pattern=SCORE_PATTERNS[args.corpus],
    )
    print(json.dumps(manifest["splits"], indent=2))
    total = sum(s["bytes"] for s in manifest["shards"])
    print(f"{len(manifest['shards'])} shards, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
