"""Make a corpus directory self-contained and portable, then verify that it is.

Three things stop these corpora being turnkey, and all three pass a naive file count:

1. **The images are symlinks.** `convert_ossq.py` calls `link_image`, so a staff-crop
   corpus contains links into the machine that built it, not pixels. Copied without
   dereferencing, 38,421 files arrive and every one is dangling - a dataset that looks
   complete and fails on first read.
2. **Index files carry absolute paths.** `<build machine>/workspace/...` means nothing on
   anyone else's disk.
3. **Ground-truth records reference pages by absolute path.** Same problem, one level
   further in.

So this rewrites paths to be relative to the dataset root, materialises any symlink as a
real file, and then *checks its own work* - because the failure mode here is silent, and
a packaging step that reports success on a broken dataset is worse than no packaging
step. `verify()` is the part that matters: it re-reads what was written and fails on a
dangling link, an absolute path, or a reference that does not resolve.

Nothing here is corpus-specific. It takes an index, a root, and rewrites in place.
"""

# flake8: noqa: T201

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def relative_to_root(path: str, root: Path) -> str:
    """`path` expressed relative to `root`, or unchanged if it lies outside.

    Returned with forward slashes regardless of platform: an index is data, and a
    dataset written on Linux must read on Windows.
    """
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path


def materialise(path: Path) -> bool:
    """Replace a symlink with the file it points at. True if it was one.

    Copies to a temporary name and replaces, so an interrupted run cannot leave a
    half-written file where a working link used to be.
    """
    if not path.is_symlink():
        return False
    target = Path(os.path.realpath(path))
    if not target.is_file():
        raise FileNotFoundError(f"dangling symlink: {path} -> {target}")
    temporary = path.with_suffix(path.suffix + ".materialising")
    shutil.copy2(target, temporary)
    path.unlink()
    temporary.rename(path)
    return True


def relocate(path: str, index_dir: Path, root: Path) -> str:
    """One index field, rewritten to a path that resolves on the reader's disk.

    The awkward case, and the reason this is not just `relative_to_root`: the paths in
    these indexes describe the *build machine's* layout (`/workspace/b0/phase7bar/...`),
    while the files ship laid out beside their index. So a plain relative_to() finds no
    common prefix and leaves the row absolute - correctly refused by verify(), but never
    actually repaired.

    Resolution order, most trustworthy first:
      1. already relative and resolves - leave it, it is portable.
      2. a file of that name sits beside the index - the shipped layout.
      3. the path really is under root - relative_to() it.
    Anything else is returned unchanged, so verify() reports it rather than this
    function inventing a path that happens to look plausible.
    """
    if not path.startswith("/") and (index_dir / path).exists():
        return path
    name = Path(path).name
    if (index_dir / name).exists():
        return name
    return relative_to_root(path, root)


def rewrite_index(index_path: Path, root: Path) -> int:
    """Rewrite every path in an `a,b` index to resolve locally. Returns rows rewritten.

    Split from the right: image paths in this corpus contain commas (OSSQ files pages by
    composer), so splitting from the left silently truncates them.
    """
    index_dir = index_path.parent
    rows, changed = [], 0
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.rsplit(",", 1)
        new = (
            f"{relocate(left, index_dir, root)},{relocate(right, index_dir, root)}"
        )
        changed += new != line
        rows.append(new)
    index_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return changed


def rewrite_ground_truth(doc_path: Path, root: Path) -> int:
    """Rewrite absolute `page_image` references in an OCR-first ground-truth file."""
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    changed = 0
    for match in doc.get("matches", []):
        page = match.get("page_image", "")
        if page.startswith("/"):
            match["page_image"] = relative_to_root(page, root)
            changed += 1
    if changed:
        doc_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return changed


def verify(root: Path) -> list[str]:
    """Every reason this dataset is not yet portable. Empty means it is.

    Deliberately re-reads from disk rather than trusting what the rewrite returned: the
    whole point is to catch a packaging step that thought it succeeded.
    """
    problems: list[str] = []

    for path in root.rglob("*"):
        if path.is_symlink():
            problems.append(f"symlink not materialised: {path.relative_to(root)}")

    for index_path in root.rglob("*index*.txt"):
        index_dir = index_path.parent
        for number, line in enumerate(
            index_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            for field in line.rsplit(",", 1):
                if field.startswith("/"):
                    problems.append(f"absolute path {index_path.name}:{number}")
                    break
                # Resolved the way a reader resolves it: relative to the index that
                # names it. Checking against root instead would answer a different
                # question than the one the training loader will ask.
                if not (index_dir / field).exists():
                    problems.append(f"missing file {index_path.name}:{number} -> {field}")
                    break
    return problems


def write_manifest(root: Path, out: Path) -> dict:
    """Checksums and counts, so a downloader can prove they got what we shipped.

    The manifest excludes itself. Otherwise the digest would depend on whether a
    manifest happened to be present when it was computed - ours is written after the
    walk, the downloader's is present before it, and the two never agree. A checksum
    that cannot be reproduced by the person checking it is worse than none.
    """
    files, total = 0, 0
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != out):
        files += 1
        total += path.stat().st_size
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    manifest = {"files": files, "bytes": total, "tree_digest": digest.hexdigest()}
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="Dataset root to package.")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Report problems without changing anything.",
    )
    args = parser.parse_args()

    if not args.verify_only:
        links = 0
        for path in list(args.root.rglob("*")):
            if path.is_symlink():
                materialise(path)
                links += 1
        print(f"materialised {links:,} symlink(s)")

        indexes = sum(rewrite_index(p, args.root) for p in args.root.rglob("*index*.txt"))
        print(f"rewrote {indexes:,} index row(s) to relative paths")

        docs = sum(
            rewrite_ground_truth(p, args.root)
            for p in args.root.rglob("*.json")
            if p.name != "MANIFEST.json"
        )
        print(f"rewrote {docs:,} ground-truth page reference(s)")

        manifest = write_manifest(args.root, args.root / "MANIFEST.json")
        print(f"manifest: {manifest['files']:,} files, {manifest['bytes'] / 2**30:.2f} GB")

    problems = verify(args.root)
    if problems:
        print(f"\nNOT PORTABLE - {len(problems):,} problem(s):")
        for line in problems[:20]:
            print(f"  {line}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20:,} more")
        raise SystemExit(1)
    print("\nverified portable: no symlinks, no absolute paths, every reference resolves")


if __name__ == "__main__":
    main()
