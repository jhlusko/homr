"""Replace only the PDMX rows of the pinned joint-training index with v2."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-index", type=Path, required=True)
    p.add_argument("--v2-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--v2-prefix", default="datasets/pdmx-homr-v2")
    a = p.parse_args()
    expected_old = "f8c7f272b0fddd2397c5db169698a28e1ebaca82f6cd7a8a2baba5e7441202ca"
    expected_v2 = "de41b00b4c8c0efb0cd0fb4aff6b76ee39607d9ef284b97687ae30ff8f2102b2"
    if digest(a.old_index) != expected_old or digest(a.v2_root / "index_train.txt") != expected_v2:
        raise ValueError("Source index digest differs from the audited corpus")
    if not (a.v2_root / "READY").exists():
        raise ValueError("PDMX-v2 shards are not verified and extracted")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with a.output.open("w") as target:
        for line in a.old_index.read_text().splitlines():
            if "/scanned/" in line:
                target.write(line + "\n"); counts["ossq"] += 1
            elif "/pairs/" in line:
                target.write(line + "\n"); counts["lieder"] += 1
        for line in (a.v2_root / "index_train.txt").read_text().splitlines():
            image, token = line.split(",")
            target.write(f"{a.v2_prefix}/{image},{a.v2_prefix}/{token}\n")
            counts["pdmx"] += 1
    if counts != {"ossq": 31496, "lieder": 3093, "pdmx": 565522}:
        raise ValueError(f"Unexpected corpus counts: {counts}")
    a.output.with_suffix(".identity.json").write_text(json.dumps({
        "old_index_sha256": expected_old, "v2_index_sha256": expected_v2,
        "new_index_sha256": digest(a.output), "counts": dict(counts),
        "v2_release_sha256": digest(a.v2_root / "release.json"),
        "sampling": "60% OSSQ, 25% Lieder, 15% PDMX-v2; 40290 crop draws per epoch",
    }, indent=2) + "\n")
    print(json.dumps({"index": str(a.output), "sha256": digest(a.output), "counts": dict(counts)}))


if __name__ == "__main__":
    main()
