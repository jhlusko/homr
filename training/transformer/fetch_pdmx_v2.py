"""Fetch and verify the public PDMX-v2 release on a training instance.

Archives are downloaded and extracted on the instance, never routed through the
operator's laptop. A verified marker is written only after every shard succeeds.
"""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    temporary = path.with_suffix(path.suffix + ".part")
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "5", "--retry-delay", "3", "--output", str(temporary), url], check=True)
    os.replace(temporary, path)


def shard(entry, root, base):
    marker = root / "verified" / (entry["name"] + ".json")
    if marker.exists():
        return entry["name"], "already verified"
    archive = root / "archives" / entry["name"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = base + "/shards/" + entry["name"]
    if entry["url"] not in (url, base + "/" + entry["name"]):
        raise ValueError(f"Unexpected shard URL: {entry['name']}")
    if not archive.exists() or archive.stat().st_size != entry["bytes"] or sha256(archive) != entry["sha256"]:
        download(url, archive)
    if archive.stat().st_size != entry["bytes"] or sha256(archive) != entry["sha256"]:
        raise ValueError(f"Shard integrity mismatch: {entry['name']}")
    subprocess.run(["tar", "-I", "zstd", "-xf", str(archive), "-C", str(root / "data"), "--no-same-owner"], check=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"name": entry["name"], "bytes": entry["bytes"], "sha256": entry["sha256"]}) + "\n")
    archive.unlink()
    return entry["name"], "verified and extracted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release-url", default="https://pub-d31b33903cd04fec8cc87ee0667c1981.r2.dev/pdmx-homr-v2/release.json")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "data").mkdir(exist_ok=True)
    base = args.release_url.rsplit("/", 1)[0]
    for name in ("release.json", "index_train.txt", "index_valid.txt"):
        path = args.root / name
        if not path.exists():
            download(base + "/" + name, path)
    release = json.loads((args.root / "release.json").read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(shard, entry, args.root, base) for entry in release["shards"]]
        for future in as_completed(futures):
            name, status = future.result()
            print(name, status, flush=True)
    expected = {entry["name"] for entry in release["shards"]}
    found = {path.name.removesuffix(".json") for path in (args.root / "verified").glob("*.json")}
    if expected != found:
        raise ValueError(f"Missing verified shards: {expected - found}")
    (args.root / "READY").write_text(json.dumps({"release_url": args.release_url, "shards": len(expected), "index_train_sha256": sha256(args.root / "index_train.txt")}) + "\n")
    print("READY", flush=True)


if __name__ == "__main__":
    main()
