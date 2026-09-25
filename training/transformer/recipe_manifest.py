"""Record which corpus indexes and code version a training run actually used.

E1 (reproducibility, `OTS_HOMR_RELEASE_SEQUENCE_2026-09-25.md`) needs to state exactly
what inputs a checkpoint was built from - not what the launcher's defaults claim, since
a caller can override any index path on the command line. Digesting the index *file*
(not the corpus it lists) is deliberate: the index is the caller-visible contract, and a
mismatched index catches a stale checkout of a corpus repo the same way a mismatched
corpus itself would, without hashing gigabytes of images/audio on every run.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def _sha256_of_file(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_commit(git_root: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", git_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_recipe_manifest(
    checkpoint_dir: str,
    git_root: str,
    indexes: dict[str, str],
    extra: dict | None = None,
) -> str:
    """Write `recipe_manifest.json` into `checkpoint_dir`. Returns the path written.

    `indexes` maps a human-readable role ("ossq", "lieder_train", ...) to the index
    file path actually passed to `train_transformer`. Each entry records the resolved
    path and that file's own SHA-256, so a later comparison can tell whether the same
    bytes were used even if the path or corpus repo moved.
    """
    manifest = {
        "commit": _current_commit(git_root),
        "indexes": {
            role: {"path": path, "sha256": _sha256_of_file(path)}
            for role, path in indexes.items()
        },
    }
    if extra:
        manifest.update(extra)
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recipe_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return str(out_path)
