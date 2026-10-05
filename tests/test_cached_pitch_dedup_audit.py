import json

import pytest

from training.transformer.audit_cached_pitch_dedup import load, load_manifest


def test_rejects_duplicate_cache_rows(tmp_path):
    path = tmp_path / "cache.jsonl"
    row = json.dumps({"id": "crop", "status": "ok"}) + "\n"
    path.write_text(row + row)
    with pytest.raises(ValueError, match="Duplicate or failed"):
        load(path)


def test_rejects_failed_cache_rows(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"id": "crop", "status": "failed"}) + "\n")
    with pytest.raises(ValueError, match="Duplicate or failed"):
        load(path)


def test_rejects_duplicate_manifest_ids(tmp_path):
    path = tmp_path / "manifest.jsonl"
    row = json.dumps({"id": "crop"}) + "\n"
    path.write_text(row + row)
    with pytest.raises(ValueError, match="Duplicate manifest"):
        load_manifest(path)
