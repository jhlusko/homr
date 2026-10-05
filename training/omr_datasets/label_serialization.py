"""Versioned opt-in annotation serialization with token-hash binding."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from homr.transformer.vocabulary import EncodedSymbol, serialized_chords
from training.transformer.training_vocabulary import (
    ANNOTATION_POLICIES,
    LEGACY_ANNOTATION_POLICY,
    check_token_lines,
    read_token_lines,
    token_lines_to_str,
)

SCHEMA = "homr.label-serialization.v1"
SUFFIX = ".labels.json"


def validate_label_policy(token_path: str | Path, *, expected_policy: str | None = None) -> str:
    token_path = Path(token_path)
    manifest = Path(str(token_path) + SUFFIX)
    if expected_policy is not None and expected_policy not in ANNOTATION_POLICIES:
        raise ValueError("Unknown expected annotation policy")
    policy = LEGACY_ANNOTATION_POLICY
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        policy = payload.get("annotationPolicy")
        if payload.get("schemaVersion") != SCHEMA or policy not in ANNOTATION_POLICIES:
            raise ValueError("Unsupported label serialization manifest")
        if payload.get("tokenSha256") != hashlib.sha256(token_path.read_bytes()).hexdigest():
            raise ValueError("Label serialization token hash mismatch")
    if expected_policy is not None and policy != expected_policy:
        raise ValueError(f"Annotation policy mismatch: expected {expected_policy}, found {policy}")
    return policy


def write_label_bundle(
    token_path: str | Path, symbols: list[EncodedSymbol], *, annotation_policy: str
) -> dict:
    """Write new labels and existing sidecars; require an explicit policy.

    Refuses existing bundle members and validates vocabulary/identity before writing.
    A manifest binds the annotation semantics to these exact token bytes.
    """
    from training.omr_datasets.notation_sidecar import write_sidecar

    path = Path(token_path)
    members = [
        path,
        Path(str(path) + SUFFIX),
        Path(str(path) + ".notation.json"),
        Path(str(path) + ".timing.json"),
    ]
    if any(p.exists() for p in members):
        raise ValueError("Refusing to overwrite label bundle")
    check_token_lines([symbol for symbol in symbols if symbol.rhythm != "chord"])
    text = token_lines_to_str(list(symbols), annotation_policy=annotation_policy)
    parsed = read_token_lines(text.splitlines())
    check_token_lines(parsed)
    before = [
        (s.rhythm, s.pitch, s.lift, s.position) for c in serialized_chords(list(symbols)) for s in c
    ]
    after = [(s.rhythm, s.pitch, s.lift, s.position) for c in serialized_chords(parsed) for s in c]
    if before != after:
        raise ValueError("Serialization changed symbol identity or count")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schemaVersion": SCHEMA,
        "annotationPolicy": annotation_policy,
        "tokenSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    created = []
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(text, encoding="utf-8")
        write_sidecar(staged, symbols)
        Path(str(staged) + SUFFIX).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        # Publish the token last. A partial bundle fails closed at its manifest,
        # rather than exposing an unmarked v2 token as an old legacy label.
        try:
            for target in [*members[2:], members[1], path]:
                source = Path(directory) / target.name
                if source.exists():
                    os.link(source, target)
                    created.append(target)
        except Exception:
            for target in reversed(created):
                target.unlink()
            raise
    return manifest
