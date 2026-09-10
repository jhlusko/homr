"""Do the running job's labels change under the new code?

The seven trained heads read beam levels, stem and slur slots. Ties are a new field with
no head, so the question is whether anything those heads consume has moved.
"""
import json
from pathlib import Path

from homr.transformer.structured_notation import TieState
from training.architecture.transformer.structured_targets import build_targets, notation_positions
from training.omr_datasets.notation_sidecar import attach_sidecar, sidecar_path
from training.transformer.training_vocabulary import read_tokens

lines = [l for l in Path("/workspace/b0/phase2/valid/index.txt").read_text().splitlines() if l.strip()][:200]
schemas, ties, ok = set(), set(), 0
for line in lines:
    tokens = line.split(",")[1]
    schemas.add(json.loads(sidecar_path(tokens).read_text())["schemaVersion"])
    symbols = read_tokens(tokens)
    if not attach_sidecar(tokens, symbols):
        continue
    ties.update(s.notation.tie for s in symbols if s.notation is not None)
    built = build_targets([notation_positions(symbols, 608)], 4, 2)
    ok += all(v.shape == (1, 608) for v in built.values())
print("sidecar schemas on disk:", schemas)
print("tie values decoded:", {str(t) for t in ties})
print(f"examples whose targets still build: {ok}/{len(lines)}")
print("target keys:", sorted(build_targets([[None]], 4, 2)))
