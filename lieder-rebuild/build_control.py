"""Matched control for the naturals probe: identical 3880 pairs, naturals stripped.

Confounds the naturals experiment could not rule out on its own: does restoring naturals
specifically cause the PDMX regression, or would any 12-epoch fine-tune on this exact
page/system/voice selection regress PDMX by roughly the same amount? Every other PDMX
score in this project comes from an unrelated checkpoint lineage (450, 456, 459, 447,
448, 426 - none is a 464-based, same-corpus, same-epoch fine-tune), so none of them
answers this. Stripping naturals from the ALREADY-BUILT naturals corpus keeps every
other variable - which pages, which systems, corpus size, split - fixed by construction.
"""
import sys
sys.path.insert(0, "/workspace/b0/homr")
from pathlib import Path
from homr.circle_of_fifths import strip_naturals
from training.transformer.training_vocabulary import read_tokens, token_lines_to_str

SRC_INDEX = Path("/workspace/b0/imslp_train_index_naturals.txt")
OUT_DIR = Path("/workspace/b0/olimpic-probe/stage2_pairs_naturals_control")
OUT_INDEX = Path("/workspace/b0/imslp_train_index_naturals_control.txt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_lines = []
for line in SRC_INDEX.read_text().splitlines():
    if not line.strip():
        continue
    png, tokens_path = line.split(",", 1)
    symbols = read_tokens(tokens_path)
    stripped = strip_naturals(symbols)
    out_tokens = OUT_DIR / Path(tokens_path).name
    out_tokens.write_text(token_lines_to_str(stripped), encoding="utf-8")
    out_lines.append(f"{png},{out_tokens}")

OUT_INDEX.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print(f"wrote {len(out_lines)} pairs to {OUT_INDEX}")

n_before = sum(1 for line in SRC_INDEX.read_text().splitlines() if line.strip()
               for s in read_tokens(line.split(",", 1)[1]) if s.lift == "N")
print(f"N lifts before strip: {n_before}")
