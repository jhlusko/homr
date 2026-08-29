#!/usr/bin/env bash
# A/B the end-on-a-divider decode constraint on checkpoint 456.
#
# The post-hoc simulation halved bar-count mismatches on the dense cut while token
# accuracy moved -0.03pp. That was an upper bound with a divider bolted on; this is the
# real thing, where suppressing EOS lets the model emit the divider itself and continue
# properly. It could do better than the simulation or worse - worse if the model, denied
# EOS, produces junk instead of a barline, which is what the cap exists to bound.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
CK="$A/pytorch_model_456-c4bc89945f5cbe8f8edb1581ec3322b60dbda0cb.pth"
cd /workspace/b0/homr
while pgrep -f 'train_lieder_only|base_predictions' >/dev/null; do sleep 60; done
sleep 20
for MODE in off on; do
  python3 - "$MODE" <<'PY'
import re, sys, pathlib
p = pathlib.Path("/workspace/b0/homr/homr/transformer/configs.py")
s = p.read_text()
want = "True" if sys.argv[1] == "on" else "False"
s = re.sub(r"self\.enforce_final_divider = \w+", f"self.enforce_final_divider = {want}", s)
p.write_text(s)
print(f"enforce_final_divider = {want}")
PY
  echo "=== scoring with constraint $MODE $(date +%H:%M:%S) ==="
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_456_divider_$MODE.jsonl" --checkpoint "$CK" \
    > "$R/general_456_divider_$MODE.log" 2>&1
  echo "=== scored $MODE $(date +%H:%M:%S) ==="
done
# Leave it enabled, which is the default.
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("/workspace/b0/homr/homr/transformer/configs.py")
p.write_text(re.sub(r"self\.enforce_final_divider = \w+",
                    "self.enforce_final_divider = True", p.read_text()))
PY
echo "=== DIVIDER AB DONE $(date +%H:%M:%S) ==="
