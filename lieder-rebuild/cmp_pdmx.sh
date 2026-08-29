#!/usr/bin/env bash
# Runs ON the instance under setsid: the previous waiter held an ssh connection open
# for 40 minutes and died with it (exit 255, broken pipe) while the scoring it was
# waiting on carried on fine. Nothing that has to outlive a network hiccup should be
# tied to the client connection.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
while pgrep -f base_predictions >/dev/null; do sleep 30; done
.venv/bin/python -m training.transformer.compare_checkpoints \
  --name "PDMX held-out (public, split predates this work)" \
  --run 426="$R/pdmx_old.jsonl" --run 447="$R/pdmx_mid.jsonl" --run 448="$R/pdmx_new.jsonl" \
  --out "$R/cmp_pdmx.json"
echo "=== 448 vs 447 ==="
.venv/bin/python -m training.transformer.compare_checkpoints \
  --name "PDMX, 448 against 447" \
  --run 447="$R/pdmx_mid.jsonl" --run 448="$R/pdmx_new.jsonl" \
  --out "$R/cmp_pdmx_448v447.json"
echo "=== PDMX COMPARISON DONE ==="
