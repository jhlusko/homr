#!/usr/bin/env bash
# Download and convert the four corpora this box lacks, WITH NATURALS KEPT.
#
# Preparing for a from-scratch run that fixes the naturals inconsistency properly. Four
# of five converters call strip_naturals and convert_ossq does not, so no training corpus
# contains a natural, no checkpoint has ever predicted one, and OSSQ alone is scored
# against a symbol the pipeline discards. Fixing it in one corpus would teach the model
# two conventions for the same mark, so every corpus has to be rebuilt.
#
# HOMR_KEEP_NATURALS=1 is exported for the whole run: the switch lives inside
# strip_naturals, so it reaches all 13 call sites at once and none can drift.
#
# Sources: grandstaff (grfia.dlsi.ua.es, ~tgz), Camera-PrIMuS (grfia.dlsi.ua.es, ~tgz),
# MuseTrainer (github zip), Lieder (needs a MuseScore AppImage to render).
#
# Each corpus is converted independently and a failure in one must not abort the others -
# PrIMuS and GrandStaff are old academic hosts and may be slow or down.
set -uo pipefail
cd /workspace/b0/homr
export HOMR_KEEP_NATURALS=1
export HOMR_MAX_TUPLET_RATIO=0.95
echo "=== fetch start $(date +%H:%M:%S), $(df --output=avail -BG /workspace | tail -1) free ==="

convert () {
  local NAME="$1" MOD="$2" FN="$3"
  echo "=== $NAME start $(date +%H:%M:%S) ==="
  if .venv/bin/python -c "
from $MOD import $FN
$FN()
" > "/workspace/b0/lieder-rebuild/fetch_$NAME.log" 2>&1; then
    echo "=== $NAME OK $(date +%H:%M:%S)  $(du -sh datasets/$NAME 2>/dev/null | cut -f1) ==="
  else
    echo "=== $NAME FAILED $(date +%H:%M:%S) - see fetch_$NAME.log ==="
    tail -5 "/workspace/b0/lieder-rebuild/fetch_$NAME.log" | sed 's/^/    /'
  fi
  df --output=avail -BG /workspace | tail -1 | sed 's/^/    free: /'
}

convert musetrainer training.omr_datasets.convert_musetrainer convert_musetrainer
convert grandstaff training.omr_datasets.convert_grandstaff convert_grandstaff
convert primus     training.omr_datasets.convert_primus     convert_primus_dataset
convert lieder     training.omr_datasets.convert_lieder      convert_lieder

echo "=== all indexes ==="
for d in datasets/*/; do
  for f in index.txt index_train.txt; do
    [ -f "$d$f" ] && echo "  $d$f: $(wc -l < "$d$f") rows"
  done
done
echo "=== FETCH DONE $(date +%H:%M:%S) ==="
