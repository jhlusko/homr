#!/usr/bin/env bash
# Reverse fingerprinting over the whole corpus: assign every labelled bar to the
# system crop that contains it.  4 workers, staggered - each ONNX process holds
# ~300 threads against a cgroup pids.max of 3840.
set -euo pipefail
trap 'code=$?; [ $code -ne 0 ] && echo "!!! ABORTED (exit $code)" >&2; exit $code' EXIT

REBUILD=/workspace/b0/lieder-rebuild
HOMR=/workspace/b0/homr
N=4
cd "$REBUILD"

if ps -eo cmd | grep -q "[r]everse_fingerprint.*rev_shard_"; then
  echo "already running - kill by PID first" >&2; exit 1
fi

"$HOMR/.venv/bin/python" -c "
import json
a = json.load(open('$REBUILD/system_alignment_v2.json'))
print('\n'.join(sorted(a['scores'])))
" > rev_score_ids
echo "scores: $(wc -l < rev_score_ids)"
rm -f rev_shard_[0-9][0-9]
split -n "r/$N" -d -a 2 rev_score_ids rev_shard_

for k in $(seq 0 $((N - 1))); do
  i=$(printf '%02d' "$k")
  echo "launching reverse shard $i ($(wc -l < "rev_shard_$i") scores)"
  ( cd "$HOMR" && setsid nohup .venv/bin/python -m training.omr_datasets.reverse_fingerprint \
      --alignment "$REBUILD/system_alignment_v2.json" \
      --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
      --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
      --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
      --systems /workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes \
      --pngs /workspace/b0/olimpic-probe/imslp_pngs /workspace/b0/olimpic-probe/imslp_pngs_new \
      --score-ids "$REBUILD/rev_shard_$i" \
      --prediction-cache "$REBUILD/crop_readings_v3_$i.json" \
      --out /workspace/b0/olimpic-probe/stage2_pairs_reverse_v3 \
      --manifest "$REBUILD/stage2_reverse_manifest_v3_$i.txt" \
      --report "$REBUILD/stage2_reverse_report_v3_$i.json" \
      > "$REBUILD/reverse_v3_$i.log" 2>&1 & )
  if [ "$k" -lt "$((N - 1))" ]; then sleep 45; fi
done
sleep 20
echo "workers: $(ps -eo cmd | grep -c '[r]everse_fingerprint.*rev_shard_')"
echo "threads: $(ps -eLf | wc -l) / $(head -1 /sys/fs/cgroup/pids.max)"
