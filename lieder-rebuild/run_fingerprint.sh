#!/usr/bin/env bash
# Full fingerprint recovery pass over every system measure-count alignment could not
# place.  4 workers, 45s stagger - the same limits the 2026-08-27 pid-ceiling
# incident established (each ONNX process holds ~300 threads; cgroup pids.max 3840).
set -euo pipefail
trap 'code=$?; [ $code -ne 0 ] && echo "!!! ABORTED line $LINENO (exit $code)" >&2; exit $code' EXIT

REBUILD=/workspace/b0/lieder-rebuild
HOMR=/workspace/b0/homr
N=4
cd "$REBUILD"

if pgrep -f 'recover_by_fingerprint.*fp_shard_' >/dev/null; then
  echo "already running - kill by PID first" >&2; exit 1
fi

"$HOMR/.venv/bin/python" - <<'PY' > fp_score_ids
import json
a = json.load(open("/workspace/b0/lieder-rebuild/system_alignment_v2.json"))
for sid, sc in sorted(a["scores"].items()):
    for item in sc.get("systems", []):
        st = item.get("status")
        if st in ("ambiguous", "count_mismatch") or (st == "skipped" and item.get("detected_measures")):
            print(sid)
            break
PY
echo "scores with recoverable systems: $(wc -l < fp_score_ids)"
split -n "r/$N" -d -a 2 fp_score_ids fp_shard_

for k in $(seq 0 $((N - 1))); do
  i=$(printf '%02d' "$k")
  echo "launching fp shard $i ($(wc -l < "fp_shard_$i") scores)"
  ( cd "$HOMR" && setsid nohup .venv/bin/python -m training.omr_datasets.recover_by_fingerprint \
      --alignment "$REBUILD/system_alignment_v2.json" \
      --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
      --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
      --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
      --systems /workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes \
      --pngs /workspace/b0/olimpic-probe/imslp_pngs /workspace/b0/olimpic-probe/imslp_pngs_new \
      --clean-manifest "$REBUILD/stage2_clean_manifest.txt" \
      --score-ids "$REBUILD/fp_shard_$i" \
      --out /workspace/b0/olimpic-probe/stage2_pairs_fingerprint \
      --manifest "$REBUILD/stage2_fingerprint_manifest_$i.txt" \
      --report "$REBUILD/stage2_fingerprint_report_$i.json" \
      > "$REBUILD/fingerprint_$i.log" 2>&1 & )
  # NB: `$((N - 1) )` with a stray space is not arithmetic expansion - bash reads
  # `$((` ... `)` as a subshell, prints "N: command not found", and the stagger
  # silently never happens.  All four workers then start in the same second, which
  # is exactly what exhausted the pid ceiling on 2026-08-27.
  if [ "$k" -lt "$((N - 1))" ]; then sleep 45; fi
done
sleep 20
echo "workers: $(pgrep -fc 'recover_by_fingerprint.*fp_shard_' || true)"
echo "threads: $(ps -eLf | wc -l) / $(cat /sys/fs/cgroup/pids.max)"
