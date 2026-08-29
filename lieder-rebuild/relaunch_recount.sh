#!/usr/bin/env bash
# Relaunch the Lieder physical-bar recount at SAFE concurrency.
#
# Why 4 and not 12: on 2026-08-27 all 12 shards were launched in the same second.
# Each ONNX Runtime process holds 206-307 threads (sized from nproc=128) and the
# container's cgroup pids.max is 3840.  The startup burst hit that ceiling,
# pthread_create returned EAGAIN for 130 of 330 scores, and every surviving worker
# then DEADLOCKED - all threads sleeping, main thread in futex_wait_queue, zero CPU
# ticks - waiting on pool workers that were never created.  Nothing was produced.
#
# 4 workers x ~310 threads = ~1240, comfortably under 3840, and the 45s stagger
# keeps the session initialisations from coinciding even at that width.
set -euo pipefail

REBUILD=/workspace/b0/lieder-rebuild
HOMR=/workspace/b0/homr
NSHARDS=4
STAGGER=45

cd "$REBUILD"

if pgrep -f 'compare_bar_counts.*score_ids_' >/dev/null; then
  echo "ERROR: compare_bar_counts is still running - kill it first, by PID." >&2
  exit 1
fi

# Nothing from the 07:48 run survived; the two rows files it left are empty ("[]").
# Move them aside rather than delete, so the merge glob cannot pick them up.
mkdir -p failed_run_20260827_0748
for f in bar_count_rows_v2_[0-9][0-9].json bar_count_v2_[0-9][0-9].log; do
  [ -e "$f" ] && mv "$f" failed_run_20260827_0748/
done

cat score_ids_shard_* | sort -u > ids_all
total=$(wc -l < ids_all)
echo "relaunching $total scores across $NSHARDS workers"

if [ ! -e "ids_v3_shard_00" ]; then
  split -n "r/$NSHARDS" -d -a 2 ids_all ids_v3_shard_
fi

# `seq -w 0 3` pads to the width of the largest value - one digit - so it yields
# "0 1 2 3", not "00 01 02 03", while `split -a 2` writes two-digit suffixes.
# Format the index explicitly instead.
for k in $(seq 0 $((NSHARDS - 1))); do
  i=$(printf '%02d' "$k")
  n=$(wc -l < "ids_v3_shard_$i")
  echo "launching shard $i ($n scores)"
  ( cd "$HOMR" && setsid nohup .venv/bin/python -m training.omr_datasets.compare_bar_counts \
      --ground-truth /workspace/b0/olimpic-probe/imslp_lieder_ground_truth \
      --systems /workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes \
      --pngs /workspace/b0/olimpic-probe/imslp_pngs /workspace/b0/olimpic-probe/imslp_pngs_new \
      --score-ids "$REBUILD/ids_v3_shard_$i" \
      --rows-out "$REBUILD/bar_count_rows_v3_$i.json" \
      --failed-out "$REBUILD/bar_count_failed_v3_$i.json" \
      > "$REBUILD/bar_count_v3_$i.log" 2>&1 & )
  if [ "$k" -lt $((NSHARDS - 1)) ]; then sleep "$STAGGER"; fi
done

sleep 20
echo "--- workers: $(pgrep -fc 'compare_bar_counts.*ids_v3_shard_') ---"
echo "--- threads: $(ps -eLf | wc -l) / pids.max $(cat /sys/fs/cgroup/pids.max) ---"
