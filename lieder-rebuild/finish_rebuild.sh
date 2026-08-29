#!/usr/bin/env bash
# Finish the Lieder alignment rebuild: merge the recount, GATE ON COVERAGE, then
# align -> build clean pairs -> audit -> split -> 100-item review set.
#
# The coverage gate is the point of this script.  `align_lieder_systems` builds its
# per-score map purely from the rows it is given, so a score missing from the rows
# file simply never appears in the output - it is not an error, it is invisible.
# The 07:48 run lost 39% of the corpus that way and would have produced an
# under-covered alignment that reads as conservatism rather than as lost data.
set -euo pipefail
trap 'code=$?; [ $code -ne 0 ] && echo "!!! finish_rebuild.sh ABORTED at line $LINENO (exit $code)" >&2; exit $code' EXIT

REBUILD=/workspace/b0/lieder-rebuild
PROBE=/workspace/b0/olimpic-probe
cd /workspace/b0/homr

echo "=== merging v3 shards ==="
jq -s 'add' "$REBUILD"/bar_count_rows_v3_0[0-9].json > "$REBUILD/bar_count_rows_v2.json"

echo "=== coverage gate ==="
jq -r '.[].score_id' "$REBUILD/bar_count_rows_v2.json" | sort -u > "$REBUILD/ids_final"
# Scores knowingly and reproducibly outside this run, one id per line with a reason
# comment.  Anything NOT listed here is an unexplained loss and stops the build.
touch "$REBUILD/permanent_exclusions.txt"
# grep exits 1 when nothing matches, and an exclusions file that is empty or all
# comments is the GOOD state - with `set -e -o pipefail` that killed the whole run
# silently at the gate on 2026-08-27, after coverage had come back 330/330.
{ grep -vE '^\s*(#|$)' "$REBUILD/permanent_exclusions.txt" || true; } | sort -u > "$REBUILD/.excl"
comm -23 "$REBUILD/ids_all" "$REBUILD/ids_final" > "$REBUILD/.gap" || true
missing=$(comm -23 "$REBUILD/.gap" "$REBUILD/.excl" || true)
excluded=$(comm -12 "$REBUILD/.gap" "$REBUILD/.excl" || true)
[ -n "$excluded" ] && echo "known exclusions honoured: $(echo "$excluded" | wc -l)"
if [ -n "$missing" ]; then
  echo "MISSING $(echo "$missing" | wc -l) of $(wc -l < "$REBUILD/ids_all") scores:" >&2
  echo "$missing" >&2
  echo >&2
  echo "Recorded failures:" >&2
  jq -s 'add | map({score_id, reason: (.reason[0:80])})' \
     "$REBUILD"/bar_count_failed_v3_0[0-9].json 2>/dev/null >&2 || true
  echo >&2
  echo "Refusing to build an under-covered alignment. Rerun the missing ids, or" >&2
  echo "add them to permanent_exclusions.txt with a reason." >&2
  [ "${1:-}" = "--allow-missing" ] || exit 1
  echo "--allow-missing given; continuing with a KNOWN-INCOMPLETE corpus." >&2
else
  echo "COVERAGE OK: $(wc -l < "$REBUILD/ids_final") / $(wc -l < "$REBUILD/ids_all") scores" \
       "($(wc -l < "$REBUILD/.excl") known exclusions)"
fi

echo "=== align (model-independent, min-margin 2.0) ==="
.venv/bin/python -m training.omr_datasets.align_lieder_systems \
  --rows "$REBUILD/bar_count_rows_v2.json" \
  --ground-truth "$PROBE/imslp_lieder_ground_truth" \
  --out "$REBUILD/system_alignment_v2.json"

echo "=== build clean pairs (quarantining recovered) ==="
.venv/bin/python -m training.omr_datasets.build_clean_stage2_pairs \
  --alignment "$REBUILD/system_alignment_v2.json" \
  --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
  --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
  --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
  --systems "$PROBE/imslp_systems_with_staff_boxes" \
  --pngs "$PROBE/imslp_pngs" "$PROBE/imslp_pngs_new" \
  --out "$PROBE/stage2_pairs_clean" \
  --manifest "$REBUILD/stage2_clean_manifest.txt" \
  --report "$REBUILD/stage2_clean_build_report.json" \
  --recovered-manifest "$PROBE/stage2_recovered_manifest.txt" \
  --quarantine-manifest "$REBUILD/stage2_recovered_QUARANTINED.txt" \
  --quarantine-report "$REBUILD/stage2_recovered_quarantine.json"

echo "=== audit ==="
.venv/bin/python -m training.omr_datasets.audit_clean_stage2_pairs \
  --manifest "$REBUILD/stage2_clean_manifest.txt" \
  --alignment "$REBUILD/system_alignment_v2.json" \
  --recovered-manifest "$PROBE/stage2_recovered_manifest.txt" \
  --out "$REBUILD/stage2_clean_audit.json"

echo "=== score-disjoint split ==="
.venv/bin/python -m training.omr_datasets.split_pairs_by_score \
  --manifest "$REBUILD/stage2_clean_manifest.txt" \
  --train-out "$REBUILD/stage2_clean_train_manifest.txt" \
  --val-out "$REBUILD/stage2_clean_val_candidate_manifest.txt" \
  --val-fraction 0.1

echo "=== 100-item review set ==="
.venv/bin/python -m training.omr_datasets.make_alignment_review \
  --manifest "$REBUILD/stage2_clean_val_candidate_manifest.txt" \
  --alignment "$REBUILD/system_alignment_v2.json" \
  --old-manifest "$PROBE/stage2_pairs_manifest.txt" \
                 "$PROBE/stage2_recovered_manifest.txt" \
  --out "$REBUILD/review_alignment" \
  --limit 100

echo
echo "=== summary ==="
echo "clean pairs:      $(wc -l < "$REBUILD/stage2_clean_manifest.txt")"
echo "train:            $(wc -l < "$REBUILD/stage2_clean_train_manifest.txt")"
echo "val candidates:   $(wc -l < "$REBUILD/stage2_clean_val_candidate_manifest.txt")"
echo "quarantined:      $(wc -l < "$REBUILD/stage2_recovered_QUARANTINED.txt")"
echo "review items:     $(ls "$REBUILD/review_alignment" | wc -l)"
