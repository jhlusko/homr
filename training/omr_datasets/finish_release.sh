#!/usr/bin/env bash
# Finish the distribution once every transfer has landed.
#
# Run this after the rsyncs complete. It packages each shipped directory, verifies it,
# and refuses to declare success if anything is still unportable. Safe to re-run: the
# packager is idempotent, and a row it already fixed is left alone.
#
#   bash training/omr_datasets/finish_release.sh ~/workspace/homr-artifacts

set -uo pipefail

ROOT="${1:?usage: finish_release.sh <artifacts-dir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

TARGETS=(
  "$ROOT/corpora/ossq_scanned_corrected"
  "$ROOT/corpora/lieder_scanned_pairs"
  "$ROOT/datasets/ossq_instrumental_text"
  "$ROOT/datasets/lieder_vocal_text"
  "$ROOT/ground_truth/ossq_boxes"
)

# Check first, change nothing. A transfer that is still running looks exactly like a
# corpus with missing files, and packaging a half-arrived tree bakes the gap into the
# index. Better to stop and say so.
echo "=== checking for in-flight transfers ==="
if pgrep -f "rsync.*175\.155\.64\.164" >/dev/null 2>&1; then
  echo "rsync is still running - wait for it to finish, or the manifest will record"
  echo "a partial tree as if it were complete. Aborting."
  exit 2
fi

failed=0
for target in "${TARGETS[@]}"; do
  if [ ! -d "$target" ]; then
    echo "SKIP (absent): $target"
    continue
  fi
  echo
  echo "=== $target ==="
  if ! python3 -m training.omr_datasets.package_dataset --root "$target"; then
    failed=1
  fi
done

echo
if [ "$failed" -ne 0 ]; then
  echo "FAILED - at least one directory is not portable. Nothing above should ship."
  exit 1
fi
echo "All directories packaged and verified portable."
echo "Next: tar each one and publish alongside DATASET_DISTRIBUTION.md."
