#!/usr/bin/env bash
# Refuse to start a GPU run from code that is not committed and pushed.
#
# Twice on 2026-09-24 an uncommitted laptop fix never reached the rented instance, and a
# day went into rediscovering it. Source this (or run it) at the top of every instance
# launcher, from the checkout the run will import:
#
#   /path/to/checkout/tools/gpu_preflight.sh [expected-commit]
#
# Fails when the tree has any tracked or untracked change, when HEAD is on no remote
# branch after a fetch, or when HEAD is not the expected commit. On success it prints
# the full commit so the launcher's log records exactly what ran.
#
# The rented instance has no push credentials. There, an unpushed commit may run only if
# PREFLIGHT_BUNDLE names a `git bundle` that contains HEAD and lives under
# /workspace/context/results/ (which the laptop pulls), so the code that ran is
# recoverable and gets pushed from the laptop afterwards:
#
#   git bundle create /workspace/context/results/bundles/$(git rev-parse --short HEAD).bundle \
#     origin/ossq-benchmark..HEAD
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
head=$(git rev-parse HEAD)

if [ -n "$(git status --porcelain)" ]; then
  echo "preflight: refusing to run from a dirty tree:" >&2
  git status --short >&2
  exit 1
fi
if [ $# -ge 1 ] && [ "$head" != "$(git rev-parse --verify "$1^{commit}" 2>/dev/null)" ]; then
  echo "preflight: HEAD $head is not the expected commit $1" >&2
  exit 1
fi
git fetch --quiet origin || echo "preflight: fetch failed; checking cached remote refs" >&2
if [ -n "$(git branch --remotes --contains "$head")" ]; then
  echo "preflight: ok $head"
  exit 0
fi
bundle=${PREFLIGHT_BUNDLE:-}
case "$bundle" in
  /workspace/context/results/*) ;;
  *) echo "preflight: HEAD $head is not on any origin branch; push it, or set" \
       "PREFLIGHT_BUNDLE to a bundle under /workspace/context/results/" >&2
     exit 1 ;;
esac
git bundle verify --quiet "$bundle" >/dev/null
if ! git bundle list-heads "$bundle" | grep -q "^$head "; then
  echo "preflight: bundle $bundle does not contain HEAD $head" >&2
  exit 1
fi
echo "preflight: ok $head (unpushed; recoverable from $bundle)"
