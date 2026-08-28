#!/usr/bin/env bash
# Fetch the renderer review.html engraves with.
#
# Verovio turns a MusicXML label into SVG with one <g class="measure"> per measure, so
# the bar a review item is about can be shaded directly. The player cannot do that: it
# renders a score but exposes no handle on an individual measure, which left the
# overfull sets saying "look at bar 5" and the reviewer counting barlines.
#
# Not committed, for the same reason score-editor/ is not: it is a 7.3MB build.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p vendor
VERSION=6.3.0
URL="https://cdn.jsdelivr.net/npm/verovio@${VERSION}/dist/verovio-toolkit-wasm.js"
echo "fetching verovio ${VERSION}"
curl -sSL --fail -o vendor/verovio-toolkit-wasm.js "$URL"
# A truncated or error-page download would fail at runtime as a blank pane rather than
# as a missing file, so check it is what it claims before leaving it in place.
if ! head -c 400 vendor/verovio-toolkit-wasm.js | grep -q "verovio\|Module"; then
  rm -f vendor/verovio-toolkit-wasm.js
  echo "downloaded file does not look like the verovio toolkit" >&2
  exit 1
fi
ls -la vendor/verovio-toolkit-wasm.js
echo "done - now: python3 -m http.server 8899, then open http://127.0.0.1:8899/review.html"
