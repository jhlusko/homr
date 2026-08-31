#!/usr/bin/env bash
# Prepare deterministic source bundles for CDN upload. Does not upload anything.
#
# Required environment:
#   OSSQ_SOURCE       OpenScore StringQuartets checkout
#   LIEDER_SOURCE     OpenScore Lieder checkout
#   LIEDER_PDFS       directory holding IMSLP<n>.pdf source files
#   LIEDER_MANIFESTS  directory holding the frozen v4 manifests/audit outputs
#   OUT               empty/new output directory
#
# The Lieder all-source bundle is private. The public-candidate bundle contains only
# the 25 exact sources from imslp_release_candidate_v4_score_ids.txt.
set -euo pipefail

: "${OSSQ_SOURCE:?}"
: "${LIEDER_SOURCE:?}"
: "${LIEDER_PDFS:?}"
: "${LIEDER_MANIFESTS:?}"
: "${OUT:?}"

for tool in tar zstd sha256sum python3; do command -v "$tool" >/dev/null; done
for path in "$OSSQ_SOURCE" "$LIEDER_SOURCE" "$LIEDER_PDFS" "$LIEDER_MANIFESTS"; do
  test -e "$path" || { echo "missing: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "OUT must not already exist: $OUT" >&2; exit 2; }

mkdir -p "$OUT/staging" "$OUT/archives"
stage="$OUT/staging"
archives="$OUT/archives"
fixed_tar=(tar -h --sort=name --mtime='UTC 2026-08-31' --owner=0 --group=0 --numeric-owner)

make_archive() {
  local name="$1"; shift
  "${fixed_tar[@]}" -C "$stage" -cf - "$@" | zstd -T0 -19 -o "$archives/$name.tar.zst"
}

# OSSQ includes the complete frozen OpenScore source tree (scores, PDFs, and metadata).
ln -s "$OSSQ_SOURCE" "$stage/ossq-source"
make_archive ossq-source-v1 ossq-source
rm "$stage/ossq-source"

# Full Lieder bundle: OpenScore source tree + only source PDFs represented by the v4
# corpus, never the wider unreviewed PDF cache.
mkdir -p "$stage/lieder-all-source/pdfs" "$stage/lieder-all-source/manifests"
ln -s "$LIEDER_SOURCE" "$stage/lieder-all-source/openscore-lieder"
python3 - "$LIEDER_MANIFESTS" "$LIEDER_PDFS" "$stage/lieder-all-source/pdfs" <<'PY'
from pathlib import Path
import re, shutil, sys
manifests, pdf_root, out = map(Path, sys.argv[1:])
ids = set()
for name in ('imslp_train_index_v4_boundary_safe.txt', 'imslp_val_index_v4_boundary_safe.txt'):
    ids.update(re.findall(r'IMSLP\d+', (manifests / name).read_text()))
for score_id in sorted(ids):
    source = pdf_root / f'{score_id}.pdf'
    if not source.is_file():
        raise SystemExit(f'missing source PDF: {source}')
    shutil.copy2(source, out / source.name)
print(f'copied {len(ids)} all-source PDFs')
PY
cp "$LIEDER_MANIFESTS"/imslp_{train,val}_index_v4_boundary_safe.txt "$stage/lieder-all-source/manifests/"
cp "$LIEDER_MANIFESTS"/imslp_rights_audit_v4.csv "$stage/lieder-all-source/manifests/"
make_archive lieder-v4-all-sources lieder-all-source
rm -rf "$stage/lieder-all-source"

# Conservative public candidate: only the 25 auditable IMSLP scans and their matching
# OpenScore score files, scores.yaml, and frozen filtered indexes.
mkdir -p "$stage/lieder-v4-pd25-source/pdfs" "$stage/lieder-v4-pd25-source/openscore" "$stage/lieder-v4-pd25-source/manifests"
python3 - "$LIEDER_MANIFESTS" "$LIEDER_PDFS" "$LIEDER_SOURCE" "$stage/lieder-v4-pd25-source" <<'PY'
from pathlib import Path
import re, shutil, sys, yaml
manifests, pdf_root, lieder_root, out = map(Path, sys.argv[1:])
ids = {line.strip() for line in (manifests / 'imslp_release_candidate_v4_score_ids.txt').read_text().splitlines() if line.startswith('IMSLP')}
for score_id in sorted(ids):
    source = pdf_root / f'{score_id}.pdf'
    if not source.is_file(): raise SystemExit(f'missing source PDF: {source}')
    shutil.copy2(source, out / 'pdfs' / source.name)
scores_yaml = lieder_root / 'data' / 'scores.yaml'
scores = yaml.safe_load(scores_yaml.read_text())
keys = {str(key) for key, entry in scores.items() if f"IMSLP{str(entry.get('imslp','')).lstrip('#')}" in ids}
shutil.copy2(scores_yaml, out / 'openscore' / 'scores.yaml')
for key in keys:
    found = list((lieder_root / 'scores').glob(f'**/lc{key}.*'))
    if not found: raise SystemExit(f'missing OpenScore files for {key}')
    for source in found:
        destination = out / 'openscore' / source.relative_to(lieder_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
print(f'copied {len(ids)} public-candidate PDFs and {len(keys)} OpenScore entries')
PY
cp "$LIEDER_MANIFESTS"/imslp_release_candidate_v4_score_ids.txt "$stage/lieder-v4-pd25-source/manifests/"
cp "$LIEDER_MANIFESTS"/imslp_{train,val}_index_v4_release_candidate_pd.txt "$stage/lieder-v4-pd25-source/manifests/"
cp "$LIEDER_MANIFESTS"/imslp_rights_audit_v4.csv "$stage/lieder-v4-pd25-source/manifests/"
make_archive lieder-v4-pd25-sources lieder-v4-pd25-source
rm -rf "$stage/lieder-v4-pd25-source"

(cd "$archives" && sha256sum *.tar.zst > SHA256SUMS)
python3 - "$archives" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
files = []
for path in sorted(root.glob('*.tar.zst')):
    files.append({'path': path.name, 'bytes': path.stat().st_size,
                  'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()})
(root / 'release.json').write_text(json.dumps({'version': '2026-08-31', 'archives': files}, indent=2) + '\n')
PY
echo "Prepared archives in $archives"
