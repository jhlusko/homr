"""Build a review of audited exclusions, preserving the original rendered crops."""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

from training.transformer.synthetic_transfer_review import locations
from training.transformer.training_vocabulary import read_tokens

NAMES = {'accent': 'accent', 'arpeggiate': 'arpeggio', 'fermata': 'fermata',
         'staccato': 'staccato (dot)', 'staccatissimo': 'staccatissimo (wedge)',
         'tenuto': 'tenuto (line)', 'trill': 'trill'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assets = args.output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    cards, records, seen = [], [], {}
    for n, line in enumerate(args.audit.read_text().splitlines(), 1):
        row = json.loads(line)
        relative = Path('out') / row['tokens'].split('/out/', 1)[1]
        token = args.corpus / relative
        symbols = read_tokens(str(token))
        positions = locations(symbols)
        unsupported = {value for branch, value in row['errors'] if branch == 'articulation'}
        matches = [{'location': positions.get(i, 'Unmapped position'), 'staff': s.position,
                    'articulation': s.articulation}
                   for i, s in enumerate(symbols) if s.articulation in unsupported]
        assert matches, f'No matching unsupported symbol: {token}'
        prefix = f'case-{n}'
        for suffix in ('.jpg', '.tokens', '.tokens.notation.json'):
            source = token.with_suffix(suffix)
            shutil.copy2(source, assets / (prefix + suffix))
        digest = hashlib.sha256(token.read_bytes()).hexdigest()
        duplicate = seen.get(digest)
        seen.setdefault(digest, n)
        description = '; '.join(' + '.join(NAMES.get(part, part) for part in value.split('_'))
                                for value in sorted(unsupported))
        labels = ''.join(f'<li>Crop-local {html.escape(m["location"])}; '
                         f'{html.escape(m["staff"])} staff. Combined label: '
                         f'<code>{html.escape(m["articulation"])}</code>.</li>' for m in matches)
        duplicate_note = (f'<p>Token content is identical to case {duplicate}; the original crop '
                          'is shown separately.</p>' if duplicate else '')
        cards.append(f'''<article id="case-{n}"><h2>{n}. {html.escape(description)}</h2>
<p class="id">{html.escape(token.stem)}</p><ul>{labels}</ul>{duplicate_note}
<p>Click the original image to zoom. Locations refer to measures within this crop;
printed score measure numbers may differ. The named pitch carries the pooled label,
which may describe several simultaneous notes.</p>
<a href="assets/{prefix}.jpg" target="_blank"><img src="assets/{prefix}.jpg"
alt="Original score crop, case {n}" loading="lazy"></a>
<details><summary>Original labels and downloads</summary><pre>{html.escape(token.read_text())}</pre>
<a href="assets/{prefix}.tokens">Tokens</a> ·
<a href="assets/{prefix}.tokens.notation.json">Notation sidecar</a></details></article>''')
        records.append(dict(case=n, crop=str(relative), unsupported=sorted(unsupported),
                            matches=matches, token_sha256=digest, same_tokens_as=duplicate))
    (args.output / 'cases.json').write_text(json.dumps(records, indent=2) + '\n')
    (args.output / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Six excluded PDMX articulation crops</title><style>
body{font:17px/1.55 system-ui;margin:30px auto;max-width:1500px;padding:0 24px;background:#f5f6fa;color:#17243b}
article{background:white;border:1px solid #d3dbe8;border-radius:12px;padding:24px;margin:24px 0}
img{width:100%;height:auto}code,pre{font-family:monospace}pre{overflow:auto;font-size:13px}
.id{overflow-wrap:anywhere;font-size:13px;color:#53627a}a{color:#1658a5}
</style><h1>Six excluded PDMX articulation crops</h1>
<p>These six crops were removed from the current training index after a full token audit.
The published corpus is unchanged, and the other crops from these scores remain eligible.</p>
<p><strong>What failed:</strong> the articulation classifier assigns one class to a complete
combination of marks. Every individual mark below is supported, but these three combinations
have no class, causing a lookup error. This does not establish that the scores are defective.</p>
<p><strong>How to read the labels:</strong> the token writer pools articulation marks from
simultaneous notes on the same staff and attaches their union to the first encoded note.
Consequently, a combination can arise from different notes or voices. The images below are the
original corpus renders, without reconstruction or renderer overrides.</p>
<p>No individual accept/reject decisions are required. This page is for checking the notation
behind the exclusions and deciding whether supporting these combinations merits a future
vocabulary or representation change. The current training run keeps its existing vocabulary.</p>
''' + ''.join(cards) + '</html>')
    print(f'Wrote {len(records)} cases to {args.output / "index.html"}')


if __name__ == '__main__':
    main()
