"""Compare a source-grid upgrade with retained legacy window metadata."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from training.omr_datasets.exact_timing_labels import SCHEMA, semantic_timing, validate_source_grid


def verify(previous, current):
    old_files = {p.relative_to(previous) for p in previous.rglob('*.tokens')}
    new_files = {p.relative_to(current) for p in current.rglob('*.tokens')}
    if not old_files or old_files != new_files:
        raise ValueError('Selected token window inventory changed')
    counts = Counter()
    for relative in sorted(old_files):
        old_token, new_token = previous / relative, current / relative
        if old_token.read_bytes() != new_token.read_bytes():
            raise ValueError('Legacy token bytes changed: ' + str(relative))
        old_path, new_path = Path(str(old_token)+'.timing.json'), Path(str(new_token)+'.timing.json')
        if old_path.exists() != new_path.exists():
            raise ValueError('Timing sidecar availability changed')
        counts['identical_token_windows'] += 1
        if not old_path.exists():
            continue
        old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
        token_hash = hashlib.sha256(new_token.read_bytes()).hexdigest()
        if (old['schemaVersion'] != new['schemaVersion'] or
                old['tokenSha256'] != token_hash or new['tokenSha256'] != token_hash or
                len(old['symbols']) != len(new['symbols'])):
            raise ValueError('Sidecar envelope changed')
        for before, after in zip(old['symbols'], new['symbols'], strict=True):
            if before['identity'] != after['identity']:
                raise ValueError('Serialized symbol identity changed')
            a, b = before['timing'], after['timing']
            if a is None or b is None:
                if a != b:
                    raise ValueError('Timing null alignment changed')
                counts['unchanged_null_records'] += 1
                continue
            validate_source_grid(a)
            validate_source_grid(b)
            if b['schema'] != SCHEMA or semantic_timing(a) != semantic_timing(b):
                raise ValueError('Reference timing semantics changed')
            counts['unchanged_timed_records_with_source_grid'] += 1
        counts['verified_sidecars'] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite comparison')
    result = verify(args.previous, args.current)
    result.update(previous=str(args.previous), current=str(args.current),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
