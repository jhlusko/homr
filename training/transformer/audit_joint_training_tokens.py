"""Audit a joint-training index against the unchanged decoder vocabulary.

Reads token files with the training parser. Writes an exclusion inventory and a
proposed filtered index; never edits the published corpus or starts training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time


def initialize():
    global read_tokens, vocab, max_length
    from training.transformer.training_vocabulary import read_tokens, vocab
    from homr.transformer.configs import default_config
    max_length = default_config.max_seq_len


def inspect_row(item):
    line, token_path, source = item
    errors = set()
    try:
        symbols = read_tokens(token_path)
        if len(symbols) + 2 > max_length:
            errors.add(('sequence_length', str(len(symbols) + 2)))
        for symbol in symbols:
            for branch in ('rhythm', 'pitch', 'lift', 'articulation', 'slur', 'position'):
                value = getattr(symbol, branch)
                if value not in getattr(vocab, branch):
                    errors.add((branch, value))
    except Exception as error:
        errors.add(('parse_error', f'{type(error).__name__}: {error}'))
    return line, token_path, source, sorted(errors)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    rows = []
    for line in args.index.read_text().splitlines():
        if not line.strip():
            continue
        _, tokens = line.split(',')
        path = str(root / tokens)
        source = 'ossq' if '/scanned/' in path else 'lieder' if '/pairs/' in path else 'pdmx'
        rows.append((line, path, source))
    started = time.monotonic()
    counts = Counter(); invalid = Counter(); reasons = Counter(); done = 0
    with (args.out / 'invalid_rows.jsonl').open('w') as failures, \
         (args.out / 'proposed_index_train.txt').open('w') as filtered, \
         mp.Pool(args.workers, initializer=initialize) as pool:
        for line, path, source, errors in pool.imap(inspect_row, rows, chunksize=250):
            counts[source] += 1; done += 1
            if errors:
                invalid[source] += 1
                for branch, value in errors:
                    reasons[f'{branch}:{value}'] += 1
                failures.write(json.dumps(dict(source=source, tokens=path, index_row=line, errors=errors)) + '\n')
            else:
                filtered.write(line + '\n')
            if done % 5000 == 0 or done == len(rows):
                elapsed = time.monotonic() - started
                progress = dict(state='running', checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                completed=done, total=len(rows), invalid_rows=sum(invalid.values()),
                                elapsed_seconds=elapsed, remaining_seconds=elapsed / done * (len(rows) - done))
                write_json(args.out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    report = dict(state='complete', completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  index_sha256=hashlib.sha256(args.index.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows_by_source=dict(counts), invalid_by_source=dict(invalid),
                  invalid_rows=sum(invalid.values()), reasons=dict(reasons),
                  elapsed_seconds=time.monotonic() - started,
                  proposed_index_sha256=hashlib.sha256((args.out / 'proposed_index_train.txt').read_bytes()).hexdigest(),
                  policy='Audit only; no label remapping, corpus mutation, or training restart')
    write_json(args.out / 'report.json', report)
    (args.out / 'COMPLETE').write_text('complete\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
