"""Guarded per-symbol reference timing beside unchanged legacy token files."""
import hashlib
import json
from pathlib import Path

from homr.transformer.vocabulary import serialized_chords
from training.omr_datasets.exact_timing_labels import validate_source_grid

FILE_SCHEMA = 'homr.exact-timing-sidecar.v1'


class ExactTimingMismatch(ValueError):
    pass


def timing_sidecar_path(token_path):
    return Path(str(token_path) + '.timing.json')


def _identity(symbol):
    # Articulations and slurs are hoisted by the legacy text writer; these
    # four fields retain individual note identity in its serialized order.
    return [symbol.rhythm, symbol.pitch, symbol.lift, symbol.position]


def write_exact_timing_sidecar(token_path, symbols):
    ordered = [s for chord in serialized_chords(list(symbols)) for s in chord
               if s.rhythm.startswith(('note', 'rest'))]
    path = timing_sidecar_path(token_path)
    if not any(s.exact_timing is not None for s in ordered):
        # A regenerated crop must not inherit a prior run's reference labels.
        path.unlink(missing_ok=True)
        return None
    payload = dict(schemaVersion=FILE_SCHEMA,
        tokenSha256=hashlib.sha256(Path(token_path).read_bytes()).hexdigest(),
        symbols=[dict(identity=_identity(s), timing=s.exact_timing) for s in ordered])
    path.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
    return path


def attach_exact_timing_sidecar(token_path, symbols):
    path = timing_sidecar_path(token_path)
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schemaVersion') != FILE_SCHEMA:
        raise ExactTimingMismatch('Unsupported exact-timing sidecar schema')
    if payload['tokenSha256'] != hashlib.sha256(Path(token_path).read_bytes()).hexdigest():
        raise ExactTimingMismatch('Exact-timing token hash mismatch')
    candidates = [s for s in symbols if s.rhythm.startswith(('note', 'rest'))]
    records = payload['symbols']
    if len(candidates) != len(records) or any(
            _identity(s) != r['identity'] for s, r in zip(candidates, records)):
        raise ExactTimingMismatch('Exact-timing symbol identity/order mismatch')
    for row in records:
        if row['timing'] is not None:
            try:
                validate_source_grid(row['timing'])
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
                raise ExactTimingMismatch('Invalid exact-timing label: ' + str(error)) from error
    # Mutate only after every pairing has been validated.
    for symbol, record in zip(candidates, records, strict=True):
        symbol.exact_timing = record['timing']
    return sum(row['timing'] is not None for row in records)
