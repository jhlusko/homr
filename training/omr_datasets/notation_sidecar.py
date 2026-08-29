"""
Carrying structured notation alongside a dataset's token files.

Training never parses MusicXML: the conversion step writes token text files, and the data
loader reads them back through `read_tokens`, which rebuilds each symbol from its six
fields. Notation attached during parsing dies there unless it is written out too.

It is written beside the token file rather than inside it. The token line format packs a
chord onto one line and hoists articulations and slurs into per-position sets, so a
per-note field cannot be appended without escaping problems or a second parser - and
legacy files have to keep loading byte-identically regardless.

The pairing is by position, which is the thing carrying notation on the symbol was meant
to avoid, so it is guarded rather than trusted: the number of note-bearing symbols is
recorded when writing and checked when reading. A mismatch means our own writer and
reader disagree, and is refused rather than silently attaching one note's beams to
another. Notation is an optional enrichment, so a token file with no sidecar loads
exactly as it always did.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from homr.transformer.structured_notation import (
    AdvanceClass,
    BeamLevelState,
    DynamicMark,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    TieState,
)
from homr.transformer.vocabulary import EncodedSymbol

SCHEMA_VERSION = "homr.notation-sidecar.v4"

#: Schemas this reader still understands. v1 predates tie extraction, v2 predates
#: dynamics extraction, and v3 predates advance extraction, so their records decode with
#: no tie / no dynamic / no advance respectively - which is correct for them: the field
#: was not merely absent from the file, it was absent from the pipeline that wrote it.
READABLE_SCHEMAS = (
    SCHEMA_VERSION,
    "homr.notation-sidecar.v3",
    "homr.notation-sidecar.v2",
    "homr.notation-sidecar.v1",
)

SIDECAR_SUFFIX = ".notation.json"


class SidecarMismatch(RuntimeError):
    pass


def sidecar_path(token_path: str | Path) -> Path:
    return Path(str(token_path) + SIDECAR_SUFFIX)


def _encode(notation: NoteNotation) -> dict:
    return {
        "beams": [str(state) for state in notation.beam_levels],
        "stem": str(notation.stem),
        "slurs": [[str(event), str(side)] for event, side in notation.slurs],
        "tie": str(notation.tie),
        "dynamic": str(notation.dynamic),
        "advance": str(notation.advance),
    }


def _decode(record: dict) -> NoteNotation:
    return NoteNotation(
        beam_levels=tuple(BeamLevelState(state) for state in record["beams"]),
        stem=StemDirection(record["stem"]),
        slurs=tuple((SlurEvent(event), SlurSide(side)) for event, side in record["slurs"]),
        tie=TieState(record.get("tie", TieState.NONE)),
        dynamic=DynamicMark(record.get("dynamic", DynamicMark.NONE)),
        advance=AdvanceClass(record.get("advance", AdvanceClass.NOT_APPLICABLE)),
    )


def write_sidecar(token_path: str | Path, symbols: Sequence[EncodedSymbol]) -> Path | None:
    """Write the notation of `symbols`, or nothing when none of them carry any.

    Absence of a sidecar is meaningful - it says this dataset predates the labels - so an
    empty one is not written just to have the file exist.
    """
    records = [_encode(s.notation) for s in symbols if s.notation is not None]
    if not records:
        return None
    path = sidecar_path(token_path)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        # The count the reader checks against. Written explicitly rather than inferred
        # from the list so a truncated file is caught rather than read as a short one.
        "annotatedSymbols": len(records),
        "notation": records,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def attach_sidecar(token_path: str | Path, symbols: Sequence[EncodedSymbol]) -> int:
    """Attach notation from the sidecar to `symbols` in place; returns how many landed.

    Returns 0 when there is no sidecar, which is the ordinary case for a dataset built
    before the labels existed.
    """
    path = sidecar_path(token_path)
    if not path.is_file():
        return 0

    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schemaVersion")
    if schema not in READABLE_SCHEMAS:
        raise SidecarMismatch(f"unsupported notation sidecar schema {schema!r} at {path}")

    records = payload["notation"]
    declared = payload["annotatedSymbols"]
    if len(records) != declared:
        raise SidecarMismatch(
            f"{path} declares {declared} annotated symbols but carries {len(records)}"
        )

    # The reader's own notion of which symbols are note-bearing must match the writer's.
    # If it does not, one note's beams would land on another, so refuse.
    candidates = [symbol for symbol in symbols if _is_note_bearing(symbol)]
    if len(candidates) != declared:
        raise SidecarMismatch(
            f"{path} has notation for {declared} symbols but the token file yields "
            f"{len(candidates)} note-bearing ones - writer and reader disagree"
        )

    for symbol, record in zip(candidates, records, strict=True):
        symbol.notation = _decode(record)
    return declared


def round_trips(token_path: str | Path) -> bool:
    """Whether this token file and its sidecar agree about which symbols carry notation.

    The count guard in attach_sidecar refuses a mismatch rather than attaching one note's
    beams to another - which is right, and means a mismatched pair raises inside a
    DataLoader worker partway through training rather than at conversion. A converter
    should find out while it can still drop the example.

    A pair with no sidecar round-trips trivially: absence is a valid state.
    """
    from training.transformer.training_vocabulary import read_tokens  # noqa: PLC0415

    try:
        attach_sidecar(token_path, read_tokens(str(token_path)))
    except (SidecarMismatch, OSError, ValueError):
        return False
    return True


def _is_note_bearing(symbol: EncodedSymbol) -> bool:
    """Symbols a note in the source can become, and only those.

    Barlines, clefs, key and time signatures never carry notation, so they must not be
    counted on either side of the pairing.
    """
    return symbol.rhythm.startswith(("note", "rest"))
