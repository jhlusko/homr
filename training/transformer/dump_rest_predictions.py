"""A `evaluate_structured_heads.evaluate()` sink that keeps rests, for A3.

`evaluate_structured_heads.dump_predictions`'s own sink deliberately drops every rest:
its `supervised` list keeps only notes whose `beam_levels` are not all `NOT_APPLICABLE`,
and a rest's `beam_levels` are *always* `NOT_APPLICABLE` (confirmed against files on
disk, `results/next_actions_3_4.md`) - but so are an ordinary unbeamed note's (a
half note, say). `NOT_APPLICABLE` therefore means "not part of a beam group," not
"is a rest," and is the wrong signal to filter rests back in by.

The token file itself is the right signal: `EncodedSymbol.rhythm` says "note" or "rest"
directly (`"rest" in symbol.rhythm`, the convention `homr/transformer/vocabulary.py`
already uses). This sink re-reads each staff's token file, walks its note-bearing
symbols in the same order `evaluate()`'s decoded `reference`/`predicted` lists are in
(both derive from the same sidecar-aligned positions - `notation_sidecar.py`'s own
"number of note-bearing symbols" check is what keeps this trustworthy), and writes only
the rest positions: the model's actual predicted beam state where a rest sits, which
`dump_predictions` never records at all.
"""

# flake8: noqa: T201

import json
from collections.abc import Callable, Sequence
from typing import Any

from homr.transformer.structured_notation import NoteNotation
from training.transformer.training_vocabulary import read_tokens


def _rest_indices(token_path: str, expected_count: int) -> list[int]:
    symbols = read_tokens(token_path)
    note_bearing = [s for s in symbols if "note" in s.rhythm or "rest" in s.rhythm]
    if len(note_bearing) != expected_count:
        raise ValueError(
            f"{token_path}: {len(note_bearing)} note-bearing symbols on disk, "
            f"{expected_count} decoded positions - reader and decoder disagree, "
            "refusing to guess which position is which"
        )
    return [index for index, symbol in enumerate(note_bearing) if "rest" in symbol.rhythm]


def dump_rest_predictions(
    batches: Any, beam_levels: int, handle: Any
) -> Callable[..., None]:
    """Writes one JSON record per staff that has at least one rest, with the model's
    predicted `beam_levels` at every rest position (and the reference, for a sanity
    check that it is `NOT_APPLICABLE` every time, per the already-confirmed semantics).

    Identity and ordering match `evaluate_structured_heads.dump_predictions` exactly
    (an unshuffled loader, `position` counted along the index) so the two sinks can run
    in the same pass and their output joined by `tokens` path.
    """
    entries = batches.dataset.inner.corpus_list
    position = 0

    def write(
        predicted: Sequence[NoteNotation],
        reference: Sequence[NoteNotation],
        stem_confidence: list[float] | None = None,
    ) -> None:
        nonlocal position
        if position >= len(entries):
            return
        token_path = entries[position]["tokens"]
        rest_indices = _rest_indices(token_path, len(reference))
        if rest_indices:
            record = {
                "tokens": token_path,
                "rest_positions": rest_indices,
                "reference_beam": [
                    [str(s) for s in reference[i].beam_levels[:beam_levels]]
                    for i in rest_indices
                ],
                "predicted_beam": [
                    [str(s) for s in predicted[i].beam_levels[:beam_levels]]
                    for i in rest_indices
                ],
            }
            handle.write(json.dumps(record) + "\n")
        position += 1

    return write
