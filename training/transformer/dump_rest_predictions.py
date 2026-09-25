"""A `evaluate_structured_heads.evaluate()` sink that keeps the rests, for A3.

A3's gate is the head's precision on **rest-spanning beams**: groups whose begin and end
notes sit on either side of a rest, the one shape the beam rule structurally cannot emit
(`OTS_HOMR_PUBLIC_RELEASE_ROADMAP.md` §1). Scoring it needs the head's beam states on
the *notes* around each rest, laid out in token order with the rests still in place.
`dump_predictions` cannot supply that: it keeps only positions whose reference beam is
supervised, so rests (and every unbeamed note) vanish and a group can no longer be told
apart from one that spans a rest.

**What a position is.** `evaluate()` decodes one entry per decoder output position, and
`structured_dataset._positions` builds the targets from `read_tokens(token_path)` via
`notation_positions` - BOS, then *every* symbol in order, then EOS, then padding - after
which `align_to_decoder_output` drops BOS. So decoded position `t` is symbol `t` of the
token file, for every symbol (barlines and clefs included), and positions past the last
symbol are EOS/padding. The first version of this sink assumed decoded positions were the
note-bearing symbols only; on a real staff that was 34 symbols against 607 positions, and
its count check refused, correctly.

**Why the head's state *at* a rest is not recorded.** A rest's beam target is always
`NOT_APPLICABLE`, which `structured_targets` stores as ignored, and `decode_predictions`
masks the prediction wherever the target is ignored. The head was never trained there
either, so an unmasked argmax at a rest would be noise. The gate is about the notes'
states; the rest positions only say where the rests are.
"""

# flake8: noqa: T201

import json
from collections.abc import Callable, Sequence
from typing import Any

from homr.transformer.structured_notation import NoteNotation
from training.transformer.training_vocabulary import read_tokens

NOTE, REST, OTHER = "n", "r", "o"


def symbol_kinds(token_path: str, decoded_length: int) -> list[str]:
    """`n`/`r`/`o` for each decoded position that holds a real symbol, in token order.

    A staff longer than the decoder's window is truncated exactly as its targets were.
    """
    symbols = read_tokens(token_path)
    kinds = []
    for symbol in symbols[:decoded_length]:
        if "rest" in symbol.rhythm:
            kinds.append(REST)
        elif "note" in symbol.rhythm:
            kinds.append(NOTE)
        else:
            kinds.append(OTHER)
    return kinds


def dump_rest_predictions(batches: Any, beam_levels: int, handle: Any) -> Callable[..., None]:
    """Writes one JSON record per staff that has at least one rest: the symbol kinds and,
    per beam level, the predicted and reference state at every symbol position.

    Identity and ordering match `evaluate_structured_heads.dump_predictions` exactly (an
    unshuffled loader, `position` counted along the index), so both sinks can run in the
    same pass and be joined by `tokens` path.
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
        position += 1
        kinds = symbol_kinds(token_path, len(reference))
        if REST not in kinds:
            return
        length = len(kinds)
        record = {
            "tokens": token_path,
            "kinds": "".join(kinds),
            "reference_beam": [
                [str(reference[t].beam_levels[level]) for t in range(length)]
                for level in range(beam_levels)
            ],
            "predicted_beam": [
                [str(predicted[t].beam_levels[level]) for t in range(length)]
                for level in range(beam_levels)
            ],
        }
        handle.write(json.dumps(record) + "\n")

    return write
