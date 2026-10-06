"""Native-note output scoring; verse permutation is evaluation-only, never inference."""

from collections import Counter
from fractions import Fraction
from itertools import permutations


def event_key(event, offset=0):
    return (
        event["physical_measure"] + offset,
        Fraction(event["onset_quarters"]),
        event["pitch"],
        Fraction(event.get("alter", "0")),
        Fraction(event["duration_quarters"]),
    )


def score_native_outputs(references, predictions, measure_offset):
    """Match literal text at exact exported timing/pitch/duration identity.

    This cannot independently score box ownership or detector misses without
    certified pixel links. The best page-global physical-row/verse permutation
    is an oracle grouping diagnostic and is explicitly not verse-ID accuracy.
    Counter intersection caps duplicate predictions at reference multiplicity.
    """
    native_identities = {}
    for ref in references:
        native_identities.setdefault(event_key(ref), set()).add(ref.get("source_note_ordinal", 0))
    ambiguous = {key for key, identities in native_identities.items() if len(identities) > 1}
    refs = Counter(
        (event_key(r), r["verse"], r["text"]) for r in references if event_key(r) not in ambiguous
    )
    verses = sorted({r["verse"] for r in references})
    rows = sorted({p["physical_row_order"] for p in predictions})
    mappings = []
    if len(rows) >= len(verses):
        for selected in permutations(rows, len(verses)):
            mappings.append(dict(zip(selected, verses)))
    else:
        for selected in permutations(verses, len(rows)):
            mappings.append(dict(zip(rows, selected)))
    if not mappings:
        mappings = [{}]
    best, best_map = -1, {}
    for mapping in mappings:
        candidates = Counter(
            (event_key(p["note"], measure_offset), mapping.get(p["physical_row_order"]), p["text"])
            for p in predictions
        )
        matched = sum((candidates & refs).values())
        if matched > best:
            best, best_map = matched, mapping
    owner_keys = {event_key(r) for r in references} - ambiguous
    no_verse_refs = Counter(
        (key, text) for (key, verse, text), count in refs.items() for _ in range(count)
    )
    no_verse_predictions = Counter(
        (event_key(p["note"], measure_offset), p["text"]) for p in predictions
    )
    matching_notes = [p for p in predictions if event_key(p["note"], measure_offset) in owner_keys]
    return {
        "reference_lyric_targets": len(references),
        "ambiguous_native_note_keys": len(ambiguous),
        "reference_targets_on_ambiguous_native_notes": sum(
            event_key(r) in ambiguous for r in references
        ),
        "predicted_attachments": len(predictions),
        "predictions_on_native_lyric_notes": len(matching_notes),
        "predictions_without_native_lyric_note_identity": len(predictions) - len(matching_notes),
        "unique_native_lyric_notes_reached": len(
            {event_key(p["note"], measure_offset) for p in matching_notes}
        ),
        "joint_literal_matches_after_oracle_page_verse_permutation": best,
        "joint_literal_matches_ignoring_verse": sum(
            (no_verse_refs & no_verse_predictions).values()
        ),
        "unrecovered_native_targets": len(references) - best,
        "oracle_page_verse_mapping": best_map,
        "native_verse_id_accuracy": None,
        "independent_box_ownership_accuracy": None,
        "detector_misses": None,
    }
