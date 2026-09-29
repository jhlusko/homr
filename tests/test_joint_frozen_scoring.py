from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.compare_joint_frozen_decode import edit_distance, metrics


def test_exact_edit_distance():
    assert edit_distance([], ['a']) == 1
    assert edit_distance(['a', 'b', 'c'], ['a', 'x', 'c']) == 1
    assert edit_distance(['a', 'b', 'c'], ['a', 'c']) == 1
    assert edit_distance(['a', 'b'], ['b', 'a']) == 2


def test_pitch_error_does_not_become_rhythm_error():
    ref = [EncodedSymbol('note_4', 'C4', '_')]
    pred = [EncodedSymbol('note_4', 'D4', '_')]
    scores, alignment = metrics(ref, pred)
    assert scores['symbol']['errors'] == 1
    assert scores['pitch']['errors'] == 1
    assert scores['rhythm']['errors'] == 0
    assert scores['accidental']['errors'] == 0
    assert alignment['matched_notes'] == 0


def test_missing_note_counts_as_error():
    scores, alignment = metrics([EncodedSymbol('note_4', 'C4', '_')], [])
    assert scores['symbol']['errors'] == 1
    assert scores['rhythm']['errors'] == 1
    assert scores['pitch']['reference'] == 1
    assert alignment['reference_notes'] == 1
