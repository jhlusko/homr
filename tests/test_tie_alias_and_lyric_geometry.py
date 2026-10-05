from collections import Counter

import numpy as np

from training.onnx.audit_tie_phrase_aliases import classify_pairs
from training.onnx.gate_lyrics_by_staff_geometry import gap_for_box, same_row, staff_groups


def test_duplicate_predicted_tie_can_be_a_true_phrase():
    pair = ((1, 0, '0', 'C', '4', '0', '1', False),
            (1, 0, '1', 'C', '4', '0', '1', False))
    pred = {name: Counter({pair: 1}) for name in ('phrase', 'sounding_tie', 'engraved_tie')}
    truth = {'phrase': Counter({pair: 1}), 'sounding_tie': Counter()}
    assert classify_pairs(truth, pred, set(pair))[0]['category'] == 'source_phrase_only'
    truth['sounding_tie'][pair] = 1
    assert classify_pairs(truth, pred, set(pair))[0]['category'] == 'source_phrase_and_tie'
    truth['phrase'].clear()
    assert classify_pairs(truth, pred, set(pair))[0]['category'] == 'source_tie_only'
    assert classify_pairs(truth, pred, set()) == []


def test_staff_extraction_rejects_incomplete_group():
    image = np.full((250, 400), 255, dtype=np.uint8)
    for y in (30, 40, 50, 60, 70, 150, 160, 170, 180):
        image[y, 20:380] = 0
    good, rejected = staff_groups(image)
    assert good == [[30, 40, 50, 60, 70]]
    assert rejected == [[150, 160, 170, 180]]


def test_header_and_on_staff_boxes_have_no_staff_gap():
    staves = [[100, 110, 120, 130, 140], [240, 250, 260, 270, 280]]
    assert gap_for_box([[0, 30], [10, 50]], staves) is None
    assert gap_for_box([[0, 100], [10, 140]], staves) is None
    assert gap_for_box([[0, 170], [10, 190]], staves) == 0
    assert gap_for_box([[0, 290], [10, 300]], staves) is None


def test_row_recovery_rejects_separate_verse_and_zero_height():
    first = [[0, 100], [10, 120]]
    assert same_row(first, [[30, 102], [50, 119]])
    assert not same_row(first, [[30, 125], [50, 145]])
    assert not same_row(first, [[30, 110], [50, 110]])
