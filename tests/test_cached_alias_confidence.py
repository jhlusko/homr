from training.onnx.freeze_c1_layout_packet import choose
from training.onnx.test_cached_alias_suppression import suppress


def payload(tie, probability=.95):
    return {'final_notation_slurs': [['none', 'above'], ['none', 'below']],
            'head_choices': [
                {'head': 'slur.slot.1.event', 'value': 'none', 'probability': probability},
                {'head': 'slur.slot.2.event', 'value': 'none', 'probability': probability},
                {'head': 'tie.state', 'value': tie, 'probability': probability}]}


def test_both_endpoint_events_and_ties_are_required():
    first, last = payload('start'), payload('stop')
    assert suppress(first, last, .9)
    assert not suppress(first, payload('stop', .85), .9)
    assert not suppress(first, payload('start'), .9)
    assert not suppress(first, None, .9)


def test_true_structured_phrase_is_protected_even_with_confident_tie():
    first, last = payload('start'), payload('stop')
    first['final_notation_slurs'][0][0] = 'start'
    assert not suppress(first, last, .9)
    first['final_notation_slurs'][0][0] = 'none'
    first['head_choices'][0]['value'] = 'start'
    assert not suppress(first, last, .9)


def test_selection_is_deterministic_and_distinct_by_score():
    eligible = []
    for i, stratum in enumerate(('multi-opening', 'multi-continuation',
                                'single-opening', 'single-continuation')):
        for j in range(3):
            eligible.append({'stratum': stratum, 'page': {
                'score': f'{i}-{j}', 'pixel_sha256': f'{i}{j}'}})
    chosen = choose(eligible)
    assert chosen == choose(list(reversed(eligible)))
    assert len(chosen) == len({r['page']['score'] for r in chosen}) == 8
    assert all(sum(r['stratum'] == s for r in chosen) == 2
               for s in {r['stratum'] for r in eligible})
