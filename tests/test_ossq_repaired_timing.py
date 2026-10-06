import pytest

from training.onnx.bind_ossq_repaired_timing import timing_map


def payload(divisions=24, ticks=24, elapsed="1", tuplet='<tuplet type="start" />'):
    return dict(
        source_divisions=str(divisions),
        source_duration_ticks=str(ticks),
        elapsed_quarters=elapsed,
        tuplet_notations_xml=[tuplet],
        tuplet_containers=[],
    )


def test_grid_rescaling_restores_canonical_payload():
    canonical = payload(tuplet='<tuplet type="start" bracket="no" show-number="none" />')
    assert list(timing_map([canonical], [payload(4, 4)]).values()) == [canonical]


def test_changed_duration_refused():
    with pytest.raises(ValueError, match="timing differ"):
        timing_map([payload()], [payload(elapsed="2")])


def test_ambiguous_source_grid_refused():
    with pytest.raises(ValueError, match="Ambiguous"):
        timing_map([payload(), payload(4, 4)], [payload(), payload(4, 4)])
