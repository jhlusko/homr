import pytest

from training.onnx.freeze_c1_layout_packet import choose


def test_fixed_rule_packet_uses_hash_order_and_distinct_scores_per_stratum():
    rows = [
        {"page": {"score": f"score-{s}-{n}", "pixel_sha256": f"{n}{s}"}, "stratum": s}
        for s in ["multi-opening", "multi-continuation", "single-opening", "single-continuation"]
        for n in [2, 1]
    ]
    selected = choose(rows, per_stratum=1)
    assert len(selected) == 4
    assert all(r["page"]["score"].endswith("-1") for r in selected)
    assert len(choose(rows)) == 8
    with pytest.raises(ValueError):
        choose(rows, per_stratum=0)
