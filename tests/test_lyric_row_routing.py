from training.onnx.lyric_row_routing import route_rows


def transforms():
    return [
        {
            "staff_call": call,
            "original_to_processed_ratio": [1, 1],
            "staff_grid": [{"x": x, "y": [top + i * 10 for i in range(5)]} for x in (0, 200)],
        }
        for call, top in [(0, 0), (1, 140)]
    ]


def row(top, bottom):
    return {"box": [[50, top], [100, top], [100, bottom], [50, bottom]]}


def test_physical_row_order_is_not_native_verse():
    result = route_rows([row(90, 110), row(60, 80), row(62, 82)], transforms())
    assert [r["physical_row_order"] for r in result["routed"]] == [2, 1, 1]
    assert all(r["native_verse"] is None for r in result["routed"])


def test_staff_intrusion_and_unbounded_last_staff_stay_unresolved():
    result = route_rows([row(30, 65), row(190, 210)], transforms())
    assert result["routed"] == []
    assert len(result["unresolved"]) == 2


def test_neighbor_recovery_does_not_cross_staff_lines_or_chain():
    result = route_rows(
        [row(42, 60), row(48, 65), row(35, 60), row(130, 150)],
        transforms(),
        recover_row_neighbors=True,
    )
    assert {r["index"] for r in result["routed"]} == {0, 1}
    assert next(r for r in result["routed"] if r["index"] == 0)["seed_indices"] == [1]
    assert {r["index"] for r in result["unresolved"]} == {2, 3}
    assert route_rows([row(42, 60)], transforms(), recover_row_neighbors=True)["routed"] == []
