from training.onnx.build_semantic_expansion_review import reference_overlap
from training.onnx.prepare_semantic_role_controls import select


def page(score, lyrics, directions, partition="calibration"):
    return dict(
        id=score + ":pixels",
        score=score,
        corpus="lieder",
        partition=partition,
        ground_truth=[dict(semanticFamily="Lyrics") for _ in range(lyrics)]
        + [dict(semanticFamily="DirectionText") for _ in range(directions)],
    )


def test_selection_is_score_disjoint_and_never_uses_formal_test():
    rows = [page("dual" + str(i), 4, 1) for i in range(6)] + [
        page("direction" + str(i), 0, 2) for i in range(4)
    ]
    rows.extend([page("excluded", 100, 10), page("test", 100, 10, "test"), rows[0]])
    chosen = select(rows, {"excluded"})
    assert len(chosen) == 8
    assert len({r["score"] for r in chosen}) == 8
    assert all(r["score"] not in {"excluded", "test"} for r in chosen)
    assert chosen == select(list(reversed(rows)), {"excluded"})


def test_reference_overlap_accepts_full_ocr_line_without_claiming_a_role():
    assert reference_overlap([10, 10, 20, 20], [0, 0, 100, 100]) == 1
    assert reference_overlap([10, 10, 20, 20], [20, 0, 100, 100]) == 0
