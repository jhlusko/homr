from homr.transformer import vocabulary
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.ablate_cached_duplicate_cleanup import cleanup_without_pitch_dedup


def test_preserves_unison_members_without_changing_default_cleanup():
    def symbols():
        return [
            EncodedSymbol("clef_G2"),
            EncodedSymbol("note_1", "D4", "_", "_", "_", "upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_1", "D4", "_", "_", "_", "upper"),
            EncodedSymbol("barline"),
        ]

    original = vocabulary._remove_duplicated_piches
    control = vocabulary.remove_duplicated_symbols(symbols())
    ablated = cleanup_without_pitch_dedup(symbols())
    assert sum(s.rhythm.startswith("note") for s in control) == 1
    assert sum(s.rhythm.startswith("note") for s in ablated) == 2
    assert sum(s.rhythm == "chord" for s in ablated) == 1
    assert vocabulary._remove_duplicated_piches is original
    assert len(vocabulary.remove_duplicated_symbols(symbols())) == len(control)


def test_redundant_headers_are_still_cleaned():
    symbols = [
        EncodedSymbol("clef_G2"),
        EncodedSymbol("keySignature_0"),
        EncodedSymbol("note_4", "C4", "_", "_", "_", "upper"),
        EncodedSymbol("barline"),
        EncodedSymbol("clef_G2"),
        EncodedSymbol("keySignature_0"),
    ]
    result = cleanup_without_pitch_dedup(symbols)
    assert sum(s.rhythm == "clef_G2" for s in result) == 1
    assert sum(s.rhythm == "keySignature_0" for s in result) == 1


def test_trace_refuses_existing_output_before_loading_models(tmp_path):
    import pytest

    from training.onnx.trace_page_duplicate_cleanup import run

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="replace cleanup trace"):
        run(
            pages_path=tmp_path / "absent",
            models_path=tmp_path / "absent",
            original=tmp_path / "absent",
            index=0,
            arm="accepted",
            out=output,
        )


def test_trace_rejects_mismatched_original_page_without_inference(tmp_path):
    import json

    import pytest

    from training.onnx.trace_page_duplicate_cleanup import run

    pages = tmp_path / "pages.json"
    models = tmp_path / "models.json"
    pages.write_text(json.dumps([{"score": "sq1"}]))
    models.write_text(json.dumps({"accepted": {}}))
    original = tmp_path / "original"
    original.mkdir()
    (original / "identity.json").write_text(json.dumps({"page": {"score": "sq2"}, "models": {}}))
    with pytest.raises(ValueError, match="Different page/model"):
        run(
            pages_path=pages,
            models_path=models,
            original=original,
            index=0,
            arm="accepted",
            out=tmp_path / "out",
        )
