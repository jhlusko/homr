import pytest

from training.onnx.lyric_syllable_positions import propose_spatial_splits, split_positioned_text


def chars(text, centers=None):
    visible = [c for c in text if not c.isspace()]
    positions = centers or list(range(len(visible)))
    return [
        dict(index=i, text=c, box=[[x - 1, 0], [x + 1, 0], [x + 1, 2], [x - 1, 2]])
        for i, (c, x) in enumerate(zip(visible, positions))
    ]


def test_positions_use_ctc_geometry_instead_of_uniform_text_width():
    result = split_positioned_text("I were", chars("I were", [10, 100, 110, 120, 130]))
    assert [r["text"] for r in result] == ["I", "were"]
    assert [r["center"][0] for r in result] == [10, 115]


def test_hyphens_split_syllables_but_apostrophes_and_punctuation_survive():
    text = "Ah! I'd swift -ly_ fly"
    assert [r["text"] for r in split_positioned_text(text, chars(text))] == [
        "Ah!",
        "I'd",
        "swift",
        "ly_",
        "fly",
    ]


def test_missing_character_positions_are_not_guessed():
    with pytest.raises(ValueError, match="disagrees"):
        split_positioned_text("star-ling", chars("star-lin"))


def test_missing_separator_can_be_proposed_without_changing_recognized_characters():
    value = propose_spatial_splits("nemÁb", chars("nemÁb", [10, 30, 50, 110, 130]))
    assert [r["text"] for r in value["syllables"]] == ["nem", "Áb"]
    assert "".join(r["text"] for r in value["syllables"]) == "nemÁb"
    assert not value["production_qualified"]


def test_regular_spacing_short_words_and_nonmonotonic_positions_remain_intact():
    for text, positions in [
        ("were", [10, 30, 50, 70]),
        ("I'd", [10, 80, 100]),
        ("health", [10, 30, 50, 49, 70, 90]),
    ]:
        assert propose_spatial_splits(text, chars(text, positions))["proposals"] == []
