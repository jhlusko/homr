"""Tests for homr.tuplet_repair, both the token-level algorithm and the EncodedSymbol
adapter that wires it into the real decode pipeline (homr/main.py)."""

from fractions import Fraction

from homr.transformer.vocabulary import EncodedSymbol
from homr.tuplet_repair import (
    bar_duration,
    is_single_staff,
    repair,
    repair_symbols,
    split_bars,
)

BAR = "barline"


def _staff(*bars: list[str]) -> list[str]:
    out: list[str] = []
    for bar in bars:
        out.extend(bar)
        out.append(BAR)
    return out


def test_split_bars_roundtrips_exactly() -> None:
    tokens = _staff(["note_4", "note_4"], ["note_8"] * 4)
    assert [t for bar in split_bars(tokens) for t in bar] == tokens


def test_fires_on_a_clean_triplet() -> None:
    # prevailing 1/2 (two quarters); one bar has a written eighth-triplet plus a
    # quarter = 5/8, excess exactly 1/8 = (3-2)*(1/8).
    tokens = _staff(
        ["note_8", "note_8", "note_8", "note_4"],
        ["note_4", "note_4"],
        ["note_4", "note_4"],
        ["note_4", "note_4"],
    )
    fixed, rewrites = repair(tokens)
    assert len(fixed) == len(tokens)
    assert fixed[:3] == ["note_12", "note_12", "note_12"]
    assert fixed[3] == "note_4"
    assert len(rewrites) == 1


def test_declines_when_two_windows_are_ambiguous() -> None:
    # Two separate 3-eighth runs in one bar, split by a quarter; the excess (1/8)
    # matches a (3:2) rewrite of EITHER run alone, and nothing picks between them.
    tokens = _staff(
        ["note_4", "note_4", "note_4", "note_8"],
        ["note_4", "note_4", "note_4", "note_8"],
        ["note_4", "note_4", "note_4", "note_8"],
        ["note_8", "note_8", "note_8", "note_4", "note_8", "note_8", "note_8"],
    )
    _, rewrites = repair(tokens)
    assert rewrites == []


def test_declines_when_more_than_one_bar_is_overfull() -> None:
    # Two separately-fixable-looking overfull bars: something is wrong with the whole
    # staff (e.g. a dropped barline), not just an implied tuplet in one bar.
    # Three plain bars establish the prevailing length unambiguously; two overfull
    # bars share the arithmetic signature of an implied triplet each.
    overfull_bar = ["note_8", "note_8", "note_8", "note_4"]
    tokens = _staff(
        ["note_4", "note_4"],
        ["note_4", "note_4"],
        ["note_4", "note_4"],
        overfull_bar,
        overfull_bar,
    )
    _, rewrites = repair(tokens)
    assert rewrites == []
    # The guard is opt-out: disabling it recovers both.
    _, forced = repair(tokens, max_overfull=None)
    assert len(forced) == 2


def test_too_few_bars_never_fires() -> None:
    tokens = _staff(["note_8"] * 6, ["note_4", "note_4"])
    _, rewrites = repair(tokens)
    assert rewrites == []


def test_bar_duration_matches_prevailing_after_repair() -> None:
    tokens = _staff(
        ["note_8", "note_8", "note_8", "note_4"],
        ["note_4", "note_4"],
        ["note_4", "note_4"],
        ["note_4", "note_4"],
    )
    fixed, _ = repair(tokens)
    bars = split_bars(fixed)
    assert bar_duration(bars[0]) == Fraction(1, 2)


def test_is_single_staff_true_with_no_lower_position() -> None:
    symbols = [EncodedSymbol("note_4"), EncodedSymbol("note_4", position="upper")]
    assert is_single_staff(symbols)


def test_is_single_staff_false_with_a_lower_symbol() -> None:
    symbols = [EncodedSymbol("note_4", position="upper"), EncodedSymbol("note_4", position="lower")]
    assert not is_single_staff(symbols)


def test_repair_symbols_skips_grand_staff_voices() -> None:
    symbols = [
        EncodedSymbol("note_8", position="upper"),
        EncodedSymbol("note_8", position="lower"),
        EncodedSymbol("note_8", position="upper"),
        EncodedSymbol("note_4", position="upper"),
        EncodedSymbol(BAR),
        EncodedSymbol("note_4", position="upper"),
        EncodedSymbol("note_4", position="upper"),
        EncodedSymbol(BAR),
        EncodedSymbol("note_4", position="upper"),
        EncodedSymbol("note_4", position="upper"),
        EncodedSymbol(BAR),
    ]
    out, rewrites = repair_symbols(symbols)
    assert rewrites == []
    assert out is symbols


def test_repair_symbols_rewrites_rhythm_only_and_preserves_other_branches() -> None:
    symbols = [
        EncodedSymbol("note_8", pitch="C4", articulation="staccato"),
        EncodedSymbol("note_8", pitch="D4"),
        EncodedSymbol("note_8", pitch="E4"),
        EncodedSymbol("note_4", pitch="F4"),
        EncodedSymbol(BAR),
        EncodedSymbol("note_4", pitch="G4"),
        EncodedSymbol("note_4", pitch="A4"),
        EncodedSymbol(BAR),
        EncodedSymbol("note_4", pitch="B4"),
        EncodedSymbol("note_4", pitch="C5"),
        EncodedSymbol(BAR),
        EncodedSymbol("note_4", pitch="D5"),
        EncodedSymbol("note_4", pitch="E5"),
        EncodedSymbol(BAR),
    ]
    out, rewrites = repair_symbols(symbols)
    assert len(rewrites) == 1
    assert [s.rhythm for s in out[:4]] == ["note_12", "note_12", "note_12", "note_4"]
    # Pitch, articulation and every symbol after the rewritten bar are untouched.
    assert [s.pitch for s in out] == [s.pitch for s in symbols]
    assert out[0].articulation == "staccato"
    # Symbols the repair didn't touch keep object identity - only the rewritten
    # bar's symbols were copied.
    assert out[5] is symbols[5]


def test_repair_symbols_no_rewrite_returns_same_object() -> None:
    symbols = [
        EncodedSymbol("note_4"), EncodedSymbol("note_4"), EncodedSymbol(BAR),
        EncodedSymbol("note_4"), EncodedSymbol("note_4"), EncodedSymbol(BAR),
        EncodedSymbol("note_4"), EncodedSymbol("note_4"), EncodedSymbol(BAR),
    ]
    out, rewrites = repair_symbols(symbols)
    assert rewrites == []
    assert out is symbols


def test_bar_duration_counts_a_chord_once_not_per_member() -> None:
    # A two-note chord naively summed (1/4 + 1/4) plus a rest and a note would read as
    # 1 whole note; correctly counted (chord = one simultaneity, minimum duration) it is
    # 3/4. This is the exact bug that made two otherwise-correct OSSQ staves look
    # overfull and get corrupted by the repair.
    bar = ["note_4", "chord", "note_4", "rest_4", "note_4", BAR]
    assert bar_duration(bar) == Fraction(3, 4)


def test_repair_does_not_fire_on_a_correct_bar_with_a_chord() -> None:
    tokens = _staff(
        ["note_4", "note_4"],
        ["note_4", "note_4"],
        ["note_4", "chord", "note_4", "rest_4", "note_4"],
        ["note_4", "note_4"],
    )
    _, rewrites = repair(tokens)
    assert rewrites == []


class TestTupletRepairConfigDefault:
    """The pipeline-level flag (homr/transformer/configs.py), not the algorithm above.

    Reconfirmed against OSSQ (training/omr_datasets/eval_tuplet_repair.py) after the
    chord-duration double-counting fix above: exact staves +3/+4/+7/+6 with ZERO losses
    across all four checkpoints, precision 71.7-93.3%. The earlier OFF-by-default
    measurement (+1/+2/+4/+4, up to 2 losses per run) was entirely an artifact of that
    bug - every loss it recorded was a correct bar wrongly flagged overfull because a
    chord's duration was counted once per member instead of once per simultaneity.
    """

    def test_on_by_default(self) -> None:
        import os
        from unittest import mock

        from homr.transformer.configs import Config

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOMR_TUPLET_REPAIR", None)
            assert Config().tuplet_repair is True

    def test_explicit_0_disables(self) -> None:
        import os
        from unittest import mock

        from homr.transformer.configs import Config

        with mock.patch.dict(os.environ, {"HOMR_TUPLET_REPAIR": "0"}):
            assert Config().tuplet_repair is False

    def test_explicit_1_also_enables(self) -> None:
        import os
        from unittest import mock

        from homr.transformer.configs import Config

        with mock.patch.dict(os.environ, {"HOMR_TUPLET_REPAIR": "1"}):
            assert Config().tuplet_repair is True
