import json
import xml.etree.ElementTree as ET

import pytest

from homr import music_xml_generator as generator
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.qualify_joint_export import sha
from training.onnx.trace_cached_lyric_export import trace_cached_export


def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMR_STRUCTURED_EXPORT", "none")
    rows = [
        {
            "rhythm": rhythm,
            "pitch": pitch,
            "lift": "_",
            "articulation": "_",
            "slur": "_",
            "position": "upper",
        }
        for rhythm, pitch in [
            ("clef_G2", "_"),
            ("timeSignature/4", "_"),
            ("note_4", "C5"),
            ("chord", "_"),
            ("note_4", "G4"),
            ("barline", "_"),
        ]
    ]
    (tmp_path / "decoded-staves.json").write_text(json.dumps([rows]))
    policy = {
        "HOMR_STEM_ARBITRATION": "0",
        "HOMR_SLUR_SIDE": "0",
        "HOMR_TUPLET_REPAIR": "0",
        "HOMR_BEAM_REPAIR": "0",
        "HOMR_TIE_REPAIR": "0",
        "HOMR_SLUR_CROSSING": "0",
        "HOMR_STRUCTURED_EXPORT": "none",
    }
    (tmp_path / "identity.json").write_text(
        json.dumps(
            {
                "policy": policy,
                "code_sha256": {"homr/music_xml_generator.py": sha(generator.__file__)},
            }
        )
    )
    score = generator.generate_xml(
        generator.XmlGeneratorArguments(), [[EncodedSymbol(**row) for row in rows]], ""
    )
    ET.ElementTree(score).write(tmp_path / "page.musicxml")


def test_sorted_chord_xml_retains_original_token_identity(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    result = trace_cached_export(tmp_path)
    assert result["exact_xml"]
    assert [r["origin"] for r in result["records"]] == [(0, 4), (0, 2)]


def test_export_provenance_refuses_nonidentical_replay(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    tree = ET.parse(tmp_path / "page.musicxml")
    tree.getroot().find(".//pitch/octave").text = "9"
    tree.write(tmp_path / "page.musicxml")
    with pytest.raises(ValueError, match="original XML"):
        trace_cached_export(tmp_path)
