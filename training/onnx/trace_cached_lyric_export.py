"""Trace cached token identity through the actual exporter; require exact XML replay."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from homr import music_xml_generator as generator
from homr.beam_repair import repair_beams
from homr.slur_crossing import repair_crossings
from homr.tie_repair import repair_ties
from homr.transformer.structured_notation import (
    AdvanceClass,
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    TieState,
)
from homr.transformer.vocabulary import EncodedSymbol
from homr.tuplet_repair import repair_symbols
from training.onnx.qualify_joint_export import sha


def trace_cached_export(folder):
    identity = json.loads((folder / "identity.json").read_text())
    policy = identity["policy"]
    if policy["HOMR_STEM_ARBITRATION"] != "0" or policy["HOMR_SLUR_SIDE"] != "0":
        raise ValueError("Cached replay does not restore confidence-dependent policies")
    root = Path(__file__).resolve().parents[2]
    if (
        sha(root / "homr/music_xml_generator.py")
        != identity["code_sha256"]["homr/music_xml_generator.py"]
    ):
        raise ValueError("Exporter code differs from page decode")
    rows = json.loads((folder / "decoded-staves.json").read_text())
    voices = []
    for voice_index, voice in enumerate(rows):
        restored = []
        for token_index, row in enumerate(voice):
            kwargs = {
                key: row[key]
                for key in ("rhythm", "pitch", "lift", "articulation", "slur", "position")
            }
            n = row.get("notation")
            if n:
                kwargs["notation"] = NoteNotation(
                    beam_levels=tuple(BeamLevelState(x) for x in n["beams"]),
                    stem=StemDirection(n["stem"]),
                    slurs=tuple((SlurEvent(e), SlurSide(s)) for e, s in n["slurs"]),
                    tie=TieState(n["tie"]),
                    advance=AdvanceClass(n["advance"]),
                )
            symbol = EncodedSymbol(**kwargs)
            symbol._lyric_export_origin = (voice_index, token_index)
            restored.append(symbol)
        voices.append(restored)
    old_environment = {key: os.environ.get(key) for key in policy}
    original = generator.build_note_or_rest
    captured = {}

    def capture(symbol, *args, **kwargs):
        element = original(symbol, *args, **kwargs)
        captured[id(element)] = getattr(symbol, "_lyric_export_origin", None)
        return element

    try:
        os.environ.update(policy)
        generator.build_note_or_rest = capture
        for index, voice in enumerate(voices):
            if policy["HOMR_TUPLET_REPAIR"] == "1":
                voice, _ = repair_symbols(voice)
                voices[index] = voice
            if policy["HOMR_BEAM_REPAIR"] == "1":
                repair_beams(voice)
            if policy["HOMR_TIE_REPAIR"] == "1":
                repair_ties(voice)
            if policy["HOMR_SLUR_CROSSING"] == "1":
                repair_crossings(voice)
        replay = generator.generate_xml(generator.XmlGeneratorArguments(), voices, "")
        expected = ET.parse(folder / "page.musicxml").getroot()
        if ET.tostring(replay) != ET.tostring(expected):
            raise ValueError("Cached policy/export replay does not reproduce the original XML")
        records = []
        for part_index, part in enumerate(replay.findall("part")):
            notes = [n for n in part.findall("./measure/note") if n.find("pitch") is not None]
            for exported_index, element in enumerate(notes):
                records.append(
                    {
                        "part_index": part_index,
                        "exported_index": exported_index,
                        "origin": captured.get(id(element)),
                    }
                )
        return {
            "exact_xml": True,
            "records": records,
            "exporter_sha256": sha(root / "homr/music_xml_generator.py"),
        }
    finally:
        generator.build_note_or_rest = original
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
