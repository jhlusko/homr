"""
Which corpora may carry notation labels, and why the answer is not "all of them".

`music_xml_parser` attaches beam, stem and slur labels for every corpus that goes through
it, so it is tempting to have every converter write a sidecar. That would be wrong for
some of them, and wrong in a way no test of the parser could catch: the labels describe
the *source* engraving, and a corpus is only eligible if its training image shows that
same engraving.

  lieder    renders SVG and MusicXML from one source .mscx, so the staff image is the
            score's own engraving. Eligible.
  pdmx      *was* ineligible: it regenerated MusicXML from the tokens and rendered that
            with Verovio, so the image showed Verovio's automatic beaming rather than the
            score's. It now renders the source window instead, which is what makes it
            eligible - so the test is that it renders from source, not merely that it
            writes a sidecar.
  musetrainer  still regenerates from tokens. Not eligible.

These tests pin the reasoning to the code, so that a later change to how a corpus builds
its images cannot silently make its labels wrong.
"""

import ast
import unittest
from pathlib import Path

CONVERTERS = Path(__file__).resolve().parent.parent / "training" / "omr_datasets"


def _calls(module: str) -> set[str]:
    tree = ast.parse((CONVERTERS / module).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                found.add(target.id)
            elif isinstance(target, ast.Attribute):
                found.add(target.attr)
    return found


class TestEligibleCorporaWriteSidecars(unittest.TestCase):
    def test_lieder_writes_a_sidecar(self) -> None:
        self.assertIn("write_sidecar", _calls("convert_lieder.py"))

    def test_ossq_writes_a_sidecar(self) -> None:
        self.assertIn("write_sidecar", _calls("convert_ossq.py"))

    def test_pdmx_writes_a_sidecar_and_renders_from_source(self) -> None:
        # Both halves matter. A sidecar without source rendering is the anti-signal this
        # module exists to prevent, so the two are asserted together rather than apart.
        calls = _calls("convert_pdmx.py")
        self.assertIn("write_sidecar", calls)
        self.assertIn("extract_window", calls)
        self.assertNotIn("_tokens_to_svg", calls)


class TestIneligibleCorporaDoNot(unittest.TestCase):
    """A corpus whose image is re-rendered from tokens must not carry source labels.

    If one of these ever starts writing a sidecar, either the rendering changed and the
    reasoning above needs revisiting, or a mistake has been made that would put
    systematically wrong beams into training - so this fails loudly rather than trusting
    a comment.
    """

    def test_musetrainer_does_not_write_a_sidecar(self) -> None:
        self.assertNotIn("write_sidecar", _calls("convert_musetrainer.py"))

    def test_musetrainer_still_renders_from_tokens(self) -> None:
        # The reason it is ineligible. If this stops being true the decision changes.
        self.assertIn("_tokens_to_svg", _calls("convert_musetrainer.py"))


if __name__ == "__main__":
    unittest.main()


class TestConvertersImportWhatTheyUse(unittest.TestCase):
    """A NameError inside a rendering helper is swallowed and looks like a bad score.

    _source_to_svg catches exceptions and returns None so one unrenderable window cannot
    kill a 10,000-file conversion. That is right, and it means a missing import produces a
    silent zero-output run rather than a crash - which is exactly what happened: every
    window failed with "name 'inject_musicxml_markings' is not defined" while the
    conversion reported progress normally.
    """

    def test_pdmx_imports_every_name_its_renderer_uses(self) -> None:
        import training.omr_datasets.convert_pdmx as module

        for name in (
            "inject_musicxml_markings",
            "_render_svg_in_subprocess",
            "_VEROVIO_FONTS",
            "_svg_to_png",
            "extract_window",
        ):
            self.assertTrue(hasattr(module, name), f"convert_pdmx is missing {name}")

    def test_the_renderer_compiles_against_those_names(self) -> None:
        # Importing is not enough - the function body has to resolve them.
        import xml.etree.ElementTree as ET

        from training.omr_datasets.convert_pdmx import _source_to_svg

        # A score Verovio may or may not render; what matters is that it fails on the
        # rendering rather than on a NameError.
        score = ET.fromstring(
            '<score-partwise version="3.1"><part-list><score-part id="P1">'
            "<part-name>P</part-name></score-part></part-list>"
            '<part id="P1"><measure number="1"><attributes><divisions>1</divisions>'
            '<clef><sign>G</sign><line>2</line></clef></attributes>'
            "<note><pitch><step>C</step><octave>5</octave></pitch>"
            "<duration>1</duration><type>quarter</type></note></measure></part>"
            "</score-partwise>"
        )
        try:
            _source_to_svg(score)
        except NameError as missing:  # pragma: no cover - the failure this pins
            self.fail(f"renderer references an unimported name: {missing}")
