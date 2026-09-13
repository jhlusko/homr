"""The post-decode passes must actually be reachable from the pipeline.

This exists because both `beam_repair` and `stem_arbitration` were written, tested, and
left with no caller - the same defect they were built to fix in four other modules. A unit
test proves a function works; only this proves anything calls it.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Modules that change decoded notation and must be invoked from the pipeline.
POST_DECODE_PASSES = {
    "homr.beam_repair": "repair_beams",
    "homr.stem_arbitration": "arbitrate_stems",
    "homr.tuplet_repair": "repair_symbols",
}


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


class TestPostDecodeWiring(unittest.TestCase):
    def test_every_post_decode_pass_is_called_from_main(self) -> None:
        called = _called_names(ROOT / "homr" / "main.py")
        for module, function in sorted(POST_DECODE_PASSES.items()):
            with self.subTest(module=module):
                self.assertIn(
                    function,
                    called,
                    f"{module}.{function} is never called from homr/main.py - it would "
                    f"ship as unreachable code",
                )

    def test_each_pass_has_a_config_flag(self) -> None:
        """So a regression can be switched off without a deploy."""
        configs = (ROOT / "homr" / "transformer" / "configs.py").read_text(encoding="utf-8")
        for flag in ("beam_repair", "stem_arbitration", "tuplet_repair"):
            with self.subTest(flag=flag):
                self.assertIn(f"self.{flag} = ", configs)


class TestBothDecodePathsMaskTheSameWay(unittest.TestCase):
    """Every path that attaches notation must silence the untrained beam levels.

    There are two - the shipping ONNX decode and the torch decode the galleries and
    evaluations run on - and they attach notation in separate files. When only one of
    them masked, every figure measured through the other described a pipeline nobody
    ships.
    """

    ATTACHING_DECODERS = (
        Path("homr") / "transformer" / "decoder_inference.py",
        Path("training") / "architecture" / "transformer" / "decoder.py",
    )

    def test_every_decoder_that_decodes_a_note_also_masks_it(self) -> None:
        for relative in self.ATTACHING_DECODERS:
            with self.subTest(path=str(relative)):
                called = _called_names(ROOT / relative)
                self.assertIn("decode_note", called, f"{relative} no longer attaches notation")
                self.assertIn(
                    "mask_untrained_beams",
                    called,
                    f"{relative} decodes notation without masking the beam levels the "
                    f"head was never supervised on",
                )


if __name__ == "__main__":
    unittest.main()
