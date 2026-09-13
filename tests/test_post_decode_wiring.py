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
    "homr.slur_side": "choose_slur_sides",
    "homr.slur_crossing": "repair_crossings",
    "homr.tie_repair": "repair_ties",
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
        for flag in (
            "beam_repair",
            "stem_arbitration",
            "tuplet_repair",
            "slur_side",
            "slur_crossing",
            "tie_repair",
        ):
            with self.subTest(flag=flag):
                self.assertIn(f"self.{flag} = ", configs)


class TestTheVerificationHarnessReplaysTheSameSequence(unittest.TestCase):
    """The harness that verifies these passes must run all of them.

    `training/transformer/end_to_end_passes.py` hand-copies `main.py`'s post-decode
    sequence so it can run it twice over one decode. A hand-copy of another file's
    sequence rots: `choose_slur_sides` was wired into `main.py` and left out of the
    harness, and the verification run duly reported the pass changing nothing at all.
    A pass that looks worthless because the measurement never called it is worse than
    one that is merely unwired - it comes with evidence against itself.
    """

    HARNESS = Path("training") / "transformer" / "end_to_end_passes.py"

    def test_the_harness_calls_every_pass_main_calls(self) -> None:
        called = _called_names(ROOT / self.HARNESS)
        for module, function in sorted(POST_DECODE_PASSES.items()):
            with self.subTest(module=module):
                self.assertIn(
                    function,
                    called,
                    f"{module}.{function} runs in main.py but not in the harness that "
                    f"verifies these passes, so its effect would measure as zero",
                )


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
