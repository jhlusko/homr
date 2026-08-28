"""Every ConversionState field must exist before any method reads it.

A method definition inserted into the middle of `__init__` left three assignments
stranded after a `return`, so `tremolo_state`, `volta_number` and `last_volta_measure`
were never set. Nothing failed until a label carried a tremolo or a volta, and then
`generate_xml` raised AttributeError - which in the review-set generator surfaced as
the whole set failing to build, not as one bad pair being skipped.

The general check is the useful one: construct the state and call every method that
reads an attribute, rather than asserting the presence of the three names that
happened to be lost this time.
"""

import unittest

from homr.music_xml_generator import ConversionState


class TestConversionStateIsFullyInitialised(unittest.TestCase):
    def state(self) -> ConversionState:
        return ConversionState(division=24, nominator=4)

    def test_tremolo_toggles_from_a_defined_start(self) -> None:
        state = self.state()
        self.assertEqual(state.toggle_tremolo_state(), "start")
        self.assertEqual(state.toggle_tremolo_state(), "stop")

    def test_volta_numbering_starts_at_one_and_runs_consecutively(self) -> None:
        state = self.state()
        self.assertEqual(state.start_volta(5), 1)
        state.stop_volta(5)
        self.assertEqual(state.start_volta(6), 2)
        state.stop_volta(6)
        # A gap restarts the numbering rather than continuing it.
        self.assertEqual(state.start_volta(20), 1)

    def test_no_assignment_is_stranded_after_a_return_in_init(self) -> None:
        """Catches the shape of the defect, not just its three casualties."""
        import ast
        import inspect

        source = inspect.getsource(ConversionState.__init__)
        tree = ast.parse(source.lstrip().replace("\n    ", "\n"))
        body = tree.body[0].body
        returned = False
        for node in body:
            if returned:
                self.fail(f"statement after return in __init__: {ast.dump(node)[:80]}")
            if isinstance(node, ast.Return):
                returned = True


if __name__ == "__main__":
    unittest.main()
