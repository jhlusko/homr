"""Replay crops with the pre-fix grace cursor under otherwise current code.

Diagnostic only: no corpus writes or runtime switch. Arguments pass through to
 audit_lieder_rest_stretch; retain the source snapshot alongside the report.
"""
from unittest.mock import patch

from training.omr_datasets.music_xml_parser import TokensMeasure
from training.onnx.audit_lieder_rest_stretch import main


def legacy_grace_cursor(original):
    def append(self, staff, is_chord, duration, invisible, symbol):
        grace = 'G' in symbol.rhythm
        onset = self.symbols[-1].position if is_chord else self.current_position
        before = len(self.symbols)
        original(self, staff, is_chord, duration, invisible, symbol)
        if grace:
            self.current_position = onset + duration
            for entry in self.symbols[before:]:
                entry.grace_order = 0
    return append


if __name__ == '__main__':
    with patch.object(TokensMeasure, 'append_note', legacy_grace_cursor(TokensMeasure.append_note)):
        main()
