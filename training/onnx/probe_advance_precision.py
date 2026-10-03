"""Reproduce printed-duration precision loss without modifying runtime source."""
import argparse
from pathlib import Path

from homr.transformer.structured_notation import AdvanceClass, NoteNotation, StemDirection
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.ablate_real_reference_advance import export_with_counts
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_full_page_musicxml import staff_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    voice = [EncodedSymbol('timeSignatureBeats_4'), EncodedSymbol('timeSignature/4'),
        EncodedSymbol('note_12', 'C4', '_', '_', '_', 'upper'), EncodedSymbol('chord'),
        EncodedSymbol('note_4', 'E4', '_', '_', '_', 'upper',
            notation=NoteNotation((), StemDirection.NOT_APPLICABLE, (), advance=AdvanceClass.EIGHTH)),
        EncodedSymbol('note_8', 'G4', '_', '_', '_', 'upper'), EncodedSymbol('barline')]
    cases = {}
    for name, enabled in [('baseline', False), ('advance', True)]:
        xml, _, _ = export_with_counts([voice], enabled)
        cases[name] = dict(divisions=[x.text for x in xml.findall('.//divisions')],
                           events=staff_events(xml))
    write(args.out, dict(cases=cases, code_sha256=sha(__file__),
                        limitation='Synthetic mixed-duration HOMR group, not model predictions.'))


if __name__ == '__main__':
    main()
