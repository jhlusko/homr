"""Locate grace groups hidden by a structural metadata symbol during XML export."""
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from homr.music_xml_generator import group_into_chords
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = ET.parse(args.reference).getroot()
    parts = music_xml_string_to_tokens(ET.tostring(root, encoding='unicode'))
    groups = []
    for part_index, part in enumerate(parts):
        for measure_index, measure in enumerate(part):
            for group in group_into_chords(list(measure)):
                if (any(s.rhythm.startswith('note') and s.rhythm.endswith('G') for s in group.symbols)
                        and not group.symbols[0].rhythm.startswith(('note', 'rest'))):
                    groups.append(dict(part_index=part_index, source_measure_index=measure_index,
                        first=group.symbols[0].rhythm, symbols=[str(s) for s in group.symbols]))
    write(args.out, dict(reference=str(args.reference), reference_sha256=sha(args.reference),
        groups=groups, code_sha256=sha(__file__),
        limitation='Locates current parser/export grouping collisions; no source mutation.'))


if __name__ == '__main__':
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    main()
