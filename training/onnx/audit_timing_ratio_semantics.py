"""Audit reference-side ratio labels against co-onset tremolo annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path

from homr import music_xml_generator as generator
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    for row in json.loads(args.evaluation.read_text()):
        symbols = []
        for i, raw in enumerate(row["reference"]):
            symbol = EncodedSymbol(**raw)
            symbol.probe_index = i
            symbols.append(symbol)
        for group in generator.group_into_chords(symbols):
            tremolo = any("tremolo" in s.articulation for s in group.symbols)
            for symbol in group.symbols:
                if row["targets"][symbol.probe_index].get("timing.ratio") == "2:1":
                    counts[(row["score"], tremolo)] += 1
    write(
        args.out,
        dict(
            counts=[dict(score=k[0], co_onset_tremolo=k[1], support=v) for k, v in counts.items()],
            evaluation_sha256=sha(args.evaluation),
            source_sha256=sha(__file__),
            limitation="Reference-side diagnostic, never an inference feature. Co-onset ornament association; not ordinary-duplet qualification.",
        ),
    )


if __name__ == "__main__":
    main()
