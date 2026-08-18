"""
The automatic-beaming rule, scored on OLiMPiC's whole scores - so beam.level.1's 0.815 on
OLiMPiC (27.82) can be judged against a baseline, the way Gate C judged it against OSSQ.

beam_baseline.py's rule itself (measure_part) is generic MusicXML, needing only a <part>
element's own measures and durations - nothing about it is OSSQ-specific. Only its file
discovery (score_files) is, globbing OSSQ's scores/*/* layout. This reuses the rule
unchanged and supplies OLiMPiC's own files instead.

Whole scores, not the systemwise samples this design has used for training and scoring
everywhere else. score_files's own docstring already measured why: segmenting at a system
break cuts beam groups and restarts the divisions/time-signature context, which inflates the
rule - 91.9% measured on segments against 79.4% on whole scores, for OSSQ. The systemwise
files under olimpic-1.0-scanned/samples/ are exactly that kind of segment. 27.41's whole-
score OpenScore Lieder .mxl downloads - kept for the lyric join, unrelated to this track -
are the whole scores this rule needs, and piano_part_id from lieder_voice.py selects the
same part OLiMPiC's own build selects, so the rule sees what the heads were trained to
predict.
"""

# flake8: noqa: T201

import argparse
from pathlib import Path

from training.omr_datasets.beam_baseline import Baseline, measure_part
from training.omr_datasets.convert_olimpic import partition
from training.omr_datasets.lieder_voice import piano_part_id, read_mxl


def score_id_of(sample: str) -> str:
    """The Lieder score id a sample name starts with: "samples/6583512/p3-s4" -> "6583512"."""
    return sample.split("/")[1]


def measure_olimpic(mxl_dir: Path, score_ids: set[str]) -> Baseline:
    baseline = Baseline()
    for score_id in sorted(score_ids):
        mxl_path = mxl_dir / f"lc{score_id}.mxl"
        if not mxl_path.is_file():
            continue
        try:
            full_score = read_mxl(mxl_path)
        except Exception as broken:  # noqa: BLE001 - reported, not fatal to the run
            print(f"skipping {score_id}: {broken}")
            continue
        target_id = piano_part_id(full_score)
        piano_parts = [
            part for part in full_score.getroot().findall("part") if part.get("id") == target_id
        ]
        for part in piano_parts:
            measure_part(part, baseline)
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--samples-root", type=Path, required=True, help="An olimpic-1.0-scanned dir."
    )
    parser.add_argument("--mxl", type=Path, required=True, help="Dir of lc<id>.mxl whole scores.")
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    samples = partition(args.samples_root, args.split)
    score_ids = {score_id_of(sample) for sample in samples}
    print(f"{len(score_ids)} distinct scores in the {args.split} split")

    baseline = measure_olimpic(args.mxl, score_ids)
    print(baseline.describe())


if __name__ == "__main__":
    main()
