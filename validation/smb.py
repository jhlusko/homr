# flake8: noqa: S101

"""
PRAIG/SMB benchmark: measures OMR quality on the SMB dataset.

Data source: PRAIG/SMB (gated HuggingFace dataset - see below for access instructions).
Tool:        selectable via --tool (default: music21).
Check:       ned_benchmark.run_benchmark

Before running:
  1. Request/accept access at https://huggingface.co/datasets/PRAIG/SMB
  2. Run `huggingface-cli login` with a read-permission token.
"""

import argparse
import tempfile
from pathlib import Path

from validation.ned_benchmark import Sample, run_benchmark, update_ned_scores
from validation.tools import TOOLS


def _sample_kern(sample: dict[object, object]) -> str:
    """Return the full-page reference transcription from an SMB dataset row.

    SMB publishes a page-level ``**kern`` transcription as well as per-region
    annotations.  The page reference is the benchmark target; joining regions
    is retained only for older dataset revisions that lack it.
    """
    page = sample.get("page")
    if isinstance(page, dict):
        kern = page.get("kern")
        if isinstance(kern, str) and kern.strip():
            return kern

    kern = sample.get("kern")
    if isinstance(kern, str) and kern.strip():
        return kern

    regions = sample.get("regions")
    if not isinstance(regions, list):
        return ""
    return "\n".join(
        kern
        for region in regions
        if isinstance(region, dict)
        and isinstance(kern := region.get("kern"), str)
        and kern.strip()
    )


def get_smb_samples(image_dir: Path) -> list[Sample]:
    from datasets import Image, load_dataset  # type: ignore  # noqa: PLC0415

    ds = load_dataset("PRAIG/SMB")["test"]
    ds = ds.cast_column("image", Image(decode=False))  # Don't decode on image colume.
    result = []
    for i, sample in enumerate(ds):
        assert isinstance(sample, dict)
        assert sample["image"]["bytes"]

        kern = _sample_kern(sample)
        if not kern:
            raise ValueError(f"SMB sample {i} has no page or region **kern transcription")
        image_path = image_dir / f"{i}.png"
        image_path.write_bytes(sample["image"]["bytes"])

        result.append(Sample(str(i), kern, image_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="OMR-NED benchmark for PRAIG/SMB.")
    parser.add_argument("--limit", type=int, default=None, help="Only process N samples.")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--verbose", action="store_true", help="Print traceback on failure.")
    parser.add_argument("--output", type=str, default=None, help="Path to SQLite output file.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Recompute NED from stored kern/output data without re-running the tool.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Skip samples already present in --output and append new results.",
    )
    parser.add_argument(
        "--tool",
        choices=list(TOOLS),
        default="music21",
        help="OMR tool to benchmark (default: music21).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Samples per batch for tools that support batch_run (default: 10).",
    )
    parser.add_argument(
        "--kern-parser",
        choices=["native", "music21"],
        default="native",
        dest="kern_parser",
        help=(
            "Parser for the kern ground-truth side of NED comparison. "
            "'native' (default) = built-in humdrum parser; 'music21' = music21-based parser."
        ),
    )
    parser.add_argument(
        "--xml-parser",
        choices=["native", "music21", "musicdiff", "musicdiff_detailed"],
        default="native",
        dest="xml_parser",
        help=(
            "Parser/method for the NED comparison. "
            "'native' (default) = built-in token pipeline; "
            "'music21' = music21-based token pipeline; "
            "'musicdiff' = musicdiff holistic OMR-NED (component NEDs not available)."
            "'musicdiff_detailed' = musicdiff holistic OMR-NED."
        ),
    )
    args = parser.parse_args()
    output_db = args.output or f"smb_{args.tool}.db"

    if args.update:
        update_ned_scores(
            output_db,
            verbose=args.verbose,
            kern_parser=args.kern_parser,
            xml_parser=args.xml_parser,
            limit=args.limit,
        )
        return

    tool = TOOLS[args.tool]
    with tempfile.TemporaryDirectory() as image_dir:
        run_benchmark(
            get_smb_samples(image_dir=Path(image_dir)),
            tool,
            args.workers,
            limit=args.limit,
            verbose=args.verbose,
            output_db=output_db,
            continue_run=args.continue_run,
            batch_size=args.batch_size,
            kern_parser=args.kern_parser,
            xml_parser=args.xml_parser,
        )


if __name__ == "__main__":
    main()
