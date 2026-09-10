"""
Does staff detection agree with the music about how many staves a system has?

`convert_ossq.py` converts a system only when its staff crops number exactly as many as
the MusicXML parts, because the crop-to-part pairing is positional and a miscount shifts
every pair after it. That guard is only affordable if it rarely fires, and whether it
rarely fires is a property of the detector on a given track - 27.14 measured scans
reporting five to nine staves in a four-part system, which synthetic renders do not do.

This reads the detections already on disk and compares them against the part counts, so
the cost of the guard is known before a training run is spent on whatever survives it.

Detections are counted above the confidence threshold *without* merging overlapping
boxes, which the cropping stage does. So the count here is an upper bound on the crops a
system will produce, and the disagreement it reports is a lower bound.
"""

# flake8: noqa: T201

import argparse
import collections
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

#: What omr-data-preprocessor writes per system: class x1 y1 x2 y2 confidence.
BBOX_SUFFIX = "_yolo_bboxs.txt"


@dataclass
class Agreement:
    matched: int = 0
    mismatched: int = 0
    delta: collections.Counter[int] = field(default_factory=collections.Counter)
    pairs: collections.Counter[tuple[int, int]] = field(default_factory=collections.Counter)

    @property
    def total(self) -> int:
        return self.matched + self.mismatched

    @property
    def rate(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def observe(self, parts: int, detected: int) -> None:
        self.delta[detected - parts] += 1
        self.pairs[(parts, detected)] += 1
        if parts == detected:
            self.matched += 1
        else:
            self.mismatched += 1

    def describe(self) -> str:
        lines = [
            f"systems compared: {self.total:,}",
            f"  detections match the part count: {self.matched:,} ({self.rate:.1%})",
            f"  mismatch (system would be skipped): {self.mismatched:,} "
            f"({1 - self.rate:.1%})",
            "",
            "detected minus parts:",
        ]
        for difference, count in sorted(self.delta.items()):
            lines.append(f"  {difference:+d}: {count:,} ({count / max(self.total, 1):.1%})")
        return "\n".join(lines)


def count_detections(bbox_path: Path, threshold: float) -> int:
    """Boxes at or above the confidence threshold.

    A line with no parsable confidence is counted rather than dropped: it is a box the
    detector emitted, and silently ignoring it would understate the disagreement this
    exists to measure.
    """
    detected = 0
    for line in bbox_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            confidence = float(fields[-1])
        except ValueError:
            detected += 1
            continue
        if confidence >= threshold:
            detected += 1
    return detected


def measure(dataset_root: Path, track: str, threshold: float) -> Agreement:
    agreement = Agreement()
    pattern = f"scores/*/*/images/{track}/systemwise/*{BBOX_SUFFIX}"
    for bbox_path in sorted(dataset_root.glob(pattern)):
        stem = bbox_path.name.removesuffix(BBOX_SUFFIX)
        segment = bbox_path.parents[3] / "musicxml" / "unaligned" / f"{stem}.musicxml"
        if not segment.is_file():
            continue
        try:
            parts = len(ET.parse(segment).getroot().findall("part"))  # noqa: S314
        except ET.ParseError:
            continue
        agreement.observe(parts, count_detections(bbox_path, threshold))
    return agreement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--track", choices=["synthetic", "scanned"], default="synthetic")
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    args = parser.parse_args()

    agreement = measure(args.dataset_root, args.track, args.confidence_threshold)
    print(agreement.describe())
    if agreement.pairs:
        print()
        print("most common (parts, detected) pairs:")
        for (parts, detected), count in agreement.pairs.most_common(8):
            print(f"  parts={parts} detected={detected}: {count:,}")


if __name__ == "__main__":
    main()
