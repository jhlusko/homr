"""Run convert_ossq's real path over real segments, before the crops exist.

The converter has only ever been exercised on hand-built MusicXML. Real segments carry
attributes, partial measures at system boundaries, tuplets, and the 256th notes 27.10
says homr's rhythm vocabulary cannot represent. If any of those raise, the conversion run
fails partway through rather than at the start.

Crops are faked so the guard passes; nothing here touches images.
"""
import shutil, sys, tempfile, traceback
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from training.omr_datasets.convert_ossq import CROP_NAME, build
from training.omr_datasets.ossq_splits import load_split_manifest

source = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
manifest = load_split_manifest()

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "corpus"
    out = Path(tmp) / "out"
    copied = 0
    for segment in sorted(source.glob("scores/*/*/musicxml/unaligned/*.musicxml")):
        score_id, page, system = segment.stem.split(":")
        if manifest.split_for(score_id, "synthetic") is None:
            continue
        work = root / "scores" / segment.parents[2].parent.name / segment.parents[2].name
        (work / "musicxml" / "unaligned").mkdir(parents=True, exist_ok=True)
        shutil.copy(segment, work / "musicxml" / "unaligned" / segment.name)
        crops = work / "images" / "synthetic" / "partwise"
        crops.mkdir(parents=True, exist_ok=True)
        parts = len(ET.parse(segment).getroot().findall("part"))
        for index in range(parts):
            name = CROP_NAME.format(
                score=score_id, page=int(page), system=int(system), part=index + 1
            )
            (crops / name).write_bytes(b"")
        copied += 1
        if copied >= limit:
            break

    print(f"segments staged: {copied}")
    try:
        examples = build(root, out, track="synthetic")
    except Exception:
        traceback.print_exc()
        raise SystemExit("conversion raised - the real run would fail partway through")

    print(f"examples: {len(examples)}")
    lengths = [len(Path(e.tokens).read_text().splitlines()) for e in examples]
    if lengths:
        lengths.sort()
        print(f"token lines: min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]}")
    sidecars = sum(1 for e in examples if Path(str(e.tokens) + ".notation.json").is_file())
    print(f"examples with a notation sidecar: {sidecars} of {len(examples)}")
    by_split = Counter(e.split for e in examples)
    print("by split:", dict(by_split))
