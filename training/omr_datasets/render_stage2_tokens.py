"""Renders `extract_stage2_pairs.py`'s own `.tokens` files back into real notation
images, via the same `generate_xml` (tokens -> MusicXML) this project already uses
elsewhere (`training/validate_music_xml_conversion.py`, `convert_musetrainer.py`)
plus MuseScore's own renderer (`render_lieder_ground_truth.py`'s own `xvfb-run -a
mscore` pattern - `-platform offscreen` does not work on this box, confirmed this
session).

Built so a human reviewing `stage2_pair_review_server.py` can see real notation
next to the scan crop, instead of reading the raw six-column token table by eye -
raw tokens are how this pipeline's own correctness was first spot-checked earlier
this session, but they don't scale to a systematic review pass.

Batches many token files into one MuseScore `-j`/`--job` conversion file rather
than invoking `mscore` once per file - confirmed this session that batching cuts
render time roughly 3x by avoiding a fresh Qt/app startup per file (2.5s for 3
files batched vs. a much larger per-file constant otherwise). MuseScore's own
`-o file.png` always appends `-1` to a single-page export's filename; renamed to
the plain expected name here so the review server doesn't need to special-case it.
"""

# flake8: noqa: T201

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.omr_datasets.stage2_pair_review_server import parse_stem
from training.transformer.training_vocabulary import read_tokens

#: How many files one `mscore -j` invocation renders - large enough to amortize
#: startup cost, small enough that one hung/slow file doesn't stall a huge batch.
BATCH_SIZE = 200


def build_job(
    tokens_paths: list[Path], musicxml_dir: Path, out_dir: Path, title: bool = True
) -> list[dict]:
    """One job-file entry per token file: writes its generated MusicXML into
    `musicxml_dir` and returns the `{"in": ..., "out": ...}` entries `mscore -j`
    expects - `out` is the plain expected png path (before MuseScore's own `-1`
    suffix is added and then stripped back off by `_destage_output`).
    """
    job = []
    for tokens_path in tokens_paths:
        stem = tokens_path.stem
        symbols = read_tokens(str(tokens_path))
        # `title=False` renders the staff alone. A page titled with its own filename
        # is useful when a human is browsing renderings by name, and pure noise when
        # the rendering sits beside a photograph of the same staff for comparison.
        xml = generate_xml(
            XmlGeneratorArguments(None, None, None), [symbols], stem if title else ""
        )
        musicxml_path = musicxml_dir / f"{stem}.musicxml"
        ET.ElementTree(xml).write(musicxml_path, encoding="unicode", xml_declaration=True)
        job.append({"in": str(musicxml_path), "out": str(out_dir / f"{stem}.png")})
    return job


def destage_output(out_dir: Path, stem: str) -> None:
    """Renames MuseScore's own `{stem}-1.png` (its single-page export naming) to
    the plain `{stem}.png` the review server actually looks up."""
    staged = out_dir / f"{stem}-1.png"
    if staged.exists():
        staged.rename(out_dir / f"{stem}.png")


def render_batch(
    tokens_paths: list[Path], musicxml_dir: Path, out_dir: Path, title: bool = True
) -> None:
    job = build_job(tokens_paths, musicxml_dir, out_dir, title)
    job_path = musicxml_dir / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    subprocess.run(  # noqa: S603
        ["xvfb-run", "-a", "mscore", "-j", str(job_path)],  # noqa: S607
        check=True, capture_output=True,
    )
    for entry in job:
        destage_output(out_dir, Path(entry["in"]).stem)


def render_order_key(path: Path) -> tuple:
    """Numeric `(score_id, system, voice)` order - the same order a human reads a
    score's own page top to bottom - rather than plain filename sort, which puts
    `"sys10"` before `"sys2"` and made early systems in a review lag noticeably
    behind later ones for no good reason. An unparseable stem (shouldn't happen
    given these are our own extractor's own filenames) sorts last rather than
    raising, so one odd file can't block the whole batch.
    """
    parsed = parse_stem(path.stem)
    return parsed if parsed is not None else (path.stem, 10**9, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--tokens-dir", type=Path, required=True,
        help="extract_stage2_pairs.py's --out dir (holds the .tokens files).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Where rendered pngs go.")
    parser.add_argument(
        "--scratch", type=Path, required=True,
        help="Scratch dir for generated .musicxml + job files (not needed afterward).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    all_tokens = sorted(args.tokens_dir.glob("*.tokens"), key=render_order_key)
    pending = [p for p in all_tokens if not (args.out / f"{p.stem}.png").exists()]
    print(f"{len(all_tokens)} token files, {len(pending)} not yet rendered")

    ok = failed = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        try:
            render_batch(batch, args.scratch, args.out)
        except subprocess.CalledProcessError:
            # `mscore -j` under `check=True` discards the whole batch when any one file
            # in it fails, so batching for throughput silently costs the good files
            # alongside the bad one - and its own message is unhelpful (empty stderr,
            # "batch at 400 FAILED"), pointing away from that being what happened. This
            # cost 1,515 of 3,715 pairs on the first full run; shrinking the batch
            # recovered 78% of them, retrying singly recovers all but the genuinely
            # unrenderable.
            recovered = 0
            for one in batch:
                try:
                    render_batch([one], args.scratch, args.out)
                    recovered += 1
                except subprocess.CalledProcessError:  # noqa: PERF203
                    failed += 1
            print(f"batch at {start} failed; {recovered}/{len(batch)} recovered singly")
        rendered_now = sum(1 for p in batch if (args.out / f"{p.stem}.png").exists())
        ok += rendered_now
        failed += len(batch) - rendered_now
        print(f"{start + len(batch)}/{len(pending)} done ({ok} rendered, {failed} missing)")

    print(f"{ok} rendered, {failed} failed/missing")


if __name__ == "__main__":
    main()
