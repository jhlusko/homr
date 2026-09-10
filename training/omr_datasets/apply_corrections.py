"""Turn reviewer-corrected MusicXML back into corpus labels.

`docs/writeups/review_server.py` stores what a reviewer saved from the editor, exactly
as they saved it, and stops there. This is the step that converts those files into the
token format and patches the corpus - separately, explicitly, and on the machine that
has both the corpus and torch.

Why it is a separate step rather than something the server does:

* A browser POST is not an auditable way to mutate training data. A correction that
  silently replaced a label would be indistinguishable from the truncation defect the
  rebuild exists to undo.
* MusicXML -> tokens is lossy in ways worth seeing before committing to them. The
  round trip drops anything the vocabulary cannot express, so a correction can quietly
  come back shorter than it went in. The dry run prints that.
* The corpus has a contract the audit enforces: a label's measure-divider count must
  equal its aligned span. A correction that changes the bar count breaks it, and that
  is a decision, not a detail - so it is refused unless asked for explicitly.

Nothing is written without `--apply`, and every file rewritten is backed up first.
"""

# flake8: noqa: T201

import argparse
import json
import shutil
from pathlib import Path

from homr.circle_of_fifths import strip_naturals
from training.omr_datasets.audit_clean_stage2_pairs import MEASURE_DIVIDERS
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.omr_datasets.notation_sidecar import write_sidecar
from training.transformer.training_vocabulary import read_tokens, token_lines_to_str


def dividers(symbols: list) -> int:
    return sum(symbol.rhythm in MEASURE_DIVIDERS for symbol in symbols)


def corpus_tokens_for(stem: str, manifests: list[Path]) -> Path | None:
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image, tokens = line.split(",", 1)
            if Path(image).stem == stem:
                return Path(tokens)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corrections", type=Path, required=True,
                        help="the store review_server.py writes")
    parser.add_argument("--manifest", type=Path, nargs="+", required=True,
                        help="corpus manifest(s) naming the .tokens each stem lives in")
    parser.add_argument("--apply", action="store_true",
                        help="write. Without this nothing is modified.")
    parser.add_argument("--allow-span-change", action="store_true",
                        help="accept a correction that changes the measure count")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    files = sorted(args.corrections.rglob("*.musicxml"))
    # `id.vN.musicxml` are superseded earlier attempts kept for provenance, not
    # corrections to apply.
    files = [f for f in files if ".v" not in f.stem.rsplit("-", 1)[-1] or not f.stem.split(".v")[-1].isdigit()]
    print(f"{len(files)} correction(s) in {args.corrections}")

    results = []
    applied = skipped = failed = 0
    for path in files:
        stem = path.stem
        entry: dict = {"stem": stem, "correction": str(path)}
        target = corpus_tokens_for(stem, args.manifest)
        if target is None:
            entry.update(status="no corpus row", note="stem not in any manifest given")
            results.append(entry); skipped += 1
            print(f"  SKIP {stem}: not in any manifest")
            continue
        try:
            voices = music_xml_string_to_tokens(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            entry.update(status="unparseable", error=str(exc))
            results.append(entry); failed += 1
            print(f"  FAIL {stem}: {exc}")
            continue
        if len(voices) != 1:
            # A corrected crop is one staff-voice. More than one means the reviewer
            # saved a different score, or the editor added a part.
            entry.update(status="not a single voice", voices=len(voices))
            results.append(entry); failed += 1
            print(f"  FAIL {stem}: {len(voices)} voices, expected 1")
            continue

        # `music_xml_string_to_tokens` yields Measures; the corpus format is the flat
        # symbol stream MeasureCutter produces, with the clef/key/time preamble it
        # maintains. Taking the whole voice is the same call the builder makes for a
        # span, so a correction is converted exactly the way the label it replaces was.
        corrected = strip_naturals(slice_voice_measures(voices[0], 0, len(voices[0])))
        if not corrected:
            entry.update(status="empty after conversion", measures=len(voices[0]))
            results.append(entry); failed += 1
            print(f"  FAIL {stem}: converted to an empty token stream")
            continue
        before = read_tokens(str(target)) if target.is_file() else []
        entry.update(
            target=str(target),
            bars_before=dividers(before),
            bars_after=dividers(corrected),
            symbols_before=len(before),
            symbols_after=len(corrected),
        )
        span_changed = entry["bars_before"] != entry["bars_after"]
        if span_changed and not args.allow_span_change:
            entry["status"] = "refused: measure count changed"
            results.append(entry); skipped += 1
            print(f"  REFUSE {stem}: bars {entry['bars_before']} -> {entry['bars_after']}"
                  f" (pass --allow-span-change to accept)")
            continue

        entry["status"] = "would apply" if not args.apply else "applied"
        print(f"  {'APPLY ' if args.apply else 'DRY   '}{stem}: "
              f"bars {entry['bars_before']}->{entry['bars_after']}, "
              f"symbols {entry['symbols_before']}->{entry['symbols_after']}")
        if args.apply:
            backup = target.with_suffix(target.suffix + ".pre-correction")
            if target.is_file() and not backup.exists():
                shutil.copy2(target, backup)
            target.write_text(token_lines_to_str(corrected), encoding="utf-8")
            write_sidecar(target, corrected)
            applied += 1
        else:
            applied += 1
        results.append(entry)

    print(f"\n{applied} to apply, {skipped} skipped, {failed} failed")
    if not args.apply:
        print("dry run - nothing written. Re-run with --apply.")
    if args.report:
        args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
