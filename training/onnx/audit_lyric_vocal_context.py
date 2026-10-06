"""Report context-selector abstentions as well as supported-page regressions."""

import argparse
import json
from pathlib import Path

from training.onnx.lyric_row_routing import route_rows
from training.onnx.lyric_vocal_context import vocal_voice_candidates
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_expanded_lyric_attachments import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pages", "decode", "selected", "positions", "baseline", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results, hashes = [], {str(args.pages): sha(args.pages)}
    for index, page in enumerate(json.loads(args.pages.read_text())):
        folder = args.decode / f"page-{index}"
        paths = [
            args.selected / f"{page['score']}.json",
            args.positions / f"{page['score']}.json",
            args.baseline / f"{page['score']}-predictions.json",
            folder / "staff-transforms.json",
            folder / "final-note-anchors.json",
        ]
        selected, positioned, old, transforms, final = [json.loads(p.read_text()) for p in paths]
        routing = route_rows(selected["all_page_text"], transforms, recover_row_neighbors=True)
        record = {
            "score": page["score"],
            "candidates": vocal_voice_candidates(final, transforms, routing),
            "baseline_attachments": len(old["predictions"]),
            "native_targets": page["native_targets"],
        }
        try:
            result = predict(folder, selected, positioned, trace_export=True, vocal_context=True)
            record.update(
                status="resolved",
                predictions_unchanged=result["predictions"] == old["predictions"],
                unresolved_candidates_unchanged=result["unresolved_syllables"]
                == old["unresolved_syllables"],
            )
            write(args.out / f"{page['score']}-predictions.json", result)
        except ValueError as error:
            record.update(status="unresolved_context", reason=str(error))
        results.append(record)
        hashes.update(
            {
                str(p): sha(p)
                for p in [
                    *paths,
                    folder / "page.musicxml",
                    folder / "decoded-staves.json",
                    folder / "identity.json",
                ]
            }
        )
    write(
        args.out / "report.json",
        {
            "pages": results,
            "input_sha256": hashes,
            "code_sha256": {
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    Path(__file__).with_name("lyric_vocal_context.py"),
                    Path(__file__).with_name("score_expanded_lyric_attachments.py"),
                ]
            },
            "production_qualified": False,
            "scope": (
                "Diagnostic abstention audit; unresolved pages remain in denominator, no fallback voice assignment"
            ),
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
