"""Do the two post-decode passes move the numbers they were measured to move?

`homr/stem_arbitration.py` and `homr/beam_repair.py` were each measured where they were
built: the stem arbitration over a dumped prediction file scored against engraved labels,
the beam repair over the same dump's vectors. Neither had ever run where it ships - on the
output of a live decode of a real scan, after tuplet repair, inside `homr/main.py`'s
sequence. A measurement that does not transfer is indistinguishable from a pass that is
not wired in, which is exactly how both spent their first day.

This runs the production decode once per crop and then replays `main.py`'s post-decode
sequence twice over copies of it: once with both flags off, once with both on. The decode
is shared deliberately - the arms must differ only by the passes, and a second decode
would add sampling noise to a comparison whose whole point is the difference.

Scored against the corrected sidecar by aligning predicted symbols to engraved ones with
`difflib` over (rhythm, pitch) and counting only matched notes. Notes the decode got wrong
are excluded from both arms identically: this measures what the passes do to notation, not
what the decoder does to tokens, and a stem on a note that is not there has no truth to
compare against.
"""

# flake8: noqa: T201

import argparse
import copy
import difflib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from homr.beam_repair import LEVELS, repair_beams
from homr.staff_parsing import add_image_into_tr_omr_canvas
from homr.stem_arbitration import arbitrate_stems
from homr.transformer.beam_validation import validate_voice
from homr.transformer.configs import Config
from homr.transformer.structured_notation import BeamLevelState
from homr.transformer.vocabulary import EncodedSymbol
from homr.tuplet_repair import repair_symbols
from training.omr_datasets.notation_sidecar import attach_sidecar


@dataclass
class ArmTotals:
    """Everything one arm is judged on, accumulated over crops."""

    staves: int = 0
    notes: int = 0
    stem_correct: int = 0
    beam_level1_correct: int = 0
    beam_vector_correct: int = 0
    invalid_staves: int = 0
    #: Staves the other arm had valid and this one does not. A pass meant to remove
    #: impossibilities can introduce them at the seam between what it rewrote and what it
    #: left alone, and a net rate hides that entirely.
    newly_invalid: int = 0
    findings: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "staves": self.staves,
            "notes": self.notes,
            "stem_accuracy": _ratio(self.stem_correct, self.notes),
            "beam_level1_accuracy": _ratio(self.beam_level1_correct, self.notes),
            "beam_vector_accuracy": _ratio(self.beam_vector_correct, self.notes),
            "invalid_stave_rate": _ratio(self.invalid_staves, self.staves),
            "newly_invalid_staves": self.newly_invalid,
            "findings": dict(self.findings),
        }


def read_tokens(path: Path) -> list[EncodedSymbol]:
    """`training.transformer.training_vocabulary.read_token_lines`, without torch.

    Inlined rather than imported because that module pulls in torch for the tensor half
    of the training vocabulary, and nothing here trains. The chord convention matters and
    is kept exactly: members after the first are preceded by a bare `chord` symbol, which
    is not note-bearing and so does not take a sidecar record.
    """
    symbols: list[EncodedSymbol] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for index, entry in enumerate(line.split("&")):
            if "tieSlur" in entry:
                continue
            parts = entry.strip().split()
            if not parts:
                continue
            if len(parts) == 6:
                rhythm, pitch, lift, articulation, slur, position = parts
            elif len(parts) == 5:
                rhythm, pitch, lift, articulation, slur = parts
                position = "upper"
            elif len(parts) == 4:
                rhythm, pitch, lift, articulation = parts
                slur, position = ".", "upper"
            else:
                continue
            if index:
                symbols.append(EncodedSymbol("chord"))
            symbols.append(EncodedSymbol(rhythm, pitch, lift, articulation, slur, position))
    return symbols


@dataclass
class Crosstab:
    """Where a pass moved a note, judged both before and after.

    A headline that moves the wrong way is not yet a finding: a pass that rewrites many
    notes can lose accuracy while fixing the thing it was built to fix. What separates
    the two is how the rewritten notes were doing beforehand, which only a crosstab says.
    """

    right_to_wrong: int = 0
    wrong_to_right: int = 0
    wrong_to_wrong: int = 0
    right_to_right: int = 0

    @property
    def touched(self) -> int:
        return self.right_to_wrong + self.wrong_to_right + self.wrong_to_wrong + self.right_to_right

    def add(self, before_right: bool, after_right: bool) -> None:
        if before_right and after_right:
            self.right_to_right += 1
        elif before_right:
            self.right_to_wrong += 1
        elif after_right:
            self.wrong_to_right += 1
        else:
            self.wrong_to_wrong += 1

    def as_dict(self) -> dict:
        return {
            "touched": self.touched,
            "wrong_to_right": self.wrong_to_right,
            "right_to_wrong": self.right_to_wrong,
            "wrong_to_wrong": self.wrong_to_wrong,
            "right_to_right": self.right_to_right,
            "net": self.wrong_to_right - self.right_to_wrong,
        }


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _key(symbol: EncodedSymbol) -> tuple[str, str]:
    return (symbol.rhythm, symbol.pitch)


def _notes(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    return [s for s in symbols if s.rhythm.startswith(("note", "rest"))]


def _vectors(symbols: list[EncodedSymbol]) -> list[tuple[BeamLevelState, ...]]:
    out = []
    for symbol in symbols:
        notation = symbol.notation
        out.append(tuple(notation.beam_levels) if notation is not None else ())
    return out


def _score(
    predicted: list[EncodedSymbol], engraved: list[EncodedSymbol], totals: ArmTotals
) -> bool:
    """Add one staff's contribution to `totals`; returns whether the staff validated."""
    totals.staves += 1

    valid = True
    vectors = [v for v in _vectors(predicted) if v]
    if vectors:
        counts = _interior_findings(validate_voice(vectors, LEVELS), vectors)
        if counts:
            valid = False
            totals.invalid_staves += 1
            totals.findings.update(counts)

    left, right = _notes(predicted), _notes(engraved)
    matcher = difflib.SequenceMatcher(
        None, [_key(s) for s in left], [_key(s) for s in right], autojunk=False
    )
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            got, want = left[block.a + offset], right[block.b + offset]
            if got.notation is None or want.notation is None:
                continue
            totals.notes += 1
            if got.notation.stem == want.notation.stem:
                totals.stem_correct += 1
            # Truncated to the trained levels on both sides: the sidecar carries all six
            # the representation supports and the heads exist for four, so comparing full
            # vectors would score every note wrong for a level no head predicts.
            got_levels = tuple(got.notation.beam_levels)[:LEVELS]
            want_levels = tuple(want.notation.beam_levels)[:LEVELS]
            if got_levels[:1] == want_levels[:1]:
                totals.beam_level1_correct += 1
            if got_levels == want_levels:
                totals.beam_vector_correct += 1
    return valid


#: The kinds `training/transformer/beam_validity_audit.py` reports, so the two numbers
#: are the same number.
KINDS = ("unopened", "unclosed", "nested", "hook_at_primary_level", "single_note_groups")


def _crosstab(
    off: list[EncodedSymbol],
    on: list[EncodedSymbol],
    engraved: list[EncodedSymbol],
    beams: Crosstab,
    stems: Crosstab,
    regressions: list[dict] | None = None,
    name: str = "",
) -> None:
    """Count only the notes a pass actually changed, on both arms at once.

    The two arms share a decode and a length - the passes rewrite notation in place and
    never insert or drop a symbol - so they align to the engraved staff by the same
    matching blocks, and a note is comparable in both or neither.
    """
    left, right = _notes(off), _notes(engraved)
    after = _notes(on)
    if len(after) != len(left):
        return
    matcher = difflib.SequenceMatcher(
        None, [_key(s) for s in left], [_key(s) for s in right], autojunk=False
    )
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            before, now, want = (
                left[block.a + offset],
                after[block.a + offset],
                right[block.b + offset],
            )
            if before.notation is None or want.notation is None or now.notation is None:
                continue
            before_beams = tuple(before.notation.beam_levels)[:LEVELS]
            now_beams = tuple(now.notation.beam_levels)[:LEVELS]
            want_beams = tuple(want.notation.beam_levels)[:LEVELS]
            if before_beams != now_beams:
                beams.add(before_beams == want_beams, now_beams == want_beams)
                if regressions is not None and (before_beams == want_beams) != (
                    now_beams == want_beams
                ):
                    regressions.append(
                        {
                            "staff": name,
                            "direction": (
                                "right_to_wrong" if before_beams == want_beams else "wrong_to_right"
                            ),
                            "note": block.a + offset,
                            "rhythm": before.rhythm,
                            "pitch": before.pitch,
                            "engraved": [str(state) for state in want_beams],
                            "head": [str(state) for state in before_beams],
                            "rule": [str(state) for state in now_beams],
                            "context": [
                                {
                                    "rhythm": symbol.rhythm,
                                    "head": (
                                        [
                                            str(state)
                                            for state in symbol.notation.beam_levels[:LEVELS]
                                        ]
                                        if symbol.notation
                                        else []
                                    ),
                                }
                                for symbol in left[
                                    max(0, block.a + offset - 4) : block.a + offset + 5
                                ]
                            ],
                        }
                    )
            if before.notation.stem != now.notation.stem:
                stems.add(
                    before.notation.stem == want.notation.stem,
                    now.notation.stem == want.notation.stem,
                )


def _interior_findings(
    findings: object, vectors: list[tuple[BeamLevelState, ...]]
) -> dict[str, int]:
    """Findings away from the crop's edges, counted per kind.

    A crop is one staff cut out of a system, so a group crossing either edge is correctly
    open there. The audit this is compared against excludes everything before the first
    BEGIN and after the last END for that reason; excluding it differently here would
    make the two figures incomparable.
    """
    first_begin = next(
        (i for i, v in enumerate(vectors) if v and v[0] == BeamLevelState.BEGIN), len(vectors)
    )
    last_end = next(
        (
            i
            for i in range(len(vectors) - 1, -1, -1)
            if vectors[i] and vectors[i][0] == BeamLevelState.END
        ),
        -1,
    )
    counts: dict[str, int] = {}
    for kind in KINDS:
        inside = [i for i in getattr(findings, kind, ()) if first_begin <= i <= last_end]
        if inside:
            counts[kind] = len(inside)
    return counts


def _post_decode(symbols: list[EncodedSymbol], *, passes: bool) -> list[EncodedSymbol]:
    """`homr/main.py`'s sequence, on a copy, with the two flags on or off.

    Tuplet repair runs in both arms because it runs in both configurations of the thing
    under test; only the two passes are switched.
    """
    voice = copy.deepcopy(symbols)
    voice = repair_symbols(voice)[0]
    if passes:
        repair_beams(voice)
        arbitrate_stems(voice)
    return voice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="directory of crops")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--regressions",
        type=Path,
        help="write every note beam repair moved, in either direction, for reading",
    )
    args = parser.parse_args()

    config = Config()
    config.use_gpu_inference = False
    if not os.path.exists(config.filepaths.structured_heads_path):
        raise SystemExit(
            "no structured heads at "
            f"{config.filepaths.structured_heads_path} - both passes would no-op, and an "
            "arm that cannot move is not a control"
        )

    # Imported here so the missing-heads check above fails before a model is loaded.
    from homr.transformer.staff2score import Staff2Score

    inference = Staff2Score(config)

    off, on = ArmTotals(), ArmTotals()
    beam_moves, stem_moves = Crosstab(), Crosstab()
    regressions: list[dict] = []
    sidecars = sorted(args.corpus.glob("*.txt.notation.json"))[: args.limit]
    for index, sidecar in enumerate(sidecars, start=1):
        tokens_path = Path(str(sidecar)[: -len(".notation.json")])
        image_path = tokens_path.with_suffix(".png")
        if not image_path.is_file():
            continue
        engraved = read_tokens(tokens_path)
        try:
            attach_sidecar(tokens_path, engraved)
        except Exception as ex:  # a writer/reader disagreement, not our business here
            print(f"  skipped {tokens_path.name}: {ex}")
            continue

        # Prepared by the shipping canvas function, not a local resize: the encoder takes
        # a fixed 256x1280 and how a crop is fitted into it changes what is decoded.
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        predicted = inference.predict(add_image_into_tr_omr_canvas(image))

        without = _post_decode(predicted, passes=False)
        with_passes = _post_decode(predicted, passes=True)
        was_valid = _score(without, engraved, off)
        is_valid = _score(with_passes, engraved, on)
        if was_valid and not is_valid:
            on.newly_invalid += 1
        if is_valid and not was_valid:
            off.newly_invalid += 1
        _crosstab(
            without,
            with_passes,
            engraved,
            beam_moves,
            stem_moves,
            regressions if args.regressions else None,
            tokens_path.name,
        )

        if index % 25 == 0:
            print(f"  [{index}/{len(sidecars)}] {on.notes:,} scored notes")

    off_report, on_report = off.as_dict(), on.as_dict()
    report = {
        "corpus": str(args.corpus),
        "off": off_report,
        "on": on_report,
        "beam_repair_moves": beam_moves.as_dict(),
        "stem_arbitration_moves": stem_moves.as_dict(),
    }
    print()
    print(f"{'metric':<26}{'off':>12}{'on':>12}{'delta':>12}")
    for name in ("stem_accuracy", "beam_level1_accuracy", "beam_vector_accuracy"):
        a, b = off_report[name], on_report[name]
        print(f"{name:<26}{a:>11.2%}{b:>12.2%}{b - a:>+12.2%}")
    for name in ("invalid_stave_rate",):
        a, b = off_report[name], on_report[name]
        print(f"{name:<26}{a:>11.2%}{b:>12.2%}{b - a:>+12.2%}")
    print(f"\n{off.staves:,} staves, {off.notes:,} scored notes")
    print(f"\nstaves the passes made invalid that were not: {on.newly_invalid:,}")
    print(f"staves the passes made valid that were not:   {off.newly_invalid:,}")
    print(f"findings off: {dict(off.findings)}")
    print(f"findings on:  {dict(on.findings)}")
    for label, table in (("beam repair", beam_moves), ("stem arbitration", stem_moves)):
        counts = table.as_dict()
        print(
            f"\n{label}: changed {counts['touched']:,} notes - "
            f"{counts['wrong_to_right']:,} wrong to right, "
            f"{counts['right_to_wrong']:,} right to wrong, "
            f"{counts['wrong_to_wrong']:,} wrong either way, "
            f"{counts['right_to_right']:,} right either way (net {counts['net']:+,})"
        )

    if args.regressions:
        args.regressions.write_text(json.dumps(regressions, indent=2), encoding="utf-8")
        print(f"wrote {len(regressions):,} regressions to {args.regressions}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
