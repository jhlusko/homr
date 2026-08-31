"""Render isolated structured-head A/Bs from the same held-out crop decode.

This deliberately bypasses staff detection: the structured heads were trained and
scored on already-cropped staves.  Each control and treatment therefore shares identical
pixels, core autoregressive symbols, and structured logits; only one field is retained
when MusicXML is generated.  It is an export-integration review, not a claim about the
full detector pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Iterable

import torch

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string
from homr.transformer.configs import Config
from homr.transformer.structured_notation import (
    AdvanceClass,
    DynamicMark,
    NoteNotation,
    StemDirection,
    TieState,
    empty_beam_levels,
)
from training.architecture.transformer.tromr_arch import TrOMR
from training.transformer.data_loader import DataLoader
from training.transformer.train_structured_heads import load_pinned


FIELDS = ("tie", "stem", "slur")


def image_for(tokens: str) -> Path:
    base = Path(tokens.removesuffix(".tokens"))
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = base.with_suffix(extension)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No crop beside {tokens}")


def notation_for(field: str, notation: NoteNotation) -> NoteNotation:
    """Keep exactly one structured field; all other fields are render no-ops."""
    return NoteNotation(
        beam_levels=empty_beam_levels(),
        stem=notation.stem if field == "stem" else StemDirection.NOT_APPLICABLE,
        slurs=notation.slurs if field == "slur" else (),
        tie=notation.tie if field == "tie" else TieState.NONE,
        dynamic=DynamicMark.NONE,
        advance=AdvanceClass.NOT_APPLICABLE,
    )


def symbols_for(field: str | None, symbols: Iterable) -> list:
    result = copy.deepcopy(list(symbols))
    for symbol in result:
        if field is None:
            symbol.notation = None
        elif symbol.notation is not None:
            symbol.notation = notation_for(field, symbol.notation)
    return result


def count(xml: str, tag: str) -> int:
    return xml.count(f"<{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokens = [line.strip() for line in args.sample.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crops").mkdir(exist_ok=True)
    config = Config()
    config.enable_structured_heads = True
    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    assert model.decoder.structured_heads is not None
    model.decoder.structured_heads.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
    )
    model.eval_mode()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    manifest = []
    for number, token_path in enumerate(tokens, start=1):
        image = image_for(token_path)
        item = DataLoader([f"{image},{token_path}"], config, is_validation=True)[0]
        with torch.no_grad():
            decoded = model.generate(item["inputs"].unsqueeze(0).to(device))
        key = f"pdmx_{number:02d}"
        crop = args.out / "crops" / f"{key}{image.suffix}"
        shutil.copy2(image, crop)
        control = xml_to_string(generate_xml(XmlGeneratorArguments(), [symbols_for(None, decoded)], key))
        (args.out / f"{key}__none.musicxml").write_text(control, encoding="utf-8")
        record = {"id": key, "tokens": token_path, "crop": crop.name, "effects": {}}
        for field in FIELDS:
            rendered = xml_to_string(
                generate_xml(XmlGeneratorArguments(), [symbols_for(field, decoded)], key)
            )
            (args.out / f"{key}__{field}.musicxml").write_text(rendered, encoding="utf-8")
            tag = {"tie": "tie", "stem": "stem", "slur": "slur"}[field]
            record["effects"][field] = {"control": count(control, tag), "with": count(rendered, tag)}
        manifest.append(record)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
