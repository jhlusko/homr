"""Regenerate live Phase-1 cross-staff-reranking examples with the deployed model.

The script deliberately runs the same detector/encoder/decoder twice only at the
candidate-selection boundary: greedy candidates are compared with the candidates
selected by ``rerank_staff_candidates``.  It writes a small, reviewable gallery
bundle (source scan plus before/after MusicXML) for systems whose structural
finding count genuinely decreases.  It is not an accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import homr.transformer.staff2score as staff2score_module
from homr.cross_staff_consistency import (
    check_barline_positions,
    check_measure_durations,
    staves_by_system,
)
from homr.cross_staff_rerank import rerank_staff_candidates, rhythm_candidates_for_staff
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string
from homr.staff_parsing import _plan_systems, parse_staffs
from homr.transformer.configs import Config as TransformerConfig


def _call_index_to_system_voice(plan, number_of_voices: int) -> dict[int, tuple[int, int]]:
    """Mirror parse_staffs' voice-major decode order."""
    mapping: dict[int, tuple[int, int]] = {}
    call_index = 0
    for voice in range(number_of_voices):
        for system in range(len(plan.systems)):
            if plan.staff_for_voice(system, voice) is not None:
                mapping[call_index] = (system, voice)
                call_index += 1
    return mapping


def _finding_count(staves) -> int:
    return len(check_barline_positions(staves)) + len(check_measure_durations(staves))


def build_examples(images: list[Path], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    original_predict = staff2score_module.Staff2Score.predict
    manifest: list[dict] = []

    for page_index, image_path in enumerate(images, 1):
        captured: list[dict] = []

        def capturing_predict(self, image):
            transformed = staff2score_module._transform(image=image)
            context = self.encoder.generate(transformed)
            dtype = np.float16 if self.decoder.fp16 else np.float32
            if context.dtype != dtype:
                context = context.astype(dtype)
            captured.append({"context": context, "decoder": self.decoder})
            return original_predict(self, image)

        staff2score_module.Staff2Score.predict = capturing_predict
        try:
            config = ProcessingConfig(False, False, False, False, -1, True, True, False, False, None)
            multi_staffs, image, debug, _title_future, _ = detect_staffs_in_image(str(image_path), config)
            voices = parse_staffs(
                debug,
                multi_staffs,
                image,
                config=TransformerConfig(),
                enable_phase1_rerank=False,
            )
        finally:
            staff2score_module.Staff2Score.predict = original_predict

        plan = _plan_systems(multi_staffs)
        presence = [
            [plan.staff_for_voice(system, voice) is not None for voice in range(len(voices))]
            for system in range(len(plan.systems))
        ]
        before_systems = list(staves_by_system(voices, presence))
        mapping = _call_index_to_system_voice(plan, len(voices))
        by_system: dict[int, dict[int, dict]] = {}
        for call_index, location in mapping.items():
            by_system.setdefault(location[0], {})[location[1]] = captured[call_index]

        candidates: list[tuple[int, int, int, list, list]] = []
        for system_index, before_staves in enumerate(before_systems):
            before = _finding_count(before_staves)
            available = by_system.get(system_index, {})
            voices_here = sorted(available)
            if before == 0 or len(voices_here) < 3:
                continue
            candidate_staves = {}
            for staff_index, voice in enumerate(voices_here):
                capture = available[voice]
                candidate_staves[staff_index] = rhythm_candidates_for_staff(
                    capture["decoder"],
                    np.array([[1]], dtype=np.int64),
                    np.array([[0]], dtype=np.int64),
                    max_forks=3,
                    context=capture["context"],
                )
            reranked = rerank_staff_candidates(candidate_staves)
            greedy = [candidate_staves[index][0] for index in range(len(voices_here))]
            after = [reranked[index] for index in range(len(voices_here))]
            after_count = _finding_count(after)
            if after_count < before:
                candidates.append((before - after_count, system_index, before, greedy, after))

        if not candidates:
            print(f"[{page_index}/{len(images)}] {image_path.name}: no improved system", flush=True)
            continue
        reduction, system_index, before, greedy, after = max(candidates, key=lambda row: (row[0], row[2]))
        stem = f"{page_index:02d}_{image_path.stem}_system{system_index + 1}"
        before_path = output / f"{stem}__before.musicxml"
        after_path = output / f"{stem}__after.musicxml"
        before_path.write_text(xml_to_string(generate_xml(XmlGeneratorArguments(None, None, None), greedy, "Before Phase-1 reranking")))
        after_path.write_text(xml_to_string(generate_xml(XmlGeneratorArguments(None, None, None), after, "After Phase-1 reranking")))
        scan_path = output / f"{stem}__scan{image_path.suffix.lower()}"
        shutil.copy2(image_path, scan_path)
        entry = {
            "id": stem,
            "source": str(image_path),
            "system": system_index + 1,
            "before_findings": before,
            "after_findings": before - reduction,
            "reduction": reduction,
        }
        manifest.append(entry)
        print(f"[{page_index}/{len(images)}] {image_path.name}: system {system_index + 1}, {before} -> {before - reduction}", flush=True)

    manifest.sort(key=lambda row: (row["reduction"], row["before_findings"]), reverse=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, help="one absolute source-image path per line")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    images = [Path(line.strip()) for line in args.images.read_text().splitlines() if line.strip()]
    build_examples(images, args.output)


if __name__ == "__main__":
    main()
