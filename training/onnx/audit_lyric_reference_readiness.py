"""Bind native lyric references to fixed scan pages without recognizer predictions.

Count-exact page boundaries are not pixel-level syllable/note certification.
OSSQ lyric absence is a negative-control proposal, never positive lyric recall.
"""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import yaml

from training.onnx.audit_full_page_references import certified_page, read_mxl


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_events(root):
    events = []
    for part in root.findall("part"):
        divisions = Fraction(1)
        for measure_index, measure in enumerate(part.findall("measure"), 1):
            cursor = previous = Fraction(0)
            ordinal = 0
            for element in measure:
                if element.tag == "attributes":
                    divisions = Fraction(element.findtext("divisions", str(divisions)))
                elif element.tag in ("backup", "forward"):
                    delta = Fraction(element.findtext("duration", "0")) / divisions
                    cursor += delta if element.tag == "forward" else -delta
                elif element.tag == "note":
                    ordinal += 1
                    duration = Fraction(element.findtext("duration", "0")) / divisions
                    chord = element.find("chord") is not None
                    onset = previous if chord else cursor
                    if not chord:
                        previous = onset
                        cursor += duration
                    pitch = element.find("pitch")
                    for lyric in element.findall("lyric"):
                        text = "".join(t.text or "" for t in lyric if t.tag in ("text", "elision"))
                        if not text.strip():
                            continue
                        events.append(
                            {
                                "source_part": part.get("id"),
                                "physical_measure": measure_index,
                                "source_note_ordinal": ordinal,
                                "voice": element.findtext("voice"),
                                "staff": element.findtext("staff"),
                                "onset_quarters": str(onset),
                                "duration_quarters": str(duration),
                                "pitch": (
                                    None
                                    if pitch is None
                                    else pitch.findtext("step") + pitch.findtext("octave")
                                ),
                                "alter": None if pitch is None else pitch.findtext("alter", "0"),
                                "verse": lyric.get("number", "1"),
                                "text": text,
                                "syllabic": lyric.findtext("syllabic"),
                                "segments": [
                                    {
                                        "kind": segment.tag,
                                        "text": segment.text,
                                        "attributes": dict(segment.attrib),
                                    }
                                    for segment in lyric
                                ],
                                "pixel_link_certified": False,
                            }
                        )
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    base = workspace / "homr-artifacts/gpu-roadmap-20260919/instance-results/workspace"
    alignment_path = workspace / "lieder-omr-data/provenance/alignment_v4_boundary_safe.json"
    alignment = json.loads(alignment_path.read_text())
    if alignment.get("model_predictions_used") is not False:
        raise ValueError("Alignment must not use recognition predictions")
    source_dir = workspace / "lieder-omr-data/sources/lieder-all-source/openscore-lieder"
    catalogue_path = source_dir / "data/scores.yaml"
    catalogue = yaml.safe_load(catalogue_path.read_text())
    paths = defaultdict(list)
    for number, entry in catalogue.items():
        digits = re.fullmatch(r"#?(\d+)", str(entry.get("imslp", "")))
        if digits:
            paths["IMSLP" + digits[1]].append(
                source_dir / "scores" / entry["path"] / f"lc{number}.mxl"
            )
    ossq_paths = defaultdict(list)
    for path in (workspace / "ossq-omr/scores").glob("*/*/sq*.musicxml"):
        ossq_paths[path.stem].append(path)
    inputs = {str(p): sha(p) for p in (args.pages, alignment_path, catalogue_path)}
    reviewed_scores = set()
    for path in sorted(
        (base / "guarded-tuplet-20261004/review").glob("lyric-anchor-judgement*-key.json")
    ):
        inputs[str(path)] = sha(path)
        reviewed_scores.update(x["score"] for x in json.loads(path.read_text()))
    indexes = list(
        (base / "train-20260928/public-core-reproducibility/provenance/indexes").glob(
            "*/index_*.txt"
        )
    )
    indexes += list((workspace / "homr-artifacts/corpora/lieder_scanned_pairs").glob("stage2*.txt"))
    for relative in [
        "homr-artifacts/instance-salvage-20261001/files/runs/public-joint-core-heads-20260928/frozen-train-index.txt",
        "homr-artifacts/instance-salvage-20261001/files/runs/headmix-audit-20260927/train_index.txt",
    ]:
        path = workspace / relative
        if path.exists():
            indexes.append(path)
    exposures = defaultdict(list)
    for path in sorted(set(indexes)):
        inputs[str(path)] = sha(path)
        for score in set(re.findall(r"(?:IMSLP|sq)\d+", path.read_text())):
            exposures[score].append(str(path))
    pages = [json.loads(line) for line in args.pages.read_text().splitlines()]
    if any(p["partition"] != "test" for p in pages):
        raise ValueError("Only the fixed test pool is supported")
    if len({p["id"] for p in pages}) != len(pages):
        raise ValueError("Duplicate page identities")
    args.out.mkdir(parents=True, exist_ok=False)
    records, references, cache = [], {}, {}
    for page in pages:
        score, corpus = page["score"], page["corpus"]
        image = Path(page["image"])
        if sha(image) != page["image_sha256"]:
            raise ValueError("Frozen scan page changed")
        inputs[str(image)] = sha(image)
        choices = paths[score] if corpus == "lieder" else ossq_paths[score]
        record = {
            "id": page["id"],
            "score": score,
            "corpus": corpus,
            "image": str(image),
            "image_sha256": sha(image),
            "previously_anchor_reviewed_score": score in reviewed_scores,
            "known_index_exposure": exposures[score],
            "untouched_test_verified": False,
            "pixel_link_certified": False,
            "exclusions": [],
        }
        if len(choices) != 1 or not choices[0].is_file():
            record["exclusions"].append("missing_or_ambiguous_native_source")
            records.append(record)
            continue
        source = choices[0]
        if source not in cache:
            root = read_mxl(source) if source.suffix == ".mxl" else ET.parse(source).getroot()
            if root.tag != "score-partwise" or any(
                "homr" in (s.text or "").lower() for s in root.findall(".//software")
            ):
                raise ValueError("Recognizer output or unsupported native source")
            cache[source] = (root, native_events(root))
        root, events = cache[source]
        inputs[str(source)] = sha(source)
        record.update(
            native_source=str(source),
            native_source_sha256=sha(source),
            source_lyric_targets=len(events),
            source_lyric_parts=sorted({e["source_part"] for e in events}),
        )
        if corpus == "ossq":
            record["role"] = "proposed_no_lyrics_control" if not events else "requires_lyric_audit"
            record["scan_edition_and_page_link_verified"] = False
            record["exclusions"].append("scan_edition_and_layout_unverified")
        else:
            alignment_record = alignment["scores"].get(score)
            systems = (
                []
                if alignment_record is None
                else [
                    s
                    for s in alignment_record["systems"]
                    if Path(s["page_image"]).name == image.name
                ]
            )
            counts = [len(p.findall("measure")) for p in root.findall("part")]
            if (
                alignment_record is None
                or not counts
                or len(set(counts)) != 1
                or counts[0] != sum(alignment_record["source_counts"])
            ):
                record["exclusions"].append("native_measure_count_or_alignment_mismatch")
            if not certified_page(systems):
                record["exclusions"].append("incomplete_or_ambiguous_page_boundaries")
            if not record["exclusions"]:
                start, end = min(s["start_measure"] for s in systems), max(
                    s["end_measure"] for s in systems
                )
                selected = [e for e in events if start < e["physical_measure"] <= end]
                record.update(
                    start_measure=start,
                    end_measure=end,
                    systems=systems,
                    page_native_lyric_targets=len(selected),
                    role="native_page_window_reference",
                )
                references[page["id"]] = selected
            else:
                record["role"] = "native_source_only"
        records.append(record)
    ready = [
        r
        for r in records
        if r.get("role") == "native_page_window_reference"
        and r["page_native_lyric_targets"]
        and not r["previously_anchor_reviewed_score"]
    ]
    chosen = {}
    for record in sorted(
        ready, key=lambda r: hashlib.sha256(("lyric-link-20261008:" + r["id"]).encode()).hexdigest()
    ):
        chosen.setdefault(record["score"], record)
    selected = list(chosen.values())[:10]
    report = {
        "schemaVersion": "homr.lyric-native-reference-readiness.v1",
        "model_predictions_used": False,
        "source_used_for_inference": False,
        "inputSha256": inputs,
        "sourceSha256": sha(Path(__file__)),
        "summary": {
            "pages": len(records),
            "corpora": dict(Counter(r["corpus"] for r in records)),
            "roles": dict(Counter(r.get("role", "unavailable") for r in records)),
            "exclusions": dict(Counter(reason for r in records for reason in r["exclusions"])),
            "newly_reviewable_native_window_pages": len(ready),
            "newly_reviewable_native_window_scores": len(chosen),
            "fixed_candidate_selection": len(selected),
            "verified_untouched_pages": 0,
        },
        "records": records,
        "scope": (
            "Native text/note ground truth and count-exact page-window proposals; "
            "no exact scan-to-native note-pixel certification, OCR-box recall, or release gate."
        ),
    }
    for name, data in [
        ("report.json", report),
        ("native-page-lyrics.json", references),
        ("candidate-pages.json", selected),
    ]:
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n")
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
