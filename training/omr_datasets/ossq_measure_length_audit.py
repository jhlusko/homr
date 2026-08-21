"""
Audit ossq-omr ground-truth MusicXML for measures where parts disagree on total
length - this should never happen in valid notation (every part in a system must
span the same duration per measure), so any occurrence here is a genuine ground-truth
labeling defect, not a legitimate irregularity.

For each part, walks note/backup/forward in document order, tracking a cursor and the
running maximum it reaches within the measure (backup can rewind for a second voice;
the measure's true length is the peak the cursor reaches, not just where it ends).
Normalizes by that part's own current <divisions> value (divisions can differ between
parts, and can change mid-piece) so comparisons are in quarter-note units regardless
of each part's own tick resolution.
"""
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path


def measure_length_by_part(measure: ET.Element, divisions: dict) -> dict[str, Fraction]:
    """part_id -> length in quarter notes, for every part with a <staff>/notes rooted
    directly under this measure element (single-part-per-<part> MusicXML, the ossq-omr
    convention)."""
    div = divisions.get("current", 1)
    cursor = Fraction(0)
    peak = Fraction(0)
    for el in measure:
        if el.tag == "attributes":
            d = el.find("divisions")
            if d is not None:
                div = int(d.text)
                divisions["current"] = div
        elif el.tag == "note":
            dur_el = el.find("duration")
            if dur_el is None:
                continue
            dur = Fraction(int(dur_el.text), div)
            is_chord = el.find("chord") is not None
            if not is_chord:
                cursor += dur
                peak = max(peak, cursor)
        elif el.tag == "backup":
            dur_el = el.find("duration")
            if dur_el is not None:
                cursor -= Fraction(int(dur_el.text), div)
        elif el.tag == "forward":
            dur_el = el.find("duration")
            if dur_el is not None:
                cursor += Fraction(int(dur_el.text), div)
                peak = max(peak, cursor)
    return peak


def audit_file(path: Path) -> list[dict]:
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    parts = root.findall(".//part")
    if len(parts) < 2:
        return []

    part_names = {}
    for sp in root.findall(".//score-part"):
        pid = sp.get("id")
        name_el = sp.find("part-name")
        part_names[pid] = name_el.text if name_el is not None and name_el.text else pid

    per_part_divisions = {p.get("id"): {"current": 1} for p in parts}
    per_part_measures = {p.get("id"): p.findall("measure") for p in parts}

    max_len = max(len(m) for m in per_part_measures.values())
    findings = []
    for i in range(max_len):
        lengths = {}
        for pid, measures in per_part_measures.items():
            if i >= len(measures):
                continue
            m = measures[i]
            length = measure_length_by_part(m, per_part_divisions[pid])
            lengths[pid] = (m.get("number"), length)
        distinct = {length for _, length in lengths.values()}
        if len(distinct) > 1:
            findings.append(
                {
                    "file": str(path),
                    "measure_index": i,
                    "per_part": {
                        f"{pid} ({part_names.get(pid, pid)})": (num, str(length))
                        for pid, (num, length) in lengths.items()
                    },
                }
            )
    return findings


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Flag every measure where ossq-omr ground-truth parts disagree "
        "on total length - never legitimate notation, always a labeling defect."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2].parent / "ossq-omr",
        help="Path to an ossq-omr checkout (default: ../ossq-omr next to this repo).",
    )
    args = parser.parse_args()

    files = sorted(
        p for p in args.dataset_root.rglob("*.musicxml") if "/musicxml/" not in str(p)
    )
    if not files:
        raise SystemExit(f"No ground-truth MusicXML found under {args.dataset_root}.")
    print(f"scanning {len(files)} ground-truth files")
    all_findings = []
    for idx, path in enumerate(files, 1):
        findings = audit_file(path)
        all_findings.extend(findings)
        if idx % 50 == 0:
            print(f"  [{idx}/{len(files)}] ... {len(all_findings)} findings so far")

    print(f"\n===== {len(all_findings)} measure(s) with disagreeing part lengths, "
          f"across {len({f['file'] for f in all_findings})} file(s) =====")
    for f in all_findings:
        print(f"{f['file']}  measure_index={f['measure_index']}")
        for part, (num, length) in f["per_part"].items():
            print(f"    {part}: measure {num} = {length} quarter notes")


if __name__ == "__main__":
    main()
