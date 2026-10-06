import json
import sys

from training.onnx.recover_ossq_archive_mapping import main


def test_archive_candidates_do_not_certify_or_confuse_filename_suffix_with_pixels(
    tmp_path, monkeypatch
):
    archive = tmp_path / "archive"
    score = archive / "scores/composer/work"
    score.mkdir(parents=True)
    (score / "sq123.musicxml").write_text("<score-partwise/>")
    metadata = score / "metadata/scanned/systemwise"
    metadata.mkdir(parents=True)
    (metadata / "sq123:0001:0001.yaml").write_text(
        "score_id: sq123\npage_idx: 1\nsystem_idx: 1\n"
        "number_of_parts: 4\nmeasure_numbers: [3, 4]\n"
    )
    pixels = score / "images/scanned/partwise"
    pixels.mkdir(parents=True)
    (pixels / "sq123:0001:0001:1.png").write_bytes(b"original")
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    public = tmp_path / "public"
    for split, content in (("train", b"original"), ("valid", b"different")):
        image = f"scanned/{split}/sq123_0001_0001_1-pathhash.png"
        (indexes / f"index_{split}.txt").write_text(
            f"{image},scanned/{split}/sq123_0001_0001_1.txt\n"
        )
        target = public / image
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--archive",
            str(archive),
            "--indexes",
            str(indexes),
            "--public-corpus",
            str(public),
            "--out",
            str(out),
        ],
    )
    main()
    report = json.loads(out.read_text())
    assert report["certified_crop_mappings"] is False
    assert report["summary"]["train"]["public_pixels_match"] == 1
    assert report["summary"]["valid"]["public_pixels_match"] == 0
    assert all(row["metadata_identity_matches"] for row in report["crops"])
