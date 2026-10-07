import argparse
import json

from training.onnx.qualify_free_timing_probe import legacy_timing, match_events, score
from training.onnx.qualify_joint_export import sha, write


def note(pitch="C4", rhythm="note_4"):
    return dict(
        rhythm=rhythm,
        pitch=pitch,
        lift="lift_null",
        articulation="none",
        slur="none",
        position="upper",
    )


def divider():
    return note(rhythm="barline")


def test_duration_changes_match_but_pitch_chord_and_measure_errors_do_not():
    reference = [note(), note("E4"), divider()]
    prediction = [note(rhythm="note_8"), note("E4"), divider()]
    assert match_events(reference, prediction)[0] == {0: 0, 1: 1}
    assert match_events(reference, [note("D4"), note("E4"), divider()])[0] == {}
    assert match_events(reference, prediction[:-1])[1]["reason"] == "measure-count-mismatch"
    assert match_events(reference, [note(), note(rhythm="chord"), note("E4"), divider()])[0] == {}


def test_raw_legacy_duration_tuplets_and_full_rest_flag_are_distinct():
    assert legacy_timing(note(rhythm="note_12")) == {
        "timing.elapsed": "1/3",
        "timing.written": "eighth:0",
        "timing.ratio": "3:2",
    }
    assert legacy_timing(note(rhythm="rest_1"))["timing.full_rest"] == "no"
    assert legacy_timing(note(rhythm="rest_0"))["timing.full_rest"] == "yes"
    assert legacy_timing(note(rhythm="note_16G")) == {}


def test_scoring_retains_unmatched_targets_and_never_uses_head_outputs_to_align(tmp_path):
    root = tmp_path
    safe = dict(id="test", source="lieder", score="IMSLP1", image="scan.png", image_sha256="a" * 64)
    write(root / "inference-pilot.json", [safe])
    targets = [{"timing.elapsed": "1"}, {"timing.elapsed": "1"}, {}]
    reference = [note(), note("E4"), divider()]
    write(root / "evaluation.json", [{**safe, "reference": reference, "targets": targets}])
    write(
        root / "preparation.json",
        dict(
            manifests_sha256={
                name: sha(root / name) for name in ("inference-pilot.json", "evaluation.json")
            }
        ),
    )
    folder = root / "pilot/part-00"
    folder.mkdir(parents=True)
    record = dict(
        input=safe,
        reference_used=False,
        prediction=[note("D4"), note("E4"), divider()],
        timing={"epoch6": [{"timing.elapsed": "1"}, {"timing.elapsed": "1"}, {}]},
    )
    write(folder / "crop.json", record)
    score(argparse.Namespace(run=root, scope="pilot"))
    result = json.loads((root / "pilot-report.json").read_text())
    assert result["timing"]["lieder"]["epoch6/timing.elapsed"] == {
        "support": 2,
        "matched": 0,
        "correct": 0,
    }
    assert result["recognition"]["lieder"]["pitch"]["errors"] == 1


def test_paired_audit_distinguishes_regressions_and_keeps_unmatched_visible():
    from training.onnx.audit_free_timing_probe import audit

    observations = []
    for index, baseline, candidate, predicted_index in [
        (0, "2", "1", 0),
        (1, "1", "2", 1),
        (2, None, None, None),
    ]:
        for arm, value in [("legacy", baseline), ("epoch6", candidate)]:
            observations.append(
                dict(
                    reference_token_index=index,
                    predicted_token_index=predicted_index,
                    head="timing.elapsed",
                    arm=arm,
                    target="1",
                    prediction=value,
                )
            )
    result = audit(
        dict(cases=[dict(id="crop", source="lieder", score="IMSLP1", timing=observations)])
    )
    counts = result["paired"]["lieder"]["epoch6/timing.elapsed"]
    assert counts == dict(support=3, improvement=1, regression=1, unmatched=1)
    assert len(result["changes"]) == 2
    assert result["classes"]["lieder"]["epoch6/timing.elapsed"]["1"]["support"] == 3
