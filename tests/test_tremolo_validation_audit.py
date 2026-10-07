from training.onnx.audit_tremolo_validation import native_measures


def note(measure, duration="1"):
    return dict(measure=measure, onset="0", duration=duration, grace=False, pitch=("C", 4, "0"))


def test_native_audit_cannot_hide_missing_measures_or_staffs():
    reference = [[note(0), note(1)], [note(0)]]
    baseline = [[note(0, "2")]]
    candidate = [[note(0)]]
    rows = native_measures(reference, baseline, candidate)
    assert len(rows) == 2
    assert rows[0]["control"]["rhythm"]["errors"] == 2
    assert rows[0]["selective"]["rhythm"]["errors"] == 1
    assert rows[1]["selective"]["event"]["errors"] == 1
    assert rows[1]["selective"]["event"]["support"] == 1
    assert all(not r["regressed"] for r in rows)


def test_native_audit_reports_local_regression_despite_other_gains():
    reference = [[note(0), note(1)]]
    baseline = [[note(0, "2"), note(1)]]
    candidate = [[note(0), note(1, "2")]]
    rows = native_measures(reference, baseline, candidate)
    assert rows[0]["regressed"] == []
    assert "rhythm" in rows[1]["regressed"]
    assert "strict_onset_event" in rows[1]["regressed"]
