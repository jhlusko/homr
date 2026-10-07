"""Research-only, prediction-derived bar spans; never infer meter notation."""

from collections import Counter
from fractions import Fraction

from training.onnx.probe_selective_timing_export import bars


def peer_span(voices, part_index, bar_index, timeline, safe_bar):
    target = voices[part_index]
    # Any declaration prevents fallback: partial or changing declarations remain guarded.
    if any(s.rhythm.startswith(("timeSignature/", "timeSignatureBeats_")) for s in target):
        return None, dict(reason="declared-or-partial-meter-no-peer-fallback")
    count = len(bars(target))
    witnesses = []
    for pi, voice in enumerate(voices):
        grouped = bars(voice)
        if pi == part_index or len(grouped) != count or any(s.position == "lower" for s in voice):
            continue
        lengths = []
        for bi, bar in enumerate(grouped):
            symbols = [s for g in bar for s in g.symbols]
            if bi == bar_index or any("G" in s.rhythm for s in symbols):
                continue
            if not any(s.rhythm.startswith("note") for s in symbols):
                continue
            length = sum((g.get_render_duration() * 4 for g in bar), Fraction())
            # Only clean, complete serialized single-voice bars are evidence.
            if length > 0 and safe_bar(timeline, pi, bi, 1, length):
                lengths.append(length)
        votes = Counter(lengths)
        if len(lengths) < 3:
            continue
        span, support = votes.most_common(1)[0]
        if support * 4 < len(lengths) * 3:
            continue
        witnesses.append(dict(part=pi + 1, span=str(span), support=support, total=len(lengths)))
    spans = {w["span"] for w in witnesses}
    if len(spans) != 1:
        return None, dict(reason="conflicting-or-insufficient-peer-spans", witnesses=witnesses)
    span = Fraction(next(iter(spans)))
    # Require at least one peer's candidate bar itself to confirm the same span.
    confirming = [
        w["part"] for w in witnesses if safe_bar(timeline, w["part"] - 1, bar_index, 1, span)
    ]
    if not confirming:
        return None, dict(reason="peer-candidate-bar-does-not-confirm-span", witnesses=witnesses)
    return span, dict(
        reason="prediction-only-peer-span",
        witnesses=witnesses,
        confirming_parts=confirming,
        candidate_excluded=True,
        meter_notation_unchanged=True,
    )
