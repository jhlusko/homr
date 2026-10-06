"""Split decoded lyric text using CTC character boxes; never use source lyrics."""

import numpy as np


def split_positioned_text(text, characters):
    """Keep apostrophes/punctuation; spaces and printed hyphens delimit candidates.

    Boxes are approximate CTC cells, not certified glyph boundaries. Refuse any
    disagreement between visible character records and the recognizer's text.
    """
    expected = [char for char in text if not char.isspace()]
    if [row["text"] for row in characters] != expected:
        raise ValueError("Character stream disagrees with decoded line")
    result, current = [], []

    def finish():
        if not current:
            return
        polygons = [np.asarray(row["box"], dtype=float) for row in current]
        if any(p.shape != (4, 2) or not np.isfinite(p).all() for p in polygons):
            raise ValueError("Character geometry must be finite quadrilaterals")
        points = np.concatenate(polygons)
        result.append(
            dict(
                text="".join(row["text"] for row in current),
                center=np.mean([p.mean(axis=0) for p in polygons], axis=0).tolist(),
                bounds=[
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                ],
                character_indices=[row["index"] for row in current],
                geometry="approximate CTC character cells",
            )
        )
        current.clear()

    index = 0
    for char in text:
        if char.isspace():
            finish()
            continue
        row = characters[index]
        index += 1
        if char in "-–—":
            finish()
        else:
            current.append(row)
    finish()
    return result


def propose_spatial_splits(text, characters, min_gap_ratio=2.5):
    """Opt-in calibration proposal for missing separators; preserve every character.

    A boundary needs two characters on each side and a gap at least the specified
    multiple of the token's median character-center spacing. Short/nonmonotonic
    tokens stay intact. This threshold is a calibration hypothesis, not a frozen
    production policy or a guarantee that a wide gap denotes a syllable.
    """
    if not np.isfinite(min_gap_ratio) or min_gap_ratio <= 1:
        raise ValueError("Gap ratio must be finite and greater than one")
    baseline = split_positioned_text(text, characters)
    lookup = {row["index"]: row for row in characters}
    if len(lookup) != len(characters):
        raise ValueError("Character indices must be unique")
    result, proposals = [], []
    for syllable in baseline:
        rows = [lookup[index] for index in syllable["character_indices"]]
        centers = np.array([np.asarray(row["box"]).mean(axis=0)[0] for row in rows])
        gaps = np.diff(centers)
        cuts = []
        if len(rows) >= 4 and np.all(gaps > 0):
            median = float(np.median(gaps))
            cuts = [
                index + 1
                for index, gap in enumerate(gaps)
                if index >= 1 and len(rows) - index - 1 >= 2 and gap >= min_gap_ratio * median
            ]
            # Do not create short fragments between multiple candidate cuts.
            if any(b - a < 2 for a, b in zip([0, *cuts], [*cuts, len(rows)])):
                cuts = []
        if not cuts:
            result.append(syllable)
            continue
        pieces = []
        for start, end in zip([0, *cuts], [*cuts, len(rows)]):
            part = rows[start:end]
            value = split_positioned_text("".join(row["text"] for row in part), part)[0]
            value["spatial_split_proposal"] = True
            pieces.append(value)
        result.extend(pieces)
        proposals.append(
            dict(
                original=syllable["text"],
                proposed=[p["text"] for p in pieces],
                boundaries=cuts,
                gap_ratios=[float(gaps[cut - 1] / median) for cut in cuts],
            )
        )
    return dict(syllables=result, proposals=proposals, production_qualified=False)
