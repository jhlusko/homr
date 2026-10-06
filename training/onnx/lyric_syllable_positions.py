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
