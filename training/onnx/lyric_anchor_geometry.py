"""Inverse of captured staff preprocessing, for diagnostic page overlays."""

import numpy as np


def inverse_dewarp(point, source, destination, simplices):
    """Invert using the exact forward triangles, refusing uncovered/ambiguous points."""
    if source is None:
        return np.asarray(point, dtype=float)
    source, destination = np.asarray(source), np.asarray(destination)
    matches = []
    for simplex in simplices:
        dst = destination[simplex]
        matrix = np.vstack((dst.T, np.ones(3)))
        if abs(np.linalg.det(matrix)) < 1e-8:
            continue
        weights = np.linalg.solve(matrix, [*point, 1])
        if np.all(weights >= -1e-6):
            matches.append(weights @ source[simplex])
    if not matches:
        raise ValueError("Point outside dewarp coverage")
    if any(np.linalg.norm(candidate - matches[0]) > 0.01 for candidate in matches[1:]):
        raise ValueError("Point has ambiguous inverse dewarp")
    return matches[0]


def canvas_to_page(point, transform):
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Attention coordinate must be a finite pair")
    canvas = transform["canvas"]
    point -= canvas["offset"]
    if np.any(point < 0) or np.any(point >= canvas["resized_size"]):
        raise ValueError("Attention coordinate lies in canvas padding")
    point *= np.asarray(canvas["input_size"]) / canvas["resized_size"]
    point += transform["crop_origins"][1]
    dewarp = transform["dewarp"]
    point = inverse_dewarp(point, dewarp["source"], dewarp["destination"], dewarp["simplices"])
    point += transform["crop_origins"][0]
    point /= transform["staff_scale"]
    point *= transform["original_to_processed_ratio"]
    return [float(value) for value in point]
