"""Deterministic spatially-correlated randomness: one uniform number per grid cell, from an integer hash.
The same cell always gets the same draw for a given seed, so results do not depend on graph indexing."""
from __future__ import annotations

import numpy as np

_M = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def cell_uniform(seed: int, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Uniform [0,1) per integer cell (cx, cy)."""
    a = np.asarray(cx, dtype=np.int64).astype(np.uint64)
    b = np.asarray(cy, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = np.uint64(seed & 0xFFFFFFFFFFFFFFFF) ^ (a * np.uint64(0x517CC1B727220A95)) ^ (b * np.uint64(0x6D2B79F5E385D853))
    return (splitmix64(h) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
