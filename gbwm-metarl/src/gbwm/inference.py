"""Inference helpers."""

from __future__ import annotations

import numpy as np


def median_action(actions: list[float] | np.ndarray) -> float:
    values = np.asarray(actions, dtype=float)
    if values.size == 0:
        raise ValueError("actions must not be empty")
    return float(np.median(values))

