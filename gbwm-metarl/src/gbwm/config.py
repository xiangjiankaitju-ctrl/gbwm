"""Shared constants for the GBWM reproduction."""

from __future__ import annotations

import math

EPS = 1e-12
GOAL_ACTION_THRESHOLD = 0.5
DEFAULT_H = 1.0
DEFAULT_FORWARD_SIM_N = 11

# Paper default: L = [[0], [1], [2], [3], [4,5], [6,7,8,9], [10:]]
DEFAULT_TIME_BLOCKS = (
    (0,),
    (1,),
    (2,),
    (3,),
    (4, 5),
    (6, 7, 8, 9),
    slice(10, None),
)


def logistic(x: float) -> float:
    """Numerically stable logistic used for g_sim."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

