"""Synthetic training scenario generation from the paper curriculum."""

from __future__ import annotations

import numpy as np

from .scenario import Scenario
from .state_features import discount_sum

GOAL_COUNT_VALUES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1], dtype=int)
GOAL_COUNT_PROBS = np.array([0.22, 0.15, 0.12, 0.10, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.20], dtype=float)


def generate_training_scenario(mu: np.ndarray, sigma: np.ndarray, rng: np.random.Generator, name: str = "generated") -> Scenario:
    """Generate one Algorithm-5-style all-or-nothing training scenario."""

    T = int(rng.integers(5, 51))
    C = np.zeros(T + 1, dtype=float)
    U = np.zeros(T + 1, dtype=float)
    I = np.zeros(T + 1, dtype=float)

    sampled = int(rng.choice(GOAL_COUNT_VALUES, p=GOAL_COUNT_PROBS))
    goal_count = T if sampled == -1 else min(sampled, T)
    goal_times = {T}
    if goal_count > 1:
        candidates = np.arange(1, T, dtype=int)
        chosen = rng.choice(candidates, size=goal_count - 1, replace=False)
        goal_times.update(int(x) for x in chosen)

    for t in sorted(goal_times):
        u1 = float(rng.uniform(0.0, 1.0))
        u2 = float(rng.uniform(0.0, 1.0))
        C[t] = 100.0 * u1 * (1.03**t)
        U[t] = 0.3 * C[t] / (1.03**t) + 25.0 * u2

    probe = Scenario(T=T, W0=1.0, C=C, U=U, I=I, mu=mu, sigma=sigma, name=f"{name}-probe")
    low = discount_sum(C, 1, probe.P - 1, 2.0, probe)
    high = discount_sum(C, 1, 0, -2.0, probe)
    lower, upper = sorted((float(low), float(high)))
    if upper <= lower:
        upper = lower + 1.0
    W0 = float(rng.uniform(lower, upper))
    return Scenario(T=T, W0=W0, C=C, U=U, I=I, mu=mu, sigma=sigma, name=name)

