"""Paper Appendix C test cases.

The PDF table text contains compact formula rows. This module keeps those
formula rows as code so the 66 scenarios are reproducible without manually
expanding hundreds of repeated goals and infusions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .efficient_frontier import baseline_efficient_frontier, simulated_efficient_frontier
from .scenario import Scenario


def _spec(case_id: int, T: int, W0: float, goals: list[list[float]], infusions: list[list[float]] | None = None) -> dict[str, Any]:
    return {
        "case_id": int(case_id),
        "T": int(T),
        "W0": float(W0),
        "goals": goals,
        "infusions": [] if infusions is None else infusions,
    }


def _goal_range(start: int, stop: int, cost_fn, utility_fn) -> list[list[float]]:
    return [[int(t), float(cost_fn(t)), float(utility_fn(t))] for t in range(start, stop + 1)]


def _even_year_goals(T: int, cost: float) -> list[list[float]]:
    return _goal_range(2, T, lambda _t: cost, lambda _t: 1.0)[::2]


def _formula_infusions(T: int, W0: float) -> list[list[float]]:
    # Appendix D explicitly describes case 57 as (10 / 59) * 1.03^t.
    # That corresponds to W0 / (10 * (T - 1)) * 1.03^t for W0=100,T=60.
    base = float(W0) / (10.0 * float(T - 1))
    return [[int(t), float(base * (1.03**t))] for t in range(1, T)]


def _base_specs_1_to_33() -> list[dict[str, Any]]:
    specs = [
        _spec(1, 10, 100.0, [[10, 150.0, 1.0]]),
        _spec(2, 10, 100.0, [[10, 200.0, 1.0]]),
        _spec(3, 10, 100.0, [[10, 400.0, 1.0]]),
        _spec(4, 40, 100.0, [[40, 600.0, 1.0]]),
        _spec(5, 40, 100.0, [[40, 1200.0, 1.0]]),
        _spec(6, 40, 100.0, [[40, 2400.0, 1.0]]),
        _spec(7, 100, 100.0, [[100, 50000.0, 1.0]]),
        _spec(8, 100, 100.0, [[100, 500000.0, 1.0]]),
        _spec(9, 100, 100.0, [[100, 20000000.0, 1.0]]),
        _spec(10, 3, 100.0, [[2, 75.0, 0.9], [3, 75.0, 1.0]]),
        _spec(11, 20, 100.0, [[10, 200.0, 1.3], [20, 500.0, 1.0]]),
        _spec(12, 20, 100.0, [[15, 200.0, 1.3], [20, 300.0, 1.0]]),
        _spec(13, 35, 100.0, [[5, 50.0, 1.0], [25, 500.0, 0.5], [35, 1000.0, 1.0]]),
        _spec(14, 60, 100.0, [[15, 300.0, 0.7], [30, 6000.0, 1.2], [45, 5000.0, 0.2], [60, 20000.0, 1.0]]),
        _spec(15, 25, 100.0, [[3, 30.0, 0.2], [5, 70.0, 0.3], [8, 70.0, 0.3], [25, 1000.0, 1.0]]),
        _spec(16, 40, 100.0, [[10, 150.0, 1.5], [30, 400.0, 1.0], [35, 500.0, 1.0], [40, 600.0, 1.0]]),
        _spec(17, 20, 100.0, _even_year_goals(20, 15.0)),
        _spec(18, 20, 100.0, _even_year_goals(20, 25.0)),
        _spec(19, 20, 100.0, _even_year_goals(20, 50.0)),
        _spec(20, 20, 100.0, _even_year_goals(20, 75.0)),
        _spec(21, 60, 100.0, _goal_range(1, 60, lambda _t: 20.0, lambda _t: 1.0)),
        _spec(22, 60, 100.0, _goal_range(1, 60, lambda t: t, lambda _t: 1.0)),
        _spec(23, 60, 100.0, _goal_range(1, 60, lambda t: t, lambda t: 100.0 + t)),
        _spec(24, 60, 100.0, _goal_range(1, 60, lambda t: t, lambda t: 100.0 - t)),
        _spec(25, 60, 100.0, _goal_range(1, 60, lambda t: 60.0 - t / 2.0, lambda _t: 1.0)),
        _spec(26, 60, 100.0, _goal_range(1, 60, lambda t: 60.0 - t / 2.0, lambda t: 100.0 + t)),
        _spec(27, 60, 100.0, _goal_range(1, 60, lambda t: 60.0 - t / 2.0, lambda t: 100.0 - t)),
        _spec(
            28,
            30,
            100.0,
            [
                [3, 35.0, 1.5],
                [6, 35.0, 1.3],
                [9, 5.0, 0.4],
                [12, 50.0, 1.0],
                [15, 15.0, 0.7],
                [18, 5.0, 0.3],
                [21, 45.0, 0.6],
                [24, 120.0, 0.9],
                [27, 170.0, 1.1],
                [30, 160.0, 1.0],
            ],
        ),
        _spec(29, 16, 12.0, [[16, 34.25, 26.0]]),
        _spec(30, 16, 21.63, [[8, 18.50, 18.0], [16, 34.25, 26.0]]),
        _spec(31, 16, 38.99, [[4, 13.60, 14.0], [8, 18.50, 18.0], [12, 25.18, 22.0], [16, 34.25, 26.0]]),
        _spec(
            32,
            16,
            70.27,
            [[2, 11.66, 12.0], [4, 13.60, 14.0], [6, 15.87, 16.0], [8, 18.50, 18.0], [10, 21.59, 20.0], [12, 25.18, 22.0], [14, 29.37, 24.0], [16, 34.25, 26.0]],
        ),
        _spec(
            33,
            16,
            126.67,
            [[1, 10.8, 11.0], [2, 11.66, 12.0], [3, 12.60, 13.0], [4, 13.60, 14.0], [5, 14.69, 15.0], [6, 15.87, 16.0], [7, 17.14, 17.0], [8, 18.50, 18.0], [9, 19.99, 19.0], [10, 21.59, 20.0], [11, 23.32, 21.0], [12, 25.18, 22.0], [13, 27.20, 23.0], [14, 29.37, 24.0], [15, 31.72, 25.0], [16, 34.25, 26.0]],
        ),
    ]
    return specs


def paper_case_specs() -> list[dict[str, Any]]:
    """Return the 66 Appendix C case specifications."""

    base_specs = _base_specs_1_to_33()
    infusion_specs: dict[int, list[list[float]] | str] = {
        34: [[1, 10.0]],
        35: "formula",
        36: [[1, 10.0]],
        37: "formula",
        38: [[6, 12.0]],
        39: "formula",
        40: [[21, 19.0]],
        41: "formula",
        42: [[27, 22.0]],
        43: "formula",
        44: [[6, 12.0]],
        45: "formula",
        46: [[13, 15.0]],
        47: "formula",
        48: [[11, 14.0]],
        49: "formula",
        50: [[10, 13.0]],
        51: "formula",
        52: [[11, 14.0]],
        53: "formula",
        54: [[38, 31.0]],
        55: "formula",
        56: [[41, 34.0]],
        57: "formula",
        58: [[45, 38.0]],
        59: "formula",
        60: [[49, 43.0]],
        61: "formula",
        62: [[14, 3.0]],
        63: "formula",
        64: [[15, 6.0]],
        65: "formula",
        66: [[16, 21.0]],
    }
    specs = list(base_specs)
    for case_id in range(34, 67):
        base = base_specs[case_id - 34]
        infusions = infusion_specs[case_id]
        if infusions == "formula":
            infusions = _formula_infusions(int(base["T"]), float(base["W0"]))
        specs.append(_spec(case_id, int(base["T"]), float(base["W0"]), list(base["goals"]), list(infusions)))
    return specs


def load_case_specs(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["cases"])


def scenario_from_case_spec(spec: dict[str, Any], P: int = 15, frontier_source: str = "baseline") -> Scenario:
    T = int(spec["T"])
    C = np.zeros(T + 1, dtype=float)
    U = np.zeros(T + 1, dtype=float)
    I = np.zeros(T + 1, dtype=float)
    for t, cost, utility in spec.get("goals", []):
        C[int(t)] = float(cost)
        U[int(t)] = float(utility)
    for t, amount in spec.get("infusions", []):
        I[int(t)] = float(amount)
    if frontier_source == "baseline":
        mu, sigma = baseline_efficient_frontier(P=P)
    elif frontier_source == "simulated":
        mu, sigma = simulated_efficient_frontier(P=P)
    else:
        raise ValueError("frontier_source must be 'baseline' or 'simulated'")
    return Scenario(T=T, W0=float(spec["W0"]), C=C, U=U, I=I, mu=mu, sigma=sigma, name=f"case-{spec['case_id']}")
