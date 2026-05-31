"""NumPy dynamic-programming baseline over a wealth grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import evolve_wealth
from .scenario import Scenario
from .state_features import normal_midpoint_nodes


@dataclass(frozen=True)
class DPResult:
    wealth_grid: np.ndarray
    value_grid: np.ndarray
    goal_policy_grid: np.ndarray
    portfolio_policy_grid: np.ndarray


def default_z_grid(n: int = 11) -> tuple[np.ndarray, np.ndarray]:
    nodes = normal_midpoint_nodes(n)
    weights = np.full(n, 1.0 / n, dtype=float)
    return nodes, weights


def _interp_value(wealth_grid: np.ndarray, values: np.ndarray, wealth: np.ndarray) -> np.ndarray:
    return np.interp(wealth, wealth_grid, values, left=values[0], right=values[-1])


def solve_dp(
    scenario: Scenario,
    wealth_grid: np.ndarray,
    z_nodes: np.ndarray | None = None,
    z_weights: np.ndarray | None = None,
) -> DPResult:
    """Solve the all-or-nothing problem with a simple wealth-grid DP."""

    wealth_grid = np.asarray(wealth_grid, dtype=float)
    if wealth_grid.ndim != 1 or wealth_grid.size < 2:
        raise ValueError("wealth_grid must be a one-dimensional array with at least two points")
    if np.any(np.diff(wealth_grid) <= 0):
        raise ValueError("wealth_grid must be strictly increasing")
    if z_nodes is None or z_weights is None:
        z_nodes, z_weights = default_z_grid()
    z_nodes = np.asarray(z_nodes, dtype=float)
    z_weights = np.asarray(z_weights, dtype=float)
    z_weights = z_weights / np.sum(z_weights)

    n_w = wealth_grid.size
    value = np.zeros((scenario.T + 2, n_w), dtype=float)
    goal_policy = np.zeros((scenario.T + 1, n_w), dtype=int)
    portfolio_policy = np.zeros((scenario.T + 1, n_w), dtype=int)

    for t in range(scenario.T, 0, -1):
        for i, wealth in enumerate(wealth_grid):
            best_value = -np.inf
            best_g = 0
            best_p = 0
            feasible_goal_choices = [0]
            if scenario.C[t] > 0.0 and wealth + 1e-12 >= scenario.C[t]:
                feasible_goal_choices.append(1)

            for g in feasible_goal_choices:
                reward = float(g * scenario.U[t])
                wealth_after_goal = float(wealth - g * scenario.C[t])
                for p in range(scenario.P):
                    if t < scenario.T:
                        next_wealth = np.array(
                            [
                                evolve_wealth(
                                    wealth_after_goal,
                                    float(scenario.mu[p]),
                                    float(scenario.sigma[p]),
                                    float(z),
                                    scenario.h,
                                    float(scenario.I[t + 1]),
                                )
                                for z in z_nodes
                            ],
                            dtype=float,
                        )
                        continuation = float(np.sum(z_weights * _interp_value(wealth_grid, value[t + 1], next_wealth)))
                    else:
                        continuation = 0.0
                    candidate = reward + continuation
                    if candidate > best_value:
                        best_value = candidate
                        best_g = g
                        best_p = p

            value[t, i] = best_value
            goal_policy[t, i] = best_g
            portfolio_policy[t, i] = best_p

    return DPResult(
        wealth_grid=wealth_grid,
        value_grid=value,
        goal_policy_grid=goal_policy,
        portfolio_policy_grid=portfolio_policy,
    )

