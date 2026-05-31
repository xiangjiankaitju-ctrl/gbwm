from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.dp_baseline import solve_dp
from gbwm.efficient_frontier import simulated_efficient_frontier
from gbwm.metrics import evaluate_policy, rl_efficiency
from gbwm.policies import DPGridPolicy, HeuristicStatePolicy
from gbwm.scenario import Scenario
from gbwm.state_features import build_state


def build_smoke_scenario() -> Scenario:
    mu, sigma = simulated_efficient_frontier(P=5)
    T = 6
    C = np.zeros(T + 1)
    U = np.zeros(T + 1)
    I = np.zeros(T + 1)
    C[2], U[2] = 35.0, 0.8
    C[4], U[4] = 70.0, 1.2
    C[6], U[6] = 110.0, 1.6
    I[3] = 15.0
    return Scenario(T=T, W0=85.0, C=C, U=U, I=I, mu=mu, sigma=sigma, name="smoke")


def main() -> None:
    scenario = build_smoke_scenario()
    state = build_state(scenario, 1, scenario.W0)
    wealth_grid = np.linspace(0.0, 250.0, 126)

    start = time.perf_counter()
    dp = solve_dp(scenario, wealth_grid)
    dp_seconds = time.perf_counter() - start

    rng = np.random.default_rng(123)
    z_paths = rng.standard_normal((500, scenario.T))
    dp_eval = evaluate_policy(scenario, DPGridPolicy(dp), z_paths)
    heuristic_eval = evaluate_policy(scenario, HeuristicStatePolicy(), z_paths)

    summary = {
        "scenario": scenario.name,
        "state_shape": list(state.shape),
        "state_finite": bool(np.all(np.isfinite(state))),
        "dp_seconds": dp_seconds,
        "dp_mean_utility": dp_eval.mean_utility,
        "heuristic_mean_utility": heuristic_eval.mean_utility,
        "heuristic_efficiency_vs_dp": rl_efficiency(heuristic_eval, dp_eval),
        "initial_dp_goal": int(dp.goal_policy_grid[1, np.argmin(np.abs(dp.wealth_grid - scenario.W0))]),
        "initial_dp_portfolio": int(dp.portfolio_policy_grid[1, np.argmin(np.abs(dp.wealth_grid - scenario.W0))]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
