"""Monte Carlo policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .environment import GBWMEnv
from .policies import PolicyAction
from .scenario import Scenario


class Policy(Protocol):
    def __call__(self, scenario: Scenario, t: int, wealth: float) -> PolicyAction:
        ...


@dataclass(frozen=True)
class EvaluationResult:
    mean_utility: float
    path_utilities: np.ndarray


def evaluate_policy(
    scenario: Scenario,
    policy: Policy,
    z_paths: np.ndarray,
    initial_wealth: float | None = None,
) -> EvaluationResult:
    z_paths = np.asarray(z_paths, dtype=float)
    if z_paths.ndim != 2 or z_paths.shape[1] < scenario.T:
        raise ValueError("z_paths must have shape (n_paths, at least T)")
    utilities = np.zeros(z_paths.shape[0], dtype=float)
    for path_idx in range(z_paths.shape[0]):
        env = GBWMEnv(scenario)
        t, wealth = env.reset(initial_wealth)
        total = 0.0
        for step_idx in range(scenario.T):
            action = policy(scenario, t, wealth)
            record = env.step(action.goal_action, action.portfolio_action, float(z_paths[path_idx, step_idx]))
            total += record.utility_attained
            if record.done:
                break
            t = env.t
            wealth = env.wealth
        utilities[path_idx] = total
    return EvaluationResult(mean_utility=float(np.mean(utilities)), path_utilities=utilities)


def rl_efficiency(candidate: EvaluationResult, optimal: EvaluationResult) -> float:
    if optimal.mean_utility <= 0.0:
        return float("nan")
    return float(candidate.mean_utility / optimal.mean_utility)


def evaluate_policies(
    scenario: Scenario,
    policies: dict[str, Policy],
    z_paths: np.ndarray,
    initial_wealth: float | None = None,
) -> dict[str, EvaluationResult]:
    """Evaluate multiple policies on the same Monte Carlo paths."""

    return {
        name: evaluate_policy(scenario, policy, z_paths, initial_wealth=initial_wealth)
        for name, policy in policies.items()
    }


def evaluation_summary(results: dict[str, EvaluationResult], baseline_name: str = "dp") -> dict[str, float]:
    """Flatten policy results into stable scalar fields for CSV/JSON output."""

    summary: dict[str, float] = {}
    baseline = results.get(baseline_name)
    for name, result in results.items():
        summary[f"{name}_mean_utility"] = result.mean_utility
        if baseline is not None and name != baseline_name:
            summary[f"{name}_efficiency_vs_{baseline_name}"] = rl_efficiency(result, baseline)
    return summary
