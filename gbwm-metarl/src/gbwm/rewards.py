"""Appendix A reward helpers for the dual-agent MetaRL formulation."""

from __future__ import annotations

import numpy as np

from .config import EPS, GOAL_ACTION_THRESHOLD
from .scenario import Scenario
from .state_features import build_state


def extrinsic_reward(
    scenario: Scenario,
    t: int,
    wealth_before_goal: float,
    goal_action: float,
    goal_taken: int,
    a_thresh: float = GOAL_ACTION_THRESHOLD,
) -> float:
    """Return the paper's normalized extrinsic reward r_e(t)."""

    total_utility = max(float(np.sum(scenario.U)), EPS)
    if t < scenario.T:
        return float(goal_taken) * float(scenario.U[t]) / total_utility

    if float(goal_action) < a_thresh or scenario.U[t] <= 0.0 or scenario.C[t] <= 0.0:
        return 0.0
    if float(wealth_before_goal) + EPS >= float(scenario.C[t]):
        return float(scenario.U[t]) / total_utility
    return float(scenario.U[t]) / total_utility * 0.25 * max(float(wealth_before_goal), 0.0) / float(scenario.C[t])


def intrinsic_goal_reward(scenario: Scenario, t: int, wealth_before_goal: float, goal_action: float, lambda_i: float) -> float:
    state = build_state(scenario, t, wealth_before_goal)
    g_sim = float(state[-2])
    return -0.5 * float(lambda_i) * abs(g_sim - float(goal_action))


def intrinsic_portfolio_reward(scenario: Scenario, t: int, wealth_after_goal: float, portfolio_action: float, lambda_i: float) -> float:
    state = build_state(scenario, t, wealth_after_goal)
    p_sim = float(state[-1])
    return -0.5 * float(lambda_i) * abs(p_sim - float(portfolio_action))


def intrinsic_lambda(epoch_index: int, total_epochs: int, start: float = 1.0, end: float = 0.25) -> float:
    if total_epochs <= 1:
        return float(end)
    progress = min(max(float(epoch_index) / float(total_epochs - 1), 0.0), 1.0)
    return float(start + (end - start) * progress)


def reverse_returns(rewards: list[float]) -> list[float]:
    running = 0.0
    out = [0.0] * len(rewards)
    for idx in range(len(rewards) - 1, -1, -1):
        running += float(rewards[idx])
        out[idx] = running
    return out
