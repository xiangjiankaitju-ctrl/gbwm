"""Policy helpers for smoke evaluation before PPO is available."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dp_baseline import DPResult
from .environment import goal_decision
from .scenario import Scenario
from .state_features import build_state


@dataclass(frozen=True)
class PolicyAction:
    goal_action: float
    portfolio_action: float


class HeuristicStatePolicy:
    """Use g_sim and p_sim from the MetaRL state as a transparent baseline."""

    def __call__(self, scenario: Scenario, t: int, wealth: float) -> PolicyAction:
        state = build_state(scenario, t, wealth)
        return PolicyAction(goal_action=float(state[-2]), portfolio_action=float(state[-1]))


class GreedyAffordablePolicy:
    """Cheap baseline: take affordable goals and use a fixed portfolio action."""

    def __init__(self, portfolio_action: float = 0.5):
        self.portfolio_action = float(portfolio_action)

    def __call__(self, scenario: Scenario, t: int, wealth: float) -> PolicyAction:
        take = scenario.C[t] > 0.0 and wealth + 1e-12 >= scenario.C[t]
        return PolicyAction(goal_action=1.0 if take else 0.0, portfolio_action=self.portfolio_action)


class DPGridPolicy:
    """Convert a solved DP grid into continuous environment actions."""

    def __init__(self, result: DPResult):
        self.result = result

    def __call__(self, scenario: Scenario, t: int, wealth: float) -> PolicyAction:
        idx = int(np.argmin(np.abs(self.result.wealth_grid - wealth)))
        g = int(self.result.goal_policy_grid[t, idx])
        p = int(self.result.portfolio_policy_grid[t, idx])
        a_g = 1.0 if g else 0.0
        a_p = min((p + 0.5) / scenario.P, 1.0)
        return PolicyAction(goal_action=a_g, portfolio_action=a_p)


class MetaRLPolicy:
    """Median-action inference over trained seed checkpoints."""

    def __init__(self, checkpoint_paths: list[str | Path], device: str = "cpu"):
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths must not be empty")
        from .ppo import load_checkpoint_agents

        self.models = [load_checkpoint_agents(path, device=device)[:2] for path in checkpoint_paths]

    def __call__(self, scenario: Scenario, t: int, wealth: float) -> PolicyAction:
        goal_state = build_state(scenario, t, wealth)
        goal_actions = [goal.mean_action(goal_state) for goal, _portfolio in self.models]
        goal_action = float(np.median(np.asarray(goal_actions, dtype=float)))
        goal_taken = goal_decision(goal_action, wealth, scenario.C[t])
        wealth_after_goal = float(wealth - goal_taken * scenario.C[t])
        portfolio_state = build_state(scenario, t, wealth_after_goal)
        portfolio_actions = [portfolio.mean_action(portfolio_state) for _goal, portfolio in self.models]
        portfolio_action = float(np.median(np.asarray(portfolio_actions, dtype=float)))
        return PolicyAction(goal_action=goal_action, portfolio_action=portfolio_action)

    def portfolio_only_action(self, scenario: Scenario, t: int, wealth: float) -> float:
        portfolio_state = build_state(scenario, t, wealth)
        portfolio_actions = [portfolio.mean_action(portfolio_state) for _goal, portfolio in self.models]
        return float(np.median(np.asarray(portfolio_actions, dtype=float)))
