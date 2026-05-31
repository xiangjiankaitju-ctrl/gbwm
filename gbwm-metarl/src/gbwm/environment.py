"""All-or-nothing GBWM environment dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import GOAL_ACTION_THRESHOLD
from .scenario import Scenario


@dataclass(frozen=True)
class StepRecord:
    t: int
    wealth_before_goal: float
    goal_action: float
    goal_taken: int
    portfolio_action: float
    portfolio_index: int
    utility_attained: float
    wealth_after_goal: float
    z: float
    wealth_next: float
    done: bool


def goal_decision(a_g: float, wealth: float, goal_cost: float, a_thresh: float = GOAL_ACTION_THRESHOLD) -> int:
    """Map a continuous goal action to a feasible all-or-nothing decision."""

    return int(float(a_g) >= a_thresh and float(wealth) + 1e-12 >= float(goal_cost) and float(goal_cost) > 0.0)


def portfolio_decision(a_p: float, P: int) -> int:
    """Map a continuous portfolio action in [0, 1] to an ordered portfolio index."""

    if P < 1:
        raise ValueError("P must be positive")
    clipped = min(max(float(a_p), 0.0), 1.0)
    if clipped >= 1.0:
        return P - 1
    return int(math.floor(clipped * P))


def evolve_wealth(wealth_after_goal: float, mu: float, sigma: float, z: float, h: float, infusion_next: float) -> float:
    gross_return = math.exp((mu - 0.5 * sigma * sigma) * h + sigma * float(z) * math.sqrt(h))
    return float(wealth_after_goal) * gross_return + float(infusion_next)


class GBWMEnv:
    """Minimal deterministic-shell environment for a fixed GBWM scenario."""

    def __init__(self, scenario: Scenario, a_thresh: float = GOAL_ACTION_THRESHOLD):
        self.scenario = scenario
        self.a_thresh = a_thresh
        self.t = 1
        self.wealth = float(scenario.W0)
        self.history: list[StepRecord] = []

    def reset(self, wealth: float | None = None) -> tuple[int, float]:
        self.t = 1
        self.wealth = float(self.scenario.W0 if wealth is None else wealth)
        self.history = []
        return self.t, self.wealth

    def step(self, a_g: float, a_p: float, z: float) -> StepRecord:
        if self.t > self.scenario.T:
            raise RuntimeError("episode is already done")

        t = self.t
        wealth_before = float(self.wealth)
        g = goal_decision(a_g, wealth_before, self.scenario.C[t], self.a_thresh)
        utility = float(g * self.scenario.U[t])
        wealth_after = wealth_before - float(g * self.scenario.C[t])
        p = portfolio_decision(a_p, self.scenario.P)

        if t < self.scenario.T:
            wealth_next = evolve_wealth(
                wealth_after,
                float(self.scenario.mu[p]),
                float(self.scenario.sigma[p]),
                z,
                self.scenario.h,
                float(self.scenario.I[t + 1]),
            )
            done = False
            self.t += 1
            self.wealth = wealth_next
        else:
            wealth_next = wealth_after
            done = True
            self.t += 1
            self.wealth = wealth_next

        record = StepRecord(
            t=t,
            wealth_before_goal=wealth_before,
            goal_action=float(a_g),
            goal_taken=g,
            portfolio_action=float(a_p),
            portfolio_index=p,
            utility_attained=utility,
            wealth_after_goal=wealth_after,
            z=float(z),
            wealth_next=float(wealth_next),
            done=done,
        )
        self.history.append(record)
        return record

