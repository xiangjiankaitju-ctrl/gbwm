"""MetaRL state features for the all-or-nothing GBWM problem."""

from __future__ import annotations

from statistics import NormalDist
import math

import numpy as np

from .config import DEFAULT_FORWARD_SIM_N, DEFAULT_TIME_BLOCKS, EPS, logistic
from .scenario import Scenario


def normal_midpoint_nodes(n: int = DEFAULT_FORWARD_SIM_N) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    dist = NormalDist()
    return np.array([dist.inv_cdf((i + 0.5) / n) for i in range(n)], dtype=float)


def discount_vec(values: np.ndarray, t: int, p: int, z: float, scenario: Scenario) -> np.ndarray:
    """Discount values at times t..T into time-t dollars under fixed p and z."""

    future = np.asarray(values[t : scenario.T + 1], dtype=float)
    offsets = np.arange(future.shape[0], dtype=float)
    drift = float(scenario.mu[p]) - 0.5 * float(scenario.sigma[p]) ** 2
    exponent = -drift * scenario.h * offsets - float(scenario.sigma[p]) * float(z) * np.sqrt(scenario.h * offsets)
    return future * np.exp(exponent)


def discount_sum(values: np.ndarray, t: int, p: int, z: float, scenario: Scenario) -> float:
    return float(np.sum(discount_vec(values, t, p, z, scenario)))


def aggregate(values: np.ndarray, blocks=DEFAULT_TIME_BLOCKS) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = []
    for block in blocks:
        if isinstance(block, slice):
            selected = values[block]
        else:
            indexes = [idx for idx in block if idx < values.shape[0]]
            selected = values[indexes] if indexes else np.array([], dtype=float)
        out.append(float(np.sum(selected)) if selected.size else 0.0)
    return np.array(out, dtype=float)


def _safe_normalize(values: np.ndarray, denom: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if denom is None:
        denom = float(np.sum(values))
    if abs(denom) <= EPS:
        return np.zeros_like(values, dtype=float)
    return values / float(denom)


def _consume_for_goal(cash: float, infusions: np.ndarray, goal_offset: int, cost: float) -> tuple[bool, float, np.ndarray]:
    available = cash + float(np.sum(infusions[: goal_offset + 1]))
    if available + EPS < cost:
        return False, cash, infusions

    remaining = float(cost)
    updated = infusions.copy()
    for idx in range(goal_offset, -1, -1):
        use = min(float(updated[idx]), remaining)
        updated[idx] -= use
        remaining -= use
        if remaining <= EPS:
            break
    if remaining > EPS:
        cash -= remaining
    return True, cash, updated


def _utility_from_order(
    wealth: float,
    C_disc: np.ndarray,
    I_disc: np.ndarray,
    U_future: np.ndarray,
    order: list[int],
) -> float:
    cash = float(wealth)
    infusions = np.asarray(I_disc, dtype=float).copy()
    utility = 0.0
    for offset in order:
        cost = float(C_disc[offset])
        if cost <= 0.0 or U_future[offset] <= 0.0:
            continue
        paid, cash, infusions = _consume_for_goal(cash, infusions, offset, cost)
        if paid:
            utility += float(U_future[offset])
    return utility


def _utility_order(U_future: np.ndarray, force_first_offset: int | None = None, skip_offset: int | None = None) -> list[int]:
    offsets = [idx for idx, value in enumerate(U_future) if value > 0.0 and idx != skip_offset]
    offsets.sort(key=lambda idx: (-float(U_future[idx]), idx))
    if force_first_offset is not None and force_first_offset in offsets:
        offsets.remove(force_first_offset)
        offsets.insert(0, force_first_offset)
    return offsets


def compute_g_sim(scenario: Scenario, t: int, wealth_before_goal: float, n: int = DEFAULT_FORWARD_SIM_N) -> float:
    if scenario.C[t] <= 0.0 or scenario.U[t] <= 0.0:
        return 0.0

    z_nodes = normal_midpoint_nodes(n)
    U_future = scenario.U[t : scenario.T + 1]
    take_order = _utility_order(U_future, force_first_offset=0)
    skip_order = _utility_order(U_future, skip_offset=0)
    e_take = np.zeros(scenario.P, dtype=float)
    e_skip = np.zeros(scenario.P, dtype=float)

    for p in range(scenario.P):
        take_values = []
        skip_values = []
        for z in z_nodes:
            c_disc = discount_vec(scenario.C, t, p, float(z), scenario)
            i_disc = discount_vec(scenario.I, t, p, float(z), scenario)
            take_values.append(_utility_from_order(wealth_before_goal, c_disc, i_disc, U_future, take_order))
            skip_values.append(_utility_from_order(wealth_before_goal, c_disc, i_disc, U_future, skip_order))
        e_take[p] = float(np.mean(take_values))
        e_skip[p] = float(np.mean(skip_values))

    max_take = float(np.max(e_take))
    max_skip = float(np.max(e_skip))
    if max_take <= EPS:
        return 0.0
    return float(logistic((max_take - max_skip) / max_take))


def compute_p_sim(scenario: Scenario, t: int, wealth_before_goal: float, n: int = DEFAULT_FORWARD_SIM_N) -> float:
    if scenario.P == 1:
        return 0.0

    z_nodes = normal_midpoint_nodes(n)
    U_future = scenario.U[t : scenario.T + 1]
    order = _utility_order(U_future)
    e = np.zeros(scenario.P, dtype=float)
    for p in range(scenario.P):
        values = []
        for z in z_nodes:
            c_disc = discount_vec(scenario.C, t, p, float(z), scenario)
            i_disc = discount_vec(scenario.I, t, p, float(z), scenario)
            values.append(_utility_from_order(wealth_before_goal, c_disc, i_disc, U_future, order))
        e[p] = float(np.mean(values))
    return float(int(np.argmax(e)) / (scenario.P - 1))


def build_state(scenario: Scenario, t: int, wealth_before_goal: float) -> np.ndarray:
    """Build the paper's 26-dimensional normalized state vector."""

    if t < 1 or t > scenario.T:
        raise ValueError("t must be in 1..T")
    p_cons = 0
    p_aggr = scenario.P - 1
    c_cons = discount_vec(scenario.C, t, p_cons, -1.0, scenario)
    c_aggr = discount_vec(scenario.C, t, p_aggr, 1.0, scenario)

    t_norm = float(t / scenario.T)
    W_min = float(wealth_before_goal / max(float(np.sum(c_cons)), EPS))
    W_max = float(wealth_before_goal / max(float(np.sum(c_aggr)), EPS))

    U_future = scenario.U[t : scenario.T + 1]
    U_agg_raw = aggregate(U_future)
    U_agg = _safe_normalize(U_agg_raw, float(np.sum(U_future)))

    C_min_raw = aggregate(c_aggr)
    C_max_raw = aggregate(c_cons)
    C_min = _safe_normalize(C_min_raw, float(np.sum(c_aggr)))
    C_max = _safe_normalize(C_max_raw, float(np.sum(c_cons)))

    g_sim = compute_g_sim(scenario, t, wealth_before_goal)
    p_sim = compute_p_sim(scenario, t, wealth_before_goal)

    state = np.concatenate(
        [
            np.array([t_norm, W_min, W_max], dtype=float),
            U_agg,
            C_min,
            C_max,
            np.array([g_sim, p_sim], dtype=float),
        ]
    )
    if state.shape != (26,):
        raise RuntimeError(f"state shape must be (26,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise RuntimeError(f"state contains non-finite values: {state}")
    return state.astype(np.float32)

