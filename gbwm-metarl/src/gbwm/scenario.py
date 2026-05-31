"""Scenario data model for all-or-nothing GBWM problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_H


@dataclass(frozen=True)
class Scenario:
    """A single all-or-nothing GBWM scenario.

    Arrays use length ``T + 1`` and indexes ``0..T``. Index 0 is reserved for
    initial conditions and should have zero goal cost, utility, and infusion.
    Effective decisions are made at times ``1..T``.
    """

    T: int
    W0: float
    C: np.ndarray
    U: np.ndarray
    I: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    h: float = DEFAULT_H
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "C", np.asarray(self.C, dtype=float))
        object.__setattr__(self, "U", np.asarray(self.U, dtype=float))
        object.__setattr__(self, "I", np.asarray(self.I, dtype=float))
        object.__setattr__(self, "mu", np.asarray(self.mu, dtype=float))
        object.__setattr__(self, "sigma", np.asarray(self.sigma, dtype=float))
        self._validate()

    @property
    def P(self) -> int:
        return int(self.mu.shape[0])

    @property
    def total_utility(self) -> float:
        return float(np.sum(self.U))

    def _validate(self) -> None:
        if self.T < 1:
            raise ValueError("T must be at least 1")
        expected = self.T + 1
        for field_name in ("C", "U", "I"):
            value = getattr(self, field_name)
            if value.shape != (expected,):
                raise ValueError(f"{field_name} must have shape ({expected},)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{field_name} contains non-finite values")
        if self.C[0] != 0 or self.U[0] != 0 or self.I[0] != 0:
            raise ValueError("C[0], U[0], and I[0] must all be zero")
        if np.any(self.C < 0) or np.any(self.U < 0) or np.any(self.I < 0):
            raise ValueError("C, U, and I must be non-negative")
        if self.mu.ndim != 1 or self.sigma.ndim != 1 or self.mu.shape != self.sigma.shape:
            raise ValueError("mu and sigma must be one-dimensional arrays with matching shapes")
        if self.P < 1:
            raise ValueError("at least one portfolio is required")
        if np.any(self.sigma < 0):
            raise ValueError("sigma must be non-negative")
        if self.W0 < 0:
            raise ValueError("W0 must be non-negative")
        if self.h <= 0:
            raise ValueError("h must be positive")

    @classmethod
    def from_lists(
        cls,
        *,
        T: int,
        W0: float,
        C: list[float],
        U: list[float],
        I: list[float] | None,
        mu: list[float],
        sigma: list[float],
        h: float = DEFAULT_H,
        name: str = "",
    ) -> "Scenario":
        if I is None:
            I = [0.0] * (T + 1)
        return cls(T=T, W0=W0, C=np.array(C), U=np.array(U), I=np.array(I), mu=np.array(mu), sigma=np.array(sigma), h=h, name=name)

