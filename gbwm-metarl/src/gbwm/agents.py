"""Agent interfaces for the planned torch PPO implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkSpec:
    input_dim: int = 26
    actor_hidden: tuple[int, ...] = (256, 64, 16)
    critic_hidden: tuple[int, ...] = (64, 16)
    hidden_activation: str = "tanh"
    actor_output: str = "sigmoid"
    critic_output: str = "linear"


DEFAULT_NETWORK_SPEC = NetworkSpec()

