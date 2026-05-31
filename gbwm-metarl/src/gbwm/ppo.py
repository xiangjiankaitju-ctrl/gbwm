"""Torch PPO implementation for the Appendix A/B dual-agent MetaRL algorithm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .agents import DEFAULT_NETWORK_SPEC, NetworkSpec
from .environment import GBWMEnv
from .rewards import extrinsic_reward, intrinsic_goal_reward, intrinsic_lambda, intrinsic_portfolio_reward
from .scenario_generation import generate_training_scenario
from .state_features import build_state


class MissingTorchDependency(RuntimeError):
    pass


def require_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingTorchDependency(
            "PPO training requires torch. Install torch, e.g. in Colab with `pip install -r requirements-colab.txt`."
        ) from exc
    return torch


@dataclass(frozen=True)
class PPOConfig:
    mode: str = "smoke"
    epochs: int = 2
    episodes_per_epoch: int = 8
    seeds: tuple[int, ...] = (0,)
    curriculum_seed: int = 260502300
    learning_rate: float = 1e-4
    clip: float = 0.2
    discount: float = 1.0
    action_std: float = 0.10
    ppo_update_epochs: int = 4
    intrinsic_start: float = 1.0
    intrinsic_end: float = 0.25
    checkpoint_dir: str = "outputs/checkpoints"
    device: str = "cpu"
    network: NetworkSpec = field(default_factory=lambda: DEFAULT_NETWORK_SPEC)

    @classmethod
    def preset(cls, mode: str) -> "PPOConfig":
        if mode == "smoke":
            return cls(mode="smoke", epochs=2, episodes_per_epoch=8, seeds=(0,))
        if mode == "mini":
            return cls(mode="mini", epochs=20, episodes_per_epoch=50, seeds=(0, 15))
        if mode == "paper-like":
            return cls(mode="paper-like", epochs=1000, episodes_per_epoch=500, seeds=(0, 15, 722, 1021, 5069))
        raise ValueError("mode must be 'smoke', 'mini', or 'paper-like'")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["network"] = asdict(self.network)
        payload["seeds"] = list(self.seeds)
        return payload


def _make_mlp(torch, input_dim: int, hidden: tuple[int, ...], output_dim: int, output_activation: str):
    import torch.nn as nn  # type: ignore

    layers: list[Any] = []
    prev = input_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.Tanh())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    if output_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif output_activation == "linear":
        pass
    else:
        raise ValueError(f"Unsupported output activation: {output_activation}")
    return nn.Sequential(*layers)


class ActorCriticAgent:
    """One scalar actor and one scalar critic, matching Appendix B.1."""

    def __init__(self, name: str, config: PPOConfig):
        self.torch = require_torch()
        self.name = name
        self.config = config
        self.device = self.torch.device(config.device)
        spec = config.network
        self.actor = _make_mlp(self.torch, spec.input_dim, spec.actor_hidden, 1, spec.actor_output).to(self.device)
        self.critic = _make_mlp(self.torch, spec.input_dim, spec.critic_hidden, 1, spec.critic_output).to(self.device)
        self.actor_optimizer = self.torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = self.torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)

    def _state_tensor(self, state: np.ndarray):
        return self.torch.as_tensor(state, dtype=self.torch.float32, device=self.device).view(1, -1)

    def mean_action(self, state: np.ndarray) -> float:
        with self.torch.no_grad():
            mean = self.actor(self._state_tensor(state)).squeeze()
        return float(mean.detach().cpu().item())

    def sample_action(self, state: np.ndarray) -> tuple[float, float]:
        state_t = self._state_tensor(state)
        mean = self.actor(state_t).squeeze(-1)
        std = self.torch.full_like(mean, float(self.config.action_std))
        dist = self.torch.distributions.Normal(mean, std)
        raw_action = dist.sample()
        action = self.torch.clamp(raw_action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum()
        return float(action.detach().cpu().item()), float(log_prob.detach().cpu().item())

    def update(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray, returns: np.ndarray) -> dict[str, float]:
        torch = self.torch
        if len(states) == 0:
            return {f"{self.name}_actor_loss": 0.0, f"{self.name}_critic_loss": 0.0}
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device).view(-1, 1)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device).view(-1, 1)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).view(-1, 1)

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        for _ in range(self.config.ppo_update_epochs):
            values = self.critic(states_t)
            advantages = returns_t - values.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            mean = self.actor(states_t)
            std = torch.full_like(mean, float(self.config.action_std))
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions_t)
            ratios = torch.exp(log_probs - old_log_probs_t)
            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1.0 - self.config.clip, 1.0 + self.config.clip) * advantages
            actor_loss = -torch.mean(torch.minimum(unclipped, clipped))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            critic_loss = torch.mean((self.critic(states_t) - returns_t) ** 2)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu().item())
            critic_loss_value = float(critic_loss.detach().cpu().item())

        return {f"{self.name}_actor_loss": actor_loss_value, f"{self.name}_critic_loss": critic_loss_value}

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])


class GoalAgent(ActorCriticAgent):
    def __init__(self, config: PPOConfig):
        super().__init__("goal", config)


class PortfolioAgent(ActorCriticAgent):
    def __init__(self, config: PPOConfig):
        super().__init__("portfolio", config)


def _portfolio_returns(extrinsic: list[float], intrinsic_p: list[float]) -> list[float]:
    out: list[float] = []
    for idx in range(len(intrinsic_p)):
        out.append(float(np.sum(extrinsic[idx + 1 :]) + np.sum(intrinsic_p[idx:])))
    return out


def _goal_returns(extrinsic: list[float], intrinsic_g: list[float]) -> list[float]:
    rewards = [float(e + i) for e, i in zip(extrinsic, intrinsic_g)]
    running = 0.0
    out = [0.0] * len(rewards)
    for idx in range(len(rewards) - 1, -1, -1):
        running += rewards[idx]
        out[idx] = running
    return out


class PPOTrainer:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, config: PPOConfig | None = None, frontier_hash: str | None = None):
        self.config = config or PPOConfig.preset("smoke")
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.frontier_hash = frontier_hash
        self.torch = require_torch()

    def _train_seed(self, seed: int) -> dict:
        self.torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        goal_agent = GoalAgent(self.config)
        portfolio_agent = PortfolioAgent(self.config)
        logs: list[dict] = []

        for epoch in range(self.config.epochs):
            scenario_rng = np.random.default_rng(int(self.config.curriculum_seed) + int(epoch))
            base_scenario = generate_training_scenario(self.mu, self.sigma, scenario_rng, name=f"epoch-{epoch}")
            lambda_i = intrinsic_lambda(epoch, self.config.epochs, self.config.intrinsic_start, self.config.intrinsic_end)

            goal_states: list[np.ndarray] = []
            goal_actions: list[float] = []
            goal_old_log_probs: list[float] = []
            goal_returns_all: list[float] = []
            portfolio_states: list[np.ndarray] = []
            portfolio_actions: list[float] = []
            portfolio_old_log_probs: list[float] = []
            portfolio_returns_all: list[float] = []
            attained_utilities: list[float] = []

            for _episode in range(self.config.episodes_per_epoch):
                initial_wealth = float(rng.uniform(0.8 * base_scenario.W0, 1.2 * base_scenario.W0))
                env = GBWMEnv(base_scenario)
                t, wealth = env.reset(initial_wealth)
                episode_goal_states: list[np.ndarray] = []
                episode_goal_actions: list[float] = []
                episode_goal_log_probs: list[float] = []
                episode_portfolio_states: list[np.ndarray] = []
                episode_portfolio_actions: list[float] = []
                episode_portfolio_log_probs: list[float] = []
                episode_extrinsic: list[float] = []
                episode_intrinsic_g: list[float] = []
                episode_intrinsic_p: list[float] = []
                episode_attained = 0.0

                while t <= base_scenario.T:
                    wealth_before_goal = float(wealth)
                    g_state = build_state(base_scenario, t, wealth_before_goal)
                    a_g, log_g = goal_agent.sample_action(g_state)
                    goal_taken = int(a_g >= 0.5 and wealth_before_goal + 1e-12 >= base_scenario.C[t] and base_scenario.C[t] > 0.0)
                    wealth_after_goal = wealth_before_goal - float(goal_taken * base_scenario.C[t])
                    p_state = build_state(base_scenario, t, wealth_after_goal)
                    a_p, log_p = portfolio_agent.sample_action(p_state)
                    z = float(rng.standard_normal())
                    record = env.step(a_g, a_p, z)

                    episode_goal_states.append(g_state)
                    episode_goal_actions.append(a_g)
                    episode_goal_log_probs.append(log_g)
                    episode_portfolio_states.append(p_state)
                    episode_portfolio_actions.append(a_p)
                    episode_portfolio_log_probs.append(log_p)
                    episode_extrinsic.append(extrinsic_reward(base_scenario, t, wealth_before_goal, a_g, record.goal_taken))
                    episode_intrinsic_g.append(intrinsic_goal_reward(base_scenario, t, wealth_before_goal, a_g, lambda_i))
                    episode_intrinsic_p.append(intrinsic_portfolio_reward(base_scenario, t, wealth_after_goal, a_p, lambda_i))
                    episode_attained += float(record.utility_attained)

                    if record.done:
                        break
                    t = env.t
                    wealth = env.wealth

                goal_states.extend(episode_goal_states)
                goal_actions.extend(episode_goal_actions)
                goal_old_log_probs.extend(episode_goal_log_probs)
                goal_returns_all.extend(_goal_returns(episode_extrinsic, episode_intrinsic_g))
                portfolio_states.extend(episode_portfolio_states)
                portfolio_actions.extend(episode_portfolio_actions)
                portfolio_old_log_probs.extend(episode_portfolio_log_probs)
                portfolio_returns_all.extend(_portfolio_returns(episode_extrinsic, episode_intrinsic_p))
                attained_utilities.append(episode_attained)

            goal_losses = goal_agent.update(
                np.asarray(goal_states, dtype=np.float32),
                np.asarray(goal_actions, dtype=np.float32),
                np.asarray(goal_old_log_probs, dtype=np.float32),
                np.asarray(goal_returns_all, dtype=np.float32),
            )
            portfolio_losses = portfolio_agent.update(
                np.asarray(portfolio_states, dtype=np.float32),
                np.asarray(portfolio_actions, dtype=np.float32),
                np.asarray(portfolio_old_log_probs, dtype=np.float32),
                np.asarray(portfolio_returns_all, dtype=np.float32),
            )
            logs.append(
                {
                    "epoch": int(epoch),
                    "seed": int(seed),
                    "lambda_i": float(lambda_i),
                    "mean_attained_utility": float(np.mean(attained_utilities)),
                    **goal_losses,
                    **portfolio_losses,
                }
            )

        checkpoint = {
            "seed": int(seed),
            "config": self.config.to_dict(),
            "frontier_hash": self.frontier_hash,
            "goal": goal_agent.state_dict(),
            "portfolio": portfolio_agent.state_dict(),
            "logs": logs,
        }
        out_dir = Path(self.config.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"metarl_{self.config.mode}_seed_{seed}.pt"
        self.torch.save(checkpoint, path)
        return {"seed": int(seed), "checkpoint": str(path.resolve()), "logs": logs}

    def train(self) -> dict:
        results = [self._train_seed(seed) for seed in self.config.seeds]
        return {"mode": self.config.mode, "seeds": list(self.config.seeds), "results": results}


def load_checkpoint_agents(path: str | Path, device: str = "cpu") -> tuple[GoalAgent, PortfolioAgent, dict]:
    torch = require_torch()
    payload = torch.load(Path(path), map_location=device)
    raw_config = dict(payload["config"])
    network_payload = raw_config.pop("network", None)
    if network_payload:
        raw_config["network"] = NetworkSpec(**network_payload)
    if "seeds" in raw_config:
        raw_config["seeds"] = tuple(raw_config["seeds"])
    raw_config["device"] = device
    config = PPOConfig(**raw_config)
    goal = GoalAgent(config)
    portfolio = PortfolioAgent(config)
    goal.load_state_dict(payload["goal"])
    portfolio.load_state_dict(payload["portfolio"])
    goal.actor.eval()
    portfolio.actor.eval()
    return goal, portfolio, payload
