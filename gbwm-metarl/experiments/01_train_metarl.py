from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.efficient_frontier import baseline_efficient_frontier_spec, simulated_efficient_frontier_spec  # noqa: E402
from gbwm.ppo import PPOConfig, PPOTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Appendix A/B dual PPO MetaRL agents.")
    parser.add_argument("--mode", choices=("smoke", "mini", "paper-like"), default="smoke")
    parser.add_argument("--frontier-source", choices=("baseline", "simulated"), default="baseline")
    parser.add_argument("--frontier-csv", default=str(ROOT / "data" / "frontiers" / "baseline_1998_2017.csv"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(PPOConfig.preset(args.mode), checkpoint_dir=args.checkpoint_dir, device=args.device)
    spec = (
        baseline_efficient_frontier_spec(path=args.frontier_csv)
        if args.frontier_source == "baseline"
        else simulated_efficient_frontier_spec()
    )
    trainer = PPOTrainer(spec.mu, spec.sigma, config=config, frontier_hash=spec.frontier_hash)
    result = trainer.train()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
