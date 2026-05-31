# Reproduction Tracker

## Phase 1: All-or-Nothing GBWM MetaRL

Goal: reproduce the paper's main all-or-nothing pipeline with a strict frontier,
dual PPO training, DP comparison, Table 1/Table 2 style outputs, and case 20/57
policy-grid data.

Current status:

- Done: GBWM environment, 26-dimensional Appendix A state, 66 Appendix C cases.
- Done: DP baseline and Monte Carlo policy evaluator.
- Done: strict frontier builder from `data/raw/frontier_nav_monthly.csv`.
- Done: explicit simulated frontier path for local debug only.
- Done: torch-gated dual PPO implementation and checkpoint loader.
- Done: evaluation scripts now target DP vs MetaRL rather than greedy results.
- Pending: user NAV CSV for strict numeric frontier reproduction.
- Pending: Colab torch run for PPO smoke, mini, and paper-like training.

Next concrete gates:

1. Provide `data/raw/frontier_nav_monthly.csv`.
2. Run `experiments/00_build_frontier.py --frontier-source csv`.
3. In Colab, install `requirements-colab.txt` and run PPO smoke.
4. Run 66-case evaluation with trained checkpoints and 10,000 MC paths.

## Phase 2: Extensions

Partial goals, concurrent goals, stochastic inflation, and robustness frontiers
remain out of scope until Phase 1 has trained MetaRL checkpoints and formal
DP-vs-MetaRL outputs.
