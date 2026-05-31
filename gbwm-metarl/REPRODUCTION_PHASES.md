# Single-Phase Reproduction Tracker

Goal: run the paper's main all-or-nothing pipeline with a deterministic
synthetic baseline frontier, dual PPO training, DP comparison, Table 1/Table 2
style outputs, and case 20/57 policy-grid data.

Current status:

- Done: GBWM environment, 26-dimensional Appendix A state, 66 Appendix C cases.
- Done: DP baseline and Monte Carlo policy evaluator.
- Done: strict frontier builder from `data/raw/frontier_nav_monthly.csv`.
- Done: synthetic baseline builder for `data/frontiers/baseline_1998_2017.csv`.
- Done: torch-gated dual PPO implementation and checkpoint loader.
- Done: evaluation scripts now target DP vs MetaRL rather than greedy results.
- Done: formal gates require the full paper-like checkpoint set by default.
- Done: `experiments/08_run_mainline_reproduction.py` encodes the one-phase run.
- Done: synthetic baseline frontier CSV and manifest can be used for baseline training.
- Pending: Colab torch run for PPO smoke, mini, and paper-like training.

Next concrete gates:

1. Run `experiments/00_build_frontier.py --frontier-source simulated`.
2. In Colab, install `requirements-colab.txt` and run smoke then mini PPO training.
3. Run paper-like PPO training.
4. Run 66-case evaluation with trained checkpoints and 10,000 MC paths.
5. Export case 20/57 DP-vs-MetaRL heatmap data.

Completion command:

```bash
python experiments/08_run_mainline_reproduction.py --device cuda
```

Out of scope: concurrent goals, partial goals, stochastic inflation, and
frontier robustness.

Note: synthetic-baseline outputs are not true NAV-based paper numeric
reproduction, and reports must keep `frontier_status=synthetic_baseline_frontier`.
