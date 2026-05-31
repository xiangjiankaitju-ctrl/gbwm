# GBWM MetaRL Mainline Reproduction

This project reproduces the all-or-nothing GBWM MetaRL algorithm from
*A Meta Reinforcement Learning Approach to Goals-Based Wealth Management*.

Scope is intentionally limited to the paper's main all-or-nothing pipeline. The
extension experiments are not part of the completion target.

The formal single-phase path is:

1. Build the baseline efficient frontier from user-supplied monthly NAV data.
2. Train the dual PPO MetaRL agents in a torch environment, such as Colab.
3. Evaluate the 66 paper cases with DP and MetaRL on shared Monte Carlo paths.
4. Export case 20/57 DP-vs-MetaRL policy-grid data.

Synthetic frontier and heuristic policies are clearly marked in output
provenance. The current runnable mainline uses a deterministic synthetic
baseline frontier when raw NAV data is unavailable.

## Synthetic Baseline

Build the synthetic baseline frontier:

```bash
python experiments/00_build_frontier.py --frontier-source simulated
```

This writes `data/frontiers/baseline_1998_2017.csv` with 15 portfolio points and
marks the manifest as `synthetic_baseline_frontier`. Results from this path are
synthetic-baseline experiments, not true NAV-based paper numeric reproduction.

## One Command

After the synthetic baseline is built and torch is installed, the mainline run is:

```bash
python experiments/08_run_mainline_reproduction.py --device cuda
```

The script runs local checks, builds the baseline frontier, trains the 5-seed
paper-like PPO preset, evaluates 66 cases with 10,000 Monte Carlo paths, and
exports case 20/57 heatmap data. If paper-like checkpoints already exist:

```bash
python experiments/08_run_mainline_reproduction.py --skip-training --device cuda
```

## Frontier

Expected raw CSV:

```text
data/raw/frontier_nav_monthly.csv
date,VTSMX,VTBIX,VGTSX
1998-01-31,...
```

Build the paper baseline frontier:

```powershell
& 'C:\Users\xiangjiankai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments\00_build_frontier.py --frontier-source csv
```

Build the synthetic baseline frontier:

```powershell
& 'C:\Users\xiangjiankai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments\00_build_frontier.py --frontier-source simulated
```

## Colab Training

```bash
pip install -r requirements-colab.txt
python experiments/01_train_metarl.py --mode smoke --frontier-source baseline
python experiments/01_train_metarl.py --mode mini --frontier-source baseline
python experiments/01_train_metarl.py --mode paper-like --frontier-source baseline
```

The paper-like preset uses seeds `0,15,722,1021,5069` and
`1000 epochs * 500 episodes/epoch`.
The formal evaluator defaults to this exact checkpoint set:
`outputs/checkpoints/metarl_paper-like_seed_*.pt`.

## Evaluation

Generate Table 2 style DP vs MetaRL efficiency results:

```bash
python experiments/02_eval_66_cases.py --mode quick --mc-paths 10000 --checkpoint-dir outputs/checkpoints
```

Export case 20/57 policy-grid data:

```bash
python experiments/07_export_heatmap_data.py --case-ids 20,57 --checkpoint-dir outputs/checkpoints
```

## Local Checks

The local runtime does not need torch for non-PPO tests:

```powershell
& 'C:\Users\xiangjiankai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests
& 'C:\Users\xiangjiankai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' experiments\00_smoke_test.py
```
