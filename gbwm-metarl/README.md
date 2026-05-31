# GBWM MetaRL Reproduction

This project reproduces the all-or-nothing GBWM MetaRL algorithm from
*A Meta Reinforcement Learning Approach to Goals-Based Wealth Management*.

The formal path is:

1. Build the baseline efficient frontier from user-supplied monthly NAV data.
2. Train the dual PPO MetaRL agents in a torch environment, such as Colab.
3. Evaluate the 66 paper cases with DP and MetaRL on shared Monte Carlo paths.

Synthetic frontier and heuristic policies are debug-only helpers. They are not
used as formal reproduction outputs unless explicitly requested by command-line
flags that mark them as simulated/debug provenance.

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

For local smoke/debug only:

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
