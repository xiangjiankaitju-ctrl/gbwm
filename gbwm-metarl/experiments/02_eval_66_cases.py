from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.dp_baseline import solve_dp
from gbwm.dp_baseline import default_z_grid
from gbwm.metrics import evaluate_policies, evaluation_summary
from gbwm.paper_cases import paper_case_specs, scenario_from_case_spec
from gbwm.policies import DPGridPolicy, MetaRLPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Appendix C with DP and trained MetaRL policies.")
    parser.add_argument("--mode", choices=("quick", "calibration"), default="quick", help="Run one DP config or a DP sensitivity calibration.")
    parser.add_argument("--grid-size", type=int, default=91, help="Number of wealth-grid points for DP.")
    parser.add_argument("--z-nodes", type=int, default=7, help="Number of deterministic normal midpoint nodes for DP.")
    parser.add_argument("--mc-paths", type=int, default=10000, help="Monte Carlo paths per case.")
    parser.add_argument("--seed", type=int, default=260502300, help="Random seed for Monte Carlo paths.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "tables"), help="Directory for JSON and CSV outputs.")
    parser.add_argument("--calibration-configs", default="51:5,91:7,151:11", help="Comma-separated grid:z pairs used in calibration mode.")
    parser.add_argument("--case-ids", default="", help="Optional comma-separated case IDs to evaluate, e.g. 20,57.")
    parser.add_argument("--frontier-source", choices=("baseline", "simulated"), default="baseline")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "checkpoints"), help="Directory containing MetaRL seed checkpoints.")
    parser.add_argument("--checkpoint-paths", default="", help="Optional comma-separated checkpoint paths. Overrides --checkpoint-dir.")
    parser.add_argument("--device", default="cpu", help="Torch device for MetaRL inference.")
    parser.add_argument("--stable-threshold", type=float, default=0.03, help="Relative DP utility spread considered stable.")
    parser.add_argument("--mild-threshold", type=float, default=0.07, help="Relative DP utility spread considered mildly sensitive.")
    parser.add_argument("--high-threshold", type=float, default=0.15, help="Relative DP utility spread considered highly sensitive.")
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_paths.strip():
        paths = [Path(chunk.strip()) for chunk in args.checkpoint_paths.split(",") if chunk.strip()]
    else:
        paths = sorted(Path(args.checkpoint_dir).glob("*.pt"))
    if not paths:
        raise ValueError("No MetaRL checkpoints found. Run experiments/01_train_metarl.py first.")
    return paths


def selected_specs(args: argparse.Namespace) -> list[dict]:
    specs = paper_case_specs()
    if not args.case_ids.strip():
        return specs
    wanted = {int(chunk.strip()) for chunk in args.case_ids.split(",") if chunk.strip()}
    selected = [spec for spec in specs if int(spec["case_id"]) in wanted]
    missing = wanted.difference(int(spec["case_id"]) for spec in selected)
    if missing:
        raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    return selected


def wealth_grid_for_case(scenario, grid_size: int) -> np.ndarray:
    max_goal_cost = float(np.sum(scenario.C))
    max_infusion = float(np.sum(scenario.I))
    upper = max(2.5 * max(scenario.W0 + max_infusion, max_goal_cost), 200.0)
    low_count = max(20, int(round(grid_size * 0.6)))
    high_count = max(10, grid_size - low_count)
    low_upper = max(10.0 * scenario.W0 + max_infusion, 200.0)
    low_upper = min(low_upper, upper)
    low_grid = np.linspace(0.0, low_upper, low_count)
    if upper <= low_upper:
        grid = low_grid
    else:
        start = max(low_upper, 1.0)
        high_grid = np.geomspace(start, upper, high_count)
        grid = np.concatenate([low_grid, high_grid])
    anchors = [0.0, float(scenario.W0), upper]
    anchors.extend(float(x) for x in scenario.C[scenario.C > 0.0])
    anchors.extend(float(x) for x in np.cumsum(scenario.I))
    grid = np.concatenate([grid, np.array(anchors, dtype=float)])
    grid = np.unique(np.round(grid, 10))
    grid = grid[(grid >= 0.0) & (grid <= upper)]
    return np.sort(grid)


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_66_cases_summary.json"
    csv_path = output_dir / "eval_66_cases_rows.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def parse_calibration_configs(raw: str) -> list[tuple[int, int]]:
    configs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        grid_raw, z_raw = chunk.split(":", 1)
        grid_size = int(grid_raw)
        z_nodes = int(z_raw)
        if grid_size < 2 or z_nodes < 1:
            raise ValueError("calibration configs must use grid_size >= 2 and z_nodes >= 1")
        configs.append((grid_size, z_nodes))
    if not configs:
        raise ValueError("at least one calibration config is required")
    return configs


def dp_value_at_initial_wealth(dp, scenario) -> float:
    return float(np.interp(scenario.W0, dp.wealth_grid, dp.value_grid[1], left=dp.value_grid[1, 0], right=dp.value_grid[1, -1]))


def monotonicity_violations(dp) -> int:
    relevant = dp.value_grid[1 : dp.goal_policy_grid.shape[0]]
    diffs = np.diff(relevant, axis=1)
    return int(np.sum(diffs < -1e-8))


def classify_stability(dp_utilities: list[float], metarl_utility: float, args: argparse.Namespace) -> tuple[str, bool, float, bool]:
    finite = np.array([x for x in dp_utilities if np.isfinite(x)], dtype=float)
    if finite.size == 0:
        return "needs_review", False, float("nan"), True
    scale = max(float(np.max(np.abs(finite))), 1e-12)
    rel_spread = float((np.max(finite) - np.min(finite)) / scale)
    reference_dp = float(finite[-1])
    dp_below_metarl = bool(reference_dp + 1e-9 < metarl_utility)
    if dp_below_metarl or (reference_dp <= 0.0 and metarl_utility > 0.0):
        return "needs_review", False, rel_spread, dp_below_metarl
    if rel_spread <= args.stable_threshold:
        return "stable", True, rel_spread, False
    if rel_spread <= args.mild_threshold:
        return "mildly_sensitive", True, rel_spread, False
    if rel_spread <= args.high_threshold:
        return "highly_sensitive", False, rel_spread, False
    return "needs_review", False, rel_spread, False


def evaluate_one_config(scenario, wealth_grid: np.ndarray, z_nodes_count: int, z_paths: np.ndarray, metarl_policy) -> tuple[dict, object]:
    z_nodes, z_weights = default_z_grid(z_nodes_count)
    start = time.perf_counter()
    dp = solve_dp(scenario, wealth_grid, z_nodes=z_nodes, z_weights=z_weights)
    dp_seconds = time.perf_counter() - start
    portfolio_only_times = []
    goal_and_portfolio_times = []
    probe_wealth = float(scenario.W0)
    for t in range(1, scenario.T + 1):
        start_action = time.perf_counter()
        if scenario.C[t] > 0.0:
            metarl_policy(scenario, t, probe_wealth)
            goal_and_portfolio_times.append(time.perf_counter() - start_action)
        else:
            metarl_policy.portfolio_only_action(scenario, t, probe_wealth)
            portfolio_only_times.append(time.perf_counter() - start_action)
    results = evaluate_policies(
        scenario,
        {
            "dp": DPGridPolicy(dp),
            "metarl": metarl_policy,
        },
        z_paths,
    )
    row = {
        "dp_seconds": dp_seconds,
        "metarl_portfolio_only_mean_seconds": float(np.mean(portfolio_only_times)) if portfolio_only_times else float("nan"),
        "metarl_goal_and_portfolio_mean_seconds": float(np.mean(goal_and_portfolio_times)) if goal_and_portfolio_times else float("nan"),
        "dp_value_at_W0": dp_value_at_initial_wealth(dp, scenario),
        "monotonicity_violations": monotonicity_violations(dp),
    }
    row.update(evaluation_summary(results, baseline_name="dp"))
    return row, dp


def run_quick(args: argparse.Namespace) -> dict:
    specs = selected_specs(args)
    metarl_policy = MetaRLPolicy(checkpoint_paths(args), device=args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    suite_start = time.perf_counter()
    for spec in specs:
        scenario = scenario_from_case_spec(spec, frontier_source=args.frontier_source)
        wealth_grid = wealth_grid_for_case(scenario, args.grid_size)
        z_paths = rng.standard_normal((args.mc_paths, scenario.T))
        config_row, _dp = evaluate_one_config(scenario, wealth_grid, args.z_nodes, z_paths, metarl_policy)
        rows.append(
            {
                "case_id": int(spec["case_id"]),
                "T": scenario.T,
                "goals": int(np.count_nonzero(scenario.C)),
                "wealth_grid_size_requested": args.grid_size,
                "wealth_grid_size_actual": int(wealth_grid.size),
                "z_nodes": args.z_nodes,
                "mc_paths": args.mc_paths,
                "calibration_status": "not_run",
                "calibration_passed": False,
                **config_row,
            }
        )

    efficiencies = np.array([row["metarl_efficiency_vs_dp"] for row in rows if np.isfinite(row["metarl_efficiency_vs_dp"])], dtype=float)
    portfolio_times = np.array([row["metarl_portfolio_only_mean_seconds"] for row in rows if np.isfinite(row["metarl_portfolio_only_mean_seconds"])], dtype=float)
    goal_portfolio_times = np.array([row["metarl_goal_and_portfolio_mean_seconds"] for row in rows if np.isfinite(row["metarl_goal_and_portfolio_mean_seconds"])], dtype=float)
    dp_times = np.array([row["dp_seconds"] for row in rows if np.isfinite(row["dp_seconds"])], dtype=float)
    return {
        "mode": "quick",
        "evaluated_cases": len(rows),
        "case_ids": [int(spec["case_id"]) for spec in specs],
        "grid_size": args.grid_size,
        "mc_paths": args.mc_paths,
        "frontier_source": args.frontier_source,
        "checkpoint_count": len(checkpoint_paths(args)),
        "seed": args.seed,
        "z_nodes": args.z_nodes,
        "suite_seconds": time.perf_counter() - suite_start,
        "metarl_efficiency_mean": float(np.mean(efficiencies)) if efficiencies.size else None,
        "metarl_efficiency_min": float(np.min(efficiencies)) if efficiencies.size else None,
        "metarl_efficiency_max": float(np.max(efficiencies)) if efficiencies.size else None,
        "table1_runtime_seconds": {
            "metarl_portfolio_only_mean": float(np.mean(portfolio_times)) if portfolio_times.size else None,
            "metarl_goal_and_portfolio_mean": float(np.mean(goal_portfolio_times)) if goal_portfolio_times.size else None,
            "dp_backward_pass_mean": float(np.mean(dp_times)) if dp_times.size else None,
        },
        "output_dir": str(Path(args.output_dir).resolve()),
        "rows": rows,
    }


def run_calibration(args: argparse.Namespace) -> dict:
    specs = selected_specs(args)
    configs = parse_calibration_configs(args.calibration_configs)
    metarl_policy = MetaRLPolicy(checkpoint_paths(args), device=args.device)
    rng = np.random.default_rng(args.seed)
    rows = []
    detail_rows = []
    suite_start = time.perf_counter()
    for spec in specs:
        scenario = scenario_from_case_spec(spec, frontier_source=args.frontier_source)
        z_paths = rng.standard_normal((args.mc_paths, scenario.T))
        case_details = []
        for grid_size, z_nodes_count in configs:
            wealth_grid = wealth_grid_for_case(scenario, grid_size)
            config_row, _dp = evaluate_one_config(scenario, wealth_grid, z_nodes_count, z_paths, metarl_policy)
            detail = {
                "case_id": int(spec["case_id"]),
                "T": scenario.T,
                "goals": int(np.count_nonzero(scenario.C)),
                "wealth_grid_size_requested": grid_size,
                "wealth_grid_size_actual": int(wealth_grid.size),
                "z_nodes": z_nodes_count,
                "mc_paths": args.mc_paths,
                **config_row,
            }
            detail_rows.append(detail)
            case_details.append(detail)

        dp_utilities = [float(row["dp_mean_utility"]) for row in case_details]
        metarl_utility = float(case_details[-1]["metarl_mean_utility"])
        status, passed, rel_spread, dp_below_metarl = classify_stability(dp_utilities, metarl_utility, args)
        reference = case_details[-1]
        rows.append(
            {
                "case_id": int(spec["case_id"]),
                "T": scenario.T,
                "goals": int(np.count_nonzero(scenario.C)),
                "calibration_status": status,
                "calibration_passed": passed,
                "dp_relative_spread": rel_spread,
                "dp_below_metarl": dp_below_metarl,
                "reference_grid_size_requested": int(reference["wealth_grid_size_requested"]),
                "reference_grid_size_actual": int(reference["wealth_grid_size_actual"]),
                "reference_z_nodes": int(reference["z_nodes"]),
                "mc_paths": args.mc_paths,
                "dp_mean_utility": float(reference["dp_mean_utility"]),
                "metarl_mean_utility": float(reference["metarl_mean_utility"]),
                "metarl_efficiency_vs_dp": float(reference["metarl_efficiency_vs_dp"]),
                "dp_value_at_W0": float(reference["dp_value_at_W0"]),
                "monotonicity_violations": int(reference["monotonicity_violations"]),
                "dp_seconds_total": float(sum(row["dp_seconds"] for row in case_details)),
            }
        )

    passed_rows = [row for row in rows if row["calibration_passed"]]
    efficiencies = np.array([row["metarl_efficiency_vs_dp"] for row in passed_rows if np.isfinite(row["metarl_efficiency_vs_dp"])], dtype=float)
    return {
        "mode": "calibration",
        "calibration_configs": [{"grid_size": grid, "z_nodes": z} for grid, z in configs],
        "evaluated_cases": len(rows),
        "case_ids": [int(spec["case_id"]) for spec in specs],
        "passed_cases": len(passed_rows),
        "mc_paths": args.mc_paths,
        "frontier_source": args.frontier_source,
        "checkpoint_count": len(checkpoint_paths(args)),
        "seed": args.seed,
        "suite_seconds": time.perf_counter() - suite_start,
        "stable_threshold": args.stable_threshold,
        "mild_threshold": args.mild_threshold,
        "high_threshold": args.high_threshold,
        "passed_metarl_efficiency_mean": float(np.mean(efficiencies)) if efficiencies.size else None,
        "output_dir": str(Path(args.output_dir).resolve()),
        "rows": rows,
        "detail_rows": detail_rows,
    }


def main() -> None:
    args = parse_args()
    summary = run_quick(args) if args.mode == "quick" else run_calibration(args)
    write_outputs(summary["rows"], summary, Path(args.output_dir))
    if args.mode == "calibration":
        detail_path = Path(args.output_dir) / "eval_66_cases_calibration_details.csv"
        with detail_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["detail_rows"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["detail_rows"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
