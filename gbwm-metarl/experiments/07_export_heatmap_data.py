from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.dp_baseline import default_z_grid, solve_dp
from gbwm.paper_cases import paper_case_specs, scenario_from_case_spec
from gbwm.policies import DPGridPolicy, MetaRLPolicy
from gbwm.reproduction import (
    FORMAL_CHECKPOINT_MODE,
    resolve_checkpoint_paths,
    validate_baseline_frontier_artifacts,
    validate_checkpoint_metadata,
)


def load_eval_module():
    path = ROOT / "experiments" / "02_eval_66_cases.py"
    spec = importlib.util.spec_from_file_location("eval_66_cases", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Could not load 02_eval_66_cases.py")
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export heatmap-ready policy data for selected paper cases.")
    parser.add_argument("--case-ids", default="20,57")
    parser.add_argument("--grid-size", type=int, default=151)
    parser.add_argument("--z-nodes", type=int, default=11)
    parser.add_argument("--frontier-source", choices=("baseline", "simulated"), default="baseline")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--checkpoint-paths", default="")
    parser.add_argument("--checkpoint-mode", choices=("smoke", "mini", "paper-like"), default=FORMAL_CHECKPOINT_MODE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "heatmaps"))
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    return resolve_checkpoint_paths(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_paths=args.checkpoint_paths,
        mode=args.checkpoint_mode,
    )


def selected_specs(case_ids_raw: str) -> list[dict]:
    wanted = {int(chunk.strip()) for chunk in case_ids_raw.split(",") if chunk.strip()}
    specs = [spec for spec in paper_case_specs() if int(spec["case_id"]) in wanted]
    missing = wanted.difference(int(spec["case_id"]) for spec in specs)
    if missing:
        raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    return specs


def export_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_module = load_eval_module()
    paths = checkpoint_paths(args)
    frontier_info = validate_baseline_frontier_artifacts() if args.frontier_source == "baseline" else None
    checkpoint_info = validate_checkpoint_metadata(
        paths,
        mode=args.checkpoint_mode,
        frontier_hash=frontier_info["frontier_hash"] if frontier_info else None,
        device=args.device,
    )
    metarl_policy = MetaRLPolicy(paths, device=args.device)

    z_nodes, z_weights = default_z_grid(args.z_nodes)
    output_dir = Path(args.output_dir)
    manifest = {
        "case_ids": [],
        "grid_size_requested": args.grid_size,
        "z_nodes": args.z_nodes,
        "policies": ["dp", "metarl"],
        "frontier_source": args.frontier_source,
        "frontier_status": frontier_info["frontier_status"] if frontier_info else "explicit_simulated",
        "frontier": frontier_info,
        "checkpoint_count": len(paths),
        "checkpoint_mode": args.checkpoint_mode,
        "checkpoint_seeds": checkpoint_info["seeds"],
        "files": [],
    }
    start = time.perf_counter()
    for spec in selected_specs(args.case_ids):
        scenario = scenario_from_case_spec(spec, frontier_source=args.frontier_source)
        wealth_grid = eval_module.wealth_grid_for_case(scenario, args.grid_size)
        dp = solve_dp(scenario, wealth_grid, z_nodes=z_nodes, z_weights=z_weights)
        policy_objects = {"dp": DPGridPolicy(dp), "metarl": metarl_policy}

        rows = []
        for t in range(1, scenario.T + 1):
            for wealth in wealth_grid:
                for policy_name, policy in policy_objects.items():
                    action = policy(scenario, t, float(wealth))
                    portfolio_index = min(int(action.portfolio_action * scenario.P), scenario.P - 1)
                    rows.append(
                        {
                            "case_id": int(spec["case_id"]),
                            "policy": policy_name,
                            "t": t,
                            "wealth": float(wealth),
                            "goal_action": float(action.goal_action),
                            "goal_take_decision": int(action.goal_action >= 0.5 and wealth + 1e-12 >= scenario.C[t] and scenario.C[t] > 0.0),
                            "portfolio_action": float(action.portfolio_action),
                            "portfolio_index": portfolio_index,
                        }
                    )
        out_path = output_dir / f"case_{int(spec['case_id']):02d}_heatmap_data.csv"
        export_rows(out_path, rows)
        manifest["case_ids"].append(int(spec["case_id"]))
        manifest["files"].append(str(out_path.resolve()))
    manifest["seconds"] = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "heatmap_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
