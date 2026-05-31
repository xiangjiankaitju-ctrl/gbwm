from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.efficient_frontier import baseline_efficient_frontier_spec, simulated_efficient_frontier_spec


def load_eval_module():
    path = ROOT / "experiments" / "02_eval_66_cases.py"
    spec = importlib.util.spec_from_file_location("eval_66_cases", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Could not load 02_eval_66_cases.py")
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DP baseline audit from calibration runs.")
    parser.add_argument("--calibration-configs", default="51:5,91:7,151:11")
    parser.add_argument("--mc-paths", type=int, default=300)
    parser.add_argument("--seed", type=int, default=260502300)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "tables" / "dp_baseline_audit"))
    parser.add_argument("--frontier-source", choices=("baseline", "simulated"), default="baseline")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--checkpoint-paths", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stable-threshold", type=float, default=0.03)
    parser.add_argument("--mild-threshold", type=float, default=0.07)
    parser.add_argument("--high-threshold", type=float, default=0.15)
    return parser.parse_args()


def write_audit(summary: dict, output_dir: Path, frontier_source: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier = (
        baseline_efficient_frontier_spec().as_dict()
        if frontier_source == "baseline"
        else simulated_efficient_frontier_spec().as_dict()
    )
    audit = {
        "audit_type": "dp_metarl_calibration",
        "frontier": frontier,
        "calibration": {k: v for k, v in summary.items() if k not in {"rows", "detail_rows"}},
        "rows": summary["rows"],
    }
    with (output_dir / "dp_baseline_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if summary["rows"]:
        with (output_dir / "dp_baseline_audit.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["rows"])
    if summary.get("detail_rows"):
        with (output_dir / "dp_baseline_audit_details.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["detail_rows"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["detail_rows"])


def main() -> None:
    args = parse_args()
    eval_module = load_eval_module()
    eval_args = argparse.Namespace(
        mode="calibration",
        grid_size=91,
        z_nodes=7,
        mc_paths=args.mc_paths,
        seed=args.seed,
        output_dir=args.output_dir,
        calibration_configs=args.calibration_configs,
        case_ids=args.case_ids,
        stable_threshold=args.stable_threshold,
        mild_threshold=args.mild_threshold,
        high_threshold=args.high_threshold,
        frontier_source=args.frontier_source,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_paths=args.checkpoint_paths,
        device=args.device,
    )
    summary = eval_module.run_calibration(eval_args)
    output_dir = Path(args.output_dir)
    eval_module.write_outputs(summary["rows"], summary, output_dir)
    if summary.get("detail_rows"):
        detail_path = output_dir / "eval_66_cases_calibration_details.csv"
        with detail_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["detail_rows"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["detail_rows"])
    write_audit(summary, output_dir, args.frontier_source)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "detail_rows"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
