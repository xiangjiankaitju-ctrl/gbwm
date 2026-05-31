from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-phase all-or-nothing GBWM MetaRL reproduction.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mc-paths", type=int, default=10000)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--tables-dir", default=str(ROOT / "outputs" / "tables" / "mainline_paper_like"))
    parser.add_argument("--heatmaps-dir", default=str(ROOT / "outputs" / "heatmaps" / "mainline_case_20_57"))
    parser.add_argument("--skip-local-checks", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="Use only when paper-like checkpoints already exist.")
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> dict:
    print(f"\n== {name} ==")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    if completed.stdout.strip():
        print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr, file=sys.stderr)
    return {"name": name, "command": command, "returncode": completed.returncode}


def main() -> None:
    args = parse_args()
    steps: list[tuple[str, list[str]]] = []
    if not args.skip_local_checks:
        steps.extend(
            [
                ("unit tests", [str(PYTHON), "-m", "unittest", "discover", "-s", "tests"]),
                ("smoke test", [str(PYTHON), "experiments/00_smoke_test.py"]),
            ]
        )
    steps.append(("build synthetic baseline frontier", [str(PYTHON), "experiments/00_build_frontier.py", "--frontier-source", "simulated"]))
    if not args.skip_training:
        steps.append(
            (
                "paper-like PPO training",
                [
                    str(PYTHON),
                    "experiments/01_train_metarl.py",
                    "--mode",
                    "paper-like",
                    "--frontier-source",
                    "baseline",
                    "--checkpoint-dir",
                    args.checkpoint_dir,
                    "--device",
                    args.device,
                ],
            )
        )
    steps.extend(
        [
            (
                "66-case DP vs MetaRL evaluation",
                [
                    str(PYTHON),
                    "experiments/02_eval_66_cases.py",
                    "--mode",
                    "quick",
                    "--mc-paths",
                    str(args.mc_paths),
                    "--frontier-source",
                    "baseline",
                    "--checkpoint-mode",
                    "paper-like",
                    "--checkpoint-dir",
                    args.checkpoint_dir,
                    "--output-dir",
                    args.tables_dir,
                    "--device",
                    args.device,
                ],
            ),
            (
                "case 20/57 heatmap data",
                [
                    str(PYTHON),
                    "experiments/07_export_heatmap_data.py",
                    "--case-ids",
                    "20,57",
                    "--frontier-source",
                    "baseline",
                    "--checkpoint-mode",
                    "paper-like",
                    "--checkpoint-dir",
                    args.checkpoint_dir,
                    "--output-dir",
                    args.heatmaps_dir,
                    "--device",
                    args.device,
                ],
            ),
        ]
    )

    results = [run_step(name, command) for name, command in steps]
    print(json.dumps({"status": "complete", "steps": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
