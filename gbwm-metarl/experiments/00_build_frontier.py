from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.efficient_frontier import (  # noqa: E402
    DEFAULT_BASELINE_FRONTIER_CSV,
    DEFAULT_BASELINE_MANIFEST,
    DEFAULT_RAW_FRONTIER_CSV,
    build_frontier_from_nav_csv,
    build_simulated_frontier_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the baseline efficient frontier for GBWM reproduction.")
    parser.add_argument("--frontier-source", choices=("csv", "simulated"), default="csv")
    parser.add_argument("--raw-csv", default=str(ROOT / DEFAULT_RAW_FRONTIER_CSV))
    parser.add_argument("--output-csv", default=str(ROOT / DEFAULT_BASELINE_FRONTIER_CSV))
    parser.add_argument("--manifest", default=str(ROOT / DEFAULT_BASELINE_MANIFEST))
    parser.add_argument("--portfolios", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frontier_source == "csv":
        spec = build_frontier_from_nav_csv(
            raw_csv=args.raw_csv,
            P=args.portfolios,
            output_csv=args.output_csv,
            manifest_path=args.manifest,
        )
    else:
        spec = build_simulated_frontier_files(
            P=args.portfolios,
            output_csv=args.output_csv,
            manifest_path=args.manifest,
        )
    print(json.dumps(spec.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
