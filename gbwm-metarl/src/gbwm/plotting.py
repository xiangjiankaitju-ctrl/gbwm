"""Output helpers that avoid optional plotting dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dp_baseline import DPResult


def export_dp_grids_csv(result: DPResult, output_dir: str | Path) -> None:
    """Export DP grids as CSV files without requiring matplotlib."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "wealth_grid.csv", result.wealth_grid, delimiter=",")
    np.savetxt(output / "value_grid.csv", result.value_grid, delimiter=",")
    np.savetxt(output / "goal_policy_grid.csv", result.goal_policy_grid, delimiter=",", fmt="%d")
    np.savetxt(output / "portfolio_policy_grid.csv", result.portfolio_policy_grid, delimiter=",", fmt="%d")

