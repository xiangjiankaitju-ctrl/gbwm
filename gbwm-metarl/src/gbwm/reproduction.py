"""Strict gates for the single-phase all-or-nothing reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .efficient_frontier import (
    DEFAULT_BASELINE_FRONTIER_CSV,
    DEFAULT_BASELINE_MANIFEST,
    PAPER_FRONTIER_STATUS,
    SYNTHETIC_BASELINE_STATUS,
    FrontierDataError,
    baseline_efficient_frontier_spec,
)

PPO_PRESET_SEEDS: dict[str, tuple[int, ...]] = {
    "smoke": (0,),
    "mini": (0, 15),
    "paper-like": (0, 15, 722, 1021, 5069),
}
FORMAL_CHECKPOINT_MODE = "paper-like"
FORMAL_MC_PATHS = 10000
ALLOWED_BASELINE_FRONTIER_STATUSES = {PAPER_FRONTIER_STATUS, SYNTHETIC_BASELINE_STATUS}


class ReproductionGateError(RuntimeError):
    """Raised when a requested formal reproduction artifact is incomplete."""


def checkpoint_filename(mode: str, seed: int) -> str:
    return f"metarl_{mode}_seed_{int(seed)}.pt"


def expected_checkpoint_paths(checkpoint_dir: str | Path, mode: str = FORMAL_CHECKPOINT_MODE) -> list[Path]:
    seeds = PPO_PRESET_SEEDS[mode]
    directory = Path(checkpoint_dir)
    return [directory / checkpoint_filename(mode, seed) for seed in seeds]


def resolve_checkpoint_paths(
    *,
    checkpoint_dir: str | Path,
    checkpoint_paths: str = "",
    mode: str = FORMAL_CHECKPOINT_MODE,
    require_complete: bool = True,
) -> list[Path]:
    """Resolve checkpoints and enforce the expected preset seed set by default."""

    if mode not in PPO_PRESET_SEEDS:
        raise ReproductionGateError(f"Unknown checkpoint mode: {mode}")

    if checkpoint_paths.strip():
        paths = [Path(chunk.strip()) for chunk in checkpoint_paths.split(",") if chunk.strip()]
    else:
        paths = expected_checkpoint_paths(checkpoint_dir, mode=mode)

    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise ReproductionGateError(f"Missing MetaRL checkpoint(s): {formatted}")

    if require_complete:
        expected_names = {checkpoint_filename(mode, seed) for seed in PPO_PRESET_SEEDS[mode]}
        actual_names = {path.name for path in paths}
        if actual_names != expected_names:
            raise ReproductionGateError(
                f"Checkpoint set for mode={mode!r} is incomplete or mixed. "
                f"Expected {sorted(expected_names)}, found {sorted(actual_names)}."
            )
    return paths


def validate_checkpoint_metadata(
    paths: list[str | Path],
    *,
    mode: str = FORMAL_CHECKPOINT_MODE,
    frontier_hash: str | None = None,
    device: str = "cpu",
) -> dict:
    """Load checkpoint metadata and reject mixed modes, seeds, or frontiers."""

    from .ppo import require_torch

    torch = require_torch()
    seeds: list[int] = []
    modes: set[str] = set()
    frontier_hashes: set[str | None] = set()
    for path in paths:
        payload = torch.load(Path(path), map_location=device)
        seed = int(payload.get("seed"))
        config = dict(payload.get("config", {}))
        seeds.append(seed)
        modes.add(str(config.get("mode")))
        frontier_hashes.add(payload.get("frontier_hash"))

    expected_seeds = set(PPO_PRESET_SEEDS[mode])
    if set(seeds) != expected_seeds:
        raise ReproductionGateError(f"Checkpoint seeds do not match {mode}: expected {sorted(expected_seeds)}, found {sorted(seeds)}")
    if modes != {mode}:
        raise ReproductionGateError(f"Checkpoint modes are mixed: {sorted(modes)}")
    if frontier_hash is not None and frontier_hashes != {frontier_hash}:
        raise ReproductionGateError("Checkpoint frontier hash does not match the loaded baseline frontier.")
    return {
        "checkpoint_count": len(paths),
        "mode": mode,
        "seeds": sorted(seeds),
        "frontier_hashes": sorted(str(item) for item in frontier_hashes),
    }


def validate_baseline_frontier_artifacts(
    *,
    frontier_csv: str | Path = DEFAULT_BASELINE_FRONTIER_CSV,
    manifest_path: str | Path = DEFAULT_BASELINE_MANIFEST,
    expected_portfolios: int = 15,
) -> dict:
    """Validate that formal frontier artifacts exist and are not debug data."""

    spec = baseline_efficient_frontier_spec(P=expected_portfolios, path=frontier_csv)
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FrontierDataError(f"Baseline frontier manifest is missing: {manifest_file}")
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    frontier_status = str(manifest.get("frontier_status") or manifest.get("numeric_status"))
    if frontier_status not in ALLOWED_BASELINE_FRONTIER_STATUSES:
        raise FrontierDataError(
            f"Baseline frontier manifest status must be one of {sorted(ALLOWED_BASELINE_FRONTIER_STATUSES)}; "
            f"found {frontier_status!r}"
        )
    if frontier_status == PAPER_FRONTIER_STATUS and manifest.get("source_mode") != "csv":
        raise FrontierDataError("NAV paper frontier manifest source_mode must be csv")
    if frontier_status == SYNTHETIC_BASELINE_STATUS and manifest.get("source_mode") != "simulated":
        raise FrontierDataError("Synthetic baseline frontier manifest source_mode must be simulated")
    if int(manifest.get("portfolio_count", 0)) != expected_portfolios:
        raise FrontierDataError(f"Expected {expected_portfolios} frontier portfolios in manifest")
    if manifest.get("frontier_hash") != spec.frontier_hash:
        raise FrontierDataError("Baseline frontier CSV hash does not match its manifest")
    if not np.all(np.isfinite(spec.mu)) or not np.all(np.isfinite(spec.sigma)):
        raise FrontierDataError("Baseline frontier contains non-finite mu/sigma values")
    if np.any(spec.sigma < 0.0):
        raise FrontierDataError("Baseline frontier contains negative sigma values")
    if np.any(np.diff(spec.mu) < -1e-10):
        raise FrontierDataError("Baseline frontier mu values must be non-decreasing")
    if np.any(np.diff(spec.sigma) < -1e-10):
        raise FrontierDataError("Baseline frontier sigma values must be non-decreasing")

    return {
        "frontier_csv": str(Path(frontier_csv).resolve()),
        "manifest": str(manifest_file.resolve()),
        "portfolio_count": expected_portfolios,
        "frontier_status": frontier_status,
        "numeric_status": frontier_status,
        "frontier_hash": spec.frontier_hash,
        "mu_min": float(np.min(spec.mu)),
        "mu_max": float(np.max(spec.mu)),
        "sigma_min": float(np.min(spec.sigma)),
        "sigma_max": float(np.max(spec.sigma)),
    }
