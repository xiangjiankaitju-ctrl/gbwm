"""Efficient-frontier construction and loading.

The formal reproduction path is strict: callers must either load a previously
built baseline frontier or explicitly request a simulated debug frontier.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

ASSET_COLUMNS = ("VTSMX", "VTBIX", "VGTSX")
DEFAULT_RAW_FRONTIER_CSV = Path("data/raw/frontier_nav_monthly.csv")
DEFAULT_BASELINE_FRONTIER_CSV = Path("data/frontiers/baseline_1998_2017.csv")
DEFAULT_BASELINE_MANIFEST = Path("data/frontiers/baseline_1998_2017_manifest.json")
PAPER_FRONTIER_STATUS = "paper_reproduction_frontier"
SYNTHETIC_BASELINE_STATUS = "synthetic_baseline_frontier"
DEBUG_FRONTIER_STATUS = "debug_not_paper_reproduction"


class FrontierDataError(RuntimeError):
    """Raised when the strict frontier reproduction inputs are unavailable."""


@dataclass(frozen=True)
class FrontierSpec:
    """Efficient-frontier values with provenance for result reporting."""

    name: str
    mu: np.ndarray
    sigma: np.ndarray
    source: str
    numeric_status: str
    notes: str
    weights: np.ndarray | None = None
    target_return: np.ndarray | None = None
    frontier_hash: str | None = None

    def as_dict(self) -> dict:
        payload = {
            "name": self.name,
            "source": self.source,
            "numeric_status": self.numeric_status,
            "notes": self.notes,
            "mu": [float(x) for x in self.mu],
            "sigma": [float(x) for x in self.sigma],
            "frontier_hash": self.frontier_hash,
        }
        if self.weights is not None:
            payload["weights"] = [[float(x) for x in row] for row in self.weights]
        if self.target_return is not None:
            payload["target_return"] = [float(x) for x in self.target_return]
        return payload


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _frontier_hash(rows: Iterable[dict]) -> str:
    encoded = json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_frontier_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FrontierDataError(
            f"Baseline frontier file is missing: {resolved}. "
            "Build it from data/raw/frontier_nav_monthly.csv or explicitly request a simulated frontier."
        )

    rows: list[dict] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mu", "sigma", "target_return", *(f"w_{name}" for name in ASSET_COLUMNS)}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise FrontierDataError(f"Frontier CSV missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    if not rows:
        raise FrontierDataError(f"Frontier CSV has no rows: {resolved}")

    mu = np.array([float(row["mu"]) for row in rows], dtype=float)
    sigma = np.array([float(row["sigma"]) for row in rows], dtype=float)
    target_return = np.array([float(row["target_return"]) for row in rows], dtype=float)
    weights = np.array([[float(row[f"w_{name}"]) for name in ASSET_COLUMNS] for row in rows], dtype=float)
    digest = _frontier_hash(rows)
    return mu, sigma, weights, target_return, digest


def baseline_efficient_frontier(P: int = 15, path: str | Path = DEFAULT_BASELINE_FRONTIER_CSV) -> tuple[np.ndarray, np.ndarray]:
    """Load the formal baseline frontier from disk.

    This function intentionally does not fall back to a synthetic curve.
    """

    mu, sigma, _weights, _target_return, _digest = _read_frontier_csv(path)
    if mu.shape[0] != P:
        raise FrontierDataError(f"Expected {P} frontier portfolios, found {mu.shape[0]} in {Path(path)}")
    return mu, sigma


def baseline_efficient_frontier_spec(P: int = 15, path: str | Path = DEFAULT_BASELINE_FRONTIER_CSV) -> FrontierSpec:
    mu, sigma, weights, target_return, digest = _read_frontier_csv(path)
    if mu.shape[0] != P:
        raise FrontierDataError(f"Expected {P} frontier portfolios, found {mu.shape[0]} in {Path(path)}")
    return FrontierSpec(
        name="baseline_1998_2017",
        mu=mu,
        sigma=sigma,
        weights=weights,
        target_return=target_return,
        source=str(_resolve(path)),
        numeric_status="paper_reproduction_frontier",
        notes="Loaded from a generated Markowitz frontier file; see manifest for raw data provenance.",
        frontier_hash=digest,
    )


def simulated_efficient_frontier(P: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Return an explicit deterministic debug frontier.

    This is not a paper reproduction input. It exists so local tests and smoke
    scripts can run before the user's NAV CSV is supplied.
    """

    if P < 1:
        raise ValueError("P must be positive")
    x = np.linspace(0.0, 1.0, P)
    sigma = 0.035 + 0.215 * x
    mu = 0.018 + 0.075 * (x**0.85)
    return mu.astype(float), sigma.astype(float)


def simulated_efficient_frontier_spec(P: int = 15) -> FrontierSpec:
    mu, sigma = simulated_efficient_frontier(P=P)
    weights = np.zeros((P, len(ASSET_COLUMNS)), dtype=float)
    weights[:, 0] = np.linspace(0.0, 1.0, P)
    weights[:, 1] = 1.0 - weights[:, 0]
    rows = [
        {
            "portfolio_index": i,
            "mu": float(mu[i]),
            "sigma": float(sigma[i]),
            "target_return": float(mu[i]),
            **{f"w_{name}": float(weights[i, j]) for j, name in enumerate(ASSET_COLUMNS)},
        }
        for i in range(P)
    ]
    return FrontierSpec(
        name="simulated_debug",
        mu=mu,
        sigma=sigma,
        weights=weights,
        target_return=mu.copy(),
        source="explicit deterministic simulated frontier",
        numeric_status=DEBUG_FRONTIER_STATUS,
        notes="Use only when --frontier-source simulated is explicitly selected.",
        frontier_hash=_frontier_hash(rows),
    )


def _load_monthly_nav(path: str | Path) -> tuple[list[str], np.ndarray]:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FrontierDataError(
            f"Raw NAV CSV is missing: {resolved}. Expected columns: date,{','.join(ASSET_COLUMNS)}"
        )
    dates: list[str] = []
    values: list[list[float]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", *ASSET_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise FrontierDataError(f"Raw NAV CSV missing required columns: {sorted(missing)}")
        for row in reader:
            ym = str(row["date"])[:7]
            if "1998-01" <= ym <= "2017-12":
                dates.append(str(row["date"]))
                values.append([float(row[name]) for name in ASSET_COLUMNS])
    if len(values) < 3:
        raise FrontierDataError("Need at least three monthly NAV rows between 1998-01 and 2017-12")
    order = np.argsort(np.array(dates, dtype=str))
    sorted_dates = [dates[int(i)] for i in order]
    nav = np.array(values, dtype=float)[order]
    if np.any(nav <= 0.0):
        raise FrontierDataError("NAV values must be positive")
    return sorted_dates, nav


def annualized_return_cov_from_nav(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    dates, nav = _load_monthly_nav(path)
    returns = nav[1:] / nav[:-1] - 1.0
    if not np.all(np.isfinite(returns)):
        raise FrontierDataError("Monthly returns contain non-finite values")
    annual_mu = 12.0 * np.mean(returns, axis=0)
    annual_cov = 12.0 * np.cov(returns, rowvar=False, ddof=1)
    meta = {
        "raw_path": str(_resolve(path)),
        "start_date": dates[0],
        "end_date": dates[-1],
        "monthly_return_rows": int(returns.shape[0]),
        "assets": list(ASSET_COLUMNS),
    }
    return annual_mu.astype(float), annual_cov.astype(float), meta


def _solve_min_variance(mu: np.ndarray, cov: np.ndarray, target_return: float | None = None) -> np.ndarray:
    n = int(mu.shape[0])
    best_w: np.ndarray | None = None
    best_var = float("inf")
    for mask in range(1, 1 << n):
        active = [idx for idx in range(n) if mask & (1 << idx)]
        cov_s = cov[np.ix_(active, active)]
        mu_s = mu[active]
        constraints = [np.ones(len(active), dtype=float)]
        b = [1.0]
        if target_return is not None:
            constraints.append(mu_s)
            b.append(float(target_return))
        a = np.vstack(constraints)
        b_vec = np.array(b, dtype=float)
        try:
            inv_cov = np.linalg.pinv(cov_s)
            middle = np.linalg.pinv(a @ inv_cov @ a.T)
            w_s = inv_cov @ a.T @ middle @ b_vec
        except np.linalg.LinAlgError:
            continue
        if np.any(w_s < -1e-9):
            continue
        w_s = np.maximum(w_s, 0.0)
        total = float(np.sum(w_s))
        if total <= 0.0:
            continue
        w_s = w_s / total
        if target_return is not None and abs(float(mu_s @ w_s) - float(target_return)) > 1e-7:
            continue
        w = np.zeros(n, dtype=float)
        w[active] = w_s
        var = float(w @ cov @ w)
        if var < best_var:
            best_var = var
            best_w = w
    if best_w is None:
        raise FrontierDataError(f"No long-only Markowitz solution for target_return={target_return}")
    return best_w


def markowitz_frontier(mu_assets: np.ndarray, cov_assets: np.ndarray, P: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a long-only frontier with target returns equally spaced as in the paper."""

    mu_assets = np.asarray(mu_assets, dtype=float)
    cov_assets = np.asarray(cov_assets, dtype=float)
    if mu_assets.shape != (len(ASSET_COLUMNS),) or cov_assets.shape != (len(ASSET_COLUMNS), len(ASSET_COLUMNS)):
        raise ValueError("asset mu/cov shapes must match the three baseline assets")
    if P < 1:
        raise ValueError("P must be positive")

    min_var_w = _solve_min_variance(mu_assets, cov_assets)
    start_return = float(mu_assets @ min_var_w)
    end_return = float(np.max(mu_assets))
    target_returns = np.linspace(start_return, end_return, P)
    weights = np.vstack([_solve_min_variance(mu_assets, cov_assets, float(target)) for target in target_returns])
    frontier_mu = weights @ mu_assets
    variances = np.einsum("ij,jk,ik->i", weights, cov_assets, weights)
    frontier_sigma = np.sqrt(np.maximum(variances, 0.0))
    order = np.argsort(frontier_mu)
    return frontier_mu[order], frontier_sigma[order], weights[order], target_returns[order]


def write_frontier_files(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    target_returns: np.ndarray,
    source_mode: str,
    output_csv: str | Path = DEFAULT_BASELINE_FRONTIER_CSV,
    manifest_path: str | Path = DEFAULT_BASELINE_MANIFEST,
    source_meta: dict | None = None,
) -> FrontierSpec:
    output = _resolve(output_csv)
    manifest = _resolve(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(mu.shape[0]):
        row = {
            "portfolio_index": int(idx),
            "mu": float(mu[idx]),
            "sigma": float(sigma[idx]),
            "target_return": float(target_returns[idx]),
            **{f"w_{name}": float(weights[idx, j]) for j, name in enumerate(ASSET_COLUMNS)},
        }
        rows.append(row)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["portfolio_index", "mu", "sigma", "target_return", *(f"w_{name}" for name in ASSET_COLUMNS)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _written_mu, _written_sigma, _written_weights, _written_targets, digest = _read_frontier_csv(output)
    status_by_mode = {
        "csv": PAPER_FRONTIER_STATUS,
        "simulated": SYNTHETIC_BASELINE_STATUS,
    }
    if source_mode not in status_by_mode:
        raise ValueError(f"Unsupported frontier source_mode: {source_mode}")
    manifest_payload = {
        "frontier_csv": str(output),
        "frontier_hash": digest,
        "source_mode": source_mode,
        "frontier_status": status_by_mode[source_mode],
        "numeric_status": status_by_mode[source_mode],
        "source_meta": source_meta or {},
        "assets": list(ASSET_COLUMNS),
        "portfolio_count": int(mu.shape[0]),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
    return FrontierSpec(
        name="baseline_1998_2017" if source_mode == "csv" else "synthetic_baseline_1998_2017",
        mu=mu,
        sigma=sigma,
        weights=weights,
        target_return=target_returns,
        source=str(output),
        numeric_status=manifest_payload["numeric_status"],
        notes=f"Generated from source_mode={source_mode}; see manifest for provenance.",
        frontier_hash=digest,
    )


def build_frontier_from_nav_csv(
    raw_csv: str | Path = DEFAULT_RAW_FRONTIER_CSV,
    P: int = 15,
    output_csv: str | Path = DEFAULT_BASELINE_FRONTIER_CSV,
    manifest_path: str | Path = DEFAULT_BASELINE_MANIFEST,
) -> FrontierSpec:
    mu_assets, cov_assets, meta = annualized_return_cov_from_nav(raw_csv)
    mu, sigma, weights, target_returns = markowitz_frontier(mu_assets, cov_assets, P=P)
    meta["asset_annual_mu"] = {name: float(mu_assets[i]) for i, name in enumerate(ASSET_COLUMNS)}
    meta["asset_annual_cov"] = cov_assets.tolist()
    return write_frontier_files(
        mu=mu,
        sigma=sigma,
        weights=weights,
        target_returns=target_returns,
        source_mode="csv",
        output_csv=output_csv,
        manifest_path=manifest_path,
        source_meta=meta,
    )


def build_simulated_frontier_files(
    P: int = 15,
    output_csv: str | Path = DEFAULT_BASELINE_FRONTIER_CSV,
    manifest_path: str | Path = DEFAULT_BASELINE_MANIFEST,
) -> FrontierSpec:
    spec = simulated_efficient_frontier_spec(P=P)
    if spec.weights is None or spec.target_return is None:
        raise RuntimeError("simulated frontier spec is incomplete")
    return write_frontier_files(
        mu=spec.mu,
        sigma=spec.sigma,
        weights=spec.weights,
        target_returns=spec.target_return,
        source_mode="simulated",
        output_csv=output_csv,
        manifest_path=manifest_path,
        source_meta={
            "warning": "deterministic synthetic baseline frontier; not a true NAV-based paper numeric reproduction",
            "formula": {
                "x": "i / (P - 1), i=0..P-1",
                "sigma": "0.035 + 0.215 * x",
                "mu": "0.018 + 0.075 * x**0.85",
                "target_return": "mu",
                "w_VTSMX": "x",
                "w_VTBIX": "1 - x",
                "w_VGTSX": "0",
            },
        },
    )
