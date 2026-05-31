from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import importlib.util

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.dp_baseline import solve_dp
from gbwm.efficient_frontier import (
    FrontierDataError,
    baseline_efficient_frontier,
    build_frontier_from_nav_csv,
    build_simulated_frontier_files,
    markowitz_frontier,
    simulated_efficient_frontier,
    simulated_efficient_frontier_spec,
)
from gbwm.environment import GBWMEnv, goal_decision, portfolio_decision
from gbwm.paper_cases import load_case_specs, paper_case_specs, scenario_from_case_spec
from gbwm.reproduction import (
    ReproductionGateError,
    checkpoint_filename,
    expected_checkpoint_paths,
    resolve_checkpoint_paths,
    validate_baseline_frontier_artifacts,
)
from gbwm.scenario import Scenario
from gbwm.scenario_generation import generate_training_scenario
from gbwm.state_features import build_state

EVAL_SCRIPT = ROOT / "experiments" / "02_eval_66_cases.py"
spec = importlib.util.spec_from_file_location("eval_66_cases", EVAL_SCRIPT)
eval_66_cases = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(eval_66_cases)


def simple_scenario() -> Scenario:
    mu, sigma = simulated_efficient_frontier(P=3)
    T = 3
    C = np.array([0.0, 0.0, 50.0, 80.0])
    U = np.array([0.0, 0.0, 1.0, 2.0])
    I = np.array([0.0, 0.0, 0.0, 0.0])
    return Scenario(T=T, W0=100.0, C=C, U=U, I=I, mu=mu, sigma=sigma, name="unit")


class EnvironmentTests(unittest.TestCase):
    def test_goal_decision_threshold_and_affordability(self) -> None:
        self.assertEqual(goal_decision(0.49, 100.0, 50.0), 0)
        self.assertEqual(goal_decision(0.50, 49.0, 50.0), 0)
        self.assertEqual(goal_decision(0.50, 50.0, 50.0), 1)

    def test_portfolio_decision_bounds(self) -> None:
        self.assertEqual(portfolio_decision(0.0, 5), 0)
        self.assertEqual(portfolio_decision(1.0, 5), 4)
        self.assertEqual(portfolio_decision(-1.0, 5), 0)
        self.assertEqual(portfolio_decision(2.0, 5), 4)

    def test_env_step_gbm_zero_shock(self) -> None:
        scenario = simple_scenario()
        env = GBWMEnv(scenario)
        env.reset()
        record = env.step(0.0, 0.0, 0.0)
        expected = scenario.W0 * np.exp((scenario.mu[0] - 0.5 * scenario.sigma[0] ** 2) * scenario.h)
        self.assertAlmostEqual(record.wealth_next, expected)


class StateFeatureTests(unittest.TestCase):
    def test_state_is_26_and_finite(self) -> None:
        scenario = simple_scenario()
        for t in range(1, scenario.T + 1):
            state = build_state(scenario, t, scenario.W0)
            self.assertEqual(state.shape, (26,))
            self.assertTrue(np.all(np.isfinite(state)))

    def test_state_handles_no_future_goals(self) -> None:
        mu, sigma = simulated_efficient_frontier(P=2)
        scenario = Scenario(T=2, W0=10.0, C=np.zeros(3), U=np.zeros(3), I=np.zeros(3), mu=mu, sigma=sigma)
        state = build_state(scenario, 1, scenario.W0)
        self.assertEqual(state.shape, (26,))
        self.assertTrue(np.all(np.isfinite(state)))


class ScenarioGenerationTests(unittest.TestCase):
    def test_generated_scenario_has_terminal_goal(self) -> None:
        rng = np.random.default_rng(7)
        mu, sigma = simulated_efficient_frontier()
        scenario = generate_training_scenario(mu, sigma, rng)
        self.assertGreater(scenario.C[scenario.T], 0.0)
        self.assertGreater(scenario.U[scenario.T], 0.0)
        self.assertGreaterEqual(scenario.T, 5)
        self.assertLessEqual(scenario.T, 50)


class DPTests(unittest.TestCase):
    def test_value_is_non_decreasing_in_wealth(self) -> None:
        scenario = simple_scenario()
        result = solve_dp(scenario, np.linspace(0.0, 200.0, 81))
        diffs = np.diff(result.value_grid[1])
        self.assertTrue(np.all(diffs >= -1e-9))

    def test_dp_is_reproducible(self) -> None:
        scenario = simple_scenario()
        grid = np.linspace(0.0, 200.0, 41)
        a = solve_dp(scenario, grid)
        b = solve_dp(scenario, grid)
        self.assertTrue(np.array_equal(a.goal_policy_grid, b.goal_policy_grid))
        self.assertTrue(np.array_equal(a.portfolio_policy_grid, b.portfolio_policy_grid))
        self.assertTrue(np.allclose(a.value_grid, b.value_grid))


class PaperCaseTests(unittest.TestCase):
    def test_partial_case_scaffold_loads(self) -> None:
        specs = load_case_specs(ROOT / "data" / "test_cases_66.json")
        self.assertGreaterEqual(len(specs), 1)
        scenario = scenario_from_case_spec(specs[0], frontier_source="simulated")
        self.assertGreaterEqual(scenario.T, 1)
        self.assertEqual(scenario.C.shape, (scenario.T + 1,))

    def test_generated_paper_suite_has_66_cases(self) -> None:
        specs = paper_case_specs()
        self.assertEqual(len(specs), 66)
        self.assertEqual([spec["case_id"] for spec in specs], list(range(1, 67)))
        case57 = scenario_from_case_spec(specs[56], frontier_source="simulated")
        self.assertEqual(case57.name, "case-57")
        self.assertEqual(int(np.count_nonzero(case57.C)), 60)
        self.assertEqual(int(np.count_nonzero(case57.I)), 59)


class EvaluationScriptTests(unittest.TestCase):
    def test_calibration_config_parser(self) -> None:
        self.assertEqual(eval_66_cases.parse_calibration_configs("51:5,91:7"), [(51, 5), (91, 7)])

    def test_case_id_selection(self) -> None:
        class Args:
            case_ids = "20,57"

        specs = eval_66_cases.selected_specs(Args())
        self.assertEqual([spec["case_id"] for spec in specs], [20, 57])

    def test_stability_classifier(self) -> None:
        class Args:
            stable_threshold = 0.03
            mild_threshold = 0.07
            high_threshold = 0.15

        status, passed, rel_spread, dp_below_greedy = eval_66_cases.classify_stability([10.0, 10.1, 10.2], 9.0, Args())
        self.assertEqual(status, "stable")
        self.assertTrue(passed)
        self.assertFalse(dp_below_greedy)
        self.assertLess(rel_spread, 0.03)

        status, passed, _rel_spread, dp_below_greedy = eval_66_cases.classify_stability([10.0, 10.1, 10.2], 11.0, Args())
        self.assertEqual(status, "needs_review")
        self.assertFalse(passed)
        self.assertTrue(dp_below_greedy)


class FrontierTests(unittest.TestCase):
    def test_simulated_frontier_spec_marks_debug_status(self) -> None:
        spec = simulated_efficient_frontier_spec(P=5)
        self.assertEqual(spec.numeric_status, "debug_not_paper_reproduction")
        self.assertEqual(spec.mu.shape, (5,))
        self.assertEqual(spec.sigma.shape, (5,))

    def test_baseline_frontier_missing_file_fails(self) -> None:
        with self.assertRaises(FrontierDataError):
            baseline_efficient_frontier(P=15, path=ROOT / "does_not_exist.csv")

    def test_markowitz_frontier_is_long_only_and_target_monotone(self) -> None:
        mu_assets = np.array([0.08, 0.03, 0.06])
        cov_assets = np.array(
            [
                [0.0400, 0.0020, 0.0100],
                [0.0020, 0.0100, 0.0010],
                [0.0100, 0.0010, 0.0300],
            ]
        )
        mu, sigma, weights, targets = markowitz_frontier(mu_assets, cov_assets, P=15)
        self.assertEqual(mu.shape, (15,))
        self.assertEqual(weights.shape, (15, 3))
        self.assertTrue(np.all(weights >= -1e-9))
        self.assertTrue(np.allclose(np.sum(weights, axis=1), 1.0))
        self.assertTrue(np.all(np.diff(targets) >= -1e-10))
        self.assertTrue(np.all(np.isfinite(sigma)))

    def test_build_frontier_from_nav_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "frontier_nav_monthly.csv"
            rows = ["date,VTSMX,VTBIX,VGTSX"]
            for idx in range(36):
                year = 1998 + idx // 12
                month = idx % 12 + 1
                rows.append(
                    f"{year:04d}-{month:02d}-28,{100 * (1.01 ** idx):.6f},{100 * (1.004 ** idx):.6f},{100 * (1.007 ** idx):.6f}"
                )
            raw.write_text("\n".join(rows), encoding="utf-8")
            spec = build_frontier_from_nav_csv(
                raw_csv=raw,
                P=15,
                output_csv=tmp_path / "baseline.csv",
                manifest_path=tmp_path / "manifest.json",
            )
            self.assertEqual(spec.mu.shape, (15,))
            self.assertEqual(spec.weights.shape, (15, 3))
            self.assertEqual(spec.numeric_status, "paper_reproduction_frontier")

    def test_build_simulated_frontier_files_marks_synthetic_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec = build_simulated_frontier_files(
                P=15,
                output_csv=tmp_path / "baseline_1998_2017.csv",
                manifest_path=tmp_path / "baseline_1998_2017_manifest.json",
            )
            self.assertEqual(spec.mu.shape, (15,))
            self.assertEqual(spec.numeric_status, "synthetic_baseline_frontier")
            info = validate_baseline_frontier_artifacts(
                frontier_csv=tmp_path / "baseline_1998_2017.csv",
                manifest_path=tmp_path / "baseline_1998_2017_manifest.json",
            )
            self.assertEqual(info["frontier_status"], "synthetic_baseline_frontier")


class ReproductionGateTests(unittest.TestCase):
    def test_expected_paper_like_checkpoint_set_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            for path in expected_checkpoint_paths(checkpoint_dir, mode="paper-like"):
                path.write_bytes(b"placeholder")
            paths = resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, mode="paper-like")
            self.assertEqual([path.name for path in paths], [checkpoint_filename("paper-like", seed) for seed in (0, 15, 722, 1021, 5069)])

    def test_incomplete_checkpoint_set_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            (checkpoint_dir / checkpoint_filename("paper-like", 0)).write_bytes(b"placeholder")
            with self.assertRaises(ReproductionGateError):
                resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, mode="paper-like")


class PPOGateTests(unittest.TestCase):
    def test_ppo_missing_torch_is_explicit(self) -> None:
        if importlib.util.find_spec("torch"):
            self.skipTest("torch is installed in this environment")
        from gbwm.ppo import MissingTorchDependency, PPOConfig, PPOTrainer

        mu, sigma = simulated_efficient_frontier()
        with self.assertRaises(MissingTorchDependency):
            PPOTrainer(mu, sigma, config=PPOConfig.preset("smoke"))


if __name__ == "__main__":
    unittest.main()
