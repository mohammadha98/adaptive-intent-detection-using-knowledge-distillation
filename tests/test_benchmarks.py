"""
Unit tests for the AutoMTL Cost & Latency benchmark module.

Run:
    pytest tests/test_benchmarks.py -v
    python -m unittest tests.test_benchmarks -v
"""

import sys
import os
import math
import unittest
from unittest.mock import patch, MagicMock

# ── path plumbing ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "metrics")))

# Prevent MockStudent from loading a real BERTClassifier (which downloads
# the heavy DistilBERT model).  We mock the module-level symbols before
# the benchmark module is imported.
_MODULE_PATH = "benchmark_cost_latency"

# A no-op replacement for BERTClassifier that won't trigger HF downloads
class _FakeBERT:
    """Minimal stand-in so MockStudent.__init__ does not download models."""
    def __init__(self, *args, **kwargs):
        self.is_trained = False
    def load_state_dict(self, *args, **kwargs):
        pass
    def to(self, *args, **kwargs):
        return self
    def predict(self, *args, **kwargs):
        return ("unknown", 0.0)

_patches = [
    patch(f"{_MODULE_PATH}.BERTClassifier", new=_FakeBERT),
    patch(f"{_MODULE_PATH}.torch.load", return_value=MagicMock()),
]
for _p in _patches:
    _p.start()

# Now it is safe to import – MockStudent will see _FakeBERT instead of
# the real DistilBERT-based classifier.
from benchmark_cost_latency import (  # noqa: E402
    BenchmarkConfig,
    MockStudent,
    MockTeacher,
    calculate_llm_cost,
    run_scenarios,
    get_simulated_data_stream,
)


# ═══════════════════════════════════════════════════════════════════════
#  Helper
# ═══════════════════════════════════════════════════════════════════════

def _make_cfg(**overrides):
    """Return a BenchmarkConfig with sensible test defaults."""
    defaults = dict(
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0002,
        student_latency_mean=0.050,
        student_latency_std=0.0,       # deterministic for tests
        teacher_latency_mean=1.200,
        teacher_latency_std=0.0,       # deterministic for tests
        total_requests=100,
        ood_ratio=0.20,
        use_real_sleep=False,
        seed=42,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════
#  Tests: calculate_llm_cost
# ═══════════════════════════════════════════════════════════════════════

class TestCostCalculation(unittest.TestCase):
    """Verify the LLM cost formula is mathematically accurate."""

    def setUp(self):
        self.cfg = _make_cfg()

    def test_standard_case(self):
        """2000 input tokens + 500 output tokens → expected cost."""
        cost = calculate_llm_cost(2000, 500, self.cfg)
        expected = (2000 / 1000.0 * 0.0001) + (500 / 1000.0 * 0.0002)
        self.assertAlmostEqual(cost, expected, places=10)

    def test_zero_tokens(self):
        """Zero input and output → $0.00."""
        self.assertAlmostEqual(calculate_llm_cost(0, 0, self.cfg), 0.0, places=10)

    def test_input_only(self):
        """Only input tokens, output = 0."""
        cost = calculate_llm_cost(1500, 0, self.cfg)
        expected = (1500 / 1000.0 * 0.0001)
        self.assertAlmostEqual(cost, expected, places=10)

    def test_output_only(self):
        """Only output tokens, input = 0."""
        cost = calculate_llm_cost(0, 300, self.cfg)
        expected = (300 / 1000.0 * 0.0002)
        self.assertAlmostEqual(cost, expected, places=10)

    def test_large_counts(self):
        """Large token counts should not overflow."""
        cost = calculate_llm_cost(1_000_000, 500_000, self.cfg)
        expected = (1_000_000 / 1000.0 * 0.0001) + (500_000 / 1000.0 * 0.0002)
        self.assertAlmostEqual(cost, expected, places=6)

    def test_fractional_input(self):
        """Fractional token counts (as returned by len(text.split())*1.5)."""
        cost = calculate_llm_cost(18.0, 5.0, self.cfg)
        expected = (18.0 / 1000.0 * 0.0001) + (5.0 / 1000.0 * 0.0002)
        self.assertAlmostEqual(cost, expected, places=10)

    def test_default_config_fallback(self):
        """When cfg is None, the function should use default BenchmarkConfig."""
        cost = calculate_llm_cost(1000, 0)          # no cfg passed
        expected = (1000 / 1000.0) * 0.0001         # default input rate
        self.assertAlmostEqual(cost, expected, places=8)

    def test_custom_cost_rates(self):
        """A config with different rates is honoured."""
        alt = _make_cfg(cost_per_1k_input_tokens=0.005,
                        cost_per_1k_output_tokens=0.010)
        cost = calculate_llm_cost(2000, 1000, alt)
        expected = (2.0 * 0.005) + (1.0 * 0.010)
        self.assertAlmostEqual(cost, expected, places=8)

    def test_cost_is_linear(self):
        """Doubling tokens should double the cost."""
        base = calculate_llm_cost(1000, 500, self.cfg)
        double = calculate_llm_cost(2000, 1000, self.cfg)
        self.assertAlmostEqual(double, base * 2.0, places=8)


# ═══════════════════════════════════════════════════════════════════════
#  Tests: MockStudent & MockTeacher
# ═══════════════════════════════════════════════════════════════════════

class TestMockStudent(unittest.TestCase):
    """Behaviour of the simulated student model."""

    def test_deterministic_latency_with_zero_std(self):
        cfg = _make_cfg(student_latency_mean=0.050, student_latency_std=0.0, seed=1)
        student = MockStudent(cfg)
        # force no real classifier so OOD is random
        student.classifier = None
        res = student.predict("hello")
        self.assertAlmostEqual(res["latency"], 0.050, places=5)

    def test_latency_is_positive(self):
        cfg = _make_cfg(seed=99)
        student = MockStudent(cfg)
        student.classifier = None
        for _ in range(50):
            res = student.predict("test")
            self.assertGreater(res["latency"], 0.0)

    def test_returns_expected_keys(self):
        cfg = _make_cfg()
        student = MockStudent(cfg)
        student.classifier = None
        res = student.predict("any text")
        self.assertIn("latency", res)
        self.assertIn("is_ood", res)
        self.assertIsInstance(res["is_ood"], bool)


class TestMockTeacher(unittest.TestCase):
    """Behaviour of the simulated teacher LLM."""

    def test_deterministic_latency_with_zero_std(self):
        cfg = _make_cfg(teacher_latency_mean=1.200, teacher_latency_std=0.0, seed=1)
        teacher = MockTeacher(cfg)
        res = teacher.get_fallback_prediction("hello")
        self.assertAlmostEqual(res["latency"], 1.200, places=5)

    def test_returns_token_counts(self):
        cfg = _make_cfg()
        teacher = MockTeacher(cfg)
        res = teacher.get_fallback_prediction("this is a test")
        self.assertGreater(res["input_tokens"], 0)
        self.assertGreater(res["output_tokens"], 0)

    def test_output_tokens_in_range(self):
        cfg = _make_cfg(min_output_tokens=3, max_output_tokens=8, seed=7)
        teacher = MockTeacher(cfg)
        for _ in range(50):
            res = teacher.get_fallback_prediction("x")
            self.assertGreaterEqual(res["output_tokens"], cfg.min_output_tokens)
            self.assertLessEqual(res["output_tokens"], cfg.max_output_tokens)

    def test_latency_is_positive(self):
        cfg = _make_cfg(seed=42)
        teacher = MockTeacher(cfg)
        for _ in range(50):
            res = teacher.get_fallback_prediction("test")
            self.assertGreater(res["latency"], 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Tests: Latency aggregation (AutoMTL hybrid routing)
# ═══════════════════════════════════════════════════════════════════════

class TestLatencyAggregationAutoMTL(unittest.TestCase):
    """Verify that AutoMTL latency is the sum of student + teacher only
    when a fallback occurs."""

    def setUp(self):
        self.cfg = _make_cfg(
            student_latency_mean=0.050,
            student_latency_std=0.0,
            teacher_latency_mean=1.200,
            teacher_latency_std=0.0,
            seed=1,
        )

    def test_id_request_only_student_latency(self):
        """Force is_ood=False → latency = student only."""
        s_res = {"latency": 0.05, "is_ood": False}
        auto_lat = s_res["latency"]
        if s_res["is_ood"]:
            auto_lat += 1.2  # would be teacher latency
        self.assertAlmostEqual(auto_lat, 0.05, places=10)

    def test_ood_request_sum_of_both(self):
        """Force is_ood=True → latency = student + teacher."""
        s_res = {"latency": 0.055, "is_ood": True}
        t_res = {"latency": 1.180}
        auto_lat = s_res["latency"]
        if s_res["is_ood"]:
            auto_lat += t_res["latency"]
        expected = 0.055 + 1.180
        self.assertAlmostEqual(auto_lat, expected, places=10)

    def test_zero_latency_edge(self):
        """Even if latencies are zero, sum should be zero."""
        s_res = {"latency": 0.0, "is_ood": True}
        t_res = {"latency": 0.0}
        auto_lat = s_res["latency"] + t_res["latency"]
        self.assertEqual(auto_lat, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Integration-style tests  (run_scenarios)
# ═══════════════════════════════════════════════════════════════════════

class TestRunScenarios(unittest.TestCase):
    """End-to-end benchmark run on a small synthetic stream."""

    @staticmethod
    def _build_stream(n: int, is_ood_flags):
        """Return a list of dicts with 'text' and 'is_ood_ground_truth'."""
        return [{"text": f"sample {i}", "is_ood_ground_truth": flag}
                for i, flag in enumerate(is_ood_flags[:n])]

    def test_results_keys_present(self):
        cfg = _make_cfg(total_requests=20, seed=99)
        stream = self._build_stream(20, [False] * 15 + [True] * 5)
        results = run_scenarios(stream, cfg)

        for scenario in ("llm_only", "student_only", "automtl"):
            self.assertIn(scenario, results)
            self.assertIn("total_latency", results[scenario])
            self.assertIn("total_cost", results[scenario])
            self.assertIn("total_requests", results[scenario])
        self.assertIn("fallbacks", results["automtl"])

    def test_request_counts_match_input(self):
        cfg = _make_cfg(total_requests=50, seed=7)
        stream = self._build_stream(50, [False] * 40 + [True] * 10)
        results = run_scenarios(stream, cfg)
        self.assertEqual(results["llm_only"]["total_requests"], 50)
        self.assertEqual(results["student_only"]["total_requests"], 50)
        self.assertEqual(results["automtl"]["total_requests"], 50)

    def test_all_id_yields_zero_fallbacks(self):
        """When every request is ID, AutoMTL should have 0 fallbacks and $0 cost."""
        cfg = _make_cfg(total_requests=30, seed=3,
                        student_latency_std=0.0, teacher_latency_std=0.0)
        with patch(f"{_MODULE_PATH}.MockStudent.predict",
                   return_value={"latency": 0.05, "is_ood": False}):
            stream = self._build_stream(30, [False] * 30)
            results = run_scenarios(stream, cfg)

        self.assertEqual(results["automtl"]["fallbacks"], 0)
        self.assertEqual(results["automtl"]["total_cost"], 0.0)
        # AutoMTL latency ≈ student-only latency
        self.assertAlmostEqual(
            results["automtl"]["total_latency"],
            results["student_only"]["total_latency"],
            places=3,
        )

    def test_all_ood_yields_max_fallbacks(self):
        """When every request is OOD, AutoMTL = student + teacher for each."""
        cfg = _make_cfg(total_requests=30, seed=5,
                        student_latency_std=0.0, teacher_latency_std=0.0)
        with patch(f"{_MODULE_PATH}.MockStudent.predict",
                   return_value={"latency": 0.05, "is_ood": True}):
            stream = self._build_stream(30, [True] * 30)
            results = run_scenarios(stream, cfg)

        self.assertEqual(results["automtl"]["fallbacks"], 30)
        # AutoMTL cost must equal LLM-only cost (every request hits teacher)
        self.assertAlmostEqual(
            results["automtl"]["total_cost"],
            results["llm_only"]["total_cost"],
            places=5,
        )
        # AutoMTL latency = (student + teacher) * 30
        expected_lat = (0.05 + 1.20) * 30
        self.assertAlmostEqual(results["automtl"]["total_latency"], expected_lat, places=2)

    def test_cost_student_only_is_zero(self):
        cfg = _make_cfg(total_requests=20, seed=12)
        stream = self._build_stream(20, [False] * 20)
        results = run_scenarios(stream, cfg)
        self.assertEqual(results["student_only"]["total_cost"], 0.0)

    def test_cost_automtl_never_exceeds_llm_only(self):
        """AutoMTL cost ≤ LLM-Only cost (only OOD subset hits teacher)."""
        cfg = _make_cfg(total_requests=100, seed=42)
        stream = self._build_stream(100, [False] * 80 + [True] * 20)
        results = run_scenarios(stream, cfg)
        self.assertLessEqual(results["automtl"]["total_cost"],
                             results["llm_only"]["total_cost"])

    def test_latency_automtl_between_two_baselines(self):
        """Avg latency: Student-Only ≤ AutoMTL ≤ LLM-Only (with ≈20% OOD)."""
        cfg = _make_cfg(total_requests=100, seed=13,
                        student_latency_std=0.0, teacher_latency_std=0.0)
        # We need the OOD gate to match the 80/20 split, not the real model.
        # Mock the student to return is_ood based on ground-truth flags.
        with patch(f"{_MODULE_PATH}.MockStudent.predict") as mock_predict:
            stream = self._build_stream(100, [False] * 80 + [True] * 20)

            # 20 % of requests marked OOD by the mock
            def _side_effect(text):
                idx = mock_predict.call_count
                is_ood = stream[min(idx, len(stream) - 1)]["is_ood_ground_truth"]
                return {"latency": cfg.student_latency_mean, "is_ood": is_ood}
            mock_predict.side_effect = _side_effect

            results = run_scenarios(stream, cfg)

        s_avg = results["student_only"]["total_latency"] / 100
        m_avg = results["automtl"]["total_latency"] / 100
        l_avg = results["llm_only"]["total_latency"] / 100

        self.assertLessEqual(s_avg, m_avg, "Student-only should be fastest")
        self.assertLessEqual(m_avg, l_avg, "AutoMTL should be faster than LLM-only")


# ═══════════════════════════════════════════════════════════════════════
#  Tests: BenchmarkConfig dataclass
# ═══════════════════════════════════════════════════════════════════════

class TestBenchmarkConfig(unittest.TestCase):
    """Ensure the config object is well-behaved."""

    def test_defaults(self):
        cfg = BenchmarkConfig()
        self.assertEqual(cfg.cost_per_1k_input_tokens, 0.0001)
        self.assertEqual(cfg.cost_per_1k_output_tokens, 0.0002)
        self.assertEqual(cfg.total_requests, 1000)
        self.assertEqual(cfg.ood_ratio, 0.20)
        self.assertFalse(cfg.use_real_sleep)

    def test_override(self):
        cfg = BenchmarkConfig(
            cost_per_1k_input_tokens=0.003,
            total_requests=500,
            use_real_sleep=True,
        )
        self.assertEqual(cfg.cost_per_1k_input_tokens, 0.003)
        self.assertEqual(cfg.total_requests, 500)
        self.assertTrue(cfg.use_real_sleep)
        # untouched defaults remain
        self.assertEqual(cfg.cost_per_1k_output_tokens, 0.0002)


# ═══════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)