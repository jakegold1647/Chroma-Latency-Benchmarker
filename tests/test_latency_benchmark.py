"""
Unit tests for benchmarks/latency_benchmark.py
"""

import pytest
from unittest.mock import MagicMock, patch, call
from benchmarks.latency_benchmark import (
    BenchmarkReport,
    LatencyBenchmark,
    DIMENSION_LABELS,
    SUPPORTED_DIMENSIONS,
)


# ---------------------------------------------------------------------------
# BenchmarkReport — computed properties
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def _report(self, latencies):
        return BenchmarkReport(
            dimension=384,
            collection_size=1000,
            n_queries=len(latencies),
            n_results=10,
            latencies_ms=latencies,
        )

    def test_mean_ms_correct(self):
        r = self._report([1.0, 2.0, 3.0])
        assert r.mean_ms == pytest.approx(2.0)

    def test_p50_ms_correct_odd(self):
        r = self._report([1.0, 2.0, 3.0])
        assert r.p50_ms == pytest.approx(2.0)

    def test_p50_ms_correct_even(self):
        r = self._report([1.0, 2.0, 3.0, 4.0])
        assert r.p50_ms == pytest.approx(2.5)

    def test_p95_ms_single_element(self):
        r = self._report([5.0])
        assert r.p95_ms == pytest.approx(5.0)

    def test_p95_ms_100_elements(self):
        # 100 values [1..100]; p95 index = int(100*0.95)=95 → value 96
        r = self._report([float(i) for i in range(1, 101)])
        assert r.p95_ms == pytest.approx(96.0)

    def test_p99_ms_100_elements(self):
        # p99 index = int(100*0.99)=99 → value 100
        r = self._report([float(i) for i in range(1, 101)])
        assert r.p99_ms == pytest.approx(100.0)

    def test_stdev_ms_correct(self):
        import statistics
        lats = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = self._report(lats)
        assert r.stdev_ms == pytest.approx(statistics.stdev(lats))

    def test_stdev_ms_single_value_is_zero(self):
        r = self._report([3.5])
        assert r.stdev_ms == 0.0

    def test_empty_latencies_all_zero(self):
        r = self._report([])
        assert r.mean_ms == 0.0
        assert r.p50_ms == 0.0
        assert r.p95_ms == 0.0
        assert r.p99_ms == 0.0
        assert r.stdev_ms == 0.0

    def test_to_row_keys(self):
        r = self._report([1.0, 2.0, 3.0])
        row = r.to_row()
        expected_keys = {
            "Dimension", "Model (approx.)", "Coll. Size",
            "Queries", "n_results", "Mean (ms)", "p50 (ms)",
            "p95 (ms)", "p99 (ms)", "StdDev",
        }
        assert set(row.keys()) == expected_keys

    def test_to_row_dimension_label_known(self):
        r = BenchmarkReport(384, 1000, 10, 10, [1.0])
        row = r.to_row()
        assert row["Model (approx.)"] == DIMENSION_LABELS[384]

    def test_to_row_dimension_label_unknown_falls_back(self):
        r = BenchmarkReport(999, 1000, 10, 10, [1.0])
        row = r.to_row()
        assert row["Model (approx.)"] == "999"

    def test_to_row_values_are_formatted_strings_for_floats(self):
        r = self._report([1.0, 2.0, 3.0])
        row = r.to_row()
        # Mean of 1,2,3 = 2.000
        assert row["Mean (ms)"] == "2.000"


# ---------------------------------------------------------------------------
# LatencyBenchmark — construction and defaults
# ---------------------------------------------------------------------------

class TestLatencyBenchmarkDefaults:
    def test_default_dimensions(self):
        lb = LatencyBenchmark()
        assert lb.dimensions == SUPPORTED_DIMENSIONS

    def test_default_collection_sizes(self):
        lb = LatencyBenchmark()
        assert lb.collection_sizes == [1_000, 10_000, 50_000]

    def test_default_n_queries(self):
        lb = LatencyBenchmark()
        assert lb.n_queries == 50

    def test_default_n_results(self):
        lb = LatencyBenchmark()
        assert lb.n_results == 10

    def test_custom_dimensions(self):
        lb = LatencyBenchmark(dimensions=[384])
        assert lb.dimensions == [384]

    def test_custom_collection_sizes(self):
        lb = LatencyBenchmark(collection_sizes=[100, 200])
        assert lb.collection_sizes == [100, 200]


# ---------------------------------------------------------------------------
# LatencyBenchmark — run() integration (small scale)
# ---------------------------------------------------------------------------

class TestLatencyBenchmarkRun:
    def test_run_returns_correct_number_of_reports(self):
        lb = LatencyBenchmark(
            dimensions=[384],
            collection_sizes=[100, 200],
            n_queries=3,
            n_results=5,
        )
        reports = lb.run()
        assert len(reports) == 2  # 1 dim × 2 sizes

    def test_run_reports_have_correct_dimensions(self):
        lb = LatencyBenchmark(
            dimensions=[384, 768],
            collection_sizes=[100],
            n_queries=3,
        )
        reports = lb.run()
        dims = [r.dimension for r in reports]
        assert 384 in dims
        assert 768 in dims

    def test_run_reports_have_correct_collection_sizes(self):
        lb = LatencyBenchmark(
            dimensions=[384],
            collection_sizes=[50, 150],
            n_queries=3,
        )
        reports = lb.run()
        sizes = {r.collection_size for r in reports}
        assert sizes == {50, 150}

    def test_run_latencies_count_matches_n_queries(self):
        lb = LatencyBenchmark(
            dimensions=[384],
            collection_sizes=[100],
            n_queries=7,
        )
        reports = lb.run()
        assert len(reports[0].latencies_ms) == 7

    def test_run_all_latencies_positive(self):
        lb = LatencyBenchmark(
            dimensions=[384],
            collection_sizes=[100],
            n_queries=5,
        )
        reports = lb.run()
        assert all(ms > 0 for ms in reports[0].latencies_ms)

    def test_n_results_clamped_to_collection_size(self):
        """n_results > collection_size should not raise."""
        lb = LatencyBenchmark(
            dimensions=[384],
            collection_sizes=[10],
            n_queries=3,
            n_results=100,  # larger than collection
        )
        reports = lb.run()
        assert len(reports) == 1

    def test_print_summary_does_not_raise(self, capsys):
        lb = LatencyBenchmark(dimensions=[384], collection_sizes=[50], n_queries=2)
        reports = lb.run()
        lb.print_summary(reports)
        out = capsys.readouterr().out
        assert "384" in out
