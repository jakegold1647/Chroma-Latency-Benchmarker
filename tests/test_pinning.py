"""
Pinning / regression tests
==========================
These tests lock down deterministic outputs. If any of these fail after
a code change, the behaviour of a core function has shifted — review
intentionally before updating the pinned values.
"""

import pytest
import numpy as np
from benchmarks.utils import generate_random_vectors, format_results_table
from benchmarks.latency_benchmark import BenchmarkReport, DIMENSION_LABELS, SUPPORTED_DIMENSIONS
from benchmarks.stress_test import IngestionStats, LoadQueryStats


# ---------------------------------------------------------------------------
# PIN: generate_random_vectors output with fixed seed
# ---------------------------------------------------------------------------

class TestPinnedVectorOutput:
    """
    Values computed once and frozen. Seed=42, count=3, dim=4.
    If numpy or the generation logic changes these will fail.
    """

    PINNED = [
        [0.188174, -0.642228, 0.463431, 0.580833],
        [-0.8231,  -0.549362, 0.053933, -0.133416],
        [-0.011577, -0.58778, 0.605939, 0.535928],
    ]

    def test_first_vector_matches_pinned(self):
        vecs = generate_random_vectors(3, 4, seed=42)
        np.testing.assert_allclose(vecs[0], self.PINNED[0], atol=1e-4)

    def test_second_vector_matches_pinned(self):
        vecs = generate_random_vectors(3, 4, seed=42)
        np.testing.assert_allclose(vecs[1], self.PINNED[1], atol=1e-4)

    def test_third_vector_matches_pinned(self):
        vecs = generate_random_vectors(3, 4, seed=42)
        np.testing.assert_allclose(vecs[2], self.PINNED[2], atol=1e-4)

    def test_all_rows_unit_norm(self):
        vecs = generate_random_vectors(3, 4, seed=42)
        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_384d_seed42_first_vector_is_unit(self):
        vecs = generate_random_vectors(1, 384, seed=42)
        norm = np.linalg.norm(vecs[0])
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_768d_seed99_first_vector_is_unit(self):
        vecs = generate_random_vectors(1, 768, seed=99)
        norm = np.linalg.norm(vecs[0])
        assert norm == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# PIN: DIMENSION_LABELS mapping
# ---------------------------------------------------------------------------

class TestPinnedDimensionLabels:
    def test_384_label(self):
        assert DIMENSION_LABELS[384] == "MiniLM-L6 (384d)"

    def test_768_label(self):
        assert DIMENSION_LABELS[768] == "BGE-base / E5-base (768d)"

    def test_1536_label(self):
        assert DIMENSION_LABELS[1536] == "text-embedding-ada-002 (1536d)"

    def test_supported_dimensions_unchanged(self):
        assert SUPPORTED_DIMENSIONS == [384, 768, 1536]


# ---------------------------------------------------------------------------
# PIN: BenchmarkReport.to_row() format strings
# ---------------------------------------------------------------------------

class TestPinnedBenchmarkReportRow:
    def test_mean_formatted_to_3dp(self):
        r = BenchmarkReport(384, 1000, 3, 10, [1.0, 2.0, 3.0])
        assert r.to_row()["Mean (ms)"] == "2.000"

    def test_p50_formatted_to_3dp(self):
        r = BenchmarkReport(384, 1000, 3, 10, [1.0, 2.0, 3.0])
        assert r.to_row()["p50 (ms)"] == "2.000"

    def test_stdev_formatted_to_3dp(self):
        import statistics
        lats = [1.0, 2.0, 3.0]
        r = BenchmarkReport(384, 1000, 3, 10, lats)
        expected = f"{statistics.stdev(lats):.3f}"
        assert r.to_row()["StdDev"] == expected

    def test_dimension_stored_as_int(self):
        r = BenchmarkReport(768, 1000, 2, 5, [1.0, 1.5])
        assert r.to_row()["Dimension"] == 768

    def test_coll_size_stored_as_int(self):
        r = BenchmarkReport(384, 5000, 2, 5, [1.0, 1.5])
        assert r.to_row()["Coll. Size"] == 5000


# ---------------------------------------------------------------------------
# PIN: IngestionStats.to_row() format strings
# ---------------------------------------------------------------------------

class TestPinnedIngestionStatsRow:
    def test_elapsed_formatted_seconds_2dp(self):
        s = IngestionStats(384, 1000, 500, 3750.0, 10.0, 5.0)
        assert s.to_row()["Elapsed (s)"] == "3.75"

    def test_peak_ram_formatted_1dp(self):
        s = IngestionStats(384, 1000, 500, 1000.0, 123.456, 50.0)
        assert s.to_row()["Peak RAM (MB)"] == "123.5"

    def test_retained_ram_formatted_1dp(self):
        s = IngestionStats(384, 1000, 500, 1000.0, 100.0, 67.89)
        assert s.to_row()["Retained RAM (MB)"] == "67.9"

    def test_throughput_formatted_no_decimal(self):
        s = IngestionStats(384, 10000, 500, 2000.0, 50.0, 20.0)
        # 10000 / 2.0s = 5000 v/s
        row = s.to_row()
        assert row["Throughput (v/s)"] == "5000"


# ---------------------------------------------------------------------------
# PIN: format_results_table structural invariants
# ---------------------------------------------------------------------------

class TestPinnedTableFormat:
    ROWS = [{"Dim": 384, "ms": "1.234"}, {"Dim": 768, "ms": "2.345"}]

    def test_separator_line_starts_and_ends_with_plus(self):
        out = format_results_table(self.ROWS)
        sep_lines = [ln for ln in out.splitlines() if ln.startswith("+")]
        assert all(ln.endswith("+") for ln in sep_lines)

    def test_data_lines_start_and_end_with_pipe(self):
        out = format_results_table(self.ROWS)
        data_lines = [ln for ln in out.splitlines() if ln.startswith("|")]
        assert all(ln.endswith("|") for ln in data_lines)

    def test_column_count_consistent_across_rows(self):
        out = format_results_table(self.ROWS)
        pipe_counts = [ln.count("|") for ln in out.splitlines() if ln.startswith("|")]
        assert len(set(pipe_counts)) == 1  # all rows have same pipe count

    def test_title_separator_length_matches_table_width(self):
        rows = [{"X": "val"}]
        out = format_results_table(rows, title="T")
        lines = out.splitlines()
        eq_lines = [ln for ln in lines if ln.startswith("=")]
        plus_lines = [ln for ln in lines if ln.startswith("+")]
        assert len(eq_lines[0]) == len(plus_lines[0])
