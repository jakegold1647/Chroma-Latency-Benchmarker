"""
Unit tests for benchmarks/stress_test.py
"""

import pytest
from benchmarks.stress_test import IngestionStats, LoadQueryStats, StressTest


# ---------------------------------------------------------------------------
# IngestionStats — computed properties
# ---------------------------------------------------------------------------

class TestIngestionStats:
    def _stats(self, elapsed_ms=2000.0, total=1000):
        return IngestionStats(
            dimension=384,
            total_vectors=total,
            batch_size=500,
            elapsed_ms=elapsed_ms,
            peak_memory_mb=50.0,
            retained_memory_mb=20.0,
        )

    def test_throughput_vps_correct(self):
        s = self._stats(elapsed_ms=1000.0, total=5000)
        assert s.throughput_vps == pytest.approx(5000.0)

    def test_throughput_vps_zero_elapsed(self):
        s = self._stats(elapsed_ms=0.0, total=1000)
        assert s.throughput_vps == 0.0

    def test_throughput_vps_scales_with_count(self):
        slow = self._stats(elapsed_ms=2000.0, total=1000)
        fast = self._stats(elapsed_ms=500.0, total=1000)
        assert fast.throughput_vps > slow.throughput_vps

    def test_to_row_has_required_keys(self):
        row = self._stats().to_row()
        for key in ("Dimension", "Vectors", "Batch Size", "Elapsed (s)",
                    "Throughput (v/s)", "Peak RAM (MB)", "Retained RAM (MB)"):
            assert key in row

    def test_to_row_elapsed_formatted_in_seconds(self):
        s = self._stats(elapsed_ms=3500.0)
        row = s.to_row()
        assert row["Elapsed (s)"] == "3.50"

    def test_to_row_throughput_no_decimal(self):
        s = self._stats(elapsed_ms=1000.0, total=12345)
        row = s.to_row()
        assert "." not in row["Throughput (v/s)"]


# ---------------------------------------------------------------------------
# LoadQueryStats — computed properties
# ---------------------------------------------------------------------------

class TestLoadQueryStats:
    def _stats(self, latencies):
        return LoadQueryStats(
            dimension=768,
            collection_size=10000,
            concurrent_batch=1,
            latencies_ms=latencies,
        )

    def test_mean_ms_correct(self):
        s = self._stats([2.0, 4.0, 6.0])
        assert s.mean_ms == pytest.approx(4.0)

    def test_mean_ms_empty(self):
        s = self._stats([])
        assert s.mean_ms == 0.0

    def test_p95_ms_empty(self):
        s = self._stats([])
        assert s.p95_ms == 0.0

    def test_p95_ms_100_values(self):
        lats = [float(i) for i in range(1, 101)]
        s = self._stats(lats)
        assert s.p95_ms == pytest.approx(96.0)

    def test_to_row_has_required_keys(self):
        row = self._stats([1.0, 2.0]).to_row()
        for key in ("Dimension", "Coll. Size", "Query Batch",
                    "Mean (ms)", "p95 (ms)", "Total Queries"):
            assert key in row

    def test_to_row_total_queries_count(self):
        s = self._stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s.to_row()["Total Queries"] == 5


# ---------------------------------------------------------------------------
# StressTest — construction and defaults
# ---------------------------------------------------------------------------

class TestStressTestDefaults:
    def test_default_dimensions(self):
        st = StressTest()
        assert st.dimensions == [384, 768, 1536]

    def test_default_total_vectors(self):
        st = StressTest()
        assert st.total_vectors == 100_000

    def test_default_batch_size(self):
        st = StressTest()
        assert st.batch_size == 500

    def test_default_query_rounds(self):
        st = StressTest()
        assert st.query_rounds == 100

    def test_custom_dimensions(self):
        st = StressTest(dimensions=[384])
        assert st.dimensions == [384]

    def test_custom_total_vectors(self):
        st = StressTest(total_vectors=5000)
        assert st.total_vectors == 5000


# ---------------------------------------------------------------------------
# StressTest — run_ingestion_stress (small scale)
# ---------------------------------------------------------------------------

class TestStressTestIngestion:
    def test_returns_one_stat_per_dimension(self):
        st = StressTest(dimensions=[384, 768], total_vectors=200, batch_size=100, query_rounds=3)
        stats = st.run_ingestion_stress()
        assert len(stats) == 2

    def test_stat_dimensions_match_input(self):
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=50, query_rounds=2)
        stats = st.run_ingestion_stress()
        assert stats[0].dimension == 384

    def test_stat_total_vectors_matches(self):
        st = StressTest(dimensions=[384], total_vectors=150, batch_size=50, query_rounds=2)
        stats = st.run_ingestion_stress()
        assert stats[0].total_vectors == 150

    def test_elapsed_ms_positive(self):
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=100, query_rounds=2)
        stats = st.run_ingestion_stress()
        assert stats[0].elapsed_ms > 0

    def test_peak_memory_non_negative(self):
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=100, query_rounds=2)
        stats = st.run_ingestion_stress()
        assert stats[0].peak_memory_mb >= 0

    def test_throughput_positive(self):
        st = StressTest(dimensions=[384], total_vectors=200, batch_size=100, query_rounds=2)
        stats = st.run_ingestion_stress()
        assert stats[0].throughput_vps > 0


# ---------------------------------------------------------------------------
# StressTest — run_load_queries (small scale)
# ---------------------------------------------------------------------------

class TestStressTestLoadQueries:
    def test_returns_one_stat_per_dimension(self):
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=100, query_rounds=3)
        q_stats = st.run_load_queries()
        assert len(q_stats) == 1

    def test_latencies_count_matches_query_rounds(self):
        rounds = 5
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=100, query_rounds=rounds)
        q_stats = st.run_load_queries()
        assert len(q_stats[0].latencies_ms) == rounds

    def test_all_latencies_positive(self):
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=100, query_rounds=4)
        q_stats = st.run_load_queries()
        assert all(ms > 0 for ms in q_stats[0].latencies_ms)

    def test_collection_size_recorded_correctly(self):
        st = StressTest(dimensions=[384], total_vectors=200, batch_size=100, query_rounds=3)
        q_stats = st.run_load_queries()
        assert q_stats[0].collection_size == 200

    def test_print_ingestion_summary_does_not_raise(self, capsys):
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=100, query_rounds=2)
        stats = st.run_ingestion_stress()
        st.print_ingestion_summary(stats)
        out = capsys.readouterr().out
        assert "384" in out

    def test_print_query_summary_does_not_raise(self, capsys):
        st = StressTest(dimensions=[384], total_vectors=100, batch_size=100, query_rounds=2)
        q_stats = st.run_load_queries()
        st.print_query_summary(q_stats)
        out = capsys.readouterr().out
        assert "384" in out
