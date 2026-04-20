"""
Unit tests for benchmarks/utils.py
"""

import time
import pytest
import numpy as np

from benchmarks.utils import (
    generate_random_vectors,
    generate_metadata,
    timer,
    memory_tracker,
    format_results_table,
)


# ---------------------------------------------------------------------------
# generate_random_vectors
# ---------------------------------------------------------------------------

class TestGenerateRandomVectors:
    def test_returns_correct_count(self):
        vecs = generate_random_vectors(10, 384)
        assert len(vecs) == 10

    def test_returns_correct_dimension(self):
        vecs = generate_random_vectors(5, 768)
        assert all(len(v) == 768 for v in vecs)

    def test_vectors_are_unit_normalized(self):
        vecs = generate_random_vectors(20, 512, seed=7)
        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_with_same_seed(self):
        a = generate_random_vectors(5, 64, seed=0)
        b = generate_random_vectors(5, 64, seed=0)
        assert a == b

    def test_different_seeds_produce_different_vectors(self):
        a = generate_random_vectors(5, 64, seed=1)
        b = generate_random_vectors(5, 64, seed=2)
        assert a != b

    def test_returns_list_of_lists_of_floats(self):
        vecs = generate_random_vectors(3, 4, seed=42)
        assert isinstance(vecs, list)
        assert all(isinstance(v, list) for v in vecs)
        assert all(isinstance(x, float) for v in vecs for x in v)

    def test_single_vector(self):
        vecs = generate_random_vectors(1, 1536, seed=99)
        assert len(vecs) == 1
        assert len(vecs[0]) == 1536

    def test_all_supported_dimensions(self):
        for dim in [384, 768, 1536]:
            vecs = generate_random_vectors(2, dim)
            assert len(vecs) == 2
            assert len(vecs[0]) == dim


# ---------------------------------------------------------------------------
# generate_metadata
# ---------------------------------------------------------------------------

class TestGenerateMetadata:
    def test_returns_correct_count(self):
        meta = generate_metadata(15)
        assert len(meta) == 15

    def test_required_keys_present(self):
        meta = generate_metadata(3)
        for entry in meta:
            assert "doc_id" in entry
            assert "source" in entry
            assert "chunk" in entry
            assert "score" in entry

    def test_doc_id_is_sequential(self):
        meta = generate_metadata(5)
        assert [m["doc_id"] for m in meta] == [0, 1, 2, 3, 4]

    def test_source_wraps_at_100(self):
        meta = generate_metadata(105)
        assert meta[0]["source"] == "doc_0.txt"
        assert meta[99]["source"] == "doc_99.txt"
        assert meta[100]["source"] == "doc_0.txt"

    def test_chunk_wraps_at_20(self):
        meta = generate_metadata(25)
        assert meta[0]["chunk"] == 0
        assert meta[19]["chunk"] == 19
        assert meta[20]["chunk"] == 0

    def test_score_in_valid_range(self):
        meta = generate_metadata(100)
        for m in meta:
            assert 0.5 <= m["score"] <= 1.0

    def test_score_is_float(self):
        meta = generate_metadata(5)
        for m in meta:
            assert isinstance(m["score"], float)

    def test_empty_metadata(self):
        meta = generate_metadata(0)
        assert meta == []


# ---------------------------------------------------------------------------
# timer
# ---------------------------------------------------------------------------

class TestTimer:
    def test_elapsed_ms_key_present(self):
        with timer() as t:
            pass
        assert "elapsed_ms" in t

    def test_elapsed_is_positive(self):
        with timer() as t:
            pass
        assert t["elapsed_ms"] > 0

    def test_elapsed_reflects_sleep(self):
        with timer() as t:
            time.sleep(0.05)
        assert t["elapsed_ms"] >= 45  # allow 5ms slack

    def test_result_populated_after_block(self):
        result = {}
        with timer() as t:
            result["during"] = "elapsed_ms" not in t
        assert result["during"] is True
        assert "elapsed_ms" in t


# ---------------------------------------------------------------------------
# memory_tracker
# ---------------------------------------------------------------------------

class TestMemoryTracker:
    def test_peak_mb_key_present(self):
        with memory_tracker() as m:
            _ = list(range(1000))
        assert "peak_mb" in m

    def test_peak_mb_is_non_negative(self):
        with memory_tracker() as m:
            pass
        assert m["peak_mb"] >= 0

    def test_large_allocation_increases_peak(self):
        with memory_tracker() as m_small:
            _ = list(range(10))
        with memory_tracker() as m_large:
            _ = list(range(500_000))
        assert m_large["peak_mb"] > m_small["peak_mb"]


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------

class TestFormatResultsTable:
    def test_empty_rows_returns_placeholder(self):
        assert format_results_table([]) == "(no results)"

    def test_contains_all_headers(self):
        rows = [{"Name": "Alice", "Score": 99}]
        out = format_results_table(rows)
        assert "Name" in out
        assert "Score" in out

    def test_contains_all_values(self):
        rows = [{"Dim": 384, "ms": "1.234"}]
        out = format_results_table(rows)
        assert "384" in out
        assert "1.234" in out

    def test_title_appears_when_provided(self):
        rows = [{"A": "x"}]
        out = format_results_table(rows, title="My Table")
        assert "My Table" in out

    def test_no_title_section_when_omitted(self):
        rows = [{"A": "x"}]
        out = format_results_table(rows)
        assert "====" not in out

    def test_multiple_rows(self):
        rows = [{"dim": 384}, {"dim": 768}, {"dim": 1536}]
        out = format_results_table(rows)
        assert "384" in out
        assert "768" in out
        assert "1536" in out

    def test_column_widths_accommodate_longest_value(self):
        rows = [{"key": "short"}, {"key": "a_very_long_value_here"}]
        out = format_results_table(rows)
        assert "a_very_long_value_here" in out

    def test_table_has_border_characters(self):
        rows = [{"X": 1}]
        out = format_results_table(rows)
        assert "+" in out
        assert "|" in out
        assert "-" in out
