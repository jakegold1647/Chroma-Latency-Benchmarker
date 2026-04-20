import numpy as np
import time
import tracemalloc
from typing import List, Dict, Any
from contextlib import contextmanager


def generate_random_vectors(count: int, dimensions: int, seed: int = 42) -> List[List[float]]:
    """Generate normalized random vectors to simulate real embedding distributions."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dimensions)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors.tolist()


def generate_metadata(count: int) -> List[Dict[str, Any]]:
    """Generate synthetic metadata payloads."""
    return [
        {
            "doc_id": i,
            "source": f"doc_{i % 100}.txt",
            "chunk": i % 20,
            "score": float(np.random.uniform(0.5, 1.0)),
        }
        for i in range(count)
    ]


@contextmanager
def timer():
    """Context manager that yields elapsed time in milliseconds."""
    start = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000


@contextmanager
def memory_tracker():
    """Context manager that tracks peak memory usage in MB."""
    tracemalloc.start()
    result = {}
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_mb"] = peak / 1024 / 1024


def format_results_table(rows: List[Dict[str, Any]], title: str = "") -> str:
    """Render a list of dicts as a fixed-width ASCII table."""
    if not rows:
        return "(no results)"

    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    sep = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"
    header_row = "|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|"

    lines = []
    if title:
        lines.append(f"\n{'=' * len(sep)}")
        lines.append(f" {title}")
        lines.append("=" * len(sep))
    lines.append(sep)
    lines.append(header_row)
    lines.append(sep)
    for row in rows:
        lines.append("|" + "|".join(f" {str(row[h]):<{col_widths[h]}} " for h in headers) + "|")
    lines.append(sep)
    return "\n".join(lines)
