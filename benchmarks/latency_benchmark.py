"""
Latency Benchmark
=================
Measures ChromaDB query latency across vector dimensions that correspond
to common LLM embedding models:

    384  — MiniLM / all-MiniLM-L6-v2
    768  — BGE-base, E5-base, OpenAI ada-001
    1536 — OpenAI text-embedding-ada-002, text-embedding-3-small
"""

import uuid
import statistics
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb

from .utils import generate_random_vectors, generate_metadata, timer, format_results_table

logger = logging.getLogger(__name__)

SUPPORTED_DIMENSIONS = [384, 768, 1536]
DIMENSION_LABELS = {
    384: "MiniLM-L6 (384d)",
    768: "BGE-base / E5-base (768d)",
    1536: "text-embedding-ada-002 (1536d)",
}


@dataclass
class QueryResult:
    dimension: int
    n_results: int
    latency_ms: float
    collection_size: int


@dataclass
class BenchmarkReport:
    dimension: int
    collection_size: int
    n_queries: int
    n_results: int
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    def to_row(self) -> dict:
        return {
            "Dimension": self.dimension,
            "Model (approx.)": DIMENSION_LABELS.get(self.dimension, str(self.dimension)),
            "Coll. Size": self.collection_size,
            "Queries": self.n_queries,
            "n_results": self.n_results,
            "Mean (ms)": f"{self.mean_ms:.3f}",
            "p50 (ms)": f"{self.p50_ms:.3f}",
            "p95 (ms)": f"{self.p95_ms:.3f}",
            "p99 (ms)": f"{self.p99_ms:.3f}",
            "StdDev": f"{self.stdev_ms:.3f}",
        }


class LatencyBenchmark:
    """
    Benchmarks ChromaDB retrieval latency for multiple vector dimensions.

    Parameters
    ----------
    collection_sizes : list of int
        Number of documents to pre-load into each test collection.
    n_queries : int
        Number of query vectors to run per (dimension, collection_size) pair.
    n_results : int
        Number of nearest neighbours to retrieve per query.
    dimensions : list of int
        Embedding dimensions to test.
    """

    def __init__(
        self,
        collection_sizes: Optional[List[int]] = None,
        n_queries: int = 50,
        n_results: int = 10,
        dimensions: Optional[List[int]] = None,
    ):
        self.collection_sizes = collection_sizes or [1_000, 10_000, 50_000]
        self.n_queries = n_queries
        self.n_results = n_results
        self.dimensions = dimensions or SUPPORTED_DIMENSIONS
        self._client = chromadb.EphemeralClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[BenchmarkReport]:
        """Run the full benchmark matrix and return all reports."""
        reports: List[BenchmarkReport] = []

        for dim in self.dimensions:
            for size in self.collection_sizes:
                report = self._benchmark_one(dim, size)
                reports.append(report)
                logger.info(
                    "dim=%d  size=%d  mean=%.3fms  p95=%.3fms",
                    dim,
                    size,
                    report.mean_ms,
                    report.p95_ms,
                )

        return reports

    def print_summary(self, reports: List[BenchmarkReport]) -> None:
        rows = [r.to_row() for r in reports]
        print(format_results_table(rows, title="ChromaDB Latency Benchmark Results"))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _benchmark_one(self, dimension: int, collection_size: int) -> BenchmarkReport:
        collection_name = f"bench_{dimension}d_{collection_size}_{uuid.uuid4().hex[:8]}"
        collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._populate_collection(collection, collection_size, dimension)

        query_vectors = generate_random_vectors(self.n_queries, dimension, seed=99)
        latencies: List[float] = []

        for qv in query_vectors:
            with timer() as t:
                collection.query(
                    query_embeddings=[qv],
                    n_results=min(self.n_results, collection_size),
                )
            latencies.append(t["elapsed_ms"])

        self._client.delete_collection(collection_name)

        return BenchmarkReport(
            dimension=dimension,
            collection_size=collection_size,
            n_queries=self.n_queries,
            n_results=self.n_results,
            latencies_ms=latencies,
        )

    def _populate_collection(self, collection, size: int, dimension: int) -> None:
        batch_size = 1000
        for offset in range(0, size, batch_size):
            batch = min(batch_size, size - offset)
            vectors = generate_random_vectors(batch, dimension, seed=offset)
            ids = [str(offset + i) for i in range(batch)]
            metadata = generate_metadata(batch)
            documents = [f"Document chunk {offset + i}" for i in range(batch)]
            collection.add(
                embeddings=vectors,
                ids=ids,
                metadatas=metadata,
                documents=documents,
            )
