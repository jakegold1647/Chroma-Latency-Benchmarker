"""
Stress Test Module
==================
Simulates high-density data ingestion into ChromaDB and measures:

  - Ingestion throughput (vectors/second)
  - Peak memory overhead (MB) during ingestion
  - Memory retained after ingestion (resident collection footprint)
  - Query latency degradation under load

Designed to surface bottlenecks relevant to GPU-heavy inference pipelines
where embedding generation is fast but the vector store may become the
throughput ceiling.
"""

import gc
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb

from .utils import (
    generate_random_vectors,
    generate_metadata,
    timer,
    memory_tracker,
    format_results_table,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    dimension: int
    total_vectors: int
    batch_size: int
    elapsed_ms: float
    peak_memory_mb: float
    retained_memory_mb: float

    @property
    def throughput_vps(self) -> float:
        """Vectors per second."""
        return self.total_vectors / (self.elapsed_ms / 1000) if self.elapsed_ms > 0 else 0.0

    def to_row(self) -> dict:
        return {
            "Dimension": self.dimension,
            "Vectors": self.total_vectors,
            "Batch Size": self.batch_size,
            "Elapsed (s)": f"{self.elapsed_ms / 1000:.2f}",
            "Throughput (v/s)": f"{self.throughput_vps:.0f}",
            "Peak RAM (MB)": f"{self.peak_memory_mb:.1f}",
            "Retained RAM (MB)": f"{self.retained_memory_mb:.1f}",
        }


@dataclass
class LoadQueryStats:
    dimension: int
    collection_size: int
    concurrent_batch: int
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        return sorted_lat[int(len(sorted_lat) * 0.95)]

    def to_row(self) -> dict:
        return {
            "Dimension": self.dimension,
            "Coll. Size": self.collection_size,
            "Query Batch": self.concurrent_batch,
            "Mean (ms)": f"{self.mean_ms:.3f}",
            "p95 (ms)": f"{self.p95_ms:.3f}",
            "Total Queries": len(self.latencies_ms),
        }


class StressTest:
    """
    High-density ingestion and load-query stress test for ChromaDB.

    Parameters
    ----------
    dimensions : list of int
        Embedding dimensions to stress-test.
    total_vectors : int
        Total number of vectors to ingest per dimension.
    batch_size : int
        Vectors per add() call (simulates streaming ingestion).
    query_rounds : int
        Number of query rounds to run after ingestion completes.
    n_results : int
        k-NN neighbours to retrieve per query.
    """

    def __init__(
        self,
        dimensions: Optional[List[int]] = None,
        total_vectors: int = 100_000,
        batch_size: int = 500,
        query_rounds: int = 100,
        n_results: int = 10,
    ):
        self.dimensions = dimensions or [384, 768, 1536]
        self.total_vectors = total_vectors
        self.batch_size = batch_size
        self.query_rounds = query_rounds
        self.n_results = n_results
        self._client = chromadb.EphemeralClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_ingestion_stress(self) -> List[IngestionStats]:
        """Ingest `total_vectors` vectors per dimension and report overhead."""
        results: List[IngestionStats] = []

        for dim in self.dimensions:
            logger.info("Ingestion stress: dim=%d  total=%d", dim, self.total_vectors)
            stats = self._ingest_with_tracking(dim)
            results.append(stats)
            logger.info(
                "  throughput=%.0f v/s  peak=%.1f MB  retained=%.1f MB",
                stats.throughput_vps,
                stats.peak_memory_mb,
                stats.retained_memory_mb,
            )

        return results

    def run_load_queries(self, ingestion_stats: Optional[List[IngestionStats]] = None) -> List[LoadQueryStats]:
        """
        Populate a collection then hammer it with sequential queries,
        measuring per-query latency at full collection load.
        """
        results: List[LoadQueryStats] = []

        for dim in self.dimensions:
            logger.info("Load query stress: dim=%d  collection=%d", dim, self.total_vectors)
            stats = self._query_under_load(dim)
            results.append(stats)
            logger.info("  mean=%.3fms  p95=%.3fms", stats.mean_ms, stats.p95_ms)

        return results

    def print_ingestion_summary(self, stats: List[IngestionStats]) -> None:
        rows = [s.to_row() for s in stats]
        print(format_results_table(rows, title="Stress Test — Ingestion Overhead"))

    def print_query_summary(self, stats: List[LoadQueryStats]) -> None:
        rows = [s.to_row() for s in stats]
        print(format_results_table(rows, title="Stress Test — Query Latency Under Load"))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ingest_with_tracking(self, dimension: int) -> IngestionStats:
        collection_name = f"stress_{dimension}d_{uuid.uuid4().hex[:8]}"
        collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        gc.collect()

        with memory_tracker() as mem, timer() as t:
            self._batch_ingest(collection, self.total_vectors, dimension)

        peak_mb = mem["peak_mb"]
        elapsed_ms = t["elapsed_ms"]

        # Measure retained memory by forcing GC and sampling again
        gc.collect()
        import tracemalloc
        tracemalloc.start()
        _ = collection.count()
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        retained_mb = current / 1024 / 1024

        self._client.delete_collection(collection_name)

        return IngestionStats(
            dimension=dimension,
            total_vectors=self.total_vectors,
            batch_size=self.batch_size,
            elapsed_ms=elapsed_ms,
            peak_memory_mb=peak_mb,
            retained_memory_mb=retained_mb,
        )

    def _query_under_load(self, dimension: int) -> LoadQueryStats:
        collection_name = f"loadq_{dimension}d_{uuid.uuid4().hex[:8]}"
        collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._batch_ingest(collection, self.total_vectors, dimension)

        query_vectors = generate_random_vectors(self.query_rounds, dimension, seed=77)
        latencies: List[float] = []

        for qv in query_vectors:
            with timer() as t:
                collection.query(
                    query_embeddings=[qv],
                    n_results=self.n_results,
                )
            latencies.append(t["elapsed_ms"])

        self._client.delete_collection(collection_name)

        return LoadQueryStats(
            dimension=dimension,
            collection_size=self.total_vectors,
            concurrent_batch=1,
            latencies_ms=latencies,
        )

    def _batch_ingest(self, collection, total: int, dimension: int) -> None:
        for offset in range(0, total, self.batch_size):
            batch = min(self.batch_size, total - offset)
            vectors = generate_random_vectors(batch, dimension, seed=offset)
            ids = [str(offset + i) for i in range(batch)]
            metadata = generate_metadata(batch)
            documents = [f"Chunk {offset + i}" for i in range(batch)]
            collection.add(
                embeddings=vectors,
                ids=ids,
                metadatas=metadata,
                documents=documents,
            )
