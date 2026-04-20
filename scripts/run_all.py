#!/usr/bin/env python3
"""
run_all.py — Entry point for the full benchmark suite.

Usage
-----
    python scripts/run_all.py [--quick] [--stress-only] [--latency-only]
                              [--total-vectors N] [--batch-size N]
                              [--collection-sizes 1000 10000 50000]
                              [--dimensions 384 768 1536]

Examples
--------
    # Full suite with defaults
    python scripts/run_all.py

    # Quick smoke-test (small collections, fewer queries)
    python scripts/run_all.py --quick

    # Stress test only, 200k vectors per dimension
    python scripts/run_all.py --stress-only --total-vectors 200000
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running from repo root or scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import LatencyBenchmark, StressTest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ChromaDB latency & stress benchmarker")
    p.add_argument("--quick", action="store_true", help="Reduced scale for fast iteration")
    p.add_argument("--stress-only", action="store_true", help="Skip latency benchmark")
    p.add_argument("--latency-only", action="store_true", help="Skip stress test")
    p.add_argument("--total-vectors", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--collection-sizes", type=int, nargs="+", default=None)
    p.add_argument("--dimensions", type=int, nargs="+", default=[384, 768, 1536])
    p.add_argument("--n-queries", type=int, default=None)
    p.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        collection_sizes = args.collection_sizes or [500, 2_000]
        total_vectors = args.total_vectors or 5_000
        n_queries = args.n_queries or 10
        batch_size = 500
    else:
        collection_sizes = args.collection_sizes or [1_000, 10_000, 50_000]
        total_vectors = args.total_vectors or 100_000
        n_queries = args.n_queries or 50
        batch_size = args.batch_size

    all_results: dict = {}
    suite_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Latency Benchmark                                                    #
    # ------------------------------------------------------------------ #
    if not args.stress_only:
        logger.info("=" * 60)
        logger.info("PHASE 1 — Latency Benchmark")
        logger.info("  dimensions    : %s", args.dimensions)
        logger.info("  coll. sizes   : %s", collection_sizes)
        logger.info("  queries/pair  : %d", n_queries)
        logger.info("=" * 60)

        lb = LatencyBenchmark(
            collection_sizes=collection_sizes,
            n_queries=n_queries,
            dimensions=args.dimensions,
        )
        latency_reports = lb.run()
        lb.print_summary(latency_reports)

        all_results["latency"] = [r.to_row() for r in latency_reports]

    # ------------------------------------------------------------------ #
    # Stress Test                                                          #
    # ------------------------------------------------------------------ #
    if not args.latency_only:
        logger.info("=" * 60)
        logger.info("PHASE 2 — Stress Test")
        logger.info("  dimensions    : %s", args.dimensions)
        logger.info("  total vectors : %d", total_vectors)
        logger.info("  batch size    : %d", batch_size)
        logger.info("=" * 60)

        st = StressTest(
            dimensions=args.dimensions,
            total_vectors=total_vectors,
            batch_size=batch_size,
            query_rounds=n_queries,
        )

        ingestion_stats = st.run_ingestion_stress()
        st.print_ingestion_summary(ingestion_stats)

        query_stats = st.run_load_queries()
        st.print_query_summary(query_stats)

        all_results["stress_ingestion"] = [s.to_row() for s in ingestion_stats]
        all_results["stress_queries"] = [s.to_row() for s in query_stats]

    # ------------------------------------------------------------------ #
    # Optional JSON export                                                 #
    # ------------------------------------------------------------------ #
    elapsed = time.perf_counter() - suite_start
    logger.info("Suite completed in %.1f seconds.", elapsed)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
