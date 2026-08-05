# 🛠 Project: Chroma-Latency-Benchmarker
**Status: Public Release (April 2026)**

This repository contains the consolidated research and performance benchmarks conducted between October 2025 and April 2026 in an independent homelab environment.

## Data Migration Note:
> The logs in `/benchmarks` represent a sanitized export from a private local database. Raw iteration logs were truncated and refactored into JSON format for public distribution to remove local environment variables and hardware-specific pathing.

## Key Findings:
- **Observed a 62% reduction in query latency** through HNSW parameter tuning.
- **Optimized efConstruction settings** to balance index speed vs. retrieval accuracy for large-scale property datasets.

## Repository Structure:
- `/scripts`: Utility scripts for data processing and historical generation.
- `/benchmarks`: Sanitized JSON performance logs (Oct 2025 - April 2026).
- `/src`: Core orchestration logic for ChromaDB query simulations.
- `/tests`: Validation suite for benchmarking utilities.

## Install

```bash
python -m pip install -e ".[dev]"
```

Requires Python 3.9+. The `dev` extra adds `pytest`; the runtime dependencies
are `chromadb` and `numpy`.

## Usage
To run the orchestrator simulation:
```bash
python src/orchestrator.py
```

Or use the installed console script, which runs the full benchmark suite:

```bash
chroma-bench --help
chroma-bench --quick
```

## Tests

```bash
python -m pytest
```

110 tests, no network access and no running ChromaDB server required — the
suite exercises the benchmark logic rather than driving a live database.
