# Chroma Latency Benchmarker

A small benchmarking harness for ChromaDB: query latency, batch-ingestion
throughput, and memory scaling, plus a generator that produces a synthetic
dataset so the analysis tooling has something to run against.

**The sample data in `benchmarks/` is synthetic.** It is produced by
`scripts/generate_historical_data.py`, which interpolates a latency curve
between two fixed endpoints. The files are named `chroma_sim_*.json` for that
reason. They illustrate the shape of an HNSW tuning curve — they are not
measurements, and no conclusion should be drawn from them about real ChromaDB
performance on real hardware.

If you want real numbers, point the harness at your own instance and generate
them. That is what it is for.

## What's here

- `benchmarks/latency_benchmark.py` — query latency measurement
- `benchmarks/stress_test.py` — batch ingestion and memory scaling
- `benchmarks/utils.py` — shared helpers
- `scripts/generate_historical_data.py` — writes the synthetic dataset
- `scripts/analyze_results.py` — summarises a result set
- `scripts/run_all.py` — runs the suite end to end
- `src/orchestrator.py` — ties the pieces together
- `tests/` — 110 tests covering the benchmark logic

## Install

```bash
python -m pip install -e ".[dev]"
```

Python 3.9+. Runtime dependencies are `chromadb` and `numpy`; the `dev` extra
adds `pytest`.

## Usage

```bash
python src/orchestrator.py     # run the orchestrator
chroma-bench --help
chroma-bench --quick
```

## Tests

```bash
python -m pytest
```

110 tests. No network access and no running ChromaDB server required — the
suite exercises the benchmark logic rather than driving a live database.

## Status

A learning project. The harness and its tests are real; the bundled dataset is
not. Treat it as a starting point for benchmarking your own instance rather
than as a source of published results.

## License

MIT — see [LICENSE](LICENSE).
