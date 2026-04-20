# Chroma-Latency-Benchmarker

A Python-based benchmarking suite for evaluating [ChromaDB](https://www.trychroma.com/) retrieval performance across the vector dimensions used by common LLM embedding models. Includes both a latency matrix benchmark and a high-density ingestion stress test with memory overhead tracking.

---

## Research Context

This codebase was developed during a private research phase (Oct 2025 – April 2026) and has been refactored for public benchmarking use. The original work focused on evaluating vector database viability in GPU-heavy inference infrastructure, where embedding generation throughput can outpace the downstream vector store. Results from this suite informed architectural decisions around collection sharding, batch ingestion cadence, and hardware provisioning for production retrieval-augmented generation (RAG) pipelines.

---

## Motivation

GPU-accelerated embedding pipelines (e.g., vLLM + a fine-tuned encoder) can produce embeddings significantly faster than a naively configured vector store can index and serve them. This project answers two practical questions:

1. **Latency** — How does ChromaDB query latency scale across different embedding dimensions (384 → 768 → 1536) and collection sizes?
2. **Overhead** — What is the peak and retained memory footprint during high-throughput ingestion, and where does throughput saturate?

---

## Features

- Latency benchmark across three standard embedding dimensions
- Configurable collection sizes (1k → 10k → 50k vectors by default)
- p50 / p95 / p99 / StdDev per (dimension, collection size) pair
- Stress test module: ingestion throughput (vectors/sec), peak RAM, retained RAM
- Load-query phase: per-query latency measured against a fully-loaded collection
- Optional JSON export for downstream analysis or CI comparison
- `--quick` flag for fast smoke-test runs during development

---

## Embedding Dimensions Covered

| Dimension | Representative Models |
|-----------|----------------------|
| 384 | `all-MiniLM-L6-v2`, `paraphrase-MiniLM-L3-v2` |
| 768 | `bge-base-en-v1.5`, `e5-base-v2`, OpenAI `ada-001` |
| 1536 | OpenAI `text-embedding-ada-002`, `text-embedding-3-small` |

---

## Installation

```bash
git clone https://github.com/your-username/Chroma-Latency-Benchmarker.git
cd Chroma-Latency-Benchmarker
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.9+**.

---

## Usage

### Full benchmark suite (defaults)
```bash
python scripts/run_all.py
```

### Quick smoke-test
```bash
python scripts/run_all.py --quick
```

### Latency benchmark only
```bash
python scripts/run_all.py --latency-only
```

### Stress test only (200k vectors per dimension)
```bash
python scripts/run_all.py --stress-only --total-vectors 200000
```

### Custom dimensions and collection sizes
```bash
python scripts/run_all.py --dimensions 384 768 --collection-sizes 5000 25000 100000
```

### Save results to JSON
```bash
python scripts/run_all.py --output-json results/run_$(date +%Y%m%d).json
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | off | Reduced scale for fast iteration |
| `--latency-only` | off | Skip the stress test phase |
| `--stress-only` | off | Skip the latency benchmark phase |
| `--dimensions` | `384 768 1536` | Space-separated list of embedding dimensions |
| `--collection-sizes` | `1000 10000 50000` | Vectors per collection in latency benchmark |
| `--total-vectors` | `100000` | Vectors per dimension in stress test |
| `--batch-size` | `500` | Vectors per `add()` call during ingestion |
| `--n-queries` | `50` | Query repetitions per (dim, size) combination |
| `--output-json` | none | Path to write JSON results |

---

## Project Structure

```
Chroma-Latency-Benchmarker/
├── benchmarks/
│   ├── __init__.py
│   ├── latency_benchmark.py   # LatencyBenchmark class
│   ├── stress_test.py         # StressTest class
│   └── utils.py               # Vector generation, timing, memory tracking
├── scripts/
│   └── run_all.py             # CLI entry point
├── results/                   # Output directory (JSON exports)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Sample Output

```
==============================================================
 ChromaDB Latency Benchmark Results
==============================================================
+------+----------------------------+...+----------+----------+
| Dim  | Model (approx.)            |...| Mean(ms) | p95 (ms) |
+------+----------------------------+...+----------+----------+
|  384 | MiniLM-L6 (384d)           |...|    0.821 |    1.204 |
|  768 | BGE-base / E5-base (768d)  |...|    1.347 |    1.893 |
| 1536 | text-embedding-ada-002 ... |...|    2.614 |    3.501 |
+------+----------------------------+...+----------+----------+

==============================================================
 Stress Test — Ingestion Overhead
==============================================================
+------+---------+-----------+--------+--------------+----------+
| Dim  | Vectors | Batch Size | Thpt   | Peak RAM(MB) | Ret.(MB) |
+------+---------+-----------+--------+--------------+----------+
|  384 | 100000  |       500 | 12400  |        148.3 |     42.1 |
|  768 | 100000  |       500 |  8700  |        241.7 |     68.4 |
| 1536 | 100000  |       500 |  5100  |        432.9 |    121.6 |
+------+---------+-----------+--------+--------------+----------+
```

*(Actual values will vary by hardware. The above is illustrative.)*

---

## Limitations

- All tests use ChromaDB's `EphemeralClient` (in-memory); results reflect pure compute and allocator overhead, not I/O or persistence.
- Vectors are synthetically generated (unit-normalized Gaussian). Real embedding distributions will differ.
- No multi-thread or multi-process concurrency is modelled in the current query phase.
- HNSW index construction cost is amortized across the ingestion batches; cold-start latency is not isolated.

---

## Contributing

Pull requests are welcome. When adding a new benchmark module, follow the pattern in `benchmarks/latency_benchmark.py`: a class with a `run()` method returning a list of dataclasses and a `print_summary()` method.

---

## License

MIT
