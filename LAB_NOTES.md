# 📓 Internal Research Lab Notes

### Phase 1: Baseline & Environment Setup (Oct 2025)
- **Oct 12:** Initialized project. Testing ChromaDB `EphemeralClient`.
- **Oct 28:** Encountered inconsistent latency spikes during ingestion. Suspecting Windows background process interference. 
- **Goal:** Move to a dedicated Linux-based homelab node for cleaner metrics.

### Phase 2: Memory Scaling & Stress (Dec 2025)
- **Dec 05:** Collection size reached 10k vectors. Memory footprint is scaling linearly (~0.5MB per 1k records).
- **Dec 20:** Batch ingestion saturation point found at 500 vectors/batch. Higher values cause the SQLite backend to bottleneck.

### Phase 3: HNSW Optimization (Feb 2026)
- **Feb 14:** Default HNSW parameters ($M=16, ef=200$) are failing to maintain sub-100ms latency for 1536d vectors.
- **Feb 28:** Breaking through the "Latency Wall." Doubling $M$ to 32 reduced p95 latency by 18%, but index time increased by 40%.

### Phase 4: Finalization & Sanitization (April 2026)
- **April 05:** Reached the 50k vector milestone. Performance remains stable with $M=64$ and $efConstruction=600$.
- **April 18:** Preparing logs for public release. Truncating raw iteration data and removing local environment paths (e.g., `C:\Users\jake\LabNode1\..`).
- **April 20:** Final push to GitHub.
