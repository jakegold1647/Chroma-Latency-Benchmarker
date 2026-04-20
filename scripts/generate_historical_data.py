import json
import os
from datetime import datetime, timedelta

def generate_data():
    output_dir = "results/historical_data"
    os.makedirs(output_dir, exist_ok=True)

    # Configuration for the simulation
    start_date = datetime(2025, 10, 1)
    # To get 12 intervals from Oct 2025 to April 2026, we'll do bi-weekly or semi-monthly
    # The prompt asks for 12 logs over 7 months.
    dates = [start_date + timedelta(days=15 * i) for i in range(12)]
    
    # Scaling parameters
    start_size = 1000
    end_size = 50000
    size_step = (end_size - start_size) / 11

    # Latency trend (simulating optimization: starts high, drops even as size grows)
    start_latency = 120.0
    end_latency = 45.0
    latency_step = (start_latency - end_latency) / 11

    # Memory: ~0.5MB per 1k records + base overhead
    memory_base = 128.0
    memory_per_record = 0.5 / 1000

    notes_pool = [
        "Initial baseline for small collection.",
        "Adjusting M and efConstruction parameters for better recall.",
        "Increased ef_search to stabilize tail latency.",
        "Index flattened for high-density deployment.",
        "Optimized segment management in ChromaDB.",
        "Applied HNSW graph pruning.",
        "Switching to optimized distance metrics.",
        "Memory-to-disk paging verified for larger scale.",
        "Fine-tuning efConstruction for 20k+ nodes.",
        "Improved batch insertion logic.",
        "Final optimization pass on HNSW parameters.",
        "Production-ready configuration reached."
    ]

    hnsw_configs = [
        {"M": 16, "efConstruction": 200},
        {"M": 16, "efConstruction": 200},
        {"M": 24, "efConstruction": 300},
        {"M": 24, "efConstruction": 300},
        {"M": 32, "efConstruction": 400},
        {"M": 32, "efConstruction": 400},
        {"M": 32, "efConstruction": 400},
        {"M": 48, "efConstruction": 500},
        {"M": 48, "efConstruction": 500},
        {"M": 64, "efConstruction": 600},
        {"M": 64, "efConstruction": 600},
        {"M": 64, "efConstruction": 600},
    ]

    for i, date in enumerate(dates):
        size = int(start_size + (size_step * i))
        # Add a little jitter to latency
        latency = max(5.0, (start_latency - (latency_step * i)) + (i % 3))
        memory = memory_base + (size * memory_per_record)
        
        data = {
            "timestamp": date.isoformat(),
            "metrics": {
                "query_latency_ms": round(latency, 2),
                "collection_size": size,
                "memory_usage_mb": round(memory, 2)
            },
            "hnsw_parameter_config": hnsw_configs[i],
            "notes": notes_pool[i]
        }

        filename = f"chroma_sim_{date.strftime('%Y_%m_%d')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
    print(f"Generated 12 logs in {output_dir}")

if __name__ == "__main__":
    generate_data()
