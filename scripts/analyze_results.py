import json
import os
import glob

def generate_report():
    log_files = sorted(glob.glob("benchmarks/chroma_sim_*.json"))
    
    print(f"{'Date':<12} | {'Size':<8} | {'Latency (ms)':<12} | {'Memory (MB)':<12} | {'HNSW (M/ef)'}")
    print("-" * 65)
    
    first_latency = None
    last_latency = None

    for file_path in log_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            date = os.path.basename(file_path).replace("chroma_sim_", "").replace(".json", "")
            metrics = data['metrics']
            hnsw = data['hnsw_parameter_config']
            
            latency = metrics['query_latency_ms']
            if first_latency is None: first_latency = latency
            last_latency = latency
            
            print(f"{date:<12} | {metrics['collection_size']:<8} | {latency:<12.2f} | {metrics['memory_usage_mb']:<12.1f} | M:{hnsw['M']}, ef:{hnsw['efConstruction']}")

    improvement = ((first_latency - last_latency) / first_latency) * 100
    print("-" * 65)
    print(f"Total Latency Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    generate_report()
