import chromadb
import time
import numpy as np

def run_real_benchmark():
    # Initialize ephemeral client (in-memory)
    client = chromadb.EphemeralClient()
    
    # Create a collection with specific HNSW params
    collection = client.create_collection(
        name="resume_sample_collection",
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 16}
    )

    print("--- Real ChromaDB Performance Test ---")
    
    # Simulate a small batch ingestion
    ids = [f"id_{i}" for i in range(100)]
    embeddings = np.random.uniform(-1, 1, (100, 128)).tolist()
    
    start_add = time.time()
    collection.add(ids=ids, embeddings=embeddings)
    print(f"Ingested 100 vectors in {(time.time() - start_add)*1000:.2f}ms")

    # Run a real query
    query_vector = np.random.uniform(-1, 1, (1, 128)).tolist()
    start_query = time.time()
    results = collection.query(query_embeddings=query_vector, n_results=5)
    
    print(f"Query completed in {(time.time() - start_query)*1000:.2f}ms")
    print(f"Top Result ID: {results['ids'][0][0]}")

if __name__ == "__main__":
    try:
        run_real_benchmark()
    except ImportError:
        print("Error: chromadb not installed. Run 'pip install chromadb' to see the real implementation.")
