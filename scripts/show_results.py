from experiments.recorded_results import FINAL_RETRIEVAL, GENERATION, RERANKING

print("Retrieval")
for name, metrics in FINAL_RETRIEVAL.items():
    print(f"{name:15} Recall@20={metrics['recall20']:.4f} nDCG@20={metrics['ndcg20']:.4f}")

print("\nReranking")
for name, metrics in RERANKING.items():
    print(f"{name:10} Recall@3={metrics['recall3']:.4f} nDCG@3={metrics['ndcg3']:.4f}")

print("\nGeneration")
for name, metrics in GENERATION.items():
    print(f"{name:35} accuracy={metrics['overall']:.3f}")
