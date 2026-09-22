"""Metrics recorded in the saved project notebooks."""

DENSE_BENCHMARK = {
    "baseline_mpnet_recall20": 0.8530,
    "finetuned_mpnet_recall20": 0.9109,
}

FINAL_RETRIEVAL = {
    "BM25": {"ndcg20": 0.6919, "recall20": 0.8521},
    "Dense MPNet": {"ndcg20": 0.7186, "recall20": 0.8946},
    "Dense MiniLM": {"ndcg20": 0.6630, "recall20": 0.8373},
    "Hybrid MPNet": {"ndcg20": 0.7600, "recall20": 0.9279},
    "Hybrid MiniLM": {"ndcg20": 0.7319, "recall20": 0.8767},
}

QUERY_REWRITING = {
    "no_rewrite": {"ndcg20": 0.7600, "recall20": 0.9279, "mrr": 0.7139},
    "qwen": {"ndcg20": 0.7561, "recall20": 0.9336, "mrr": 0.7085},
    "llama": {"ndcg20": 0.7475, "recall20": 0.9286, "mrr": 0.6960},
}

RERANKING = {
    "no_rerank": {"ndcg3": 0.6955, "recall3": 0.7591},
    "MiniLM": {"ndcg3": 0.6577, "recall3": 0.7094},
    "BGE": {"ndcg3": 0.7546, "recall3": 0.7988},
}

GENERATION = {
    "Qwen2.5-3B-Instruct": {"overall": 0.543, "supports": 0.758, "refutes": 0.391, "nei": 0.393, "mean_latency": 3.22},
    "Llama-3.2-3B-Instruct": {"overall": 0.427, "supports": 0.815, "refutes": 0.422, "nei": 0.000, "mean_latency": 2.83},
    "Qwen2.5-3B-Instruct + SpanBERT": {"overall": 0.570, "supports": 0.702, "refutes": 0.469, "nei": 0.482, "mean_latency": 3.74},
}
