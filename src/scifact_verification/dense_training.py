"""Fine-tuning recipe for the MPNet dense retriever."""

from __future__ import annotations


def build_positive_pairs(query_map: dict[str, str], qrels: dict[str, set[str]], documents: dict[str, str]) -> list[tuple[str, str]]:
    pairs = []
    for query_id, positive_ids in qrels.items():
        if query_id not in query_map:
            continue
        for doc_id in positive_ids:
            if doc_id in documents:
                pairs.append((query_map[query_id], documents[doc_id]))
    return pairs


def mpnet_training_config() -> dict:
    """Configuration used by the separate dense-retriever benchmark run."""
    return {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "loss": "MultipleNegativesRankingLoss",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
    }
