"""Ranking metrics used throughout the retrieval and reranking experiments."""

from __future__ import annotations

import math


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def ndcg_at_k(predicted_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    gains = [1 if doc_id in relevant_doc_ids else 0 for doc_id in predicted_doc_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = dcg_at_k([1] * min(len(relevant_doc_ids), k), k)
    return dcg / ideal if ideal else 0.0


def recall_at_k(predicted_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    return len(set(predicted_doc_ids[:k]) & relevant_doc_ids) / len(relevant_doc_ids)


def precision_at_k(predicted_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(doc_id in relevant_doc_ids for doc_id in predicted_doc_ids[:k]) / k


def reciprocal_rank(predicted_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(predicted_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def evaluate_ranking(predicted_doc_ids: list[str], relevant_doc_ids: set[str], k_values=(1, 3, 5, 10, 20)) -> dict[str, float]:
    result = {"MRR": reciprocal_rank(predicted_doc_ids, relevant_doc_ids)}
    for k in k_values:
        result[f"Recall@{k}"] = recall_at_k(predicted_doc_ids, relevant_doc_ids, k)
        result[f"Precision@{k}"] = precision_at_k(predicted_doc_ids, relevant_doc_ids, k)
        result[f"nDCG@{k}"] = ndcg_at_k(predicted_doc_ids, relevant_doc_ids, k)
    return result
