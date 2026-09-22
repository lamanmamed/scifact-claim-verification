"""Cross-encoder reranking and pairwise training helpers."""

from __future__ import annotations

import random

import torch


def rerank_candidates(
    claim: str,
    candidates: list[tuple[str, float, float, float]],
    document_text: dict[str, str],
    reranker,
    *,
    top_n: int = 20,
) -> list[tuple[str, float, float, float, float]]:
    selected = candidates[:top_n]
    pairs = [(claim, document_text.get(str(doc_id), "")) for doc_id, *_ in selected]
    scores = reranker.predict(pairs)
    rows = [
        (str(doc_id), float(cross_score), float(hybrid), float(bm25), float(dense))
        for (doc_id, hybrid, bm25, dense), cross_score in zip(selected, scores)
    ]
    return sorted(rows, key=lambda row: row[1], reverse=True)


def pairwise_logistic_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(positive_scores - negative_scores) + 1e-8).mean()


def sample_random_negatives(
    all_doc_ids: list[str],
    positive_ids: set[str],
    hard_negative_ids: list[str],
    *,
    count: int = 2,
    rng: random.Random | None = None,
) -> list[str]:
    rng = rng or random
    blocked = set(positive_ids) | set(hard_negative_ids)
    available = [doc_id for doc_id in all_doc_ids if doc_id not in blocked]
    return rng.sample(available, min(count, len(available)))


def bge_training_config() -> dict:
    return {
        "base_model": "BAAI/bge-reranker-large",
        "epochs": 2,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "max_length": 512,
        "hard_negatives_per_query": 2,
        "random_negatives_per_query": 2,
    }
