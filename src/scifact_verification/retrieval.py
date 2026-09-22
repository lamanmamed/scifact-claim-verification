"""Sparse, dense, and hybrid retrieval helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np


def normalize_bm25_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def normalize_dense_query(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def minmax_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if high == low:
        return {str(doc_id): 1.0 for doc_id in scores}
    return {str(doc_id): float((score - low) / (high - low)) for doc_id, score in scores.items()}


def fuse_scores(
    bm25_scores: Mapping[str, float],
    dense_scores: Mapping[str, float],
    *,
    alpha: float = 0.5,
) -> list[tuple[str, float, float, float]]:
    """Min-max normalize both retrievers and combine them with a weighted sum."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    sparse = minmax_normalize(bm25_scores)
    dense = minmax_normalize(dense_scores)
    doc_ids = set(sparse) | set(dense)
    rows = [
        (
            doc_id,
            alpha * sparse.get(doc_id, 0.0) + (1 - alpha) * dense.get(doc_id, 0.0),
            sparse.get(doc_id, 0.0),
            dense.get(doc_id, 0.0),
        )
        for doc_id in doc_ids
    ]
    return sorted(rows, key=lambda row: row[1], reverse=True)


def bm25_search(searcher, query: str, k: int = 20) -> dict[str, float]:
    """Run a Pyserini/Lucene searcher without importing Pyserini at module import time."""
    return {str(hit.docid): float(hit.score) for hit in searcher.search(query, k)}


def dense_search(encoder, faiss_index, doc_ids: list[str], query: str, k: int = 20) -> dict[str, float]:
    embedding = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = faiss_index.search(np.asarray(embedding, dtype="float32"), k)
    return {
        str(doc_ids[index]): float(score)
        for score, index in zip(scores[0], indices[0])
        if index != -1
    }


def hybrid_search(
    *,
    sparse_query: str,
    dense_query: str,
    bm25_searcher,
    dense_encoder,
    faiss_index,
    doc_ids: list[str],
    k_bm25: int = 20,
    k_dense: int = 20,
    alpha: float = 0.5,
):
    sparse = bm25_search(bm25_searcher, sparse_query, k_bm25)
    dense = dense_search(dense_encoder, faiss_index, doc_ids, dense_query, k_dense)
    return fuse_scores(sparse, dense, alpha=alpha)
