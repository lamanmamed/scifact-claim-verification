"""Configuration and composition helpers for the final claim-verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    retrieval_method: str = "hybrid_mpnet"
    rewrite_method: str = "qwen"
    reranker_method: str = "bge"
    generator_model: str = "Qwen/Qwen2.5-3B-Instruct"
    use_spanbert: bool = True
    retrieval_k: int = 20
    evidence_k: int = 3

    def validate(self) -> None:
        if self.retrieval_method not in {"bm25", "dense_mpnet", "dense_minilm", "hybrid_mpnet", "hybrid_minilm"}:
            raise ValueError(f"Unknown retrieval method: {self.retrieval_method}")
        if self.rewrite_method not in {"none", "qwen", "llama"}:
            raise ValueError(f"Unknown rewrite method: {self.rewrite_method}")
        if self.reranker_method not in {"none", "minilm", "bge"}:
            raise ValueError(f"Unknown reranker method: {self.reranker_method}")
        if self.retrieval_k < 1 or self.evidence_k < 1:
            raise ValueError("k values must be positive")


FINAL_CONFIG = PipelineConfig()
