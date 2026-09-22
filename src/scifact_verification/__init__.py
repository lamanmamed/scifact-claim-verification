"""Scientific claim verification components built around the SciFact dataset."""

from .metrics import evaluate_ranking
from .pipeline import FINAL_CONFIG, PipelineConfig
from .retrieval import fuse_scores

__all__ = ["evaluate_ranking", "fuse_scores", "PipelineConfig", "FINAL_CONFIG"]
