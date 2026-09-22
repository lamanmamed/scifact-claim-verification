import random
import unittest

import torch

from scifact_verification.reranking import pairwise_logistic_loss, rerank_candidates, sample_random_negatives


class DummyReranker:
    def predict(self, pairs):
        return [len(document) for _, document in pairs]


class RerankingTests(unittest.TestCase):
    def test_pairwise_loss_prefers_positive_scores(self):
        good = pairwise_logistic_loss(torch.tensor([3.0]), torch.tensor([0.0]))
        bad = pairwise_logistic_loss(torch.tensor([0.0]), torch.tensor([3.0]))
        self.assertLess(good.item(), bad.item())

    def test_reranker_orders_by_cross_score(self):
        candidates = [("a", 0.9, 1.0, 0.8), ("b", 0.8, 0.7, 0.9)]
        docs = {"a": "short", "b": "a much longer document"}
        result = rerank_candidates("claim", candidates, docs, DummyReranker())
        self.assertEqual(result[0][0], "b")

    def test_random_negatives_exclude_positive_and_hard(self):
        result = sample_random_negatives(
            ["a", "b", "c", "d"], {"a"}, ["b"], count=2, rng=random.Random(1)
        )
        self.assertEqual(set(result), {"c", "d"})


if __name__ == "__main__":
    unittest.main()
