import unittest

from scifact_verification.metrics import evaluate_ranking, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.pred = ["d1", "d2", "d3", "d4"]
        self.gold = {"d2", "d4"}

    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(self.pred, self.gold, 2), 0.5)

    def test_precision_at_k(self):
        self.assertEqual(precision_at_k(self.pred, self.gold, 2), 0.5)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(self.pred, self.gold), 0.5)

    def test_ndcg_is_bounded(self):
        value = ndcg_at_k(self.pred, self.gold, 4)
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 1)

    def test_evaluate_ranking(self):
        metrics = evaluate_ranking(self.pred, self.gold, k_values=(1, 2))
        self.assertIn("Recall@2", metrics)
        self.assertIn("nDCG@1", metrics)
        self.assertIn("MRR", metrics)


if __name__ == "__main__":
    unittest.main()
