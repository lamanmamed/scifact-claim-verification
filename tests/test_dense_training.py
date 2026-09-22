import unittest

from scifact_verification.dense_training import build_positive_pairs, mpnet_training_config


class DenseTrainingTests(unittest.TestCase):
    def test_positive_pairs(self):
        pairs = build_positive_pairs({"q1": "claim"}, {"q1": {"d1", "d2"}}, {"d1": "doc1", "d2": "doc2"})
        self.assertEqual(set(pairs), {("claim", "doc1"), ("claim", "doc2")})

    def test_recorded_benchmark_recipe(self):
        config = mpnet_training_config()
        self.assertEqual(config["model"], "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(config["loss"], "MultipleNegativesRankingLoss")
        self.assertEqual(config["epochs"], 3)


if __name__ == "__main__":
    unittest.main()
