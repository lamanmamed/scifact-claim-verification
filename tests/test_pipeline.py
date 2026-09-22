import unittest

from scifact_verification.pipeline import FINAL_CONFIG, PipelineConfig


class PipelineTests(unittest.TestCase):
    def test_final_config(self):
        self.assertEqual(FINAL_CONFIG.retrieval_method, "hybrid_mpnet")
        self.assertEqual(FINAL_CONFIG.reranker_method, "bge")
        self.assertTrue(FINAL_CONFIG.use_spanbert)
        FINAL_CONFIG.validate()

    def test_bad_config_raises(self):
        with self.assertRaises(ValueError):
            PipelineConfig(retrieval_method="unknown").validate()


if __name__ == "__main__":
    unittest.main()
