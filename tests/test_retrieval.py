import unittest

from scifact_verification.retrieval import fuse_scores, minmax_normalize, normalize_bm25_query, normalize_dense_query


class RetrievalTests(unittest.TestCase):
    def test_query_normalization(self):
        self.assertEqual(normalize_bm25_query("  Vitamin   D  "), "vitamin d")
        self.assertEqual(normalize_dense_query("Vitamin\nD"), "Vitamin D")

    def test_constant_scores_map_to_one(self):
        self.assertEqual(minmax_normalize({"a": 2.0, "b": 2.0}), {"a": 1.0, "b": 1.0})

    def test_hybrid_fusion_uses_union(self):
        rows = fuse_scores({"a": 3.0, "b": 1.0}, {"b": 1.0, "c": 4.0})
        ids = {row[0] for row in rows}
        self.assertEqual(ids, {"a", "b", "c"})

    def test_hybrid_fusion_respects_alpha(self):
        rows = fuse_scores({"a": 10.0, "b": 0.0}, {"a": 0.0, "b": 10.0}, alpha=1.0)
        self.assertEqual(rows[0][0], "a")

    def test_bad_alpha_raises(self):
        with self.assertRaises(ValueError):
            fuse_scores({}, {}, alpha=1.1)


if __name__ == "__main__":
    unittest.main()
