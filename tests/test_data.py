import tempfile
import unittest
from pathlib import Path

from scifact_verification.data import build_full_text, make_bm25_text, make_raw_text, prepare_document, load_qrels


class DataTests(unittest.TestCase):
    def test_full_text_avoids_double_period(self):
        self.assertEqual(build_full_text("A title.", "Abstract text."), "A title. Abstract text.")

    def test_raw_text_preserves_unicode_and_removes_controls(self):
        self.assertEqual(make_raw_text("β\n  value"), "β value")

    def test_bm25_text_lowercases_and_normalizes_dashes(self):
        self.assertEqual(make_bm25_text("Vitamin D — Risk"), "vitamin d - risk")

    def test_prepare_document_keeps_all_retrieval_views(self):
        row = prepare_document({"_id": "7", "title": "Title", "text": "Text"})
        self.assertEqual(row["doc_id"], "7")
        self.assertEqual(row["dense_text"], "Title. Text")
        self.assertEqual(row["bm25_text"], "title. text")

    def test_load_qrels_ignores_nonpositive_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.tsv"
            path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n", encoding="utf-8")
            self.assertEqual(load_qrels(path), {"q1": {"d1"}})


if __name__ == "__main__":
    unittest.main()
