import unittest

from scifact_verification.rewriting import extract_first_json, has_relation, parse_rewrite_output


class RewritingTests(unittest.TestCase):
    def test_relation_detection(self):
        self.assertTrue(has_relation("Vitamin D may reduce influenza risk"))
        self.assertFalse(has_relation("vitamin D influenza"))

    def test_extract_first_balanced_json(self):
        result = extract_first_json('prefix {"sparse_query":"a","dense_query":"b"} trailing {"x":1}')
        self.assertEqual(result["dense_query"], "b")

    def test_parse_falls_back_to_original_claim(self):
        result = parse_rewrite_output("not json", "original claim")
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["dense_query"], "original claim")

    def test_parse_success(self):
        result = parse_rewrite_output('{"sparse_query":"aspirin prevent stroke","dense_query":"Aspirin prevents stroke."}', "x")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["sparse_query"], "aspirin prevent stroke")


if __name__ == "__main__":
    unittest.main()
