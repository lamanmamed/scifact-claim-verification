import unittest

from scifact_verification.generation import build_verdict_prompt, format_evidence, parse_verdict_output


class GenerationTests(unittest.TestCase):
    def test_evidence_is_numbered(self):
        text = format_evidence([{"text": "alpha"}, {"text": "beta"}])
        self.assertEqual(text, "[1] alpha\n[2] beta")

    def test_prompt_requires_evidence_only(self):
        prompt = build_verdict_prompt("claim", [{"text": "evidence"}])
        self.assertIn("using only the evidence", prompt)
        self.assertIn("SUPPORTS", prompt)
        self.assertIn("REFUTES", prompt)

    def test_verdict_parser(self):
        response = """Verdict: NOT ENOUGH INFO\n\nEvidence Summary:\n[1] Something\n\nReasoning:\nInsufficient.\n\nCounter Evidence:\nNone"""
        result = parse_verdict_output(response)
        self.assertEqual(result["verdict"], "NOT_ENOUGH_INFO")
        self.assertEqual(result["reasoning"], "Insufficient.")
        self.assertEqual(result["counter_evidence"], "None")

    def test_unknown_verdict(self):
        self.assertEqual(parse_verdict_output("free-form text")["verdict"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
