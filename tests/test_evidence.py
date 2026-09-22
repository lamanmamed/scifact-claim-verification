import unittest

from scifact_verification.evidence import best_qa_span, sentence_containing_span, snippet_around_span, split_into_sentences


class EvidenceTests(unittest.TestCase):
    def test_sentence_split(self):
        self.assertEqual(split_into_sentences("One. Two? Three!"), ["One.", "Two?", "Three!"])

    def test_sentence_for_span(self):
        text = "First sentence. Second sentence here. Third sentence."
        start = text.index("Second")
        end = start + len("Second")
        self.assertEqual(sentence_containing_span(text, start, end), "Second sentence here.")

    def test_snippet_window(self):
        text = "First sentence. Second sentence here. Third sentence."
        start = text.index("Second")
        end = start + len("Second")
        self.assertEqual(snippet_around_span(text, start, end, window=1), text)

    def test_best_qa_span_respects_max_length(self):
        start_logits = [0.0, 3.0, 1.0, 0.0]
        end_logits = [0.0, 0.0, 4.0, 10.0]
        start, end, _ = best_qa_span(start_logits, end_logits, [1, 2, 3], max_answer_len=2)
        self.assertEqual((start, end), (2, 3))


if __name__ == "__main__":
    unittest.main()
