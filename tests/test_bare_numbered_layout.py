"""Regression: bare-number PYQ layouts (1 / a) / 2 a)) must extract fully."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.question_extractor import prepare_page_text_for_extraction, extract_questions_from_page_text
from rag.hybrid_question_extraction import hybrid_extract_document


MAY2024_LIKE = """
1
(3) All questions carry equal marks.
1
[20]
a)
What are Feed Forward Neural Network?
b) Explain Gradient Descent in Deep Learning.
c)
Explain the dropout method and its advantages.
d) What are Undercomplete Autoencoders?
e)
Explain Pooling operation in CNN.
2 a)
What are the Three Classes of Deep Learning, explain each?
[10]
b) Explain the architecture of CNN with the help of a diagram.
[10]
3 a)
What are the different types of Gradient Descent methods, explain any three of them.
[10]
b) Explain main components of an Autoencoder and its architecture.
[10]
4 a)
Explain LSTM model, how it overcomes the limitation of RNN.
[10]
b) What are the issues faced by Vanilla GAN models?
[10]
5 a)
What are L1 and L2 regularization methods?
[10]
b) Explain any three types of Autoencoders.
[10]
6 a)
What is the significance of Activation Functions in Neural Networks, explain different types Activation functions used in NN.
[10]
b) What are Generative Adversarial Networks, comment on its applications.
[10]
"""


class TestBareNumberedLayout(unittest.TestCase):
    def test_bare_layout_extracts_fifteen(self):
        prep = prepare_page_text_for_extraction(MAY2024_LIKE)
        self.assertIn("Q1(a)", prep)
        self.assertIn("Q1(e)", prep)
        self.assertIn("Q6(b)", prep)
        acc, rej = extract_questions_from_page_text(MAY2024_LIKE, 1, "may.pdf", "ws", year=2024)
        ids = [q["question_id"] for q in acc]
        self.assertEqual(len(acc), 15, f"ids={ids} rej={len(rej)}")
        self.assertIn("Q1(a)", ids)
        q1a = next(q for q in acc if q["question_id"] == "Q1(a)")
        self.assertIn("Feed Forward", q1a["exact_text"])

    def test_orphan_a_assumes_q1(self):
        text = "a) Explain something alone without parent.\nb) Explain Y.\n2 a) Explain Z."
        prep = prepare_page_text_for_extraction(text)
        self.assertIn("Q1(a)", prep)
        self.assertIn("Q2(a)", prep)

    def test_hybrid_complete(self):
        pages = [{
            "page": 1,
            "raw_native_text": MAY2024_LIKE,
            "raw_ocr_text": "",
            "reconstructed_text": prepare_page_text_for_extraction(MAY2024_LIKE),
            "ocr_used": False,
        }]
        from unittest.mock import patch
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(pages, filename="may.pdf", workspace_id="ws", year=2024)
        self.assertEqual(result["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(len(result["accepted_questions"]), 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
