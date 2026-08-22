"""OCR split-layout reconstruction for real scanned/font-encoded PYQ pages."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.question_extractor import (
    prepare_page_text_for_extraction,
    extract_questions_from_page_text,
)


OCR_SKELETON_SAMPLE = """
Engineering (Artificial Intelligence & Machine Learning)
(3) All questions carry equal marks.
Q1.
a.
b.
c.
d.
e.
Q2. a.
b.
Q3. a.
b.
Q4 a.
b.
Q5 a.
b.
Q6 a.
b.
Design AND gate using Perceptron.
Suppose we have N input-output pairs. Our goal is to find the parameters that predict the output y from the input x.
Explain dropout. How does it solve the problem of overfitting?
Explain denoising auto encoder model.
Describe sequence learning problem.
Explain Gated Recurrent Unit in detail.
What is an activation function? Describe any four activation functions.
Explain CNN architecture in detail.
Explain early stopping, batch normalization, and data augmentation.
Explain RNN architecture in detail.
Explain the working of Generative Adversarial Network.
Explain Stochastic Gradient Descent and momentum based gradient descent optimization techniques.
Explain LSTM architecture.
Describe LeNET architecture.
Explain vanishing and exploding gradient in RNNs.
"""


class TestOCRLayoutExtraction(unittest.TestCase):
    def test_skeleton_rebuild_extracts_all_subquestions(self):
        prepared = prepare_page_text_for_extraction(OCR_SKELETON_SAMPLE)
        self.assertIn("Q1(a) Design AND gate", prepared)
        self.assertIn("Q1(e) Describe sequence", prepared)
        acc, rej = extract_questions_from_page_text(OCR_SKELETON_SAMPLE, 1, "ocr.pdf", "ws-ocr", year=2023)
        ids = [q["question_id"] for q in acc]
        self.assertGreaterEqual(len(acc), 14, f"ids={ids} rej={len(rej)}")
        self.assertIn("Q1(a)", ids)
        self.assertIn("Q6(b)", ids)
        # Exact source wording preserved
        q1a = next(q for q in acc if q["question_id"] == "Q1(a)")
        self.assertIn("Design AND gate using Perceptron", q1a["exact_text"])
        # Must not invent adverb-only topics as the question text
        self.assertNotEqual(q1a["exact_text"].strip().lower(), "carefully")


if __name__ == "__main__":
    unittest.main(verbosity=2)
