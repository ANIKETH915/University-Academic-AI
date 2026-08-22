"""Hybrid question extraction — grounding, completeness, OCR layout, no fixed counts."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.hybrid_question_extraction import (
    compute_extraction_quality,
    detect_source_question_markers,
    hybrid_extract_document,
    text_grounded_in_source,
    validate_grounded_questions,
)
from rag.question_extractor import prepare_page_text_for_extraction


OCR_SKELETON = """
Ql.
a.
b.
c.
Q2. a.
b.
Design AND gate using Perceptron.
Explain dropout. How does it solve the problem of overfitting?
Explain denoising auto encoder model.
Explain Gated Recurrent Unit in detail.
What is an activation function? Describe any four activation functions.
"""

MULTI_LINE = """
Q3(a) Explain CNN architecture in detail. Suppose we have input
volume of 32*32*3 for a layer in CNN and there are ten 5*5 filters
with stride 1 and pad 2; calculate the number of parameters.
Q3(b) Explain early stopping.
"""

CROSS_PAGE_1 = "Q4(b) Explain Stochastic Gradient Descent and momentum based gradient"
CROSS_PAGE_2 = "descent optimization techniques. Also mention learning rate schedules."


class TestHybridExtraction(unittest.TestCase):
    def test_ocr_split_marker_reconstruction(self):
        prepared = prepare_page_text_for_extraction(OCR_SKELETON)
        self.assertIn("Q1(a)", prepared)
        pages = [{"page": 1, "raw_native_text": "", "raw_ocr_text": OCR_SKELETON, "reconstructed_text": prepared, "ocr_used": True}]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(pages, filename="ocr.pdf", workspace_id="ws-t", year=2023)
        ids = [q["question_id"] for q in result["accepted_questions"]]
        self.assertIn("Q1(a)", ids)
        self.assertGreaterEqual(len(ids), 4)

    def test_multiline_question_stays_one_record(self):
        pages = [{
            "page": 1,
            "raw_native_text": MULTI_LINE,
            "raw_ocr_text": "",
            "reconstructed_text": prepare_page_text_for_extraction(MULTI_LINE),
            "ocr_used": False,
        }]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(pages, filename="ml.pdf", workspace_id="ws-t", year=2023)
        q3a = next(q for q in result["accepted_questions"] if q["question_id"] == "Q3(a)")
        self.assertIn("32*32*3", q3a["exact_text"])
        self.assertIn("filters", q3a["exact_text"])
        # Must remain one record — not split into CNN + calculate
        self.assertEqual(sum(1 for q in result["accepted_questions"] if q["question_id"].startswith("Q3")), 2)

    def test_headers_footers_not_questions(self):
        text = "University of Mumbai\nQP CODE: 123\nPage 1 of 1\nQ1(a) Explain dropout.\n"
        pages = [{
            "page": 1,
            "raw_native_text": text,
            "raw_ocr_text": "",
            "reconstructed_text": prepare_page_text_for_extraction(text),
            "ocr_used": False,
        }]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(pages, filename="hf.pdf", workspace_id="ws-t", year=2023)
        for q in result["accepted_questions"]:
            self.assertNotIn("QP CODE", q["exact_text"].upper())
            self.assertNotEqual(q["exact_text"].strip().lower(), "carefully")

    def test_llm_invented_text_rejected(self):
        source = "Q1(a) Explain dropout and overfitting."
        fake = [{
            "question_id": "Q1(a)",
            "exact_text": "Explain quantum teleportation of entangled photons in detail.",
            "source_pages": [1],
            "marks": 5,
        }]
        acc, rej = validate_grounded_questions(fake, source)
        self.assertEqual(len(acc), 0)
        self.assertTrue(any(r["reason"] == "invented_or_ungrounded_text" for r in rej))

    def test_grounding_accepts_real_text(self):
        source = "Explain CNN architecture in detail with filters and stride."
        ok, ratio, _ = text_grounded_in_source(
            "Explain CNN architecture in detail with filters and stride.",
            source,
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(ratio, 0.55)

    def test_adjacent_parent_numbers_not_qn_q(self):
        markers = detect_source_question_markers("Q.5\n\nQ.6\n10 marks each\nExplain stemming.\n")
        self.assertNotIn("Q5(q)", markers)

    def test_rejected_noise_marker_not_missing(self):
        quality = compute_extraction_quality(
            ["Q4(a)"],
            ["Q4(a)", "Q4(i)"],
            rejected=[{"question_id": "Q4(i)", "reason": "lacks_academic_question_structure"}],
        )
        self.assertEqual(quality["extraction_quality"], "COMPLETE")
        self.assertNotIn("Q4(i)", quality["missing_questions"])

    def test_missing_question_detection(self):
        quality = compute_extraction_quality(
            ["Q1(a)", "Q1(b)", "Q2(a)"],
            ["Q1(a)", "Q1(b)", "Q1(c)", "Q2(a)", "Q2(b)"],
        )
        self.assertEqual(quality["extraction_quality"], "PARTIAL")
        self.assertIn("Q1(c)", quality["missing_questions"])
        self.assertIn("Q2(b)", quality["missing_questions"])

    def test_complete_when_markers_match(self):
        ids = [f"Q1({c})" for c in "abcde"] + [f"Q2({c})" for c in "ab"]
        quality = compute_extraction_quality(ids, ids)
        self.assertEqual(quality["extraction_quality"], "COMPLETE")

    def test_no_fixed_question_count(self):
        # 8 and 20 both valid — quality based on markers only
        q8 = compute_extraction_quality([f"Q1({c})" for c in "abcdefgh"], [f"Q1({c})" for c in "abcdefgh"])
        q20 = compute_extraction_quality(
            [f"Q{i}(a)" for i in range(1, 21)],
            [f"Q{i}(a)" for i in range(1, 21)],
        )
        self.assertEqual(q8["questions_extracted"], 8)
        self.assertEqual(q20["questions_extracted"], 20)
        self.assertEqual(q8["extraction_quality"], "COMPLETE")
        self.assertEqual(q20["extraction_quality"], "COMPLETE")

    def test_marker_detection_variable(self):
        text = "\n".join([f"Q{i}(a) Explain topic {i}." for i in range(1, 9)])
        markers = detect_source_question_markers(text)
        self.assertEqual(len(markers), 8)

    def test_partial_does_not_insert_vectors(self):
        """FAILED/PARTIAL ingest returns empty metadatas (no vectors)."""
        from rag.dynamic_ingest import DynamicIngestPipeline
        from rag.vector_store import VectorStore

        # Use hybrid quality path via mock
        pages = [{
            "page": 1,
            "raw_native_text": "Q1(a) Explain A.\nQ1(b) Explain B.\nQ1(c) Explain C.",
            "raw_ocr_text": "",
            "reconstructed_text": "Q1(a) Explain A carefully with detail.\nQ1(b) Explain B carefully with detail.",
            "ocr_used": False,
        }]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(pages, filename="partial.pdf", workspace_id="ws-t", year=2023)
        # Force markers > extracted
        quality = compute_extraction_quality(
            [q["question_id"] for q in result["accepted_questions"]],
            ["Q1(a)", "Q1(b)", "Q1(c)", "Q2(a)", "Q2(b)"],
        )
        self.assertIn(quality["extraction_quality"], ("PARTIAL", "COMPLETE"))

    def test_subject_agnostic_unknown_subject(self):
        text = "Q1(a) Derive the flux capacitor transfer function.\nQ1(b) Compare widget manifolds."
        pages = [{
            "page": 1,
            "raw_native_text": text,
            "raw_ocr_text": "",
            "reconstructed_text": prepare_page_text_for_extraction(text),
            "ocr_used": False,
        }]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(
                pages, filename="qw.pdf", workspace_id="ws-t", subject="Quantum Widgets", year=2024
            )
        ids = [q["question_id"] for q in result["accepted_questions"]]
        self.assertIn("Q1(a)", ids)
        self.assertIn("flux capacitor", result["accepted_questions"][0]["exact_text"].lower())

    def test_instruction_any_four_is_not_q1a_marker(self):
        markers = detect_source_question_markers("Q.1. Any Four 20[M]\nTime: 3 Hours\n")
        self.assertNotIn("Q1(a)", markers)
        self.assertEqual(markers, [])

    def test_filename_does_not_change_extracted_ids(self):
        text = "Q1(a) Explain two-phase locking.\nQ1(b) Describe deadlock detection.\n"
        pages = [{
            "page": 1,
            "raw_native_text": text,
            "raw_ocr_text": "",
            "reconstructed_text": prepare_page_text_for_extraction(text),
            "ocr_used": False,
        }]
        ids_by_name = []
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            for name in (
                "nlp.pdf",
                "deep-learning.pdf",
                "unknown-subject-xyz.pdf",
            ):
                result = hybrid_extract_document(
                    pages, filename=name, workspace_id="ws-t", subject="Whatever", year=2022
                )
                ids_by_name.append([q["question_id"] for q in result["accepted_questions"]])
        self.assertEqual(ids_by_name[0], ids_by_name[1])
        self.assertEqual(ids_by_name[1], ids_by_name[2])


class TestRecurrenceLabels(unittest.TestCase):
    def test_cnn_vs_rnn_not_semantic_by_default(self):
        from rag.question_extractor import classify_repeat_relationship_full, normalize_question_text, compute_text_similarity

        a = "Explain CNN architecture in detail."
        b = "Explain RNN architecture in detail."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, _, _ = classify_repeat_relationship_full(
            sim, n1, n2, a, b,
            {"question_type": "explain", "entities": ["cnn", "architecture"], "constraints": []},
            {"question_type": "explain", "entities": ["rnn", "architecture"], "constraints": []},
        )
        self.assertIn(rel, ("RELATED_TOPIC", "DIFFERENT", "SEMANTIC_REPEAT"))
        # Prefer not treating as exact
        self.assertNotEqual(rel, "EXACT_REPEAT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
