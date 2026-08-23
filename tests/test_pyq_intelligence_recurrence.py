"""
Regression suite for PYQ Intelligence rebuild:
extraction validation, exact/semantic/related separation, OCR rejection,
multi-concept integrity, workspace isolation, priority variation,
and no extracted-question dump in main intelligence API.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.question_extractor import (
    extract_questions_from_page_text,
    normalize_question_text,
    classify_repeat_relationship,
    classify_repeat_relationship_full,
    compute_text_similarity,
    validate_question_candidate,
    CanonicalConceptExtractor,
    build_question_representation,
)
from rag.pyq_intelligence import PYQIntelligenceEngine, calculate_deterministic_priority_score
from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.workspace_db import WorkspaceDB
import fitz


def make_pdf(path: str, pages: list) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        y = 40
        for raw_line in text.splitlines():
            # Wrap long lines without truncating content
            while raw_line:
                chunk = raw_line[:95]
                raw_line = raw_line[95:]
                if y > 780:
                    page = doc.new_page()
                    y = 40
                page.insert_text((40, y), chunk, fontsize=9)
                y += 12
    doc.save(path)
    doc.close()


class TestExtractionValidation(unittest.TestCase):
    def test_08_ocr_garbage_rejected(self):
        ok, reason, _ = validate_question_candidate("94X525Yada494X525Yada junk fragment xx")
        self.assertFalse(ok)

    def test_09_header_footer_rejected(self):
        samples = [
            "University of Mumbai B.E. Computer Engineering",
            "QP CODE: 89231 Seat No: ____",
            "Dec-2023 10:30 am Engineering Artificial Intelligence",
            "*** End of Paper ***",
        ]
        for s in samples:
            ok, reason, _ = validate_question_candidate(s)
            self.assertFalse(ok, f"Should reject: {s} ({reason})")

    def test_10_continuation_line_merged(self):
        sample = (
            "Q3(a) Explain CNN architecture in detail. Suppose, we have input volume\n"
            "of 32*32*3 for a layer in CNN with 10 filters of size 5*5*3 and stride of 1.\n"
            "Calculate the number of parameters.\n"
            "Q3(b) Explain early stopping, batch normalization, and data augmentation.\n"
        )
        accepted, rejected = extract_questions_from_page_text(
            sample, 1, "paper.pdf", "ws-merge", year=2023
        )
        ids = [q["question_id"] for q in accepted]
        self.assertIn("Q3(a)", ids)
        self.assertIn("Q3(b)", ids)
        self.assertNotIn("Q3(2)", ids)
        q3a = next(q for q in accepted if q["question_id"] == "Q3(a)")
        self.assertIn("32*32*3", q3a["exact_text"])
        self.assertIn("parameters", q3a["exact_text"].lower())
        q3b = next(q for q in accepted if q["question_id"] == "Q3(b)")
        self.assertIn("early stopping", q3b["exact_text"].lower())

    def test_be_line_not_question(self):
        sample = (
            "University of Mumbai\n"
            "B.E. (Computer Engineering) Sem VII\n"
            "Q1(a) Explain gradient descent in detail with examples.\n"
        )
        accepted, _ = extract_questions_from_page_text(sample, 1, "p.pdf", "ws", year=2024)
        ids = [q["question_id"] for q in accepted]
        self.assertEqual(ids, ["Q1(a)"])
        self.assertNotIn("Q1(b)", ids)

    def test_numeric_subquestion_rejected(self):
        sample = (
            "Q3(a) Explain CNN architecture in detail. Suppose, we have input volume\n"
            "*32*3 for a layer in CNN with 10 filters. Calculate parameters.\n"
            "b Explain early stopping, batch normalization, and data augmentation.\n"
            "Q2(6)\n"
            "Dec-2023 10:30 am Engineering Artificial Intelligence\n"
        )
        accepted, rejected = extract_questions_from_page_text(sample, 1, "p.pdf", "ws", year=2023)
        ids = [q["question_id"] for q in accepted]
        self.assertNotIn("Q2(6)", ids)
        self.assertNotIn("Q3(2)", ids)
        self.assertIn("Q3(a)", ids)
        self.assertIn("Q3(b)", ids)

    def test_07_multi_concept_remains_one(self):
        sample = "Q3(b) Explain early stopping, batch normalization, and data augmentation.\n"
        accepted, _ = extract_questions_from_page_text(sample, 1, "p.pdf", "ws", year=2023)
        self.assertEqual(len(accepted), 1)
        topics = accepted[0]["detected_topics"]
        # May detect multiple topic labels, but still ONE question record
        self.assertEqual(accepted[0]["question_id"], "Q3(b)")


class TestRecurrenceClassification(unittest.TestCase):
    def test_01_exact_repeat_capitalization(self):
        a = "Explain Gradient Descent in Deep Learning."
        b = "explain gradient descent in deep learning."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, conf, _ = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertEqual(rel, "EXACT_REPEAT")
        self.assertGreaterEqual(conf, 0.9)

    def test_02_exact_repeat_punctuation(self):
        a = "What is the significance of Activation Functions in Neural Networks, explain different types Activation functions used in NN."
        b = "What is the significance of Activation Functions in Neural Networks, explain different types of Activation functions used in NN."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, conf, _ = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertEqual(rel, "EXACT_REPEAT")

    def test_03_semantic_paraphrase(self):
        a = "Explain the working of backpropagation."
        b = "Describe how the backpropagation algorithm works."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, conf, reason = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_03b_dropout_advantages_vs_overfitting_is_related(self):
        a = "Explain the dropout method and its advantages."
        b = "Explain dropout. How does it solve the problem of overfitting?"
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, _, reason = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertNotIn(rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"}, reason)
        self.assertEqual(rel, "RELATED_TOPIC", reason)

    def test_04_related_not_same_intent(self):
        a = "Explain LSTM architecture."
        b = "Differentiate between LSTM and GRU."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, _, reason = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"}, reason)
        self.assertNotEqual(rel, "EXACT_REPEAT")
        self.assertNotEqual(rel, "SEMANTIC_REPEAT")

    def test_05_cnn_architecture_vs_parameter_calc(self):
        a = "Explain CNN architecture in detail."
        b = "Calculate the number of parameters in a convolutional layer."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, _, reason = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"}, reason)
        self.assertNotIn(rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"})

    def test_06_lstm_vs_gru_compare_not_same(self):
        a = "Explain LSTM architecture."
        b = "Differentiate between LSTM and GRU."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, _, _ = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertNotEqual(rel, "SEMANTIC_REPEAT")

    def test_gan_architecture_vs_applications(self):
        a = "Explain GAN architecture."
        b = "Explain applications of GAN."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, _, reason = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"}, reason)

    def test_cnn_vs_pooling_not_same(self):
        a = "Explain CNN architecture."
        b = "Explain pooling operation in CNN."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        sim = compute_text_similarity(n1, n2)
        rel, _, _, reason = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertNotEqual(rel, "EXACT_REPEAT")
        # Pooling vs full architecture should not be forced as semantic repeat
        if rel == "SEMANTIC_REPEAT":
            self.fail(f"False semantic repeat: {reason}")


class TestIntelligenceAPIAndPriority(unittest.TestCase):
    WS_A = "ws-recurrence-regression-a"
    WS_B = "ws-recurrence-regression-b"

    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()
        cls.ingest = DynamicIngestPipeline(vector_store=cls.store)
        cls.engine = PYQIntelligenceEngine(vector_store=cls.store)
        cls.ws_db = WorkspaceDB()
        cls.store.delete_by_workspace(cls.WS_A)
        cls.store.delete_by_workspace(cls.WS_B)
        cls.tmpdir = tempfile.TemporaryDirectory()

        # Prefer slightly shorter stems so wrapped PDF lines stay intact
        p2024 = os.path.join(cls.tmpdir.name, "Subj_2024_May.pdf")
        p2025 = os.path.join(cls.tmpdir.name, "Subj_2025_May.pdf")
        make_pdf(
            p2024,
            [
                "UNIVERSITY EXAM 2024\n"
                "Q1(a) What is the significance of Activation Functions in Neural Networks, explain different types Activation functions used in NN.\n"
                "Q1(b) Explain the dropout method and its advantages.\n"
                "Q2(a) Explain CNN architecture in detail.\n"
                "Q2(b) Explain LSTM architecture.\n"
                "Q3(a) Explain Gradient Descent in Deep Learning.\n"
            ],
        )
        make_pdf(
            p2025,
            [
                "UNIVERSITY EXAM 2025\n"
                "Q6(a) What is the significance of Activation Functions in Neural Networks, explain different types of Activation functions used in NN.\n"
                "Q6(b) Explain dropout. How does it solve the problem of overfitting?\n"
                "Q7(a) Calculate the number of parameters in a convolutional layer.\n"
                "Q7(b) Differentiate between LSTM and GRU.\n"
                "Q8(a) Explain the gradient descent algorithm used in neural network. Also discuss types of gradient descent in detail.\n"
            ],
        )

        ws = cls.ws_db.get_or_create(cls.WS_A, subject="Deep Learning", semester="Semester 7")
        cls.ingest.parse_pyq_pdf(p2024, ws)
        cls.ingest.parse_pyq_pdf(p2025, ws)

        # Different workspace with unrelated question
        p_other = os.path.join(cls.tmpdir.name, "Other_2024.pdf")
        make_pdf(p_other, ["Q1(a) Explain paging and segmentation in operating systems in detail.\n"])
        ws_b = cls.ws_db.get_or_create(cls.WS_B, subject="Operating Systems", semester="Semester 5")
        cls.ingest.parse_pyq_pdf(p_other, ws_b)

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_by_workspace(cls.WS_A)
        cls.store.delete_by_workspace(cls.WS_B)
        cls.tmpdir.cleanup()

    def test_11_cross_pdf_linking(self):
        analysis = self.engine.get_pyq_analysis(self.WS_A, subject="Deep Learning")
        self.assertEqual(analysis["total_papers"], 2)
        linked = int(analysis["exact_repeat_count"]) + int(analysis["semantic_repeat_count"])
        self.assertGreaterEqual(linked, 1)
        exact_blob = " ".join(g.get("exact_text", "") for g in analysis["exact_repeats"]).lower()
        sem_blob = " ".join(
            " ".join((oq.get("text") or "") for oq in (g.get("original_questions") or []))
            for g in analysis["semantic_repeats"]
        ).lower()
        self.assertTrue(
            "activation" in exact_blob
            or "activation" in sem_blob
            or "gradient" in exact_blob
            or "gradient" in sem_blob
            or linked >= 1
        )

    def test_12_workspace_isolation(self):
        a = self.engine.get_pyq_analysis(self.WS_A, subject="Deep Learning")
        b = self.engine.get_pyq_analysis(self.WS_B, subject="Operating Systems")
        a_texts = " ".join(q.get("sample_question", "") for q in a.get("topics", [])).lower()
        self.assertNotIn("paging", a_texts)
        self.assertEqual(b["total_papers"], 1)
        b_qs = self.engine.get_source_questions(self.WS_B)
        self.assertTrue(any("paging" in q["exact_text"].lower() for q in b_qs))

    def test_13_priority_scores_not_identical(self):
        analysis = self.engine.get_pyq_analysis(self.WS_A, subject="Deep Learning")
        scores = [t["priority_score"] for t in analysis.get("topics", [])]
        self.assertGreaterEqual(len(scores), 2)
        self.assertGreater(len(set(scores)), 1, f"Priority scores artificially identical: {scores}")

    def test_14_no_hardcoded_15_assumption(self):
        # 5 questions in 2024 paper — count must follow PDF, not assume 15
        analysis = self.engine.get_pyq_analysis(self.WS_A)
        self.assertEqual(analysis["total_valid_questions"], 10)

    def test_17_no_extracted_dump_in_main_api(self):
        analysis = self.engine.get_pyq_analysis(self.WS_A)
        self.assertEqual(analysis.get("extracted_questions"), [])
        self.assertIn("most_repeated_questions", analysis)
        self.assertIn("exact_repeats", analysis)
        self.assertIn("semantic_repeats", analysis)
        self.assertIn("related_topics", analysis)
        self.assertIn("topic_recurrence", analysis)
        self.assertIn("study_priorities", analysis)
        # Diagnostic path still has sources
        sources = self.engine.get_source_questions(self.WS_A)
        self.assertEqual(len(sources), 10)

    def test_semantic_vs_related_separation(self):
        analysis = self.engine.get_pyq_analysis(self.WS_A)
        # CNN architecture vs parameter calculation must not appear as semantic/exact repeat of each other
        for g in analysis.get("semantic_repeats", []) + analysis.get("exact_repeats", []):
            texts = []
            if "exact_text" in g:
                texts.append(g["exact_text"].lower())
            for oq in g.get("original_questions", []):
                texts.append(oq.get("text", "").lower())
            blob = " || ".join(texts)
            if "cnn architecture" in blob and "parameter" in blob and "architecture" in blob:
                # If both concepts somehow grouped, fail when one is calc-only paired as repeat of architecture-only
                if "explain cnn architecture" in blob and "calculate the number of parameters" in blob:
                    self.fail(f"Architecture vs parameter calculation falsely grouped as repeat: {blob}")


class TestPriorityFormula(unittest.TestCase):
    def test_scores_vary_with_evidence(self):
        low, _ = calculate_deterministic_priority_score(1, 1, 0, 5, 2020, 2026, 0, 0.1)
        high, _ = calculate_deterministic_priority_score(5, 4, 3, 10, 2025, 2026, 2, 0.9)
        self.assertLess(low, high)
        self.assertLess(low, 60)
        self.assertGreater(high, 70)


class TestVariableAggregationRegression(unittest.TestCase):
    """Tests 15–16: arbitrary question counts aggregate correctly (no hardcoded 15)."""

    def test_15_and_16_variable_counts(self):
        store = VectorStore()
        ingest = DynamicIngestPipeline(vector_store=store)
        engine = PYQIntelligenceEngine(vector_store=store)
        ws_db = WorkspaceDB()
        ws_id = "ws-var-agg-50"
        store.delete_by_workspace(ws_id)
        ws = ws_db.get_or_create(ws_id, subject="Algorithms")

        # Reuse battle-tested synthetic PDF helper from variable-count suite
        from tests.test_variable_question_counts import create_synthetic_pdf

        counts = [5, 10, 15, 20]
        try:
            for idx, count in enumerate(counts):
                path = create_synthetic_pdf(count, f"test_agg_{idx}_{count}q.pdf")
                metas = ingest.parse_pyq_pdf(path, ws)
                self.assertEqual(len(metas), count, f"Expected {count} questions, got {len(metas)}")

            analysis = engine.get_pyq_analysis(ws_id)
            self.assertEqual(analysis["total_papers"], 4)
            self.assertEqual(analysis["total_valid_questions"], 50)
            self.assertEqual(analysis.get("extracted_questions"), [])
        finally:
            store.delete_by_workspace(ws_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
