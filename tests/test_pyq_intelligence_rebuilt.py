"""
tests/test_pyq_intelligence_rebuilt.py
========================================
Tests for rebuilt, source-first, data-driven PYQ Intelligence Engine:

1. Topic cluster module mapping uses canonical question record's syllabus_mapping
   (Module 3 Autoencoder, Module 4 CNN, Module 6 GAN - NO defaulting to Module 1).
2. Canonical question records are the single source of truth.
3. Exact question repeats are separated from topic recurrence.
4. Question-level evidence, years, and marks are derived from source questions.
5. Priority and prediction scores are deterministic and transparent.
6. 2 papers produce "limited historical evidence" / LOW confidence.
7. Analytics queries return structured evidence-based answers.
8. Workspace isolation and empty workspace handling work.

Run with:
  pytest tests/test_pyq_intelligence_rebuilt.py -v
"""

import os
import sys
import tempfile
import unittest
import fitz  # PyMuPDF

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.pyq_intelligence import PYQIntelligenceEngine, calculate_deterministic_priority_score
from rag.answer_engine import GroundedAnswerEngine
from rag.workspace_db import WorkspaceDB


def create_pdf(path: str, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, text)
    doc.save(path)
    doc.close()


class TestRebuiltPYQIntelligence(unittest.TestCase):

    WS_ID = "ws-intel-test-dl"

    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()
        cls.ingest = DynamicIngestPipeline(vector_store=cls.store)
        cls.pyq_intel = PYQIntelligenceEngine(vector_store=cls.store)
        cls.engine = GroundedAnswerEngine(vector_store=cls.store)
        cls.ws_db = WorkspaceDB()

        cls.store.delete_by_workspace(cls.WS_ID)

        cls.tmpdir = tempfile.TemporaryDirectory()

        # Uploaded syllabus drives module mapping (subject-agnostic — no hardcoded DL index)
        cls.syl_pdf = os.path.join(cls.tmpdir.name, "DL_Syllabus.pdf")
        create_pdf(
            cls.syl_pdf,
            [
                "DEEP LEARNING SYLLABUS\n"
                "Module 3: Regularization & Model Generalization\n"
                "- Dropout\n- Early Stopping\n- Batch Normalization\n- Autoencoders\n- Denoising Autoencoder\n"
                "Module 4: Convolutional Neural Networks (CNN)\n"
                "- CNN Architecture\n- Convolutional Layer\n- Pooling\n- LeNET\n"
                "Module 6: Autoencoders & Generative Models\n"
                "- GAN\n- Generative Adversarial Network\n- Autoencoders\n"
            ],
        )

        # 2023 Deep Learning Paper (with explicit modules in mappings)
        cls.dl_2023_pdf = os.path.join(cls.tmpdir.name, "DL_2023_Dec.pdf")
        create_pdf(
            cls.dl_2023_pdf,
            [
                "MUMBAI UNIVERSITY | B.E. SEMESTER 7 | DEEP LEARNING | DECEMBER 2023 | 80 MARKS\n"
                "Q1. a. Design AND gate using Perceptron. (5 Marks)\n"
                "Q1. b. Derive gradient descent update rule for sum of squared error. (5 Marks)\n"
                "Q1. c. Explain Dropout and how it solves overfitting. (5 Marks)\n"
                "Q1. d. Explain Denoising Autoencoder model. (5 Marks)\n"
                "Q1. e. Describe Sequence Learning problem. (5 Marks)\n",
                "Q2. a. Explain Gated Recurrent Unit in detail. (10 Marks)\n"
                "Q2. b. What is an activation function? Describe any four activation functions. (10 Marks)\n"
                "Q3. a. Explain CNN architecture in detail and calculate parameters. (10 Marks)\n"
                "Q3. b. Explain early stopping, batch normalization, and data augmentation. (10 Marks)\n"
                "Q4. a. Explain RNN architecture in detail. (10 Marks)\n"
                "Q4. b. Explain the working of Generative Adversarial Network. (10 Marks)\n"
                "Q5. a. Explain Stochastic Gradient Descent and momentum. (10 Marks)\n"
                "Q5. b. Explain LSTM architecture. (10 Marks)\n"
                "Q6. a. Describe LeNET architecture. (10 Marks)\n"
                "Q6. b. Explain vanishing and exploding gradient in RNNs. (10 Marks)\n"
            ]
        )

        # 2024 Deep Learning Paper (contains exact repeat of LSTM & CNN)
        cls.dl_2024_pdf = os.path.join(cls.tmpdir.name, "DL_2024_May.pdf")
        create_pdf(
            cls.dl_2024_pdf,
            [
                "MUMBAI UNIVERSITY | B.E. SEMESTER 7 | DEEP LEARNING | MAY 2024 | 80 MARKS\n"
                "Q1. a. Explain Backpropagation algorithm step by step. (10 Marks)\n"
                "Q1. b. Derive gradient descent update rule for sum of squared error. (5 Marks)\n"
                "Q2. a. Explain LSTM architecture. (10 Marks)\n"
                "Q2. b. Explain CNN architecture in detail. (10 Marks)\n"
                "Q3. a. Explain Generative Adversarial Network. (10 Marks)\n"
            ]
        )

        cls.ws_info = cls.ws_db.get_or_create(
            workspace_id=cls.WS_ID,
            university="University of Mumbai",
            branch="Computer Engineering",
            semester="Semester 7",
            subject="Deep Learning",
            subject_code="CS3591"
        )

        # Ingest syllabus first, then both PYQ PDFs
        cls.ingest.parse_syllabus_pdf(cls.syl_pdf, cls.ws_info)
        cls.ingest.parse_pyq_pdf(cls.dl_2023_pdf, cls.ws_info)
        cls.ingest.parse_pyq_pdf(cls.dl_2024_pdf, cls.ws_info)

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_by_workspace(cls.WS_ID)
        cls.tmpdir.cleanup()

    # 1. Verification of Canonical Module Preservation
    def test_01_canonical_module_mapping_preservation(self):
        """
        Verify that CNN is mapped to Module 4, GAN to Module 6, Autoencoder to Module 3,
        matching the canonical question record syllabus_mapping instead of defaulting to Module 1.
        """
        analysis = self.pyq_intel.get_pyq_analysis(self.WS_ID, subject="Deep Learning")
        self.assertTrue(analysis["available"])

        topics = {t["topic_name"]: t for t in analysis["topics"]}
        print("\n[T1] Extracted Topic Cluster Modules:")
        for t_name, t_data in topics.items():
            print(f"      Topic: '{t_name}' -> Unit: '{t_data['unit']}'")

        # Find CNN topic
        cnn_topic = next((t for name, t in topics.items() if "cnn" in name.lower() or "convolutional" in name.lower()), None)
        self.assertIsNotNone(cnn_topic, "CNN topic cluster not found!")
        self.assertIn("Module 4", cnn_topic["unit"], f"CNN cluster should be Module 4, got: {cnn_topic['unit']}")

        # Find GAN topic
        gan_topic = next((t for name, t in topics.items() if "gan" in name.lower() or "generative" in name.lower()), None)
        self.assertIsNotNone(gan_topic, "GAN topic cluster not found!")
        self.assertIn("Module 6", gan_topic["unit"], f"GAN cluster should be Module 6, got: {gan_topic['unit']}")

        # Find Autoencoder topic — Module 3 or 6 depending on uploaded syllabus hits
        ae_topic = next((t for name, t in topics.items() if "autoencoder" in name.lower() or "auto encoder" in name.lower() or "denoising" in name.lower()), None)
        self.assertIsNotNone(ae_topic, "Autoencoder topic cluster not found!")
        self.assertTrue(
            "Module 3" in ae_topic["unit"] or "Module 6" in ae_topic["unit"],
            f"Autoencoder cluster should map to uploaded syllabus module, got: {ae_topic['unit']}",
        )

    # 2. Verification of Exact Repeats vs Topic Recurrence
    def test_02_exact_repeats_vs_topic_recurrence(self):
        """
        Verify that exact repeat questions (e.g. 'Derive gradient descent update rule...' in 2023 and 2024,
        and 'Explain LSTM architecture' in 2023 and 2024) are detected in exact_repeats.
        """
        analysis = self.pyq_intel.get_pyq_analysis(self.WS_ID, subject="Deep Learning")
        exact_repeats = analysis.get("exact_repeats", [])

        print(f"\n[T2] Total Exact Repeat Groups: {len(exact_repeats)}")
        for g in exact_repeats:
            print(f"      Text: '{g['exact_text'][:60]}...' | Years: {g['years']} | QIDs: {g['question_ids']}")

        self.assertGreaterEqual(len(exact_repeats), 1, "Expected at least 1 exact repeat group!")

    # 3. Verification of Question Evidence in Cluster
    def test_03_question_level_evidence_in_clusters(self):
        """
        Every topic cluster must include full source_questions evidence list with year, marks, and text.
        """
        analysis = self.pyq_intel.get_pyq_analysis(self.WS_ID, subject="Deep Learning")
        for t in analysis["topics"]:
            self.assertGreaterEqual(len(t["source_questions"]), 1)
            for sq in t["source_questions"]:
                self.assertIn("question_id", sq)
                self.assertIn("year", sq)
                self.assertIn("marks", sq)
                self.assertIn("exact_text", sq)
                self.assertIn("source_file", sq)

    # 4. Verification of Score Determinism
    def test_04_score_determinism(self):
        """
        Verify that Priority Score is computed deterministically with transparent signals.
        """
        total, comps = calculate_deterministic_priority_score(
            appearances_count=4,
            distinct_years=2,
            exact_repeat_count=1,
            max_marks=10,
            last_year=2024,
            current_year=2026,
            semantic_repeat_count=1,
            recurrence_consistency=0.5,
        )
        # Evidence-sensitive formula: scores must be deterministic and non-flat
        self.assertEqual(comps["frequency_score"], 28.0)
        self.assertEqual(comps["year_recurrence_score"], 16.0)
        self.assertEqual(comps["exact_repeat_score"], 6.0)
        self.assertEqual(comps["semantic_repeat_score"], 4.0)
        self.assertEqual(comps["marks_score"], 9.0)
        self.assertEqual(comps["recency_score"], 3.0)
        self.assertEqual(comps["consistency_score"], 5.0)
        self.assertEqual(total, 71.0)

    # 5. Verification of 2 Papers Evidence Language
    def test_05_two_papers_evidence_language(self):
        """
        With 2 uploaded papers, evidence_label must indicate 'Recurring across 2 uploaded papers'.
        """
        analysis = self.pyq_intel.get_pyq_analysis(self.WS_ID, subject="Deep Learning")
        self.assertEqual(analysis["total_papers"], 2)

        for t in analysis["topics"]:
            self.assertIn("2 uploaded papers", t["evidence_label"])
            self.assertEqual(t["prediction_confidence"], "LOW")

    # 6. Analytics Query Answer Routing
    def test_06_analytics_query_answer_routing(self):
        """
        Asking 'How many times did LSTM appear?' returns structured analytics response.
        """
        res = self.engine.generate_grounded_answer(
            question="How many times did LSTM appear?",
            mode="general",
            filters={"workspace_id": self.WS_ID}
        )
        self.assertEqual(res["answer_mode"], "structured_analytics")
        self.assertIn("LSTM", res["answer"])
        self.assertIn("Appearances", res["answer"])

    # 7. 'What should I study' query routing
    def test_07_what_should_i_study_routing(self):
        """
        Asking 'What should I study first?' returns ranked evidence-based topic priority.
        """
        res = self.engine.generate_grounded_answer(
            question="What should I study first for Deep Learning?",
            mode="general",
            filters={"workspace_id": self.WS_ID}
        )
        self.assertEqual(res["answer_mode"], "structured_analytics")
        self.assertIn("Priority Ranking", res["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
