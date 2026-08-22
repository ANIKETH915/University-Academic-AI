"""
tests/test_pdf_only_architecture.py
====================================
Comprehensive tests verifying PDF-Upload-Only architecture:

1. Clean workspace starts with 0 questions, 0 papers, 0 topics, 0 vectors.
2. Workspace creation ONLY creates workspace metadata (starts completely empty).
3. PDF upload is the ONLY entry point for populating vector store and PYQ analytics.
4. Uploading 2023 Deep Learning paper extracts exactly 15 subquestions (Q1(a)...Q6(b)), excluding parent Q1-Q6 containers.
5. Every uploaded record receives active workspace metadata.
6. Re-uploading the same PDF replaces old records cleanly without accumulating duplicate vectors.
7. RAG query answers using evidence from uploaded PDF.
8. Single-paper mode is correctly reported when 1 paper is uploaded.
9. Multi-paper recurrence works when a 2nd paper is uploaded.
10. Negative test: Empty workspace returns insufficient_evidence / no PYQs uploaded.
11. Workspace isolation: Workspace A data NEVER leaks into Workspace B.

Run with:
  pytest tests/test_pdf_only_architecture.py -v
"""

import os
import sys
import tempfile
import unittest
import fitz  # PyMuPDF

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from rag.api import app
from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.answer_engine import GroundedAnswerEngine
from rag.pyq_intelligence import PYQIntelligenceEngine
from rag.workspace_db import WorkspaceDB


def create_pdf(path: str, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, text)
    doc.save(path)
    doc.close()


class TestPDFOnlyArchitecture(unittest.TestCase):

    WS_CLEAN_A = "ws-clean-arch-dl"
    WS_CLEAN_B = "ws-clean-arch-cn-empty"

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.store = VectorStore()
        cls.ingest = DynamicIngestPipeline(vector_store=cls.store)
        cls.engine = GroundedAnswerEngine(vector_store=cls.store)
        cls.pyq_intel = PYQIntelligenceEngine(vector_store=cls.store)
        cls.ws_db = WorkspaceDB()

        # Clean workspace vectors
        cls.store.delete_by_workspace(cls.WS_CLEAN_A)
        cls.store.delete_by_workspace(cls.WS_CLEAN_B)

        cls.tmpdir = tempfile.TemporaryDirectory()

        # Generate sample 2023 Deep Learning paper (15 subquestions)
        cls.dl_2023_pdf = os.path.join(cls.tmpdir.name, "MU_DL_2023_Dec.pdf")
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
                "Q5. b. Explain LSTM architecture in detail. (10 Marks)\n"
                "Q6. a. Describe LeNET architecture. (10 Marks)\n"
                "Q6. b. Explain vanishing and exploding gradient in RNNs. (10 Marks)\n"
            ]
        )

        # Generate sample 2024 Deep Learning paper
        cls.dl_2024_pdf = os.path.join(cls.tmpdir.name, "MU_DL_2024_May.pdf")
        create_pdf(
            cls.dl_2024_pdf,
            [
                "MUMBAI UNIVERSITY | B.E. SEMESTER 7 | DEEP LEARNING | MAY 2024 | 80 MARKS\n"
                "Q1. a. Explain Backpropagation algorithm step by step. (10 Marks)\n"
                "Q1. b. Explain Gradient Descent optimization techniques. (10 Marks)\n"
                "Q2. a. Explain LSTM model and how it overcomes RNN limitations. (10 Marks)\n"
                "Q2. b. Explain Convolutional Neural Network architecture. (10 Marks)\n"
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_by_workspace(cls.WS_CLEAN_A)
        cls.store.delete_by_workspace(cls.WS_CLEAN_B)
        cls.tmpdir.cleanup()

    # 1. Clean workspace starts with zero questions, zero vectors
    def test_01_new_workspace_starts_empty(self):
        ws_info = self.ws_db.get_or_create(
            workspace_id=self.WS_CLEAN_B,
            university="University of Mumbai",
            branch="Computer Engineering",
            semester="Semester 5",
            subject="Computer Networks",
            subject_code="CSC501"
        )
        self.assertEqual(len(ws_info["syllabus_files"]), 0)
        self.assertEqual(len(ws_info["pyq_files"]), 0)

        # PYQ Analysis must report zero
        analysis = self.pyq_intel.get_pyq_analysis(
            workspace_id=self.WS_CLEAN_B,
            subject="Computer Networks",
            semester="Semester 5"
        )
        self.assertFalse(analysis["available"])
        self.assertEqual(analysis["total_questions_analyzed"], 0)
        self.assertEqual(analysis["unique_topic_clusters"], 0)
        self.assertEqual(len(analysis["topics"]), 0)
        self.assertEqual(len(analysis["extracted_questions"]), 0)

        # Vectors must be 0
        vectors = self.store.collection.get(where={"workspace_id": {"$eq": self.WS_CLEAN_B}})
        self.assertEqual(len(vectors.get("ids", [])), 0)

    # 2. PDF upload creates exact 15 subquestion records
    def test_02_pdf_ingest_creates_15_subquestions(self):
        ws_info = self.ws_db.get_or_create(
            workspace_id=self.WS_CLEAN_A,
            university="University of Mumbai",
            branch="Computer Engineering",
            semester="Semester 7",
            subject="Deep Learning",
            subject_code="CS3591"
        )

        metas = self.ingest.parse_pyq_pdf(self.dl_2023_pdf, ws_info)
        self.assertEqual(len(metas), 15, f"Expected 15 extracted records, got {len(metas)}")

        q_ids = [m["question_id"] for m in metas]
        expected_ids = [
            "Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q1(e)",
            "Q2(a)", "Q2(b)",
            "Q3(a)", "Q3(b)",
            "Q4(a)", "Q4(b)",
            "Q5(a)", "Q5(b)",
            "Q6(a)", "Q6(b)"
        ]
        self.assertEqual(q_ids, expected_ids)

        # Parent Q1-Q6 must not exist in q_ids
        for parent in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
            self.assertNotIn(parent, q_ids)

    # 3. Vector database receives correct metadata
    def test_03_vectors_have_strict_workspace_metadata(self):
        res = self.store.collection.get(where={"workspace_id": {"$eq": self.WS_CLEAN_A}})
        ids = res.get("ids", [])
        metas = res.get("metadatas", [])

        self.assertEqual(len(ids), 15)
        for meta in metas:
            self.assertEqual(meta["workspace_id"], self.WS_CLEAN_A)
            self.assertEqual(meta["university"], "University of Mumbai")
            self.assertEqual(meta["subject"], "Deep Learning")
            self.assertEqual(meta["semester"], "Semester 7")
            self.assertIn("MU_DL_2023_Dec.pdf", meta["source_file"])
            self.assertTrue(meta["question_id"].startswith("Q"))

    # 4. Duplicate upload replaces old vectors without duplicating
    def test_04_reingestion_replaces_vectors_without_duplicates(self):
        ws_info = self.ws_db.get_by_id(self.WS_CLEAN_A)
        # Re-ingest exact same file
        self.ingest.parse_pyq_pdf(self.dl_2023_pdf, ws_info)

        res = self.store.collection.get(where={"workspace_id": {"$eq": self.WS_CLEAN_A}})
        self.assertEqual(len(res.get("ids", [])), 15, "Re-ingestion accumulated duplicate vectors!")

    # 5. RAG query answers strictly from uploaded PDF
    def test_05_rag_answers_from_uploaded_pdf(self):
        result = self.engine.generate_grounded_answer(
            question="Explain LSTM architecture in detail.",
            mode="general",
            doc_type="pyq",
            filters={"workspace_id": self.WS_CLEAN_A},
            debug=True
        )
        self.assertNotEqual(result["answer_mode"], "insufficient_evidence")
        self.assertIn("lstm", result["answer"].lower())
        self.assertTrue(any("MU_DL_2023_Dec.pdf" in c.get("source_file", "") for c in result["citations"]))

    # 6. Single-paper mode is reported accurately
    def test_06_single_paper_mode(self):
        analysis = self.pyq_intel.get_pyq_analysis(
            workspace_id=self.WS_CLEAN_A,
            subject="Deep Learning",
            semester="Semester 7"
        )
        self.assertTrue(analysis["available"])
        self.assertTrue(analysis["single_paper_mode"])
        self.assertEqual(analysis["total_papers"], 1)

    # 7. Uploading a second paper enables multi-paper analysis
    def test_07_multi_paper_ingestion(self):
        ws_info = self.ws_db.get_by_id(self.WS_CLEAN_A)
        self.ingest.parse_pyq_pdf(self.dl_2024_pdf, ws_info)

        analysis = self.pyq_intel.get_pyq_analysis(
            workspace_id=self.WS_CLEAN_A,
            subject="Deep Learning",
            semester="Semester 7"
        )
        self.assertTrue(analysis["available"])
        self.assertFalse(analysis["single_paper_mode"])
        self.assertEqual(analysis["total_papers"], 2)

    # 8. Negative test: Empty workspace returns insufficient_evidence
    def test_08_empty_workspace_returns_negative(self):
        result = self.engine.generate_grounded_answer(
            question="What questions appeared in Computer Networks?",
            mode="general",
            doc_type="both",
            filters={"workspace_id": self.WS_CLEAN_B},
            debug=True
        )
        self.assertEqual(result["answer_mode"], "insufficient_evidence")
        self.assertIn("insufficient", result["answer"].lower())

    # 9. Workspace isolation: Workspace A data NEVER leaks into Workspace B
    def test_09_workspace_isolation(self):
        search_res = self.store.search(
            query="Explain LSTM architecture",
            doc_type="pyq",
            top_k=10,
            filters={"workspace_id": self.WS_CLEAN_B}
        )
        self.assertEqual(len(search_res), 0, "Workspace B retrieved chunks from Workspace A!")

    # 10. Unique upload-only phrase is retrievable
    def test_10_unique_upload_only_phrase_retrievable(self):
        ws_test_id = "ws-test-unique-phrase"
        self.store.delete_by_workspace(ws_test_id)
        ws_info = self.ws_db.get_or_create(
            workspace_id=ws_test_id,
            university="University of Mumbai",
            branch="Computer Engineering",
            semester="Semester 7",
            subject="Deep Learning"
        )
        unique_pdf = os.path.join(self.tmpdir.name, "Unique_Topic_Paper.pdf")
        unique_phrase = "XYZ_UPLOAD_ONLY_TOPIC_987654"
        create_pdf(
            unique_pdf,
            [
                f"MUMBAI UNIVERSITY | DEEP LEARNING 2024\n"
                f"Q1. a. Explain {unique_phrase} in detail. (10 Marks)\n"
            ]
        )
        metas = self.ingest.parse_pyq_pdf(unique_pdf, ws_info)
        self.assertGreater(len(metas), 0)

        # Retrieve phrase
        search_res = self.store.search(
            query=unique_phrase,
            doc_type="pyq",
            top_k=5,
            filters={"workspace_id": ws_test_id}
        )
        self.assertGreater(len(search_res), 0)
        self.assertIn(unique_phrase, search_res[0]["text"])
        self.assertEqual(search_res[0]["metadata"]["workspace_id"], ws_test_id)

        # Cleanup test workspace
        self.store.delete_by_workspace(ws_test_id)
        self.ws_db.delete_workspace(ws_test_id)

    # 11. Zero dummy workspaces & dummy vectors health assertion
    def test_11_zero_dummy_workspaces_and_vectors_after_cleanup(self):
        all_ws = self.ws_db.get_all()
        # Verify no demo workspaces exist in database
        demo_workspaces = [w for w in all_ws if w.get("is_demo") is True]
        self.assertEqual(len(demo_workspaces), 0, "Demo workspaces present in database!")

    # 12. Deleting document purges ChromaDB vectors
    def test_12_deleting_document_purges_vectors(self):
        ws_test_id = "ws-test-doc-purge"
        self.store.delete_by_workspace(ws_test_id)
        ws_info = self.ws_db.get_or_create(workspace_id=ws_test_id)

        purge_pdf = os.path.join(self.tmpdir.name, "Purge_Test_Paper.pdf")
        create_pdf(purge_pdf, ["Q1. a. Explain Perceptron training algorithm. (10 Marks)"])
        self.ingest.parse_pyq_pdf(purge_pdf, ws_info)

        # Vectors before purge
        v_before = self.store.collection.get(where={"workspace_id": {"$eq": ws_test_id}})
        self.assertGreater(len(v_before.get("ids", [])), 0)

        # Purge by source file
        filename = os.path.basename(purge_pdf)
        self.store.delete_by_source_file(filename)

        v_after = self.store.collection.get(where={"workspace_id": {"$eq": ws_test_id}})
        self.assertEqual(len(v_after.get("ids", [])), 0)

        self.ws_db.delete_workspace(ws_test_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)

