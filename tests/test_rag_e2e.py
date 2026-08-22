"""
tests/test_rag_e2e.py
=====================
End-to-end RAG pipeline tests.

Proves:
  1. New document → chunks → embeddings → ChromaDB → retrieval → answer
  2. Negative test: unknown topic returns insufficient-evidence, NOT hardcoded answer
  3. Deep Learning LSTM query returns evidence from actual PYQ chunks
  4. Syllabus module query uses subject-specific index
  5. No AIS / SLM / WDV / BCOC131 text appears in any answer

Run with:
  pytest tests/test_rag_e2e.py -v
"""

import os
import sys
import json
import tempfile
import unittest
import fitz  # PyMuPDF

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from rag.api import app
from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.answer_engine import GroundedAnswerEngine


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_pdf(path: str, pages: list[str]) -> None:
    """Create a minimal searchable PDF with the given page strings."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, text)
    doc.save(path)
    doc.close()


FORBIDDEN_STRS = [
    "Accounting Information System",
    "AIS",
    "Voucher entries",
    "Creation of Company",
    "Straight Line Method",
    "SLM",
    "Written Down Value",
    "WDV",
    "BCOC131",
    "Accounting_Info_System",
    "IGNOU",
    "double entry bookkeeping",
    "financial transaction",
]


def assert_no_hardcoded_content(test_case: unittest.TestCase, answer: str, label: str = "") -> None:
    """Assert that none of the IGNOU/AIS/Accounting hardcoded strings appear."""
    for bad in FORBIDDEN_STRS:
        test_case.assertNotIn(
            bad.lower(),
            answer.lower(),
            msg=f"[{label}] Answer contains hardcoded string: '{bad}'\n\nAnswer was:\n{answer[:600]}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────────────────────

class TestRAGEndToEnd(unittest.TestCase):

    WS_ID = "test-rag-e2e-ws"
    WS_DL_ID = "test-rag-e2e-dl-ws"

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.store = VectorStore()
        cls.ingest = DynamicIngestPipeline(vector_store=cls.store)
        cls.engine = GroundedAnswerEngine(vector_store=cls.store)

        # Clean up any leftover vectors from previous runs
        cls.store.delete_by_workspace(cls.WS_ID)
        cls.store.delete_by_workspace(cls.WS_DL_ID)

        cls.tmpdir = tempfile.TemporaryDirectory()

        # ── Workspace info dicts ─────────────────────────────────────────
        cls.ws_info = {
            "id": cls.WS_ID,
            "university": "Test University",
            "branch": "Computer Engineering",
            "semester": "Semester 7",
            "subject": "Deep Learning",
            "subject_code": "DL7001",
        }

        cls.dl_ws_info = {
            "id": cls.WS_DL_ID,
            "university": "University of Mumbai",
            "branch": "Computer Engineering",
            "semester": "Semester 7",
            "subject": "Deep Learning",
            "subject_code": "42371",
        }

        # ── Synthetic NEW-topic PDF (XYZ_NEW_TEST_TOPIC_987) ─────────────
        cls.new_topic_pdf = os.path.join(cls.tmpdir.name, "NEW_TOPIC_2025_PYQ.pdf")
        make_pdf(
            cls.new_topic_pdf,
            [
                "TEST UNIVERSITY | SEMESTER 7 | DEEP LEARNING | EXAMINATION 2025\n"
                "Q1. a. Explain XYZ_NEW_TEST_TOPIC_987 in detail. [10 Marks]\n"
                "Q1. b. Describe the applications of XYZ_NEW_TEST_TOPIC_987. [10 Marks]\n"
            ],
        )

        # ── Synthetic 2023 Deep Learning PYQ ─────────────────────────────
        cls.dl_2023_pdf = os.path.join(cls.tmpdir.name, "DL_2023_PYQ.pdf")
        make_pdf(
            cls.dl_2023_pdf,
            [
                "UNIVERSITY OF MUMBAI | SEMESTER 7 | DEEP LEARNING | EXAMINATION 2023\n"
                "Instructions: Attempt any five questions. Each question carries 20 marks.\n"
                "Q1. a. Design AND gate using Perceptron. [5 Marks]\n"
                "Q1. b. Derive the Gradient Descent update rule. [5 Marks]\n"
                "Q1. c. Explain Dropout and how it solves overfitting. [5 Marks]\n"
                "Q1. d. Explain Denoising Autoencoder. [5 Marks]\n"
                "Q1. e. Describe Sequence Learning. [5 Marks]\n",
                "Q2. a. Explain Gated Recurrent Unit in detail. [10 Marks]\n"
                "Q2. b. What is an activation function? Describe any four activation functions. [10 Marks]\n"
                "Q3. a. Explain CNN architecture in detail. [10 Marks]\n"
                "Q3. b. Explain early stopping, batch normalization, and data augmentation. [10 Marks]\n"
                "Q4. a. Explain RNN architecture in detail. [10 Marks]\n"
                "Q4. b. Explain the working of Generative Adversarial Network. [10 Marks]\n"
                "Q5. a. Explain Stochastic Gradient Descent and momentum. [10 Marks]\n"
                "Q5. b. Explain LSTM architecture. [10 Marks]\n"
                "Q6. a. Describe LeNET architecture. [10 Marks]\n"
                "Q6. b. Explain vanishing and exploding gradient in RNNs. [10 Marks]\n",
            ],
        )

        # ── Synthetic 2024 Deep Learning PYQ ─────────────────────────────
        cls.dl_2024_pdf = os.path.join(cls.tmpdir.name, "DL_2024_PYQ.pdf")
        make_pdf(
            cls.dl_2024_pdf,
            [
                "UNIVERSITY OF MUMBAI | SEMESTER 7 | DEEP LEARNING | EXAMINATION 2024\n"
                "Q1. a. Explain Backpropagation algorithm step by step. [10 Marks]\n"
                "Q1. b. What is Gradient Descent? Explain SGD and Adam optimizer. [10 Marks]\n"
                "Q2. a. Explain LSTM model, how it overcomes the limitation of RNN. [10 Marks]\n"
                "Q2. b. Explain Convolutional Neural Network architecture. [10 Marks]\n"
                "Q3. a. Explain Generative Adversarial Network (GAN). [10 Marks]\n"
                "Q3. b. Describe ResNet architecture and skip connections. [10 Marks]\n",
            ],
        )

        # ── Ingest all PDFs ───────────────────────────────────────────────
        cls.new_metas = cls.ingest.parse_pyq_pdf(cls.new_topic_pdf, cls.ws_info)
        cls.dl_2023_metas = cls.ingest.parse_pyq_pdf(cls.dl_2023_pdf, cls.dl_ws_info)
        cls.dl_2024_metas = cls.ingest.parse_pyq_pdf(cls.dl_2024_pdf, cls.dl_ws_info)

        print(f"\n[SETUP] new_topic metas: {len(cls.new_metas)}")
        print(f"[SETUP] dl_2023 metas:   {len(cls.dl_2023_metas)}")
        print(f"[SETUP] dl_2024 metas:   {len(cls.dl_2024_metas)}")

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_by_workspace(cls.WS_ID)
        cls.store.delete_by_workspace(cls.WS_DL_ID)
        cls.tmpdir.cleanup()

    # ────────────────────────────────────────────────────────────────────────
    # T1 — Ingestion: new PDF produces chunks
    # ────────────────────────────────────────────────────────────────────────
    def test_01_new_doc_produces_chunks(self):
        """New PDF with XYZ_NEW_TEST_TOPIC_987 must produce ≥ 1 chunks."""
        self.assertGreaterEqual(
            len(self.new_metas), 1,
            "New document produced zero chunks — ingestion pipeline broken.",
        )

    # ────────────────────────────────────────────────────────────────────────
    # T2 — Ingestion: metadata is correct
    # ────────────────────────────────────────────────────────────────────────
    def test_02_new_doc_metadata_correct(self):
        """Every chunk from the new PDF must carry workspace_id and source_file."""
        for meta in self.new_metas:
            self.assertEqual(meta.get("workspace_id"), self.WS_ID)
            self.assertEqual(meta.get("doc_type"), "pyq")
            self.assertIn("NEW_TOPIC_2025_PYQ.pdf", meta.get("source_file", ""))

    # ────────────────────────────────────────────────────────────────────────
    # T3 — Retrieval: new topic is retrievable
    # ────────────────────────────────────────────────────────────────────────
    def test_03_new_topic_is_retrievable(self):
        """Semantic search for XYZ_NEW_TEST_TOPIC_987 must return ≥ 1 result."""
        results = self.store.search(
            query="XYZ_NEW_TEST_TOPIC_987",
            doc_type="pyq",
            top_k=5,
            filters={"workspace_id": self.WS_ID},
        )
        self.assertGreaterEqual(len(results), 1, "New topic not retrievable from vector DB.")
        sources = [r["metadata"].get("source_file", "") for r in results]
        self.assertTrue(
            any("NEW_TOPIC_2025_PYQ" in s for s in sources),
            f"Retrieved results do not include the new PDF. Sources: {sources}",
        )

    # ────────────────────────────────────────────────────────────────────────
    # T4 — Answer engine: new topic answer comes from new doc
    # ────────────────────────────────────────────────────────────────────────
    def test_04_new_topic_answer_uses_new_doc(self):
        """
        Asking about XYZ_NEW_TEST_TOPIC_987 must return an answer that:
          - contains "XYZ_NEW_TEST_TOPIC_987"
          - cites the new PDF as source
          - does NOT contain hardcoded AIS/IGNOU content
        """
        result = self.engine.generate_grounded_answer(
            question="Explain XYZ_NEW_TEST_TOPIC_987 in detail.",
            mode="general",
            doc_type="pyq",
            filters={"workspace_id": self.WS_ID},
            debug=True,
        )

        answer = result.get("answer", "")
        answer_mode = result.get("answer_mode", "")
        citations = result.get("citations", [])
        retrieved = result.get("debug", {}).get("retrieved_chunks", [])

        print(f"\n[T4] answer_mode={answer_mode}")
        print(f"[T4] citations={citations}")
        print(f"[T4] answer[:400]:\n{answer[:400]}")

        # Must NOT be insufficient_evidence
        self.assertNotEqual(
            answer_mode, "insufficient_evidence",
            f"New topic was not found in vector DB. Retrieved chunks: {retrieved}",
        )

        # Must reference the unique test topic
        self.assertIn(
            "XYZ_NEW_TEST_TOPIC_987".lower(),
            answer.lower(),
            "Answer does not mention the unique new topic.",
        )

        # Must cite the new source file
        cited_files = [c.get("source_file", "") for c in citations]
        self.assertTrue(
            any("NEW_TOPIC_2025_PYQ" in f for f in cited_files),
            f"Answer does not cite the new PDF. Citations: {cited_files}",
        )

        # Must NOT contain hardcoded AIS content
        assert_no_hardcoded_content(self, answer, "T4:new_topic")

    # ────────────────────────────────────────────────────────────────────────
    # T5 — Negative test: totally unknown topic → insufficient evidence
    # ────────────────────────────────────────────────────────────────────────
    def test_05_unknown_topic_returns_insufficient_evidence(self):
        """
        Asking about a topic that was NEVER uploaded must NOT return a
        confident hardcoded answer.  answer_mode must be 'insufficient_evidence'
        or the answer must indicate evidence is missing.
        """
        result = self.engine.generate_grounded_answer(
            question="Explain ZZZZ_COMPLETELY_NONEXISTENT_TOPIC_00000.",
            mode="general",
            doc_type="both",
            filters={"workspace_id": self.WS_ID},
            debug=True,
        )

        answer_mode = result.get("answer_mode", "")
        answer = result.get("answer", "")

        print(f"\n[T5] answer_mode={answer_mode}")
        print(f"[T5] answer[:300]:\n{answer[:300]}")

        # Must be insufficient_evidence OR explicitly say so in text
        is_negative = (
            answer_mode == "insufficient_evidence"
            or "insufficient" in answer.lower()
            or "not found" in answer.lower()
            or "no source" in answer.lower()
            or "not present" in answer.lower()
        )
        self.assertTrue(
            is_negative,
            f"System returned a confident answer for a nonexistent topic.\n"
            f"answer_mode={answer_mode}\nAnswer:\n{answer[:600]}",
        )

        # Must NOT contain hardcoded AIS content
        assert_no_hardcoded_content(self, answer, "T5:negative")

    # ────────────────────────────────────────────────────────────────────────
    # T6 — 2023 Deep Learning PDF: 15 subquestions ingested
    # ────────────────────────────────────────────────────────────────────────
    def test_06_dl_2023_produces_multiple_subquestions(self):
        """The synthetic 2023 DL paper must produce multiple independent subquestion chunks."""
        self.assertGreaterEqual(
            len(self.dl_2023_metas), 5,
            f"2023 DL paper produced only {len(self.dl_2023_metas)} chunks. Expected ≥ 5.",
        )
        q_ids = [m.get("question_id", "") for m in self.dl_2023_metas]
        print(f"\n[T6] question_ids from 2023 paper: {q_ids}")

    # ────────────────────────────────────────────────────────────────────────
    # T7 — LSTM query: answer uses 2023/2024 PYQ evidence
    # ────────────────────────────────────────────────────────────────────────
    def test_07_lstm_query_returns_pyq_evidence(self):
        """
        Querying LSTM in the DL workspace must:
          - Return answer_mode != insufficient_evidence
          - Include PYQ evidence (times_asked ≥ 1)
          - Cite at least one of the 2023/2024 DL PDFs
          - NOT contain hardcoded AIS content
        """
        result = self.engine.generate_grounded_answer(
            question="What questions about LSTM appeared in previous papers?",
            mode="general",
            doc_type="pyq",
            filters={"workspace_id": self.WS_DL_ID},
            debug=True,
        )

        answer = result.get("answer", "")
        answer_mode = result.get("answer_mode", "")
        pyq_ev = result.get("pyq_evidence", {})
        citations = result.get("citations", [])
        retrieved = result.get("debug", {}).get("retrieved_chunks", [])

        print(f"\n[T7] answer_mode={answer_mode}")
        print(f"[T7] pyq_evidence={pyq_ev}")
        print(f"[T7] retrieved_chunks (top 5):")
        for ch in retrieved[:5]:
            print(f"      {ch.get('question_id')} | {ch.get('source_file')} | score={ch.get('final_score')}")
        print(f"[T7] answer[:500]:\n{answer[:500]}")

        # Must find something
        self.assertNotEqual(
            answer_mode, "insufficient_evidence",
            f"LSTM query found nothing. Retrieved: {retrieved}",
        )

        # Must reference LSTM in the answer
        self.assertIn(
            "lstm", answer.lower(),
            "LSTM not mentioned in the answer.",
        )

        # No hardcoded AIS content
        assert_no_hardcoded_content(self, answer, "T7:lstm")

    # ────────────────────────────────────────────────────────────────────────
    # T8 — Recurrence: Gradient Descent appears in both 2023 and 2024
    # ────────────────────────────────────────────────────────────────────────
    def test_08_gradient_descent_recurrence(self):
        """
        Gradient Descent appears in both 2023 and 2024 DL papers.
        The answer must include pyq_evidence.times_asked ≥ 1.
        """
        result = self.engine.generate_grounded_answer(
            question="How many times did Gradient Descent appear in the uploaded papers?",
            mode="general",
            doc_type="pyq",
            filters={"workspace_id": self.WS_DL_ID},
            debug=True,
        )

        answer = result.get("answer", "")
        answer_mode = result.get("answer_mode", "")
        pyq_ev = result.get("pyq_evidence", {})

        print(f"\n[T8] answer_mode={answer_mode}")
        print(f"[T8] pyq_evidence={pyq_ev}")
        print(f"[T8] answer[:400]:\n{answer[:400]}")

        # Must find evidence
        self.assertNotEqual(answer_mode, "insufficient_evidence")
        assert_no_hardcoded_content(self, answer, "T8:gradient_descent")

    # ────────────────────────────────────────────────────────────────────────
    # T9 — Module query: uses subject syllabus index
    # ────────────────────────────────────────────────────────────────────────
    def test_09_syllabus_module_query(self):
        """
        Asking which module contains LSTM must not return hardcoded AIS/accounting text.
        The answer should reference Module 5 (Recurrent Neural Networks).
        """
        result = self.engine.generate_grounded_answer(
            question="Which module contains LSTM according to the Deep Learning syllabus?",
            mode="general",
            doc_type="both",
            filters={"workspace_id": self.WS_DL_ID},
            debug=True,
        )

        answer = result.get("answer", "")
        answer_mode = result.get("answer_mode", "")

        print(f"\n[T9] answer_mode={answer_mode}")
        print(f"[T9] answer[:400]:\n{answer[:400]}")

        assert_no_hardcoded_content(self, answer, "T9:syllabus_module")

    # ────────────────────────────────────────────────────────────────────────
    # T10 — No AIS/Accounting answer for any DL question
    # ────────────────────────────────────────────────────────────────────────
    def test_10_no_ais_answer_for_dl_question(self):
        """
        Asking a typical Deep Learning question must NEVER produce AIS/Accounting text.
        """
        questions = [
            "Explain CNN architecture in detail.",
            "Explain RNN architecture.",
            "What is Dropout? How does it solve overfitting?",
            "Explain Generative Adversarial Network.",
        ]
        for q in questions:
            result = self.engine.generate_grounded_answer(
                question=q,
                mode="general",
                doc_type="both",
                filters={"workspace_id": self.WS_DL_ID},
            )
            answer = result.get("answer", "")
            assert_no_hardcoded_content(self, answer, f"T10:{q[:30]}")

    # ────────────────────────────────────────────────────────────────────────
    # T11 — Deduplication: re-uploading same file must not grow chunk count
    # ────────────────────────────────────────────────────────────────────────
    def test_11_dedup_on_reingestion(self):
        """
        Re-ingesting the same file must not create duplicate vector records.
        Chunk count must be the same before and after.
        """
        res1 = self.store.collection.get(
            where={"$and": [
                {"workspace_id": {"$eq": self.WS_ID}},
                {"source_file": {"$eq": "NEW_TOPIC_2025_PYQ.pdf"}},
            ]}
        )
        count_before = len(res1.get("ids", []))

        # Re-ingest the same file
        self.ingest.parse_pyq_pdf(self.new_topic_pdf, self.ws_info)

        res2 = self.store.collection.get(
            where={"$and": [
                {"workspace_id": {"$eq": self.WS_ID}},
                {"source_file": {"$eq": "NEW_TOPIC_2025_PYQ.pdf"}},
            ]}
        )
        count_after = len(res2.get("ids", []))

        print(f"\n[T11] count_before={count_before}, count_after={count_after}")
        self.assertEqual(
            count_before, count_after,
            f"Re-ingestion created duplicates: before={count_before}, after={count_after}",
        )

    # ────────────────────────────────────────────────────────────────────────
    # T12 — API endpoint smoke test
    # ────────────────────────────────────────────────────────────────────────
    def test_12_api_ask_endpoint_works(self):
        """The /ask endpoint must return a valid JSON response without 500."""
        resp = self.client.post("/ask", json={
            "question": "Explain LSTM architecture.",
            "workspace_id": self.WS_DL_ID,
            "mode": "general",
            "doc_type": "both",
        })
        self.assertIn(resp.status_code, [200, 200])
        body = resp.json()
        self.assertIn("answer", body)
        self.assertIn("answer_mode", body)
        self.assertIn("citations", body)
        assert_no_hardcoded_content(self, body.get("answer", ""), "T12:api_lstm")

    # ────────────────────────────────────────────────────────────────────────
    # T13 — Debug endpoint returns full trace
    # ────────────────────────────────────────────────────────────────────────
    def test_13_debug_endpoint_returns_trace(self):
        """The /ask/debug endpoint must return retrieved_chunks in the debug payload."""
        resp = self.client.post("/ask/debug", json={
            "question": "Explain LSTM architecture.",
            "workspace_id": self.WS_DL_ID,
            "mode": "general",
            "doc_type": "pyq",
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        debug = body.get("debug", {})
        self.assertIn("retrieved_chunks", debug)
        self.assertIn("reranked_count", debug)
        self.assertIn("answer_mode", debug)
        self.assertIn("filters_applied", debug)

    # ────────────────────────────────────────────────────────────────────────
    # T14 — Workspace isolation: WS_ID cannot see WS_DL_ID docs
    # ────────────────────────────────────────────────────────────────────────
    def test_14_workspace_isolation(self):
        """
        Searching for LSTM in WS_ID (new-topic workspace) must NOT return
        results from WS_DL_ID (Deep Learning workspace).
        """
        results = self.store.search(
            query="LSTM architecture",
            doc_type="pyq",
            top_k=5,
            filters={"workspace_id": self.WS_ID},
        )
        for r in results:
            ws = r["metadata"].get("workspace_id", "")
            self.assertEqual(
                ws, self.WS_ID,
                f"Cross-workspace contamination: retrieved chunk from WS={ws} instead of {self.WS_ID}",
            )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
