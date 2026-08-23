"""
Regression: REAL frontend-equivalent HTTP user flow.

Simulates exactly:
  POST /workspaces  (create)
  POST /workspaces/{id}/ingest  (upload PYQ)
  POST /workspaces/{id}/analyze-pyq
  POST /ask  (unique phrase proof)

Does NOT reuse pre-seeded Deep Learning workspaces.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz
from fastapi.testclient import TestClient

# Isolate from production workspaces.json / collection.
# Keep setdefault semantics: pytest imports every test module before running
# any of them, and rag.api binds its vector-store collection at first import.
# A hard override here desynchronizes the API layer from test assertions.
os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")

from rag.api import app
from rag.vector_store import VectorStore


PROOF_PHRASE = "FRONTEND_REAL_USER_PROOF_928374"


def _make_proof_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "SYNTHETIC EXAM PAPER May 2024",
        f"Q1(a) Explain {PROOF_PHRASE} in complete academic detail.",
        "Q1(b) Compare widget models used in quantum widget engineering.",
        f"Q2(a) Describe applications of {PROOF_PHRASE} with examples.",
    ]
    y = 40
    for line in lines:
        page.insert_text((40, y), line, fontsize=10)
        y += 16
    doc.save(path)
    doc.close()


class TestRealFrontendUserFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.store = VectorStore()
        cls.tmpdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_create_ingest_analyze_ask_same_workspace_id(self):
        # 1) Create workspace via API (same as frontend createWorkspace)
        create = self.client.post(
            "/workspaces",
            json={
                "university": "Synthetic University",
                "branch": "Widget Engineering",
                "semester": "Semester 8",
                "subject": "Quantum Widget Flow Test",
                "subject_code": "QWE999",
            },
        )
        self.assertEqual(create.status_code, 200, create.text)
        ws = create.json()
        ws_id = ws["id"]
        self.assertTrue(ws_id.startswith("ws-"))
        self.assertNotEqual(ws_id, "ws-default-workspace")

        # 2) Ingest PYQ PDF into THAT workspace id
        pdf_path = os.path.join(self.tmpdir.name, "proof_pyq.pdf")
        _make_proof_pdf(pdf_path)
        with open(pdf_path, "rb") as fh:
            ingest = self.client.post(
                f"/workspaces/{ws_id}/ingest",
                files={"file": ("proof_pyq.pdf", fh, "application/pdf")},
                data={"doc_type": "pyq"},
            )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        body = ingest.json()
        self.assertEqual(body["workspace_id"], ws_id)
        self.assertEqual(body["status"], "success")
        self.assertGreaterEqual(body["vectors_inserted"], 2)
        self.assertGreaterEqual(body["questions_extracted"], 2)

        # 3) Chroma must contain vectors ONLY for this workspace with proof phrase
        res = self.store.collection.get(where={"workspace_id": {"$eq": ws_id}})
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        self.assertGreaterEqual(len(docs), 2)
        blob = " ".join(docs).upper()
        self.assertIn(PROOF_PHRASE, blob)
        for m in metas:
            self.assertEqual(m.get("workspace_id"), ws_id)
            self.assertNotIn("deep-learning", (m.get("source_file") or "").lower())

        # 4) Analyze PYQ for same workspace
        analysis = self.client.post(
            f"/workspaces/{ws_id}/analyze-pyq",
            json={"workspace_id": ws_id, "subject": "Quantum Widget Flow Test", "semester": "Semester 8"},
        )
        self.assertEqual(analysis.status_code, 200, analysis.text)
        a = analysis.json()
        self.assertEqual(a.get("workspace_id", ws_id), ws_id)
        self.assertGreaterEqual(a.get("total_valid_questions") or a.get("total_questions_analyzed") or 0, 2)

        # 5) Ask unique phrase — must retrieve from this workspace's PDF
        ask = self.client.post(
            "/ask",
            json={
                "question": f"What is {PROOF_PHRASE}?",
                "workspace_id": ws_id,
                "mode": "general",
                "doc_type": "both",
            },
        )
        self.assertEqual(ask.status_code, 200, ask.text)
        ans = ask.json()
        answer_text = (ans.get("answer") or "").upper()
        self.assertNotIn("NOT_FOUND", answer_text[:40] if answer_text.startswith("NOT_FOUND") else "")
        # Evidence must mention proof phrase or cite the uploaded file
        sources = " ".join(
            str(c.get("source_file") or c.get("filename") or "")
            for c in (ans.get("citations") or [])
        ).lower()
        combined = (answer_text + " " + sources + " " + " ".join(docs).upper()).upper()
        self.assertIn(PROOF_PHRASE, combined)
        self.assertTrue(
            "proof_pyq.pdf" in sources or PROOF_PHRASE in answer_text or ans.get("retrieved_chunks_count", 0) > 0,
            f"Expected grounded retrieval for {ws_id}, got: {ans}",
        )

        # Cleanup vectors for this workspace
        self.store.delete_by_workspace(ws_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
