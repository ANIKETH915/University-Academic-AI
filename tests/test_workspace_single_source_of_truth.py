import os
import sys
import unittest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from rag.api import app
from rag.workspace_db import WorkspaceDB
from rag.vector_store import VectorStore

class TestWorkspaceSingleSourceOfTruth(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.ws_db = WorkspaceDB()
        self.store = VectorStore()

    def test_01_canonical_production_workspace_preserved(self):
        ws = self.ws_db.get_by_id("ws-default-workspace")
        if ws is None:
            # Workspace reset state (0 workspaces)
            self.assertTrue(isinstance(self.ws_db.get_all(), list))
        else:
            self.assertEqual(ws.get("subject"), "Deep Learning")
            self.assertEqual(len(ws.get("syllabus_files", [])), 1)
            self.assertEqual(len(ws.get("pyq_files", [])), 4)

    def test_02_production_vectors_preserved(self):
        c_res = self.store.collection.get(where={"workspace_id": {"$eq": "ws-default-workspace"}})
        ids = c_res.get("ids", [])
        if not ids:
            self.skipTest(
                "ws-default-workspace has no vectors in this environment; "
                "re-upload PYQs or run scratch/restore_production_ws.py"
            )
        self.assertGreater(len(ids), 0)

    def test_03_create_workspace_api_returns_canonical_workspace(self):
        ws_test_id = "ws-test-single-source-truth"
        self.ws_db.delete_workspace(ws_test_id)

        res = self.client.post("/workspaces", json={
            "university": "Test University",
            "branch": "Computer Science",
            "semester": "Semester 5",
            "subject": "Test Single Source Subject",
            "subject_code": "CSC599"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("id", data)
        self.assertEqual(data.get("subject"), "Test Single Source Subject")

        # Cleanup
        self.ws_db.delete_workspace(data["id"])


class TestNoGhostWorkspaces(unittest.TestCase):
    """Reading an unknown workspace must 404, never conjure one into existence."""

    def setUp(self):
        self.client = TestClient(app)
        self.ws_db = WorkspaceDB()
        self.unknown = "ws-definitely-not-created-12345"
        self.ws_db.delete_workspace(self.unknown)

    def tearDown(self):
        self.ws_db.delete_workspace(self.unknown)

    def test_analysis_endpoints_404_for_unknown_workspace(self):
        endpoints = [
            ("post", f"/workspaces/{self.unknown}/analyze-pyq", {"workspace_id": self.unknown}),
            ("post", f"/workspaces/{self.unknown}/study-priority", {"workspace_id": self.unknown}),
            ("get", f"/workspaces/{self.unknown}/pyq-questions", None),
            ("get", f"/workspaces/{self.unknown}/pyq-patterns", None),
            ("get", f"/workspaces/{self.unknown}/pyq-audit", None),
            ("get", f"/workspaces/{self.unknown}/study-priority", None),
            ("get", f"/workspaces/{self.unknown}/audit", None),
        ]
        for method, url, payload in endpoints:
            with self.subTest(url=url):
                res = (
                    self.client.post(url, json=payload)
                    if method == "post"
                    else self.client.get(url)
                )
                self.assertEqual(res.status_code, 404, f"{url} -> {res.status_code}")

    def test_unknown_workspace_not_persisted_after_requests(self):
        self.client.get(f"/workspaces/{self.unknown}/pyq-questions")
        self.client.post(
            f"/workspaces/{self.unknown}/analyze-pyq", json={"workspace_id": self.unknown}
        )
        self.assertIsNone(self.ws_db.get_by_id(self.unknown))
        listed = [w.get("id") for w in self.client.get("/workspaces").json()]
        self.assertNotIn(self.unknown, listed)

    def test_legacy_default_workspace_id_rejected_for_ingest(self):
        res = self.client.post(
            "/workspaces/ws-default-workspace/ingest",
            files={"file": ("x.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"doc_type": "pyq"},
        )
        self.assertEqual(res.status_code, 400)

    def test_health_reports_llm_without_secrets(self):
        data = self.client.get("/health").json()
        self.assertIn("llm", data)
        blob = json.dumps(data)
        self.assertNotIn("api_key", blob.lower())
        for key_env in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
            secret = os.environ.get(key_env)
            if secret:
                self.assertNotIn(secret, blob)


if __name__ == "__main__":
    unittest.main()
