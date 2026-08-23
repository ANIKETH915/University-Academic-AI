import os
import unittest
from fastapi.testclient import TestClient

from rag.api import app, store
from rag.workspace_db import workspace_db

client = TestClient(app)

class TestDocumentDeletion(unittest.TestCase):
    def setUp(self):
        self.ws_a_id = "ws-test-delete-a-101"
        self.ws_b_id = "ws-test-delete-b-202"
        
        # Clean up if existed
        workspace_db.delete_workspace(self.ws_a_id)
        workspace_db.delete_workspace(self.ws_b_id)

        # Create Workspace A
        workspace_db.get_or_create(
            workspace_id=self.ws_a_id,
            university="Test University",
            branch="Computer Science",
            semester="Semester 5",
            subject="Data Structures",
            subject_code="CS101",
        )

        # Create Workspace B
        workspace_db.get_or_create(
            workspace_id=self.ws_b_id,
            university="Test University",
            branch="Computer Science",
            semester="Semester 5",
            subject="Operating Systems",
            subject_code="CS102",
        )

    def tearDown(self):
        # Purge test vectors
        store.delete_by_workspace(self.ws_a_id)
        store.delete_by_workspace(self.ws_b_id)
        workspace_db.delete_workspace(self.ws_a_id)
        workspace_db.delete_workspace(self.ws_b_id)

    def test_pending_document_deletion(self):
        # Pending un-built file -> Delete allowed
        pending_file = {"id": "doc-pending-1", "name": "PendingPaper.pdf", "status": "PENDING", "build_started": False}
        workspace_db.add_file(self.ws_a_id, pending_file, doc_type="pyq")

        response = client.delete(
            f"/workspaces/{self.ws_a_id}/documents/doc-pending-1?doc_type=pyq&filename=PendingPaper.pdf"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

        ws_updated = workspace_db.get_by_id(self.ws_a_id)
        pyq_ids = [f["id"] for f in ws_updated.get("pyq_files", [])]
        self.assertNotIn("doc-pending-1", pyq_ids)

    def test_built_document_deletion_rejected_with_409(self):
        # Document with build_started=True and status=VERIFIED -> DELETE rejected with 409
        built_file = {"id": "doc-built-1", "name": "BuiltPaper.pdf", "status": "VERIFIED", "build_started": True}
        workspace_db.add_file(self.ws_a_id, built_file, doc_type="pyq")

        # Also add vector to ChromaDB
        store.add_documents(
            documents=["Explain stack traversal."],
            metadatas=[{"workspace_id": self.ws_a_id, "source_file": "BuiltPaper.pdf", "question_id": "Q100"}],
            ids=[f"{self.ws_a_id}-built-q100"]
        )

        response = client.delete(
            f"/workspaces/{self.ws_a_id}/documents/doc-built-1?doc_type=pyq&filename=BuiltPaper.pdf"
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("cannot be deleted after build processing has started", response.json()["detail"])

        # Document and vectors MUST still exist
        ws_check = workspace_db.get_by_id(self.ws_a_id)
        pyq_ids = [f["id"] for f in ws_check.get("pyq_files", [])]
        self.assertIn("doc-built-1", pyq_ids)

    def test_processing_document_deletion_rejected_with_409(self):
        # Document in PROCESSING state -> DELETE rejected with 409
        proc_file = {"id": "doc-proc-1", "name": "ProcessingPaper.pdf", "status": "PROCESSING", "build_started": True}
        workspace_db.add_file(self.ws_a_id, proc_file, doc_type="pyq")

        response = client.delete(
            f"/workspaces/{self.ws_a_id}/documents/doc-proc-1?doc_type=pyq&filename=ProcessingPaper.pdf"
        )
        self.assertEqual(response.status_code, 409)

    def test_two_documents_one_built_one_pending(self):
        # Document 1: Built (locked)
        doc1 = {"id": "doc-1", "name": "Paper1.pdf", "status": "READY", "build_started": True}
        # Document 2: Pending (deletable)
        doc2 = {"id": "doc-2", "name": "Paper2.pdf", "status": "PENDING", "build_started": False}

        workspace_db.add_file(self.ws_a_id, doc1, doc_type="pyq")
        workspace_db.add_file(self.ws_a_id, doc2, doc_type="pyq")

        # Deleting pending Document 2 -> Succeeds 200
        res2 = client.delete(f"/workspaces/{self.ws_a_id}/documents/doc-2?doc_type=pyq&filename=Paper2.pdf")
        self.assertEqual(res2.status_code, 200)

        # Deleting built Document 1 -> Fails 409
        res1 = client.delete(f"/workspaces/{self.ws_a_id}/documents/doc-1?doc_type=pyq&filename=Paper1.pdf")
        self.assertEqual(res1.status_code, 409)

        # Document 1 remains in workspace
        ws_check = workspace_db.get_by_id(self.ws_a_id)
        pyq_ids = [f["id"] for f in ws_check.get("pyq_files", [])]
        self.assertIn("doc-1", pyq_ids)
        self.assertNotIn("doc-2", pyq_ids)

    def test_three_documents_isolated_deletion(self):
        doc_a = {"id": "doc-a", "name": "Paper-A.pdf", "status": "PENDING", "build_started": False}
        doc_b = {"id": "doc-b", "name": "Paper-B.pdf", "status": "PENDING", "build_started": False}
        doc_c = {"id": "doc-c", "name": "Paper-C.pdf", "status": "PENDING", "build_started": False}

        workspace_db.add_file(self.ws_a_id, doc_a, doc_type="pyq")
        workspace_db.add_file(self.ws_a_id, doc_b, doc_type="pyq")
        workspace_db.add_file(self.ws_a_id, doc_c, doc_type="pyq")

        # Verify initial list
        ws_init = workspace_db.get_by_id(self.ws_a_id)
        pyq_names_init = [f["name"] for f in ws_init.get("pyq_files", [])]
        self.assertEqual(sorted(pyq_names_init), ["Paper-A.pdf", "Paper-B.pdf", "Paper-C.pdf"])

        # Delete ONLY Paper-B.pdf
        res = client.delete(f"/workspaces/{self.ws_a_id}/documents/doc-b?doc_type=pyq&filename=Paper-B.pdf")
        self.assertEqual(res.status_code, 200)

        # Verify ONLY Paper-B.pdf is removed; Paper-A.pdf and Paper-C.pdf MUST remain
        ws_after_b = workspace_db.get_by_id(self.ws_a_id)
        pyq_names_after_b = [f["name"] for f in ws_after_b.get("pyq_files", [])]
        self.assertNotIn("Paper-B.pdf", pyq_names_after_b)
        self.assertIn("Paper-A.pdf", pyq_names_after_b)
        self.assertIn("Paper-C.pdf", pyq_names_after_b)
        self.assertEqual(len(pyq_names_after_b), 2)

        # Delete Paper-A.pdf
        res_a = client.delete(f"/workspaces/{self.ws_a_id}/documents/doc-a?doc_type=pyq&filename=Paper-A.pdf")
        self.assertEqual(res_a.status_code, 200)

        # Verify ONLY Paper-C.pdf remains
        ws_after_a = workspace_db.get_by_id(self.ws_a_id)
        pyq_names_after_a = [f["name"] for f in ws_after_a.get("pyq_files", [])]
        self.assertEqual(pyq_names_after_a, ["Paper-C.pdf"])


if __name__ == "__main__":
    unittest.main()
