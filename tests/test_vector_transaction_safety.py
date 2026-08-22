"""
Vector safety: never destroy valid vectors before a replacement succeeds,
and never let one workspace's document affect another's.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.vector_store import VectorStore


def meta(ws: str, source: str, qid: str) -> dict:
    return {
        "workspace_id": ws,
        "source_file": source,
        "question_id": qid,
        "doc_type": "pyq",
    }


class TestSafeReplacement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()

    def setUp(self):
        self.ws_a = "test-vec-safety-a"
        self.ws_b = "test-vec-safety-b"
        self.store.delete_by_workspace(self.ws_a)
        self.store.delete_by_workspace(self.ws_b)

    def tearDown(self):
        self.store.delete_by_workspace(self.ws_a)
        self.store.delete_by_workspace(self.ws_b)

    def _count(self, ws: str) -> int:
        res = self.store.collection.get(where={"workspace_id": {"$eq": ws}})
        return len(res.get("ids") or [])

    def _ids(self, ws: str):
        res = self.store.collection.get(where={"workspace_id": {"$eq": ws}})
        return sorted(res.get("ids") or [])

    def test_replacement_removes_only_stale_vectors(self):
        old_docs = ["old q1", "old q2", "old q3"]
        old_ids = ["ws-a-paper-1", "ws-a-paper-2", "ws-a-paper-3"]
        self.store.add_documents(
            old_docs, [meta(self.ws_a, "paper.pdf", f"Q{i}(a)") for i in (1, 2, 3)], old_ids
        )
        self.assertEqual(self._count(self.ws_a), 3)

        new_docs = ["new q1", "new q2"]
        new_ids = ["ws-a-paper-1", "ws-a-paper-2"]
        inserted = self.store.replace_documents_for_source(
            new_docs,
            [meta(self.ws_a, "paper.pdf", f"Q{i}(a)") for i in (1, 2)],
            new_ids,
            source_file="paper.pdf",
            workspace_id=self.ws_a,
        )
        self.assertEqual(inserted, 2)
        self.assertEqual(self._ids(self.ws_a), sorted(new_ids))

    def test_failed_insert_leaves_old_vectors_intact(self):
        old_ids = ["ws-a-keep-1", "ws-a-keep-2"]
        self.store.add_documents(
            ["keep one", "keep two"],
            [meta(self.ws_a, "paper.pdf", "Q1(a)"), meta(self.ws_a, "paper.pdf", "Q1(b)")],
            old_ids,
        )
        self.assertEqual(self._count(self.ws_a), 2)

        with patch.object(
            self.store, "add_documents", side_effect=RuntimeError("embedding backend down")
        ):
            with self.assertRaises(RuntimeError):
                self.store.replace_documents_for_source(
                    ["replacement"],
                    [meta(self.ws_a, "paper.pdf", "Q2(a)")],
                    ["ws-a-new-1"],
                    source_file="paper.pdf",
                    workspace_id=self.ws_a,
                )

        # Nothing was purged: the previous good vectors survive the failure.
        self.assertEqual(self._ids(self.ws_a), sorted(old_ids))

    def test_no_duplicate_coexistence_after_replace(self):
        self.store.add_documents(
            ["v1 a", "v1 b"],
            [meta(self.ws_a, "paper.pdf", "Q1(a)"), meta(self.ws_a, "paper.pdf", "Q1(b)")],
            ["old-a", "old-b"],
        )
        self.store.replace_documents_for_source(
            ["v2 a", "v2 b"],
            [meta(self.ws_a, "paper.pdf", "Q1(a)"), meta(self.ws_a, "paper.pdf", "Q1(b)")],
            ["new-a", "new-b"],
            source_file="paper.pdf",
            workspace_id=self.ws_a,
        )
        res = self.store.collection.get(where={"workspace_id": {"$eq": self.ws_a}})
        docs = res.get("documents") or []
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(d.startswith("v2") for d in docs), docs)

    def test_same_filename_across_workspaces_is_isolated(self):
        self.store.add_documents(
            ["ws a content"], [meta(self.ws_a, "shared.pdf", "Q1(a)")], ["a-shared-1"]
        )
        self.store.add_documents(
            ["ws b content"], [meta(self.ws_b, "shared.pdf", "Q1(a)")], ["b-shared-1"]
        )

        self.store.replace_documents_for_source(
            ["ws a replaced"],
            [meta(self.ws_a, "shared.pdf", "Q1(a)")],
            ["a-shared-2"],
            source_file="shared.pdf",
            workspace_id=self.ws_a,
        )
        self.assertEqual(self._ids(self.ws_a), ["a-shared-2"])
        self.assertEqual(self._ids(self.ws_b), ["b-shared-1"])

    def test_delete_by_source_file_is_workspace_scoped(self):
        self.store.add_documents(
            ["ws a content"], [meta(self.ws_a, "shared.pdf", "Q1(a)")], ["a-only-1"]
        )
        self.store.add_documents(
            ["ws b content"], [meta(self.ws_b, "shared.pdf", "Q1(a)")], ["b-only-1"]
        )
        self.store.delete_by_source_file("shared.pdf", workspace_id=self.ws_a)
        self.assertEqual(self._count(self.ws_a), 0)
        self.assertEqual(self._count(self.ws_b), 1)

    def test_empty_replacement_is_a_no_op(self):
        self.store.add_documents(
            ["keep me"], [meta(self.ws_a, "paper.pdf", "Q1(a)")], ["keep-1"]
        )
        result = self.store.replace_documents_for_source(
            [], [], [], source_file="paper.pdf", workspace_id=self.ws_a
        )
        self.assertEqual(result, 0)
        self.assertEqual(self._count(self.ws_a), 1)


class TestIncompleteExtractionDoesNotTouchVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()

    def test_partial_extraction_keeps_previous_vectors(self):
        from rag.dynamic_ingest import DynamicIngestPipeline

        ws_id = "test-vec-safety-partial"
        self.store.delete_by_workspace(ws_id)
        self.store.add_documents(
            ["previous good question"],
            [meta(ws_id, "paper.pdf", "Q1(a)")],
            ["prev-good-1"],
        )

        pipeline = DynamicIngestPipeline(vector_store=self.store)
        pipeline.last_pyq_questions_audit = {}

        import fitz

        tmp = os.path.join(os.environ.get("TEMP", "."), "partial_probe.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 550, 700),
            "Q1(a) Explain the first concept in detail.\n"
            "Q2(a) \nQ3(a) \nQ4(a) \nQ5(a) \n",
        )
        doc.save(tmp)
        doc.close()

        pipeline.parse_pyq_pdf(tmp, {"id": ws_id, "subject": "Any Subject"})
        audit = pipeline.last_pyq_questions_audit or {}
        quality = audit.get("extraction_quality")

        res = self.store.collection.get(where={"workspace_id": {"$eq": ws_id}})
        ids = res.get("ids") or []
        if quality in ("PARTIAL", "FAILED"):
            self.assertIn("prev-good-1", ids, "incomplete extraction must not purge vectors")
        self.store.delete_by_workspace(ws_id)
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
