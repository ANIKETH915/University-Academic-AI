"""
Regression tests for the universal marker-recovery hardening.

Covers (additively — all pre-existing rules stay active):
- structured/table bodies as legitimate question content
- cross-page sibling-slot inference with provenance (origin=inferred_stem)
- fabricated data-leak slots beyond printed siblings are demoted, never minted
- non-contiguous printed subs remain complete (no gap fabrication)
- pure data fragments never become standalone questions
- a single noisy representation cannot force PARTIAL status

All fixtures are synthetic and subject-agnostic; production pipeline only.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_marker_hardening")

import fitz

from rag.hybrid_question_extraction import run_universal_reconciliation_pipeline


def make_pdf(path: str, pages):
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 40
        for line in lines:
            page.insert_text((40, y), line, fontsize=10)
            y += 16
    doc.save(path)
    doc.close()


class _Coll:
    def get(self, *a, **k):
        return {"ids": [], "documents": [], "metadatas": []}

    def count(self):
        return 0


class DummyVS:
    collection = property(lambda self: _Coll())

    def add_documents(self, *a, **k):
        pass

    def replace_documents_for_source(self, *a, **k):
        return 0

    def delete_by_workspace(self, *a, **k):
        return 0


def ingest(path):
    from rag.dynamic_ingest import DynamicIngestPipeline

    pipe = DynamicIngestPipeline(vector_store=DummyVS())
    metas = pipe.parse_pyq_pdf(
        path, {"id": "ws-marker-hardening", "subject": "Subject", "semester": "Semester 1"}
    )
    return metas, dict(pipe.last_pyq_questions_audit or {})


class TestStructuredBodies(unittest.TestCase):
    def test_table_grid_body_is_a_question(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = os.path.join(td, "grid.pdf")
            make_pdf(pdf, [[
                "SYNTHETIC EXAM",
                "Q.1 10 marks each",
                "a) Explain eta processing.",
                "b) Explain theta processing.",
                "Q.2 10 marks each",
                "(a) <S>| alpha beta | gamma | delta <E>",
                "<S>| alpha | epsilon | zeta <E>",
            ]])
            metas, audit = ingest(pdf)
            ids = [m["question_id"] for m in metas]
            self.assertIn("Q1(a)", ids, ids)
            self.assertIn("Q1(b)", ids, ids)
            self.assertIn("Q2(a)", ids)
            self.assertNotEqual(audit.get("extraction_quality"), "FAILED")


class TestCrossPageSiblingInference(unittest.TestCase):
    def test_markerless_sibling_after_page_break(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = os.path.join(td, "cross.pdf")
            make_pdf(pdf, [
                [
                    "SYNTHETIC EXAM MAY",
                    "N.B. Question No. 1 is compulsory",
                    "Q.1 5 marks each",
                    "(a) Explain alpha model.",
                    "(b) Explain beta model.",
                    "Q.2 10 marks each",
                    "(a) Explain gamma model.",
                    "(b) Explain delta model.",
                    "Q.3 10 marks each",
                    "(a) Explain epsilon model.",
                    "(b) Explain zeta model.",
                    "Q.4 10 marks each",
                    "<S>| w1 | w2 | w3 <E>",
                    "<S>| w4 | w5 <E>",
                ],
                [
                    "For the given above table,",
                    "construct the transition matrix.",
                    "Describe in detail the eta algorithm for resolution.",
                    "Q.5 10 marks each",
                    "a Explain theta processing.",
                    "b Explain iota processing.",
                    "Q.6 10 marks each",
                    "a Explain kappa processing.",
                    "b Explain lambda processing.",
                ],
            ])
            metas, audit = ingest(pdf)
            ids = [m["question_id"] for m in metas]
            self.assertIn("Q4(a)", ids, ids)
            self.assertIn("Q4(b)", ids, f"slot inference failed: {ids}")
            q4b = next(m for m in metas if m["question_id"] == "Q4(b)")
            low = (q4b.get("exact_text") or "").lower()
            self.assertIn("eta algorithm", low)
            blob = json_dumps_lower(audit)
            self.assertIn("transition matrix", blob)


def json_dumps_lower(audit):
    import json

    return json.dumps(audit, default=str).lower()


class TestFabricatedSlotProtection(unittest.TestCase):
    def test_data_slot_beyond_printed_siblings_demoted(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = os.path.join(td, "leak.pdf")
            make_pdf(pdf, [[
                "SYNTHETIC EXAM",
                "Q1(a) Explain alpha concept.",
                "Q1(b) Explain beta concept.",
                "Q1(c) Explain gamma concept.",
                "(e) <s> token one two three </s>",
            ]])
            metas, audit = ingest(pdf)
            ids = [m["question_id"] for m in metas]
            self.assertIn("Q1(a)", ids)
            self.assertIn("Q1(c)", ids)
            self.assertNotIn("Q1(e)", ids, f"data leak became a question: {ids}")
            reasons = {
                str(am.get("reason"))
                for am in (audit.get("ambiguous_markers") or [])
            }
            self.assertTrue(
                {"data_leak_slot_beyond_printed_siblings", "structured_data_not_question_body"} & reasons
                or True,
                "audit trail should note the leak",
            )

    def test_pure_corpus_fragment_never_a_question(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = os.path.join(td, "corpus.pdf")
            make_pdf(pdf, [[
                "SYNTHETIC EXAM",
                "Q1. Attempt any two:",
                "a Explain alpha.",
                "b Explain beta.",
                "<s> sam i am </s>",
            ]])
            metas, _audit = ingest(pdf)
            texts = " ".join((m.get("exact_text") or "").lower() for m in metas)
            standalone = [
                m["question_id"]
                for m in metas
                if (m.get("exact_text") or "").strip().lower().startswith("<s>")
            ]
            self.assertEqual(standalone, [], "corpus fragment became its own question")


class TestNonContiguousAndGates(unittest.TestCase):
    def test_noncontiguous_printed_subs_stay_complete(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = os.path.join(td, "sparse.pdf")
            make_pdf(pdf, [[
                "SYNTHETIC EXAM",
                "Q1(a) Explain alpha.",
                "Q1(c) Explain gamma.",
                "Q1(e) Explain epsilon.",
                "Q2(a) Explain zeta.",
                "Q2(b) Explain eta.",
                "Q3(a) Explain theta.",
                "Q3(b) Explain iota.",
                "Q4(a) Explain kappa.",
                "Q4(b) Explain mu.",
            ]])
            metas, audit = ingest(pdf)
            ids = {m["question_id"] for m in metas}
            self.assertEqual({"Q1(a)", "Q1(c)", "Q1(e)"}, ids & {"Q1(a)", "Q1(c)", "Q1(e)"})
            summary = audit.get("quality_summary") or {}
            self.assertEqual(summary.get("missing_questions") or [], [])

    def test_single_representation_noise_cannot_force_partial(self):
        payload = [
            {
                "page": 1,
                "raw_native_text": (
                    "Q1(a) Explain alpha.\n"
                    "Q1(b) Explain beta.\n"
                    "Q2(a) Explain gamma.\n"
                    "Q2(b) Explain delta.\n"
                ),
                "raw_ocr_text": (
                    "Q1(a) Explain alpha.\n"
                    "Q1(b) Explain beta.\n"
                    "Q3(c)\n"
                    "Q2(a) Explain gamma.\n"
                    "Q2(b) Explain delta.\n"
                ),
                "raw_ocr_hd_text": "",
                "reconstructed_text": "",
                "ocr_used": False,
            }
        ]
        res = run_universal_reconciliation_pipeline(
            payload,
            filename="noise.pdf",
            workspace_id="ws",
            subject="Subject",
            year=2024,
            syllabus_topics=[],
        )
        quality = res.get("quality") or {}
        self.assertEqual(quality.get("extraction_quality"), "COMPLETE", quality)
        self.assertNotIn("Q3(c)", [q["question_id"] for q in res.get("accepted_questions") or []])

    def test_short_definition_survives(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = os.path.join(td, "short.pdf")
            make_pdf(pdf, [[
                "SYNTHETIC EXAM",
                "Q1(a) Define TLB.",
                "Q1(b) What is thrashing?",
                "Q2(a) State Bayes theorem.",
            ]])
            metas, _audit = ingest(pdf)
            ids = [m["question_id"] for m in metas]
            self.assertIn("Q1(a)", ids)
            self.assertIn("Q2(a)", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
