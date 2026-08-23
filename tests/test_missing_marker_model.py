"""
Missing-marker model tests.

Rule under test:
    Only markers with actual evidence in the source may be reported.
    missing_questions = SOURCE-PROVEN-MISSING only.
    NOT-OBSERVED letters are never inferred, never reported.

No subject, filename, question-number or count assumptions anywhere.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")

from rag.hybrid_question_extraction import hybrid_extract_document


def _run(pages_text):
    pages = [
        {
            "page": i + 1,
            "raw_native_text": txt,
            "raw_ocr_text": "",
            "reconstructed_text": txt,
            "ocr_used": False,
        }
        for i, txt in enumerate(pages_text)
    ]
    with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
        return hybrid_extract_document(
            pages, filename="mm.pdf", workspace_id="ws-mm", year=2024
        )


def _q6_paper(subs):
    """A single parent block 'Q6.' with exactly the given printed subs."""
    lines = ["Q6. Attempt any four:", "Explain the introductory framing of the topic."]
    for s in subs:
        lines.append(f"Q6({s}) Explain {s * 3} concept with adequate supporting detail.")
    return "\n".join(lines) + "\n"


class TestNeverInferAlphabeticRanges(unittest.TestCase):
    def test_single_sub_only(self):
        res = _run([_q6_paper(["a"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])
        ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q6(a)", ids)
        for bad in ("Q6(b)", "Q6(c)", "Q6(d)", "Q6(e)", "Q6(f)"):
            self.assertNotIn(bad, ids)

    def test_two_subs_no_cde_inference(self):
        res = _run([_q6_paper(["a", "b"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_three_subs(self):
        res = _run([_q6_paper(["a", "b", "c"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_five_subs(self):
        res = _run([_q6_paper(["a", "b", "c", "d", "e"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_six_subs(self):
        res = _run([_q6_paper(["a", "b", "c", "d", "e", "f"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_two_sub_parents_only(self):
        # Two parents with two subs each: no third sub may be inferred.
        text = (
            "Q5(a) Explain the first mechanism with sufficient technical detail.\n"
            "Q5(b) Compare the second mechanism against the first one clearly.\n"
            "Q7(a) Describe the deployment pipeline stages in chronological order.\n"
            "Q7(b) Evaluate rollback strategies for failed deployment releases.\n"
        )
        res = _run([text])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])
        ids = [q["question_id"] for q in res["accepted_questions"]]
        for bad in ("Q5(c)", "Q7(c)"):
            self.assertNotIn(bad, ids)

    def test_variable_totals(self):
        three = "\n".join(
            f"Q{i}(a) Discuss distinct topic number {i} with relevant depth." for i in range(1, 4)
        )
        twelve = "\n".join(
            f"Q{i}(a) Discuss distinct topic number {i} with relevant depth."
            for i in range(1, 13)
        )
        for label, text in (("three", three), ("twelve", twelve)):
            res = _run([text + "\n"])
            self.assertEqual(
                res["quality"]["extraction_quality"],
                "COMPLETE",
                f"{label}: {res['quality']}",
            )
            self.assertEqual(res["quality"]["missing_questions"], [])


class TestSourceProvenMissing(unittest.TestCase):
    def test_proven_marker_with_unrecoverable_body_is_partial(self):
        # Source prints Q6(a-e); Q6(e)'s body is repetitive noise that the
        # validation gate correctly rejects. Q6(e) is then SOURCE-PROVEN-
        # MISSING: reported, never fabricated, status PARTIAL.
        text = (
            "Q6(a) Explain alpha mechanism with adequate supporting detail.\n"
            "Q6(b) Explain beta mechanism with adequate supporting detail.\n"
            "Q6(c) Explain gamma mechanism with adequate supporting detail.\n"
            "Q6(d) Explain delta mechanism with adequate supporting detail.\n"
            "Q6(e) Explain explain explain explain explain explain explain explain explain\n"
        )
        res = _run([text])
        quality = res["quality"]
        self.assertEqual(quality["extraction_quality"], "PARTIAL")
        self.assertEqual(quality["missing_questions"], ["Q6(e)"])
        ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertNotIn("Q6(e)", ids)
        for good in ("Q6(a)", "Q6(b)", "Q6(c)", "Q6(d)"):
            self.assertIn(good, ids)

    def test_ambiguous_marker_not_reported_missing(self):
        # A detected-but-unattributable stray letter stays in the audit,
        # never silently becomes a missing genuine question.
        text = (
            "Q6(a) Explain alpha mechanism with adequate supporting detail.\n"
            "Q6(b) Explain beta mechanism with adequate supporting detail.\n"
            "z) fragmentary OCR shred with no real content here\n"
        )
        res = _run([text])
        self.assertNotEqual(res["quality"]["extraction_quality"], "FAILED")
        self.assertNotIn("Q6(z)", res["quality"].get("missing_questions") or [])


class TestFlexibleSubquestionRangesAndGaps(unittest.TestCase):
    def test_q6_a_b(self):
        res = _run([_q6_paper(["a", "b"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_q6_a_c(self):
        res = _run([_q6_paper(["a", "b", "c"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_q6_a_e(self):
        res = _run([_q6_paper(["a", "b", "c", "d", "e"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_q6_a_f(self):
        res = _run([_q6_paper(["a", "b", "c", "d", "e", "f"])])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_q6_a_b_e_no_source_evidence(self):
        # Q6(a), Q6(b), Q6(e) present, but c and d have NO source evidence.
        # c and d must NOT be reported as missing.
        text = (
            "Q6. Attempt any four:\n"
            "Q6(a) Explain alpha mechanism with adequate supporting detail.\n"
            "Q6(b) Explain beta mechanism with adequate supporting detail.\n"
            "Q6(e) Explain epsilon mechanism with adequate supporting detail.\n"
        )
        res = _run([text])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q6(a)", extracted_ids)
        self.assertIn("Q6(b)", extracted_ids)
        self.assertIn("Q6(e)", extracted_ids)

    def test_q6_a_b_e_with_source_evidence_for_c(self):
        # Q6(a), Q6(b), Q6(e) present, and Q6(c) exists as printed marker in source text.
        # Q6(c) MUST be marked SOURCE-PROVEN-MISSING. Q6(d) has no evidence, so not missing.
        text = (
            "Q6. Attempt any four:\n"
            "Q6(a) Explain alpha mechanism with adequate supporting detail.\n"
            "Q6(b) Explain beta mechanism with adequate supporting detail.\n"
            "Q6(c)\n"
            "Q6(e) Explain epsilon mechanism with adequate supporting detail.\n"
        )
        res = _run([text])
        self.assertEqual(res["quality"]["extraction_quality"], "PARTIAL")
        self.assertIn("Q6(c)", res["quality"]["missing_questions"])
        self.assertNotIn("Q6(d)", res["quality"]["missing_questions"])

    def test_q6_a_b_f_no_source_evidence(self):
        # Q6(a), Q6(b), Q6(f) present. c, d, e have NO source evidence.
        # c, d, e must NOT be reported as missing.
        text = (
            "Q6. Attempt any four:\n"
            "Q6(a) Explain alpha mechanism with adequate supporting detail.\n"
            "Q6(b) Explain beta mechanism with adequate supporting detail.\n"
            "Q6(f) Explain zeta mechanism with adequate supporting detail.\n"
        )
        res = _run([text])
        self.assertEqual(res["quality"]["extraction_quality"], "COMPLETE")
        self.assertEqual(res["quality"]["missing_questions"], [])
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q6(a)", extracted_ids)
        self.assertIn("Q6(b)", extracted_ids)
        self.assertIn("Q6(f)", extracted_ids)

    def test_real_pdf_only_q6_a(self):
        import fitz
        from rag.dynamic_ingest import DynamicIngestPipeline
        from rag.workspace_db import WorkspaceDB

        doc = fitz.open()
        page = doc.new_page()
        text = (
            "UNIVERSITY EXAMINATION\n"
            "SUBJECT: DEEP LEARNING | SEMESTER 7\n"
            "Duration: 3 Hours | Max Marks: 80\n"
            "Instructions: Attempt all questions.\n"
            "Q6(a) Explain Convolutional Neural Networks for image classification. [10 Marks]\n"
        )
        page.insert_text((50, 50), text, fontsize=10)
        os.makedirs("scratch", exist_ok=True)
        pdf_path = os.path.join("scratch", "test_real_q6_a.pdf")
        doc.save(pdf_path)
        doc.close()

        ws_db = WorkspaceDB()
        ws = ws_db.get_or_create("ws-test-q6a", subject="Deep Learning")
        pipeline = DynamicIngestPipeline()
        metas = pipeline.parse_pyq_pdf(pdf_path, ws)
        audit = pipeline.last_pyq_questions_audit or {}

        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0]["question_number"], "Q6(a)")
        self.assertEqual(audit.get("quality_summary", {}).get("missing_questions"), [])

        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                pass

    def test_real_pdf_q5_a_e(self):
        import fitz
        from rag.dynamic_ingest import DynamicIngestPipeline
        from rag.workspace_db import WorkspaceDB

        doc = fitz.open()
        page = doc.new_page()
        lines = [
            "UNIVERSITY EXAMINATION",
            "SUBJECT: NATURAL LANGUAGE PROCESSING",
            "Duration: 3 Hours | Max Marks: 80",
            "Q5. Answer any four of the following:",
            "Q5(a) Explain Word Embeddings and Word2Vec models with diagrams.",
            "Q5(b) Discuss Recurrent Neural Networks and Vanishing Gradient Problem.",
            "Q5(c) Describe Long Short Term Memory Networks architecture.",
            "Q5(d) Explain Transformer Self Attention and Scaled Dot Product.",
            "Q5(e) Evaluate BERT vs GPT models for language modeling tasks.",
        ]
        page.insert_text((50, 50), "\n".join(lines), fontsize=10)
        os.makedirs("scratch", exist_ok=True)
        pdf_path = os.path.join("scratch", "test_real_q5_ae.pdf")
        doc.save(pdf_path)
        doc.close()

        ws_db = WorkspaceDB()
        ws = ws_db.get_or_create("ws-test-q5ae", subject="NLP")
        pipeline = DynamicIngestPipeline()
        metas = pipeline.parse_pyq_pdf(pdf_path, ws)
        audit = pipeline.last_pyq_questions_audit or {}

        self.assertEqual(len(metas), 5)
        extracted_ids = [m["question_number"] for m in metas]
        for expected in ("Q5(a)", "Q5(b)", "Q5(c)", "Q5(d)", "Q5(e)"):
            self.assertIn(expected, extracted_ids)
        self.assertEqual(audit.get("quality_summary", {}).get("missing_questions"), [])

        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)

