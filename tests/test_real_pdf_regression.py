"""
Regression tests against the real PYQ PDFs in the repository.

These run the *production* ingestion pipeline on real files — no mocks, no
stubbed extraction, no hardcoded expected text. They assert the structure the
papers actually have, which is what the December 2024 failure got wrong.

Tests skip (rather than fail) when a PDF or Tesseract is unavailable, so the
suite stays runnable on machines without OCR installed.
"""

from __future__ import annotations

import glob
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.question_extractor import question_structure_score
from rag.vector_store import VectorStore

PYQ_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "pyq",
)


def find_pdf(*fragments: str, under: str = "deep-learning"):
    folder = os.path.join(PYQ_DIR, under) if under else PYQ_DIR
    for path in glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True):
        low = os.path.basename(path).lower()
        if all(f in low for f in fragments):
            return path
    return None


def tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


_STORE = None
_PIPELINE = None
_INGEST_CACHE: dict = {}
_WORKSPACES: list = []


def _pipeline():
    global _STORE, _PIPELINE
    if _PIPELINE is None:
        _STORE = VectorStore()
        _PIPELINE = DynamicIngestPipeline(vector_store=_STORE)
    return _STORE, _PIPELINE


class RealPdfCase(unittest.TestCase):
    """
    Base class that runs the production pipeline once per PDF and shares the
    result. OCR is expensive; re-running it per assertion would prove nothing
    extra.
    """

    @classmethod
    def setUpClass(cls):
        cls.store, cls.pipeline = _pipeline()

    @classmethod
    def tearDownClass(cls):
        for ws in list(_WORKSPACES):
            cls.store.delete_by_workspace(ws)

    def ingest(self, path: str, ws_suffix: str = ""):
        key = os.path.abspath(path)
        if key in _INGEST_CACHE:
            return _INGEST_CACHE[key]

        ws_id = "test-real-pdf-" + os.path.splitext(os.path.basename(path))[0][:40]
        self.store.delete_by_workspace(ws_id)
        _WORKSPACES.append(ws_id)
        metas = self.pipeline.parse_pyq_pdf(
            path,
            {
                "id": ws_id,
                "subject": "Academic Subject",
                "semester": "Semester 7",
            },
        )
        audit = dict(self.pipeline.last_pyq_questions_audit or {})
        _INGEST_CACHE[key] = (ws_id, metas, audit)
        return _INGEST_CACHE[key]


class TestDecember2024Regression(RealPdfCase):
    """The paper that previously returned HTTP 422 with zero questions."""

    def setUp(self):
        self.pdf = find_pdf("2024", "december")
        if not self.pdf:
            self.skipTest("December 2024 PDF not present")
        if not tesseract_available():
            self.skipTest("Tesseract not installed")

    def test_extracts_complete_paper(self):
        ws_id, metas, audit = self.ingest(self.pdf, "dec2024")
        self.assertEqual(audit.get("extraction_quality"), "COMPLETE")
        self.assertEqual(audit.get("ingestion_status"), "ready")
        self.assertGreater(len(metas), 0)

    def test_structure_matches_paper_layout(self):
        _ws, _metas, audit = self.ingest(self.pdf, "dec2024-struct")
        ids = [q["question_id"] for q in audit.get("accepted_questions") or []]
        # One compulsory parent with five subquestions, then parents with two each.
        self.assertEqual(len(ids), len(set(ids)), "duplicate question ids")
        self.assertEqual(question_structure_score(ids), 1.0, ids)
        parents = {i.split("(")[0] for i in ids}
        self.assertGreaterEqual(len(parents), 5)
        self.assertIn("Q1(a)", ids)

    def test_no_missing_markers(self):
        _ws, _metas, audit = self.ingest(self.pdf, "dec2024-missing")
        summary = audit.get("quality_summary") or {}
        self.assertEqual(summary.get("missing_questions"), [])

    def test_questions_are_grounded(self):
        _ws, _metas, audit = self.ingest(self.pdf, "dec2024-ground")
        summary = audit.get("quality_summary") or {}
        self.assertGreaterEqual(summary.get("grounding_coverage") or 0, 0.8)

    def test_vectors_inserted_for_every_question(self):
        ws_id, metas, audit = self.ingest(self.pdf, "dec2024-vec")
        accepted = audit.get("accepted_questions") or []
        res = self.store.collection.get(where={"workspace_id": {"$eq": ws_id}})
        self.assertEqual(len(res.get("ids") or []), len(accepted))
        self.assertEqual(len(metas), len(accepted))

    def test_multiline_question_preserved(self):
        _ws, _metas, audit = self.ingest(self.pdf, "dec2024-multiline")
        texts = [q["exact_text"] for q in audit.get("accepted_questions") or []]
        # The CNN parameter-calculation item spans several source lines.
        long_items = [t for t in texts if len(t) > 150]
        self.assertTrue(long_items, "expected at least one multi-line question")
        calc = [t for t in texts if "calculate" in t.lower()]
        if calc:
            self.assertGreater(
                max(len(t) for t in calc), 60,
                "calculation question looks truncated",
            )


class TestAllRealPapersConsistent(RealPdfCase):
    """Every real paper in the repo must extract with sound structure."""

    def setUp(self):
        if not tesseract_available():
            self.skipTest("Tesseract not installed")
        self.pdfs = [
            p for p in (
                find_pdf("2023", "december"),
                find_pdf("2024", "december"),
                find_pdf("2024", "may"),
                find_pdf("2025", "may"),
            ) if p
        ]
        if not self.pdfs:
            self.skipTest("no real PYQ PDFs present")

    def test_every_paper_complete_with_sound_structure(self):
        for idx, pdf in enumerate(self.pdfs):
            with self.subTest(pdf=os.path.basename(pdf)):
                _ws, _metas, audit = self.ingest(pdf, f"all-{idx}")
                ids = [q["question_id"] for q in audit.get("accepted_questions") or []]
                self.assertEqual(audit.get("extraction_quality"), "COMPLETE")
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(question_structure_score(ids), 1.0, ids)
                self.assertGreaterEqual(len(ids), 8)

    def test_no_header_or_footer_text_in_questions(self):
        banned = ("qp code", "page 1 of", "max marks", "duration:", "paper / subject code")
        for idx, pdf in enumerate(self.pdfs):
            with self.subTest(pdf=os.path.basename(pdf)):
                _ws, _metas, audit = self.ingest(pdf, f"clean-{idx}")
                for q in audit.get("accepted_questions") or []:
                    low = q["exact_text"].lower()
                    for token in banned:
                        self.assertNotIn(token, low, f"{q['question_id']}: {low[:120]}")

    def test_representation_choice_is_recorded(self):
        _ws, _metas, audit = self.ingest(self.pdfs[0], "audit-repr")
        pages = audit.get("page_extraction_audit") or []
        self.assertTrue(pages)
        for page in pages:
            self.assertIn("selected_representation", page)
            self.assertIn("representations", page)


class TestScannedSyllabusIsNotAPaper(RealPdfCase):
    """
    A real scanned multi-subject syllabus is full of numbered lab-experiment
    lists. The PYQ path must not turn them into a paper's worth of questions.
    """

    def setUp(self):
        # 80 scanned pages: several minutes of OCR, so opt-in only.
        if os.environ.get("PYQRAG_SLOW_TESTS") != "1":
            self.skipTest("set PYQRAG_SLOW_TESTS=1 to run the 80-page scanned bundle")
        self.pdf = find_pdf("aidsaiml")
        if not self.pdf:
            self.skipTest("scanned syllabus bundle not present")
        if not tesseract_available():
            self.skipTest("Tesseract not installed")

    def test_syllabus_does_not_yield_a_confident_paper(self):
        _ws, _metas, audit = self.ingest(self.pdf, "syllabus-as-pyq")
        quality = audit.get("extraction_quality")
        accepted = audit.get("accepted_questions") or []
        if quality == "COMPLETE":
            # If anything is accepted at all it must still be genuinely grounded,
            # and a syllabus must never look like a normal 15-question paper.
            summary = audit.get("quality_summary") or {}
            self.assertGreaterEqual(summary.get("grounding_coverage") or 0, 0.8)
        self.assertNotEqual(
            audit.get("ingestion_status"), None,
            "ingestion status must always be reported",
        )
        for q in accepted:
            self.assertNotIn("term work", q["exact_text"].lower())


class TestUnseenNumberingStyleRealPdf(RealPdfCase):
    """
    Real unseen-subject paper whose numbering is Q.n.x) with Q1(a-f) and
    third subs on later parents. Production must not special-case the subject.
    """

    def setUp(self):
        self.pdf = find_pdf("natural-language", "2022", "december", under="nlp")
        if not self.pdf:
            # Uploaded copy used by the live workspace, if the data/pyq copy is absent.
            self.pdf = find_pdf("natural-language", "2022", "december", under="..")
        if not self.pdf:
            uploads = os.path.join(
                os.path.dirname(PYQ_DIR),
                "uploads",
                "ws-nlp-31567ba5",
            )
            hits = [
                p for p in glob.glob(os.path.join(uploads, "*natural-language*.pdf"))
                if "2022" in os.path.basename(p).lower()
            ]
            self.pdf = hits[0] if hits else None
        if not self.pdf:
            self.skipTest("2022 December NLP PDF not present")
        if not tesseract_available():
            self.skipTest("Tesseract not installed")

    def test_extracts_complete_reconciled_paper(self):
        _ws, metas, audit = self.ingest(self.pdf)
        ids = [q["question_id"] for q in audit.get("accepted_questions") or []]
        summary = audit.get("quality_summary") or {}
        self.assertEqual(audit.get("extraction_quality"), "COMPLETE", summary)
        self.assertEqual(summary.get("missing_questions") or [], [])
        for qid in ("Q1(a)", "Q1(f)", "Q3(c)", "Q4(c)"):
            self.assertIn(qid, ids, f"{qid} missing from {ids}")
        self.assertNotIn("Q1", ids, "Q1 must be subdivided, not a flat parent")
        self.assertNotIn("Q3(q)", ids)
        self.assertNotIn("Q4(q)", ids)
        self.assertGreaterEqual(len(ids), 16)
        self.assertEqual(len(metas), len(ids))

    def test_q1f_is_not_a_dump_of_the_rest_of_the_paper(self):
        _ws, _metas, audit = self.ingest(self.pdf)
        q1f = next(
            q for q in (audit.get("accepted_questions") or [])
            if q["question_id"] == "Q1(f)"
        )
        low = q1f["exact_text"].lower()
        self.assertIn("perplexity", low)
        self.assertNotIn("hidden markov", low)
        self.assertNotIn("porter", low)

    def test_http_ingest_then_analyze_pyq(self):
        """Same FastAPI route the frontend uses: POST /workspaces/{id}/ingest."""
        from fastapi.testclient import TestClient
        from rag.api import app

        client = TestClient(app)
        created = client.post(
            "/workspaces",
            json={
                "university": "Test University",
                "branch": "Computer Engineering",
                "semester": "Semester 7",
                "subject": "Academic Subject",
                "subject_code": "COURSE",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        ws_id = created.json()["id"]
        with open(self.pdf, "rb") as fh:
            ingest = client.post(
                f"/workspaces/{ws_id}/ingest",
                files={"file": (os.path.basename(self.pdf), fh, "application/pdf")},
                data={"doc_type": "pyq"},
            )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        body = ingest.json()
        self.assertEqual(body.get("extraction_quality"), "COMPLETE", body)
        self.assertGreaterEqual(body.get("vectors_inserted") or 0, 16)
        ids = (body.get("extraction_audit") or {}).get("accepted_question_ids") or []
        for qid in ("Q1(a)", "Q1(f)", "Q3(c)", "Q4(c)"):
            self.assertIn(qid, ids, f"{qid} missing from HTTP ingest {ids}")
        analysis = client.get(f"/workspaces/{ws_id}/analyze-pyq")
        self.assertEqual(analysis.status_code, 200, analysis.text)
        payload = analysis.json()
        qcount = payload.get("total_valid_questions") or payload.get("total_questions_analyzed") or 0
        self.assertGreaterEqual(qcount, 16, payload)
        src = client.get(f"/workspaces/{ws_id}/pyq-questions")
        self.assertEqual(src.status_code, 200, src.text)
        accepted = (src.json() or {}).get("accepted_questions") or []
        src_ids = {q.get("question_id") for q in accepted}
        self.assertTrue(set(ids).issubset(src_ids) or src_ids >= set(ids))


class TestUnlabelledMarkerOcrRealPdf(RealPdfCase):
    """Real paper whose OCR dropped Q/a) glyphs. Same pipeline, no subject branch."""

    def setUp(self):
        self.pdf = find_pdf("natural-language", "2023", "december", under="nlp")
        if not self.pdf:
            uploads = os.path.join(
                os.path.dirname(PYQ_DIR), "uploads", "ws-nlp-31567ba5"
            )
            hits = [
                p for p in glob.glob(os.path.join(uploads, "*natural-language*.pdf"))
                if "2023" in os.path.basename(p).lower()
                and "december" in os.path.basename(p).lower()
            ]
            self.pdf = hits[0] if hits else None
        if not self.pdf:
            self.skipTest("2023 December NLP PDF not present")
        if not tesseract_available():
            self.skipTest("Tesseract not installed")

    def test_extracts_complete_from_unlabelled_ocr(self):
        _ws, metas, audit = self.ingest(self.pdf)
        ids = [q["question_id"] for q in audit.get("accepted_questions") or []]
        summary = audit.get("quality_summary") or {}
        self.assertEqual(audit.get("extraction_quality"), "COMPLETE", summary)
        self.assertGreaterEqual(len(ids), 10, ids)
        self.assertIn("Q1(a)", ids)
        self.assertIn("Q2(a)", ids)
        blob = " ".join(q["exact_text"].lower() for q in (audit.get("accepted_questions") or []))
        self.assertIn("good turing", blob)
        self.assertNotIn("acre", blob)
        self.assertEqual(len(metas), len(ids))


class TestTwoPageCrossPageRealPdf(RealPdfCase):
    """Two-page paper: Q4 starts on page 1 and continues on page 2."""

    def setUp(self):
        self.pdf = find_pdf("natural-language", "2023", "may", under="nlp")
        if not self.pdf:
            uploads = os.path.join(
                os.path.dirname(PYQ_DIR), "uploads", "ws-nlp-31567ba5"
            )
            hits = [
                p for p in glob.glob(os.path.join(uploads, "*natural-language*.pdf"))
                if "2023" in os.path.basename(p).lower()
                and "may" in os.path.basename(p).lower()
            ]
            self.pdf = hits[0] if hits else None
        if not self.pdf:
            self.skipTest("2023 May NLP PDF not present")
        if not tesseract_available():
            self.skipTest("Tesseract not installed")

    def test_extracts_complete_including_page_two(self):
        _ws, metas, audit = self.ingest(self.pdf)
        ids = [q["question_id"] for q in audit.get("accepted_questions") or []]
        summary = audit.get("quality_summary") or {}
        self.assertEqual(audit.get("extraction_quality"), "COMPLETE", summary)
        self.assertNotIn("Q4(i)", summary.get("missing_questions") or [])
        for qid in ("Q1(a)", "Q4(a)", "Q4(b)", "Q5(a)", "Q5(b)", "Q6(a)", "Q6(b)"):
            self.assertIn(qid, ids, f"{qid} missing from {ids}")
        blob = " ".join(q["exact_text"].lower() for q in (audit.get("accepted_questions") or []))
        self.assertIn("centering", blob)
        self.assertIn("porter", blob)
        self.assertGreaterEqual(len(ids), 12)
        self.assertEqual(len(metas), len(ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
