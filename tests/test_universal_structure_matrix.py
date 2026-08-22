"""
Structural universality regression tests.

These lock in behaviour that is independent of subject, university, year,
filename and question count. Each PDF is authored here, so the expected
question ids are known by construction rather than copied from a previous run.

They are structural probes, not evidence of real-world universality: they vary
layout, numbering style, sub-letter depth, page breaks and question count, not
subject matter.
"""
import os
import shutil
import tempfile
import unittest

import fitz

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.workspace_db import WorkspaceDB

HEADER = [
    "NATIONAL INSTITUTE OF ENGINEERING",
    "END SEMESTER EXAMINATION - MAY 2027",
    "Duration: 3 Hours                       Max Marks: 80",
    "Instructions: Attempt any four questions.",
]

BODIES = [
    "Explain the layered reference model and justify each layer with a diagram.",
    "Describe the scheduling policy and derive its average waiting time.",
    "Differentiate between the two normalisation forms with a worked example.",
    "Discuss the error control mechanism and analyse its overhead.",
    "Derive the expression for throughput and state all assumptions clearly.",
    "Compare the two indexing structures and evaluate their lookup complexity.",
]


def body(i: int) -> str:
    return BODIES[i % len(BODIES)]


class StructureMatrixBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
        os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_structure_matrix")
        cls.tmp = tempfile.mkdtemp(prefix="pyqrag_structure_")
        cls.pipe = DynamicIngestPipeline()
        cls.ws_db = WorkspaceDB()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def write_pdf(self, name: str, pages: list) -> str:
        path = os.path.join(self.tmp, name)
        doc = fitz.open()
        for lines in pages:
            page = doc.new_page()
            y = 55
            for ln in lines:
                if y > page.rect.height - 60:
                    page = doc.new_page()
                    y = 55
                page.insert_textbox(fitz.Rect(45, y, 550, y + 30), ln, fontsize=9)
                y += 30
        doc.save(path)
        doc.close()
        return path

    def ingest(self, name: str, pages: list):
        path = self.write_pdf(name, pages)
        ws_id = f"ws-struct-{name.replace('.pdf', '')}"
        try:
            self.ws_db.delete_workspace(ws_id)
        except Exception:
            pass
        ws = self.ws_db.get_or_create(ws_id, subject="Structure Subject")
        self.addCleanup(lambda: self._drop(ws_id))
        return self.pipe.parse_pyq_pdf(path, ws)

    def _drop(self, ws_id):
        try:
            self.ws_db.delete_workspace(ws_id)
        except Exception:
            pass


class TestCrossPageTailRecovery(StructureMatrixBase):
    """A page break must never destroy a question (no false 422)."""

    def _split_paper(self):
        p1 = list(HEADER)
        for i, s in enumerate("abc"):
            p1.append(f"Q1({s}) {body(i)} [6]")
        # Last question on page 1 is cut mid-sentence by the page break.
        p1.append("Q2(a) Explain the transaction recovery procedure and describe how the")
        p2 = ["system restores a consistent checkpoint after an unexpected failure. [8]"]
        for i, s in enumerate("bc"):
            p2.append(f"Q2({s}) {body(i + 4)} [6]")
        return [p1, p2]

    def test_question_split_by_page_break_is_recovered(self):
        metas = self.ingest("crosspage.pdf", self._split_paper())
        ids = sorted(m["question_id"] for m in metas)
        self.assertEqual(
            ids,
            ["Q1(a)", "Q1(b)", "Q1(c)", "Q2(a)", "Q2(b)", "Q2(c)"],
            "A question interrupted by a page break must still be extracted",
        )

    def test_recovered_question_is_complete_not_truncated(self):
        metas = self.ingest("crosspage_text.pdf", self._split_paper())
        q2a = next(m for m in metas if m["question_id"] == "Q2(a)")
        text = q2a["exact_text"]
        self.assertIn("transaction recovery procedure", text)
        self.assertIn(
            "consistent checkpoint",
            text,
            "The continuation from the next page must be joined to the question",
        )
        self.assertFalse(
            text.rstrip().endswith("the"),
            "Recovered question must not remain a truncated fragment",
        )

    def test_recovered_question_records_its_page_span(self):
        metas = self.ingest("crosspage_span.pdf", self._split_paper())
        q2a = next(m for m in metas if m["question_id"] == "Q2(a)")
        self.assertEqual(q2a.get("source_page_start") or q2a.get("source_page"), 1)
        self.assertEqual(q2a.get("source_page_end"), 2)

    def test_continuation_does_not_become_its_own_question(self):
        metas = self.ingest("crosspage_orphan.pdf", self._split_paper())
        for m in metas:
            self.assertFalse(
                m["exact_text"].strip().startswith("system restores"),
                "The continuation fragment must not be emitted as a separate question",
            )


class TestFlatNumberingNotFabricated(StructureMatrixBase):
    """A paper without subquestions must not acquire invented (a) ids."""

    def test_flat_paper_keeps_bare_parent_ids(self):
        lines = list(HEADER)
        for p in range(1, 8):
            lines.append(f"Q{p}. {body(p)} [10]")
        metas = self.ingest("flat.pdf", [lines])
        ids = sorted((m["question_id"] for m in metas), key=lambda s: int(s[1:]))
        self.assertEqual(ids, [f"Q{i}" for i in range(1, 8)])
        for m in metas:
            # Chroma rejects None, so "no subquestion" is stored as "".
            self.assertFalse(
                m.get("subquestion"),
                f"{m['question_id']} must not invent a subquestion",
            )

    def test_subdivided_paper_still_labels_leading_part_a(self):
        """When the parent IS subdivided, its leading text is genuinely (a)."""
        lines = list(HEADER)
        lines.append(f"Q1. {body(0)} [6]")
        lines.append(f"b) {body(1)} [6]")
        lines.append(f"c) {body(2)} [6]")
        lines.append(f"Q2. {body(3)} [6]")
        lines.append(f"b) {body(4)} [6]")
        metas = self.ingest("subdivided.pdf", [lines])
        ids = sorted(m["question_id"] for m in metas)
        self.assertEqual(ids, ["Q1(a)", "Q1(b)", "Q1(c)", "Q2(a)", "Q2(b)"])


class TestArbitraryQuestionCounts(StructureMatrixBase):
    """No fixed count, and no upper limit."""

    def _paper(self, total: int):
        lines = list(HEADER)
        truth, made, p = [], 0, 1
        while made < total:
            width = min((p * 2 + 1) % 5 + 2, total - made)
            for si in range(width):
                s = "abcdefghij"[si]
                lines.append(f"Q{p}({s}) {body(p + si)} [5]")
                truth.append(f"Q{p}({s})")
            made += width
            p += 1
        return lines, truth

    def test_counts_from_tiny_to_large(self):
        for total in (3, 7, 13, 18, 31):
            with self.subTest(total=total):
                lines, truth = self._paper(total)
                metas = self.ingest(f"count{total}.pdf", [lines])
                got = sorted(m["question_id"] for m in metas)
                self.assertEqual(
                    got, sorted(truth), f"{total}-question paper mis-extracted"
                )


class TestSubLetterDepth(StructureMatrixBase):
    """Sub-letters beyond (b) must survive, including a-f and a-j."""

    def test_parent_with_six_subs_and_third_subs(self):
        lines = list(HEADER)
        truth = []
        for i, s in enumerate("abcdef"):
            lines.append(f"Q1({s}) {body(i)} [5]")
            truth.append(f"Q1({s})")
        for p in (2, 3, 4):
            for i, s in enumerate("abc"):
                lines.append(f"Q{p}({s}) {body(p + i)} [6]")
                truth.append(f"Q{p}({s})")
        metas = self.ingest("subs_af.pdf", [lines])
        got = sorted(m["question_id"] for m in metas)
        self.assertEqual(got, sorted(truth))
        for qid in ("Q1(f)", "Q3(c)", "Q4(c)"):
            self.assertIn(qid, got, f"{qid} was lost")

    def test_sub_letters_to_j(self):
        lines = list(HEADER)
        truth = []
        for i, s in enumerate("abcdefghij"):
            lines.append(f"Q1({s}) {body(i)} [4]")
            truth.append(f"Q1({s})")
        metas = self.ingest("subs_aj.pdf", [lines])
        got = sorted(m["question_id"] for m in metas)
        self.assertEqual(got, sorted(truth))


class TestNumberingStyles(StructureMatrixBase):
    """Numbering style must not change what is recovered."""

    def _expected(self):
        return sorted(f"Q{p}({s})" for p in (1, 2, 3) for s in "abc")

    def test_no_q_prefix(self):
        lines = list(HEADER)
        for p in (1, 2, 3):
            for i, s in enumerate("abc"):
                lines.append(f"{p}({s}) {body(p + i)} [6]")
        metas = self.ingest("style_noq.pdf", [lines])
        self.assertEqual(sorted(m["question_id"] for m in metas), self._expected())

    def test_space_before_paren(self):
        lines = list(HEADER)
        for p in (1, 2, 3):
            for i, s in enumerate("abc"):
                lines.append(f"{p} {s}) {body(p + i)} [6]")
        metas = self.ingest("style_space.pdf", [lines])
        self.assertEqual(sorted(m["question_id"] for m in metas), self._expected())

    def test_roman_subquestions(self):
        lines = list(HEADER)
        for p in (1, 2, 3):
            for i, s in enumerate(["i", "ii", "iii"]):
                lines.append(f"Q{p}({s}) {body(p + i)} [6]")
        metas = self.ingest("style_roman.pdf", [lines])
        ids = sorted(m["question_id"] for m in metas)
        self.assertEqual(len(ids), 9, f"roman subquestions lost: {ids}")


class TestFalseMarkersRejected(StructureMatrixBase):
    """Decoys must not become questions (no false success)."""

    def test_degrees_dimensions_years_are_not_questions(self):
        lines = list(HEADER) + [
            "Note: B.E. (Sem 7) candidates must answer in the ruled book.",
            "Refer to figure 5*5 and matrix 32*32*3 where applicable.",
            "Page 1 of 2",
        ]
        truth = []
        for p in (1, 2, 3):
            for i, s in enumerate("ab"):
                lines.append(f"Q{p}({s}) {body(p + i)} [6]")
                truth.append(f"Q{p}({s})")
        lines += ["2024", "10"]
        metas = self.ingest("decoys.pdf", [lines])
        got = sorted(m["question_id"] for m in metas)
        self.assertEqual(got, sorted(truth))
        for m in metas:
            self.assertNotIn("Page 1 of 2", m["exact_text"])
            self.assertFalse(m["exact_text"].strip().startswith("B.E."))


if __name__ == "__main__":
    unittest.main()
