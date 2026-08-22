"""
Subject-agnostic PYQ Intelligence regression suite.

Proves the same engine works for Deep Learning, DBMS, Operating Systems,
and Computer Networks — with workspace isolation and no cross-contamination.
"""

import os
import re
import sys
import tempfile
import unittest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz
from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.pyq_intelligence import PYQIntelligenceEngine
from rag.workspace_db import WorkspaceDB
from rag.question_extractor import (
    extract_questions_from_page_text,
    classify_repeat_relationship_full,
    normalize_question_text,
    compute_text_similarity,
)
from rag.syllabus_index import (
    build_syllabus_index_from_chunks,
    map_question_to_syllabus_index,
    empty_syllabus_index,
)


def make_pdf(path: str, lines: list) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 40
    for line in lines:
        while line:
            chunk, line = line[:95], line[95:]
            if y > 780:
                page = doc.new_page()
                y = 40
            page.insert_text((40, y), chunk, fontsize=9)
            y += 12
    doc.save(path)
    doc.close()


class TestSubjectAgnosticFormats(unittest.TestCase):
    def test_variable_id_formats(self):
        samples = [
            ("Q1(a) Explain normalization up to 3NF in detail.\nQ1(b) Define ACID properties.\n", ["Q1(a)", "Q1(b)"]),
            ("1(a) Explain deadlock prevention techniques.\n1(b) Describe paging and segmentation.\n", ["Q1(a)", "Q1(b)"]),
            ("Q2(i) Compare TCP and UDP protocols.\nQ2(ii) Explain sliding window protocol.\n", ["Q2(i)", "Q2(ii)"]),
            ("Q3-A Explain hashing in databases.\nQ3-B Explain B+ trees for indexing.\n", ["Q3(a)", "Q3(b)"]),
        ]
        for text, expected_ids in samples:
            acc, _ = extract_questions_from_page_text(text, 1, "p.pdf", "ws", year=2024)
            got = [q["question_id"] for q in acc]
            for eid in expected_ids:
                self.assertIn(eid, got, f"Expected {eid} in {got} for sample: {text[:40]}")

    def test_no_fixed_question_count(self):
        lines = [f"Q1({chr(97+i)}) Explain concept family {i} analysis method {i}.\n" for i in range(7)]
        acc, _ = extract_questions_from_page_text("".join(lines), 1, "p.pdf", "ws", year=2024)
        self.assertEqual(len(acc), 7)

    def test_dbms_exact_and_semantic(self):
        a = "Explain normalization up to 3NF."
        b = "Explain the normalization process including 3NF."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        rel, _, conf, _ = classify_repeat_relationship_full(
            compute_text_similarity(n1, n2), n1, n2, a, b
        )
        self.assertIn(rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"})

    def test_os_related_not_repeat(self):
        a = "Explain deadlock prevention."
        b = "Explain deadlock detection and recovery."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        rel, _, _, _ = classify_repeat_relationship_full(
            compute_text_similarity(n1, n2), n1, n2, a, b
        )
        self.assertNotEqual(rel, "EXACT_REPEAT")

    def test_cn_compare_vs_explain_related(self):
        a = "Explain TCP protocol in detail."
        b = "Compare TCP and UDP."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        rel, _, _, reason = classify_repeat_relationship_full(
            compute_text_similarity(n1, n2), n1, n2, a, b
        )
        self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"}, reason)

    def test_syllabus_mapping_dynamic_unmapped_without_upload(self):
        mapping, conf = map_question_to_syllabus_index(
            "Explain normalization up to 3NF.",
            ["Normalization"],
            empty_syllabus_index("DBMS"),
        )
        self.assertEqual(mapping["module"], "Unmapped")
        self.assertEqual(conf, 0.0)

    def test_syllabus_mapping_from_uploaded_structure(self):
        index = build_syllabus_index_from_chunks(
            [
                {
                    "text": "Syllabus Document [x]\nModule 3: Normalization\n- 1NF\n- 2NF\n- 3NF\n- BCNF",
                    "metadata": {"unit": "Module 3: Normalization"},
                }
            ],
            subject="DBMS",
        )
        mapping, conf = map_question_to_syllabus_index(
            "Explain normalization up to 3NF.",
            ["Normalization", "3NF"],
            index,
        )
        self.assertEqual(mapping["module"], "Module 3")
        self.assertGreater(conf, 0.0)


class TestCrossSubjectIsolation(unittest.TestCase):
    """Critical: DL and DBMS workspaces never contaminate each other."""

    WS_DL = "ws-agnostic-dl"
    WS_DB = "ws-agnostic-dbms"

    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()
        cls.ingest = DynamicIngestPipeline(vector_store=cls.store)
        cls.engine = PYQIntelligenceEngine(vector_store=cls.store)
        cls.ws_db = WorkspaceDB()
        cls.store.delete_by_workspace(cls.WS_DL)
        cls.store.delete_by_workspace(cls.WS_DB)
        cls.tmpdir = tempfile.TemporaryDirectory()

        dl_pdf = os.path.join(cls.tmpdir.name, "DL_2024.pdf")
        make_pdf(
            dl_pdf,
            [
                "DEEP LEARNING EXAM 2024",
                "Q1(a) Explain CNN architecture in detail.",
                "Q1(b) Explain LSTM architecture.",
                "Q2(a) Explain Gradient Descent in neural networks.",
            ],
        )
        db_pdf = os.path.join(cls.tmpdir.name, "DBMS_2024.pdf")
        make_pdf(
            db_pdf,
            [
                "DATABASE MANAGEMENT SYSTEMS EXAM 2024",
                "Q1(a) Explain normalization up to 3NF.",
                "Q1(b) Explain ACID properties of transactions.",
                "Q2(a) Compare B-trees and B+ trees for indexing.",
            ],
        )
        os_pdf = os.path.join(cls.tmpdir.name, "OS_2023.pdf")
        make_pdf(
            os_pdf,
            [
                "OPERATING SYSTEMS EXAM 2023",
                "Q1(a) Explain deadlock prevention techniques.",
                "Q1(b) Explain paging and segmentation.",
            ],
        )
        cn_pdf = os.path.join(cls.tmpdir.name, "CN_2022.pdf")
        make_pdf(
            cn_pdf,
            [
                "COMPUTER NETWORKS EXAM 2022",
                "Q1(a) Compare TCP and UDP protocols.",
                "Q1(b) Explain sliding window protocol.",
            ],
        )

        ws_dl = cls.ws_db.get_or_create(cls.WS_DL, subject="Deep Learning", semester="Semester 7")
        ws_db = cls.ws_db.get_or_create(cls.WS_DB, subject="Database Management Systems", semester="Semester 5")
        cls.WS_OS = "ws-agnostic-os"
        cls.WS_CN = "ws-agnostic-cn"
        cls.store.delete_by_workspace(cls.WS_OS)
        cls.store.delete_by_workspace(cls.WS_CN)
        ws_os = cls.ws_db.get_or_create(cls.WS_OS, subject="Operating Systems", semester="Semester 4")
        ws_cn = cls.ws_db.get_or_create(cls.WS_CN, subject="Computer Networks", semester="Semester 5")

        cls.ingest.parse_pyq_pdf(dl_pdf, ws_dl)
        cls.ingest.parse_pyq_pdf(db_pdf, ws_db)
        cls.ingest.parse_pyq_pdf(os_pdf, ws_os)
        cls.ingest.parse_pyq_pdf(cn_pdf, ws_cn)

    @classmethod
    def tearDownClass(cls):
        for wid in (cls.WS_DL, cls.WS_DB, cls.WS_OS, cls.WS_CN):
            cls.store.delete_by_workspace(wid)
        cls.tmpdir.cleanup()

    def test_dl_intelligence_has_no_dbms(self):
        analysis = self.engine.get_pyq_analysis(self.WS_DL, subject="Deep Learning", include_source_questions=True)
        blob = " ".join(q["exact_text"].lower() for q in analysis.get("extracted_questions", []))
        self.assertIn("cnn", blob)
        self.assertNotIn("normalization", blob)
        self.assertNotIn("acid", blob)

    def test_dbms_intelligence_has_no_dl(self):
        analysis = self.engine.get_pyq_analysis(self.WS_DB, subject="Database Management Systems", include_source_questions=True)
        blob = " ".join(q["exact_text"].lower() for q in analysis.get("extracted_questions", []))
        self.assertIn("normalization", blob)
        self.assertNotIn("cnn", blob)
        self.assertNotIn("lstm", blob)

    def test_os_and_cn_work(self):
        os_a = self.engine.get_pyq_analysis(self.WS_OS, subject="Operating Systems")
        cn_a = self.engine.get_pyq_analysis(self.WS_CN, subject="Computer Networks")
        self.assertGreaterEqual(os_a["total_valid_questions"], 2)
        self.assertGreaterEqual(cn_a["total_valid_questions"], 2)
        self.assertEqual(os_a.get("extracted_questions"), [])
        self.assertEqual(cn_a.get("extracted_questions"), [])

    def test_reupload_dedup_same_workspace_only(self):
        """Re-ingest same filename into DBMS must not wipe DL vectors."""
        db_pdf = os.path.join(self.tmpdir.name, "DBMS_2024.pdf")
        ws_db = self.ws_db.get_by_id(self.WS_DB)
        before_dl = len(self.engine.get_source_questions(self.WS_DL))
        self.ingest.parse_pyq_pdf(db_pdf, ws_db)
        after_dl = len(self.engine.get_source_questions(self.WS_DL))
        self.assertEqual(before_dl, after_dl)
        self.assertGreaterEqual(len(self.engine.get_source_questions(self.WS_DB)), 2)

    def test_multi_pdf_aggregation_variable(self):
        ws_id = "ws-agnostic-multi"
        self.store.delete_by_workspace(ws_id)
        ws = self.ws_db.get_or_create(ws_id, subject="Algorithms")
        counts = [3, 5, 8]
        for i, c in enumerate(counts):
            lines = [f"ALGORITHMS PAPER {i}"]
            for j in range(1, c + 1):
                lines.append(f"Q1({chr(96+j)}) Explain algorithm technique set {i}-{j} with examples.")
            path = os.path.join(self.tmpdir.name, f"alg_{i}_{c}.pdf")
            make_pdf(path, lines)
            metas = self.ingest.parse_pyq_pdf(path, ws)
            self.assertEqual(len(metas), c)
        analysis = self.engine.get_pyq_analysis(ws_id)
        self.assertEqual(analysis["total_papers"], 3)
        self.assertEqual(analysis["total_valid_questions"], sum(counts))
        self.store.delete_by_workspace(ws_id)


class TestUnknownSubjectQuantumWidgets(unittest.TestCase):
    """
    Completely synthetic subject that must NEVER appear in production dictionaries.
    Proves the pipeline is subject-agnostic.
    """

    WS = "ws-quantum-widget-eng"

    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()
        cls.ingest = DynamicIngestPipeline(vector_store=cls.store)
        cls.engine = PYQIntelligenceEngine(vector_store=cls.store)
        cls.ws_db = WorkspaceDB()
        cls.store.delete_by_workspace(cls.WS)
        cls.tmpdir = tempfile.TemporaryDirectory()

        syl = os.path.join(cls.tmpdir.name, "QWE_syllabus.pdf")
        make_pdf(
            syl,
            [
                "QUANTUM WIDGET ENGINEERING SYLLABUS",
                "Unit 1: Widget Fundamentals",
                "- Widget atoms",
                "- Classical widgets",
                "Unit 2: Quantum Widgets",
                "- Quantum widget architecture",
                "- Quantum widget models",
                "Unit 3: Widget Optimization",
                "- Widget annealing",
                "- Coherence tuning",
            ],
        )
        pyq_a = os.path.join(cls.tmpdir.name, "QWE_2024_May.pdf")
        make_pdf(
            pyq_a,
            [
                "QUANTUM WIDGET ENGINEERING EXAM May 2024",
                "Q1(a) Explain quantum widget architecture.",
                "Q1(b) Compare quantum widget models.",
                "Q2(a) Explain widget annealing for optimization.",
            ],
        )
        pyq_b = os.path.join(cls.tmpdir.name, "QWE_2025_Nov.pdf")
        make_pdf(
            pyq_b,
            [
                "QUANTUM WIDGET ENGINEERING EXAM November 2025",
                "Q1(a) Explain quantum widget architecture.",
                "Q1(b) Describe coherence tuning methods.",
            ],
        )

        ws = cls.ws_db.get_or_create(
            cls.WS,
            subject="Quantum Widget Engineering",
            semester="Semester 8",
            university="Synthetic University",
        )
        cls.ingest.parse_syllabus_pdf(syl, ws)
        cls.ingest.parse_pyq_pdf(pyq_a, ws)
        cls.ingest.parse_pyq_pdf(pyq_b, ws)

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_by_workspace(cls.WS)
        cls.tmpdir.cleanup()

    def test_unknown_subject_end_to_end(self):
        analysis = self.engine.get_pyq_analysis(
            self.WS,
            subject="Quantum Widget Engineering",
            include_source_questions=True,
        )
        self.assertGreaterEqual(analysis["total_valid_questions"], 5)
        self.assertEqual(analysis["total_papers"], 2)

        blob = " ".join(q["exact_text"].lower() for q in analysis.get("extracted_questions", []))
        self.assertIn("quantum widget architecture", blob)

        # Exact repeat of architecture question across papers
        exact = analysis.get("exact_repeats") or []
        exact_blob = json_dump(exact).lower()
        self.assertIn("architecture", exact_blob)

        # Must not invent Module 1; Unmapped OR Unit from uploaded syllabus only
        for q in analysis.get("extracted_questions", []):
            m = str((q.get("syllabus_mapping") or {}).get("module", ""))
            self.assertFalse(
                re.match(r"^Module\s+1$", m, re.I) and "widget" not in m.lower(),
                f"Invented module mapping: {m}",
            )

        self.assertTrue(
            analysis.get("topic_recurrence") or analysis.get("study_priority") or exact,
            "Expected recurrence or priority from uploaded evidence",
        )


def json_dump(obj) -> str:
    return json.dumps(obj, default=str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
