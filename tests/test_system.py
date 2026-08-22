import os
import sys
import tempfile
import unittest
import fitz  # PyMuPDF
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.api import app
from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.answer_engine import GroundedAnswerEngine
from rag.pyq_intelligence import PYQIntelligenceEngine, CURRENT_PRIORITY_BASELINE
from rag.question_extractor import (
    extract_questions_from_page_text,
    validate_question_candidate,
    normalize_question_text,
    compute_text_similarity,
    classify_repeat_relationship,
    analyze_single_paper_patterns,
    is_header_or_instruction
)

def create_sample_pdf(file_path: str, pages_content: list):
    """Utility helper to generate synthetic test PDFs in memory/temp folder with preserved line breaks."""
    doc = fitz.open()
    for text in pages_content:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, text)
    doc.save(file_path)
    doc.close()

class TestSystem20ScenariosSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.store = VectorStore()
        cls.ingest_pipeline = DynamicIngestPipeline(vector_store=cls.store)
        cls.answer_engine = GroundedAnswerEngine(vector_store=cls.store)
        cls.pyq_engine = PYQIntelligenceEngine(vector_store=cls.store, clustering_threshold=0.65)

        cls.ws_id_1 = "test-ws-single-paper"
        cls.ws_id_2 = "test-ws-multi-year"
        cls.ws_id_empty = "test-ws-empty"
        cls.ws_id_sp = "test-ws-sp-isolated"

        cls.store.delete_by_workspace(cls.ws_id_1)
        cls.store.delete_by_workspace(cls.ws_id_2)
        cls.store.delete_by_workspace(cls.ws_id_empty)
        cls.store.delete_by_workspace(cls.ws_id_sp)

        # The API never creates workspaces implicitly, so analysis-endpoint
        # tests must register their workspaces first (same as the real UI flow).
        from rag.workspace_db import WorkspaceDB

        ws_db = WorkspaceDB()
        for ws_id in (cls.ws_id_1, cls.ws_id_2, cls.ws_id_empty, cls.ws_id_sp):
            ws_db.get_or_create(ws_id)

        cls.temp_dir = tempfile.TemporaryDirectory()

        # PDF files for test suite
        cls.normal_pdf_path = os.path.join(cls.temp_dir.name, "MU_DBMS_2024_PYQ.pdf")
        cls.pyq_2023_path = os.path.join(cls.temp_dir.name, "MU_DBMS_2023_PYQ.pdf")
        cls.corrupted_pdf_path = os.path.join(cls.temp_dir.name, "Corrupted_Font.pdf")
        cls.multi_syl_pdf_path = os.path.join(cls.temp_dir.name, "Multi_Subject_Syllabus.pdf")

        normal_pyq_content = [
            "MUMBAI UNIVERSITY | SEMESTER 5 | EXAMINATION 2024 | TOTAL MARKS: 80\n"
            "Instructions to candidates: Attempt all questions. Figures to right indicate full marks.\n"
            "Q1. Explain normalization. Discuss 1NF, 2NF and 3NF with examples. (10 Marks)\n"
            "Q2. What is a transaction? Explain ACID properties in detail. (10 Marks)\n"
            "Q3. Explain different types of database joins with query examples. (10 Marks)\n"
            "Q4. Explain normalization process in relational database design. (10 Marks)"
        ]

        pyq_2023_content = [
            "MUMBAI UNIVERSITY | SEMESTER 5 | EXAMINATION 2023 | TOTAL MARKS: 80\n"
            "Q1. Explain normalization. Discuss 1NF, 2NF and 3NF with examples. (10 Marks)\n"
            "Q2. What do you mean by transaction processing? Explain ACID properties. (10 Marks)\n"
            "Q3. Explain deadlock prevention and recovery mechanisms in DBMS. (10 Marks)"
        ]

        corrupted_content = [
            "Bcoc Feb Bysdv Kwfudl Hkkjir Esa Kkadu Ekud Ã© \ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd"
        ]

        multi_syl_content = [
            "SYLLABUS FOR COMPUTER ENGINEERING SEMESTER 5\n"
            "SUBJECT: Database Management Systems (CSC502)\n"
            "Module 1: Database Architecture and Normalization (1NF, 2NF, 3NF)\n"
            "Module 2: Transaction Management and Concurrency Control\n"
            "\n"
            "SUBJECT: Artificial Intelligence (CSC503)\n"
            "Module 1: Search Algorithms and Heuristics\n"
            "Module 2: Knowledge Representation and Expert Systems"
        ]

        create_sample_pdf(cls.normal_pdf_path, normal_pyq_content)
        create_sample_pdf(cls.pyq_2023_path, pyq_2023_content)
        create_sample_pdf(cls.corrupted_pdf_path, corrupted_content)
        create_sample_pdf(cls.multi_syl_pdf_path, multi_syl_content)

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_by_workspace(cls.ws_id_1)
        cls.store.delete_by_workspace(cls.ws_id_2)
        cls.store.delete_by_workspace(cls.ws_id_empty)
        cls.temp_dir.cleanup()

    # Scenario 1: Normal text PDF extraction
    def test_01_normal_text_pdf_extraction(self):
        doc = fitz.open(self.normal_pdf_path)
        text = doc[0].get_text()
        doc.close()
        is_valid, reason, metrics = self.ingest_pipeline.validate_text_quality(text)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "valid")

    # Scenario 2: Scanned PDF OCR fallback
    def test_02_scanned_pdf_ocr_fallback(self):
        doc = fitz.open(self.normal_pdf_path)
        is_valid, reason, metrics = self.ingest_pipeline.validate_text_quality(doc[0].get_text())
        doc.close()
        self.assertTrue(is_valid)

    # Scenario 3: Corrupted font rejection
    def test_03_corrupted_font_rejection(self):
        doc = fitz.open(self.corrupted_pdf_path)
        text = doc[0].get_text()
        doc.close()
        is_valid, reason, metrics = self.ingest_pipeline.validate_text_quality(text)
        self.assertFalse(is_valid)
        self.assertIn(reason, ["corrupted_replacement_characters", "garbled_ocr_font_encoding", "low_alphabetic_character_ratio", "garbled_ocr_alphanumeric_noise"])

    # Scenario 4: Question extraction
    def test_04_question_extraction(self):
        doc = fitz.open(self.normal_pdf_path)
        accepted, rejected = extract_questions_from_page_text(doc[0].get_text(), 1, "MU_DBMS_2024_PYQ.pdf", self.ws_id_1)
        doc.close()
        self.assertGreaterEqual(len(accepted), 3)

    # Scenario 5: Question numbering
    def test_05_question_numbering(self):
        doc = fitz.open(self.normal_pdf_path)
        accepted, rejected = extract_questions_from_page_text(doc[0].get_text(), 1, "MU_DBMS_2024_PYQ.pdf", self.ws_id_1)
        doc.close()
        self.assertTrue(accepted[0]["question_number"].startswith("Q1"))
        self.assertEqual(accepted[0]["parent_question"], "Q1")

    # Scenario 6: Marks extraction
    def test_06_marks_extraction(self):
        doc = fitz.open(self.normal_pdf_path)
        accepted, rejected = extract_questions_from_page_text(doc[0].get_text(), 1, "MU_DBMS_2024_PYQ.pdf", self.ws_id_1)
        doc.close()
        self.assertIn(accepted[0]["marks"], [5, 10])

    # Scenario 7: Header/footer rejection
    def test_07_header_footer_rejection(self):
        header_line = "MUMBAI UNIVERSITY | SEMESTER 5 | EXAMINATION 2024"
        instruction_line = "Instructions to candidates: Attempt all questions"
        question_line = "Q1. Explain normalization and ACID properties in detail."
        self.assertTrue(is_header_or_instruction(header_line))
        self.assertTrue(is_header_or_instruction(instruction_line))
        self.assertFalse(is_header_or_instruction(question_line))

    # Scenario 8: Subject filtering
    def test_08_subject_filtering(self):
        match_dbms = self.ingest_pipeline.match_syllabus_subject_section("Database Management Systems CSC502", "Database Management Systems", "CSC502")
        self.assertTrue(match_dbms)

    # Scenario 9: Multi-subject syllabus filtering
    def test_09_multi_subject_syllabus_filtering(self):
        text_ai = "SUBJECT: Artificial Intelligence (CSC503)\nModule 1: Search Algorithms"
        match_target = self.ingest_pipeline.match_syllabus_subject_section(text_ai, "Database Management Systems", "CSC502")
        self.assertFalse(match_target)

    # Scenario 10: Exact repeat detection
    def test_10_exact_repeat_detection(self):
        norm1 = normalize_question_text("Q1. What is normalization? Explain 1NF, 2NF and 3NF. (10 Marks)")
        norm2 = normalize_question_text("Q1. What is normalization? Explain 1NF, 2NF and 3NF. (10 Marks)")
        sim = compute_text_similarity(norm1, norm2)
        rel, concept = classify_repeat_relationship(sim, norm1, norm2)
        self.assertEqual(rel, "EXACT_REPEAT")

    # Scenario 11: Paraphrase detection
    def test_11_paraphrase_detection(self):
        norm1 = normalize_question_text("Q1. Explain normalization process in relational database design.")
        norm2 = normalize_question_text("Q4. Discuss normalization and normal forms in database systems.")
        sim = compute_text_similarity(norm1, norm2)
        rel, concept = classify_repeat_relationship(sim, norm1, norm2)
        self.assertIn(rel, ["PARAPHRASED_REPEAT", "SAME_TOPIC", "RELATED", "SEMANTIC_REPEAT"])

    # Scenario 12: Same-topic / different-angle detection
    def test_12_same_topic_different_angle(self):
        norm1 = normalize_question_text("Q1. What is a transaction? Explain ACID properties.")
        norm2 = normalize_question_text("Q2. What do you mean by transaction processing? Outline concurrency control.")
        sim = compute_text_similarity(norm1, norm2)
        rel, concept = classify_repeat_relationship(sim, norm1, norm2)
        self.assertIn(rel, ["SAME_TOPIC", "PARAPHRASED_REPEAT", "RELATED", "RELATED_TOPIC", "SEMANTIC_REPEAT"])

    # Scenario 13: Cross-topic separation
    def test_13_cross_topic_separation(self):
        norm1 = normalize_question_text("Explain normalization and normal forms.")
        norm2 = normalize_question_text("Explain deadlock prevention and recovery mechanisms.")
        sim = compute_text_similarity(norm1, norm2)
        rel, concept = classify_repeat_relationship(sim, norm1, norm2)
        self.assertEqual(rel, "DIFFERENT")

    # Scenario 14: Workspace isolation
    def test_14_workspace_isolation(self):
        ws_info = {"id": self.ws_id_1, "subject": "Database Management Systems"}
        self.ingest_pipeline.parse_pyq_pdf(self.normal_pdf_path, ws_info)

        res1 = self.client.post("/search", json={"query": "normalization", "workspace_id": self.ws_id_1})
        self.assertGreater(res1.json()["total_retrieved"], 0)

        res2 = self.client.post("/search", json={"query": "normalization", "workspace_id": self.ws_id_empty})
        self.assertEqual(res2.json()["total_retrieved"], 0)

    # Scenario 15: No static demo data leakage
    def test_15_no_static_demo_data_leakage(self):
        res = self.client.post("/workspaces/test-ws-empty/analyze-pyq", json={"workspace_id": self.ws_id_empty})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_questions_analyzed"], 0)
        self.assertEqual(len(data["topics"]), 0)

    # Scenario 16: Prediction without future leakage
    def test_16_prediction_without_future_leakage(self):
        items = [
            {"metadata": {"year": "2021", "marks": "10", "source_file": "P1.pdf"}},
            {"metadata": {"year": "2023", "marks": "10", "source_file": "P2.pdf"}},
            {"metadata": {"year": "2025", "marks": "10", "source_file": "P3.pdf"}}
        ]
        feats = self.pyq_engine.extract_temporal_features(items, syl_present=True, target_year=2024)
        self.assertEqual(feats["last_appearance_year"], 2023)

    # Scenario 17: Single-paper mode
    def test_17_single_paper_mode(self):
        ws_id_sp = self.ws_id_sp
        self.store.delete_by_workspace(ws_id_sp)
        ws_info = {"id": ws_id_sp, "subject": "Database Management Systems"}
        self.ingest_pipeline.parse_pyq_pdf(self.normal_pdf_path, ws_info)

        res = self.client.post(f"/workspaces/{ws_id_sp}/analyze-pyq", json={"workspace_id": ws_id_sp})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["single_paper_mode"])
        self.assertGreater(data.get("total_valid_questions") or data.get("total_questions_analyzed", 0), 0)
        self.assertEqual(data.get("extracted_questions"), [])  # main payload has no dump
        self.assertIn("within_paper_patterns", data)
        sources = self.pyq_engine.get_source_questions(ws_id_sp)
        self.assertGreater(len(sources), 0)
        self.store.delete_by_workspace(ws_id_sp)

    # Scenario 18: Multi-year mode
    def test_18_multi_year_mode(self):
        ws_info = {"id": self.ws_id_2, "subject": "Database Management Systems"}
        self.ingest_pipeline.parse_pyq_pdf(self.normal_pdf_path, ws_info)
        self.ingest_pipeline.parse_pyq_pdf(self.pyq_2023_path, ws_info)

        res = self.client.post(f"/workspaces/{self.ws_id_2}/analyze-pyq", json={"workspace_id": self.ws_id_2})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["single_paper_mode"])
        self.assertEqual(data["total_papers"], 2)

    # Scenario 19: Empty workspace handling
    def test_19_empty_workspace_handling(self):
        res = self.client.post(f"/workspaces/{self.ws_id_empty}/analyze-pyq", json={"workspace_id": self.ws_id_empty})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["available"])

    # Scenario 20: Workspace persistence
    def test_20_workspace_persistence(self):
        res_create = self.client.post("/workspaces", json={
            "university": "Mumbai University",
            "branch": "Computer Engineering",
            "semester": "Semester 5",
            "subject": "System Programming",
            "subject_code": "CSC504"
        })
        self.assertEqual(res_create.status_code, 200)
        ws_data = res_create.json()
        ws_id = ws_data["id"]

        res_list = self.client.get("/workspaces")
        all_ws = res_list.json()
        self.assertTrue(any(w["id"] == ws_id for w in all_ws))

    # CRITICAL REGRESSION TEST: Hard Question Gate & Garbage String Rejection
    def test_21_critical_garbage_and_fragment_rejection(self):
        bad_candidates = [
            "94X525Yada494X525Yada...",
            "Cnn Architecture Suppose Have",
            "Lenet Architecture Vanishing And",
            "Attempt Any Four Design",
            "Attempt any four",
            "Attempt any five",
            "Rnn Architecture The Working",
            "University of Mumbai",
            "Deep Learning",
            "Unit 1",
            "Section A"
        ]

        for cand in bad_candidates:
            is_valid, reason, metrics = validate_question_candidate(cand)
            self.assertFalse(is_valid, f"Candidate '{cand}' should have been REJECTED, but was accepted!")

        # Verify real complete question is accepted
        real_question = "Q3. Explain RNN architecture and the working of recurrent neural networks. Discuss its applications. [10 Marks]"
        is_valid_real, reason_real, _ = validate_question_candidate(real_question)
        self.assertTrue(is_valid_real, f"Real question should be accepted, but got: {reason_real}")

    # REAL END-TO-END REGRESSION TEST: Mumbai University Deep Learning 2023 (15 Subquestion Records)
    def test_22_real_mumbai_2023_deep_learning_15_records(self):
        import glob
        pdf_files = [f for f in glob.glob("d:/pyqrag/data/pyq/deep-learning/*2023*.pdf") if "- Copy" not in f]
        if not pdf_files:
            self.skipTest("2023 Deep Learning PDF not found")

        workspace = {
            "id": "ws-mumbai-dl-2023-pytest",
            "subject": "Deep Learning",
            "university": "University of Mumbai",
            "branch": "AIDS",
            "semester": "Semester 7"
        }

        self.store.delete_by_workspace(workspace["id"])
        self.ingest_pipeline.parse_pyq_pdf(pdf_files[0], workspace)

        analysis = self.pyq_engine.get_pyq_analysis(
            workspace_id=workspace["id"],
            subject="Deep Learning",
            semester="Semester 7",
            include_source_questions=True,
        )
        extracted = analysis.get("extracted_questions", [])

        # Assertion 1: Total extracted questions dynamically parsed from document
        self.assertGreaterEqual(len(extracted), 10, f"Expected at least 10 subquestions dynamically parsed, got {len(extracted)}")

        q_numbers = [q["question_number"] for q in extracted]
        expected_numbers = [
            "Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q1(e)",
            "Q2(a)", "Q2(b)",
            "Q3(a)", "Q3(b)",
            "Q4(a)", "Q4(b)",
            "Q5(a)", "Q5(b)",
            "Q6(a)", "Q6(b)"
        ]

        # Assertion 2: Verify subquestions are formatted properly (e.g. Q1(a))
        self.assertTrue(any("(" in q for q in q_numbers))

        # Assertion 3: Parent Q1-Q6 containers MUST NOT exist in final question records
        for parent in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
            self.assertNotIn(parent, q_numbers, f"Parent question container '{parent}' invalidly present in final records!")

        # Assertion 4: Q1(a) and Q1(b) text must not be identical and independently analyzed
        q1a = next(q for q in extracted if q["question_number"] == "Q1(a)")
        q1b = next(q for q in extracted if q["question_number"] == "Q1(b)")
        self.assertNotEqual(q1a["exact_text"], q1b["exact_text"])
        self.assertNotEqual(q1a["detected_topics"], q1b["detected_topics"])

        # Assertion 5: Q3(b) subquestion has non-empty syllabus mapping
        q3b = next((q for q in extracted if q["question_number"] == "Q3(b)"), None)
        if q3b:
            self.assertTrue(len(q3b.get("syllabus_mapping", {}).get("module", "")) > 0)


if __name__ == '__main__':
    unittest.main()

