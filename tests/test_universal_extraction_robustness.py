"""
Unit tests for Universal PYQ Extraction — Final Master Hardening Pass.
Verifies:
1. Question-level partial recovery (1 bad candidate doesn't poison valid candidates).
2. Body-first recovery for damaged/weak markers with strong question bodies.
3. Adaptive failure recovery pass when 0 questions are initially extracted.
4. Candidate lifecycle status tracking (DETECTED, VALIDATED, GROUNDED, ADMITTED, REJECTED, AMBIGUOUS).
5. Debug audit report structure and exact rejection reason tracking.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_robustness")

from rag.hybrid_question_extraction import run_universal_reconciliation_pipeline, hybrid_extract_document
from rag.evidence_fusion import evaluate_question_level_evidence, body_strength_score, noise_penalty_score


class TestUniversalExtractionRobustness(unittest.TestCase):
    def test_partial_recovery_one_bad_candidate_does_not_poison_valid(self):
        """
        Verify Critical Requirement 2: Q1(a), Q1(c), Q1(e) are valid, Q2 is ambiguous/damaged.
        Result MUST be PARTIAL with 3 valid questions, NEVER FAILED with 0 questions.
        """
        raw_text = (
            "Q1(a) Explain the fundamental architecture of Convolutional Neural Networks. [10]\n"
            "Q1(c) Derive the backpropagation equations for Recurrent Neural Networks. [10]\n"
            "Q1(e) Compare supervised and unsupervised learning algorithms with examples. [10]\n"
            "Q2(ambiguous_garbage_noise_without_valid_body_or_text)\n"
        )
        pages = [
            {
                "page": 1,
                "raw_native_text": raw_text,
                "raw_ocr_text": raw_text,
                "reconstructed_text": raw_text,
            }
        ]
        res = run_universal_reconciliation_pipeline(
            pages,
            filename="test_partial_paper.pdf",
            workspace_id="ws-test-partial",
            subject="Deep Learning",
            year=2024,
        )
        accepted = res.get("accepted_questions") or []
        extracted_ids = [q["question_id"] for q in accepted]
        quality = res.get("quality") or {}
        status = quality.get("extraction_quality")

        self.assertGreaterEqual(len(accepted), 3)
        self.assertIn("Q1(a)", extracted_ids)
        self.assertIn("Q1(c)", extracted_ids)
        self.assertIn("Q1(e)", extracted_ids)
        self.assertNotEqual(status, "FAILED")
        self.assertIn(status, ("COMPLETE", "RECOVERED", "PARTIAL"))

    def test_body_first_recovery_imperative_verbs(self):
        """
        Verify Section 3 & 4: Question body with academic commands (Define, Explain, Calculate)
        and layout support is admitted even without standard '?' or clean markers.
        """
        b_score = body_strength_score("Explain deadlock prevention and avoidance algorithms in operating systems")
        self.assertGreaterEqual(b_score, 0.6)

        b_score_calc = body_strength_score("Calculate the time complexity of QuickSort in worst case")
        self.assertGreaterEqual(b_score_calc, 0.6)

        b_score_def = body_strength_score("Define database normalization up to Third Normal Form")
        self.assertGreaterEqual(b_score_def, 0.6)

    def test_candidate_status_lifecycle_and_evidence(self):
        """
        Verify Section 1 & 2: Candidate status assigned correctly (ADMITTED, REJECTED, AMBIGUOUS).
        """
        cand_valid = {
            "question_id": "Q1(a)",
            "exact_text": "Explain the concept of page replacement algorithms in operating systems.",
            "grounding_score": 0.95,
            "extraction_method": "ocr_layout",
            "parent_question": "Q1",
        }
        eval_valid = evaluate_question_level_evidence(
            cand_valid,
            source_blob="Q1(a) Explain the concept of page replacement algorithms in operating systems.",
            all_ids=["Q1(a)", "Q1(b)"],
            known_parents={"Q1"},
        )
        self.assertEqual(eval_valid["status"], "ADMITTED")

        cand_noisy = {
            "question_id": "Q99(z)",
            "exact_text": "Duration 3 hours Page 1 of 4 University Code 12345",
            "grounding_score": 0.9,
            "extraction_method": "ocr_text",
        }
        eval_noisy = evaluate_question_level_evidence(
            cand_noisy,
            source_blob="Duration 3 hours Page 1 of 4 University Code 12345",
            all_ids=["Q99(z)"],
        )
        self.assertEqual(eval_noisy["status"], "REJECTED")

    def test_adaptive_failure_recovery_pass(self):
        """
        Verify Section 25: When initial pass has non-standard marker format but valid source text,
        adaptive failure recovery pass recovers grounded questions before declaring document status.
        """
        raw_text = (
            "Examination 2024\n"
            "Question 1 Attempt any two:\n"
            "Stem A: Explain the architecture of Transformer model in NLP. [10]\n"
            "Stem B: Derive the loss function for logistic regression. [10]\n"
        )
        pages = [
            {
                "page": 1,
                "raw_native_text": raw_text,
                "raw_ocr_text": raw_text,
                "reconstructed_text": raw_text,
            }
        ]
        res = hybrid_extract_document(
            pages,
            filename="unseen_exam.pdf",
            workspace_id="ws-adaptive-test",
            subject="AI",
            year=2024,
        )
        accepted = res.get("accepted_questions") or []
        quality = res.get("quality") or {}
        status = quality.get("extraction_quality")

        self.assertGreater(len(accepted), 0)
        self.assertNotEqual(status, "FAILED")

    def test_debug_report_audit_structure(self):
        """
        Verify Section 29: Debug report structure is produced with all required keys.
        """
        raw_text = "Q1(a) Discuss database transactions and ACID properties. [10]\n"
        pages = [
            {
                "page": 1,
                "raw_native_text": raw_text,
                "raw_ocr_text": raw_text,
                "reconstructed_text": raw_text,
            }
        ]
        res = run_universal_reconciliation_pipeline(
            pages,
            filename="test_debug_report.pdf",
            workspace_id="ws-debug-report",
            subject="DBMS",
            year=2023,
        )
        audit = res.get("extraction_audit") or {}
        report = audit.get("debug_report") or res.get("debug_report") or {}

        self.assertIsNotNone(report)
        self.assertEqual(report.get("FILE"), "test_debug_report.pdf")
        self.assertIn("STATUS", report)
        self.assertIn("RECOVERED_QUESTIONS", report)
        self.assertIn("AMBIGUOUS_MARKERS", report)
        self.assertIn("REJECTED_MARKERS", report)
        self.assertIn("MISSING_SOURCE_PROVEN_QUESTIONS", report)
        self.assertIn("FABRICATED_IDS", report)
        self.assertIn("DUPLICATES", report)

    def test_multi_signal_confidence_breakdown(self):
        """
        Verify Section 27: Every canonical question carries marker_confidence,
        body_confidence, layout_confidence, grounding_confidence, overall_confidence,
        bounding_region, and marker_provenance.
        """
        raw_text = "Q1(a) Explain the difference between process and thread in operating systems. [10]\n"
        pages = [
            {
                "page": 1,
                "raw_native_text": raw_text,
                "raw_ocr_text": raw_text,
                "reconstructed_text": raw_text,
            }
        ]
        res = run_universal_reconciliation_pipeline(
            pages,
            filename="test_confidence_breakdown.pdf",
            workspace_id="ws-conf-test",
            subject="OS",
            year=2024,
        )
        accepted = res.get("accepted_questions") or []
        self.assertGreaterEqual(len(accepted), 1)
        q = accepted[0]
        self.assertIn("marker_confidence", q)
        self.assertIn("body_confidence", q)
        self.assertIn("layout_confidence", q)
        self.assertIn("grounding_confidence", q)
        self.assertIn("overall_confidence", q)
        self.assertIn("bounding_region", q)
        self.assertIn("marker_provenance", q)

    def test_universal_missing_sibling_recovery(self):
        """
        Verify missing sibling gap recovery:
        Q6(a), Q6(b), Q6(d), Q6(e) exist, Q6(c) has garbled OCR marker ('Role of DBA [05]').
        Missing sibling recovery recovers Q6(c) while suppressing non-existent Q6(f)-Q6(i).
        """
        text = (
            "Q1(a) Explain fundamental database management concepts. [10]\n"
            "Q2(a) Describe DBMS architecture with diagram. [10]\n"
            "Q3(a) Discuss SQL queries and normalization. [10]\n"
            "Q4(a) Consider employee database and write queries. [10]\n"
            "Q5(a) Explain ACID properties with suitable example. [10]\n"
            "Q6 Solve any four out of the following [20]\n"
            "a Conversion of Specialization to relational schema with suitable example [05]\n"
            "b Log based recovery [05]\n"
            "Role of DBA [05]\n"
            "d Triggers [05]\n"
            "e Types of attributes [05]\n"
        )
        pages = [
            {
                "page": 1,
                "raw_native_text": text,
                "raw_ocr_text": text,
                "reconstructed_text": text,
            }
        ]
        res = run_universal_reconciliation_pipeline(
            pages,
            filename="test_missing_sibling.pdf",
            workspace_id="ws-sibling-test",
            subject="DBMS",
            year=2024,
        )
        accepted_ids = [q.get("question_id") for q in (res.get("accepted_questions") or [])]
        self.assertIn("Q6(c)", accepted_ids)
        self.assertIn("Q6(a)", accepted_ids)
        self.assertIn("Q6(b)", accepted_ids)
        self.assertIn("Q6(d)", accepted_ids)
        self.assertIn("Q6(e)", accepted_ids)
        # Anti-hallucination: Q6(f) and beyond MUST NOT be fabricated
        self.assertNotIn("Q6(f)", accepted_ids)
        self.assertNotIn("Q6(g)", accepted_ids)
        self.assertNotIn("Q6(h)", accepted_ids)


if __name__ == "__main__":
    unittest.main()


