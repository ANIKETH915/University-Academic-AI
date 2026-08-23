"""
Unit tests for Grounding Gate validation, Canonical Question records,
Incomplete extraction safety, and Vector Transaction safety.
"""

import unittest
from rag.hybrid_question_extraction import text_grounded_in_source, validate_grounded_questions
from rag.pyq_intelligence import calculate_deterministic_priority_score


class TestGroundingGateAndCanonical(unittest.TestCase):
    def test_grounding_gate_accepts_valid_source_text(self):
        """Source text present in PDF passes grounding validation with high score."""
        source_blob = (
            "Question 1. (a) Explain the architecture of MapReduce with a neat diagram. "
            "Discuss the role of Mapper and Reducer functions in detail. [10 marks]"
        )
        cand_text = "Explain the architecture of MapReduce with a neat diagram. Discuss the role of Mapper and Reducer functions in detail."
        ok, ratio, reason = text_grounded_in_source(cand_text, source_blob)

        self.assertTrue(ok)
        self.assertGreaterEqual(ratio, 0.85)

    def test_grounding_gate_rejects_hallucinated_text(self):
        """Text invented by LLM/rule not in source blob is rejected by the Grounding Gate."""
        source_blob = "1. (a) Define paging and segmentation in Operating Systems. [5 marks]"
        hallucinated_text = "Explain the complete mathematical derivation of Bloom Filter false positive probability."
        ok, ratio, reason = text_grounded_in_source(hallucinated_text, source_blob)

        self.assertFalse(ok)
        self.assertEqual(reason, "invented_or_ungrounded_text")

    def test_validate_grounded_questions_filter(self):
        """validate_grounded_questions splits candidates into accepted vs rejected lists strictly."""
        source_blob = "Q1(a) Explain Dijkstra algorithm with example. [10 marks]"
        candidates = [
            {"question_id": "Q1(a)", "exact_text": "Explain Dijkstra algorithm with example."},
            {"question_id": "Q1(b)", "exact_text": "Invented topic about Quantum Computing in classical OS paper."},
        ]

        accepted, rejected = validate_grounded_questions(candidates, source_blob)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["question_id"], "Q1(a)")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["question_id"], "Q1(b)")

    def test_priority_score_calculation(self):
        """Priority score incorporates frequency, distinct years, exact/semantic repeats, and marks."""
        score, breakdown = calculate_deterministic_priority_score(
            appearances_count=4,
            distinct_years=3,
            exact_repeat_count=2,
            max_marks=10,
            last_year=2023,
            current_year=2026,
            semantic_repeat_count=1,
            syllabus_mapped=True,
            extraction_confidence=1.0,
        )
        self.assertGreater(score, 50.0)
        self.assertIn("frequency_score", breakdown)
        self.assertIn("year_recurrence_score", breakdown)
        self.assertIn("exact_repeat_score", breakdown)


if __name__ == "__main__":
    unittest.main()
