"""
Generic Question-Structure Identification Test Matrix.

Verifies universal identification of parent questions, instructions,
subquestion markers, and subquestion body boundaries.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.hybrid_question_extraction import hybrid_extract_document
from rag.question_structure import is_parent_instruction_line, split_embedded_subquestions


def _extract(text: str) -> dict:
    pages = [
        {
            "page": 1,
            "raw_native_text": text,
            "raw_ocr_text": "",
            "reconstructed_text": text,
            "ocr_used": False,
        }
    ]
    with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
        return hybrid_extract_document(
            pages, filename="test_generic.pdf", workspace_id="ws-gen", year=2024
        )


class TestGenericQuestionStructure(unittest.TestCase):

    def test_case_a_q1_solve_any_four(self):
        text = (
            "Q1. Solve any Four\n"
            "a. Explain the extreme programming lifecycle with diagram.\n"
            "b. Explain the development use case model.\n"
            "c. Difference between Alpha and Beta Testing.\n"
            "d. What is Software Configuration Management?\n"
            "e. Explain Six Sigma principles for software engineering.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        for expected in ("Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q1(e)"):
            self.assertIn(expected, accepted)

    def test_case_b_q6_solve_any_four(self):
        text = (
            "Q6. Solve any Four\n"
            "a. Explain the XP development cycle.\n"
            "b. Explain the use case model.\n"
            "c. Difference between Alpha and Beta testing.\n"
            "d. What is SCM?\n"
            "e. Explain Six Sigma for software engineering.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        for expected in ("Q6(a)", "Q6(b)", "Q6(c)", "Q6(d)", "Q6(e)"):
            self.assertIn(expected, accepted)

    def test_case_c_q10_attempt_any_three(self):
        text = (
            "Q10. Attempt any Three\n"
            "a. Explain convolutional layers in detail.\n"
            "b. Discuss recurrent neural networks and GRU cells.\n"
            "c. Evaluate transformer self attention mechanisms.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        for expected in ("Q10(a)", "Q10(b)", "Q10(c)"):
            self.assertIn(expected, accepted)

    def test_case_d_q20_subquestions(self):
        text = (
            "Q20(a) Describe distributed ledger technology.\n"
            "Q20(b) Discuss proof of work vs proof of stake consensus.\n"
            "Q20(c) Explain smart contract execution model.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        for expected in ("Q20(a)", "Q20(b)", "Q20(c)"):
            self.assertIn(expected, accepted)

    def test_case_e_q_dot_10_subquestions(self):
        text = (
            "Q.10(a) Describe natural language tokenization.\n"
            "Q.10(b) Discuss n-gram language models.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q10(a)", accepted)
        self.assertIn("Q10(b)", accepted)

    def test_case_f_bare_10_subquestions(self):
        text = (
            "10(a) Explain software design patterns.\n"
            "10(b) Discuss microservice architecture.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q10(a)", accepted)
        self.assertIn("Q10(b)", accepted)

    def test_case_g_non_contiguous_gaps(self):
        text = (
            "Q1. Answer any three:\n"
            "Q1(a) Explain database normalization.\n"
            "Q1(c) Discuss ACID properties of transactions.\n"
            "Q1(e) Explain indexing B-trees in relational databases.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q1(a)", accepted)
        self.assertIn("Q1(c)", accepted)
        self.assertIn("Q1(e)", accepted)
        self.assertNotIn("Q1(b)", accepted)
        self.assertNotIn("Q1(d)", accepted)
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_case_h_roman_subquestions(self):
        text = (
            "Q2(i) Define operating system process state transition.\n"
            "Q2(ii) Discuss process scheduling algorithms.\n"
            "Q2(iii) Explain deadlock prevention techniques.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q2(i)", accepted)
        self.assertIn("Q2(ii)", accepted)
        self.assertIn("Q2(iii)", accepted)

    def test_case_i_cross_page_subquestions(self):
        pages = [
            {
                "page": 1,
                "raw_native_text": "Q6. Attempt any two:\nQ6(a) Explain system architecture.",
                "raw_ocr_text": "",
                "reconstructed_text": "Q6. Attempt any two:\nQ6(a) Explain system architecture.",
                "ocr_used": False,
            },
            {
                "page": 2,
                "raw_native_text": "Q6(b) Explain data pipeline design.",
                "raw_ocr_text": "",
                "reconstructed_text": "Q6(b) Explain data pipeline design.",
                "ocr_used": False,
            },
        ]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            res = hybrid_extract_document(pages, filename="cross.pdf", workspace_id="ws-cross", year=2024)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q6(a)", accepted)
        self.assertIn("Q6(b)", accepted)

    def test_case_j_ocr_punctuation_loss(self):
        text = (
            "Q4. Answer the following:\n"
            "a Explain object oriented software engineering principles.\n"
            "b Describe unified modeling language sequence diagrams.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q4(a)", accepted)
        self.assertIn("Q4(b)", accepted)

    def test_case_m_table_cells_not_split_as_false_markers(self):
        text = (
            "Q5(a) Explain the following table data:\n"
            "Product Monday Tuesday\n"
            "A 10 20\n"
            "B 15 25\n"
            "C 30 40\n"
            "Q5(b) Discuss the vector representations derived from table.\n"
        )
        res = _extract(text)
        accepted = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q5(a)", accepted)
        self.assertIn("Q5(b)", accepted)
        self.assertNotIn("Q5(c)", accepted)


if __name__ == "__main__":
    unittest.main()
