"""Cross-page question continuation: one question, both pages, no truncation."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.hybrid_question_extraction import (
    hybrid_extract_document,
    leading_continuation_text,
    merge_cross_page_continuations,
)


def page(no: int, text: str) -> dict:
    return {
        "page": no,
        "raw_native_text": text,
        "raw_ocr_text": "",
        "reconstructed_text": text,
        "ocr_used": False,
    }


class TestLeadingContinuationDetection(unittest.TestCase):
    def test_lowercase_fragment_is_continuation(self):
        self.assertEqual(
            leading_continuation_text("its advantages, limitations and applications.\nQ5(a) Define x."),
            "its advantages, limitations and applications.",
        )

    def test_new_question_is_not_continuation(self):
        self.assertEqual(
            leading_continuation_text("Q5(a) Define a mutex properly."), ""
        )

    def test_capitalised_new_sentence_is_not_continuation(self):
        self.assertEqual(
            leading_continuation_text("Explain something new here.\nQ5(a) Define x."), ""
        )

    def test_empty_input(self):
        self.assertEqual(leading_continuation_text(""), "")


class TestMergeAcrossPages(unittest.TestCase):
    def _pages(self):
        return [
            page(1, "Q4(a) Define a race condition.\nQ4(b) Explain the working of the algorithm and discuss"),
            page(2, "its advantages, limitations and applications.\nQ5(a) Define a mutex lock."),
        ]

    def test_single_question_spans_two_pages(self):
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(
                self._pages(), filename="x.pdf", workspace_id="ws", year=0
            )
        accepted = result["accepted_questions"]
        ids = [q["question_id"] for q in accepted]
        self.assertEqual(ids.count("Q4(b)"), 1)

        q4b = next(q for q in accepted if q["question_id"] == "Q4(b)")
        self.assertIn("advantages", q4b["exact_text"].lower())
        self.assertIn("applications", q4b["exact_text"].lower())
        self.assertEqual(q4b["source_pages"], [1, 2])
        self.assertEqual(q4b["source_page_start"], 1)
        self.assertEqual(q4b["source_page_end"], 2)
        self.assertTrue(q4b.get("cross_page_merged"))

    def test_no_orphan_fragment_question_created(self):
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(
                self._pages(), filename="x.pdf", workspace_id="ws", year=0
            )
        texts = [q["exact_text"].strip().lower() for q in result["accepted_questions"]]
        self.assertFalse(
            any(t.startswith("its advantages") for t in texts),
            "continuation fragment must not become its own question",
        )

    def test_merge_count_reported(self):
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(
                self._pages(), filename="x.pdf", workspace_id="ws", year=0
            )
        self.assertEqual(result["cross_page_merges"], 1)

    def test_completed_question_is_not_extended(self):
        pages = [
            page(1, "Q4(a) Define a race condition.\nQ4(b) Explain the algorithm fully."),
            page(2, "some trailing footer text here.\nQ5(a) Define a mutex lock."),
        ]
        questions = [
            {"question_id": "Q4(b)", "exact_text": "Explain the algorithm fully.", "source_pages": [1]}
        ]
        merged, count = merge_cross_page_continuations(questions, pages)
        self.assertEqual(count, 0)
        self.assertEqual(merged[0]["exact_text"], "Explain the algorithm fully.")

    def test_single_page_document_unchanged(self):
        pages = [page(1, "Q1(a) Explain something in detail.")]
        questions = [{"question_id": "Q1(a)", "exact_text": "Explain something in detail.", "source_pages": [1]}]
        merged, count = merge_cross_page_continuations(questions, pages)
        self.assertEqual(count, 0)
        self.assertEqual(len(merged), 1)

    def test_leading_letter_sub_inherits_previous_parent(self):
        pages = [
            page(1, "Q5 a)\nDetermine communities for the given social network graph.\n"),
            page(2, "Page 2 of 2\nb)\nList and discuss various types of data structures.\nQ6 a)\nExplain collaborative filtering with an example.\n"),
        ]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(
                pages, filename="x.pdf", workspace_id="ws", year=0
            )
        ids = [q["question_id"] for q in result["accepted_questions"]]
        self.assertEqual(ids, ["Q5(a)", "Q5(b)", "Q6(a)"])


class TestMultiPagePaper(unittest.TestCase):
    def test_three_page_paper_all_questions_kept(self):
        pages = [
            page(1, "Q1(a) Explain concept one in detail.\nQ1(b) Explain concept two in detail."),
            page(2, "Q2(a) Explain concept three in detail.\nQ2(b) Explain concept four and describe"),
            page(3, "the remaining considerations in practice.\nQ3(a) Explain concept five in detail."),
        ]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(
                pages, filename="x.pdf", workspace_id="ws", year=0
            )
        ids = [q["question_id"] for q in result["accepted_questions"]]
        self.assertEqual(ids, ["Q1(a)", "Q1(b)", "Q2(a)", "Q2(b)", "Q3(a)"])
        q2b = next(q for q in result["accepted_questions"] if q["question_id"] == "Q2(b)")
        self.assertEqual(q2b["source_pages"], [2, 3])
        self.assertIn("remaining considerations", q2b["exact_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
