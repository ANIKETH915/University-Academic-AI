"""
Generic Test Suite for Source-Proven Marker Verification Layer.

Verifies that OCR noise (e.g. Q6(t), Q5(q), Q5(t)) is rejected and never enters missing_questions,
while source-proven markers pass cleanly across all 12 specified test scenarios.
"""

import unittest
from rag.hybrid_question_extraction import (
    hybrid_extract_document,
    investigate_subquestion_marker_gaps,
    compute_extraction_quality
)
from rag.question_extractor import extract_questions_from_page_text, prepare_page_text_for_extraction


class TestSourceProvenMarkerModel(unittest.TestCase):

    def test_case_1_ocr_invents_q6_t_source_does_not_contain_it(self):
        """1. OCR invents Q6(t), source does not contain it -> Q6(t) NOT missing."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q6. Attempt any two\na Explain XML structure.\nb Explain Servlet life cycle.",
            "raw_ocr_text": "Q6. Attempt any two\na Explain XML structure.\nb Explain Servlet life cycle.\nt garbage noise",
            "reconstructed_text": "Q6. Attempt any two\na Explain XML structure.\nb Explain Servlet life cycle.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        missing = res["quality"].get("missing_questions", [])
        self.assertNotIn("Q6(t)", missing)
        self.assertEqual(res["quality"].get("extraction_quality"), "COMPLETE")

    def test_case_2_source_contains_q6_t_extraction_misses_it(self):
        """2. Source contains Q6(t), extraction misses it -> Q6(t) IS missing."""
        classified_markers = [
            {"marker_id": "Q6(a)", "genuine": True},
            {"marker_id": "Q6(b)", "genuine": True},
            {"marker_id": "Q6(c)", "genuine": True},
            {"marker_id": "Q6(t)", "genuine": True},
        ]
        extracted = [
            {"question_id": "Q6(a)"},
            {"question_id": "Q6(b)"},
            {"question_id": "Q6(c)"},
        ]
        quality = compute_extraction_quality(
            [q["question_id"] for q in extracted],
            [c["marker_id"] for c in classified_markers if c["genuine"]]
        )
        self.assertIn("Q6(t)", quality.get("missing_questions", []))

    def test_case_3_source_contains_q1_a_q1_c_no_q1_b(self):
        """3. Source contains Q1(a), Q1(c), no Q1(b) -> COMPLETE."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q1(a) Explain HTML5 audio.\nQ1(c) Discuss React JS advantages.",
            "reconstructed_text": "Q1(a) Explain HTML5 audio.\nQ1(c) Discuss React JS advantages.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertEqual(set(extracted_ids), {"Q1(a)", "Q1(c)"})
        self.assertEqual(res["quality"].get("extraction_quality"), "COMPLETE")

    def test_case_4_ocr_produces_q5_q_from_watermark_noise(self):
        """4. OCR produces Q5(q) from watermark noise -> reject as ungrounded."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q5. Answer the following:\na Write JavaScript password validation code.\nb What are features of React JS.",
            "raw_ocr_text": "q noise watermark\nQ5. Answer the following:\na Write JavaScript password validation code.\nb What are features of React JS.",
            "reconstructed_text": "Q5. Answer the following:\na Write JavaScript password validation code.\nb What are features of React JS.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        missing = res["quality"].get("missing_questions", [])
        self.assertNotIn("Q5(q)", missing)

    def test_case_5_q6_a_b_only(self):
        """5. Q6(a-b) only -> COMPLETE if both are source-proven."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q6.\na Explain structure of XML Document with example.\nb Explain Servlet life cycle with neat diagram.",
            "reconstructed_text": "Q6.\na Explain structure of XML Document with example.\nb Explain Servlet life cycle with neat diagram.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertEqual(set(extracted_ids), {"Q6(a)", "Q6(b)"})
        self.assertEqual(res["quality"].get("extraction_quality"), "COMPLETE")

    def test_case_6_q1_a_f(self):
        """6. Q1(a-f) -> all six recovered."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q1. Answer any five:\n(a) Explain A.\n(b) Explain B.\n(c) Explain C.\n(d) Explain D.\n(e) Explain E.\n(f) Explain F.",
            "reconstructed_text": "Q1. Answer any five:\n(a) Explain A.\n(b) Explain B.\n(c) Explain C.\n(d) Explain D.\n(e) Explain E.\n(f) Explain F.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertEqual(len(extracted_ids), 6)
        self.assertIn("Q1(f)", extracted_ids)

    def test_case_7_q1_a_z_no_hardcoded_ae_limitation(self):
        """7. Q1(a-g) -> no hardcoded a-e limitation."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q1. Answer any five:\na Explain A.\nb Explain B.\nc Explain C.\nd Explain D.\ne Explain E.\nf Explain F.\ng Explain G.",
            "reconstructed_text": "Q1. Answer any five:\na Explain A.\nb Explain B.\nc Explain C.\nd Explain D.\ne Explain E.\nf Explain F.\ng Explain G.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q1(f)", extracted_ids)
        self.assertIn("Q1(g)", extracted_ids)

    def test_case_8_roman_subquestions(self):
        """8. Roman subquestions -> generic handling."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q2. Attempt the following:\n(i) Explain paging mechanism.\n(ii) Explain segmentation.\n(iii) Explain virtual memory.",
            "reconstructed_text": "Q2. Attempt the following:\n(i) Explain paging mechanism.\n(ii) Explain segmentation.\n(iii) Explain virtual memory.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q2(i)", extracted_ids)
        self.assertIn("Q2(ii)", extracted_ids)
        self.assertIn("Q2(iii)", extracted_ids)

    def test_case_9_dotted_markers(self):
        """9. Dotted markers -> generic handling."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q.1. Answer the following:\na. Explain HTML5.\nb. Explain DOM.",
            "reconstructed_text": "Q.1. Answer the following:\na. Explain HTML5.\nb. Explain DOM.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertEqual(set(extracted_ids), {"Q1(a)", "Q1(b)"})

    def test_case_10_cross_page_questions(self):
        """10. Cross-page questions -> preserve source structure."""
        pages = [
            {
                "page": 1,
                "raw_native_text": "Q3(a) Explain process scheduling algorithms in detail.",
                "reconstructed_text": "Q3(a) Explain process scheduling algorithms in detail.",
                "ocr_used": False,
            },
            {
                "page": 2,
                "raw_native_text": "Q3(b) Write short note on deadlocks.",
                "reconstructed_text": "Q3(b) Write short note on deadlocks.",
                "ocr_used": False,
            }
        ]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q3(a)", extracted_ids)
        self.assertIn("Q3(b)", extracted_ids)

    def test_case_11_parent_instruction_without_child_marker(self):
        """11. Parent instruction without child marker -> never invent child IDs."""
        pages = [{
            "page": 1,
            "raw_native_text": "Q6. Attempt any Four questions out of the following topics.",
            "reconstructed_text": "Q6. Attempt any Four questions out of the following topics.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        missing = res["quality"].get("missing_questions", [])
        self.assertNotIn("Q6(a)", missing)

    def test_case_12_heavy_watermark_font_garbage(self):
        """12. Heavy watermark/font garbage -> valid question text still recoverable."""
        pages = [{
            "page": 1,
            "raw_native_text": "X237Y82735CX237Y82735CX237Y82735C\nQ1. Attempt any FOUR\na Explain audio and video controls of HTML5.\nb Explain Document Object Model.",
            "reconstructed_text": "X237Y82735CX237Y82735CX237Y82735C\nQ1. Attempt any FOUR\na Explain audio and video controls of HTML5.\nb Explain Document Object Model.",
            "ocr_used": False,
        }]
        res = hybrid_extract_document(pages, filename="test.pdf", workspace_id="ws")
        extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
        self.assertIn("Q1(a)", extracted_ids)
        self.assertIn("Q1(b)", extracted_ids)


if __name__ == "__main__":
    unittest.main()
