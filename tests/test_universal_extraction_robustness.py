"""
Regression tests for universal extraction fixes:

1. Scanned papers whose subquestions are terse topics under an explicit
   instruction frame ("Write short notes on …" → "a. Ripple", "c. Corda").
2. Parent instruction frames never hijack a printed subquestion's ID.
3. Numbered N.B.-style instruction lists ("2. Answer any three out of…")
   are furniture, not question parents — while choice-parents
   ("Q.1 Solve any Four …") stay boundaries.
4. Glued OCR marker leads ("d.Blockchain for DeFi") parse as markers.
5. Status vocabulary: PARTIAL (not REVIEW_REQUIRED) when genuine markers
   are detected but not recovered.
6. Multi-resolution OCR evidence: a second high-DPI pass is reconciled,
   never blindly concatenated.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")

from rag.hybrid_question_extraction import (
    compute_extraction_quality,
    hybrid_extract_document,
)
from rag.question_extractor import (
    extract_questions_from_page_text,
    fix_ocr_question_glyphs,
    is_header_or_instruction,
    is_instruction_frame_text,
    prepare_page_text_for_extraction,
)


SHORT_NOTES_PAPER = """
N.B.: 1. Question No. 1 is compulsory.
2. Answer any three out of the remaining questions.
3. Assume suitable data if necessary.
4. Figures to the right indicate full marks.

Ql. Attempt the following (any 4): (20)
a. Distinguish between public, private, and consortium blockchain.
b. Explain the concept of double spending with a suitable example.
c. Compare hot wallets and cold wallets.
d. What is a Merkle tree? Explain the structure of a Merkle tree.
e. Write a program in solidity to find the second largest element in an array.

Q2. Attempt the following:
a. With a suitable diagram, explain the structure of a block header with a list of transactions. (10)
b. State and explain different types of cryptocurrencies. (10)

Q3. Attempt the following:
a. Describe the concept of state machine replication. How is a smart contract represented as a state machine? (10)
b. Explain Hyperledger Fabric v1 architecture. (10)

Q4. Attempt the following:
a. Describe the architecture on Ethereum. (10)
b. Write a program in solidity to implement single inheritance. (10)

Q5. Attempt the following:
a. Explain RAFT consensus mechanism for a private blockchain. (10)
b. Explain fixed and dynamic arrays in solidity with suitable examples. (10)

Q6. Write short notes (any 2): (20)
a. Ripple
b. UTXO model of Bitcoin
c. Corda
d.Blockchain for DeFi
"""


class TestInstructionFrameTopics(unittest.TestCase):
    def test_short_note_topics_are_genuine_questions(self):
        pages = [{
            "page": 1,
            "raw_native_text": "",
            "raw_ocr_text": SHORT_NOTES_PAPER,
            "reconstructed_text": prepare_page_text_for_extraction(SHORT_NOTES_PAPER),
            "ocr_used": True,
        }]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(
                pages, filename="blk.pdf", workspace_id="ws-t", subject="Blockchain", year=2023
            )
        ids = [q["question_id"] for q in result["accepted_questions"]]
        self.assertEqual(len(ids), len(set(ids)))
        for qid in ("Q1(a)", "Q1(e)", "Q2(a)", "Q5(b)", "Q6(a)", "Q6(b)", "Q6(c)", "Q6(d)"):
            self.assertIn(qid, ids, f"{qid} missing from {ids}")
        by_id = {q["question_id"]: q for q in result["accepted_questions"]}
        # Terse topic items keep their own text, never the parent frame
        self.assertEqual(by_id["Q6(a)"]["exact_text"].strip(), "Ripple")
        self.assertEqual(by_id["Q6(c)"]["exact_text"].strip(), "Corda")
        # The parent frame itself is not a question record
        for q in result["accepted_questions"]:
            self.assertFalse(is_instruction_frame_text(q["exact_text"]), q["question_id"])

    def test_glued_marker_lead_parsed(self):
        acc, _rej = extract_questions_from_page_text(
            "Q6. Write short notes on any two: (20)\na. Ripple\nd.Blockchain for DeFi\n",
            1, "g.pdf", "ws",
        )
        ids = sorted(q["question_id"] for q in acc)
        self.assertIn("Q6(a)", ids)
        self.assertIn("Q6(d)", ids)
        by_id = {q["question_id"]: q for q in acc}
        self.assertIn("Blockchain", by_id["Q6(d)"]["exact_text"])


class TestInstructionListNotParents(unittest.TestCase):
    def test_nb_list_items_are_headers(self):
        self.assertTrue(is_header_or_instruction("N.B.: 1. Question No. 1 is compulsory."))
        self.assertTrue(is_header_or_instruction("2. Answer any three out of the remaining questions."))
        self.assertTrue(is_header_or_instruction("3. Assume suitable data if necessary."))
        self.assertTrue(is_header_or_instruction("4. Figures to the right indicate full marks."))

    def test_choice_parents_are_not_headers(self):
        self.assertFalse(is_header_or_instruction("1 Attempt any four"))
        self.assertFalse(is_header_or_instruction("Q1(a) Explain dropout in detail."))
        self.assertFalse(is_header_or_instruction("What are the different types of Gradient Descent methods, explain any three of them."))

    def test_frame_detector_is_strict(self):
        # A real question that merely contains "any three" is NOT a frame
        self.assertFalse(is_instruction_frame_text(
            "What are the different types of Gradient Descent methods, explain any three of them."
        ))
        self.assertTrue(is_instruction_frame_text("Attempt any four"))
        self.assertTrue(is_instruction_frame_text("Write short notes on any two: (20)"))


class TestQuGlyphRecovery(unittest.TestCase):
    def test_qu_marker_becomes_q1(self):
        self.assertEqual(fix_ocr_question_glyphs("Qu. Attempt the following"), "Q1 Attempt the following")
        self.assertEqual(fix_ocr_question_glyphs("Qu."), "Q1")
        # Numbered variant keeps its number
        self.assertEqual(fix_ocr_question_glyphs("Qu 3 Explain"), "Q3 Explain")


class TestStatusVocabulary(unittest.TestCase):
    def test_partial_status_name(self):
        quality = compute_extraction_quality(
            ["Q1(a)", "Q1(b)", "Q2(a)"],
            ["Q1(a)", "Q1(b)", "Q1(c)", "Q2(a)", "Q2(b)"],
        )
        self.assertEqual(quality["extraction_quality"], "PARTIAL")
        self.assertIn("Q1(c)", quality["missing_questions"])

    def test_no_legacy_review_required_emitted(self):
        quality = compute_extraction_quality(["Q1(a)"], ["Q1(a)", "Q9(z)"])
        self.assertIn(quality["extraction_quality"], ("COMPLETE", "RECOVERED", "PARTIAL", "FAILED"))
        self.assertNotEqual(quality["extraction_quality"], "REVIEW_REQUIRED")


class TestMultiResolutionOcrEvidence(unittest.TestCase):
    def test_hd_representation_participates_in_reconciliation(self):
        base = "Q1(a) Explain " + "morphological parsing. " * 3 + "\n"
        hd = "Q1(a) Explain " + "morphological parsing. " * 3 + "\nQ1(b) Define suppletive inflectional morphology clearly.\n"
        pages = [{
            "page": 1,
            "raw_native_text": "",
            "raw_ocr_text": base,
            "raw_ocr_hd_text": hd,
            "reconstructed_text": "",
            "ocr_used": True,
        }]
        with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
            result = hybrid_extract_document(pages, filename="hd.pdf", workspace_id="ws-t", year=2024)
        ids = [q["question_id"] for q in result["accepted_questions"]]
        # Q1(b) exists only in the HD pass — reconciliation must recover it
        self.assertIn("Q1(b)", ids)
        sources = result.get("extraction_audit", {}).get("representation_sources") or {}
        self.assertEqual(sources.get("Q1(b)"), "ocr_text_hd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
