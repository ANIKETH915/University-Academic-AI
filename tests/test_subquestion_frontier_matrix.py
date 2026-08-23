"""
Universal A–I printed-frontier regression matrix (CASE 1–20).

A–I is a supported recognition range, never a required child count.
Missing markers are source-proven and parent-scoped only.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_frontier_matrix")

from rag.hybrid_question_extraction import (
    drop_leaping_parent_ids,
    hybrid_extract_document,
    investigate_subquestion_marker_gaps,
    normalize_marker_id,
)
from rag.subquestion_frontier import (
    is_protected_letter_context,
    maybe_correct_confused_letter,
    parent_scoped_sub_present,
    slot_inference_justified,
    split_text_into_parent_regions,
)


def _run(pages_text, ocr_pages=None):
    pages = []
    for i, txt in enumerate(pages_text):
        ocr = (ocr_pages[i] if ocr_pages else "") or ""
        pages.append(
            {
                "page": i + 1,
                "raw_native_text": txt,
                "raw_ocr_text": ocr,
                "reconstructed_text": txt,
                "ocr_used": bool(ocr),
            }
        )
    with patch("rag.hybrid_question_extraction.llm_configured", return_value=False):
        return hybrid_extract_document(
            pages, filename="frontier.pdf", workspace_id="ws-frontier", year=2024
        )


def _ids(res):
    return [q["question_id"] for q in res["accepted_questions"]]


def _frontier(res, parent):
    for row in (res.get("extraction_audit") or {}).get("parent_frontier_audit") or []:
        if row.get("parent") == parent:
            return row
    return {}


def _lettered_block(parent, letters, stem=None):
    stem = stem or f"Explain {parent} topic"
    lines = [f"{parent}. Attempt the following:"]
    for s in letters:
        lines.append(f"{parent}({s}) {stem} {s} with adequate supporting technical detail.")
    return "\n".join(lines) + "\n"


class TestCases1to11PrintedFrontier(unittest.TestCase):
    def test_case_1_ab_no_missing_ci(self):
        res = _run([_lettered_block("Q1", "ab")])
        self.assertEqual(set(_ids(res)), {"Q1(a)", "Q1(b)"})
        self.assertEqual(res["quality"]["missing_questions"], [])
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "b")
        self.assertEqual(_frontier(res, "Q1").get("source_proven_missing"), [])
        for bad in "cdefghi":
            self.assertNotIn(f"Q1({bad})", _ids(res))
            self.assertNotIn(f"Q1({bad})", res["quality"]["missing_questions"])

    def test_case_2_abc(self):
        res = _run([_lettered_block("Q1", "abc")])
        self.assertEqual(set(_ids(res)), {"Q1(a)", "Q1(b)", "Q1(c)"})
        self.assertEqual(res["quality"]["missing_questions"], [])
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "c")

    def test_case_3_a_to_e(self):
        res = _run([_lettered_block("Q1", "abcde")])
        self.assertEqual({f"Q1({s})" for s in "abcde"}, set(_ids(res)))
        self.assertEqual(res["quality"]["missing_questions"], [])
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "e")
        for bad in "fghi":
            self.assertNotIn(f"Q1({bad})", res["quality"]["missing_questions"])

    def test_case_4_a_to_i(self):
        res = _run([_lettered_block("Q1", "abcdefghi")])
        self.assertEqual({f"Q1({s})" for s in "abcdefghi"}, set(_ids(res)))
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "i")

    def test_case_5_ace_no_auto_bd(self):
        res = _run([_lettered_block("Q1", "ace")])
        self.assertEqual(set(_ids(res)), {"Q1(a)", "Q1(c)", "Q1(e)"})
        self.assertEqual(res["quality"]["missing_questions"], [])
        self.assertNotIn("Q1(b)", _ids(res))
        self.assertNotIn("Q1(d)", _ids(res))

    def test_case_6_recover_internal_c_from_body(self):
        text = (
            "Q1(a) Explain normalization in database systems with an example.\n"
            "Q1(b) Explain indexing strategies used in relational databases.\n"
            "What is denormalization and why is it used in practice?\n"
            "Q1(d) Explain hashing techniques for file organization.\n"
            "Q1(e) Explain caching strategies in storage engines.\n"
        )
        res = _run([text])
        ids = set(_ids(res))
        self.assertIn("Q1(a)", ids)
        self.assertIn("Q1(b)", ids)
        self.assertIn("Q1(d)", ids)
        self.assertIn("Q1(e)", ids)
        self.assertIn("Q1(c)", ids)
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "e")

    def test_case_7_cross_page_abcd(self):
        p1 = (
            "Q1. Attempt the following:\n"
            "Q1(a) Explain first mechanism with adequate supporting detail.\n"
            "Q1(b) Explain second mechanism with adequate supporting detail.\n"
        )
        p2 = (
            "Q1(c) Explain third mechanism with adequate supporting detail.\n"
            "Q1(d) Explain fourth mechanism with adequate supporting detail.\n"
        )
        res = _run([p1, p2])
        self.assertEqual(set(_ids(res)), {"Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)"})
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "d")
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_case_8_independent_parent_frontiers(self):
        text = _lettered_block("Q1", "abcde") + _lettered_block("Q2", "ab")
        res = _run([text])
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "e")
        self.assertEqual(_frontier(res, "Q2").get("printed_frontier"), "b")
        self.assertNotIn("Q2(c)", _ids(res))
        self.assertNotIn("Q2(c)", res["quality"]["missing_questions"])

    def test_case_9_choice_four_printed_five(self):
        text = (
            "Q1. Attempt any FOUR:\n"
            + "\n".join(
                f"({s}) Explain topic {s} with adequate supporting technical detail."
                for s in "abcde"
            )
        )
        res = _run([text])
        self.assertEqual(len([i for i in _ids(res) if i.startswith("Q1(")]), 5)
        self.assertIn("Q1(e)", _ids(res))

    def test_case_10_choice_four_printed_two(self):
        text = (
            "Q1. Attempt any FOUR:\n"
            "(a) Explain first topic with adequate supporting technical detail.\n"
            "(b) Explain second topic with adequate supporting technical detail.\n"
        )
        res = _run([text])
        kids = [i for i in _ids(res) if i.startswith("Q1(")]
        self.assertEqual(set(kids), {"Q1(a)", "Q1(b)"})
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_case_11_next_parent_stops_search(self):
        text = _lettered_block("Q6", "ab") + _lettered_block("Q7", "ab")
        res = _run([text])
        self.assertEqual(_frontier(res, "Q6").get("printed_frontier"), "b")
        for bad in "cdefghi":
            self.assertNotIn(f"Q6({bad})", _ids(res))
            self.assertNotIn(f"Q6({bad})", res["quality"]["missing_questions"])


class TestCases12to20EvidenceAndNoise(unittest.TestCase):
    def test_case_12_visual_recovers_bd(self):
        visual = _lettered_block("Q1", "abcde")
        ocr = _lettered_block("Q1", "ace")
        res = _run([visual], ocr_pages=[ocr])
        ids = set(_ids(res))
        for s in "abcde":
            self.assertIn(f"Q1({s})", ids)
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_case_13_reject_footer_q6t(self):
        native = _lettered_block("Q6", "ab")
        ocr = native + "Q6(t) Page 2 watermark University Examination\n"
        res = _run([native], ocr_pages=[ocr])
        self.assertNotIn("Q6(t)", _ids(res))
        self.assertNotIn("Q6(t)", res["quality"]["missing_questions"])
        self.assertEqual(_frontier(res, "Q6").get("printed_frontier"), "b")

    def test_case_14_no_fabricate_bd(self):
        res = _run([_lettered_block("Q1", "ace")])
        self.assertNotIn("Q1(b)", _ids(res))
        self.assertNotIn("Q1(d)", _ids(res))
        self.assertEqual(res["quality"]["missing_questions"], [])

    def test_case_15_roman_not_alpha(self):
        text = (
            "Q1. Attempt the following:\n"
            "(i) Explain paging mechanism with adequate supporting detail.\n"
            "(ii) Explain segmentation with adequate supporting detail.\n"
            "(iii) Explain virtual memory with adequate supporting detail.\n"
        )
        res = _run([text])
        ids = set(_ids(res))
        self.assertIn("Q1(i)", ids)
        self.assertIn("Q1(ii)", ids)
        self.assertIn("Q1(iii)", ids)
        self.assertNotIn("Q1(a)", ids)
        self.assertNotIn("Q1(b)", ids)
        self.assertEqual(_frontier(res, "Q1").get("marker_family"), "roman")

    def test_case_16_mixed_marker_styles(self):
        text = (
            "Q1(a) Explain first topic with adequate supporting technical detail.\n"
            "b. Explain second topic with adequate supporting technical detail.\n"
            "(c) Explain third topic with adequate supporting technical detail.\n"
            "d) Explain fourth topic with adequate supporting technical detail.\n"
        )
        res = _run([text])
        self.assertEqual(set(_ids(res)), {"Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)"})
        self.assertEqual(len(_ids(res)), 4)

    def test_case_17_table_cells_not_subquestions(self):
        text = (
            "Q1(a) Explain the following dataset with adequate supporting detail.\n"
            "| a | b | c | d |\n"
            "| 1 | 2 | 3 | 4 |\n"
            "a b c d\n"
            "10 20 30 40\n"
            "Q1(b) Discuss indexing strategies derived from the table.\n"
        )
        res = _run([text])
        ids = set(_ids(res))
        self.assertIn("Q1(a)", ids)
        self.assertIn("Q1(b)", ids)
        self.assertNotIn("Q1(c)", ids)
        self.assertNotIn("Q1(d)", ids)

    def test_case_18_diagram_labels_not_subquestions(self):
        text = (
            "Q1(a) Explain the architecture shown in the figure with supporting detail.\n"
            "Figure: (a) encoder (b) decoder (c) attention\n"
            "Q1(b) Derive the loss function used by the architecture.\n"
        )
        res = _run([text])
        ids = set(_ids(res))
        self.assertIn("Q1(a)", ids)
        self.assertIn("Q1(b)", ids)
        self.assertNotIn("Q1(c)", ids)

    def test_case_19_duplicate_q1a_one_canonical(self):
        text = (
            "Q1(a) Explain concurrency control with adequate supporting detail.\n"
            "Q1(a) Explain concurrency control with adequate supporting detail.\n"
            "Q1(b) Explain deadlock handling with adequate supporting detail.\n"
        )
        res = _run([text])
        self.assertEqual(_ids(res).count("Q1(a)"), 1)
        self.assertEqual(set(_ids(res)), {"Q1(a)", "Q1(b)"})

    def test_case_20_short_notes_choice_not_child_count(self):
        text = (
            "Q6. Write short notes on any two:\n"
            "(a) Explain public blockchain properties with supporting detail.\n"
            "(b) Explain private blockchain properties with supporting detail.\n"
            "(c) Explain hash pointers with supporting detail.\n"
            "(d) Explain merkle trees with supporting detail.\n"
        )
        res = _run([text])
        kids = [i for i in _ids(res) if i.startswith("Q6(")]
        self.assertEqual(set(kids), {"Q6(a)", "Q6(b)", "Q6(c)", "Q6(d)"})
        self.assertEqual(_frontier(res, "Q6").get("printed_frontier"), "d")


class TestParentScopedEvidenceAndHelpers(unittest.TestCase):
    def test_other_parent_b_does_not_prove_q1_b_missing(self):
        pages = [
            {
                "page": 1,
                "raw_native_text": (
                    "Q1(a) Explain first with adequate supporting detail.\n"
                    "Q1(c) Explain third with adequate supporting detail.\n"
                    "Q2(b) Explain other parent sibling with adequate detail.\n"
                ),
                "raw_ocr_text": "",
                "reconstructed_text": (
                    "Q1(a) Explain first with adequate supporting detail.\n"
                    "Q1(c) Explain third with adequate supporting detail.\n"
                    "Q2(b) Explain other parent sibling with adequate detail.\n"
                ),
            }
        ]
        extracted = [{"question_id": "Q1(a)"}, {"question_id": "Q1(c)"}, {"question_id": "Q2(b)"}]
        classified = [
            {"marker_id": qid, "genuine": True}
            for qid in ("Q1(a)", "Q1(c)", "Q2(b)")
        ]
        proven, amb = investigate_subquestion_marker_gaps(
            pages, extracted, classified, [], [], None
        )
        self.assertNotIn("Q1(b)", proven)

    def test_question_no_and_unclosed_normalize(self):
        self.assertEqual(normalize_marker_id("Question No. 1(a)"), "Q1(a)")
        self.assertEqual(normalize_marker_id("Q1(a"), "Q1(a)")
        self.assertEqual(normalize_marker_id("Q1 [b]"), "Q1(b)")

    def test_protected_table_and_figure_lines(self):
        self.assertTrue(is_protected_letter_context("| a | b | c | d |"))
        self.assertTrue(is_protected_letter_context("a b c d"))
        self.assertTrue(
            is_protected_letter_context("Figure: (a) encoder (b) decoder (c) attention")
        )
        self.assertFalse(
            is_protected_letter_context("(a) Explain normalization with an example.")
        )

    def test_parent_regions_do_not_reset_on_page_join(self):
        blob = (
            "Q6(a) Explain first with adequate supporting detail.\n"
            "Q6(b) Explain second with adequate supporting detail.\n"
            "(c) Explain third with adequate supporting detail.\n"
            "Q7(a) Explain next parent with adequate supporting detail.\n"
        )
        regions = split_text_into_parent_regions(blob)
        self.assertIn("(c)", regions["Q6"])
        self.assertNotIn("Q7", regions["Q6"])
        self.assertTrue(parent_scoped_sub_present(regions["Q6"], "c"))
        self.assertFalse(parent_scoped_sub_present(regions["Q7"], "c"))

    def test_slot_inference_not_for_complete_bodies(self):
        self.assertFalse(
            slot_inference_justified(
                "Explain servlet life cycle with a neat labelled diagram."
            )
        )
        self.assertTrue(slot_inference_justified("<S>| w1 | w2 | w3 <E>"))

    def test_confused_letter_requires_other_repr(self):
        self.assertEqual(
            maybe_correct_confused_letter(
                "e", sibling_letters=["a", "b", "d"], other_repr_letters=["c"]
            ),
            "c",
        )
        self.assertIsNone(
            maybe_correct_confused_letter(
                "e", sibling_letters=["a", "b"], other_repr_letters=[]
            )
        )

    def test_cross_parent_q1_q2_no_frontier_leak(self):
        text = _lettered_block("Q1", "abcde") + _lettered_block("Q2", "ab")
        res = _run([text])
        self.assertEqual(_frontier(res, "Q1").get("printed_frontier"), "e")
        self.assertEqual(_frontier(res, "Q2").get("printed_frontier"), "b")

    def test_isolated_leap_letter_is_noise_not_frontier(self):
        text = (
            "Q3(a) Explain first topic with adequate supporting technical detail.\n"
            "Q3(b) Explain second topic with adequate supporting technical detail.\n"
            "Q3(h) Page footer watermark residue\n"
        )
        res = _run([text])
        self.assertEqual(set(_ids(res)), {"Q3(a)", "Q3(b)"})
        self.assertNotIn("Q3(h)", _ids(res))
        self.assertEqual(_frontier(res, "Q3").get("printed_frontier"), "b")

    def test_unclosed_paren_recovers_as_c(self):
        text = (
            "Q1(a) Explain first topic with adequate supporting technical detail.\n"
            "Q1(b) Explain second topic with adequate supporting technical detail.\n"
            "(c Explain third topic with adequate supporting technical detail.\n"
            "Q1(d) Explain fourth topic with adequate supporting technical detail.\n"
        )
        res = _run([text])
        self.assertIn("Q1(c)", _ids(res))


# Printed child sequences used by the universal parent-frontier cases.
# Letters are source-printed siblings; frontier is the last printed letter.
_UNIVERSAL_PARENT_CASES = (
    ("Q1", "ab", "b"),
    ("Q2", "abcd", "d"),
    ("Q3", "ace", "e"),
    ("Q4", "abcdef", "f"),
    ("Q5", "ab", "b"),
    ("Q6", "abcdefghi", "i"),
    ("Q7", "abc", "c"),
    ("Q10", "ab", "b"),
    ("Q20", "abcde", "e"),
)


def _assert_parent_frontier(test, res, parent, letters, frontier):
    ids = set(_ids(res))
    row = _frontier(res, parent)
    for s in letters:
        test.assertIn(f"{parent}({s})", ids, f"{parent}({s}) must be recovered")
    test.assertEqual(row.get("printed_frontier"), frontier, f"{parent} frontier")
    test.assertEqual(row.get("source_proven_missing"), [], f"{parent} extra missing")
    # Past-frontier letters must not be invented or reported missing.
    after = [chr(c) for c in range(ord(frontier) + 1, ord("i") + 1)]
    for s in after:
        test.assertNotIn(f"{parent}({s})", ids)
        test.assertNotIn(f"{parent}({s})", res["quality"]["missing_questions"])
    # Independent audit aliases must exist for every parent.
    test.assertEqual(row.get("observed_children"), row.get("observed"))
    test.assertEqual(row.get("recovered_children"), row.get("recovered"))
    test.assertIn("ambiguous_candidates", row)
    test.assertIn("rejected_noise", row)
    # Non-contiguous printed letters must not be filled in.
    printed = set(letters)
    for s in "abcdefghi":
        if s not in printed and s <= frontier:
            test.assertNotIn(f"{parent}({s})", ids)
            test.assertNotIn(f"{parent}({s})", res["quality"]["missing_questions"])


class TestUniversalParentFrontier(unittest.TestCase):
    """Same frontier logic for Q1…Qn. Q6 is only one of the parents."""

    def test_combined_paper_independent_frontiers(self):
        text = "".join(
            _lettered_block(parent, letters)
            for parent, letters, _front in _UNIVERSAL_PARENT_CASES
        )
        res = _run([text])
        for parent, letters, frontier in _UNIVERSAL_PARENT_CASES:
            _assert_parent_frontier(self, res, parent, letters, frontier)
        # Q3 printed a,c,e — do not fabricate b/d.
        self.assertNotIn("Q3(b)", _ids(res))
        self.assertNotIn("Q3(d)", _ids(res))

    def test_gap_recovery_does_not_steal_another_parents_child(self):
        """Shared wording across parents must not fill Q3(b) from Q2(b)."""
        text = (
            "Q2. Attempt the following:\n"
            "Q2(a) Explain concept a with adequate supporting technical detail.\n"
            "Q2(b) Explain concept b with adequate supporting technical detail.\n"
            "Q2(c) Explain concept c with adequate supporting technical detail.\n"
            "Q2(d) Explain concept d with adequate supporting technical detail.\n"
            "Q3. Attempt the following:\n"
            "Q3(a) Explain concept a with adequate supporting technical detail.\n"
            "Q3(c) Explain concept c with adequate supporting technical detail.\n"
            "Q3(e) Explain concept e with adequate supporting technical detail.\n"
        )
        res = _run([text])
        self.assertEqual(set(i for i in _ids(res) if i.startswith("Q3(")), {"Q3(a)", "Q3(c)", "Q3(e)"})
        self.assertNotIn("Q3(b)", _ids(res))
        self.assertNotIn("Q3(d)", _ids(res))
        self.assertEqual(_frontier(res, "Q3").get("printed_frontier"), "e")
        self.assertEqual(_frontier(res, "Q2").get("printed_frontier"), "d")

    def test_each_listed_parent_in_isolation(self):
        for parent, letters, frontier in _UNIVERSAL_PARENT_CASES:
            with self.subTest(parent=parent, letters=letters):
                res = _run([_lettered_block(parent, letters)])
                _assert_parent_frontier(self, res, parent, letters, frontier)

    def test_attempt_any_four_does_not_invent_children(self):
        for parent in ("Q1", "Q4", "Q6", "Q7", "Q10", "Q20"):
            with self.subTest(parent=parent):
                text = (
                    f"{parent}. Attempt any FOUR:\n"
                    f"(a) Explain first topic with adequate supporting technical detail.\n"
                    f"(b) Explain second topic with adequate supporting technical detail.\n"
                )
                res = _run([text])
                kids = [i for i in _ids(res) if i.startswith(f"{parent}(")]
                self.assertEqual(set(kids), {f"{parent}(a)", f"{parent}(b)"})
                self.assertEqual(_frontier(res, parent).get("printed_frontier"), "b")
                for bad in "cdefghi":
                    self.assertNotIn(f"{parent}({bad})", kids)
                    self.assertNotIn(
                        f"{parent}({bad})", res["quality"]["missing_questions"]
                    )

    def test_q10_and_q20_mixed_marker_styles(self):
        text = (
            "Question 10(a) Explain tokenization with adequate supporting technical detail.\n"
            "b. Explain n-gram models with adequate supporting technical detail.\n"
            "(c) Explain TF-IDF weighting with adequate supporting technical detail.\n"
            "Q.20(a) Explain distributed ledgers with adequate supporting technical detail.\n"
            "b. Explain proof of work with adequate supporting technical detail.\n"
            "(c) Explain proof of stake with adequate supporting technical detail.\n"
            "d) Explain smart contracts with adequate supporting technical detail.\n"
            "Q.20(e) Explain consensus finality with adequate supporting technical detail.\n"
        )
        res = _run([text])
        self.assertEqual(set(i for i in _ids(res) if i.startswith("Q10(")), {"Q10(a)", "Q10(b)", "Q10(c)"})
        self.assertEqual(_frontier(res, "Q10").get("printed_frontier"), "c")
        self.assertEqual(
            set(i for i in _ids(res) if i.startswith("Q20(")),
            {"Q20(a)", "Q20(b)", "Q20(c)", "Q20(d)", "Q20(e)"},
        )
        self.assertEqual(_frontier(res, "Q20").get("printed_frontier"), "e")

    def test_question_no_and_ocr_variants_on_arbitrary_parents(self):
        text = (
            "Question No. 4(a) Explain paging with adequate supporting technical detail.\n"
            "Q4(b) Explain segmentation with adequate supporting technical detail.\n"
            "5(a) Explain deadlock handling with adequate supporting technical detail.\n"
            "b) Explain disk scheduling with adequate supporting technical detail.\n"
            "Q.7(a) Explain hashing with adequate supporting technical detail.\n"
        )
        res = _run([text])
        self.assertIn("Q4(a)", _ids(res))
        self.assertIn("Q4(b)", _ids(res))
        self.assertEqual(_frontier(res, "Q4").get("printed_frontier"), "b")
        self.assertIn("Q5(a)", _ids(res))
        self.assertIn("Q5(b)", _ids(res))
        self.assertEqual(_frontier(res, "Q5").get("printed_frontier"), "b")
        self.assertIn("Q7(a)", _ids(res))
        self.assertEqual(_frontier(res, "Q7").get("printed_frontier"), "a")

    def test_cross_page_continuation_on_non_q6_parent(self):
        p1 = (
            "Q10. Attempt the following:\n"
            "Q10(a) Explain first mechanism with adequate supporting detail.\n"
            "Q10(b) Explain second mechanism with adequate supporting detail.\n"
        )
        p2 = (
            "Q10(c) Explain third mechanism with adequate supporting detail.\n"
            "Q20(a) Explain next parent first with adequate supporting detail.\n"
            "Q20(b) Explain next parent second with adequate supporting detail.\n"
        )
        res = _run([p1, p2])
        self.assertEqual(set(i for i in _ids(res) if i.startswith("Q10(")), {"Q10(a)", "Q10(b)", "Q10(c)"})
        self.assertEqual(_frontier(res, "Q10").get("printed_frontier"), "c")
        self.assertEqual(set(i for i in _ids(res) if i.startswith("Q20(")), {"Q20(a)", "Q20(b)"})
        self.assertEqual(_frontier(res, "Q20").get("printed_frontier"), "b")
        self.assertNotIn("Q10(d)", _ids(res))
        self.assertNotIn("Q20(c)", _ids(res))

    def test_no_q6_hardcoded_parent_branch_in_extractor(self):
        rag_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag"))
        forbidden = (
            'parent == "Q6"',
            "parent == 'Q6'",
            "question_number == 6",
            '== "Q6(a-i)"',
            "if parent == Q6",
            "if question_number == 6",
        )
        hits = []
        for dirpath, _dirs, files in os.walk(rag_root):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                with open(path, encoding="utf-8") as fh:
                    body = fh.read()
                for needle in forbidden:
                    if needle in body:
                        hits.append(f"{path}: {needle}")
        self.assertEqual(hits, [], "Q6 must not be an architectural special case")


class TestDropLeapingParentIds(unittest.TestCase):
    def test_keeps_q10_and_q20_after_q1_q2(self):
        ids = ["Q1(a)", "Q2(b)", "Q10(a)", "Q20(e)"]
        self.assertEqual(drop_leaping_parent_ids(ids), ids)

    def test_drops_q53_glue_keeps_q10(self):
        ids = ["Q1(a)", "Q2(a)", "Q3(a)", "Q53(ii)", "Q10(a)"]
        out = drop_leaping_parent_ids(ids)
        self.assertNotIn("Q53(ii)", out)
        self.assertIn("Q10(a)", out)
        self.assertIn("Q1(a)", out)

    def test_keeps_sequential_q11_after_q10(self):
        ids = ["Q9(a)", "Q10(a)", "Q11(b)"]
        self.assertEqual(drop_leaping_parent_ids(ids), ids)


if __name__ == "__main__":
    unittest.main()
