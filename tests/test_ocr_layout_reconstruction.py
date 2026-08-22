"""
Layout-aware OCR reconstruction.

These tests feed synthetic *geometry* (the same shape pytesseract returns for
real papers) so the reconstruction rules are exercised without needing a
Tesseract install. No subject, year or question-count assumptions.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.ocr_layout import reconstruct_questions_from_layout
from rag.question_extractor import (
    extract_questions_from_page_text,
    prepare_page_text_for_extraction,
)


def L(text: str, x0: int, top: int) -> dict:
    return {"text": text, "x0": x0, "x1": x0 + 400, "top": top, "bottom": top + 18, "height": 18}


def ids_from(layout_text: str):
    prepared = prepare_page_text_for_extraction(layout_text)
    acc, _rej = extract_questions_from_page_text(prepared, 1, "f.pdf", "ws", year=0)
    return [q["question_id"] for q in acc]


class TestGutterParentRecovery(unittest.TestCase):
    """Parent numbers lost from the left gutter are recovered from position."""

    def _page(self):
        # Marker column at x=240, body column at x=290 (parent numbers dropped).
        return [
            L("University of Somewhere", 400, 20),
            L("1 Attempt any four [20]", 160, 100),
            L("a) Define a process control block.", 240, 130),
            L("b) Explain context switching overhead.", 240, 160),
            L("c) Describe a semaphore.", 240, 190),
            L("d) What is thrashing?", 240, 220),
            L("e) Explain paging hardware.", 240, 250),
            L("Explain deadlock detection algorithms in detail. [10]", 290, 310),
            L("b) Compare preemptive and non-preemptive scheduling. [10]", 240, 340),
            L("Explain virtual memory management with examples. [10]", 290, 400),
            L("b) Describe segmentation and its advantages. [10]", 240, 430),
        ]

    def test_parents_and_subs_recovered(self):
        text = reconstruct_questions_from_layout(self._page())
        self.assertTrue(text, "expected reconstruction from positional evidence")
        self.assertEqual(
            ids_from(text),
            ["Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q1(e)", "Q2(a)", "Q2(b)", "Q3(a)", "Q3(b)"],
        )

    def test_instruction_not_a_question(self):
        text = reconstruct_questions_from_layout(self._page())
        self.assertNotIn("Attempt any four", text)
        self.assertNotIn("University of Somewhere", text)

    def test_marks_tags_stripped(self):
        text = reconstruct_questions_from_layout(self._page())
        acc, _ = extract_questions_from_page_text(
            prepare_page_text_for_extraction(text), 1, "f.pdf", "ws", year=0
        )
        by_id = {q["question_id"]: q for q in acc}
        self.assertEqual(by_id["Q2(a)"]["marks"], 10)
        self.assertNotIn("[10]", by_id["Q2(a)"]["exact_text"])
        self.assertNotIn("[20]", by_id["Q1(a)"]["exact_text"])


class TestExplicitParentsPreserved(unittest.TestCase):
    def test_explicit_parent_numbers_win(self):
        lines = [
            L("Q1. a. Explain the OSI reference model.", 150, 100),
            L("b. Describe TCP congestion control.", 210, 130),
            L("Q2. a. Explain subnetting with an example.", 150, 190),
            L("b. Compare IPv4 and IPv6 addressing.", 210, 220),
            L("Q3. a. Explain the sliding window protocol.", 150, 280),
            L("b. Describe DNS resolution steps.", 210, 310),
        ]
        text = reconstruct_questions_from_layout(lines)
        self.assertEqual(
            ids_from(text),
            ["Q1(a)", "Q1(b)", "Q2(a)", "Q2(b)", "Q3(a)", "Q3(b)"],
        )


class TestWrapContinuation(unittest.TestCase):
    def test_wrapped_lines_stay_in_one_question(self):
        lines = [
            L("Q1. a. Explain convolutional neural networks including", 150, 100),
            L("convolution, pooling and fully connected layers in detail.", 250, 125),
            L("b. Define overfitting and describe two remedies.", 210, 160),
            L("Q2. a. Explain gradient descent variants.", 150, 220),
            L("b. Describe batch normalization.", 210, 250),
        ]
        text = reconstruct_questions_from_layout(lines)
        self.assertEqual(ids_from(text), ["Q1(a)", "Q1(b)", "Q2(a)", "Q2(b)"])
        self.assertIn("fully connected layers", text)

    def test_bulleted_continuation_not_new_question(self):
        lines = [
            L("Q1. a. Consider a CNN layer with this configuration: [10]", 150, 100),
            L("-The input has 32 channels and a spatial size of 64x64.", 280, 125),
            L("-The layer has 64 filters of size 3x3 with stride 1.", 280, 150),
            L("b. Explain pooling operations. [10]", 210, 190),
            L("Q2. a. Explain autoencoders. [10]", 150, 250),
            L("b. Explain GAN training. [10]", 210, 280),
        ]
        text = reconstruct_questions_from_layout(lines)
        self.assertEqual(ids_from(text), ["Q1(a)", "Q1(b)", "Q2(a)", "Q2(b)"])
        self.assertIn("64 filters", text)


class TestRefusesWithoutEvidence(unittest.TestCase):
    """Never fabricate structure when geometry does not support it."""

    def test_prose_page_returns_nothing(self):
        lines = [
            L("This page is continuous prose about operating systems.", 150, 100),
            L("It contains no exam markers whatsoever and should not", 150, 130),
            L("produce any reconstructed question identifiers at all.", 150, 160),
            L("Another sentence of ordinary explanatory text follows.", 150, 190),
            L("And one more line to pass the minimum length check here.", 150, 220),
        ]
        self.assertEqual(reconstruct_questions_from_layout(lines), "")

    def test_single_orphan_marker_not_promoted(self):
        lines = [
            L("a) Explain something entirely on its own.", 240, 100),
            L("Some unrelated closing remark line.", 150, 140),
            L("Another unrelated closing remark line.", 150, 170),
            L("Yet another unrelated closing remark line.", 150, 200),
            L("Final unrelated closing remark line here.", 150, 230),
        ]
        self.assertEqual(reconstruct_questions_from_layout(lines), "")

    def test_too_few_lines(self):
        self.assertEqual(reconstruct_questions_from_layout([L("Q1(a) x", 100, 10)]), "")


class TestVariableCountsFromLayout(unittest.TestCase):
    def test_no_upper_bound_on_questions(self):
        lines = [L("Instructions to candidates", 300, 20)]
        y = 100
        for parent in range(1, 13):  # 12 parents x 2 subs = 24 questions
            lines.append(L(f"Q{parent}. a. Explain topic {parent} in detail. [10]", 150, y))
            y += 30
            lines.append(L(f"b. Describe application {parent} briefly. [10]", 210, y))
            y += 30
        text = reconstruct_questions_from_layout(lines)
        self.assertEqual(len(ids_from(text)), 24)


class TestRomanSubquestions(unittest.TestCase):
    def test_roman_markers(self):
        lines = [
            L("Q1. (i) Explain two-phase locking protocol.", 150, 100),
            L("(ii) Define serializability of schedules.", 210, 130),
            L("Q2. (i) Explain timestamp ordering.", 150, 190),
            L("(ii) Describe cascading rollback.", 210, 220),
            L("Q3. (i) Explain deadlock detection in databases.", 150, 280),
            L("(ii) Describe the wait-for graph technique.", 210, 310),
        ]
        text = reconstruct_questions_from_layout(lines)
        self.assertTrue(text)
        self.assertGreaterEqual(len(ids_from(text)), 3)


class TestSubLetterDepthFromLayout(unittest.TestCase):
    """Q1(a-f) and later parents with a third sub must survive geometry reconstruction."""

    def test_six_subs_then_third_subs_on_later_parents(self):
        lines = [
            L("END SEMESTER EXAMINATION", 400, 20),
            L("1 Attempt any four [20]", 160, 80),
        ]
        top = 110
        for s, body in zip(
            "abcdef",
            [
                "Explain the layered reference model.",
                "Describe the scheduling policy.",
                "Differentiate the two normalisation forms.",
                "Discuss the error control mechanism.",
                "Derive the expression for throughput.",
                "Compare the two indexing structures.",
            ],
        ):
            lines.append(L(f"{s}) {body}", 240, top))
            top += 28
        # Later parents: OCR dropped the gutter number and the leading 'a)'
        # marker; recovery is from an unlabelled body + following 'b'/'c'.
        lines += [
            L("Explain deadlock detection algorithms in detail. [10]", 290, top),
            L("b) Compare preemptive and non-preemptive scheduling. [10]", 240, top + 30),
            L("c) Describe segmentation and its advantages. [10]", 240, top + 60),
            L("Explain virtual memory management with examples. [10]", 290, top + 120),
            L("b) Outline page replacement policies. [10]", 240, top + 150),
            L("c) Evaluate working-set window size. [10]", 240, top + 180),
            L("Explain RAID levels with a worked example. [10]", 290, top + 240),
            L("b) Describe journaled file systems. [10]", 240, top + 270),
            L("c) Compare linked and indexed allocation. [10]", 240, top + 300),
        ]
        text = reconstruct_questions_from_layout(lines)
        ids = ids_from(text)
        for qid in (
            "Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q1(e)", "Q1(f)",
            "Q3(c)", "Q4(c)",
        ):
            self.assertIn(qid, ids, f"{qid} was lost from layout reconstruction")


class TestDottedQPrefixLayout(unittest.TestCase):
    """
    Numbering class: Q.1. / Q.2.a) / Q.2b) / Q.3 b) / Q.4.c') / Q.5 a).

    Parent numbers and sub-letters are on the same visual line, with dots and
    OCR junk around the delimiter. This is a numbering style, not a subject.
    """

    def test_dotted_q_prefix_recovers_all_parents_and_third_subs(self):
        lines = [
            L("Time: 3 Hours Max. Marks: 80", 150, 20),
            L("Q.1. Any Four 20[M]", 165, 80),
            L("a Differentiate between two ambiguity types. 5M]", 191, 110),
            L("b Define affixes. Explain the types of affixes. 5M]", 191, 140),
            L("c Describe open class words with examples. 5M]", 191, 170),
            L("d What is rule based translation? 5M]", 191, 200),
            L("e Explain relationships between word meanings. 5M]", 191, 230),
            L("Homonymy, Polysemy, Synonymy, Antonymy", 256, 255),
            L("f Explain perplexity of any language model. 5M]", 191, 285),
            L("Q.2.a) Explain the role of finite state analysis?", 165, 330),
            L("Q.2b) Explain different stages of the process with an example. [10M]", 164, 360),
            L("Q.3.a) Consider the following corpus 5M]", 165, 410),
            L("<s> I tell you to sleep and rest </s>", 332, 435),
            L("List all possible bigrams. Compute conditional probabilities.", 256, 460),
            L("Q.3 b) Explain a bootstrapping approach of semi supervised learning [5M]", 164, 500),
            L("Q.3.c) What is tagging? Discuss various challenges faced. [10M]", 166, 530),
            L("Q.4a) What are the limitations of the hidden model? [5M]", 165, 580),
            L("Q.4b) Explain the different steps in text processing. [5M]", 164, 610),
            L("Q.4.c\ufffd) Compare top-down and bottom-up parsing with example. [10M]", 166, 640),
            L("Q.5 a) | What do you mean by sense disambiguation? [10M]", 165, 690),
            L("Q5 b) Explain the pronoun resolution algorithm. [10M]", 164, 720),
            L("Q.6a) Explain text summarization in detail. [10M]", 165, 770),
            L("Q.6b) Explain the stemming algorithm in detail. [10M]", 164, 800),
        ]
        text = reconstruct_questions_from_layout(lines)
        self.assertTrue(text, "dotted Q.n.x) layout produced no reconstruction")
        ids = ids_from(text)
        for qid in (
            "Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q1(e)", "Q1(f)",
            "Q2(a)", "Q2(b)",
            "Q3(a)", "Q3(b)", "Q3(c)",
            "Q4(a)", "Q4(b)", "Q4(c)",
            "Q5(a)", "Q5(b)",
            "Q6(a)", "Q6(b)",
        ):
            self.assertIn(qid, ids, f"{qid} missing from {ids}")
        q1f = next(line for line in text.splitlines() if line.startswith("Q1(f)"))
        self.assertIn("perplexity", q1f.lower())
        self.assertNotIn("finite state", q1f.lower())


class TestUnlabelledBodiesLayout(unittest.TestCase):
    """
    OCR dropped every Q/a) glyph. Recovery uses instruction + marks tags +
    academic verbs, and pairs remaining-N parents when the count matches.
    """

    def test_attempt_any_plus_remaining_five_recovers_structure(self):
        lines = [
            L("Duration: 3hrs [Max Marks: 80]", 150, 20),
            L("N.B.: (1) Question No 1 is Compulsory.", 151, 50),
            L("(2) Attempt any three questions out of the remaining five.", 226, 80),
            L("Attempt any FOUR [20]", 243, 120),
            L("What is a process control block?", 236, 150),
            L("acre", 199, 165),
            L("Explain context switching overhead.", 236, 180),
            L("Describe a semaphore with an example.", 236, 210),
            L("What is thrashing in virtual memory?", 236, 240),
            L("Explain paging hardware briefly.", 236, 270),
            L("Explain deadlock detection algorithms in detail. [10]", 236, 320),
            L("Discuss challenges in CPU scheduling. [10]", 236, 350),
            L("Consider the following page reference string [10]", 236, 400),
            L("1 2 3 2 1 4", 313, 420),
            L("Compute the number of faults using LRU.", 236, 440),
            L("What are five types of page replacement? 10]", 236, 480),
            L("Explain the banker's algorithm with an example. 10]", 236, 510),
            L("Explain two-phase locking in detail. 10]", 236, 540),
            L("Describe wait-for graphs. 10]", 236, 570),
            L("Explain RAID levels with an example. 10]", 236, 600),
            L("Explain journaled file systems. 10]", 236, 630),
            L("Explain indexed allocation in detail. 10]", 236, 660),
        ]
        text = reconstruct_questions_from_layout(lines)
        self.assertTrue(text)
        recon_ids = [ln.split()[0] for ln in text.splitlines() if ln.startswith("Q")]
        self.assertEqual(len(recon_ids), 15, text)
        ids = ids_from(text)
        self.assertIn("Q1(a)", ids)
        self.assertIn("Q1(e)", ids)
        self.assertNotIn("acre", text.lower())
        self.assertIn("Q2(a)", ids)
        self.assertIn("Q6(b)", ids)
        q3a = next(line for line in text.splitlines() if line.startswith("Q3(a)"))
        self.assertIn("LRU", q3a)
        self.assertGreaterEqual(len(ids), 12)

    def test_prose_without_instruction_or_marks_not_promoted(self):
        lines = [
            L("This chapter introduces operating systems.", 150, 40),
            L("Processes are the unit of execution.", 150, 70),
            L("Memory management includes paging.", 150, 100),
            L("File systems store data on disk.", 150, 130),
            L("Networks connect independent machines.", 150, 160),
        ]
        text = reconstruct_questions_from_layout(lines)
        self.assertEqual(text, "")


class TestParentLeadUnlabelledSubs(unittest.TestCase):
    """Q.n 10 marks each followed by unmarked bodies is still a question paper."""

    def test_first_body_after_parent_is_a_and_junk_letter_is_c(self):
        lines = [
            L("Time: 3 hours Max. Marks: 80", 150, 20),
            L("Q.1 Solve any Four out of Five 5 marks each", 174, 80),
            L("Explain the challenges of language processing.", 247, 110),
            L("b Explain how an n-gram model is used in spelling correction", 186, 140),
            L("\ufffdc Explain three types of referents in resolution.", 184, 170),
            L("d Explain translation approaches used in the course.", 186, 200),
            L("e Explain the various stages of processing.", 188, 230),
            L("Q.2. 10 marks each", 176, 280),
            L("a What is disambiguation? Explain the dictionary based approach to", 189, 310),
            L("Word Sense Disambiguation.", 251, 340),
            L("b Represent morphological analysis for regular verbs with an example.", 189, 370),
            L("Q.3. 10 marks each", 176, 430),
            L("a Explain ambiguities associated at each level with example.", 189, 460),
            L("b Explain discourse reference resolution in detail.", 189, 490),
            L("Q.4 10 marks each", 176, 540),
            L("i", 188, 570),
            L("<S> Martin can watch Will <E>", 265, 600),
            L("Create a transition matrix for the tagged corpus.", 250, 640),
            L("Apply hidden markov models and do POS tagging for given statements", 250, 670),
            L("Describe in detail the centering algorithm for resolution.", 250, 700),
            L("Q.5 10 marks each", 176, 760),
            L("For a given grammar using CYK parse the statement.", 251, 790),
            L("Explain the stemming algorithm with rules", 251, 850),
        ]
        text = reconstruct_questions_from_layout(lines)
        recon = [ln.split()[0] for ln in text.splitlines() if ln.startswith("Q")]
        for qid in ("Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q1(e)", "Q2(a)", "Q2(b)", "Q3(a)", "Q3(b)"):
            self.assertIn(qid, recon, f"{qid} missing from {recon}\n{text}")
        self.assertIn("Q4(a)", recon, text)
        self.assertIn("Q4(b)", recon, text)
        self.assertNotIn("Q4(i)", recon, text)
        q1a = next(ln for ln in text.splitlines() if ln.startswith("Q1(a)"))
        self.assertIn("challenges", q1a.lower())
        q4a = next(ln for ln in text.splitlines() if ln.startswith("Q4(a)"))
        self.assertIn("transition", q4a.lower())
        self.assertNotIn("centering", q4a.lower())
        q4b = next(ln for ln in text.splitlines() if ln.startswith("Q4(b)"))
        self.assertIn("centering", q4b.lower())
        q5a = next(ln for ln in text.splitlines() if ln.startswith("Q5(a)"))
        self.assertNotIn("stemming", q5a.lower())

    def test_page_two_parent_lead_unlabelled_pair(self):
        lines = [
            L("Q.4 10 marks each", 176, 40),
            L("Create a transition matrix for the tagged corpus.", 250, 70),
            L("Describe in detail the centering algorithm for resolution.", 250, 100),
            L("Q.5 10 marks each", 176, 160),
            L("For a given grammar using CYK parse the statement.", 251, 190),
            L("S -> NP VP", 286, 220),
            L("Explain the stemming algorithm with rules", 251, 280),
            L("Q.6 10 marks each", 177, 340),
            L("Explain information retrieval versus extraction systems", 253, 370),
            L("Explain maximum entropy models for sequence labelling", 253, 400),
        ]
        text = reconstruct_questions_from_layout(lines)
        recon = [ln.split()[0] for ln in text.splitlines() if ln.startswith("Q")]
        for qid in ("Q4(a)", "Q4(b)", "Q5(a)", "Q5(b)", "Q6(a)", "Q6(b)"):
            self.assertIn(qid, recon, f"{qid} missing from {recon}\n{text}")
        q5a = next(ln for ln in text.splitlines() if ln.startswith("Q5(a)"))
        self.assertIn("CYK", q5a)
        self.assertNotIn("stemming", q5a.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
