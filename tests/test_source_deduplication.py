"""
Regression suite: PYQ Intelligence source-paper deduplication.

Recurrence must count distinct canonical papers, never raw matching records.
No university / subject / year / filename / question-number catalog.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pyq_intelligence import PYQIntelligenceEngine
from rag.question_extractor import normalize_question_text
from rag.source_identity import (
    attach_source_identity,
    content_fingerprint,
    file_bytes_hash,
    normalize_filename_stem,
    paper_id_of,
    unique_occurrence_count,
    unique_session_identities,
    unique_years,
)


ACTIVATION = (
    "What is the significance of Activation Functions in Neural Networks, "
    "explain different types Activation functions used in NN."
)
GRADIENT = "Explain Gradient Descent in Deep Learning."
DROPOUT = "Explain the dropout method and its advantages."
CNN = "Explain CNN architecture in detail."
LSTM = "Explain LSTM architecture."
POOLING = "Explain pooling operation in CNN."
BACKPROP = "Explain the working of backpropagation."
BACKPROP_PARA = "Describe how the backpropagation algorithm works."
DEADLOCK = "Explain deadlock prevention techniques."
DEADLOCK_PARA = "Describe methods used to prevent deadlock."


def q(
    text: str,
    year: int,
    session: str,
    source_file: str,
    question_id: str = "Q1(a)",
    **extra,
) -> dict:
    rec = {
        "question_id": question_id,
        "question_number": question_id,
        "exact_text": text,
        "normalized_text": normalize_question_text(text),
        "detected_topics": extra.pop("detected_topics", None) or [],
        "year": year,
        "exam_session": session,
        "marks": extra.pop("marks", 10),
        "source_file": source_file,
        "source_page": 1,
        "university": extra.pop("university", "Example University"),
        "subject": extra.pop("subject", "Example Subject"),
        "course_code": extra.pop("course_code", ""),
        "entities": extra.pop("entities", []),
        "constraints": extra.pop("constraints", []),
        "question_type": extra.pop("question_type", "explain"),
        "syllabus_mapping": extra.pop(
            "syllabus_mapping",
            {"module": "Module 1", "chapter": "Unit", "topic": "Topic"},
        ),
        "confidence": 0.9,
    }
    rec.update(extra)
    if not rec["detected_topics"]:
        rec["detected_topics"] = [text.split()[1] if len(text.split()) > 1 else "Topic"]
    return rec


def paper(year, session, filename, questions, **meta):
    """Build a multi-question paper. `questions` is [(qid, text), ...]."""
    return [
        q(text, year, session, filename, question_id=qid, **dict(meta))
        for qid, text in questions
    ]


class TestFilenameAndFingerprintHelpers(unittest.TestCase):
    def test_copy_suffixes_collapse_generically(self):
        a = "exam-paper.pdf"
        variants = [
            "exam-paper (1).pdf",
            "exam-paper - Copy.pdf",
            "exam-paper - Copy (2).pdf",
            "exam-paper_copy.pdf",
            "exam-paper copy.pdf",
        ]
        stem = normalize_filename_stem(a)
        for name in variants:
            self.assertEqual(normalize_filename_stem(name), stem, name)

    def test_distinct_stems_stay_distinct(self):
        self.assertNotEqual(
            normalize_filename_stem("exam-2024-may.pdf"),
            normalize_filename_stem("exam-2025-may.pdf"),
        )

    def test_content_fingerprint_ignores_question_numbers(self):
        t = [normalize_question_text(ACTIVATION), normalize_question_text(GRADIENT)]
        self.assertEqual(content_fingerprint(t), content_fingerprint(list(reversed(t))))


class TestSourceIdentityRules(unittest.TestCase):
    def test_identical_bytes_merge_regardless_of_filename(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            p1 = os.path.join(tmp.name, "alpha.pdf")
            p2 = os.path.join(tmp.name, "totally-different-name.pdf")
            payload = b"%PDF-1.4 identical-bytes-payload-for-source-identity\n"
            with open(p1, "wb") as fh:
                fh.write(payload)
            shutil.copyfile(p1, p2)
            self.assertEqual(file_bytes_hash(p1), file_bytes_hash(p2))
            recs = [
                *paper(2024, "May/June", "alpha.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)], source_path=p1),
                *paper(2024, "May/June", "totally-different-name.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)], source_path=p2),
            ]
            attach_source_identity(recs)
            self.assertEqual(unique_occurrence_count(recs), 1)
            self.assertEqual(recs[0]["canonical_paper_id"], recs[2]["canonical_paper_id"])
            self.assertEqual(recs[0]["source_identity_confidence"], "high")
            self.assertIn("totally-different-name.pdf", recs[0]["duplicate_source_ids"])
        finally:
            tmp.cleanup()

    def test_same_content_copied_under_another_filename(self):
        recs = [
            *paper(2024, "May/June", "midterm.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN), ("Q3(a)", LSTM)]),
            *paper(2024, "May/June", "midterm - Copy.pdf", [("Q6(a)", ACTIVATION), ("Q7(a)", CNN), ("Q8(a)", LSTM)]),
        ]
        attach_source_identity(recs)
        self.assertEqual(unique_occurrence_count(recs), 1)
        self.assertEqual(paper_id_of(recs[0]), paper_id_of(recs[3]))

    def test_duplicate_same_session_paper_collapses(self):
        recs = [
            *paper(2024, "May/June", "session-a.pdf", [("Q1(a)", GRADIENT), ("Q2(a)", DROPOUT)]),
            *paper(2024, "May/June", "session-a (1).pdf", [("Q1(a)", GRADIENT), ("Q2(a)", DROPOUT)]),
        ]
        attach_source_identity(recs)
        self.assertEqual(unique_occurrence_count(recs), 1)

    def test_same_question_different_years_stay_separate(self):
        recs = [
            *paper(2023, "May/June", "y2023.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)]),
            *paper(2024, "May/June", "y2024.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)]),
        ]
        attach_source_identity(recs)
        self.assertEqual(unique_occurrence_count(recs), 2)
        self.assertEqual(unique_years(recs), [2023, 2024])

    def test_same_question_different_sessions_stay_separate(self):
        recs = [
            *paper(2024, "May/June", "summer.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)]),
            *paper(2024, "Nov/Dec", "winter.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)]),
        ]
        attach_source_identity(recs)
        self.assertEqual(unique_occurrence_count(recs), 2)
        self.assertEqual(len(unique_session_identities(recs)), 2)

    def test_same_question_number_different_content_not_merged(self):
        recs = [
            *paper(2024, "May/June", "paper-one.pdf", [("Q6(a)", ACTIVATION), ("Q6(b)", CNN)]),
            *paper(2024, "May/June", "paper-two.pdf", [("Q6(a)", POOLING), ("Q6(b)", LSTM)]),
        ]
        attach_source_identity(recs)
        self.assertEqual(unique_occurrence_count(recs), 2)
        ids = {paper_id_of(r) for r in recs}
        self.assertEqual(len(ids), 2)

    def test_uncertain_identity_does_not_silently_merge(self):
        recs = [
            q(ACTIVATION, 2024, "May/June", "alpha-set.pdf", "Q6(a)", university="", subject=""),
            q(ACTIVATION, 2024, "May/June", "unrelated-name.pdf", "Q6(a)", university="", subject=""),
        ]
        attach_source_identity(recs)
        self.assertEqual(unique_occurrence_count(recs), 2)
        self.assertNotEqual(paper_id_of(recs[0]), paper_id_of(recs[1]))

    def test_question_number_never_determines_identity(self):
        recs = [
            *paper(2024, "Nov/Dec", "a.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)]),
            *paper(2025, "May/June", "b.pdf", [("Q6(a)", ACTIVATION), ("Q7(a)", DROPOUT)]),
        ]
        attach_source_identity(recs)
        self.assertEqual(unique_occurrence_count(recs), 2)


class TestIntelligenceOccurrenceCounts(unittest.TestCase):
    def setUp(self):
        self.engine = PYQIntelligenceEngine(vector_store=None)

    def test_identical_pdf_uploaded_twice_is_not_a_repeat(self):
        recs = [
            *paper(2024, "May/June", "final.pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)]),
            *paper(2024, "May/June", "final (1).pdf", [("Q1(a)", ACTIVATION), ("Q2(a)", CNN)]),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(groups, [])
        self.assertEqual(unique_occurrence_count(recs), 1)

    def test_same_content_other_filename_not_a_repeat(self):
        recs = [
            *paper(2024, "Nov/Dec", "winter.pdf", [("Q3(a)", GRADIENT), ("Q3(b)", LSTM)]),
            *paper(2024, "Nov/Dec", "winter - Copy.pdf", [("Q3(a)", GRADIENT), ("Q3(b)", LSTM)]),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(groups, [])

    def test_duplicate_session_does_not_inflate_exact_repeat(self):
        recs = [
            *paper(
                2024, "Nov/Dec", "2024-nov.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", CNN)],
                detected_topics=["Activation Functions Neural"],
            ),
            *paper(
                2024, "May/June", "2024-may.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", DROPOUT)],
                detected_topics=["Activation Functions Neural"],
            ),
            *paper(
                2024, "May/June", "2024-may - Copy.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", DROPOUT)],
                detected_topics=["Activation Functions Neural"],
            ),
            *paper(
                2025, "May/June", "2025-may.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", LSTM)],
                detected_topics=["Activation Functions Neural"],
            ),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat_count"], 3)
        self.assertEqual(groups[0]["unique_occurrence_count"], 3)
        self.assertEqual(len(groups[0]["source_refs"]), 3)
        self.assertEqual(len(groups[0]["original_questions"]), 3)
        self.assertEqual(unique_years(groups[0]["original_questions"]), [2024, 2025])
        refs = groups[0]["source_refs"]
        self.assertEqual(len(refs), len(set(refs)))

    def test_gradient_style_duplicate_session_counts_two(self):
        recs = [
            *paper(2024, "May/June", "may.pdf", [("Q6(a)", GRADIENT), ("Q6(b)", CNN)]),
            *paper(2024, "May/June", "may (1).pdf", [("Q6(a)", GRADIENT), ("Q6(b)", CNN)]),
            *paper(2025, "May/June", "may25.pdf", [("Q6(a)", GRADIENT), ("Q6(b)", LSTM)]),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat_count"], 2)
        self.assertEqual(len(groups[0]["source_refs"]), 2)

    def test_same_question_different_years_counts_two(self):
        recs = [
            *paper(2023, "Nov/Dec", "a.pdf", [("Q2(a)", LSTM), ("Q2(b)", CNN)]),
            *paper(2024, "May/June", "b.pdf", [("Q4(a)", LSTM), ("Q4(b)", DROPOUT)]),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(groups[0]["repeat_count"], 2)
        self.assertEqual(groups[0]["years"], [2023, 2024])

    def test_same_question_different_sessions_counts_two(self):
        recs = [
            *paper(2024, "May/June", "s1.pdf", [("Q1(a)", CNN), ("Q1(b)", DROPOUT)]),
            *paper(2024, "Nov/Dec", "s2.pdf", [("Q5(a)", CNN), ("Q5(b)", LSTM)]),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(groups[0]["repeat_count"], 2)
        self.assertEqual(len(unique_session_identities(recs)), 2)

    def test_same_question_number_different_content_is_not_a_repeat(self):
        recs = [
            *paper(2024, "May/June", "p1.pdf", [("Q6(a)", ACTIVATION), ("Q6(b)", CNN)]),
            *paper(2024, "May/June", "p2.pdf", [("Q6(a)", POOLING), ("Q6(b)", LSTM)]),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(groups, [])

    def test_exact_repeat_across_three_genuine_papers(self):
        recs = [
            *paper(2022, "May/June", "p2022.pdf", [("Q1(a)", ACTIVATION), ("Q1(b)", CNN)]),
            *paper(2023, "Nov/Dec", "p2023.pdf", [("Q3(a)", ACTIVATION), ("Q3(b)", DROPOUT)]),
            *paper(2024, "May/June", "p2024.pdf", [("Q6(a)", ACTIVATION), ("Q6(b)", LSTM)]),
        ]
        groups = self.engine.find_exact_repeat_groups(recs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat_count"], 3)
        self.assertEqual(groups[0]["distinct_years_count"], 3)
        self.assertEqual(len(groups[0]["canonical_paper_ids"]), 3)

    def test_semantic_repeat_across_three_genuine_papers(self):
        recs = [
            q(BACKPROP, 2022, "May/June", "s2022.pdf", "Q1(a)", entities=["backpropagation"], detected_topics=["Backpropagation"]),
            q(BACKPROP_PARA, 2023, "Nov/Dec", "s2023.pdf", "Q2(a)", entities=["backpropagation"], detected_topics=["Backpropagation"]),
            q("Explain how the backpropagation algorithm works.", 2024, "May/June", "s2024.pdf", "Q3(a)", entities=["backpropagation"], detected_topics=["Backpropagation"]),
            q("Explain how the backpropagation algorithm works.", 2024, "May/June", "s2024 - Copy.pdf", "Q3(a)", entities=["backpropagation"], detected_topics=["Backpropagation"]),
        ]

        def _admit(*_args, **_kwargs):
            return "SEMANTIC_REPEAT", "Backpropagation", 0.8, "equivalent intent"

        with mock.patch("rag.pyq_intelligence.classify_repeat_relationship_full", side_effect=_admit):
            groups = self.engine.find_semantic_repeat_groups(recs)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat_count"], 3)
        self.assertEqual(groups[0]["unique_occurrence_count"], 3)
        self.assertEqual(len(groups[0]["source_refs"]), 3)
        self.assertEqual(len(groups[0]["canonical_paper_ids"]), 3)

    def test_topic_appearances_use_unique_papers(self):
        recs = [
            *paper(2024, "May/June", "may.pdf", [("Q1(a)", CNN), ("Q1(b)", DROPOUT)], detected_topics=["CNN"]),
            *paper(2024, "May/June", "may - Copy.pdf", [("Q1(a)", CNN), ("Q1(b)", DROPOUT)], detected_topics=["CNN"]),
            *paper(2025, "May/June", "may25.pdf", [("Q2(a)", CNN), ("Q2(b)", LSTM)], detected_topics=["CNN"]),
        ]
        clusters = self.engine.cluster_canonical_questions(recs)
        cnn = next(c for c in clusters if "cnn" in c["topic_name"].lower())
        self.assertEqual(cnn["appearances_count"], 2)
        self.assertEqual(cnn["unique_occurrence_count"], 2)


class TestAnalysisPayloadDedup(unittest.TestCase):
    def test_most_repeated_asked_count_and_sources(self):
        engine = PYQIntelligenceEngine(vector_store=None)
        recs = [
            *paper(
                2024, "Nov/Dec", "2024-nov.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", CNN)],
                detected_topics=["Activation Functions Neural"],
            ),
            *paper(
                2024, "May/June", "2024-may.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", DROPOUT)],
                detected_topics=["Activation Functions Neural"],
            ),
            *paper(
                2024, "May/June", "2024-may - Copy.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", DROPOUT)],
                detected_topics=["Activation Functions Neural"],
            ),
            *paper(
                2025, "May/June", "2025-may.pdf",
                [("Q6(a)", ACTIVATION), ("Q6(b)", LSTM)],
                detected_topics=["Activation Functions Neural"],
            ),
            *paper(
                2024, "May/June", "gd-may.pdf",
                [("Q3(a)", GRADIENT), ("Q3(b)", POOLING)],
                detected_topics=["Gradient Descent Deep"],
            ),
            *paper(
                2024, "May/June", "gd-may - Copy.pdf",
                [("Q3(a)", GRADIENT), ("Q3(b)", POOLING)],
                detected_topics=["Gradient Descent Deep"],
            ),
            *paper(
                2025, "May/June", "gd-2025.pdf",
                [("Q3(a)", GRADIENT), ("Q3(b)", BACKPROP)],
                detected_topics=["Gradient Descent Deep"],
            ),
        ]
        paper_stats = {
            r["source_file"]: {
                "source_file": r["source_file"],
                "exam_year": r["year"],
                "exam_session": r["exam_session"],
                "valid_questions": 1,
                "rejected_questions": 0,
                "exact_repeats": 0,
                "semantic_repeats": 0,
            }
            for r in recs
        }
        with mock.patch("rag.pyq_intelligence.embed_texts", return_value=None), mock.patch(
            "rag.pyq_intelligence.build_syllabus_index_from_workspace", return_value={}
        ):
            result = engine._compute_pyq_analysis(
                "ws-dedup-unit", "Example Subject", "Semester 1", False, recs, paper_stats
            )

        self.assertIn("source_deduplication", result)
        self.assertLess(result["source_deduplication"]["unique_papers"], result["source_deduplication"]["raw_source_files"])
        self.assertGreaterEqual(result["source_deduplication"]["collapsed_duplicate_files"], 2)

        most = result["most_repeated_questions"]
        activation = next(
            (m for m in most if "activation" in (m.get("sample_text") or "").lower() or "activation" in (m.get("title") or "").lower()),
            None,
        )
        self.assertIsNotNone(activation)
        self.assertEqual(activation["asked_count"], 3)
        self.assertEqual(activation["unique_occurrence_count"], 3)
        self.assertEqual(len(activation["sources"]), 3)
        self.assertEqual(len(activation["sources"]), len(set(activation["sources"])))
        self.assertIn("canonical_paper_ids", activation)

        gradient = next(
            (m for m in most if "gradient" in (m.get("sample_text") or "").lower() or "gradient" in (m.get("title") or "").lower()),
            None,
        )
        self.assertIsNotNone(gradient)
        self.assertEqual(gradient["asked_count"], 2)
        self.assertEqual(len(gradient["sources"]), 2)

        # Priority / frequency must use unique papers, not raw records.
        act_priority = next(
            (qp for qp in result["question_priorities"] if "activation" in (qp.get("sample_text") or "").lower()),
            None,
        )
        if act_priority:
            self.assertEqual(act_priority["unique_occurrence_count"], 3)
            self.assertLessEqual(act_priority["signals"]["frequency_score"], 3 * 5.5)

        self.assertEqual(result["total_papers"], result["source_deduplication"]["unique_papers"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
