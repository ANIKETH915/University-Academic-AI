"""
Exam year/session detection.

Papers print a syllabus revision year next to the exam date, and filenames use
separators that defeat naive word-boundary matching. Neither may produce a wrong
academic year, because year attribution drives every recurrence statistic.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.question_extractor import detect_exam_year_and_session


class TestRevisionYearNotUsed(unittest.TestCase):
    def test_revision_year_ignored_in_filename(self):
        year, session = detect_exam_year_and_session(
            "be_computer-engineering-aids_semester-7_2024_december_deep-learning-rev-2019c-scheme.pdf"
        )
        self.assertEqual(year, 2024)
        self.assertEqual(session, "Nov/Dec")

    def test_revision_year_ignored_in_header(self):
        year, _s = detect_exam_year_and_session(
            "paper.pdf",
            "B.E. (Artificial Intelligence & Machine Learning) (R-2019-20 C Scheme)\n"
            "December 2023 Examination",
        )
        self.assertEqual(year, 2023)

    def test_scheme_year_alone_is_not_an_exam_year(self):
        year, _s = detect_exam_year_and_session(
            "question-paper.pdf", "Revised Syllabus Scheme 2019 with effect from 2020"
        )
        self.assertEqual(year, 0)


class TestSeparatorHandling(unittest.TestCase):
    def test_underscore_separated_year(self):
        year, _s = detect_exam_year_and_session("sem5_2022_may_dbms.pdf")
        self.assertEqual(year, 2022)

    def test_hyphen_separated_year(self):
        year, session = detect_exam_year_and_session("os-2021-november-paper.pdf")
        self.assertEqual(year, 2021)
        self.assertEqual(session, "Nov/Dec")

    def test_no_year_returns_zero(self):
        year, _s = detect_exam_year_and_session("computer_networks_paper.pdf", "Question Paper")
        self.assertEqual(year, 0)


class TestPrintedDateWins(unittest.TestCase):
    def test_printed_exam_date_used(self):
        year, session = detect_exam_year_and_session(
            "scanned.pdf", "12/11/2024 CSE-AIML SEM-VII C SCHEME QP CODE: 10065146"
        )
        self.assertEqual(year, 2024)
        self.assertEqual(session, "Nov/Dec")

    def test_printed_date_overrides_revision_mention(self):
        year, _s = detect_exam_year_and_session(
            "paper-rev-2016-scheme.pdf", "Rev 2016 Scheme    Date: 25/05/2023"
        )
        self.assertEqual(year, 2023)

    def test_invalid_date_components_ignored(self):
        year, _s = detect_exam_year_and_session("x.pdf", "99/99/2022 nonsense, May 2021 Examination")
        self.assertEqual(year, 2021)


class TestSessionDetection(unittest.TestCase):
    def test_may_session(self):
        _y, session = detect_exam_year_and_session("paper_2025_may_subject.pdf")
        self.assertEqual(session, "May/June")

    def test_winter_session(self):
        _y, session = detect_exam_year_and_session("paper_2022_winter.pdf")
        self.assertEqual(session, "Nov/Dec")

    def test_unknown_session_reported_honestly(self):
        _y, session = detect_exam_year_and_session("paper_2022.pdf", "Total Marks 80")
        self.assertEqual(session, "Unknown session")

    def test_subject_agnostic(self):
        for name, expected in (
            ("dbms_2020_june.pdf", 2020),
            ("thermodynamics_2018_december.pdf", 2018),
            ("pharmacology-2019-may.pdf", 2019),
            ("constitutional_law_2021_november.pdf", 2021),
        ):
            with self.subTest(name=name):
                year, _s = detect_exam_year_and_session(name)
                self.assertEqual(year, expected)


class TestRealPapersDistinctYears(unittest.TestCase):
    """The four real papers must land on four different years, not all 2019."""

    def test_four_real_filenames_give_four_years(self):
        names = [
            "be_computer-engineering-aids_semester-7_2023_december_deep-learning-rev-2019c-scheme.pdf",
            "be_computer-engineering-aids_semester-7_2024_december_deep-learning-rev-2019c-scheme.pdf",
            "be_computer-engineering-aids_semester-7_2024_may_deep-learning-rev-2019c-scheme.pdf",
            "be_computer-engineering-aids_semester-7_2025_may_deep-learning-rev-2019c-scheme.pdf",
        ]
        results = [detect_exam_year_and_session(n) for n in names]
        self.assertEqual([y for y, _s in results], [2023, 2024, 2024, 2025])
        self.assertEqual(
            [s for _y, s in results],
            ["Nov/Dec", "Nov/Dec", "May/June", "May/June"],
        )
        self.assertEqual(len({(y, s) for y, s in results}), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
