"""
Layer 1–3 tests for the universal PYQ intelligence system.

Fixtures only. Production code must not depend on these cases.
No subject / university / filename rules are asserted as runtime behavior.
"""

from __future__ import annotations

import unittest

from rag.hybrid_question_extraction import score_page_representation, text_grounded_in_source
from rag.ocr_layout import _reconstruct_unlabelled_bodies, reconstruct_questions_from_layout
from rag.pyq_intelligence import (
    PYQIntelligenceEngine,
    calculate_deterministic_priority_score,
    generic_normalize_topic_title,
    quality_control_intelligence_payload,
)
from rag.question_extractor import (
    classify_repeat_relationship_full,
    looks_like_ocr_garbage_topic,
    normalize_question_text,
    structural_ocr_noise_ratio,
    validate_question_candidate,
)
from rag.visual_regions import _owner_for_region, _spans_from_questions, attach_regions_to_questions


def L(text, x0, top):
    return {"text": text, "x0": x0, "x1": x0 + 400, "top": top, "bottom": top + 18, "height": 18}


class TestLayer1UniversalExtraction(unittest.TestCase):
    def test_unlabelled_bodies_never_mint_ids(self):
        lines = [
            L("Attempt any four", 100, 40),
            L("Explain paging. [10]", 200, 80),
            L("Describe segmentation. [10]", 200, 120),
            L("Define thrashing. [10]", 200, 160),
            L("Compare FIFO and LRU. [10]", 200, 200),
        ]
        self.assertEqual(_reconstruct_unlabelled_bodies(lines), "")

    def test_non_contiguous_markers_are_complete(self):
        text = (
            "Q1(a) Explain virtual memory.\n"
            "Q1(c) Describe demand paging.\n"
            "Q1(e) What is a page fault?"
        )
        from rag.question_extractor import extract_questions_from_page_text, prepare_page_text_for_extraction

        acc, _ = extract_questions_from_page_text(
            prepare_page_text_for_extraction(text), 1, "f.pdf", "ws", year=0
        )
        ids = [q["question_id"] for q in acc]
        self.assertIn("Q1(a)", ids)
        self.assertIn("Q1(c)", ids)
        self.assertIn("Q1(e)", ids)
        self.assertNotIn("Q1(b)", ids)
        self.assertNotIn("Q1(d)", ids)

    def test_parent_instruction_is_not_a_question(self):
        text = reconstruct_questions_from_layout(
            [
                L("University Header", 400, 20),
                L("1 Attempt any four [20]", 160, 100),
                L("a) Define a process.", 240, 130),
                L("b) Explain context switching.", 240, 160),
                L("c) Describe a semaphore.", 240, 190),
                L("d) What is thrashing?", 240, 220),
                L("e) Explain paging.", 240, 250),
                L("Explain deadlock detection. [10]", 290, 310),
                L("b) Compare scheduling. [10]", 240, 340),
            ]
        )
        self.assertTrue(text)
        self.assertNotIn("Attempt any four", text)

    def test_fake_ocr_marker_is_not_source_truth(self):
        blob = "Q1(a) Explain paging.\nQ1(b) Describe segmentation."
        ok, _, _ = text_grounded_in_source("Invented Q6(t) about quantum widgets.", blob)
        self.assertFalse(ok)

    def test_representation_score_penalizes_noise_not_char_count(self):
        clean = "Q1(a) Explain demand paging and page replacement.\nQ1(b) Define thrashing."
        garbled = clean + "\n" + ("xxxx " * 80) + "\npage 3 of 3\n=====\nQ Q Q Q"
        accepted = [
            {"question_id": "Q1(a)", "exact_text": "Explain demand paging and page replacement."},
            {"question_id": "Q1(b)", "exact_text": "Define thrashing."},
        ]
        s_clean = score_page_representation(clean, accepted, [], word_quality_ratio=0.9)
        s_garbled = score_page_representation(garbled, accepted, [], word_quality_ratio=0.2)
        self.assertGreater(s_clean["score"], s_garbled["score"])
        self.assertGreater(s_garbled["noise_ratio"], s_clean["noise_ratio"])

    def test_visual_spans_are_sequential_not_full_page(self):
        qs = [
            {"question_id": "Q1(a)", "source_page": 1},
            {"question_id": "Q1(b)", "source_page": 1},
        ]
        spans = _spans_from_questions(qs, page_height=800)
        self.assertEqual(len(spans), 2)
        self.assertLess(spans[0]["y1"], 500)
        self.assertGreater(spans[1]["y0"], 300)
        region = {"page": 1, "mid_y": 700}
        self.assertEqual(_owner_for_region(region, spans), "Q1(b)")

    def test_table_attaches_to_geometry_owner(self):
        qs = [
            {"question_id": "Q2(a)", "source_page": 1, "exact_text": "Draw the table."},
            {"question_id": "Q2(b)", "source_page": 1, "exact_text": "Explain the result."},
        ]
        regions = [{
            "kind": "table",
            "page": 1,
            "x0": 0, "y0": 600, "x1": 200, "y1": 740,
            "mid_y": 670,
            "text": "A | B",
            "native_ok": True,
        }]
        out = attach_regions_to_questions(qs, regions, page_height=800)
        by_id = {q["question_id"]: q for q in out}
        self.assertIn("[TABLE]", by_id["Q2(b)"]["exact_text"])
        self.assertNotIn("[TABLE]", by_id["Q2(a)"]["exact_text"])


class TestLayer2Intelligence(unittest.TestCase):
    def test_exact_repeat_requires_safe_normalization(self):
        a = "Explain the types of Multiprocessor Systems."
        b = "Explain the types of Multiprocessor Systems."
        n1 = normalize_question_text(a)
        n2 = normalize_question_text(b)
        rel, _, conf, _ = classify_repeat_relationship_full(1.0, n1, n2, a, b)
        self.assertEqual(rel, "EXACT_REPEAT")
        self.assertEqual(conf, 1.0)

    def test_types_vs_advantages_is_related_not_semantic(self):
        a = "Explain the types of multiprocessor systems."
        b = "Discuss advantages of multiprocessor systems."
        n1 = normalize_question_text(a)
        n2 = normalize_question_text(b)
        from rag.question_extractor import compute_text_similarity

        sim = compute_text_similarity(n1, n2)
        rel, _, _, _ = classify_repeat_relationship_full(sim, n1, n2, a, b)
        self.assertIn(rel, ("RELATED_TOPIC", "DIFFERENT"))
        self.assertNotEqual(rel, "EXACT_REPEAT")

    def test_ocr_garbage_topic_suppressed(self):
        self.assertTrue(looks_like_ocr_garbage_topic("Context Switching Interrupt vs Switching Interrupt Switching"))
        self.assertTrue(looks_like_ocr_garbage_topic("ACT"))
        self.assertFalse(looks_like_ocr_garbage_topic("Multiprocessor Systems"))

    def test_topic_normalize_does_not_hardcode_subjects(self):
        t = generic_normalize_topic_title("Determine communities for the given social network using Girvan-Newman")
        self.assertTrue(t)
        self.assertNotIn("Community Detection (Girvan-Newman)", t)
        self.assertTrue("community" in t.lower() or "girvan" in t.lower())

    def test_quality_control_drops_ungrounded_groups(self):
        qs = [{
            "question_id": "Q1(a)",
            "source_file": "a.pdf",
            "exact_text": "Explain paging.",
        }]
        payload = {
            "exact_repeats": [{
                "question_ids": ["Q9(z)", "Q9(y)"],
                "source_refs": [],
                "display_title": "Invented",
            }],
            "semantic_repeats": [{
                "question_ids": ["Q1(a)"],
                "original_questions": [{"text": "Hallucinated text"}],
            }],
            "related_topics": [{
                "topic": "Paging",
                "q1": {"text": "Explain paging."},
                "q2": {"text": "Hallucinated text"},
            }],
            "topics": [{
                "topic_name": "Paging",
                "source_questions": [{"question_id": "Q1(a)", "source_file": "a.pdf"}],
                "unit": "Unmapped",
            }],
            "topic_priorities": [],
        }
        cleaned = quality_control_intelligence_payload(payload, "ws", qs)
        self.assertEqual(cleaned["exact_repeats"], [])
        self.assertEqual(cleaned["semantic_repeats"], [])
        self.assertEqual(cleaned["related_topics"], [])
        self.assertEqual(cleaned["topics"][0]["unit"], "Syllabus mapping uncertain")

    def test_engine_keeps_exact_semantic_related_separate(self):
        engine = PYQIntelligenceEngine(vector_store=None)
        qs = [
            {
                "question_id": "Q1(a)", "exact_text": "Explain the types of multiprocessor systems.",
                "normalized_text": normalize_question_text("Explain the types of multiprocessor systems."),
                "detected_topics": ["Multiprocessor Systems"], "year": 2024, "marks": 10,
                "source_file": "p1.pdf", "source_page": 1, "entities": ["multiprocessor systems"],
                "question_type": "explain", "constraints": ["types"], "exam_session": "May",
            },
            {
                "question_id": "Q2(a)", "exact_text": "Explain the types of multiprocessor systems.",
                "normalized_text": normalize_question_text("Explain the types of multiprocessor systems."),
                "detected_topics": ["Multiprocessor Systems"], "year": 2025, "marks": 10,
                "source_file": "p2.pdf", "source_page": 1, "entities": ["multiprocessor systems"],
                "question_type": "explain", "constraints": ["types"], "exam_session": "May",
            },
            {
                "question_id": "Q3(a)", "exact_text": "Discuss advantages of multiprocessor systems.",
                "normalized_text": normalize_question_text("Discuss advantages of multiprocessor systems."),
                "detected_topics": ["Multiprocessor Systems"], "year": 2025, "marks": 8,
                "source_file": "p2.pdf", "source_page": 2, "entities": ["multiprocessor systems"],
                "question_type": "discuss", "constraints": ["applications"], "exam_session": "May",
            },
        ]
        exact = engine.find_exact_repeat_groups(qs)
        exact_keys = {f"{q['source_file']}:{q['question_id']}" for g in exact for q in g["questions"]}
        semantic = engine.find_semantic_repeat_groups(qs, already_exact=exact_keys)
        related = engine.find_related_topic_pairs(qs, skip_keys=exact_keys)
        self.assertEqual(len(exact), 1)
        self.assertTrue(all(g.get("group_type") == "EXACT" for g in exact))
        self.assertTrue(all(g.get("group_type") != "EXACT" for g in semantic))
        self.assertTrue(all(r.get("is_repeat") is False for r in related))


class TestLayer3StudyPriority(unittest.TestCase):
    def test_distinct_years_outweigh_same_year_frequency(self):
        same_year, _ = calculate_deterministic_priority_score(
            appearances_count=2, distinct_years=1, exact_repeat_count=0,
            max_marks=10, last_year=2025, current_year=2026,
            syllabus_mapped=False, extraction_confidence=1.0,
        )
        three_years, _ = calculate_deterministic_priority_score(
            appearances_count=3, distinct_years=3, exact_repeat_count=0,
            max_marks=10, last_year=2025, current_year=2026,
            syllabus_mapped=False, extraction_confidence=1.0,
        )
        self.assertLess(same_year, three_years)

    def test_syllabus_and_confidence_affect_score(self):
        mapped, sig = calculate_deterministic_priority_score(
            2, 2, 0, 8, 2025, 2026, 0, 0.5, True, 1.0
        )
        unmapped, sig2 = calculate_deterministic_priority_score(
            2, 2, 0, 8, 2025, 2026, 0, 0.5, False, 0.2
        )
        self.assertGreater(mapped, unmapped)
        self.assertEqual(sig["syllabus_score"], 6.0)
        self.assertEqual(sig2["syllabus_score"], 0.0)
        self.assertGreater(sig["confidence_score"], sig2["confidence_score"])

    def test_no_guaranteed_language_in_formula_components(self):
        _, sig = calculate_deterministic_priority_score(3, 3, 1, 10, 2025, 2026)
        blob = " ".join(sig.keys()).lower()
        self.assertNotIn("guaranteed", blob)

    def test_structural_noise_detects_repetition_without_hardcoded_strings(self):
        clean = "Explain deadlock avoidance algorithms with an example."
        noisy = "deadlock deadlock deadlock xxxx xxxx page 2 of 2 ======"
        self.assertLess(structural_ocr_noise_ratio(clean), structural_ocr_noise_ratio(noisy))

    def test_garbage_only_body_is_rejected(self):
        ok, reason, _ = validate_question_candidate(
            "'' '.,'. .10 8(,ti BC D69D FAL 2CAE5 5 E5C I 67 U F'2 F7C"
        )
        self.assertFalse(ok)
        self.assertIn(reason, ("garbled_ocr_alphanumeric_noise", "low_character_quality"))

    def test_readable_formula_question_is_kept(self):
        ok, reason, _ = validate_question_candidate(
            "Suppose the stream is S = {4, 2, 5, 9}. Show how Flajolet-Martin estimates distinct elements."
        )
        self.assertTrue(ok, reason)

    def test_distinct_named_tokens_are_related_not_semantic(self):
        a = "Explain the architecture of CNN with the help of a diagram."
        b = "Explain the working of RNN with the help of a suitable diagram."
        n1, n2 = normalize_question_text(a), normalize_question_text(b)
        rel, _, _, _ = classify_repeat_relationship_full(0.55, n1, n2, a, b)
        self.assertNotEqual(rel, "SEMANTIC_REPEAT")
        self.assertNotEqual(rel, "EXACT_REPEAT")

    def test_vs_merge_collapses_repeated_side(self):
        title = generic_normalize_topic_title("Out Software Testing vs Software Testing")
        self.assertTrue(title)
        self.assertNotIn(" vs ", title.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
