"""Semantic false-positive / true-repeat regression suite.

Generic academic cases only: no subject rules, no university rules.
Verifies contradiction-first classification, validated grouping,
grounded topic labels, cache behaviour and validated study priority.
"""

import json
import unittest
import uuid
from unittest import mock

from rag.question_extractor import (
    classify_repeat_relationship_full,
    compute_text_similarity,
    normalize_question_text,
    requested_output_focus,
    topic_label_grounded_in_text,
)
from rag.pyq_intelligence import (
    PYQIntelligenceEngine,
    calculate_deterministic_priority_score,
)

try:
    from rag.vector_store import VectorStore
except Exception:
    VectorStore = None


def _judge(a, b, sem=None):
    n1, n2 = normalize_question_text(a), normalize_question_text(b)
    sim = compute_text_similarity(n1, n2)
    return classify_repeat_relationship_full(sim, n1, n2, a, b, semantic_similarity=sem)


class TestContradictionFirstClassifier(unittest.TestCase):
    def test_same_topic_different_questions_regression_case(self):
        qs = [
            "Explain regression.",
            "Derive the regression equation.",
            "Compare linear and logistic regression.",
            "Applications of regression.",
        ]
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                rel, concept, conf, reason = _judge(qs[i], qs[j])
                self.assertNotIn(
                    rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"},
                    f"{qs[i]!r} vs {qs[j]!r} wrongly grouped: {rel} {reason}",
                )
                self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"})

    def test_exact_repeat_detection_preserved(self):
        pairs = [
            ("Explain Gradient Descent in Deep Learning.", "explain gradient descent in deep learning."),
            ("What is deadlock? Explain.", "what is deadlock explain"),
        ]
        for a, b in pairs:
            rel, _, conf, reason = _judge(a, b)
            self.assertEqual(rel, "EXACT_REPEAT", reason)
            self.assertGreaterEqual(conf, 0.9)

    def test_true_paraphrase_backpropagation(self):
        rel, _, conf, reason = _judge(
            "Explain the working of backpropagation.",
            "Describe how the backpropagation algorithm works.",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_true_paraphrase_deadlock(self):
        rel, _, conf, reason = _judge(
            "Explain deadlock prevention techniques.",
            "Describe methods used to prevent deadlock.",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_cnn_vs_rnn_never_semantic(self):
        for a, b in [
            ("Explain CNN architecture.", "Explain RNN architecture."),
            ("Explain convolutional neural network architecture.", "Explain recurrent neural network architecture."),
            ("Explain the working of Kruskal's algorithm.", "Explain the working of Prim's algorithm."),
        ]:
            rel, _, _, reason = _judge(a, b)
            self.assertNotEqual(rel, "SEMANTIC_REPEAT", f"{a} vs {b}: {reason}")
            self.assertNotEqual(rel, "EXACT_REPEAT")

    def test_same_concept_different_intent_is_related(self):
        cases = [
            ("Explain TCP congestion control.", "Compare TCP congestion control algorithms."),
            ("Explain TCP congestion control.", "Derive the congestion window equation for TCP."),
            ("Define operating system.", "Explain the functions of an operating system."),
        ]
        for a, b in cases:
            rel, _, _, _ = _judge(a, b)
            self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"}, f"{a} vs {b}")

    def test_different_constraints_block_repeat(self):
        rel, _, _, reason = _judge(
            "Explain Apriori algorithm.",
            "Apply Apriori algorithm to the given transaction database.",
        )
        self.assertNotEqual(rel, "SEMANTIC_REPEAT", reason)

    def test_facet_outputs_are_not_repeats_of_plain_explanation(self):
        for facet_q, plain_q in [
            ("Applications of regression.", "Explain regression."),
            ("What are the advantages of clustering?", "Explain clustering."),
            ("List the applications of TCP.", "Explain TCP congestion control."),
        ]:
            rel, _, _, reason = _judge(facet_q, plain_q)
            self.assertNotEqual(rel, "SEMANTIC_REPEAT", reason)
            self.assertNotEqual(rel, "EXACT_REPEAT", reason)

    def test_requested_output_focus_detection(self):
        self.assertEqual(requested_output_focus("Applications of regression."), "applications")
        self.assertEqual(requested_output_focus("What are the advantages of clustering?"), "advantages_disadvantages")
        self.assertEqual(requested_output_focus("Types of operating systems."), "enumerate_types")
        self.assertEqual(requested_output_focus("Explain applications of GAN."), "applications")
        out = requested_output_focus("Explain the dropout method and its advantages.")
        self.assertEqual(out, "explanation")

    def test_unrelated_questions_stay_different(self):
        rel, _, _, _ = _judge(
            "Explain deadlock handling in databases.",
            "Construct a B-tree of order 3 for the given keys.",
        )
        self.assertIn(rel, {"DIFFERENT", "RELATED_TOPIC"})
        rel2, _, _, _ = _judge(
            "Explain the waterfall software process model.",
            "Solve the recurrence relation T(n) = 2T(n/2) + n.",
        )
        self.assertNotEqual(rel2, "SEMANTIC_REPEAT")

    def test_embedding_signal_cannot_override_vetoes_or_create_exact(self):
        rel, _, _, _ = _judge(
            "Explain CNN architecture.",
            "Explain RNN architecture.",
            sem=0.99,
        )
        self.assertNotIn(rel, {"SEMANTIC_REPEAT", "EXACT_REPEAT"})
        rel2, _, _, _ = _judge(
            "Explain the working of backpropagation.",
            "Describe how the backpropagation algorithm works.",
            sem=0.99,
        )
        self.assertNotEqual(rel2, "EXACT_REPEAT")

    def test_social_media_recommendation_paraphrase_is_semantic(self):
        rel, _, conf, reason = _judge(
            "Define a social media-based recommendation system and explain how it "
            "differs from traditional recommendation systems.",
            "What is a social media-based recommendation system and how does it "
            "differ from conventional recommendation systems?",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_backprop_algorithm_vs_procedure_is_semantic(self):
        rel, _, conf, reason = _judge(
            "Explain backpropagation algorithm.",
            "Describe the backpropagation learning procedure.",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_deadlock_definition_plus_prevention_is_semantic(self):
        rel, _, conf, reason = _judge(
            "What is deadlock and explain prevention methods.",
            "Define deadlock and discuss techniques used to prevent deadlock.",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_named_architecture_pairs_are_not_semantic(self):
        for a, b in [
            ("Explain CNN.", "Explain RNN."),
            ("Explain LSTM.", "Explain GRU."),
            ("Explain Kruskal algorithm.", "Explain Prim algorithm."),
            ("Explain linear regression.", "Explain logistic regression."),
        ]:
            rel, _, _, reason = _judge(a, b)
            self.assertNotEqual(rel, "SEMANTIC_REPEAT", f"{a} vs {b}: {reason}")
            self.assertNotEqual(rel, "EXACT_REPEAT", f"{a} vs {b}: {reason}")

    def test_regression_vs_applications_is_not_semantic(self):
        rel, _, _, reason = _judge("Explain regression.", "Explain applications of regression.")
        self.assertNotEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"})

    def test_deadlock_prevention_vs_detection_is_related(self):
        rel, _, _, reason = _judge(
            "Explain deadlock prevention.",
            "Explain deadlock detection.",
        )
        self.assertEqual(rel, "RELATED_TOPIC", reason)

    def test_hyperlink_vs_location_analytics_is_related_at_most(self):
        rel, _, _, reason = _judge(
            "Explain hyperlink analytics.",
            "Explain location analytics.",
        )
        self.assertNotEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertNotEqual(rel, "EXACT_REPEAT", reason)
        self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"})

    def test_ocr_glued_recommendation_paraphrase_is_semantic(self):
        rel, _, conf, reason = _judge(
            "What is a social media-based recommendation system and how does it differ from a traditional recommendation system?",
            "Define a social media-based recommendation system and explain how it 10 "
            "differsfromconventionalrecommendationsystems'\n[IMAGE]\n[diagram on page 1]",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_partial_entity_overlap_is_not_semantic(self):
        rel, _, _, reason = _judge(
            "Explain Social Media Action Analytics, Common Social Media Actions and Actions Analytics Tools.",
            "Explain Action Analytics with example.",
        )
        self.assertNotEqual(rel, "SEMANTIC_REPEAT", reason)

    def test_ocr_variant_same_layer_question_is_semantic(self):
        rel, _, conf, reason = _judge(
            "Explain briefly the seven layers of social media analytics with examples.",
            "Briefly describe the seven layers of social media analyics, illustrating your answer with suitable examples.",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_explain_x_vs_what_is_x_is_semantic(self):
        rel, _, conf, reason = _judge(
            "Explain backpropagation.",
            "What is backpropagation?",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_denoising_autoencoder_explain_vs_what_are_is_semantic(self):
        rel, _, conf, reason = _judge(
            "Explain denoising auto encoder model.",
            "What are Denoising Autoencoders?",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertNotEqual(rel, "RELATED_TOPIC")
        self.assertGreaterEqual(conf, 0.62)

    def test_architecture_vs_key_components_is_semantic(self):
        rel, _, conf, reason = _judge(
            "Explain RNN architecture in detail.",
            "Explain the key components of an RNN.",
        )
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)
        self.assertGreaterEqual(conf, 0.62)

    def test_lstm_gru_architecture_word_is_not_exact(self):
        rel, _, _, reason = _judge(
            "Differentiate between the architecture of LSTM and GRU network.",
            "Differentiate between the LSTM and GRU network.",
        )
        self.assertNotEqual(rel, "EXACT_REPEAT", reason)
        self.assertEqual(rel, "SEMANTIC_REPEAT", reason)

    def test_cnn_vs_rnn_working_is_not_semantic(self):
        rel, _, _, reason = _judge(
            "Explain CNN architecture.",
            "Explain RNN working.",
        )
        self.assertNotIn(rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"}, reason)

    def test_lstm_vs_gru_entities_are_not_semantic(self):
        rel, _, _, reason = _judge("Explain LSTM architecture.", "Explain GRU architecture.")
        self.assertNotIn(rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"}, reason)

    def test_kruskal_vs_prim_is_not_semantic(self):
        rel, _, _, reason = _judge(
            "Explain the working of Kruskal's algorithm.",
            "Explain the working of Prim's algorithm.",
        )
        self.assertNotIn(rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"}, reason)

    def test_linear_vs_logistic_regression_is_not_semantic(self):
        rel, _, _, reason = _judge(
            "Explain linear regression.",
            "Explain logistic regression.",
        )
        self.assertNotIn(rel, {"EXACT_REPEAT", "SEMANTIC_REPEAT"}, reason)

    def test_gan_working_vs_applications_is_related(self):
        rel, _, _, reason = _judge(
            "Explain the working of GAN.",
            "Explain applications of GAN.",
        )
        self.assertEqual(rel, "RELATED_TOPIC", reason)

    def test_cnn_architecture_vs_parameter_calculation_is_related(self):
        rel, _, _, reason = _judge(
            "Explain CNN architecture.",
            "Calculate parameters in a CNN layer.",
        )
        self.assertEqual(rel, "RELATED_TOPIC", reason)
        self.assertNotEqual(rel, "SEMANTIC_REPEAT")

    def test_activation_functions_count_vs_significance_is_not_exact(self):
        rel, _, _, reason = _judge(
            "What is an activation function? Describe any four activation functions.",
            "Explain the significance of activation functions in neural networks and "
            "different types of activation functions.",
        )
        self.assertNotEqual(rel, "EXACT_REPEAT", reason)
        self.assertIn(rel, {"RELATED_TOPIC", "DIFFERENT"}, reason)

    def test_embedding_cannot_force_architecture_drop_to_exact(self):
        rel, _, _, reason = _judge(
            "Differentiate between the architecture of LSTM and GRU network.",
            "Differentiate between the LSTM and GRU network.",
            sem=0.99,
        )
        self.assertNotEqual(rel, "EXACT_REPEAT", reason)


def make_record(text, qid, year, sf, marks=5, topics=None, session="May", **extra):
    rec = {
        "question_id": qid,
        "question_number": qid,
        "exact_text": text,
        "normalized_text": normalize_question_text(text),
        "detected_topics": topics or [],
        "year": year,
        "exam_session": session,
        "marks": marks,
        "source_file": sf,
        "source_page": 1,
        "confidence": 0.95,
        "syllabus_mapping": {"module": "Unmapped", "chapter": "Unmapped", "topic": "Unmapped"},
    }
    rec.update(extra)
    return rec


import contextlib
NO_LLM = contextlib.nullcontext()




class TestValidatedGrouping(unittest.TestCase):
    def setUp(self):
        self.engine = PYQIntelligenceEngine(vector_store=None)

    def test_regression_case_forms_no_repeat_groups(self):
        qs = [
            make_record("Explain regression.", "Q1(a)", 2023, "p2023.pdf"),
            make_record("Derive the regression equation.", "Q2(a)", 2024, "p2024.pdf"),
            make_record("Compare linear and logistic regression.", "Q3(a)", 2025, "p2025.pdf"),
            make_record("Applications of regression.", "Q4(a)", 2025, "p2025.pdf"),
        ]
        with NO_LLM:
            exact = self.engine.find_exact_repeat_groups(qs)
            exact_keys = {
                f"{q['source_file']}:{q['question_id']}" for g in exact for q in g["questions"]
            }
            semantic = self.engine.find_semantic_repeat_groups(qs, already_exact=exact_keys)
            related = self.engine.find_related_topic_pairs(qs, skip_keys=exact_keys)
        self.assertEqual(exact, [])
        self.assertEqual(semantic, [])
        self.assertGreaterEqual(len(related), 1)
        for pair in related:
            self.assertIs(pair["is_repeat"], False)
            members = pair.get("members") or []
            if members:
                self.assertGreaterEqual(len(members), 2)

    def test_related_groups_are_compact_not_pair_explosion(self):
        qs = [
            make_record("Explain the working of GAN.", "Q1(a)", 2023, "p2023.pdf"),
            make_record("Explain applications of GAN.", "Q2(a)", 2024, "p2024.pdf"),
            make_record("Explain advantages of GAN.", "Q3(a)", 2025, "p2025.pdf"),
            make_record("Explain CNN architecture.", "Q4(a)", 2023, "p2023.pdf"),
            make_record("Calculate parameters in a CNN layer.", "Q5(a)", 2024, "p2024.pdf"),
        ]
        with NO_LLM:
            related = self.engine.find_related_topic_pairs(qs)
        self.assertLessEqual(len(related), 4)
        for group in related:
            self.assertIs(group["is_repeat"], False)
            members = group.get("members") or [group.get("q1"), group.get("q2")]
            self.assertGreaterEqual(len([m for m in members if m]), 2)

    def test_topic_recurrence_without_question_repeat(self):
        qs = [
            make_record(t, q, y, sf)
            for t, q, y, sf in [
                ("Explain regression.", "Q1(a)", 2023, "p2023.pdf"),
                ("Derive the regression equation.", "Q2(a)", 2024, "p2024.pdf"),
                ("Compare linear and logistic regression.", "Q3(a)", 2025, "p2025.pdf"),
                ("Applications of regression.", "Q4(a)", 2025, "p2025.pdf"),
            ]
        ]
        with NO_LLM:
            clusters = self.engine.cluster_canonical_questions(qs)
        covered = {q["question_id"] for c in clusters for q in c["source_questions"]}
        self.assertTrue(covered, "regression questions should form topic clusters")
        self.assertEqual(len(covered), 4)

    def test_true_paraphrases_form_one_semantic_group_across_years(self):
        qs = [
            make_record("Explain the working of backpropagation.", "Q2(a)", 2023, "dl2023.pdf"),
            make_record("Describe how the backpropagation algorithm works.", "Q3(b)", 2024, "dl2024.pdf"),
            make_record("Explain CNN architecture.", "Q1(a)", 2023, "dl2023.pdf"),
            make_record("Explain LSTM architecture.", "Q4(a)", 2024, "dl2024.pdf"),
        ]
        with NO_LLM:
            exact = self.engine.find_exact_repeat_groups(qs)
            exact_keys = {
                f"{q['source_file']}:{q['question_id']}" for g in exact for q in g["questions"]
            }
            semantic = self.engine.find_semantic_repeat_groups(qs, already_exact=exact_keys)
        self.assertEqual(len(exact), 0)
        self.assertEqual(len(semantic), 1)
        group = semantic[0]
        member_ids = {f"{q['source_file']}:{q['question_id']}" for q in group["questions"]}
        self.assertEqual(
            member_ids,
            {"dl2023.pdf:Q2(a)", "dl2024.pdf:Q3(b)"},
        )
        self.assertEqual(group["years"], [2023, 2024])
        lowered = str(group.get("display_title", "")).lower()
        self.assertIn("backpropagation", lowered)

    def test_cnn_vs_rnn_not_grouped_end_to_end(self):
        qs = [
            make_record("Explain CNN architecture.", "Q1(a)", 2023, "a.pdf"),
            make_record("Explain RNN architecture.", "Q1(a)", 2024, "b.pdf"),
            make_record("Explain convolutional neural network architecture.", "Q2(a)", 2024, "b.pdf"),
        ]
        with NO_LLM:
            exact = self.engine.find_exact_repeat_groups(qs)
            semantic = self.engine.find_semantic_repeat_groups(qs)
        self.assertEqual(exact, [])
        self.assertEqual(semantic, [])

    def test_intra_group_validation_blocks_chain_merges(self):
        real_classify = "rag.pyq_intelligence.classify_repeat_relationship_full"

        L = make_record("Leader question about topic alpha mechanisms.", "Q1(a)", 2023, "a.pdf")
        M = make_record("Describe the leader question about topic alpha mechanisms.", "Q1(a)", 2024, "b.pdf")
        N = make_record("Outline the leader question about topic alpha mechanisms.", "Q1(a)", 2025, "c.pdf")
        qs = [L, M, N]

        import rag.pyq_intelligence as pmod

        orig = pmod.classify_repeat_relationship_full

        def fake(sim, n1, n2, t1="", t2="", i1=None, i2=None, **kw):
            texts = {t1, t2}
            if M["exact_text"] in texts and N["exact_text"] in texts:
                return "RELATED_TOPIC", "x", 0.3, "forced"
            return orig(sim, n1, n2, t1, t2, i1, i2, **kw)

        with NO_LLM, mock.patch(real_classify, side_effect=fake):
            groups = self.engine.find_semantic_repeat_groups(qs)
        joined = {f"{q['source_file']}:{q['question_id']}" for g in groups for q in g["questions"]}
        self.assertNotIn("c.pdf:Q1(a)", joined)

    def test_llm_judge_budget_respected(self):
        calls = {"n": 0}

        def fake_judge(a, b, **_kw):
            calls["n"] += 1
            return {"label": "RELATED_TOPIC", "confidence": 0.9, "reason": "mock"}

        qs = [
            make_record(f"Explain the k-means clustering procedure step {i}.", f"Q{i}(a)", 2023 + (i % 2), f"p{i}.pdf")
            for i in range(6)
        ]
        self.engine.find_semantic_repeat_groups(qs, llm_budget=2)
        self.assertLessEqual(calls["n"], 2)

    def test_signature_finds_paraphrase_even_with_low_lexical_overlap(self):
        a = "Explain how the Raft consensus protocol operates."
        b = "Describe the operation of the Raft consensus algorithm."
        qs = [
            make_record(a, "Q1(a)", 2023, "d1.pdf", entities=["raft consensus"], constraints=[], question_type="explain"),
            make_record(b, "Q2(a)", 2024, "d2.pdf", entities=["raft consensus"], constraints=[], question_type="explain"),
        ]
        with mock.patch("rag.pyq_intelligence.compute_text_similarity", return_value=0.15), NO_LLM:
            groups = self.engine.find_semantic_repeat_groups(qs, embeddings=None)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["repeat_count"], 2)

    def test_embedding_cannot_force_contradictory_entities(self):
        from rag.semantic_similarity import embed_texts

        a = "Explain CNN architecture."
        b = "Explain RNN architecture."
        qs = [
            make_record(a, "Q1(a)", 2023, "d1.pdf"),
            make_record(b, "Q2(a)", 2024, "d2.pdf"),
        ]
        matrix = embed_texts([normalize_question_text(a), normalize_question_text(b)])
        if matrix is None:
            self.skipTest("embedding model unavailable")
        with NO_LLM:
            groups = self.engine.find_semantic_repeat_groups(qs, embeddings=matrix)
        self.assertEqual(groups, [])


class TestTopicLabelGrounding(unittest.TestCase):
    def test_grounded_and_ungrounded_labels(self):
        src = "Explain the Banker's algorithm for deadlock avoidance."
        self.assertTrue(topic_label_grounded_in_text("Banker's Algorithm", src))
        self.assertTrue(topic_label_grounded_in_text("deadlock avoidance", src))
        self.assertFalse(topic_label_grounded_in_text("Quantum Blockchain Synergy", src))
        self.assertFalse(topic_label_grounded_in_text("", src))

    def test_enrich_topics_with_llm_filters_hallucinated_primary(self):
        from rag.hybrid_question_extraction import enrich_topics_with_llm

        good = {
            "question_id": "Q1",
            "exact_text": "Explain the Banker's algorithm for deadlock avoidance.",
        }

        def fake_call(system, user, **kw):
            if "Q1" in user:
                return {
                    "primary_topic": "Quantum Blockchain Synergy",
                    "secondary_topics": ["Blockchain Sharding"],
                    "entities": [],
                    "constraints": [],
                }
            return {
                "primary_topic": "Deadlock Avoidance",
                "secondary_topics": [],
                "entities": [],
                "constraints": [],
            }

        bad = {"question_id": "Q2", "exact_text": "Describe methods used for deadlock avoidance in operating systems."}
        enrich_topics_with_llm([good, bad])
        self.assertNotIn("primary_topic", good)


@unittest.skipIf(VectorStore is None, "VectorStore unavailable")
class TestWorkspacePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = VectorStore()
        cls.engine = PYQIntelligenceEngine(vector_store=cls.store)

    def _inject(self, ws, records):
        docs = [r["exact_text"] for r in records]
        metas = []
        ids = []
        for idx, r in enumerate(records):
            metas.append({
                "workspace_id": ws,
                "doc_type": "pyq",
                "source_file": r["source_file"],
                "year": str(r["year"]),
                "exam_session": r.get("exam_session", "May"),
                "question_id": r["question_id"],
                "question_number": r["question_id"],
                "parent_question": r["question_id"].split("(")[0],
                "exact_text": r["exact_text"],
                "normalized_text": normalize_question_text(r["exact_text"]),
                "marks": str(r.get("marks", 5)),
                "confidence": "0.95",
                "extraction_quality": "COMPLETE",
                "grounding_status": "grounded",
                "extraction_method": "native",
                "source_page": r.get("source_page", 1),
                "detected_topics": json.dumps(r.get("topics", [])),
                "entities": json.dumps(r.get("entities", []) or []),
                "constraints": json.dumps(r.get("constraints", []) or []),
            })
            ids.append(f"{ws}:{idx}:{uuid.uuid4().hex[:8]}")
        self.store.collection.add(
            documents=docs,
            metadatas=metas,
            ids=ids,
            embeddings=self.store.embed_texts(docs),
        )

    def test_cache_and_exact_repeat_pipeline(self):
        ws = f"ws-suite-cache-{uuid.uuid4().hex[:8]}"
        try:
            self._inject(ws, [
                make_record("Explain gradient descent optimization.", "Q1(a)", 2023, "g2023.pdf"),
                make_record("explain gradient descent optimization.", "Q1(a)", 2024, "g2024.pdf"),
                make_record("Explain CNN architecture.", "Q2(a)", 2023, "g2023.pdf"),
            ])
            calls = {"n": 0}
            original = self.engine._compute_pyq_analysis

            def counted(*args, **kwargs):
                calls["n"] += 1
                return original(*args, **kwargs)

            with NO_LLM, mock.patch.object(self.engine, "_compute_pyq_analysis", side_effect=counted):
                first = self.engine.get_pyq_analysis(ws)
                second = self.engine.get_pyq_analysis(ws)
            self.assertEqual(calls["n"], 1, "second identical request must be served from cache")
            self.assertTrue(first["available"])
            self.assertEqual(first["summary_stats"], second["summary_stats"])
            self.assertGreaterEqual(first["summary_stats"]["exact_repeats"], 1)
            first["summary_stats"]["exact_repeats"] = 999
            third = self.engine.get_pyq_analysis(ws)
            self.assertNotEqual(third["summary_stats"]["exact_repeats"], 999, "cached result must be isolated (deepcopy)")
        finally:
            self.store.delete_by_workspace(ws)

    def test_study_priority_consumes_only_validated_evidence(self):
        ws = f"ws-suite-prio-{uuid.uuid4().hex[:8]}"
        try:
            self._inject(ws, [
                make_record("Explain regression.", "Q1(a)", 2023, "r2023.pdf"),
                make_record("Derive the regression equation.", "Q2(a)", 2024, "r2024.pdf"),
                make_record("Compare linear and logistic regression.", "Q3(a)", 2025, "r2025.pdf"),
                make_record("Applications of regression.", "Q4(a)", 2025, "r2025.pdf"),
            ])
            with NO_LLM:
                analysis = self.engine.get_pyq_analysis(ws)
            self.assertTrue(analysis["available"])
            self.assertEqual(analysis["summary_stats"]["repeated_questions_count"], 0)
            self.assertEqual(analysis["exact_repeat_count"], 0)
            self.assertEqual(analysis["semantic_repeat_count"], 0)
            for topic in analysis["topic_priorities"]:
                self.assertEqual(topic["exact_repeat_count"], 0)
                self.assertEqual(topic["semantic_repeat_count"], 0)
                self.assertEqual(topic["signals"]["exact_repeat_score"], 0.0)
            self.assertTrue(analysis["topics"], "topic recurrence should still exist")
            with NO_LLM:
                priority = self.engine.get_study_priority(ws)
            for item in priority["top_high_priority_topics"]:
                self.assertTrue(item["why"])
                self.assertTrue(item["source_questions"])
        finally:
            self.store.delete_by_workspace(ws)

    def test_hallucinated_topic_labels_filtered_at_load(self):
        ws = f"ws-suite-topic-{uuid.uuid4().hex[:8]}"
        try:
            self._inject(ws, [
                make_record(
                    "Explain deadlock avoidance in operating systems.",
                    "Q1(a)", 2023, "os.pdf",
                    topics=["Fabricated Quantum Synergy"],
                ),
            ])
            with NO_LLM:
                qs = self.engine.get_source_questions(ws)
            self.assertEqual(len(qs), 1)
            names = [str(t).lower() for t in qs[0]["detected_topics"]]
            self.assertFalse(any("fabricated" in n or "quantum" in n for n in names), names)
        finally:
            self.store.delete_by_workspace(ws)

    def test_distinct_years_dominate_priority_formula(self):
        low, sig_low = calculate_deterministic_priority_score(
            appearances_count=4, distinct_years=1, exact_repeat_count=0,
            max_marks=10, last_year=2025, current_year=2026,
        )
        high, sig_high = calculate_deterministic_priority_score(
            appearances_count=4, distinct_years=4, exact_repeat_count=0,
            max_marks=10, last_year=2025, current_year=2026,
        )
        self.assertLess(low, high)
        self.assertGreater(sig_high["year_recurrence_score"], sig_low["year_recurrence_score"])

    def test_workspace_isolation_between_analyses(self):
        ws_a = f"ws-suite-iso-a-{uuid.uuid4().hex[:8]}"
        ws_b = f"ws-suite-iso-b-{uuid.uuid4().hex[:8]}"
        try:
            self._inject(ws_a, [
                make_record("Explain gradient descent optimization.", "Q1(a)", 2023, "ga.pdf"),
                make_record("explain gradient descent optimization.", "Q1(a)", 2024, "gb.pdf"),
            ])
            self._inject(ws_b, [
                make_record("Explain CNN architecture.", "Q1(a)", 2023, "ca.pdf"),
            ])
            with NO_LLM:
                res_a = self.engine.get_pyq_analysis(ws_a)
                res_b = self.engine.get_pyq_analysis(ws_b)
            self.assertGreaterEqual(res_a["exact_repeat_count"], 1)
            self.assertEqual(res_b["exact_repeat_count"], 0)
            blob_b = json.dumps(res_b["most_repeated_questions"])
            self.assertNotIn("gradient descent", blob_b.lower())
        finally:
            self.store.delete_by_workspace(ws_a)
            self.store.delete_by_workspace(ws_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
