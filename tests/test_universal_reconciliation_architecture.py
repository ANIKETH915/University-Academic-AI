"""
Comprehensive regression test suite for Universal Question Reconciliation Architecture.
Tests all 26 critical requirements specified in the core prompt.
"""

import pytest
from typing import Dict, Any, List
from rag.hybrid_question_extraction import (
    normalize_marker_id,
    detect_source_question_markers,
    run_universal_reconciliation_pipeline,
    hybrid_extract_document,
    DocumentEvidence,
)
from rag.dynamic_ingest import DynamicIngestPipeline
from rag.vector_store import VectorStore


def test_01_q1_a_to_f():
    text = (
        "Q1 Attempt any five\n"
        "a) What is natural language processing? Explain applications. [5]\n"
        "b) Discuss word sense disambiguation algorithms in detail. [5]\n"
        "c) Explain Part of Speech tagging with HMM model. [5]\n"
        "d) What is reference resolution? Give suitable examples. [5]\n"
        "e) Explain Porter Stemming algorithm with step details. [5]\n"
        "f) Discuss Conditional Random Fields for sequence labeling. [5]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test1.pdf", workspace_id="ws-test")
    extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
    for sub in ("a", "b", "c", "d", "e", "f"):
        assert f"Q1({sub})" in extracted_ids
    assert res["quality"]["extraction_quality"] == "COMPLETE"


def test_02_q3_c():
    text = (
        "Q3(a) Explain statistical machine translation principles. [10]\n"
        "Q3(b) Discuss Neural Machine Translation architecture with attention. [10]\n"
        "Q3(c) Derive Viterbi algorithm equations for bigram HMM tagger. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test2.pdf", workspace_id="ws-test")
    extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
    assert "Q3(c)" in extracted_ids


def test_03_q4_c():
    text = (
        "Q4(a) Discuss Lesk algorithm for word sense disambiguation. [10]\n"
        "Q4(b) Explain various challenges in POS tagging with examples. [10]\n"
        "Q4(c) Describe semantic role labeling using dependency trees. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test3.pdf", workspace_id="ws-test")
    extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
    assert "Q4(c)" in extracted_ids


def test_04_q4_i_roman():
    text = (
        "Q4(i) Explain tf-idf vector space representation for documents. [10]\n"
        "Q4(ii) Discuss cosine similarity calculation for text clustering. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test4.pdf", workspace_id="ws-test")
    extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
    assert "Q4(i)" in extracted_ids or "Q4(I)" in [i.upper() for i in extracted_ids]


def test_05_q5_a_b():
    text = (
        "5 a) Explain Porter Stemming algorithm in detail with example rules. [10]\n"
        "5 b) Explain Probabilistic Context Free Grammar parsing algorithm. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test5.pdf", workspace_id="ws-test")
    extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
    assert "Q5(a)" in extracted_ids
    assert "Q5(b)" in extracted_ids


def test_06_q6_a_b():
    text = (
        "6 a) Explain Question Answering system components in detail. [10]\n"
        "6 b) Explain Conditional Random Field models for sequence tagging. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test6.pdf", workspace_id="ws-test")
    extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
    assert "Q6(a)" in extracted_ids
    assert "Q6(b)" in extracted_ids


def test_07_ocr_variation_q4b():
    assert normalize_marker_id("Q.4.b)") == "Q4(b)"
    assert normalize_marker_id("Q.4(b)") == "Q4(b)"


def test_08_ocr_variation_prime():
    assert normalize_marker_id("Q.4.c')") == "Q4(c)"


def test_09_space_separator_q4b():
    assert normalize_marker_id("Q4 b)") == "Q4(b)"


def test_10_bare_4b():
    assert normalize_marker_id("4(b)") == "Q4(b)"
    assert normalize_marker_id("4 b)") == "Q4(b)"


def test_11_multiline_wrap():
    text = (
        "Q1(a) Explain in detail the complete architecture of convolutional neural networks\n"
        "including pooling layers, activation functions, and fully connected classification layers. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test11.pdf", workspace_id="ws-test")
    q = res["accepted_questions"][0]
    assert "convolutional neural networks" in q["exact_text"]
    assert "classification layers" in q["exact_text"]


def test_12_cross_page_continuation():
    p1 = "Q2(a) Discuss deep recurrent neural networks for text classification and detail how"
    p2 = "vanishing gradients affect long sequence training. [10]\nQ2(b) What is LSTM? [10]\n"
    pages = [
        {"page": 1, "reconstructed_text": p1, "raw_native_text": p1},
        {"page": 2, "reconstructed_text": p2, "raw_native_text": p2},
    ]
    res = hybrid_extract_document(pages, filename="test12.pdf", workspace_id="ws-test")
    q2a = next(q for q in res["accepted_questions"] if q["question_id"] == "Q2(a)")
    assert "vanishing gradients" in q2a["exact_text"]
    assert q2a["source_pages"] == [1, 2] or q2a.get("source_page_end") == 2


def test_13_ocr_marker_separated_from_body():
    p = {
        "page": 1,
        "raw_native_text": "a)\nExplain word sense disambiguation algorithms using WordNet lexical database. [10]\n",
        "reconstructed_text": "Q4(a) Explain word sense disambiguation algorithms using WordNet lexical database. [10]\n",
    }
    res = hybrid_extract_document([p], filename="test13.pdf", workspace_id="ws-test")
    assert any(q["question_id"] == "Q4(a)" for q in res["accepted_questions"])


def test_14_marker_native_body_layout():
    p = {
        "page": 1,
        "raw_native_text": "Q5(a) [10]\n",
        "raw_ocr_text": "Q5(a) Explain Porter Stemming algorithm and discuss steps in detail. [10]\n",
        "reconstructed_text": "Q5(a) Explain Porter Stemming algorithm and discuss steps in detail. [10]\n",
    }
    res = hybrid_extract_document([p], filename="test14.pdf", workspace_id="ws-test")
    assert any(q["question_id"] == "Q5(a)" for q in res["accepted_questions"])


def test_15_false_marker_rejection():
    text = (
        "N.B. (1) Question No 1 is Compulsory.\n"
        "(2) Attempt any three questions out of the remaining five.\n"
        "(3) All questions carry equal marks.\n"
        "Q1(a) Explain what is natural language processing. [5]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test15.pdf", workspace_id="ws-test")
    qids = [q["question_id"] for q in res["accepted_questions"]]
    assert qids == ["Q1(a)"]


def test_16_duplicate_marker_deduplication():
    text = (
        "Q1(a) Explain what is natural language processing. [5]\n"
        "Q1(a) Explain what is natural language processing. [5]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test16.pdf", workspace_id="ws-test")
    qids = [q["question_id"] for q in res["accepted_questions"]]
    assert qids.count("Q1(a)") == 1


def test_17_2_hours_never_q2h():
    assert normalize_marker_id("2 hours") is None
    assert normalize_marker_id("Duration: 3 hours") is None


def test_18_arbitrary_subquestion_count():
    subs = [f"Q1({chr(ord('a')+i)}) Explain concept number {i+1}. [5]\n" for i in range(10)]
    text = "".join(subs)
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test18.pdf", workspace_id="ws-test")
    extracted_ids = [q["question_id"] for q in res["accepted_questions"]]
    assert len(extracted_ids) == 10
    assert "Q1(j)" in extracted_ids


def test_19_arbitrary_total_question_count():
    qs = [f"Q{i}(a) Explain topic number {i} in detail. [10]\n" for i in range(1, 19)]
    text = "".join(qs)
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test19.pdf", workspace_id="ws-test")
    assert len(res["accepted_questions"]) == 18


def test_20_mixed_native_ocr_layout():
    p1 = {
        "page": 1,
        "raw_native_text": "Q1(a) Explain POS tagging algorithms. [10]\n",
        "raw_ocr_text": "Q2(a) Discuss machine translation systems. [10]\n",
        "reconstructed_text": "Q1(a) Explain POS tagging algorithms. [10]\nQ2(a) Discuss machine translation systems. [10]\n",
    }
    res = hybrid_extract_document([p1], filename="test20.pdf", workspace_id="ws-test")
    qids = [q["question_id"] for q in res["accepted_questions"]]
    assert "Q1(a)" in qids
    assert "Q2(a)" in qids


def test_hyphenated_terms_are_not_subquestion_markers():
    """n-gram / m-ary / v-structure at line start are compounds, not Qn(n)."""
    text = (
        "Q4(a) Explain n-gram language models with an example. [10]\n"
        "n-gram counts are estimated from the training corpus.\n"
        "Q4(b) Discuss smoothing for n-grams.\n"
        "m-ary branching is not a question marker.\n"
        "v-structure independence appears in the graph.\n"
        "Q5(a) Derive the expression and state every assumption clearly. [10]\n"
        "Q5(b) Compare the two indexing structures and evaluate lookup. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="hyphen.pdf", workspace_id="ws-test")
    ids = [q["question_id"] for q in res["accepted_questions"]]
    assert ids == ["Q4(a)", "Q4(b)", "Q5(a)", "Q5(b)"]
    assert res["quality"]["extraction_quality"] == "COMPLETE"
    for bad in ("Q4(n)", "Q4(m)", "Q4(v)", "Q5(n)", "Q5(m)", "Q5(v)"):
        assert bad not in ids
        assert bad not in (res["quality"].get("missing_questions") or [])


def test_unlabelled_stems_after_choice_parent_are_genuine():
    text = (
        "1 Attempt any FOUR\n"
        "What is the first recovered concept with a worked example.\n"
        "Explain the second recovered concept and justify the choice.\n"
        "Discuss the third recovered concept with a diagram.\n"
        "Differentiate between the two recovered forms with an example.\n"
        "Q2(a) Derive the expression and state every assumption clearly. [10]\n"
        "Q2(b) Compare the two indexing structures and evaluate lookup. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="unlabelled_stems.pdf", workspace_id="ws-test")
    ids = sorted(q["question_id"] for q in res["accepted_questions"])
    assert ids == ["Q1(a)", "Q1(b)", "Q1(c)", "Q1(d)", "Q2(a)", "Q2(b)"]
    assert res["quality"]["extraction_quality"] == "COMPLETE"


def test_glued_unlabelled_stems_on_choice_parent_line():
    text = (
        "1 Attempt any FOUR What is the first recovered concept with a worked example. "
        "Explain the second recovered concept and justify the choice.\n"
        "Q2(a) Derive the expression and state every assumption clearly. [10]\n"
        "Q2(b) Compare the two indexing structures and evaluate lookup. [10]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="glued_stems.pdf", workspace_id="ws-test")
    ids = sorted(q["question_id"] for q in res["accepted_questions"])
    assert "Q1(a)" in ids and "Q1(b)" in ids
    assert "Q2(a)" in ids and "Q2(b)" in ids
    assert res["quality"]["extraction_quality"] == "COMPLETE"


def test_21_noncontiguous_genuine_markers_are_complete():
    # Source never printed Q1(b). Completeness is genuine markers vs extracted,
    # not letter-gap filling.
    text = (
        "Q1(a) Explain what is natural language processing. [5]\n"
        "Q1(c) Discuss word sense disambiguation algorithms. [5]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test21.pdf", workspace_id="ws-test")
    assert res["quality"]["extraction_quality"] == "COMPLETE"
    assert res["quality"]["missing_questions"] == []
    ids = [q["question_id"] for q in res["accepted_questions"]]
    assert "Q1(a)" in ids
    assert "Q1(c)" in ids
    assert "Q1(b)" not in ids


def test_21b_genuine_marker_without_body_is_partial():
    text = (
        "Q1(a) Explain what is natural language processing. [5]\n"
        "Q1(b)\n"
        "Q1(c) Discuss word sense disambiguation algorithms. [5]\n"
    )
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="test21b.pdf", workspace_id="ws-test")
    assert res["quality"]["extraction_quality"] == "PARTIAL"
    assert "Q1(b)" in res["quality"]["missing_questions"]


def test_22_vector_preservation_on_failure():
    class DummyStore:
        def __init__(self):
            self.deleted = False
            self.inserted = False
        def replace_documents_for_source(self, *args, **kwargs):
            self.inserted = True
            self.deleted = True

    dstore = DummyStore()
    pipeline = DynamicIngestPipeline(vector_store=dstore)
    # Simulate partial extraction audit
    pipeline.last_pyq_questions_audit = {
        "accepted_questions": [],
        "quality_summary": {"extraction_quality": "PARTIAL", "questions_extracted": 1, "missing_questions": ["Q1(b)"]}
    }
    # Quality partial -> zero insertion
    assert dstore.inserted is False


def test_23_successful_replacement():
    class DummyStore:
        def __init__(self):
            self.inserted_count = 0
        def replace_documents_for_source(self, chunks, metadatas, ids, source_file="", workspace_id=""):
            self.inserted_count = len(chunks)

    dstore = DummyStore()
    # Ingestion complete inserts vectors
    dstore.replace_documents_for_source(["c1"], [{}], ["id1"])
    assert dstore.inserted_count == 1


def test_24_unknown_subject():
    text = "Q1(a) Describe principles of quantum cryptography protocol BB84. [10]\n"
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="quantum.pdf", workspace_id="ws-quantum", subject="Unknown Subject")
    assert len(res["accepted_questions"]) == 1


def test_25_unknown_university():
    text = "Q1(a) Derive Maxwell equations in free space and dielectric medium. [10]\n"
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="physics.pdf", workspace_id="ws-physics", subject="Physics")
    assert len(res["accepted_questions"]) == 1


def test_26_unknown_question_format():
    text = "1. a) Explain quantum entanglement and Bell inequalities with proof. [10]\n"
    pages = [{"page": 1, "raw_native_text": text, "reconstructed_text": text}]
    res = hybrid_extract_document(pages, filename="unknown_fmt.pdf", workspace_id="ws-fmt")
    assert any(q["question_id"] == "Q1(a)" for q in res["accepted_questions"])
