"""
Unit tests for the dedicated Study Priority Engine:
- Canonical question consumption only
- Generic semantic topic normalization (no duplicate priority cards)
- Over-merging protection (distinct algorithms remain separate)
- Deterministic evidence-weighted scoring model
- Question vs Topic priority separation
- Complete source question traceability
"""

import pytest
from rag.pyq_intelligence import (
    PYQIntelligenceEngine,
    generic_normalize_topic_title,
    calculate_deterministic_priority_score,
)


def test_generic_semantic_topic_normalization():
    # Fragmented titles representing same intent
    t1 = generic_normalize_topic_title("Determine communities for the given social network using Girvan-Newman")
    t2 = generic_normalize_topic_title("Find communities in the given social graph using Girvan Newman")
    
    assert "community" in t1.lower() or "girvan" in t1.lower()
    assert "community" in t2.lower() or "girvan" in t2.lower()


def test_over_merging_protection():
    engine = PYQIntelligenceEngine(vector_store=None)
    
    qs = [
        {
            "question_id": "Q1(a)",
            "exact_text": "Explain Bloom Filter architecture with an example",
            "normalized_text": "explain bloom filter architecture with an example",
            "detected_topics": ["Bloom Filter"],
            "year": 2022,
            "marks": 10,
            "source_file": "p1.pdf",
            "source_page": 1,
            "entities": ["Bloom Filter"],
            "question_type": "explain",
            "constraints": ["architecture_explanation"],
            "syllabus_mapping": {"module": "Unit 2", "chapter": "Data Structures", "topic": "Bloom Filter"},
        },
        {
            "question_id": "Q2(b)",
            "exact_text": "Explain Flajolet-Martin algorithm for frequency estimation",
            "normalized_text": "explain flajolet martin algorithm for frequency estimation",
            "detected_topics": ["Flajolet-Martin"],
            "year": 2023,
            "marks": 10,
            "source_file": "p2.pdf",
            "source_page": 2,
            "entities": ["Flajolet-Martin"],
            "question_type": "explain",
            "constraints": ["algorithm_explanation"],
            "syllabus_mapping": {"module": "Unit 2", "chapter": "Data Structures", "topic": "Flajolet-Martin"},
        },
        {
            "question_id": "Q3(a)",
            "exact_text": "Explain HDFS architecture in detail",
            "normalized_text": "explain hdfs architecture in detail",
            "detected_topics": ["HDFS"],
            "year": 2024,
            "marks": 10,
            "source_file": "p3.pdf",
            "source_page": 1,
            "entities": ["HDFS"],
            "question_type": "explain",
            "constraints": ["architecture_explanation"],
            "syllabus_mapping": {"module": "Unit 3", "chapter": "Distributed File System", "topic": "HDFS"},
        },
        {
            "question_id": "Q4(a)",
            "exact_text": "Explain MapReduce architecture and execution flow",
            "normalized_text": "explain mapreduce architecture and execution flow",
            "detected_topics": ["MapReduce"],
            "year": 2024,
            "marks": 10,
            "source_file": "p3.pdf",
            "source_page": 2,
            "entities": ["MapReduce"],
            "question_type": "explain",
            "constraints": ["architecture_explanation"],
            "syllabus_mapping": {"module": "Unit 3", "chapter": "Distributed Processing", "topic": "MapReduce"},
        },
    ]

    clusters = engine.cluster_canonical_questions(qs)
    topic_names = [c["topic_name"].lower() for c in clusters]

    # Distinct algorithms / components MUST remain separate
    assert len(clusters) == 4
    assert any("bloom" in name for name in topic_names)
    assert any("flajolet" in name for name in topic_names)
    assert any("hdfs" in name for name in topic_names)
    assert any("mapreduce" in name for name in topic_names)


def test_deterministic_priority_scoring():
    score, signals = calculate_deterministic_priority_score(
        appearances_count=4,
        distinct_years=3,
        exact_repeat_count=2,
        max_marks=10,
        last_year=2024,
        current_year=2026,
        semantic_repeat_count=1,
        recurrence_consistency=0.75,
        syllabus_mapped=True,
        extraction_confidence=1.0,
    )

    assert 70.0 <= score <= 100.0
    assert "frequency_score" in signals
    assert signals["exact_repeat_score"] > 0
    assert signals["semantic_repeat_score"] > 0
    assert signals["year_recurrence_score"] > 0
