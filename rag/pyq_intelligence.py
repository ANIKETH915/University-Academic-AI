"""
PYQ Intelligence Engine — Question-level historical intelligence.

Derives exact repeats, semantic repeats, related topics, topic recurrence,
and evidence-based study priority ONLY from active-workspace PYQ records.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.vector_store import VectorStore
from rag.question_extractor import (
    normalize_question_text,
    compute_text_similarity,
    classify_repeat_relationship_full,
    analyze_single_paper_patterns,
    detect_suspicious_alphanumeric_noise,
    CanonicalConceptExtractor,
    build_question_representation,
    detect_question_type,
    extract_entities,
    extract_constraints,
    looks_like_ocr_garbage_topic,
    validate_question_candidate,
    is_valid_question_id,
)


def CURRENT_PRIORITY_BASELINE(appearances: int, recency_weight: float, avg_marks: float) -> float:
    return min(100.0, round((appearances * 20.0) + (recency_weight * 15.0) + (avg_marks * 2.0), 1))


from rag.config import current_academic_year


def calculate_deterministic_priority_score(
    appearances_count: int,
    distinct_years: int,
    exact_repeat_count: int,
    max_marks: int,
    last_year: int,
    current_year: int | None = None,
    semantic_repeat_count: int = 0,
    recurrence_consistency: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Evidence-sensitive priority (0-100). Scores intentionally diverge with evidence.
    """
    if current_year is None:
        current_year = current_academic_year()
    freq_s = min(28.0, appearances_count * 7.0)
    year_s = min(22.0, distinct_years * 8.0)
    exact_s = min(18.0, exact_repeat_count * 6.0)
    semantic_s = min(12.0, semantic_repeat_count * 4.0)
    marks_s = min(10.0, max_marks * 0.9)
    years_ago = max(0, current_year - last_year) if last_year else current_year
    recency_s = max(0.0, 10.0 - (years_ago * 3.5))
    consistency_s = min(10.0, round(recurrence_consistency * 10.0, 1))

    total = min(
        100.0,
        round(freq_s + year_s + exact_s + semantic_s + marks_s + recency_s + consistency_s, 1),
    )
    components = {
        "frequency_score": round(freq_s, 1),
        "year_recurrence_score": round(year_s, 1),
        "exact_repeat_score": round(exact_s, 1),
        "semantic_repeat_score": round(semantic_s, 1),
        "marks_score": round(marks_s, 1),
        "recency_score": round(recency_s, 1),
        "consistency_score": round(consistency_s, 1),
        "total_priority_score": total,
    }
    return total, components


class PYQIntelligenceEngine:
    def __init__(self, vector_store: Optional[VectorStore] = None, clustering_threshold: float = 0.65):
        self.store = vector_store or VectorStore()
        self.clustering_threshold = clustering_threshold

    def extract_temporal_features(self, items: List[Dict[str, Any]], syl_present: bool = True, target_year: Optional[int] = None) -> Dict[str, Any]:
        if target_year is None:
            target_year = current_academic_year()
        raw_years = [
            int(it.get("metadata", {}).get("year"))
            for it in items
            if str(it.get("metadata", {}).get("year", "")).isdigit()
        ]
        if not raw_years:
            raw_years = [target_year]
        valid_years = [y for y in raw_years if y < target_year]
        years = sorted(valid_years) if valid_years else sorted(raw_years)
        papers = list({it.get("metadata", {}).get("source_file", "paper.pdf") for it in items})
        marks_list = [
            int(it.get("metadata", {}).get("marks", 5))
            for it in items
            if str(it.get("metadata", {}).get("marks", "")).isdigit()
        ] or [5]
        last_y = max(years)
        return {
            "total_appearances": len(items),
            "number_of_distinct_papers": len(papers),
            "number_of_distinct_years": len(set(years)),
            "last_appearance_year": last_y,
            "years_since_last_appearance": max(0, target_year - last_y),
            "average_marks": round(sum(marks_list) / len(marks_list), 1),
            "maximum_marks": max(marks_list),
            "syllabus_presence": syl_present,
        }

    def normalize_topic_title(self, raw_title: str) -> str:
        if not raw_title or looks_like_ocr_garbage_topic(raw_title):
            return ""
        if detect_suspicious_alphanumeric_noise(raw_title):
            return ""
        concepts = CanonicalConceptExtractor.extract_canonical_concepts(raw_title)
        if not concepts:
            return ""
        return concepts[0]

    def _intent_bundle(self, q: Dict[str, Any]) -> Dict[str, Any]:
        text = q.get("exact_text", "")
        return {
            "question_type": q.get("question_type") or detect_question_type(text),
            "entities": q.get("entities") or extract_entities(text),
            "constraints": q.get("constraints") or extract_constraints(text),
            "question_intent": q.get("question_intent") or build_question_representation(q.get("question_id", "Q?"), text)["question_intent"],
        }

    def _source_ref(self, q: Dict[str, Any]) -> str:
        year = q.get("year", "?")
        session = q.get("exam_session") or "Exam"
        qid = q.get("question_number") or q.get("question_id")
        # Compact: "2024 May — Q6(a)"
        sess_short = str(session).split("/")[0].split()[0] if session else "Exam"
        return f"{year} {sess_short} — {qid}"

    def find_exact_repeat_groups(self, canonical_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        used: Set[int] = set()
        n = len(canonical_questions)

        for i in range(n):
            if i in used:
                continue
            q1 = canonical_questions[i]
            n1 = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text", ""))
            if not n1:
                continue
            members = [q1]
            for j in range(i + 1, n):
                if j in used:
                    continue
                q2 = canonical_questions[j]
                # Prefer cross-paper / cross-year evidence
                n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
                sim = 1.0 if n1 == n2 else compute_text_similarity(n1, n2)
                rel, concept, conf, reason = classify_repeat_relationship_full(
                    sim, n1, n2, q1.get("exact_text", ""), q2.get("exact_text", ""),
                    self._intent_bundle(q1), self._intent_bundle(q2),
                )
                if rel == "EXACT_REPEAT" and conf >= 0.85:
                    members.append(q2)
                    used.add(j)
            if len(members) > 1:
                used.add(i)
                years = sorted({q["year"] for q in members})
                groups.append(
                    {
                        "exact_text": q1["exact_text"],
                        "display_title": (q1.get("detected_topics") or [None])[0]
                        or (self._intent_bundle(q1).get("entities") or ["Repeated Question"])[0],
                        "question_ids": [q["question_id"] for q in members],
                        "source_refs": [self._source_ref(q) for q in members],
                        "years": years,
                        "distinct_years_count": len(years),
                        "repeat_count": len(members),
                        "confidence": 1.0,
                        "reason": "Normalized wording essentially identical",
                        "questions": members,
                    }
                )
        return groups

    def find_semantic_repeat_groups(
        self, canonical_questions: List[Dict[str, Any]], already_exact: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        STEP 2–5 recurrence: intent comparison + candidate similarity + verification.
        Prefers missing a weak match over a false positive.
        """
        already_exact = already_exact or set()
        groups: List[Dict[str, Any]] = []
        used: Set[int] = set()
        n = len(canonical_questions)

        for i in range(n):
            if i in used:
                continue
            q1 = canonical_questions[i]
            key1 = f"{q1.get('source_file')}:{q1.get('question_id')}"
            if key1 in already_exact:
                continue
            members = [q1]
            reasons = []
            for j in range(i + 1, n):
                if j in used:
                    continue
                q2 = canonical_questions[j]
                key2 = f"{q2.get('source_file')}:{q2.get('question_id')}"
                if key2 in already_exact:
                    continue
                n1 = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text", ""))
                n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
                if n1 == n2:
                    continue  # exact handled elsewhere
                sim = compute_text_similarity(n1, n2)
                if sim < 0.22:
                    continue  # candidate generation threshold
                rel, concept, conf, reason = classify_repeat_relationship_full(
                    sim, n1, n2, q1.get("exact_text", ""), q2.get("exact_text", ""),
                    self._intent_bundle(q1), self._intent_bundle(q2),
                )
                # Optional LLM pairwise judge — prefer false negative over false positive
                try:
                    from rag.hybrid_question_extraction import llm_judge_question_pair
                    from rag.llm_client import llm_configured

                    if llm_configured() and rel in ("SEMANTIC_REPEAT", "RELATED_TOPIC") and sim >= 0.28:
                        verdict = llm_judge_question_pair(q1.get("exact_text", ""), q2.get("exact_text", ""))
                        if verdict:
                            label = str(verdict.get("label") or "").upper()
                            if label == "DIFFERENT":
                                continue
                            if label == "RELATED_TOPIC":
                                rel = "RELATED_TOPIC"
                            if label == "SEMANTIC_REPEAT":
                                rel = "SEMANTIC_REPEAT"
                                reason = verdict.get("reason") or reason
                            if label == "EXACT_REPEAT":
                                rel = "SEMANTIC_REPEAT"
                                reason = verdict.get("reason") or reason
                except Exception:
                    pass
                if rel == "SEMANTIC_REPEAT" and conf >= 0.62:
                    members.append(q2)
                    used.add(j)
                    reasons.append(reason)
            if len(members) > 1:
                used.add(i)
                years = sorted({q["year"] for q in members})
                title = (q1.get("detected_topics") or [None])[0] or (self._intent_bundle(q1).get("entities") or ["Semantic Repeat"])[0]
                groups.append(
                    {
                        "display_title": title,
                        "original_questions": [
                            {"text": q["exact_text"], "source_ref": self._source_ref(q), "year": q["year"], "question_id": q["question_id"]}
                            for q in members
                        ],
                        "question_ids": [q["question_id"] for q in members],
                        "source_refs": [self._source_ref(q) for q in members],
                        "years": years,
                        "repeat_count": len(members),
                        "confidence": round(sum(1 for _ in members) and 0.7, 3),
                        "reason": reasons[0] if reasons else "Shared core question intent with paraphrased wording",
                        "why_same": reasons[0] if reasons else "Same question type, entities, and constraints despite wording differences",
                        "questions": members,
                    }
                )
        return groups

    def find_related_topic_pairs(
        self, canonical_questions: List[Dict[str, Any]], skip_keys: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        skip_keys = skip_keys or set()
        related: List[Dict[str, Any]] = []
        n = len(canonical_questions)
        seen_pairs: Set[Tuple[str, str]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                q1, q2 = canonical_questions[i], canonical_questions[j]
                k1 = f"{q1.get('source_file')}:{q1.get('question_id')}"
                k2 = f"{q2.get('source_file')}:{q2.get('question_id')}"
                if k1 in skip_keys and k2 in skip_keys:
                    continue
                n1 = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text", ""))
                n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
                sim = compute_text_similarity(n1, n2)
                if sim < 0.18:
                    continue
                rel, concept, conf, reason = classify_repeat_relationship_full(
                    sim, n1, n2, q1.get("exact_text", ""), q2.get("exact_text", ""),
                    self._intent_bundle(q1), self._intent_bundle(q2),
                )
                if rel != "RELATED_TOPIC":
                    continue
                pair_key = tuple(sorted([k1, k2]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                related.append(
                    {
                        "topic": concept,
                        "q1": {"text": q1["exact_text"], "source_ref": self._source_ref(q1)},
                        "q2": {"text": q2["exact_text"], "source_ref": self._source_ref(q2)},
                        "confidence": conf,
                        "reason": reason,
                        "is_repeat": False,
                    }
                )
        related.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return related[:40]

    def cluster_canonical_questions(self, canonical_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clusters_map: Dict[str, Dict[str, Any]] = {}

        for q_rec in canonical_questions:
            exact_text = q_rec.get("exact_text", "")
            detected_topics = q_rec.get("detected_topics", [])
            if isinstance(detected_topics, str):
                try:
                    detected_topics = json.loads(detected_topics)
                except Exception:
                    detected_topics = [detected_topics]

            concepts = [c for c in (detected_topics or []) if c and not looks_like_ocr_garbage_topic(str(c))]
            if not concepts:
                concepts = CanonicalConceptExtractor.extract_canonical_concepts(exact_text)
            if not concepts:
                continue  # do not invent "Core Academic Concept"

            for concept_name in concepts[:2]:
                if concept_name not in clusters_map:
                    clusters_map[concept_name] = {"rep_name": concept_name, "source_questions": [], "module_counts": {}}
                sqs = clusters_map[concept_name]["source_questions"]
                if not any(sq["question_id"] == q_rec["question_id"] and sq["source_file"] == q_rec["source_file"] for sq in sqs):
                    sqs.append(q_rec)
                    syl_map = q_rec.get("syllabus_mapping", {})
                    mod = syl_map.get("module", "Unmapped")
                    chap = syl_map.get("chapter", "Unmapped")
                    mod_unit_str = f"{mod}: {chap}"
                    clusters_map[concept_name]["module_counts"][mod_unit_str] = (
                        clusters_map[concept_name]["module_counts"].get(mod_unit_str, 0) + 1
                    )

        clusters_list = []
        for concept_name, cdata in clusters_map.items():
            sqs = cdata["source_questions"]
            if not sqs:
                continue
            mod_counts = cdata["module_counts"]
            primary_unit = max(mod_counts.items(), key=lambda x: x[1])[0] if mod_counts else "Unmapped: Unmapped"
            years = sorted({q["year"] for q in sqs})
            sessions = sorted({q.get("exam_session", "Exam") for q in sqs})
            marks = [q["marks"] for q in sqs if isinstance(q.get("marks"), (int, float))] or [5]
            clusters_list.append(
                {
                    "topic_name": concept_name,
                    "unit": primary_unit,
                    "source_questions": sqs,
                    "appearances_count": len(sqs),
                    "years": years,
                    "exam_sessions": sessions,
                    "marks_list": marks,
                    "average_marks": round(sum(marks) / len(marks), 1),
                    "min_marks": min(marks),
                    "max_marks": max(marks),
                    "sample_question": sqs[0]["exact_text"],
                }
            )
        return clusters_list

    def _load_valid_questions(self, workspace_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Load and re-validate PYQ records for workspace. Reject stale OCR garbage."""
        pyq_chunks = []
        try:
            res = self.store.collection.get(
                where={"$and": [{"workspace_id": {"$eq": workspace_id}}, {"doc_type": {"$eq": "pyq"}}]}
            )
            if res and res.get("documents"):
                for doc_text, meta in zip(res["documents"], res["metadatas"]):
                    pyq_chunks.append({"text": doc_text, "metadata": meta})
        except Exception as ex:
            print(f"[PYQ_INTELLIGENCE] Direct collection.get failed: {ex}")

        extracted_questions: List[Dict[str, Any]] = []
        paper_stats: Dict[str, Dict[str, Any]] = {}
        seen_qkeys: Set[str] = set()

        for c in pyq_chunks:
            meta = c.get("metadata", {})
            sf = meta.get("source_file", "paper.pdf")
            if sf not in paper_stats:
                paper_stats[sf] = {
                    "source_file": sf,
                    "exam_year": int(meta.get("year")) if str(meta.get("year", "")).isdigit() else None,
                    "exam_session": meta.get("exam_session", "Exam"),
                    "valid_questions": 0,
                    "rejected_questions": 0,
                    "exact_repeats": 0,
                    "semantic_repeats": 0,
                }

            q_text = meta.get("exact_text") or c.get("text", "")
            q_text = re.sub(r"^PYQ Question Item [^\n]+\n", "", q_text).strip()

            q_num = meta.get("question_number") or meta.get("question_id") or ""
            q_id = meta.get("question_id") or q_num

            # Reject invalid IDs / garbage at analytics time (stale vectors)
            if q_id and not is_valid_question_id(str(q_id)):
                paper_stats[sf]["rejected_questions"] += 1
                continue

            is_valid, reason, _metrics = validate_question_candidate(q_text)
            if not is_valid:
                paper_stats[sf]["rejected_questions"] += 1
                continue

            parent_q = meta.get("parent_question") or (q_num.split("(")[0] if "(" in str(q_num) else q_num)
            sub_q = meta.get("subquestion") or (q_num.split("(")[1].rstrip(")") if "(" in str(q_num) else None)
            if sub_q == "":
                sub_q = None

            if not sub_q and re.match(r"^Q\d+$", str(q_num), re.I):
                has_subs = any(
                    chk.get("metadata", {}).get("parent_question") == parent_q
                    and chk.get("metadata", {}).get("subquestion")
                    for chk in pyq_chunks
                )
                if has_subs:
                    continue

            q_key = f"{sf}_{q_id}"
            if q_key in seen_qkeys:
                continue
            seen_qkeys.add(q_key)

            detected_topics = meta.get("detected_topics", [])
            if isinstance(detected_topics, str):
                try:
                    detected_topics = json.loads(detected_topics)
                except Exception:
                    detected_topics = [detected_topics]
            detected_topics = [t for t in (detected_topics or []) if t and not looks_like_ocr_garbage_topic(str(t))]
            if not detected_topics:
                detected_topics = CanonicalConceptExtractor.extract_canonical_concepts(q_text)

            entities = meta.get("entities", [])
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except Exception:
                    entities = extract_entities(q_text)
            constraints = meta.get("constraints", [])
            if isinstance(constraints, str):
                try:
                    constraints = json.loads(constraints)
                except Exception:
                    constraints = extract_constraints(q_text)

            syl_mod = meta.get("syllabus_module") or "Unmapped"
            syl_chap = meta.get("syllabus_chapter") or "Unmapped"
            if syl_mod in ("General Principles",):
                syl_mod = "Unmapped"
            # Low-confidence mappings without syllabus evidence → Unmapped
            try:
                conf = float(meta.get("confidence") or 0)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0.4 and syl_mod not in ("Unmapped", ""):
                syl_mod = "Unmapped"
                syl_chap = "Unmapped"
            syl_top = meta.get("syllabus_topic") or (detected_topics[0] if detected_topics else "Unmapped")

            year = int(meta.get("year")) if str(meta.get("year", "")).isdigit() else 0
            session = meta.get("exam_session", "Exam Session")

            extracted_questions.append(
                {
                    "question_id": q_id,
                    "question_number": q_num or q_id,
                    "parent_question": parent_q,
                    "subquestion": sub_q,
                    "exact_text": q_text,
                    "normalized_text": meta.get("normalized_text") or normalize_question_text(q_text),
                    "detected_topics": detected_topics,
                    "question_intent": meta.get("question_intent") or build_question_representation(str(q_id), q_text)["question_intent"],
                    "question_type": meta.get("question_type") or detect_question_type(q_text),
                    "entities": entities or extract_entities(q_text),
                    "constraints": constraints or extract_constraints(q_text),
                    "syllabus_mapping": {"module": syl_mod, "chapter": syl_chap, "topic": syl_top},
                    "year": year,
                    "exam_session": session,
                    "marks": int(meta.get("marks", 5)) if str(meta.get("marks", "")).isdigit() else 5,
                    "source_file": sf,
                    "source_page": meta.get("source_page", 1),
                    "source_ref": f"{year} {str(session).split('/')[0].split()[0]} — {q_num or q_id}",
                }
            )
            paper_stats[sf]["valid_questions"] += 1
            paper_stats[sf]["exam_year"] = year
            paper_stats[sf]["exam_session"] = session

        return extracted_questions, paper_stats

    def _empty_response(self, workspace_id: str, subject: Optional[str], semester: Optional[str], notice: str = "") -> Dict[str, Any]:
        return {
            "available": False,
            "single_paper_mode": False,
            "total_papers": 0,
            "pyq_paper_count": 0,
            "workspace_id": workspace_id,
            "subject": subject or "Subject",
            "semester": semester or "Semester",
            "total_questions_analyzed": 0,
            "total_questions_extracted": 0,
            "total_valid_questions": 0,
            "questions_extracted": 0,
            "unique_topic_clusters": 0,
            "unique_question_intents": 0,
            "years_covered": [],
            "exact_repeat_count": 0,
            "semantic_repeat_count": 0,
            "topics": [],
            "most_repeated_questions": [],
            "exact_repeats": [],
            "semantic_repeats": [],
            "related_topics": [],
            "topic_recurrence": [],
            "study_priorities": [],
            "papers": [],
            "within_paper_patterns": [],
            "prediction_notice": notice or "No source PYQ questions available in this workspace.",
            "extracted_questions": [],
            "source_questions_available_via": "/workspaces/{workspace_id}/pyq-questions",
        }

    def get_pyq_analysis(
        self,
        workspace_id: str,
        subject: Optional[str] = None,
        semester: Optional[str] = None,
        include_source_questions: bool = False,
    ) -> Dict[str, Any]:
        if not workspace_id or not workspace_id.strip():
            return self._empty_response("", subject, semester)

        extracted_questions, paper_stats = self._load_valid_questions(workspace_id)
        if not extracted_questions:
            return self._empty_response(workspace_id, subject, semester)

        num_papers = len(paper_stats)
        single_paper_mode = num_papers == 1
        years_covered = sorted({q["year"] for q in extracted_questions if q.get("year")})

        exact_repeat_groups = self.find_exact_repeat_groups(extracted_questions)
        exact_keys: Set[str] = set()
        for g in exact_repeat_groups:
            for q in g.get("questions", []):
                exact_keys.add(f"{q.get('source_file')}:{q.get('question_id')}")

        semantic_repeat_groups = self.find_semantic_repeat_groups(extracted_questions, already_exact=exact_keys)
        semantic_keys: Set[str] = set()
        for g in semantic_repeat_groups:
            for q in g.get("questions", []):
                semantic_keys.add(f"{q.get('source_file')}:{q.get('question_id')}")

        related = self.find_related_topic_pairs(extracted_questions, skip_keys=exact_keys | semantic_keys)
        within_paper_patterns = analyze_single_paper_patterns(extracted_questions)
        clusters = self.cluster_canonical_questions(extracted_questions)

        # Paper-level repeat tallies
        for g in exact_repeat_groups:
            for q in g.get("questions", []):
                sf = q.get("source_file")
                if sf in paper_stats:
                    paper_stats[sf]["exact_repeats"] += 1
        for g in semantic_repeat_groups:
            for q in g.get("questions", []):
                sf = q.get("source_file")
                if sf in paper_stats:
                    paper_stats[sf]["semantic_repeats"] += 1

        # Most repeated questions (merge exact + semantic groups)
        most_repeated = []
        for g in exact_repeat_groups:
            most_repeated.append(
                {
                    "title": g.get("display_title") or "Repeated Question",
                    "asked_count": g["repeat_count"],
                    "years": g["years"],
                    "exact_repeats": g["repeat_count"],
                    "semantic_repeats": 0,
                    "sources": g.get("source_refs", []),
                    "kind": "exact",
                    "confidence": g.get("confidence", 1.0),
                    "sample_text": g.get("exact_text", ""),
                }
            )
        for g in semantic_repeat_groups:
            most_repeated.append(
                {
                    "title": g.get("display_title") or "Semantic Repeat",
                    "asked_count": g["repeat_count"],
                    "years": g["years"],
                    "exact_repeats": 0,
                    "semantic_repeats": g["repeat_count"],
                    "sources": g.get("source_refs", []),
                    "kind": "semantic",
                    "confidence": g.get("confidence", 0.7),
                    "sample_text": (g.get("original_questions") or [{}])[0].get("text", ""),
                    "why_same": g.get("why_same"),
                }
            )
        most_repeated.sort(key=lambda x: (x["asked_count"], len(x["years"])), reverse=True)

        # Topic recurrence (broader than question recurrence)
        topic_recurrence = []
        topics = []
        prediction_notice = (
            "Single paper analyzed. Within-paper patterns only."
            if single_paper_mode
            else f"Analysis based on recurring patterns across {num_papers} uploaded papers."
        )

        for clus in clusters:
            topic_name = clus["topic_name"]
            source_qs = clus["source_questions"]
            appearances = clus["appearances_count"]
            years = clus["years"]
            max_m = clus["max_marks"]
            avg_m = clus["average_marks"]
            last_y = max(years) if years else current_academic_year()

            exact_in = sum(1 for sq in source_qs if f"{sq.get('source_file')}:{sq.get('question_id')}" in exact_keys)
            semantic_in = sum(1 for sq in source_qs if f"{sq.get('source_file')}:{sq.get('question_id')}" in semantic_keys)
            consistency = (len(years) / max(1, num_papers)) if num_papers else 0.0

            priority_score, score_signals = calculate_deterministic_priority_score(
                appearances_count=appearances,
                distinct_years=len(years),
                exact_repeat_count=exact_in,
                max_marks=max_m,
                last_year=last_y,
                current_year=current_academic_year(),
                semantic_repeat_count=semantic_in,
                recurrence_consistency=consistency,
            )

            if single_paper_mode:
                pred_confidence, evidence_label, pred_score = "LOW", "Single paper analyzed (Within-paper pattern only)", None
            elif num_papers == 2:
                pred_confidence, evidence_label, pred_score = "LOW", "Recurring across 2 uploaded papers (Limited historical evidence)", priority_score
            elif num_papers in (3, 4):
                pred_confidence, evidence_label, pred_score = "MEDIUM", f"Recurring across {num_papers} uploaded papers (Moderate historical evidence)", priority_score
            else:
                pred_confidence, evidence_label, pred_score = "HIGH", f"Recurring across {num_papers} uploaded papers (Strong historical evidence)", priority_score

            explanations = [
                f"Evidence Level: {evidence_label} ({num_papers} paper(s) analyzed)",
                f"Appeared {appearances} time(s) across {len(years)} past exam session(s): {', '.join(str(y) for y in years)}",
                f"Syllabus Unit: {clus['unit']}",
                f"Marks Range: {clus['min_marks']}M – {max_m}M (Avg: {avg_m}M)",
            ]
            if exact_in >= 2:
                explanations.append(f"Contains {exact_in} exact/near-exact repeated question(s)")
            if semantic_in >= 2:
                explanations.append(f"Contains {semantic_in} semantic/paraphrased repeated question(s)")

            topic_recurrence.append(
                {
                    "topic": topic_name,
                    "years": years,
                    "appearances": appearances,
                    "note": "Topic recurrence is broader than question recurrence — related questions are not automatically repeats.",
                    "source_refs": [self._source_ref(sq) for sq in source_qs],
                }
            )

            topics.append(
                {
                    "topic_name": topic_name,
                    "subject": subject or "Academic Subject",
                    "unit": clus["unit"],
                    "appearances_count": appearances,
                    "years_appeared": years,
                    "exam_sessions": clus["exam_sessions"],
                    "marks_distribution": sorted(list(set(clus["marks_list"]))),
                    "average_marks": avg_m,
                    "min_marks": clus["min_marks"],
                    "max_marks": max_m,
                    "priority_score": priority_score,
                    "prediction_score": pred_score,
                    "prediction_confidence": pred_confidence,
                    "evidence_label": evidence_label,
                    "prediction_method": "deterministic_source_first_aggregation",
                    "signals": score_signals,
                    "explanation": explanations,
                    "exact_repeat_count": exact_in,
                    "semantic_repeat_count": semantic_in,
                    "sample_question": clus["sample_question"],
                    "source_questions": source_qs if include_source_questions else [
                        {
                            "question_id": sq["question_id"],
                            "year": sq["year"],
                            "source_ref": self._source_ref(sq),
                            "marks": sq["marks"],
                            "exact_text": sq.get("exact_text", ""),
                            "source_file": sq.get("source_file", ""),
                            "source_page": sq.get("source_page", 1),
                        }
                        for sq in source_qs
                    ],
                }
            )

        topics.sort(key=lambda x: (x.get("priority_score", 0.0), x.get("appearances_count", 0)), reverse=True)

        study_priorities = []
        for idx, t in enumerate(topics[:10]):
            study_priorities.append(
                {
                    "rank": idx + 1,
                    "topic_name": t["topic_name"],
                    "priority_score": t["priority_score"],
                    "signals": t["signals"],
                    "years": t["years_appeared"],
                    "appearances": t["appearances_count"],
                    "exact_repeat_count": t.get("exact_repeat_count", 0),
                    "semantic_repeat_count": t.get("semantic_repeat_count", 0),
                    "evidence_label": t["evidence_label"],
                    "explanation": t["explanation"],
                }
            )

        unique_intents = len({q.get("question_intent") for q in extracted_questions if q.get("question_intent")})

        result: Dict[str, Any] = {
            "available": True,
            "single_paper_mode": single_paper_mode,
            "total_papers": num_papers,
            "pyq_paper_count": num_papers,
            "workspace_id": workspace_id,
            "subject": subject or "Subject",
            "semester": semester or "Semester",
            "total_questions_analyzed": len(extracted_questions),
            "total_questions_extracted": len(extracted_questions),
            "total_valid_questions": len(extracted_questions),
            "questions_extracted": len(extracted_questions),
            "unique_topic_clusters": len(topics),
            "unique_question_intents": unique_intents,
            "years_covered": years_covered,
            "exact_repeat_count": len(exact_repeat_groups),
            "semantic_repeat_count": len(semantic_repeat_groups),
            "topics": topics,
            "most_repeated_questions": most_repeated,
            "exact_repeats": [
                {k: v for k, v in g.items() if k != "questions"}
                for g in exact_repeat_groups
            ],
            "semantic_repeats": [
                {k: v for k, v in g.items() if k != "questions"}
                for g in semantic_repeat_groups
            ],
            "related_topics": related,
            "topic_recurrence": topic_recurrence,
            "study_priorities": study_priorities,
            "papers": list(paper_stats.values()),
            "within_paper_patterns": within_paper_patterns,
            "prediction_notice": prediction_notice,
            "source_questions_available_via": f"/workspaces/{workspace_id}/pyq-questions",
            "debug_trace": {
                "workspace_id": workspace_id,
                "papers_analyzed": list(paper_stats.keys()),
                "total_canonical_questions": len(extracted_questions),
                "unique_topics": len(topics),
                "exact_repeat_groups_count": len(exact_repeat_groups),
                "semantic_repeat_groups_count": len(semantic_repeat_groups),
            },
        }

        # Main intelligence API must NOT dump all extracted questions
        if include_source_questions:
            result["extracted_questions"] = extracted_questions
        else:
            result["extracted_questions"] = []

        return result

    def get_source_questions(self, workspace_id: str) -> List[Dict[str, Any]]:
        qs, _ = self._load_valid_questions(workspace_id)
        return qs

    def get_study_priority(
        self, workspace_id: str, subject: Optional[str] = None, semester: Optional[str] = None, top_n: int = 5
    ) -> Dict[str, Any]:
        analysis = self.get_pyq_analysis(workspace_id=workspace_id, subject=subject, semester=semester)
        if not analysis["available"] or not analysis.get("study_priorities"):
            return {
                "workspace_id": workspace_id,
                "subject": subject or "Subject",
                "semester": semester or "Semester",
                "single_paper_mode": analysis.get("single_paper_mode", False),
                "top_high_priority_topics": [],
                "prediction_notice": analysis.get("prediction_notice", "No source PYQ questions available in this workspace."),
            }

        ranked = []
        for t in analysis["study_priorities"][:top_n]:
            full = next((x for x in analysis["topics"] if x["topic_name"] == t["topic_name"]), t)
            rec = full.get("evidence_label", "Historical evidence")
            if analysis.get("single_paper_mode"):
                recommendation = "Within-paper pattern analysis only. No multi-year prediction is claimed."
            else:
                recommendation = f"{rec}: Priority Score {full['priority_score']}/100."
            ranked.append(
                {
                    "rank": t["rank"],
                    "topic_name": full["topic_name"],
                    "subject": full.get("subject", subject or "Subject"),
                    "unit": full.get("unit", "Unmapped"),
                    "priority_score": full["priority_score"],
                    "prediction_score": full.get("prediction_score"),
                    "prediction_confidence": full.get("prediction_confidence"),
                    "evidence_label": rec,
                    "total_appearances": full.get("appearances_count", t.get("appearances")),
                    "recent_years": full.get("years_appeared", t.get("years")),
                    "marks_pattern": full.get("marks_distribution", []),
                    "average_marks": full.get("average_marks", 0),
                    "exact_repeat_count": full.get("exact_repeat_count", 0),
                    "semantic_repeat_count": full.get("semantic_repeat_count", 0),
                    "signals": full.get("signals", t.get("signals")),
                    "explanation": full.get("explanation", t.get("explanation")),
                    "recommendation": recommendation,
                    "source_questions": full.get("source_questions", []),
                }
            )

        return {
            "workspace_id": workspace_id,
            "subject": subject or "Subject",
            "semester": semester or "Semester",
            "single_paper_mode": analysis.get("single_paper_mode", False),
            "prediction_notice": analysis.get("prediction_notice"),
            "top_high_priority_topics": ranked,
        }

    def answer_analytics_query(self, question: str, workspace_id: str) -> Dict[str, Any]:
        analysis = self.get_pyq_analysis(workspace_id=workspace_id, include_source_questions=True)
        if not analysis["available"] or not analysis["topics"]:
            msg = (
                "Insufficient source evidence: No source PYQ question papers have been uploaded to this workspace yet.\n\n"
                "Please upload past examination PDF papers to unlock automated data-driven analysis."
            )
            return {"question": question, "answer": msg, "answer_mode": "insufficient_evidence", "citations": [], "topics": []}

        q_lower = question.lower()
        topics = analysis["topics"]
        extracted_qs = analysis.get("extracted_questions") or []
        num_papers = analysis["total_papers"]

        matched_topic = None
        for t in topics:
            if t["topic_name"].lower() in q_lower or any(
                kw.lower() in q_lower for kw in t["topic_name"].split() if len(kw) > 3
            ):
                matched_topic = t
                break

        if matched_topic and any(term in q_lower for term in ["how many times", "count", "frequency", "how often", "appear"]):
            sqs = matched_topic.get("source_questions") or []
            # May be compact refs — load full if needed
            if sqs and "exact_text" not in sqs[0]:
                full_qs = {f"{q['source_file']}:{q['question_id']}": q for q in extracted_qs}
                expanded = []
                for sq in sqs:
                    # best effort match by id+year
                    hit = next((q for q in extracted_qs if q["question_id"] == sq.get("question_id") and q["year"] == sq.get("year")), None)
                    expanded.append(hit or sq)
                sqs = expanded
            ev_lines = []
            for sq in sqs:
                text = sq.get("exact_text", "")
                ev_lines.append(
                    f"- **{sq.get('year', '?')} | {sq.get('question_id', '?')}** ({sq.get('marks', '?')}M): \"{text}\" [{sq.get('source_file', '')}]"
                )
            ans_text = (
                f"### Analysis for: **{matched_topic['topic_name']}**\n\n"
                f"**Total Appearances:** {matched_topic['appearances_count']} time(s) across {len(matched_topic['years_appeared'])} paper(s)\n"
                f"**Years Appeared:** {', '.join(str(y) for y in matched_topic['years_appeared'])}\n"
                f"**Syllabus Module:** {matched_topic['unit']}\n"
                f"**Marks Range:** {matched_topic['min_marks']}M – {matched_topic['max_marks']}M (Avg: {matched_topic['average_marks']}M)\n"
                f"**Priority Score:** {matched_topic['priority_score']}/100\n\n"
                f"#### Verified Source Evidence\n" + "\n".join(ev_lines)
            )
            return {
                "question": question,
                "answer": ans_text,
                "answer_mode": "structured_analytics",
                "citations": [{"source_file": sq.get("source_file"), "source_page": sq.get("source_page")} for sq in sqs],
                "topics": [matched_topic],
            }

        if any(term in q_lower for term in ["what should i study", "kya padhna hai", "study first", "priority", "important topic", "rank"]):
            top_ranked = topics[:5]
            rank_lines = []
            for idx, t in enumerate(top_ranked, 1):
                rank_lines.append(
                    f"### {idx}. {t['topic_name']} — Priority Score: {t['priority_score']}/100\n"
                    f"- **Appearances**: {t['appearances_count']} time(s) ({', '.join(str(y) for y in t['years_appeared'])})\n"
                    f"- **Module**: {t['unit']}\n"
                    f"- **Marks**: {t['marks_distribution']} (Avg: {t['average_marks']}M)\n"
                    f"- **Sample Question**: \"{t['sample_question']}\"\n"
                )
            ans_text = (
                f"### Priority Ranking ({num_papers} Paper(s) Analyzed)\n"
                f"*{analysis.get('prediction_notice', '')}*\n\n" + "\n\n".join(rank_lines)
            )
            return {"question": question, "answer": ans_text, "answer_mode": "structured_analytics", "citations": [], "topics": top_ranked}

        top_topics_summary = "\n".join(
            [
                f"- **{t['topic_name']}** ({t['appearances_count']} appearances, {t['unit']}, Priority: {t['priority_score']}/100)"
                for t in topics[:5]
            ]
        )
        ans_text = (
            f"### Workspace Analytics Summary ({num_papers} Paper(s), {len(extracted_qs)} Question Items)\n\n"
            f"#### Top Recurring Topics\n{top_topics_summary}\n\n"
            f"Navigate to **Study Priority** to see detailed question-level evidence and mark distribution!"
        )
        return {"question": question, "answer": ans_text, "answer_mode": "structured_analytics", "citations": [], "topics": topics[:5]}
