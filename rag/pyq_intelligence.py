"""
PYQ Intelligence Engine — Question-level historical intelligence & Study Priority Engine.

Derives exact repeats, semantic repeats, related topics, topic recurrence,
and evidence-based study priority ONLY from canonical active-workspace PYQ records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from rag.config import current_academic_year
from rag.question_extractor import (
    CanonicalConceptExtractor,
    analyze_single_paper_patterns,
    build_question_representation,
    classify_repeat_relationship_full,
    compute_text_similarity,
    detect_question_type,
    detect_suspicious_alphanumeric_noise,
    extract_constraints,
    extract_entities,
    is_valid_question_id,
    looks_like_ocr_garbage_topic,
    normalize_question_text,
    topic_label_grounded_in_text,
    validate_question_candidate,
)
from rag.syllabus_index import (
    build_syllabus_index_from_workspace,
    map_question_to_syllabus_index,
)
from rag.semantic_similarity import embed_texts
from rag.source_identity import (
    attach_source_identity,
    dedupe_records_by_paper,
    dedupe_records_by_paper_and_question,
    merge_paper_stats,
    paper_id_of,
    source_dedup_summary,
    source_identity_fields,
    unique_occurrence_count,
    unique_session_identities,
    unique_years,
)
from rag.vector_store import VectorStore


def CURRENT_PRIORITY_BASELINE(appearances: int, recency_weight: float, avg_marks: float) -> float:
    return min(100.0, round((appearances * 20.0) + (recency_weight * 15.0) + (avg_marks * 2.0), 1))


def generic_normalize_topic_title(raw_title: str) -> str:
    """
    Generic semantic normalization of question/topic titles.
    Strips action verbs, filler phrases, duplicate repeated words, and OCR artifacts
    while preserving core academic entities.
    Subject-agnostic: no hardcoded topic catalog.
    """
    if not raw_title or looks_like_ocr_garbage_topic(raw_title):
        return ""
    if detect_suspicious_alphanumeric_noise(raw_title):
        return ""

    # Clean text: remove action verbs and common filler prefixes
    t = re.sub(
        r"^(?:Explain|Discuss|Describe|Write|State|Define|Determine|Find|Calculate|Compute|Show|Illustrate|Derive|Obtain|Consider|Give)\s+",
        "",
        raw_title.strip(),
        flags=re.I,
    ).strip()
    t = re.sub(
        r"^(?:a|an|the|for|in|on|with|using|about|types\s+of|concept\s+of|need\s+for|need\s+of|role\s+of|working\s+of)\s+",
        "",
        t,
        flags=re.I,
    ).strip()
    t = re.sub(
        r"\b(?:for\s+the\s+given|pseudo\s*code|pseudocode)\b",
        "",
        t,
        flags=re.I,
    ).strip(" -:,.()")

    # Remove duplicated words (e.g. "Retrieval Versus vs Retrieval" -> "Retrieval")
    words_raw = t.split()
    seen_words: List[str] = []
    for w in words_raw:
        w_clean = w.strip(" -:,.()").lower()
        if not w_clean:
            continue
        if w_clean in ("vs", "versus") and seen_words and seen_words[-1].lower() in ("vs", "versus"):
            continue
        if len(seen_words) > 0 and w_clean == seen_words[-1].lower() and len(w_clean) > 2:
            continue
        seen_words.append(w)
    t = " ".join(seen_words).strip(" -:,.()")

    t = re.sub(r"\s+", " ", t).strip()
    if not t or len(t) < 3:
        concepts = CanonicalConceptExtractor.extract_canonical_concepts(raw_title)
        return concepts[0] if concepts else ""

    t = re.sub(r"\bversus\b", "vs", t, flags=re.I)
    vs_parts = re.split(r"\s+vs\.?\s+", t, maxsplit=1, flags=re.I)
    if len(vs_parts) == 2:
        left, right = vs_parts[0].strip(), vs_parts[1].strip()
        if left and right:
            ll, rl = left.lower(), right.lower()
            if ll == rl or ll.endswith(rl) or rl.endswith(ll):
                t = left if len(left) >= len(right) else right
    t_clean = re.sub(
        r"\b(?:with\s+(?:an?\s+)?example|in\s+detail)\b",
        "",
        t,
        flags=re.I,
    ).strip(" -:,.()")

    if t_clean and len(t_clean) >= 3:
        words = t_clean.split()
        if len(words) <= 6 and not looks_like_ocr_garbage_topic(t_clean):
            return " ".join([w.capitalize() if not (w.isupper() or len(w) <= 3 or "-" in w) else w for w in words])

    words = t.split()
    stop = {"in", "the", "using", "for", "and", "or", "a", "an", "to", "of", "with"}
    kept = [w for w in words if w.lower() not in stop]
    if kept:
        phrase = " ".join(kept[:6])
        if phrase and not looks_like_ocr_garbage_topic(phrase):
            return " ".join(
                [w.capitalize() if not (w.isupper() or len(w) <= 3 or "-" in w) else w for w in phrase.split()]
            )

    concepts = CanonicalConceptExtractor.extract_canonical_concepts(raw_title)
    if concepts and not looks_like_ocr_garbage_topic(concepts[0]):
        return concepts[0]
    return ""


def calculate_deterministic_priority_score(
    appearances_count: int,
    distinct_years: int,
    exact_repeat_count: int,
    max_marks: int,
    last_year: int,
    current_year: int | None = None,
    semantic_repeat_count: int = 0,
    recurrence_consistency: float = 0.0,
    syllabus_mapped: bool = True,
    extraction_confidence: float = 1.0,
    related_topic_count: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """
    Evidence-weighted deterministic priority (0–100).

    Distinct years outweigh same-year repeats. No single metric can dominate.
    Syllabus mapping and extraction confidence are applied, not ignored.
    """
    if current_year is None:
        current_year = current_academic_year()

    freq_s = min(22.0, appearances_count * 5.5)
    year_s = min(26.0, distinct_years * 9.0)
    exact_s = min(16.0, exact_repeat_count * 5.5)
    semantic_s = min(10.0, semantic_repeat_count * 3.5)
    related_s = min(8.0, related_topic_count * 2.0)
    marks_s = min(8.0, max_marks * 0.7)
    years_ago = max(0, current_year - last_year) if last_year else current_year
    recency_s = max(0.0, 8.0 - (years_ago * 3.0))
    consistency_s = min(8.0, round(recurrence_consistency * 8.0, 1))
    syllabus_s = 6.0 if syllabus_mapped else 0.0
    try:
        conf = max(0.0, min(1.0, float(extraction_confidence)))
    except (TypeError, ValueError):
        conf = 0.5
    confidence_s = round(conf * 6.0, 1)

    total = min(
        100.0,
        round(
            freq_s + year_s + exact_s + semantic_s + related_s
            + marks_s + recency_s + consistency_s + syllabus_s + confidence_s,
            1,
        ),
    )
    components = {
        "frequency_score": round(freq_s, 1),
        "year_recurrence_score": round(year_s, 1),
        "exact_repeat_score": round(exact_s, 1),
        "semantic_repeat_score": round(semantic_s, 1),
        "related_topic_score": round(related_s, 1),
        "marks_score": round(marks_s, 1),
        "recency_score": round(recency_s, 1),
        "consistency_score": round(consistency_s, 1),
        "syllabus_score": round(syllabus_s, 1),
        "confidence_score": confidence_s,
        "total_priority_score": total,
    }
    return total, components


def quality_control_intelligence_payload(
    result: Dict[str, Any],
    workspace_id: str,
    source_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Suppress derived intelligence that cannot be traced to a workspace source
    question. Never suppress the original canonical question itself.
    """
    valid_keys = {
        f"{q.get('source_file')}:{q.get('question_id')}"
        for q in source_questions
        if q.get("question_id") and q.get("source_file")
    }
    valid_texts = {(q.get("exact_text") or "").strip() for q in source_questions if (q.get("exact_text") or "").strip()}

    def _group_ok(group: Dict[str, Any], kind: str) -> bool:
        refs = group.get("source_refs") or []
        qids = group.get("question_ids") or []
        known_ids = {k.split(":", 1)[-1] for k in valid_keys}
        grounded_ids = [qid for qid in qids if qid in known_ids]
        if len(qids) >= 2 and len(grounded_ids) < 2:
            return False
        if len(qids) < 2 and len(refs) < 2:
            return False
        if kind == "semantic":
            originals = group.get("original_questions") or []
            if len(originals) < 2:
                return False
            texts = [(oq.get("text") or "").strip() for oq in originals]
            if any(t and t not in valid_texts for t in texts if t):
                return False
        title = str(group.get("display_title") or group.get("topic") or "")
        if title and looks_like_ocr_garbage_topic(title) and kind != "exact":
            return False
        return True

    result["exact_repeats"] = [g for g in (result.get("exact_repeats") or []) if _group_ok(g, "exact")]
    result["semantic_repeats"] = [g for g in (result.get("semantic_repeats") or []) if _group_ok(g, "semantic")]
    cleaned_related = []
    for pair in result.get("related_topics") or []:
        members = pair.get("members") or []
        if members:
            texts = [(m.get("text") or "").strip() for m in members]
            live = [t for t in texts if t]
            if len(live) < 2:
                continue
            if any(t not in valid_texts for t in live):
                continue
        else:
            t1 = ((pair.get("q1") or {}).get("text") or "").strip()
            t2 = ((pair.get("q2") or {}).get("text") or "").strip()
            if not t1 or not t2 or t1 == t2:
                continue
            if t1 not in valid_texts or t2 not in valid_texts:
                continue
        if looks_like_ocr_garbage_topic(str(pair.get("topic") or "")):
            pair = {**pair, "topic": "Related concept"}
        cleaned_related.append(pair)
    result["related_topics"] = cleaned_related

    cleaned_topics = []
    for t in result.get("topics") or []:
        name = str(t.get("topic_name") or "")
        if looks_like_ocr_garbage_topic(name):
            continue
        sqs = t.get("source_questions") or []
        if not sqs:
            continue
        if any(
            f"{sq.get('source_file')}:{sq.get('question_id')}" not in valid_keys
            for sq in sqs
            if sq.get("question_id")
        ):
            continue
        if t.get("unit") in ("", None, "Unmapped"):
            t = {**t, "unit": "Syllabus mapping uncertain"}
        cleaned_topics.append(t)
    result["topics"] = cleaned_topics
    result["topic_priorities"] = [t for t in (result.get("topic_priorities") or []) if t.get("topic_name") in {x.get("topic_name") for x in cleaned_topics}]
    result["workspace_id"] = workspace_id
    result["exact_repeat_count"] = len(result.get("exact_repeats") or [])
    result["semantic_repeat_count"] = len(result.get("semantic_repeats") or [])
    return result


_ANALYSIS_CACHE: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()

def clear_pyq_analysis_cache(workspace_id: Optional[str] = None) -> None:
    with _CACHE_LOCK:
        if workspace_id:
            keys = [k for k in _ANALYSIS_CACHE if k[0] == workspace_id]
            for k in keys:
                _ANALYSIS_CACHE.pop(k, None)
        else:
            _ANALYSIS_CACHE.clear()

class PYQIntelligenceEngine:
    LLM_PAIR_JUDGE_BUDGET = 24
    ANALYSIS_CACHE_SIZE = 32

    def __init__(self, vector_store: Optional[VectorStore] = None, clustering_threshold: float = 0.65):
        self.store = vector_store or VectorStore()
        self.clustering_threshold = clustering_threshold
        self._analysis_cache = _ANALYSIS_CACHE
        self._cache_lock = _CACHE_LOCK

    def clear_cache(self, workspace_id: Optional[str] = None) -> None:
        clear_pyq_analysis_cache(workspace_id)

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
        paper_records = [
            {
                "source_file": it.get("metadata", {}).get("source_file", "paper.pdf"),
                "year": it.get("metadata", {}).get("year"),
                "exam_session": it.get("metadata", {}).get("exam_session"),
                "university": it.get("metadata", {}).get("university"),
                "subject": it.get("metadata", {}).get("subject"),
                "canonical_paper_id": it.get("metadata", {}).get("canonical_paper_id"),
            }
            for it in items
        ]
        attach_source_identity(paper_records)
        marks_list = [
            int(it.get("metadata", {}).get("marks", 5))
            for it in items
            if str(it.get("metadata", {}).get("marks", "")).isdigit()
        ] or [5]
        last_y = max(years)
        return {
            "total_appearances": unique_occurrence_count(paper_records),
            "number_of_distinct_papers": unique_occurrence_count(paper_records),
            "number_of_distinct_years": len(set(years)),
            "last_appearance_year": last_y,
            "years_since_last_appearance": max(0, target_year - last_y),
            "average_marks": round(sum(marks_list) / len(marks_list), 1),
            "maximum_marks": max(marks_list),
            "syllabus_presence": syl_present,
        }

    def normalize_topic_title(self, raw_title: str) -> str:
        return generic_normalize_topic_title(raw_title)

    def _intent_bundle(self, q: Dict[str, Any]) -> Dict[str, Any]:
        from rag.semantic_signature import build_semantic_signature

        text = q.get("exact_text", "")
        sig = build_semantic_signature(text)
        return {
            "question_type": q.get("question_type") or detect_question_type(text),
            "entities": q.get("entities") or extract_entities(text) or sig.entities,
            "constraints": q.get("constraints") or extract_constraints(text) or sig.constraints,
            "question_intent": q.get("question_intent") or build_question_representation(q.get("question_id", "Q?"), text)["question_intent"],
            "semantic_signature": sig,
        }

    def _source_ref(self, q: Dict[str, Any]) -> str:
        year = q.get("year", "?")
        session = q.get("exam_session") or "Exam"
        qid = q.get("question_number") or q.get("question_id")
        sess_str = str(session).strip()
        return f"{year} {sess_str} — {qid}"

    def _ensure_source_identity(
        self, questions: List[Dict[str, Any]], workspace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if questions and not questions[0].get("canonical_paper_id"):
            attach_source_identity(questions, workspace_id=workspace_id)
        return questions

    def _occurrence_members(self, members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return dedupe_records_by_paper(members)

    def _unique_source_refs(self, records: List[Dict[str, Any]]) -> List[str]:
        unique = self._occurrence_members(records)
        refs = [self._source_ref(q) for q in unique]
        if len(refs) == len(set(refs)):
            return refs
        out: List[str] = []
        counts = {r: refs.count(r) for r in refs}
        for q, ref in zip(unique, refs):
            if counts.get(ref, 0) <= 1:
                out.append(ref)
                continue
            stem = os.path.splitext(os.path.basename(str(q.get("source_file") or "paper")))[0]
            out.append(f"{ref} [{stem}]")
        return out

    def _original_question_payload(self, q: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "text": q.get("exact_text"),
            "source_ref": self._source_ref(q),
            "year": q.get("year"),
            "exam_session": q.get("exam_session"),
            "question_id": q.get("question_id"),
            "source_file": q.get("source_file"),
            "source_page": q.get("source_page"),
        }
        payload.update(source_identity_fields(q))
        return payload

    def _repeat_group_counts(self, members: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, List[int]]:
        unique = self._occurrence_members(members)
        years = unique_years(unique)
        return unique, len(unique), years

    def _bundles_for(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._intent_bundle(q) for q in questions]

    def _grounded_group_title(self, candidate: Any, source_text: str, fallback_label: str) -> str:
        title = str(candidate or "").strip()
        if not title or looks_like_ocr_garbage_topic(title):
            return fallback_label
        if not topic_label_grounded_in_text(title, source_text):
            return fallback_label
        return title

    def find_exact_repeat_groups(
        self,
        canonical_questions: List[Dict[str, Any]],
        bundles: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        used: Set[int] = set()
        self._ensure_source_identity(canonical_questions)
        n = len(canonical_questions)
        bundles = bundles if bundles is not None else self._bundles_for(canonical_questions)

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
                n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
                sim = 1.0 if n1 == n2 else compute_text_similarity(n1, n2)
                rel, concept, conf, reason = classify_repeat_relationship_full(
                    sim, n1, n2, q1.get("exact_text", ""), q2.get("exact_text", ""),
                    bundles[i], bundles[j],
                )
                if rel == "EXACT_REPEAT" and conf >= 0.85:
                    members.append(q2)
                    used.add(j)
            if len(members) > 1:
                unique_members, occurrence_count, years = self._repeat_group_counts(members)
                if occurrence_count < 2:
                    continue
                used.add(i)
                ident = source_identity_fields(unique_members[0])
                dup_ids: List[str] = []
                for q in unique_members:
                    for sid in q.get("duplicate_source_ids") or []:
                        if sid not in dup_ids:
                            dup_ids.append(sid)
                groups.append(
                    {
                        "exact_text": q1["exact_text"],
                        "display_title": self._grounded_group_title(
                            (q1.get("detected_topics") or [None])[0],
                            q1.get("exact_text", ""),
                            (bundles[i].get("entities") or ["Repeated Question"])[0],
                        ),
                        "group_type": "EXACT",
                        "question_ids": [q["question_id"] for q in unique_members],
                        "source_refs": self._unique_source_refs(unique_members),
                        "years": years,
                        "distinct_years_count": len(years),
                        "repeat_count": occurrence_count,
                        "unique_occurrence_count": occurrence_count,
                        "canonical_paper_id": ident.get("canonical_paper_id"),
                        "canonical_paper_ids": [paper_id_of(q) for q in unique_members],
                        "source_identity_confidence": ident.get("source_identity_confidence"),
                        "duplicate_source_ids": dup_ids,
                        "confidence": 1.0,
                        "similarity_method": "safe_normalized_equality",
                        "reason": "Normalized wording essentially identical",
                        "why_grouped": "Safe normalization produced identical wording",
                        "original_questions": [self._original_question_payload(q) for q in unique_members],
                        "questions": members,
                    }
                )
        return groups

    def find_semantic_repeat_groups(
        self,
        canonical_questions: List[Dict[str, Any]],
        already_exact: Optional[Set[str]] = None,
        bundles: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional["np.ndarray"] = None,
        llm_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from rag.semantic_signature import (
            candidate_pair_indices,
            compare_semantic_signatures,
            signature_from_bundle,
        )

        already_exact = already_exact or set()
        groups: List[Dict[str, Any]] = []
        used: Set[int] = set()
        self._ensure_source_identity(canonical_questions)
        n = len(canonical_questions)
        bundles = bundles if bundles is not None else self._bundles_for(canonical_questions)
        budget = self.LLM_PAIR_JUDGE_BUDGET if llm_budget is None else max(0, int(llm_budget))
        sem_floor = 0.62
        sigs = [
            signature_from_bundle(bundles[i], canonical_questions[i].get("exact_text", ""))
            for i in range(n)
        ]
        partner_map: Dict[int, List[int]] = defaultdict(list)
        for a, b in candidate_pair_indices(sigs, embeddings=embeddings):
            partner_map[a].append(b)
            partner_map[b].append(a)

        def sem_sim(a: int, b: int) -> Optional[float]:
            if embeddings is None:
                return None
            try:
                va, vb = embeddings[a], embeddings[b]
                denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                if denom == 0.0:
                    return None
                return float(np.dot(va, vb) / denom)
            except Exception:
                return None

        def classify_pair(i: int, j: int):
            q1, q2 = canonical_questions[i], canonical_questions[j]
            n1 = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text", ""))
            n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
            sim = compute_text_similarity(n1, n2)
            sem = sem_sim(i, j)
            rel, concept, conf, reason = classify_repeat_relationship_full(
                sim, n1, n2, q1.get("exact_text", ""), q2.get("exact_text", ""),
                bundles[i], bundles[j],
                semantic_similarity=sem,
            )
            match = compare_semantic_signatures(sigs[i], sigs[j], lexical_sim=sim, embedding_sim=sem)
            return rel, concept, conf, reason, sim, match

        for i in range(n):
            if i in used:
                continue
            q1 = canonical_questions[i]
            key1 = f"{q1.get('source_file')}:{q1.get('question_id')}"
            if key1 in already_exact:
                continue
            leader_norm = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text", ""))
            if not leader_norm:
                continue
            members = [q1]
            member_indices = [i]
            reasons = []
            confs: List[float] = []
            evidence_rows: List[Dict[str, Any]] = []
            partners = sorted({j for j in partner_map.get(i, []) if j > i})
            for j in partners:
                if j in used:
                    continue
                q2 = canonical_questions[j]
                key2 = f"{q2.get('source_file')}:{q2.get('question_id')}"
                if key2 in already_exact:
                    continue
                n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
                if not n2 or n2 == leader_norm:
                    continue
                rel, concept, conf, reason, sim, match = classify_pair(i, j)
                # Intra-group validation: every member vs every other member.
                if rel == "SEMANTIC_REPEAT" and conf >= sem_floor and len(member_indices) > 1:
                    for mi in member_indices[1:]:
                        rel_m, _, conf_m, _, _, _ = classify_pair(j, mi)
                        if rel_m != "SEMANTIC_REPEAT" or conf_m < sem_floor * 0.9:
                            rel = "RELATED_TOPIC"
                            break
                if rel == "SEMANTIC_REPEAT" and match.contradictions:
                    rel = "RELATED_TOPIC"

                if rel == "SEMANTIC_REPEAT" and conf >= sem_floor:
                    members.append(q2)
                    member_indices.append(j)
                    used.add(j)
                    reasons.append(reason)
                    confs.append(float(conf))
                    evidence_rows.append(match.evidence)
            if len(members) > 1:
                unique_members, occurrence_count, years = self._repeat_group_counts(members)
                if occurrence_count < 2:
                    continue
                used.add(i)
                joined_text = " ".join(q.get("exact_text", "") for q in unique_members)
                proto = str(sigs[i].core_entity or (bundles[i].get("entities") or ["Semantic Repeat"])[0])
                proto = re.sub(r"^(?:a|an|the|briefly|each of the)\s+", "", proto, flags=re.I).strip(" ,-")
                if proto:
                    proto = proto[:1].upper() + proto[1:]
                title = self._grounded_group_title(
                    proto,
                    joined_text,
                    (q1.get("detected_topics") or [proto])[0] or proto,
                )
                if looks_like_ocr_garbage_topic(str(title)) or not topic_label_grounded_in_text(str(title), joined_text):
                    fallback = (q1.get("detected_topics") or [None])[0] or proto
                    title = self._grounded_group_title(fallback, joined_text, "Paraphrased question")
                if looks_like_ocr_garbage_topic(str(title)):
                    title = "Paraphrased question"
                group_conf = round(sum(confs) / len(confs), 3) if confs else 0.62
                ident = source_identity_fields(unique_members[0])
                dup_ids: List[str] = []
                for q in unique_members:
                    for sid in q.get("duplicate_source_ids") or []:
                        if sid not in dup_ids:
                            dup_ids.append(sid)
                ev0 = evidence_rows[0] if evidence_rows else {}
                why = reasons[0] if reasons else "Same entity, intent, and requested output despite wording differences"
                groups.append(
                    {
                        "display_title": title,
                        "group_type": "SEMANTIC",
                        "original_questions": [self._original_question_payload(q) for q in unique_members],
                        "question_ids": [q["question_id"] for q in unique_members],
                        "source_refs": self._unique_source_refs(unique_members),
                        "years": years,
                        "repeat_count": occurrence_count,
                        "unique_occurrence_count": occurrence_count,
                        "canonical_paper_id": ident.get("canonical_paper_id"),
                        "canonical_paper_ids": [paper_id_of(q) for q in unique_members],
                        "source_identity_confidence": ident.get("source_identity_confidence"),
                        "duplicate_source_ids": dup_ids,
                        "confidence": group_conf,
                        "similarity_method": "semantic_signature",
                        "reason": why,
                        "why_same": why,
                        "why_grouped": why,
                        "semantic_evidence": {
                            "entity_match": ev0.get("entity_match"),
                            "intent_match": ev0.get("intent_match"),
                            "constraint_match": ev0.get("constraint_match"),
                            "output_match": ev0.get("output_match"),
                            "comparison_target_match": ev0.get("comparison_target_match"),
                            "embedding_support": ev0.get("embedding_support"),
                            "shared_specific_tokens": ev0.get("shared_specific_tokens") or [],
                            "core_entity": sigs[i].core_entity,
                            "intents_a": ev0.get("intents_a") or [],
                            "intents_b": ev0.get("intents_b") or [],
                        },
                        "questions": members,
                    }
                )
        return groups

    def find_related_topic_pairs(
        self,
        canonical_questions: List[Dict[str, Any]],
        skip_keys: Optional[Set[str]] = None,
        bundles: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional["np.ndarray"] = None,
    ) -> List[Dict[str, Any]]:
        from rag.semantic_signature import (
            candidate_pair_indices,
            related_canonical_key,
            signature_from_bundle,
            _is_specific,
        )

        skip_keys = skip_keys or set()
        self._ensure_source_identity(canonical_questions)
        n = len(canonical_questions)
        paper_qids: Set[str] = set()
        skip_texts: Set[str] = set()
        for q in canonical_questions:
            sf_key = f"{q.get('source_file')}:{q.get('question_id')}"
            if sf_key not in skip_keys:
                continue
            paper_qids.add(f"{paper_id_of(q)}:{q.get('question_id')}")
            nt = (q.get("normalized_text") or normalize_question_text(q.get("exact_text") or "")).strip()
            if nt:
                skip_texts.add(nt)
        if paper_qids or skip_texts:
            expanded = set(skip_keys)
            for q in canonical_questions:
                if f"{paper_id_of(q)}:{q.get('question_id')}" in paper_qids:
                    expanded.add(f"{q.get('source_file')}:{q.get('question_id')}")
                nt = (q.get("normalized_text") or normalize_question_text(q.get("exact_text") or "")).strip()
                if nt and nt in skip_texts:
                    expanded.add(f"{q.get('source_file')}:{q.get('question_id')}")
            skip_keys = expanded
        bundles = bundles if bundles is not None else self._bundles_for(canonical_questions)
        sigs = [
            signature_from_bundle(bundles[i], canonical_questions[i].get("exact_text", ""))
            for i in range(n)
        ]
        candidates = candidate_pair_indices(sigs, embeddings=embeddings)

        buckets: Dict[str, Dict[str, Any]] = {}
        for i, j in candidates:
            q1, q2 = canonical_questions[i], canonical_questions[j]
            k1 = f"{q1.get('source_file')}:{q1.get('question_id')}"
            k2 = f"{q2.get('source_file')}:{q2.get('question_id')}"
            if k1 in skip_keys and k2 in skip_keys:
                continue
            if paper_id_of(q1) == paper_id_of(q2):
                continue
            n1 = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text", ""))
            n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
            sim = compute_text_similarity(n1, n2)
            rel, concept, conf, reason = classify_repeat_relationship_full(
                sim, n1, n2, q1.get("exact_text", ""), q2.get("exact_text", ""),
                bundles[i], bundles[j],
            )
            if rel != "RELATED_TOPIC":
                continue
            shared = {t for t in (sigs[i].specific_tokens & sigs[j].specific_tokens) if _is_specific(t)}
            key = related_canonical_key(shared)
            if not key:
                continue
            keep = []
            for idx, kk in ((i, k1), (j, k2)):
                if kk not in skip_keys:
                    keep.append(idx)
            if len(keep) < 1:
                continue
            bucket = buckets.setdefault(
                key,
                {
                    "indices": set(),
                    "confs": [],
                    "reasons": [],
                    "concept": concept,
                    "shared": set(),
                    "sims": [],
                },
            )
            bucket["indices"].update(keep)
            bucket["confs"].append(float(conf or 0.0))
            bucket["reasons"].append(reason)
            bucket["shared"].update(shared)
            bucket["sims"].append(sim)
            if concept and not looks_like_ocr_garbage_topic(str(concept)):
                bucket["concept"] = concept

        related: List[Dict[str, Any]] = []
        for key, bucket in buckets.items():
            idxs = sorted(bucket["indices"])
            if len(idxs) < 2:
                continue
            members = []
            seen_member: Set[str] = set()
            for idx in idxs:
                q = canonical_questions[idx]
                mk = f"{paper_id_of(q)}:{q.get('question_id')}"
                if mk in seen_member:
                    continue
                seen_member.add(mk)
                payload = self._original_question_payload(q)
                payload.update(source_identity_fields(q))
                members.append(payload)
            if len(members) < 2:
                continue
            joined = " ".join((m.get("text") or "") for m in members)
            title = bucket.get("concept") or key
            if looks_like_ocr_garbage_topic(str(title)) or not topic_label_grounded_in_text(str(title), joined):
                title = key.replace("_", " ").title()
            conf = round(sum(bucket["confs"]) / len(bucket["confs"]), 3) if bucket["confs"] else 0.35
            sim = round(sum(bucket["sims"]) / len(bucket["sims"]), 3) if bucket["sims"] else 0.0
            reason = bucket["reasons"][0] if bucket["reasons"] else "Meaningful conceptual connection; not the same exam ask"
            related.append(
                {
                    "group_type": "RELATED",
                    "topic": title,
                    "members": members,
                    "q1": members[0],
                    "q2": members[1],
                    "confidence": conf,
                    "similarity": sim,
                    "similarity_method": "semantic_signature",
                    "reason": reason,
                    "why_grouped": reason,
                    "is_repeat": False,
                    "unique_occurrence_count": len(members),
                    "semantic_evidence": {
                        "shared_specific_tokens": sorted(bucket["shared"]),
                        "canonical_entity": key,
                    },
                }
            )
        related.sort(key=lambda x: (x.get("confidence", 0), x.get("unique_occurrence_count", 0)), reverse=True)
        return related[:12]

    def cluster_canonical_questions(
        self, canonical_questions: List[Dict[str, Any]], syllabus_index: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generic concept clustering with normalization & deduplication.
        Merges fragmented concepts sharing core intent (via token containment),
        while preserving strict over-merging boundaries for distinct algorithms/entities.
        """
        clusters_map: Dict[str, Dict[str, Any]] = {}
        self._ensure_source_identity(canonical_questions)

        for q_rec in canonical_questions:
            exact_text = q_rec.get("exact_text", "")
            detected_topics = q_rec.get("detected_topics", [])
            if isinstance(detected_topics, str):
                try:
                    detected_topics = json.loads(detected_topics)
                except Exception:
                    detected_topics = [detected_topics]

            norm_concepts = []
            for dt in (detected_topics or []):
                if dt and not looks_like_ocr_garbage_topic(str(dt)):
                    nt = generic_normalize_topic_title(str(dt))
                    if nt and nt not in norm_concepts:
                        norm_concepts.append(nt)

            if not norm_concepts:
                raw_c = CanonicalConceptExtractor.extract_canonical_concepts(exact_text)
                for rc in raw_c:
                    nt = generic_normalize_topic_title(rc)
                    if nt and nt not in norm_concepts:
                        norm_concepts.append(nt)

            if not norm_concepts:
                continue

            for concept_name in norm_concepts[:3]:
                if concept_name not in clusters_map:
                    clusters_map[concept_name] = {
                        "rep_name": concept_name,
                        "source_questions": [],
                        "module_counts": {},
                        "entities": extract_entities(exact_text),
                    }
                sqs = clusters_map[concept_name]["source_questions"]
                if not any(sq["question_id"] == q_rec["question_id"] and sq["source_file"] == q_rec["source_file"] for sq in sqs):
                    sqs.append(q_rec)
                    syl_map = q_rec.get("syllabus_mapping", {})
                    mod = syl_map.get("module", "Unmapped")
                    chap = syl_map.get("chapter", "Unmapped")
                    if mod == "Unmapped" and syllabus_index:
                        d_map, conf = map_question_to_syllabus_index(exact_text, detected_topics, syllabus_index)
                        if d_map and d_map.get("module") != "Unmapped" and conf >= 0.4:
                            mod = d_map["module"]
                            chap = d_map["chapter"]
                    mod_unit_str = f"{mod}: {chap}" if mod != "Unmapped" else "Unmapped"
                    clusters_map[concept_name]["module_counts"][mod_unit_str] = (
                        clusters_map[concept_name]["module_counts"].get(mod_unit_str, 0) + 1
                    )

        # Merge clusters sharing core intent (smart deduplication with token containment)
        concept_names = list(clusters_map.keys())
        merged_into: Dict[str, str] = {}
        stop_tokens = {"need", "given", "find", "show", "using", "types", "concept", "approach", "versus", "vs", "explain", "describe"}

        for i in range(len(concept_names)):
            c1 = concept_names[i]
            if c1 in merged_into:
                continue
            for j in range(i + 1, len(concept_names)):
                c2 = concept_names[j]
                if c2 in merged_into:
                    continue

                # Token containment check
                t1 = set(re.findall(r"[a-z0-9]{3,}", c1.lower())) - stop_tokens
                t2 = set(re.findall(r"[a-z0-9]{3,}", c2.lower())) - stop_tokens
                for e in clusters_map[c1]["entities"]:
                    t1.update(set(re.findall(r"[a-z0-9]{3,}", e.lower())) - stop_tokens)
                for e in clusters_map[c2]["entities"]:
                    t2.update(set(re.findall(r"[a-z0-9]{3,}", e.lower())) - stop_tokens)

                if not t1 or not t2:
                    continue

                common_t = t1 & t2
                min_len = min(len(t1), len(t2))
                containment = len(common_t) / min_len if min_len > 0 else 0.0

                # Check entity divergence: distinct algorithms/components MUST NOT be merged
                e1 = {x.lower() for x in clusters_map[c1]["entities"]}
                e2 = {x.lower() for x in clusters_map[c2]["entities"]}
                diff_e1 = e1 - e2
                diff_e2 = e2 - e1
                if diff_e1 and diff_e2 and not (e1 & e2) and compute_text_similarity(c1, c2) < 0.8:
                    if containment < 0.8:
                        continue

                sim = compute_text_similarity(c1, c2)
                should_merge = False
                if sim >= 0.78 or containment >= 0.80:
                    should_merge = True

                if should_merge:
                    # Prefer the more specific / descriptive concept name
                    target = c1 if len(c1) >= len(c2) else c2
                    source = c2 if target == c1 else c1
                    merged_into[source] = target
                    for sq in clusters_map[source]["source_questions"]:
                        if not any(x["question_id"] == sq["question_id"] and x["source_file"] == sq["source_file"] for x in clusters_map[target]["source_questions"]):
                            clusters_map[target]["source_questions"].append(sq)
                    for mod_k, count in clusters_map[source]["module_counts"].items():
                        clusters_map[target]["module_counts"][mod_k] = clusters_map[target]["module_counts"].get(mod_k, 0) + count

        clusters_list = []
        for concept_name, cdata in clusters_map.items():
            if concept_name in merged_into:
                continue
            sqs = cdata["source_questions"]
            if not sqs:
                continue
            mod_counts = cdata["module_counts"]
            valid_mods = {k: v for k, v in mod_counts.items() if k != "Unmapped"}
            primary_unit = max(valid_mods.items(), key=lambda x: x[1])[0] if valid_mods else "Unmapped"
            unique_sqs = self._occurrence_members(sqs)
            years = unique_years(unique_sqs)
            sessions = sorted({str(q.get("exam_session") or "Exam") for q in unique_sqs})
            marks = [q["marks"] for q in unique_sqs if isinstance(q.get("marks"), (int, float))] or [5]
            ident = source_identity_fields(unique_sqs[0]) if unique_sqs else {}
            clusters_list.append(
                {
                    "topic_name": concept_name,
                    "unit": primary_unit,
                    "source_questions": sqs,
                    "appearances_count": unique_occurrence_count(sqs),
                    "unique_occurrence_count": unique_occurrence_count(sqs),
                    "canonical_paper_ids": [paper_id_of(q) for q in unique_sqs],
                    "source_identity_confidence": ident.get("source_identity_confidence"),
                    "duplicate_source_ids": [
                        sid
                        for q in unique_sqs
                        for sid in (q.get("duplicate_source_ids") or [])
                    ],
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
                    "extraction_quality": "COMPLETE",
                    "incomplete": False,
                }

            q_text = meta.get("exact_text") or ""
            q_text = re.sub(r"^PYQ Question Item [^\n]+\n", "", q_text).strip()
            if not q_text:
                continue

            eq = str(meta.get("extraction_quality") or meta.get("paper_extraction_quality") or "").strip()
            paper_stats[sf]["extraction_quality"] = eq or paper_stats[sf].get("extraction_quality") or "COMPLETE"
            if eq in ("FAILED",):
                paper_stats[sf]["rejected_questions"] += 1
                paper_stats[sf]["incomplete"] = True
                continue
            if eq in ("PARTIAL", "RECOVERED"):
                paper_stats[sf]["incomplete"] = True

            q_num = meta.get("question_number") or meta.get("question_id") or ""
            q_id = meta.get("question_id") or q_num
            if not q_id:
                continue

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
            detected_topics = [
                t for t in detected_topics
                if topic_label_grounded_in_text(str(t), q_text)
            ]
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
            try:
                conf = float(meta.get("confidence") or 0)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < 0.4 and syl_mod not in ("Unmapped", ""):
                syl_mod = "Unmapped"
                syl_chap = "Unmapped"
            if syl_mod in ("Unmapped", ""):
                syl_mod = "Unmapped"
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
                    "source_ref": f"{year} {str(session).strip()} — {q_num or q_id}",
                    "university": meta.get("university") or "",
                    "subject": meta.get("subject") or "",
                    "course_code": meta.get("course_code") or meta.get("paper_code") or "",
                    "document_id": meta.get("document_id") or f"doc-{sf}",
                    "source_path": meta.get("source_path") or meta.get("persisted_path") or "",
                    "source_bytes_hash": meta.get("source_bytes_hash") or meta.get("file_sha256") or "",
                    "confidence": conf,
                    "grounding_status": meta.get("grounding_status") or "grounded",
                    "extraction_method": meta.get("extraction_method") or "hybrid",
                    "parent_id": meta.get("parent_id") or parent_q,
                }
            )
            paper_stats[sf]["valid_questions"] += 1
            paper_stats[sf]["exam_year"] = year
            paper_stats[sf]["exam_session"] = session

        attach_source_identity(extracted_questions, workspace_id=workspace_id)
        paper_stats = merge_paper_stats(paper_stats, extracted_questions)
        return extracted_questions, paper_stats

    def _empty_response(self, workspace_id: str, subject: Optional[str], semester: Optional[str], notice: str = "") -> Dict[str, Any]:
        return {
            "available": False,
            "extraction_incomplete": True,
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
            "recommended_study_plan": [],
            "question_priorities": [],
            "topic_priorities": [],
            "summary_stats": {
                "high_priority_topics_count": 0,
                "repeated_questions_count": 0,
                "most_repeated_topic": "N/A",
                "years_covered": [],
            },
            "papers": [],
            "within_paper_patterns": [],
            "prediction_notice": notice or "Priority analysis unavailable — question extraction is incomplete.",
            "extracted_questions": [],
            "source_questions_available_via": f"/workspaces/{workspace_id}/pyq-questions",
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

        extracted_questions, _paper_stats = self._load_valid_questions(workspace_id)
        if not extracted_questions:
            return self._empty_response(workspace_id, subject, semester, "No source PYQ questions available in this workspace.")

        signature_payload = "\n".join(
            f"{q.get('source_file')}|{q.get('question_id')}|{hash(q.get('exact_text') or '')}|{q.get('marks')}|{q.get('year')}"
            for q in extracted_questions
        )
        corpus_signature = hashlib.sha1(signature_payload.encode("utf-8", "ignore")).hexdigest()
        cache_key = (workspace_id, subject or "", semester or "", bool(include_source_questions), corpus_signature)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._analysis_cache.get(cache_key)
            if cached and now - cached[0] < 300:
                return copy.deepcopy(cached[1])
            if len(self._analysis_cache) >= self.ANALYSIS_CACHE_SIZE:
                for stale_key in sorted(self._analysis_cache, key=lambda k: self._analysis_cache[k][0])[: len(self._analysis_cache) // 2]:
                    self._analysis_cache.pop(stale_key, None)

        result = self._compute_pyq_analysis(
            workspace_id, subject, semester, include_source_questions,
            extracted_questions=extracted_questions, paper_stats=_paper_stats,
        )
        with self._cache_lock:
            self._analysis_cache[cache_key] = (time.monotonic(), copy.deepcopy(result))
        return result

    def _compute_pyq_analysis(
        self,
        workspace_id: str,
        subject: Optional[str],
        semester: Optional[str],
        include_source_questions: bool,
        extracted_questions: List[Dict[str, Any]],
        paper_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._ensure_source_identity(extracted_questions, workspace_id=workspace_id)
        paper_stats = merge_paper_stats(paper_stats, extracted_questions) or paper_stats
        num_papers = unique_occurrence_count(extracted_questions) or len(paper_stats)
        single_paper_mode = num_papers == 1
        years_covered = unique_years(extracted_questions)

        syllabus_index = build_syllabus_index_from_workspace(self.store, workspace_id, subject=subject or "Subject")

        bundles = self._bundles_for(extracted_questions)
        embeddings = embed_texts([q.get("normalized_text") or q.get("exact_text") or "" for q in extracted_questions])

        exact_repeat_groups = self.find_exact_repeat_groups(extracted_questions, bundles=bundles)
        exact_keys: Set[str] = set()
        for g in exact_repeat_groups:
            for q in g.get("questions", []):
                exact_keys.add(f"{q.get('source_file')}:{q.get('question_id')}")

        semantic_repeat_groups = self.find_semantic_repeat_groups(
            extracted_questions,
            already_exact=exact_keys,
            bundles=bundles,
            embeddings=embeddings,
        )
        semantic_keys: Set[str] = set()
        for g in semantic_repeat_groups:
            for q in g.get("questions", []):
                semantic_keys.add(f"{q.get('source_file')}:{q.get('question_id')}")

        related = self.find_related_topic_pairs(
            extracted_questions,
            skip_keys=exact_keys | semantic_keys,
            bundles=bundles,
            embeddings=embeddings,
        )
        within_paper_patterns = analyze_single_paper_patterns(extracted_questions)
        clusters = self.cluster_canonical_questions(extracted_questions, syllabus_index=syllabus_index)

        corpus_slots = len(unique_session_identities(extracted_questions))
        consistency_denominator = max(1, corpus_slots or num_papers)

        # Paper-level repeat tallies: a paper counts once per group it participates in.
        for g in exact_repeat_groups:
            for pid in {paper_id_of(q) for q in g.get("questions", [])}:
                if pid in paper_stats:
                    paper_stats[pid]["exact_repeats"] += 1
        for g in semantic_repeat_groups:
            for pid in {paper_id_of(q) for q in g.get("questions", [])}:
                if pid in paper_stats:
                    paper_stats[pid]["semantic_repeats"] += 1

        # Most repeated questions (merge exact + semantic groups)
        most_repeated = []
        for g in exact_repeat_groups:
            occurrence = int(g.get("unique_occurrence_count") or g["repeat_count"])
            most_repeated.append(
                {
                    "title": g.get("display_title") or "Repeated Question",
                    "asked_count": occurrence,
                    "unique_occurrence_count": occurrence,
                    "years": g["years"],
                    "exact_repeats": occurrence,
                    "semantic_repeats": 0,
                    "sources": g.get("source_refs", []),
                    "kind": "exact",
                    "confidence": g.get("confidence", 1.0),
                    "sample_text": g.get("exact_text", ""),
                    "canonical_paper_id": g.get("canonical_paper_id"),
                    "canonical_paper_ids": g.get("canonical_paper_ids") or [],
                    "source_identity_confidence": g.get("source_identity_confidence"),
                    "duplicate_source_ids": g.get("duplicate_source_ids") or [],
                }
            )
        for g in semantic_repeat_groups:
            occurrence = int(g.get("unique_occurrence_count") or g["repeat_count"])
            most_repeated.append(
                {
                    "title": g.get("display_title") or "Semantic Repeat",
                    "asked_count": occurrence,
                    "unique_occurrence_count": occurrence,
                    "years": g["years"],
                    "exact_repeats": 0,
                    "semantic_repeats": occurrence,
                    "sources": g.get("source_refs", []),
                    "kind": "semantic",
                    "confidence": g.get("confidence", 0.7),
                    "sample_text": (g.get("original_questions") or [{}])[0].get("text", ""),
                    "why_same": g.get("why_same"),
                    "canonical_paper_id": g.get("canonical_paper_id"),
                    "canonical_paper_ids": g.get("canonical_paper_ids") or [],
                    "source_identity_confidence": g.get("source_identity_confidence"),
                    "duplicate_source_ids": g.get("duplicate_source_ids") or [],
                }
            )
        most_repeated.sort(key=lambda x: (x["asked_count"], len(x["years"])), reverse=True)

        incomplete_papers = [
            p for p in paper_stats.values()
            if p.get("incomplete") or p.get("extraction_quality") in ("PARTIAL", "FAILED", "RECOVERED")
        ]
        extraction_incomplete = bool(incomplete_papers)
        prediction_notice = (
            "Single paper analyzed. Within-paper pattern analysis only."
            if single_paper_mode
            else f"Analysis based on recurring historical patterns across {num_papers} uploaded PYQ papers."
        )
        if extraction_incomplete:
            prediction_notice = (
                "Priority analysis limited because question extraction is incomplete. "
                + prediction_notice
            )

        topic_recurrence = []
        topics = []
        topic_priorities = []

        for clus in clusters:
            topic_name = clus["topic_name"]
            source_qs = clus["source_questions"]
            appearances = clus["appearances_count"]
            years = clus["years"]
            max_m = clus["max_marks"]
            avg_m = clus["average_marks"]
            last_y = max(years) if years else current_academic_year()

            exact_in = unique_occurrence_count(
                [sq for sq in source_qs if f"{sq.get('source_file')}:{sq.get('question_id')}" in exact_keys]
            )
            semantic_in = unique_occurrence_count(
                [sq for sq in source_qs if f"{sq.get('source_file')}:{sq.get('question_id')}" in semantic_keys]
            )
            cluster_qids = {f"{sq.get('source_file')}:{sq.get('question_id')}" for sq in source_qs}
            related_pair_keys = set()
            for r in related:
                q1 = r.get("q1") or {}
                q2 = r.get("q2") or {}
                k1 = f"{q1.get('source_file')}:{q1.get('question_id')}"
                k2 = f"{q2.get('source_file')}:{q2.get('question_id')}"
                if k1 in cluster_qids or k2 in cluster_qids:
                    related_pair_keys.add(
                        tuple(sorted((paper_id_of(q1), paper_id_of(q2), str(r.get("topic") or ""))))
                    )
            related_in = len(related_pair_keys)
            consistency = len(years) / consistency_denominator
            syllabus_mapped = clus["unit"] not in ("Unmapped", "Syllabus mapping uncertain", "")
            confs = []
            for sq in source_qs:
                try:
                    confs.append(float(sq.get("confidence") or sq.get("quality_score") or 0.85))
                except (TypeError, ValueError):
                    confs.append(0.85)
            extraction_conf = sum(confs) / len(confs) if confs else 0.85

            priority_score, score_signals = calculate_deterministic_priority_score(
                appearances_count=appearances,
                distinct_years=len(years),
                exact_repeat_count=exact_in,
                max_marks=max_m,
                last_year=last_y,
                current_year=current_academic_year(),
                semantic_repeat_count=semantic_in,
                recurrence_consistency=consistency,
                syllabus_mapped=syllabus_mapped,
                extraction_confidence=extraction_conf,
                related_topic_count=related_in,
            )

            if single_paper_mode:
                pred_confidence, evidence_label, pred_score = "LOW", "Single paper analyzed (Within-paper pattern only)", None
            elif num_papers == 2:
                pred_confidence, evidence_label, pred_score = "LOW", "Recurring across 2 uploaded papers (Limited historical evidence)", priority_score
            elif num_papers in (3, 4):
                pred_confidence, evidence_label, pred_score = "MEDIUM", f"Recurring across {num_papers} uploaded papers (Moderate historical evidence)", priority_score
            else:
                pred_confidence, evidence_label, pred_score = "HIGH", f"Recurring across {num_papers} uploaded papers (Strong historical evidence)", priority_score

            # EVIDENCE-BASED TIER CLASSIFICATION
            if priority_score >= 60.0 or (len(years) >= 3 and appearances >= 2) or (appearances >= 3 and len(years) >= 2) or exact_in > 0:
                tier, badge = "HIGH", "🔴 HIGH PRIORITY"
            elif priority_score >= 40.0 or len(years) >= 2 or appearances >= 2:
                tier, badge = "MEDIUM", "🟠 MEDIUM PRIORITY"
            elif priority_score >= 20.0:
                tier, badge = "LOWER", "🟢 LOWER PRIORITY"
            else:
                tier, badge = "LOW_EVIDENCE", "⚪ LOW EVIDENCE"

            explanations = [
                f"Evidence Level: {evidence_label} ({num_papers} paper(s) analyzed)",
                f"Appeared {appearances} time(s) across {len(years)} past exam session(s): {', '.join(str(y) for y in years)}",
                f"Syllabus Unit: {clus['unit'] if clus['unit'] not in ('Unmapped',) else 'Syllabus mapping uncertain'}",
                f"Marks Range: {clus['min_marks']}M – {max_m}M (Avg: {avg_m}M)",
            ]
            if exact_in > 0:
                explanations.append(f"Contains {exact_in} exact repeated question(s)")
            if semantic_in > 0:
                explanations.append(f"Contains {semantic_in} semantic/paraphrased repeated question(s)")
            if last_y and (current_academic_year() - last_y <= 2):
                explanations.append(f"Appeared recently in examination ({last_y})")

            why_summary = (
                f"Appeared {appearances}x across {len(years)} years ({', '.join(str(y) for y in years)}); "
                f"Marks: {clus['min_marks']}M–{max_m}M; Unit: {clus['unit']}"
            )

            source_qs_trace = []
            for sq in dedupe_records_by_paper_and_question(source_qs):
                rel = (
                    "EXACT_REPEAT"
                    if f"{sq.get('source_file')}:{sq.get('question_id')}" in exact_keys
                    else (
                        "SEMANTIC_REPEAT"
                        if f"{sq.get('source_file')}:{sq.get('question_id')}" in semantic_keys
                        else "TOPIC_MEMBER"
                    )
                )
                row = {
                    "question_id": sq["question_id"],
                    "year": sq["year"],
                    "exam_session": sq.get("exam_session", "Exam"),
                    "source_ref": self._source_ref(sq),
                    "marks": sq["marks"],
                    "exact_text": sq.get("exact_text", ""),
                    "source_file": sq.get("source_file", ""),
                    "source_page": sq.get("source_page", 1),
                    "relationship": rel,
                }
                row.update(source_identity_fields(sq))
                source_qs_trace.append(row)

            if appearances >= 2:
                topic_recurrence.append(
                    {
                        "topic": topic_name,
                        "years": years,
                        "appearances": appearances,
                        "unique_occurrence_count": appearances,
                        "note": "Topic recurrence summarizes related questions across past papers.",
                        "source_refs": self._unique_source_refs(source_qs),
                        "canonical_paper_ids": clus.get("canonical_paper_ids") or [paper_id_of(sq) for sq in self._occurrence_members(source_qs)],
                        "source_identity_confidence": clus.get("source_identity_confidence"),
                        "duplicate_source_ids": clus.get("duplicate_source_ids") or [],
                    }
                )

            display_unit = clus["unit"] if clus["unit"] not in ("Unmapped", "") else "Syllabus mapping uncertain"
            topic_item = {
                "topic_name": topic_name,
                "subject": subject or "Academic Subject",
                "unit": display_unit,
                "tier": tier,
                "tier_badge": badge,
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
                "signals": score_signals,
                "explanation": explanations,
                "why": why_summary,
                "exact_repeat_count": exact_in,
                "semantic_repeat_count": semantic_in,
                "unique_occurrence_count": appearances,
                "canonical_paper_ids": clus.get("canonical_paper_ids") or [paper_id_of(sq) for sq in self._occurrence_members(source_qs)],
                "source_identity_confidence": clus.get("source_identity_confidence"),
                "duplicate_source_ids": clus.get("duplicate_source_ids") or [],
                "sample_question": clus["sample_question"],
                "source_questions": source_qs_trace,
            }
            topics.append(topic_item)
            topic_priorities.append(topic_item)

        topics.sort(key=lambda x: (x.get("priority_score", 0.0), x.get("appearances_count", 0)), reverse=True)
        topic_priorities.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)

        # Build High-Priority Repeated QUESTIONS
        question_priorities = []
        all_repeat_groups = exact_repeat_groups + semantic_repeat_groups
        for idx, g in enumerate(all_repeat_groups):
            members = g.get("questions", [])
            if not members:
                continue
            unique_members = self._occurrence_members(members)
            q1 = unique_members[0] if unique_members else members[0]
            years = g.get("years") or unique_years(unique_members)
            repeat_cnt = int(g.get("unique_occurrence_count") or g.get("repeat_count") or len(unique_members))
            exact_cnt = repeat_cnt if g in exact_repeat_groups else 0
            semantic_cnt = repeat_cnt if g in semantic_repeat_groups else 0
            marks_list = [q.get("marks", 5) for q in members]
            max_m = max(marks_list)
            avg_m = round(sum(marks_list) / len(marks_list), 1)
            last_y = max(years) if years else current_academic_year()

            syl_map = q1.get("syllabus_mapping", {})
            unit_str = syl_map.get("module", "Unmapped")

            q_score, q_signals = calculate_deterministic_priority_score(
                appearances_count=repeat_cnt,
                distinct_years=len(years),
                exact_repeat_count=exact_cnt,
                max_marks=max_m,
                last_year=last_y,
                current_year=current_academic_year(),
                semantic_repeat_count=semantic_cnt,
                recurrence_consistency=len(years) / consistency_denominator,
                syllabus_mapped=unit_str not in ("Unmapped", "Syllabus mapping uncertain", ""),
                extraction_confidence=0.9,
            )

            if q_score >= 60.0 or len(years) >= 2:
                tier, badge = "HIGH", "🔴 HIGH PRIORITY"
            elif q_score >= 40.0:
                tier, badge = "MEDIUM", "🟠 MEDIUM PRIORITY"
            else:
                tier, badge = "LOWER", "🟢 LOWER PRIORITY"

            why_q = f"Repeated {repeat_cnt}x across {len(years)} exam paper(s) ({', '.join(str(y) for y in years)}); Asked for {max_m}M"

            question_priorities.append(
                {
                    "rank": idx + 1,
                    "question_title": g.get("display_title") or q1.get("exact_text", "")[:80],
                    "sample_text": q1.get("exact_text", ""),
                    "priority_score": q_score,
                    "tier": tier,
                    "tier_badge": badge,
                    "exact_repeat_count": exact_cnt,
                    "semantic_repeat_count": semantic_cnt,
                    "years": years,
                    "typical_marks": max_m,
                    "average_marks": avg_m,
                    "syllabus_unit": unit_str if unit_str not in ("Unmapped", "") else "Syllabus mapping uncertain",
                    "why": why_q,
                    "signals": q_signals,
                    "explanation": [
                        f"Repeated question appearing in {len(years)} distinct paper(s): {', '.join(str(y) for y in years)}",
                        f"Repeat type: {'Exact match' if exact_cnt > 0 else 'Semantic paraphrase'}",
                        f"Typical Marks: {max_m}M",
                        f"Mapped Syllabus Unit: {unit_str}",
                    ],
                    "unique_occurrence_count": repeat_cnt,
                    "canonical_paper_ids": g.get("canonical_paper_ids") or [paper_id_of(q) for q in unique_members],
                    "source_identity_confidence": g.get("source_identity_confidence"),
                    "duplicate_source_ids": g.get("duplicate_source_ids") or [],
                    "source_questions": [
                        {
                            **{
                                "question_id": q["question_id"],
                                "year": q["year"],
                                "exam_session": q.get("exam_session", "Exam"),
                                "source_ref": self._source_ref(q),
                                "marks": q["marks"],
                                "exact_text": q.get("exact_text", ""),
                                "source_file": q.get("source_file", ""),
                                "source_page": q.get("source_page", 1),
                                "relationship": "EXACT_REPEAT" if exact_cnt > 0 else "SEMANTIC_REPEAT",
                            },
                            **source_identity_fields(q),
                        }
                        for q in unique_members
                    ],
                }
            )
        question_priorities.sort(key=lambda x: x["priority_score"], reverse=True)

        def _study_band(tier: str, rank: int) -> str:
            if tier == "HIGH" or rank <= 3:
                return "STUDY_FIRST"
            if tier == "MEDIUM" or rank <= 6:
                return "STUDY_NEXT"
            if tier == "LOWER":
                return "STUDY_AFTER"
            return "OPTIONAL"

        recommended_study_plan = []
        seen_titles: Set[str] = set()

        for t in topic_priorities:
            if len(recommended_study_plan) >= 12:
                break
            title = t["topic_name"]
            if title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            rank = len(recommended_study_plan) + 1
            band = _study_band(t["tier"], rank)
            recommended_study_plan.append(
                {
                    "rank": rank,
                    "type": "topic",
                    "title": title,
                    "priority_score": t["priority_score"],
                    "tier": t["tier"],
                    "tier_badge": t["tier_badge"],
                    "study_band": band,
                    "study_order_label": {
                        "STUDY_FIRST": "Study First",
                        "STUDY_NEXT": "Study Next",
                        "STUDY_AFTER": "Study After",
                        "OPTIONAL": "Optional / Lower Priority",
                    }.get(band, band),
                    "unit": t["unit"],
                    "why": t["why"],
                    "explanation": t["explanation"],
                    "signals": t.get("signals", {}),
                    "source_questions": t["source_questions"],
                    "sample_question": t["sample_question"],
                    "original_question": t["sample_question"],
                    "years": t.get("years_appeared") or [],
                    "exact_repeat_count": t.get("exact_repeat_count", 0),
                    "semantic_repeat_count": t.get("semantic_repeat_count", 0),
                    "unique_occurrence_count": t.get("unique_occurrence_count") or t.get("appearances_count"),
                    "canonical_paper_ids": t.get("canonical_paper_ids") or [],
                    "source_identity_confidence": t.get("source_identity_confidence"),
                    "duplicate_source_ids": t.get("duplicate_source_ids") or [],
                }
            )

        study_order_groups = {
            "study_first": [x for x in recommended_study_plan if x["study_band"] == "STUDY_FIRST"],
            "study_next": [x for x in recommended_study_plan if x["study_band"] == "STUDY_NEXT"],
            "study_after": [x for x in recommended_study_plan if x["study_band"] == "STUDY_AFTER"],
            "optional": [x for x in recommended_study_plan if x["study_band"] == "OPTIONAL"],
        }

        module_wise_priority: List[Dict[str, Any]] = []
        by_mod: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in topic_priorities:
            by_mod[t.get("unit") or "Syllabus mapping uncertain"].append(t)
        for mod, items in by_mod.items():
            top = max(items, key=lambda x: x.get("priority_score") or 0)
            module_wise_priority.append(
                {
                    "module": mod,
                    "priority": round(sum(x.get("priority_score") or 0 for x in items) / len(items), 1),
                    "tier": top.get("tier"),
                    "repeated_concepts": [x["topic_name"] for x in items[:8]],
                    "important_questions": [x.get("sample_question") for x in items[:4] if x.get("sample_question")],
                    "mapping_uncertain": mod == "Syllabus mapping uncertain",
                }
            )
        module_wise_priority.sort(key=lambda x: x["priority"], reverse=True)

        study_priorities = [
            {
                "rank": item["rank"],
                "topic_name": item["title"],
                "priority_score": item["priority_score"],
                "signals": item.get("signals", {}),
                "years": unique_years(item.get("source_questions", [])),
                "appearances": unique_occurrence_count(item.get("source_questions", [])),
                "unique_occurrence_count": unique_occurrence_count(item.get("source_questions", [])),
                "exact_repeat_count": unique_occurrence_count(
                    [sq for sq in item.get("source_questions", []) if sq.get("relationship") == "EXACT_REPEAT"]
                ),
                "semantic_repeat_count": unique_occurrence_count(
                    [sq for sq in item.get("source_questions", []) if sq.get("relationship") == "SEMANTIC_REPEAT"]
                ),
                "evidence_label": item.get("tier_badge", "Priority Item"),
                "explanation": item["explanation"],
            }
            for item in recommended_study_plan
        ]

        unique_intents = len({q.get("question_intent") for q in extracted_questions if q.get("question_intent")})

        high_priority_count = sum(1 for t in topics if t.get("tier") == "HIGH" or t.get("priority_score", 0) >= 60.0)
        if high_priority_count == 0 and topics:
            high_priority_count = max(1, sum(1 for t in topics if t.get("priority_score", 0) >= 40.0))

        most_repeated_topic_name = topics[0]["topic_name"] if topics else "N/A"

        result: Dict[str, Any] = {
            "available": True,
            "extraction_incomplete": extraction_incomplete,
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
            "source_deduplication": source_dedup_summary(extracted_questions),
            "summary_stats": {
                "high_priority_topics_count": high_priority_count,
                "repeated_questions_count": len(exact_repeat_groups) + len(semantic_repeat_groups),
                "most_repeated_topic": most_repeated_topic_name,
                "years_covered": years_covered,
                "papers_analyzed": num_papers,
                "questions_analyzed": len(extracted_questions),
                "unique_concepts": len(topics),
                "exact_repeats": len(exact_repeat_groups),
                "semantic_repeats": len(semantic_repeat_groups),
                "related_topics": len(related),
                "recurring_topics": len(topic_recurrence),
            },
            "study_order_groups": study_order_groups,
            "module_wise_priority": module_wise_priority,
            "recommended_study_plan": recommended_study_plan,
            "question_priorities": question_priorities,
            "topic_priorities": topic_priorities,
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
                "papers_analyzed": [
                    p.get("source_file") or pid for pid, p in paper_stats.items()
                ] if isinstance(paper_stats, dict) else list(paper_stats.keys()),
                "unique_canonical_papers": num_papers,
                "total_canonical_questions": len(extracted_questions),
                "unique_topics": len(topics),
                "exact_repeat_groups_count": len(exact_repeat_groups),
                "semantic_repeat_groups_count": len(semantic_repeat_groups),
            },
        }

        if include_source_questions:
            result["extracted_questions"] = extracted_questions
        else:
            result["extracted_questions"] = []

        return quality_control_intelligence_payload(result, workspace_id, extracted_questions)

    def get_source_questions(self, workspace_id: str) -> List[Dict[str, Any]]:
        qs, _ = self._load_valid_questions(workspace_id)
        return qs

    def get_study_priority(
        self, workspace_id: str, subject: Optional[str] = None, semester: Optional[str] = None, top_n: int = 5
    ) -> Dict[str, Any]:
        analysis = self.get_pyq_analysis(workspace_id=workspace_id, subject=subject, semester=semester)
        if not analysis["available"]:
            return {
                "workspace_id": workspace_id,
                "subject": subject or "Subject",
                "semester": semester or "Semester",
                "single_paper_mode": analysis.get("single_paper_mode", False),
                "extraction_incomplete": True,
                "prediction_notice": analysis.get("prediction_notice", "Priority analysis unavailable — question extraction is incomplete."),
                "top_high_priority_topics": [],
                "recommended_study_plan": [],
                "question_priorities": [],
                "topic_priorities": [],
                "summary_stats": {
                    "high_priority_topics_count": 0,
                    "repeated_questions_count": 0,
                    "most_repeated_topic": "N/A",
                    "years_covered": [],
                },
            }

        plan = analysis.get("recommended_study_plan", [])[:top_n]
        return {
            "workspace_id": workspace_id,
            "subject": subject or "Subject",
            "semester": semester or "Semester",
            "single_paper_mode": analysis.get("single_paper_mode", False),
            "extraction_incomplete": bool(analysis.get("extraction_incomplete")),
            "prediction_notice": analysis.get("prediction_notice"),
            "summary_stats": analysis.get("summary_stats", {}),
            "study_order_groups": analysis.get("study_order_groups", {}),
            "module_wise_priority": analysis.get("module_wise_priority", []),
            "related_topics": analysis.get("related_topics", []),
            "recommended_study_plan": plan,
            "question_priorities": analysis.get("question_priorities", []),
            "topic_priorities": analysis.get("topic_priorities", []),
            "top_high_priority_topics": [
                {
                    "rank": item["rank"],
                    "topic_name": item["title"],
                    "subject": subject or "Subject",
                    "unit": item["unit"],
                    "priority_score": item["priority_score"],
                    "tier": item["tier"],
                    "tier_badge": item["tier_badge"],
                    "why": item["why"],
                    "explanation": item["explanation"],
                    "source_questions": item["source_questions"],
                    "recommendation": f"Priority Score {item['priority_score']}/100. {item['why']}",
                }
                for item in plan
            ],
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

        if any(term in q_lower for term in ["repeated", "repeat", "recurring question", "what was repeated"]):
            exact = analysis.get("exact_repeats") or []
            semantic = analysis.get("semantic_repeats") or []
            lines = []
            for g in exact[:8]:
                lines.append(
                    f"- EXACT: \"{g.get('exact_text') or g.get('display_title')}\" "
                    f"({', '.join(str(y) for y in g.get('years') or [])}) — {(g.get('source_refs') or [])}"
                )
            for g in semantic[:8]:
                originals = g.get("original_questions") or []
                src = "; ".join(
                    f"{oq.get('source_ref')}: \"{oq.get('text')}\"" for oq in originals[:4]
                )
                lines.append(f"- SEMANTIC: {g.get('display_title')} — {src}")
            if not lines:
                ans = "No exact or semantic repeats met the conservative evidence threshold in this workspace."
            else:
                ans = (
                    f"### Repeated questions (workspace evidence only)\n\n"
                    + "\n".join(lines)
                    + "\n\nConfidence is historical, not a guaranteed exam prediction."
                )
            return {
                "question": question,
                "answer": ans,
                "answer_mode": "structured_analytics",
                "citations": [
                    {"source_file": (oq.get("source_file") if isinstance(oq, dict) else None)}
                    for g in (exact + semantic)
                    for oq in (g.get("original_questions") or [])
                ],
                "topics": topics[:5],
            }

        matched_topic = None
        best_hits = 0
        q_tokens = set(re.findall(r"[a-z0-9]{4,}", q_lower))
        for t in topics:
            name = t["topic_name"].lower()
            if name in q_lower:
                matched_topic = t
                break
            hits = len(q_tokens & set(re.findall(r"[a-z0-9]{4,}", name)))
            if hits > best_hits:
                best_hits = hits
                matched_topic = t
        if best_hits == 0 and matched_topic and matched_topic["topic_name"].lower() not in q_lower:
            matched_topic = None

        if matched_topic and any(term in q_lower for term in ["how many times", "count", "frequency", "how often", "appear"]):
            sqs = matched_topic.get("source_questions") or []
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
