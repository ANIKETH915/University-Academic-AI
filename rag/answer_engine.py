"""
RAG Answer Engine — Retrieval-Grounded Answer Synthesis
======================================================

Answers are constructed EXCLUSIVELY from retrieved vector chunks and
structured PYQ analytics.  No hardcoded subject knowledge is used.

LLM Mode (optional):
  If the environment variable GEMINI_API_KEY or OPENAI_API_KEY is set,
  the engine calls the respective LLM with the retrieved context as the
  prompt body.  If neither key is present the engine falls back to
  deterministic extraction-mode synthesis, which still produces a
  grounded answer — it just extracts and formats text directly from the
  retrieved evidence instead of generating free-form prose.

Answer modes returned in the response:
  "rag_llm"              – LLM called with retrieved context
  "retrieval_only"       – deterministic extraction from chunks
  "structured_analytics" – answered from PYQ frequency/recurrence DB
  "insufficient_evidence"– no usable evidence found; NOT_FOUND returned
"""

import os
import re
import json
from typing import Dict, Any, List, Optional

from rag.vector_store import VectorStore
from rag.pyq_intelligence import PYQIntelligenceEngine


# ---------------------------------------------------------------------------
# Generic noise removal (subject-agnostic)
# ---------------------------------------------------------------------------

def clean_chunk_text(text: str) -> str:
    """Remove generic PDF noise from a chunk: page numbers, URLs, blank lines."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Skip generic page-number lines
        if re.fullmatch(r'page\s+\d+\s*(of\s+\d+)?', s, re.IGNORECASE):
            continue
        # Skip bare URLs
        if re.fullmatch(r'https?://\S+', s):
            continue
        cleaned.append(s)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Optional LLM caller
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Deterministic grounded synthesis without external LLM dependencies."""
    return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    question: str,
    mode: str,
    workspace_meta: Dict[str, str],
    syllabus_context: str,
    pyq_context: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the LLM."""
    university = workspace_meta.get("university", "")
    subject = workspace_meta.get("subject", "")
    semester = workspace_meta.get("semester", "")

    system_prompt = (
        "You are an academic PYQ intelligence assistant.\n"
        "Answer ONLY using the provided source context and structured analytics.\n"
        "Do not invent facts, syllabus modules, question IDs, or exam years.\n"
        f"The active workspace is: University={university}, Subject={subject}, Semester={semester}.\n"
        "If evidence is insufficient, respond with: "
        "\"Insufficient source evidence to answer this question.\""
    )

    user_prompt = (
        f"STUDENT QUESTION: {question}\n"
        f"ANSWER MODE: {mode}\n\n"
        "==================== RETRIEVED ACADEMIC CONTEXT ====================\n\n"
        f"--- SYLLABUS CURRICULUM CONTEXT ---\n{syllabus_context or '(none)'}\n\n"
        f"--- PAST EXAM (PYQ) CONTEXT ---\n{pyq_context or '(none)'}\n\n"
        "=====================================================================\n\n"
        "INSTRUCTIONS:\n"
        "1. Answer the question directly using the retrieved context above.\n"
        "2. Preserve question IDs such as Q1(a), Q5(b), exam year/session.\n"
        "3. Do not invent syllabus modules outside the provided context.\n"
        "4. If evidence is insufficient, say so explicitly.\n"
        "5. Prefer exact source wording for question text.\n"
        "6. Cite sources as [Source: <filename>, Page <page>].\n"
        "Generate the grounded answer now:\n"
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Deterministic retrieval-only synthesizer
# ---------------------------------------------------------------------------

def _synthesize_from_chunks(
    question: str,
    mode: str,
    topic: str,
    intent: str,
    syllabus_chunks: List[Dict],
    pyq_chunks: List[Dict],
) -> str:
    """
    Build an answer purely from retrieved chunk text.
    No hardcoded subject knowledge is used.
    """
    parts = []

    q_lower = question.lower()

    # ── PYQ evidence block ──────────────────────────────────────────────────
    if pyq_chunks:
        parts.append(f"### PYQ Evidence for: *{topic}*\n")
        for c in pyq_chunks[:6]:
            meta = c.get("metadata", {})
            q_id = meta.get("question_id") or meta.get("question_number", "")
            year = meta.get("year", "")
            session = meta.get("exam_session", "")
            marks = meta.get("marks", "")
            src = meta.get("source_file", "")
            page = meta.get("source_page", "")
            exact = (meta.get("exact_text") or "").strip()
            if not exact:
                # fall back to chunk text, strip prefix line
                raw = c.get("text", "")
                exact = re.sub(r'^PYQ Question Item [^\n]+\n', '', raw).strip()

            label_parts = []
            if year:
                label_parts.append(str(year))
            if session and session != "Exam Session":
                label_parts.append(session)
            if q_id:
                label_parts.append(q_id)
            if marks:
                label_parts.append(f"{marks} Marks")
            label = " | ".join(label_parts) if label_parts else "PYQ"

            parts.append(f"**{label}**")
            if exact:
                parts.append(f"> {exact}")
            if src:
                parts.append(f"[Source: {src}, Page {page}]\n")

    # ── Syllabus evidence block ─────────────────────────────────────────────
    if syllabus_chunks:
        parts.append(f"\n### Syllabus Context for: *{topic}*\n")
        for c in syllabus_chunks[:3]:
            meta = c.get("metadata", {})
            unit = meta.get("unit", meta.get("block", ""))
            src = meta.get("source_file", "")
            page = meta.get("source_page", "")
            raw = clean_chunk_text(c.get("text", ""))
            # Strip the leading "Syllabus Document [...]" header line
            raw = re.sub(r'^Syllabus Document \[[^\]]+\]\n?', '', raw).strip()
            if raw:
                if unit:
                    parts.append(f"**{unit}**")
                parts.append(raw[:600])
                if src:
                    parts.append(f"[Source: {src}, Page {page}]\n")

    # ── No evidence ─────────────────────────────────────────────────────────
    if not parts:
        return (
            f"Insufficient source evidence to answer: *{question}*\n\n"
            "Please upload relevant PYQ or syllabus documents to this workspace."
        )

    header = f"### Academic Answer: {question}\n\n"
    return header + "\n".join(parts)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class GroundedAnswerEngine:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.store = vector_store or VectorStore()
        self.pyq_intel = PYQIntelligenceEngine(vector_store=self.store)

    # ── Query Understanding ─────────────────────────────────────────────────

    def understand_query(self, question: str, mode: str = "general") -> Dict[str, Any]:
        """
        Classify intent and extract topic keywords from the user's question.
        Subject-agnostic: works for any academic domain.
        """
        q_lower = question.lower().strip()

        # Ambiguous / very broad queries
        ambiguous_patterns = [
            r'explain (a|an|any|some) (key|important|major|general)?\s*topic',
            r'tell me (an|a|some) (important|key)?\s*topic',
            r'explain something( from my syllabus)?',
            r'give me a topic',
            r'important topic from my syllabus',
            r'anything from syllabus',
            r'suggest a topic',
            r'pick a topic',
        ]
        for pat in ambiguous_patterns:
            if re.search(pat, q_lower):
                intent_type = "study_guidance" if "study" in q_lower else "ambiguous_topic"
                return {
                    "intent": intent_type,
                    "topic": None,
                    "topic_keywords": [],
                    "topic_confidence": "low",
                    "answer_type": "clarification",
                    "requested_marks": mode,
                    "is_pyq_request": False,
                    "is_analytics_request": False,
                }

        # Intent flags
        is_pyq = any(t in q_lower for t in [
            "how many times", "asked in past", "previous papers", "appeared",
            "repeated", "recurring", "pyq recurrence", "show questions about",
            "what questions", "which questions",
        ])
        is_analytics = any(t in q_lower for t in [
            "how many times", "frequency", "count", "recurrence",
            "how often", "appeared", "repeated", "what should i study",
            "what to study", "kya padhna hai", "study first", "priority",
            "important topic", "rank",
        ])
        is_comparison = any(t in q_lower for t in [
            "compare", "differentiate", "difference", "vs", "versus",
        ])
        is_numerical = any(t in q_lower for t in [
            "calculate", "solve", "numerical", "compute", "derive",
        ])
        is_definition = any(t in q_lower for t in [
            "what is", "define", "meaning of", "what do you mean",
        ])
        is_explanation = any(t in q_lower for t in [
            "explain", "describe", "discuss", "salient features",
            "characteristics", "how does", "how do",
        ])

        if is_analytics:
            intent = "pyq_analytics"
            answer_type = "analytics_summary"
        elif is_pyq:
            intent = "pyq_retrieval"
            answer_type = "pyq_summary"
        elif is_comparison:
            intent = "comparison"
            answer_type = "comparison_table"
        elif is_numerical:
            intent = "numerical"
            answer_type = "step_by_step"
        elif is_definition and not is_explanation:
            intent = "definition"
            answer_type = "short_definition"
        else:
            intent = "concept"
            answer_type = "theory"

        # Topic extraction — generic stop-word removal
        stop_words = {
            "explain", "describe", "discuss", "what", "how", "which", "from",
            "syllabus", "question", "detail", "write", "note", "following",
            "outline", "salient", "features", "mean", "you", "the", "and",
            "its", "are", "was", "were", "did", "does", "with", "about",
            "that", "this", "for", "have", "has", "had", "can", "will",
            "show", "find", "get", "give", "list", "asked", "previous",
            "papers", "times", "often", "times", "many", "appeared",
        }
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9_\-]{2,}\b', q_lower)
        topic_words = [w for w in words if w.lower() not in stop_words]

        if not topic_words:
            return {
                "intent": "ambiguous_topic",
                "topic": None,
                "topic_keywords": [],
                "topic_confidence": "low",
                "answer_type": "clarification",
                "requested_marks": mode,
                "is_pyq_request": is_pyq,
                "is_analytics_request": is_analytics,
            }

        extracted_topic = " ".join(topic_words[:5]).title()

        return {
            "intent": intent,
            "topic": extracted_topic,
            "topic_keywords": topic_words,
            "topic_confidence": "high",
            "answer_type": answer_type,
            "requested_marks": mode,
            "is_pyq_request": is_pyq,
            "is_analytics_request": is_analytics,
            "comparison_required": is_comparison,
            "numerical_required": is_numerical,
            "definition_required": is_definition,
            "explanation_required": is_explanation,
        }

    # ── Reranker (subject-agnostic) ─────────────────────────────────────────

    def rerank_and_filter_chunks(
        self,
        query_topic: str,
        topic_keywords: List[str],
        candidate_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved chunks by combining vector similarity with keyword
        relevance.  Fully subject-agnostic — no subject-specific bonuses or
        penalties, no hardcoded filenames.

        Scoring:
          final_score = 0.55 × vector_similarity
                      + 0.30 × keyword_match_ratio
                      + 0.15 × exact_phrase_bonus
        """
        if not candidate_chunks:
            return []

        # Generic administrative noise patterns (not subject-specific)
        generic_admin_patterns = [
            r'assignment submission deadline',
            r'fee structure',
            r'admission notice',
            r'hall ticket',
            r'examination schedule notice',
            r'page \d+ of \d+',
            r'counselling schedule',
            r'regional centre',
            r'programme guide \d+',
        ]

        topic_lower = (query_topic or "").lower()
        kw_set = {k.lower() for k in topic_keywords}

        reranked = []
        for c in candidate_chunks:
            raw_text = c.get("text", "")
            text_lower = clean_chunk_text(raw_text).lower()
            meta = c.get("metadata", {})

            # Generic admin penalty
            admin_penalty = 0.0
            for pat in generic_admin_patterns:
                if re.search(pat, text_lower, re.IGNORECASE):
                    admin_penalty = 0.50
                    break

            # Keyword match ratio
            kw_hits = sum(1 for kw in kw_set if kw in text_lower)
            kw_ratio = kw_hits / max(1, len(kw_set))

            # Exact phrase bonus
            phrase_bonus = 0.15 if (topic_lower and topic_lower in text_lower) else 0.0

            # Metadata topic bonus — use stored syllabus_topic / detected_topics
            meta_topic = (
                meta.get("syllabus_topic", "") + " " +
                meta.get("detected_topics", "") + " " +
                meta.get("exact_text", "")
            ).lower()
            meta_bonus = 0.10 if any(kw in meta_topic for kw in kw_set) else 0.0

            vector_score = c.get("score", 0.0)
            final_score = (
                0.55 * vector_score
                + 0.30 * kw_ratio
                + 0.15 * phrase_bonus
                + meta_bonus
                - admin_penalty
            )

            chunk_copy = dict(c)
            chunk_copy["final_score"] = round(final_score, 4)
            chunk_copy["kw_hits"] = kw_hits
            chunk_copy["_rerank_debug"] = {
                "vector_score": round(vector_score, 4),
                "kw_ratio": round(kw_ratio, 4),
                "phrase_bonus": phrase_bonus,
                "meta_bonus": meta_bonus,
                "admin_penalty": admin_penalty,
            }

            # Include chunk only if it has a genuine match signal:
            #   - decent keyword hit (kw_hits > 0) with any positive score, OR
            #   - strong vector similarity (>= 0.60), OR
            #   - exact phrase match found
            # This prevents completely unrelated queries from returning noise hits.
            has_keyword_signal = kw_hits > 0
            has_strong_vector = vector_score >= 0.60
            has_phrase = phrase_bonus > 0

            if (has_keyword_signal or has_strong_vector or has_phrase) and final_score >= 0.05:
                reranked.append(chunk_copy)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    # ── PYQ frequency helper ────────────────────────────────────────────────

    def analyze_pyq_frequency(self, pyq_chunks: List[Dict], workspace_id: str = "") -> Dict[str, Any]:
        """Count appearances across distinct canonical source papers, not raw files."""
        if not pyq_chunks:
            return {"times_asked": 0, "years": [], "frequency_summary": "Not found in PYQs."}

        from rag.source_identity import (
            attach_source_identity,
            unique_occurrence_count,
            unique_years,
        )

        records = []
        for c in pyq_chunks:
            meta = c.get("metadata", {})
            records.append(
                {
                    "source_file": meta.get("source_file", ""),
                    "year": int(meta.get("year")) if str(meta.get("year", "")).isdigit() else 0,
                    "exam_session": meta.get("exam_session"),
                    "university": meta.get("university"),
                    "subject": meta.get("subject"),
                    "course_code": meta.get("course_code"),
                    "exact_text": meta.get("exact_text") or c.get("text", ""),
                    "normalized_text": meta.get("normalized_text") or "",
                    "canonical_paper_id": meta.get("canonical_paper_id"),
                    "source_bytes_hash": meta.get("source_bytes_hash") or meta.get("file_sha256"),
                }
            )
        attach_source_identity(records, workspace_id=workspace_id)
        times = unique_occurrence_count([r for r in records if r.get("source_file") or r.get("canonical_paper_id")])
        distinct_years = unique_years(records)
        if times == 0:
            return {"times_asked": 0, "years": [], "frequency_summary": "Not found in PYQs."}

        summary = (
            f"Asked {times} time(s) across {len(distinct_years)} year(s): "
            f"{', '.join(str(y) for y in distinct_years)}."
        )
        return {
            "times_asked": times,
            "years": distinct_years,
            "frequency_summary": summary,
        }

    # ── Main entry point ────────────────────────────────────────────────────

    def generate_grounded_answer(
        self,
        question: str,
        mode: str = "general",
        doc_type: str = "both",
        filters: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Main RAG answer generation entry point.

        Flow:
          1. Understand intent / extract topic
          2. Vector retrieval (workspace-scoped)
          3. Subject-agnostic reranking
          4. Answer synthesis from retrieved chunks
             a. If LLM key is set → call LLM with retrieved context
             b. Otherwise → deterministic extraction from chunk text
          5. Return structured response with citations + debug info
        """
        ws_id = (filters or {}).get("workspace_id", "")

        # ── 1. Query understanding ──────────────────────────────────────────
        intent_info = self.understand_query(question, mode=mode)

        # ── Ambiguous query gate ────────────────────────────────────────────
        if intent_info["intent"] in ["ambiguous_topic", "study_guidance"] or not intent_info.get("topic"):
            if intent_info["intent"] == "study_guidance":
                msg = (
                    "To see the most important and frequently asked topics from your "
                    "uploaded past question papers, navigate to the **Study Priority** section.\n\n"
                    "Or ask a specific question about a topic from your uploaded documents, for example:\n"
                    "• *Explain [topic] from my syllabus.*\n"
                    "• *What questions on [topic] appeared in previous papers?*\n"
                    "• *How many times did [topic] appear in past papers?*"
                )
            else:
                msg = (
                    "Please ask a specific question about a topic from your uploaded documents.\n\n"
                    "Examples:\n"
                    "• *Explain [topic] from my syllabus.*\n"
                    "• *What questions about [topic] appeared in previous papers?*\n"
                    "• *Which module contains [topic]?*\n\n"
                    "Or navigate to **Study Priority** to view top recurring examination topics."
                )
            return {
                "question": question,
                "topic": None,
                "intent": intent_info["intent"],
                "answer": msg,
                "answer_sections": [{"title": "Clarification Required", "content": msg}],
                "pyq_evidence": {"matched": False, "times_asked": 0, "years": [], "questions": []},
                "citations": [],
                "grounding_score": 0.0,
                "answer_mode": "clarification",
                "hallucination_guard_triggered": False,
                "clarification_requested": True,
                "debug": {"query_topic": None, "retrieved_chunks": [], "filters": filters},
            }

        topic = intent_info["topic"]
        topic_kw = intent_info.get("topic_keywords", [])

        # ── 1b. Structured Analytics Routing ────────────────────────────────
        if intent_info.get("intent") == "pyq_analytics" or intent_info.get("is_analytics_request"):
            analytics_res = self.pyq_intel.answer_analytics_query(question, ws_id)
            if analytics_res.get("answer"):
                return {
                    "question": question,
                    "topic": topic,
                    "intent": intent_info["intent"],
                    "answer": analytics_res["answer"],
                    "answer_sections": [{"title": "Data-Driven Analytics Evidence", "content": analytics_res["answer"]}],
                    "pyq_evidence": {
                        "matched": len(analytics_res.get("topics", [])) > 0,
                        "times_asked": analytics_res.get("topics", [{}])[0].get("appearances_count", 0) if analytics_res.get("topics") else 0,
                        "years": analytics_res.get("topics", [{}])[0].get("years_appeared", []) if analytics_res.get("topics") else [],
                        "questions": [sq["exact_text"] for sq in analytics_res.get("topics", [{}])[0].get("source_questions", [])[:4]] if analytics_res.get("topics") else []
                    },
                    "citations": analytics_res.get("citations", []),
                    "grounding_score": 1.0,
                    "top_score": 1.0,
                    "retrieved_chunks_count": len(analytics_res.get("topics", [])),
                    "answer_mode": analytics_res.get("answer_mode", "structured_analytics"),
                    "hallucination_guard_triggered": False,
                    "clarification_requested": False,
                    "debug": {
                        "query_topic": topic,
                        "intent": intent_info["intent"],
                        "answer_mode": analytics_res.get("answer_mode", "structured_analytics"),
                        "active_workspace": ws_id
                    }
                }

        print(f"[ANSWER_ENGINE] Query='{question}' | Topic='{topic}' | Intent='{intent_info['intent']}' | WS='{ws_id}'")

        # ── 2. Vector retrieval ─────────────────────────────────────────────
        candidate_chunks = self.store.search(
            query=question,
            doc_type=doc_type,
            top_k=15,
            filters=filters,
        )

        print(f"[ANSWER_ENGINE] Retrieved {len(candidate_chunks)} candidate chunks from vector DB.")

        # ── 3. Reranking ────────────────────────────────────────────────────
        reranked_chunks = self.rerank_and_filter_chunks(topic, topic_kw, candidate_chunks)

        print(f"[ANSWER_ENGINE] After reranking: {len(reranked_chunks)} chunks survive.")

        # ── 4. No-evidence gate ─────────────────────────────────────────────
        if not reranked_chunks:
            not_found_msg = (
                f"Insufficient source evidence to answer: *{question}*\n\n"
                "The requested topic was not found in the documents uploaded to this workspace. "
                "Please upload relevant PYQ or syllabus PDFs."
            )
            return {
                "question": question,
                "topic": topic,
                "intent": intent_info["intent"],
                "answer": not_found_msg,
                "answer_sections": [{"title": "Topic Not Found", "content": not_found_msg}],
                "pyq_evidence": {"matched": False, "times_asked": 0, "years": [], "questions": []},
                "citations": [],
                "grounding_score": candidate_chunks[0]["score"] if candidate_chunks else 0.0,
                "answer_mode": "insufficient_evidence",
                "hallucination_guard_triggered": True,
                "clarification_requested": False,
                "debug": {
                    "query_topic": topic,
                    "retrieved_count": len(candidate_chunks),
                    "reranked_count": 0,
                    "filters": filters,
                    "retrieved_chunks": [
                        {
                            "source": c.get("metadata", {}).get("source_file"),
                            "score": c.get("score"),
                            "question_id": c.get("metadata", {}).get("question_id"),
                        }
                        for c in candidate_chunks[:5]
                    ],
                },
            }

        # ── Split by doc type ───────────────────────────────────────────────
        syllabus_chunks = [c for c in reranked_chunks if c.get("metadata", {}).get("doc_type") == "syllabus"]
        pyq_chunks = [c for c in reranked_chunks if c.get("metadata", {}).get("doc_type") == "pyq"]

        print(f"[ANSWER_ENGINE] syllabus_chunks={len(syllabus_chunks)}, pyq_chunks={len(pyq_chunks)}")

        # ── 5. PYQ frequency (structured analytics) ────────────────────────
        pyq_freq = self.analyze_pyq_frequency(pyq_chunks, workspace_id=ws_id)

        pyq_evidence = {
            "matched": pyq_freq["times_asked"] > 0,
            "times_asked": pyq_freq["times_asked"],
            "years": pyq_freq["years"],
            "questions": [
                (
                    c.get("metadata", {}).get("exact_text") or
                    re.sub(r'^PYQ Question Item [^\n]+\n', '', c.get("text", "")).strip()
                )[:120]
                for c in pyq_chunks[:4]
            ],
        }

        # ── 6. Build context strings for LLM / extraction ──────────────────
        syllabus_context_lines = []
        for c in syllabus_chunks[:4]:
            meta = c.get("metadata", {})
            raw = re.sub(r'^Syllabus Document \[[^\]]+\]\n?', '', c.get("text", "")).strip()
            raw = clean_chunk_text(raw)[:500]
            unit = meta.get("unit", meta.get("block", ""))
            src = meta.get("source_file", "")
            page = meta.get("source_page", "")
            entry = (f"[{unit}] " if unit else "") + raw
            if src:
                entry += f"\n[Source: {src}, Page {page}]"
            syllabus_context_lines.append(entry)
        syllabus_context = "\n\n".join(syllabus_context_lines)

        pyq_context_lines = []
        for c in pyq_chunks[:6]:
            meta = c.get("metadata", {})
            q_id = meta.get("question_id") or meta.get("question_number", "")
            year = meta.get("year", "")
            session = meta.get("exam_session", "")
            marks = meta.get("marks", "")
            src = meta.get("source_file", "")
            page = meta.get("source_page", "")
            exact = (meta.get("exact_text") or "").strip()
            if not exact:
                exact = re.sub(r'^PYQ Question Item [^\n]+\n', '', c.get("text", "")).strip()

            parts = []
            if year:
                parts.append(str(year))
            if session and session != "Exam Session":
                parts.append(session)
            if q_id:
                parts.append(q_id)
            if marks:
                parts.append(f"{marks}M")
            label = " | ".join(parts) if parts else "PYQ"
            entry = f"[{label}] {exact}"
            if src:
                entry += f"\n[Source: {src}, Page {page}]"
            pyq_context_lines.append(entry)
        pyq_context = "\n\n".join(pyq_context_lines)

        # ── 7. Infer workspace metadata from retrieved chunks ───────────────
        workspace_meta: Dict[str, str] = {}
        for c in reranked_chunks[:3]:
            meta = c.get("metadata", {})
            if not workspace_meta.get("university"):
                workspace_meta["university"] = meta.get("university", "")
            if not workspace_meta.get("subject"):
                workspace_meta["subject"] = meta.get("subject", "")
            if not workspace_meta.get("semester"):
                workspace_meta["semester"] = meta.get("semester", "")

        # ── 8. Answer synthesis ─────────────────────────────────────────────
        answer_mode: str
        ans_text: str

        sys_p, user_p = _build_llm_prompt(
            question, mode, workspace_meta, syllabus_context, pyq_context
        )
        llm_answer = _call_llm(sys_p, user_p)

        if llm_answer:
            ans_text = llm_answer.strip()
            answer_mode = "rag_llm"
            print(f"[ANSWER_ENGINE] Answer generated via LLM ({answer_mode}).")
        else:
            ans_text = _synthesize_from_chunks(
                question, mode, topic,
                intent_info["intent"],
                syllabus_chunks,
                pyq_chunks,
            )
            answer_mode = "retrieval_only"
            print(f"[ANSWER_ENGINE] Answer generated via retrieval extraction ({answer_mode}).")

        # ── 9. Frequency summary suffix ─────────────────────────────────────
        if pyq_freq["times_asked"] > 0:
            ans_text += f"\n\n**Past Exam Recurrence**: {pyq_freq['frequency_summary']}"

        # ── 10. Citations ───────────────────────────────────────────────────
        citations = []
        seen_cits: set = set()
        for c in reranked_chunks:
            meta = c.get("metadata", {})
            src_file = meta.get("source_file", "document.pdf")
            src_page = meta.get("source_page", 1)
            c_type = meta.get("doc_type", "document")
            prefix = "Syllabus Source" if c_type == "syllabus" else "PYQ Source"
            cit_str = f"[{prefix}: {src_file}, Page {src_page}]"
            if cit_str not in seen_cits:
                seen_cits.add(cit_str)
                citations.append({
                    "type": c_type,
                    "source_file": src_file,
                    "source_page": src_page,
                    "citation_str": cit_str,
                })

        top_score = reranked_chunks[0]["final_score"] if reranked_chunks else 0.0

        # ── 11. Debug payload ───────────────────────────────────────────────
        debug_payload: Dict[str, Any] = {
            "query_topic": topic,
            "intent": intent_info["intent"],
            "answer_mode": answer_mode,
            "active_workspace": ws_id,
            "filters_applied": filters,
            "retrieved_count": len(candidate_chunks),
            "reranked_count": len(reranked_chunks),
            "syllabus_chunks_used": len(syllabus_chunks),
            "pyq_chunks_used": len(pyq_chunks),
            "retrieved_chunks": [
                {
                    "source_file": c.get("metadata", {}).get("source_file"),
                    "question_id": c.get("metadata", {}).get("question_id"),
                    "year": c.get("metadata", {}).get("year"),
                    "exact_text": (
                        c.get("metadata", {}).get("exact_text", "")[:100] or
                        c.get("text", "")[:100]
                    ),
                    "vector_score": c.get("score"),
                    "final_score": c.get("final_score"),
                    "doc_type": c.get("metadata", {}).get("doc_type"),
                }
                for c in reranked_chunks[:10]
            ],
            "top_retrieved_sources": [
                f"{c.get('metadata',{}).get('source_file')} "
                f"[{c.get('metadata',{}).get('question_id','')}] "
                f"score={c.get('final_score')}"
                for c in reranked_chunks[:5]
            ],
            "final_context_syllabus": syllabus_context[:400] if debug else "[omitted]",
            "final_context_pyq": pyq_context[:600] if debug else "[omitted]",
        }

        return {
            "question": question,
            "topic": topic,
            "intent": intent_info["intent"],
            "answer": ans_text,
            "answer_sections": [{"title": "Grounded Answer", "content": ans_text}],
            "pyq_evidence": pyq_evidence,
            "pyq_frequency": pyq_freq,
            "citations": citations,
            "grounding_score": top_score,
            "top_score": top_score,
            "retrieved_chunks_count": len(reranked_chunks),
            "answer_mode": answer_mode,
            "workspace_context": workspace_meta,
            "hallucination_guard_triggered": False,
            "clarification_requested": False,
            "debug": debug_payload,
        }
