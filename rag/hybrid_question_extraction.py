"""
Hybrid PDF → Question Understanding pipeline.

Deterministic extraction first; optional LLM reconstructs/validates boundaries.
Never invents wording — all accepted text must ground in source PDF text.
Subject-agnostic. No fixed question counts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.llm_client import call_llm_json, llm_configured
from rag.question_extractor import (
    ACADEMIC_QUESTION_VERBS,
    extract_marks,
    extract_questions_from_page_text,
    is_header_or_instruction,
    is_valid_question_id,
    normalize_question_text,
    prepare_page_text_for_extraction,
    validate_question_candidate,
    _normalize_subtoken,
    _PARENT_NUM,
    _SUB_TOKEN,
)


EXTRACTION_SYSTEM = """You are a university exam question extractor.
The PDF text is the ONLY source of truth.

You MUST:
- Identify every distinct examination question / subquestion in the provided text.
- Reconstruct broken OCR layouts (e.g. Ql. / a. / b. listed separately from bodies).
- Merge multi-line wraps into ONE complete question text.
- Merge continuations that span pages into ONE question.
- Preserve the ORIGINAL wording from the source (fix OCR letter confusions only).
- Remove obvious headers, footers, page numbers, and exam instructions.
- Normalize identifiers to forms like Q1(a), Q2(b).
- Return COMPLETE question text for each subquestion.

You MUST NOT:
- Invent sentences or concepts absent from the source.
- Summarize or paraphrase away the original wording.
- Answer the questions.
- Split one multi-concept exam item into multiple PYQs.
- Create topic names as questions.
- Assume a fixed number of questions (not always 15).

Return STRICT JSON only:
{
  "questions": [
    {
      "question_id": "Q1(a)",
      "parent_question": "Q1",
      "subquestion": "a",
      "exact_text": "full original question text without the Q1(a) prefix",
      "marks": 5,
      "confidence": 0.98,
      "source_pages": [1]
    }
  ]
}
"""


def detect_source_question_markers(text: str) -> List[str]:
    """
    Detect question/subquestion marker IDs present in source text.
    Used for bidirectional completeness (markers vs extracted).
    No fixed expected count. Prefer explicit Q-/Question-prefixed markers.
    """
    if not text:
        return []
    prepared = prepare_page_text_for_extraction(text)
    found: List[str] = []
    seen: Set[str] = set()

    # Only delimited exam markers. "Q.1. Any Four" is an instruction, not Q1(a).
    patterns = [
        re.compile(
            rf"(?:^|[\n\r])\s*(?:Q\.?|Question)\s*({_PARENT_NUM})\s*[\.\):\-]?\s*(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})\s*[^\w\s]{{0,3}}[\.\)])(?!\s*\d)",
            re.I | re.M,
        ),
        re.compile(
            rf"(?:Q\.?|Question)\s*({_PARENT_NUM})\(({_SUB_TOKEN})\)",
            re.I,
        ),
        # Line-start numbered subquestions: 1(a) / 1. a) / 2 a)
        re.compile(
            rf"(?:^|[\n\r])\s*({_PARENT_NUM})\s*[\.\)]\s*(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})\s*[\.\)])",
            re.I | re.M,
        ),
    ]

    for pat in patterns:
        for m in pat.finditer(prepared):
            parent = f"Q{m.group(1)}"
            sub_raw = m.group(2) or (m.group(3) if m.lastindex and m.lastindex >= 3 else None)
            if not sub_raw:
                continue
            sub = _normalize_subtoken(sub_raw)
            # Next parent "Q.6" must not become a fake Q5(q) sub-marker.
            if sub == "q" and re.match(r"Q\s*\d", m.group(0).split()[-1] or "", re.I):
                continue
            qid = f"{parent}({sub})"
            if is_valid_question_id(qid) and qid not in seen:
                seen.add(qid)
                found.append(qid)

    current_parent = None
    for line in prepared.splitlines():
        line = line.strip()
        pm = re.match(rf"^(?:Q\.?|Question)\s*({_PARENT_NUM})\b", line, re.I)
        if pm:
            current_parent = f"Q{pm.group(1)}"
        sm = re.fullmatch(rf"\(?({_SUB_TOKEN})\)?[\.\)]?\s*", line, re.I)
        if sm and current_parent:
            qid = f"{current_parent}({_normalize_subtoken(sm.group(1))})"
            if is_valid_question_id(qid) and qid not in seen:
                seen.add(qid)
                found.append(qid)

    return found


def _significant_tokens(text: str) -> List[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "be", "by", "as", "at", "from", "that", "this", "it",
        "using", "use", "any", "all",
    }
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if len(t) > 2 and t not in stop]


def text_grounded_in_source(
    exact_text: str,
    source_blob: str,
    *,
    min_token_overlap: float = 0.55,
) -> Tuple[bool, float, str]:
    """
    Reject LLM inventions: majority of significant tokens must appear in source.
    Returns (ok, overlap_ratio, reason).
    """
    if not exact_text or not exact_text.strip():
        return False, 0.0, "empty_text"
    src = (source_blob or "").lower()
    if not src.strip():
        return False, 0.0, "empty_source"

    compact_q = re.sub(r"\s+", " ", exact_text.lower()).strip()
    compact_s = re.sub(r"\s+", " ", src)
    if len(compact_q) >= 20 and compact_q[:80] in compact_s:
        return True, 1.0, "substring_match"

    q_toks = _significant_tokens(exact_text)
    if len(q_toks) < 3:
        if q_toks and all(t in src for t in q_toks):
            return True, 1.0, "short_grounded"
        return False, 0.0, "too_short_ungrounded"

    hits = sum(1 for t in q_toks if t in src)
    ratio = hits / len(q_toks)
    # Extra invention signal: novel academic tokens not present in source
    novel = [t for t in q_toks if t not in src and len(t) > 3]
    if novel and ratio < 0.85:
        return False, ratio, "invented_or_ungrounded_text"
    if ratio < min_token_overlap:
        return False, ratio, "invented_or_ungrounded_text"
    return True, ratio, "token_overlap"


def text_not_truncated_vs_span(exact_text: str, source_span: str) -> Tuple[bool, str]:
    """
    Compare LLM candidate against a known single-question source span.
    Rejects dropping major trailing clauses (calculate/compare/including...).
    """
    if not exact_text or not source_span:
        return True, "skip"
    if len(exact_text) >= int(0.75 * len(source_span)):
        return True, "length_ok"
    e = set(_significant_tokens(exact_text))
    s = _significant_tokens(source_span)
    if not s:
        return True, "empty_span"
    cov = sum(1 for t in s if t in e) / len(s)
    if cov < 0.7:
        return False, "truncated_vs_source_span"
    return True, "coverage_ok"


def _normalize_llm_question(raw: Dict[str, Any], default_page: int) -> Optional[Dict[str, Any]]:
    qid = str(raw.get("question_id") or "").strip()
    qid = re.sub(r"^Q[lI](?=\(|$)", "Q1", qid, flags=re.I)

    if not is_valid_question_id(qid):
        parent = str(raw.get("parent_question") or "").strip()
        sub = raw.get("subquestion")
        parent = re.sub(r"^Q?[lI]$", "Q1", parent, flags=re.I)
        if parent and not parent.upper().startswith("Q"):
            parent = f"Q{parent}"
        if parent and sub is not None and str(sub).strip() != "":
            qid = f"{parent}({_normalize_subtoken(str(sub))})"
        elif parent:
            qid = parent if parent.upper().startswith("Q") else f"Q{parent}"

    if not is_valid_question_id(qid):
        return None

    pm = re.match(rf"(Q{_PARENT_NUM})(?:\(({_SUB_TOKEN})\))?$", qid, re.I)
    if not pm:
        return None
    parent = f"Q{pm.group(1)}" if not pm.group(1).upper().startswith("Q") else pm.group(1)
    # pm.group(1) already includes Q from pattern (Q\d+)
    parent = re.match(r"Q\d+", qid, re.I).group(0)
    parent = "Q" + re.search(r"\d+", parent).group(0)
    sub = _normalize_subtoken(pm.group(2)) if pm.group(2) else None
    qid = f"{parent}({sub})" if sub else parent
    if not is_valid_question_id(qid):
        return None

    exact = str(raw.get("exact_text") or "").strip()
    exact = re.sub(rf"^{re.escape(qid)}\s*[:.\-]?\s*", "", exact, flags=re.I).strip()
    if not exact:
        return None

    marks = raw.get("marks")
    try:
        marks = int(marks) if marks is not None else extract_marks(exact)
    except (TypeError, ValueError):
        marks = extract_marks(exact)
    if not (1 <= int(marks) <= 50):
        marks = 5 if parent == "Q1" else 10

    pages = raw.get("source_pages") or [default_page]
    try:
        pages = [int(p) for p in pages]
    except (TypeError, ValueError):
        pages = [default_page]

    try:
        conf = float(raw.get("confidence", 0.8))
    except (TypeError, ValueError):
        conf = 0.8

    return {
        "question_id": qid,
        "parent_question": parent,
        "subquestion": sub,
        "exact_text": exact,
        "marks": int(marks),
        "confidence": conf,
        "source_pages": pages,
        "extraction_method": "llm_hybrid",
    }


def llm_extract_questions_from_document(
    pages: List[Dict[str, Any]],
    *,
    filename: str = "",
) -> List[Dict[str, Any]]:
    """
    pages: list of {page, reconstructed_text, raw_native_text, raw_ocr_text}
    """
    if not llm_configured() or not pages:
        return []

    parts = []
    for p in pages:
        page_no = p.get("page", 1)
        recon = p.get("reconstructed_text") or ""
        native = p.get("raw_native_text") or ""
        ocr = p.get("raw_ocr_text") or ""
        parts.append(
            f"=== PAGE {page_no} ===\n"
            f"[RECONSTRUCTED]\n{recon[:6000]}\n"
            f"[NATIVE_SNIPPET]\n{native[:1500]}\n"
            f"[OCR_SNIPPET]\n{(ocr or '')[:1500]}\n"
        )
    user = (
        f"Source file: {filename}\n"
        "Extract ALL complete examination questions/subquestions from this paper.\n"
        "Merge cross-page continuations. Preserve original wording.\n\n"
        + "\n".join(parts)
    )
    data = call_llm_json(EXTRACTION_SYSTEM, user, max_tokens=6000, temperature=0.05)
    if not data:
        return []
    out = []
    for raw in data.get("questions") or []:
        if not isinstance(raw, dict):
            continue
        norm = _normalize_llm_question(raw, default_page=pages[0].get("page", 1))
        if norm:
            out.append(norm)
    return out


def validate_grounded_questions(
    candidates: List[Dict[str, Any]],
    source_blob: str,
    *,
    page_hint: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic validation + grounding after LLM/deterministic extract."""
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for c in candidates:
        qid = c.get("question_id")
        exact = (c.get("exact_text") or "").strip()
        page = (c.get("source_pages") or [page_hint])[0]

        if not is_valid_question_id(qid or ""):
            rejected.append(
                {
                    "question_id": qid,
                    "raw_text": exact,
                    "reason": "invalid_question_id",
                    "page": page,
                }
            )
            continue
        if qid in seen_ids:
            rejected.append(
                {
                    "question_id": qid,
                    "raw_text": exact,
                    "reason": "duplicate_question_id",
                    "page": page,
                }
            )
            continue

        ok_q, reason, metrics = validate_question_candidate(exact)
        if not ok_q:
            rejected.append(
                {
                    "question_id": qid,
                    "raw_text": exact,
                    "reason": reason,
                    "page": page,
                    "metrics": metrics,
                }
            )
            continue

        grounded, ratio, g_reason = text_grounded_in_source(exact, source_blob)
        if not grounded:
            rejected.append(
                {
                    "question_id": qid,
                    "raw_text": exact,
                    "reason": g_reason,
                    "page": page,
                    "metrics": {"grounding_ratio": round(ratio, 3)},
                }
            )
            continue

        # Header/footer contamination check
        if is_header_or_instruction(exact) or any(
            t in exact.lower() for t in ("page of", "qp code", "end of paper")
        ):
            rejected.append(
                {
                    "question_id": qid,
                    "raw_text": exact,
                    "reason": "header_footer_contamination",
                    "page": page,
                }
            )
            continue

        seen_ids.add(qid)
        # Strip trailing header/footer bleed attached to otherwise valid questions
        exact = _strip_trailing_contamination(exact)
        c = {**c, "exact_text": exact}
        accepted.append(c)

    return accepted, rejected


_CONTAMINATION_TAIL = re.compile(
    r"(?:\s+(?:Engineering|University|Paper\s*/\s*Subject|QP\s*CODE|Page\s+\d+|B\.E\.|Sem\s+[IVX]+).*)+$"
    r"|(?:\s+\d+\]?\s*\d{3,}\s*)+$"
    r"|(?:\s+10\]\s*\d+\s*)+$",
    re.I,
)


def _strip_trailing_contamination(text: str) -> str:
    t = (text or "").strip()
    t2 = _CONTAMINATION_TAIL.sub("", t).strip()
    return t2 if len(t2) >= 12 else t


def compute_extraction_quality(
    accepted_ids: List[str],
    source_markers: List[str],
    rejected: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Bidirectional completeness: markers in source vs extracted IDs.
    No hardcoded expected count. Rejected OCR-noise markers are not "missing".
    """
    acc_set = set(accepted_ids)
    noise_reasons = {
        "lacks_academic_question_structure",
        "header_or_instruction",
        "too_short",
        "invalid_question_id",
        "page_artifact",
    }
    noise_ids = {
        str(r.get("question_id"))
        for r in (rejected or [])
        if r.get("question_id") and (r.get("reason") or r.get("rejection_reason") or "") in noise_reasons
    }
    mark_set = set(source_markers) - noise_ids
    missing = sorted(mark_set - acc_set)
    extra = sorted(acc_set - mark_set)  # extracted without marker evidence (warn only)

    n_acc = len(acc_set)
    n_mark = len(mark_set)

    if n_acc == 0:
        quality = "FAILED"
        confidence = 0.0
    elif n_mark == 0:
        quality = "COMPLETE" if n_acc >= 1 else "FAILED"
        confidence = 0.75 if n_acc >= 3 else 0.55
    elif not missing:
        quality = "COMPLETE"
        confidence = min(0.99, 0.8 + 0.15 * (n_acc / max(n_mark, 1)))
    elif n_acc >= n_mark:
        quality = "COMPLETE"
        confidence = 0.85
    elif n_acc >= 1 and (n_acc / n_mark) >= 0.45:
        quality = "PARTIAL"
        confidence = round(n_acc / n_mark, 3)
    else:
        quality = "PARTIAL" if n_acc else "FAILED"
        confidence = round(n_acc / n_mark, 3) if n_mark else 0.0

    return {
        "questions_extracted": n_acc,
        "source_markers_detected": n_mark,
        "missing_questions": missing,
        "extra_without_marker": extra,
        "extraction_quality": quality,
        "confidence": round(confidence, 3),
        "question_extraction_confidence": round(confidence, 3),
    }


def leading_continuation_text(prepared_text: str) -> str:
    """
    Text at the top of a page that precedes the page's first question marker
    and reads like the tail of a sentence started on the previous page.
    """
    if not prepared_text:
        return ""
    lines = [ln.strip() for ln in prepared_text.splitlines() if ln.strip()]
    collected: List[str] = []
    for ln in lines:
        if re.match(rf"^Q{_PARENT_NUM}(?:\({_SUB_TOKEN}\))?\b", ln, re.I):
            break
        if is_header_or_instruction(ln):
            continue
        collected.append(ln)
    if not collected:
        return ""
    first = collected[0]
    # Only a genuine mid-sentence fragment counts as a continuation.
    if not (first[0].islower() or first.startswith(("-", "\u2022", ","))):
        return ""
    return " ".join(collected).strip()


def recover_truncated_page_tails(
    pages: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    *,
    filename: str,
    workspace_id: str,
    subject: str,
    year: int,
    syllabus_topics: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[int], int]:
    """
    Rescue questions that run off the bottom of a page.

    A question whose text is cut by a page break is rejected page-locally as a
    truncated fragment, which is correct in isolation but wrong for the document:
    the rest of the sentence is simply on the next page. When the following page
    opens with that continuation, the question is rebuilt and re-validated
    through the normal extractor. Nothing is admitted here that would not have
    been admitted had the PDF placed the question on a single page, so this
    removes a false failure without loosening any gate.

    Returns (recovered, still_rejected, consumed_page_numbers, count).
    """
    if len(pages) < 2 or not rejected:
        return [], rejected, set(), 0

    text_by_page = {
        int(p.get("page", 1)): (p.get("reconstructed_text") or "") for p in pages
    }

    recovered: List[Dict[str, Any]] = []
    still_rejected: List[Dict[str, Any]] = []
    consumed: Set[int] = set()

    for rej in rejected:
        if rej.get("reason") != "truncated_question_fragment":
            still_rejected.append(rej)
            continue

        page_no = int(rej.get("page") or 0)
        next_no = page_no + 1
        fragment = leading_continuation_text(text_by_page.get(next_no, ""))
        qid = rej.get("question_id") or ""
        body = (rej.get("raw_text") or "").strip()
        if not fragment or not qid or not body or next_no in consumed:
            still_rejected.append(rej)
            continue

        rebuilt = f"{qid} {body} {fragment}".strip()
        acc, _rej2 = extract_questions_from_page_text(
            page_text=rebuilt,
            page_num=page_no,
            source_file=filename,
            workspace_id=workspace_id,
            subject=subject,
            year=year,
            syllabus_topics=syllabus_topics,
        )
        match = next((q for q in acc if q.get("question_id") == qid), None)
        if not match:
            still_rejected.append(rej)
            continue

        match["source_pages"] = [page_no, next_no]
        match["source_page"] = page_no
        match["source_page_end"] = next_no
        match["cross_page_merged"] = True
        match["extraction_method"] = "deterministic_cross_page"
        recovered.append(match)
        consumed.add(next_no)

    return recovered, still_rejected, consumed, len(recovered)


def merge_cross_page_continuations(
    questions: List[Dict[str, Any]],
    pages: List[Dict[str, Any]],
    skip_pages: Optional[Set[int]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Attach a page's leading sentence fragment to the last question of the
    previous page. Produces ONE question spanning both pages rather than a
    truncated question plus an orphan fragment.

    Pages listed in skip_pages have already had their leading fragment consumed
    by tail recovery and must not be attached a second time.
    """
    if len(pages) < 2 or not questions:
        return questions, 0
    skip_pages = skip_pages or set()

    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for q in questions:
        page = (q.get("source_pages") or [q.get("source_page") or 1])[0]
        by_page.setdefault(int(page), []).append(q)

    merges = 0
    for p in pages:
        page_no = int(p.get("page", 1))
        if page_no in skip_pages:
            continue
        prev_qs = by_page.get(page_no - 1)
        if not prev_qs:
            continue
        fragment = leading_continuation_text(p.get("reconstructed_text") or "")
        if not fragment:
            continue
        target = prev_qs[-1]
        tail = (target.get("exact_text") or "").rstrip()
        if tail.endswith((".", "?", "!", ":")):
            continue
        target["exact_text"] = f"{tail} {fragment}".strip()
        spans = sorted({*(target.get("source_pages") or [page_no - 1]), page_no})
        target["source_pages"] = spans
        target["source_page_end"] = spans[-1]
        target["cross_page_merged"] = True
        merges += 1

    return questions, merges


def merge_prefer_longer(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge two candidate lists by question_id, keeping longer exact_text."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for q in a + b:
        qid = q.get("question_id")
        if not qid:
            continue
        if qid not in by_id or len(q.get("exact_text") or "") > len(by_id[qid].get("exact_text") or ""):
            by_id[qid] = q
    # Stable-ish order by parent/sub
    def sort_key(q: Dict[str, Any]):
        m = re.match(r"Q(\d+)(?:\((\w+)\))?", q.get("question_id", ""), re.I)
        if not m:
            return (999, "z")
        return (int(m.group(1)), (m.group(2) or "").lower())

    return sorted(by_id.values(), key=sort_key)


def hybrid_extract_document(
    pages: List[Dict[str, Any]],
    *,
    filename: str,
    workspace_id: str,
    subject: str = "Subject",
    year: int = 0,
    syllabus_topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full hybrid extraction for one PDF (multi-page).

    pages items must include:
      page, raw_native_text, raw_ocr_text, reconstructed_text, ocr_used
    """
    recon_parts = [p.get("reconstructed_text") or "" for p in pages]
    if any(part.strip() for part in recon_parts):
        source_blob = "\n\n".join(recon_parts)
    else:
        source_blob = "\n\n".join(
            (p.get("raw_ocr_text") or p.get("raw_native_text") or "")
            for p in pages
        )
    markers = detect_source_question_markers(source_blob)

    # Deterministic per-page
    det_all: List[Dict[str, Any]] = []
    det_rej: List[Dict[str, Any]] = []
    for p in pages:
        text = p.get("reconstructed_text") or ""
        if not text.strip():
            continue
        acc, rej = extract_questions_from_page_text(
            page_text=text,
            page_num=p.get("page", 1),
            source_file=filename,
            workspace_id=workspace_id,
            subject=subject,
            year=year,
            syllabus_topics=syllabus_topics,
        )
        for q in acc:
            q["source_pages"] = [p.get("page", 1)]
            q["extraction_method"] = q.get("extraction_method") or "deterministic"
        det_all.extend(acc)
        det_rej.extend(rej)

    # A page break must not destroy a question. Rebuild page-tail fragments
    # first, then attach any remaining leading fragments to accepted questions.
    recovered, det_rej, consumed_pages, tail_recoveries = recover_truncated_page_tails(
        pages,
        det_rej,
        filename=filename,
        workspace_id=workspace_id,
        subject=subject,
        year=year,
        syllabus_topics=syllabus_topics,
    )
    det_all.extend(recovered)
    det_all, cross_page_merges = merge_cross_page_continuations(
        det_all, pages, skip_pages=consumed_pages
    )
    cross_page_merges += tail_recoveries

    llm_raw: List[Dict[str, Any]] = []
    if llm_configured():
        llm_raw = llm_extract_questions_from_document(pages, filename=filename)

    # LLM candidates may only extend deterministic ones, never truncate them.
    det_by_id = {q.get("question_id"): q for q in det_all}
    llm_checked: List[Dict[str, Any]] = []
    for cand in llm_raw:
        anchor = det_by_id.get(cand.get("question_id"))
        if anchor:
            ok, why = text_not_truncated_vs_span(
                cand.get("exact_text") or "", anchor.get("exact_text") or ""
            )
            if not ok:
                det_rej.append(
                    {
                        "question_id": cand.get("question_id"),
                        "raw_text": (cand.get("exact_text") or "")[:240],
                        "reason": f"llm_{why}",
                        "page": (cand.get("source_pages") or [1])[0],
                    }
                )
                continue
        llm_checked.append(cand)

    merged = merge_prefer_longer(det_all, llm_checked) if llm_checked else det_all

    accepted, rejected = validate_grounded_questions(merged, source_blob)
    rejected.extend(det_rej)

    # Enrich accepted with canonical fields if missing (from LLM path)
    enriched = []
    for q in accepted:
        exact = q["exact_text"]
        qid = q["question_id"]
        _g_ok, g_ratio, _g_reason = text_grounded_in_source(exact, source_blob)
        q["grounding_score"] = round(g_ratio, 3)
        spans = sorted({int(p) for p in (q.get("source_pages") or [1])})
        q["source_pages"] = spans
        q["source_page_start"] = spans[0]
        q["source_page_end"] = spans[-1]
        if "normalized_text" not in q:
            from rag.question_extractor import (
                CanonicalConceptExtractor,
                build_question_representation,
            )

            q["normalized_text"] = normalize_question_text(exact)
            # Topic extraction ONLY after complete question text (separate stage)
            concepts = CanonicalConceptExtractor.extract_canonical_concepts(
                exact, syllabus_topics=syllabus_topics
            )
            intent = build_question_representation(qid, exact)
            q["detected_topics"] = concepts
            q["canonical_concepts"] = concepts
            q["question_intent"] = intent.get("question_intent", "")
            q["question_type"] = intent.get("question_type", "")
            q["entities"] = intent.get("entities", [])
            q["constraints"] = intent.get("constraints", [])
            q["year"] = year
            q["source_file"] = filename
            q["source_page"] = (q.get("source_pages") or [1])[0]
            q["workspace_id"] = workspace_id
            q["subject"] = subject
            q["rejected_question"] = False
            q["parent_marks"] = 20
            q["marks_total"] = q.get("marks", 10)
            q["question_number"] = qid
            q["question_type_struct"] = "SINGLE"
            q["quality_score"] = float(q.get("confidence") or 0.9)
            q["syllabus_mapping"] = {
                "module": "Unmapped",
                "chapter": "Unmapped",
                "topic": concepts[0] if concepts else "Unmapped",
            }
        enriched.append(q)

    # Optional LLM topic understanding (does not change question text)
    if llm_configured() and enriched:
        enrich_topics_with_llm(enriched)

    quality = compute_extraction_quality(
        [q["question_id"] for q in enriched], markers, rejected=rejected
    )

    grounding_scores = [q.get("grounding_score", 0.0) for q in enriched]
    return {
        "accepted_questions": enriched,
        "rejected_candidates": rejected,
        "source_markers": markers,
        "quality": quality,
        "llm_used": bool(llm_checked),
        "llm_candidates": len(llm_raw),
        "llm_rejected_for_truncation": len(llm_raw) - len(llm_checked),
        "cross_page_merges": cross_page_merges,
        "grounding_coverage": round(
            sum(grounding_scores) / len(grounding_scores), 3
        ) if grounding_scores else 0.0,
        "source_blob_chars": len(source_blob),
    }


TOPIC_SYSTEM = """You enrich academic exam questions with topics.
Return STRICT JSON:
{
  "primary_topic": "...",
  "secondary_topics": ["..."],
  "entities": ["..."],
  "question_intent": "explain|calculate|...",
  "question_type": "...",
  "constraints": ["..."]
}
Use only concepts present in the question text. Do not invent syllabus modules.
Do not rewrite the question.
"""


def enrich_topics_with_llm(questions: List[Dict[str, Any]], *, max_items: int = 40) -> None:
    """In-place topic enrichment; never modifies exact_text."""
    for q in questions[:max_items]:
        exact = q.get("exact_text") or ""
        if len(exact) < 20:
            continue
        data = call_llm_json(
            TOPIC_SYSTEM,
            f"Question ID: {q.get('question_id')}\nText: {exact}",
            max_tokens=500,
            temperature=0.1,
        )
        if not data:
            continue
        primary = str(data.get("primary_topic") or "").strip()
        secondary = data.get("secondary_topics") or []
        if primary:
            q["primary_topic"] = primary
            # Keep concept list but put primary first
            concepts = [primary] + [str(s) for s in secondary if str(s).strip()]
            # Dedupe
            seen = set()
            clean = []
            for c in concepts:
                k = c.lower()
                if k not in seen:
                    seen.add(k)
                    clean.append(c)
            q["canonical_concepts"] = clean[:8]
            q["detected_topics"] = clean[:8]
        if data.get("question_intent"):
            q["question_intent"] = str(data["question_intent"])
        if data.get("question_type"):
            q["question_type"] = str(data["question_type"])
        if isinstance(data.get("entities"), list):
            q["entities"] = [str(e) for e in data["entities"][:12]]
        if isinstance(data.get("constraints"), list):
            q["constraints"] = [str(c) for c in data["constraints"][:12]]


RECURRENCE_SYSTEM = """Classify the relationship between two university exam questions.
Labels ONLY: EXACT_REPEAT | SEMANTIC_REPEAT | RELATED_TOPIC | DIFFERENT

Rules:
- EXACT_REPEAT: same question (minor wording/OCR differences only).
- SEMANTIC_REPEAT: same exam intent, entities, and constraints; paraphrase OK.
- RELATED_TOPIC: same broad topic but DIFFERENT ask (e.g. CNN vs RNN architecture).
- DIFFERENT: not related as repeats.

Prefer DIFFERENT or RELATED_TOPIC over false SEMANTIC_REPEAT.
Return STRICT JSON: {"label": "...", "confidence": 0.0, "reason": "..."}
"""


def llm_judge_question_pair(text_a: str, text_b: str) -> Optional[Dict[str, Any]]:
    if not llm_configured():
        return None
    data = call_llm_json(
        RECURRENCE_SYSTEM,
        f"Question A:\n{text_a}\n\nQuestion B:\n{text_b}",
        max_tokens=300,
        temperature=0.0,
    )
    if not data:
        return None
    label = str(data.get("label") or "").upper().strip()
    if label not in {"EXACT_REPEAT", "SEMANTIC_REPEAT", "RELATED_TOPIC", "DIFFERENT"}:
        return None
    return data
