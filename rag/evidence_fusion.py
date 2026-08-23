"""
Multi-signal evidence fusion for question-boundary decisions.

This module NEVER extracts questions by itself and NEVER bypasses the
grounding / validation gate. It fuses independent evidence signals that the
rest of the pipeline has already gathered into one bounded confidence score:

    marker        - printed/normalised question-number evidence
    layout        - geometric reconstruction support (OCR word coordinates)
    parent        - parent-question context exists and is consistent
    siblings      - sibling subquestion markers of the same parent detected
    typography    - font size / weight evidence when available (neutral if not)
    semantic      - academic verb / "?" / length (SUPPORTING ONLY, optional)
    continuity    - cross-page continuation span

Fusion philosophy:
    - No single signal is mandatory.
    - Marker present + strong layout + parent context  -> high confidence.
    - Marker missing but strong layout + content + parent -> still viable.
    - Marker present but detached (no layout/parent/sibling agreement)
      -> low confidence, treated as likely OCR artifact by callers.

Callers keep full control: they decide which gates use the score. The score
only ever re-ranks or annotates candidates that already passed structural
parsing; it cannot mint new IDs.

Pure functions, no I/O, no globals — remove this file and its two call sites
to revert the enhancement entirely.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Weights sum to 1.0. Semantic content is deliberately the weakest signal.
_WEIGHTS = {
    "marker": 0.30,
    "layout": 0.20,
    "parent": 0.15,
    "siblings": 0.15,
    "typography": 0.05,
    "semantic": 0.10,
    "continuity": 0.05,
}

_PARENT_SUB_RE = re.compile(r"Q(\d+)\(([^)]+)\)$", re.I)


def _norm_sub(sub: Optional[str]) -> str:
    return (sub or "").strip().lower()


def marker_signal(question_id: str, source_blob: str) -> float:
    """Printed marker evidence: does the canonical ID appear in the source?"""
    qid = (question_id or "").strip()
    src = (source_blob or "").lower()
    if not qid or not src:
        return 0.0
    m = _PARENT_SUB_RE.match(qid)
    if m:
        parent, sub = m.group(1), m.group(2)
        patterns = [
            rf"q\.?\s*{parent}\s*[\.\):\-]?\s*\(\s*{re.escape(sub)}\s*\)",
            rf"q\.?\s*{parent}\s*[\.\):\-]?\s*{re.escape(sub)}\s*[.\)\:]",
        ]
        return 1.0 if any(re.search(p, src, re.I) for p in patterns) else 0.4
    # Parent-only ID
    parent = qid.lstrip("Qq")
    return 1.0 if re.search(rf"q\.?\s*{re.escape(parent)}\b", src) else 0.3


def layout_signal(candidate: Dict[str, Any]) -> float:
    """Geometric support: layout-aware reconstruction or multi-repr agreement."""
    method = str(candidate.get("extraction_method") or "")
    if method == "ocr_layout":
        return 1.0
    if candidate.get("geometry_support"):
        return 1.0
    if method == "crop_ocr_recovery":
        return 0.6
    if method in ("native", "ocr_text", "ocr_text_hd"):
        return 0.5
    return 0.25


def parent_signal(candidate: Dict[str, Any], known_parents: set) -> float:
    """Parent context: this item's parent block actually exists."""
    parent = str(candidate.get("parent_question") or "").upper()
    if not parent:
        return 0.2
    return 1.0 if parent in known_parents else 0.3


def sibling_signal(question_id: str, all_ids: List[str]) -> float:
    """
    Sibling structure: other printed subquestions of the same parent form a
    letter/number run. A lone sub with no siblings still scores neutral-low;
    runs of 2+ are strong structural corroboration.
    """
    m = _PARENT_SUB_RE.match(question_id or "")
    if not m:
        return 0.5  # flat parents: neutral
    parent = m.group(1).upper()
    own = _norm_sub(m.group(2))
    sibs = set()
    for other in all_ids:
        om = _PARENT_SUB_RE.match(other or "")
        if om and om.group(1).upper() == parent:
            sibs.add(_norm_sub(om.group(2)))
    sibs.discard(own)
    n = len(sibs)
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.85
    if n == 1:
        return 0.65
    return 0.35


def typography_signal(candidate: Dict[str, Any]) -> float:
    """
    Font evidence when available (bold/larger marker glyphs). Absent data is
    neutral 0.5 — never penalised, never required.
    """
    info = candidate.get("typography") or {}
    if not isinstance(info, dict) or not info:
        return 0.5
    score = 0.5
    if info.get("bold"):
        score += 0.3
    size_ratio = info.get("size_ratio")
    if isinstance(size_ratio, (int, float)) and size_ratio > 0:
        if size_ratio >= 1.15:
            score += 0.2
        elif size_ratio < 0.85:
            score -= 0.2
    return max(0.0, min(1.0, score))


def semantic_signal(exact_text: str) -> float:
    """Content plausibility. Supporting only — absence must not reject."""
    t = (exact_text or "").strip()
    if not t:
        return 0.0
    score = 0.35
    if len(t) >= 25:
        score += 0.25
    if "?" in t:
        score += 0.2
    if re.search(
        r"\b(explain|what|discuss|describe|define|differentiate|compare|"
        r"derive|calculate|design|write|state|list|draw|justify|comment|"
        r"prove|compute|solve|formulate|construct|illustrate|evaluate)\b",
        t,
        re.I,
    ):
        score += 0.2
    return min(1.0, score)


def body_strength_score(exact_text: str) -> float:
    """
    Score the strength of a question body independently of the marker.
    Identifies academic commands, definitions, explanations, derivations, etc.
    Does NOT require '?'.
    """
    t = (exact_text or "").strip()
    if not t:
        return 0.0
    score = 0.2
    if len(t) >= 20:
        score += 0.2
    if len(t) >= 60:
        score += 0.1
    if "?" in t:
        score += 0.15
    if re.search(r"\[\s*\d+\s*\]", t):  # marks notation
        score += 0.15
    if re.search(
        r"\b(explain|what|discuss|describe|define|differentiate|compare|"
        r"derive|calculate|design|write|state|list|draw|justify|comment|"
        r"prove|compute|solve|formulate|construct|illustrate|evaluate)\b",
        t,
        re.I,
    ):
        score += 0.3
    return min(1.0, score)


def noise_penalty_score(exact_text: str, source_blob: str = "") -> float:
    """
    Compute noise penalty for a candidate (header/footer text, duration, admin lines,
    corrupted OCR characters, table debris).
    """
    t = (exact_text or "").strip()
    if not t:
        return 1.0
    penalty = 0.0
    if re.search(r"\b\d+\s*(?:hours?|hrs?|mins?|minutes?)\b", t, re.I):
        penalty += 0.4
    if re.search(r"\b(?:page\s+\d+\s+of\s+\d+|university|semester|code|paper\s*/\s*subject)\b", t, re.I):
        penalty += 0.3
    hits = sum(t.count(c) for c in "<>{}|")
    if hits >= 3:
        penalty += 0.3
    return min(1.0, penalty)


def continuity_signal(candidate: Dict[str, Any]) -> float:
    """Cross-page continuation: spanning records carry merge evidence."""
    pages = candidate.get("source_pages") or []
    if len(pages) > 1 or candidate.get("cross_page_merged"):
        return 1.0
    return 0.4


def fuse_candidate_evidence(
    candidate: Dict[str, Any],
    *,
    source_blob: str = "",
    all_ids: Optional[List[str]] = None,
    known_parents: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Fuse independent signals into one confidence score plus a per-signal
    breakdown for audit/UI. Returns:
        {
          "confidence": float in [0,1],
          "signals": {name: float},
        }
    """
    ids = all_ids or []
    parents = known_parents or set()
    qid = str(candidate.get("question_id") or "")

    signals = {
        "marker": marker_signal(qid, source_blob),
        "layout": layout_signal(candidate),
        "parent": parent_signal(candidate, parents),
        "siblings": sibling_signal(qid, ids),
        "typography": typography_signal(candidate),
        "semantic": semantic_signal(candidate.get("exact_text") or ""),
        "continuity": continuity_signal(candidate),
    }
    confidence = sum(_WEIGHTS[k] * v for k, v in signals.items())
    return {
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "signals": {k: round(v, 4) for k, v in signals.items()},
    }


def evaluate_question_level_evidence(
    candidate: Dict[str, Any],
    *,
    source_blob: str = "",
    all_ids: Optional[List[str]] = None,
    known_parents: Optional[set] = None,
    rep_count: int = 1,
) -> Dict[str, Any]:
    """
    Question-Level Evidence Model evaluation per candidate.
    Calculates candidate confidence and assigns explicit candidate status:
    DETECTED, VALIDATED, GROUNDED, ADMITTED, REJECTED, AMBIGUOUS, UNVERIFIED.
    """
    fusion = fuse_candidate_evidence(
        candidate,
        source_blob=source_blob,
        all_ids=all_ids,
        known_parents=known_parents,
    )
    b_score = body_strength_score(candidate.get("exact_text") or "")
    n_penalty = noise_penalty_score(candidate.get("exact_text") or "", source_blob)
    grounded_score = float(candidate.get("grounding_score") or 0.0)

    # Conceptual candidate_confidence combining marker, body, layout, parent, sibling, grounding & noise
    raw_conf = (
        0.35 * fusion["confidence"]
        + 0.35 * b_score
        + 0.15 * min(1.0, rep_count / 2.0)
        + 0.15 * grounded_score
        - 0.30 * n_penalty
    )
    cand_confidence = round(max(0.0, min(1.0, raw_conf)), 4)

    status = "DETECTED"
    exact = (candidate.get("exact_text") or "").strip()
    if exact and len(exact) >= 3:
        status = "VALIDATED"
    if grounded_score >= 0.5:
        status = "GROUNDED"

    if status == "GROUNDED" and cand_confidence >= 0.35 and n_penalty < 0.6:
        status = "ADMITTED"
    elif n_penalty >= 0.6:
        status = "REJECTED"
    elif cand_confidence < 0.35:
        status = "AMBIGUOUS"

    marker_conf = round(float(fusion["signals"]["marker"]), 4)
    layout_conf = round(float(fusion["signals"]["layout"]), 4)
    body_conf = round(b_score, 4)
    grounding_conf = round(grounded_score, 4)

    return {
        "candidate_confidence": cand_confidence,
        "overall_confidence": cand_confidence,
        "marker_confidence": marker_conf,
        "body_confidence": body_conf,
        "layout_confidence": layout_conf,
        "grounding_confidence": grounding_conf,
        "body_strength": round(b_score, 4),
        "noise_penalty": round(n_penalty, 4),
        "fusion": fusion,
        "status": status,
    }


def fusion_bonus(confidence: float, *, max_bonus: float = 40.0) -> float:
    """
    Bounded rank adjustment for reconciliation tie-breaks. Centred at 0.55 so
    only genuinely corroborated candidates gain, and weak ones lose up to
    -max_bonus — never enough to outrank grounding failures (-500) or beat
    real content against instruction frames (-400).
    """
    return round((confidence - 0.55) * (max_bonus / 0.45), 3)

