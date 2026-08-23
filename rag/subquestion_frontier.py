"""
Universal printed-frontier tracking for lettered / roman subquestions.

Additive hardening layer. Does not invent a–i children, does not assume a
fixed child count, and does not replace existing extraction / grounding rules.

A–I is a supported recognition range, not a required sibling set.
Missing markers are reported only when the *same parent* has source evidence
that the marker existed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rag.question_extractor import (
    ACADEMIC_QUESTION_VERBS,
    _PARENT_NUM,
    _SUB_TOKEN,
    _UNAMBIGUOUS_ROMAN,
    _normalize_subtoken,
    is_header_or_instruction,
)

# Supported alphabetic child labels. Recognition range, never a required count.
ALPHABETIC_CHILD_RANGE = "abcdefghi"
ALPHABETIC_CHILD_SET = set(ALPHABETIC_CHILD_RANGE)
ROMAN_CHILD_ORDER = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
ROMAN_CHILD_SET = set(ROMAN_CHILD_ORDER)

# Gated OCR substitutions. Never applied blindly — sibling / representation
# / body evidence must support the mapping.
OCR_LETTER_CONFUSIONS: Dict[str, Tuple[str, ...]] = {
    "a": ("o",),
    "o": ("a",),
    "b": ("6",),
    "c": ("e",),
    "e": ("c",),
    "d": ("cl",),
    "f": ("t",),
    "t": ("f",),
    "g": ("q",),
    "q": ("g",),
    "h": ("n",),
    "n": ("h",),
    "i": ("l", "1", "I"),
    "l": ("i", "1"),
    "1": ("i", "l"),
}

_Q_LEAD = rf"(?:Q\.?|Question(?:\s+No\.?)?)"
_PARENT_LINE = re.compile(
    rf"^(?:"
    rf"{_Q_LEAD}\s*({_PARENT_NUM})"
    rf"|({_PARENT_NUM})\s*(?:[\.\):]\s*|\s+)(?:Attempt|Solve|Answer|Write|Question|\()"
    rf"|({_PARENT_NUM})\s*\("
    rf")",
    re.I,
)
_MULTI_PAREN_LETTERS = re.compile(rf"\(({_SUB_TOKEN})\)", re.I)
_COLUMN_HEADER_LETTERS = re.compile(rf"^[a-i](?:\s+[a-i]){{2,}}$", re.I)
_FOOTER_LINE = re.compile(
    r"\bpage\s+\d+\b|\bend\s+of\s+(?:the\s+)?(?:paper|question)\b"
    r"|\bwatermark\b|\bqp\s*code\b",
    re.I,
)


def classify_marker_style(raw_marker: str, normalized_sub: Optional[str] = None) -> str:
    """Describe the printed/OCR form without replacing the raw token."""
    s = (raw_marker or "").strip()
    sub = (normalized_sub or "").lower()
    if sub in ROMAN_CHILD_SET or (sub in _UNAMBIGUOUS_ROMAN):
        if re.search(r"\([ivx]+\)", s, re.I) or re.match(r"^[ivx]+[\.\)]", s, re.I):
            return "roman"
    if re.search(rf"{_Q_LEAD}\s*{_PARENT_NUM}", s, re.I) and re.search(r"[a-z]", s, re.I):
        return "parent_child_compound"
    if re.search(r"^\[[a-z]\]", s, re.I):
        return "bracketed_letter"
    if re.search(r"^\([a-z](?!\))", s, re.I):
        return "unclosed_paren_letter"
    if re.search(r"\([a-z]\)", s, re.I):
        return "parenthesized_letter"
    if re.search(r"^[a-z]\)", s, re.I):
        return "closing_paren_letter"
    if re.search(r"^[a-z]\.", s, re.I):
        return "dotted_letter"
    if re.search(r"^[a-z][:\-]", s, re.I):
        return "delimited_letter"
    return "letter"


def marker_provenance_record(
    *,
    marker_id: str,
    raw_marker: Optional[str] = None,
    page: Optional[int] = None,
    representation: str = "unknown",
    bounding_box: Optional[Dict[str, Any]] = None,
    confidence: float = 0.9,
    provenance: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Canonical marker candidate. Normalized fields sit beside raw evidence;
    they never replace it.
    """
    raw = (raw_marker or marker_id or "").strip()
    sub = None
    m = re.match(r"(Q\d+)\(([^)]+)\)$", str(marker_id or ""), re.I)
    if m:
        sub = _normalize_subtoken(m.group(2))
    return {
        "marker_id": marker_id,
        "raw_marker": raw,
        "normalized_marker": sub or str(marker_id or ""),
        "marker_style": classify_marker_style(raw, sub),
        "page": page,
        "bounding_box": bounding_box,
        "representation": representation,
        "confidence": round(float(confidence), 3),
        "provenance": provenance or representation,
    }


def is_protected_letter_context(line: str, lines: Optional[Sequence[str]] = None, idx: int = 0) -> bool:
    """
    True when a letter token lives in a table, diagram legend, footer, or
    multi-label caption — not as a question child marker.
    """
    s = (line or "").strip()
    if not s:
        return False
    if _FOOTER_LINE.search(s) and not any(re.search(rf"\b{v}\b", s.lower()) for v in ACADEMIC_QUESTION_VERBS):
        return True
    hits = list(_MULTI_PAREN_LETTERS.finditer(s))
    starts_as_child = bool(
        re.match(
            rf"^(?:\(({_SUB_TOKEN})\)|{_SUB_TOKEN}[\.\)]|\[{_SUB_TOKEN}\])\s+\S",
            s,
            re.I,
        )
    )
    # Question child + table/grid body is still a question, not a caption.
    if starts_as_child and len(hits) < 2:
        rest = re.sub(
            rf"^(?:\(({_SUB_TOKEN})\)|{_SUB_TOKEN}[\.\)]|\[{_SUB_TOKEN}\])\s*",
            "",
            s,
            flags=re.I,
        )
        has_grid = sum(rest.count(c) for c in "<>{}|") >= 3
        has_verb = any(re.search(rf"\b{v}\b", rest.lower()) for v in ACADEMIC_QUESTION_VERBS)
        if has_grid or has_verb or len(rest.split()) >= 4:
            return False
        # Short numeric/token remainders inside a table/figure are cells.
    if s.count("|") >= 2 or s.count("\t") >= 2:
        return True
    if _COLUMN_HEADER_LETTERS.fullmatch(s):
        return True
    window = s
    if lines:
        lo = max(0, idx - 2)
        hi = min(len(lines), idx + 3)
        window = " ".join(lines[lo:hi])
    figureish = bool(re.search(r"\b(?:fig(?:ure)?|diagram|image|table|caption|legend)\b", window, re.I))
    if len(hits) >= 2:
        if figureish:
            return True
        if not any(re.search(rf"\b{v}\b", s.lower()) for v in ACADEMIC_QUESTION_VERBS):
            return True
    if figureish and hits and not any(re.search(rf"\b{v}\b", s.lower()) for v in ACADEMIC_QUESTION_VERBS):
        return True
    return False


def split_text_into_parent_regions(text: str) -> Dict[str, str]:
    """
    Map Qn → concatenated source lines belonging to that parent until the
    next parent starts. Parent IDs are discovered dynamically (Q1, Q2, … Q10,
    Q20, …). Page breaks are not parent resets: joining pages before calling
    this preserves a cross-page frontier.
    """
    if not text:
        return {}
    lines = text.splitlines()
    starts: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        m = _PARENT_LINE.match(s)
        if not m:
            continue
        num = m.group(1) or m.group(2) or m.group(3)
        if not num:
            continue
        if is_header_or_instruction(s) and not re.search(rf"{_Q_LEAD}\s*{num}\b", s, re.I):
            continue
        starts.append((i, f"Q{int(num)}"))
    regions: Dict[str, List[str]] = {}
    for n, (i, pid) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        regions.setdefault(pid.upper(), []).extend(lines[i:end])
    return {k: "\n".join(v) for k, v in regions.items()}


def parent_region_text(full_blob: str, parent: str) -> str:
    pid = str(parent or "").upper()
    if not pid:
        return ""
    return split_text_into_parent_regions(full_blob).get(pid, "")


def _family_and_rank(sub: str) -> Tuple[Optional[str], Optional[int]]:
    s = (sub or "").strip().lower()
    if not s:
        return None, None
    if s in ROMAN_CHILD_SET and not (len(s) == 1 and s == "i"):
        return "roman", ROMAN_CHILD_ORDER.index(s)
    if s in ROMAN_CHILD_SET and s == "i":
        # Ambiguous until the caller supplies sibling context.
        return "ambiguous_i", 0
    if len(s) == 1 and s.isalpha():
        return "alpha", ord(s) - ord("a")
    return None, None


def sibling_family(subs: Iterable[str]) -> str:
    """
    Choose alphabetic vs roman for a parent. Mixed styles of the SAME family
    ( (a) / b. / (c) ) stay alphabetic. Roman (i-iii) is not mapped to a-c.
    """
    tokens = [str(s).strip().lower() for s in subs if s]
    letters = {s for s in tokens if len(s) == 1 and s.isalpha() and s != "i"}
    roman = {s for s in tokens if s in _UNAMBIGUOUS_ROMAN}
    if letters and roman:
        return "alpha"  # nested roman is not this parent's sibling family
    if roman or (tokens and all(s in ROMAN_CHILD_SET for s in tokens) and "ii" in tokens):
        return "roman"
    return "alpha"


def printed_frontier_sub(subs: Sequence[str], family: str) -> Optional[str]:
    """Highest source-proven sibling. None when the parent has no children."""
    tokens = [str(s).strip().lower() for s in subs if s]
    if family == "roman":
        ranked = [s for s in ROMAN_CHILD_ORDER if s in tokens]
        return ranked[-1] if ranked else None
    alpha = sorted(s for s in tokens if len(s) == 1 and s.isalpha())
    return alpha[-1] if alpha else None


def slot_inference_justified(last_body: str) -> bool:
    """
    Markerless page-2 prose is a NEW sibling only when the last child did not
    already carry a complete question body (table/data occupying a slot).
    Complete academic bodies continue across the page break.
    """
    t = (last_body or "").strip()
    if not t:
        return True
    if sum(t.count(c) for c in "<>{}|") >= 3:
        return True
    words = t.split()
    has_verb = any(re.search(rf"\b{v}\b", t.lower()) for v in ACADEMIC_QUESTION_VERBS)
    if has_verb and len(words) >= 6:
        return False
    if len(words) <= 8 and not has_verb:
        return True
    return False


def maybe_correct_confused_letter(
    observed: str,
    *,
    sibling_letters: Sequence[str],
    other_repr_letters: Sequence[str],
    family: str = "alpha",
) -> Optional[str]:
    """
    Return a substitute letter only when sibling sequence AND another
    representation support it. Otherwise None (leave as ambiguous).
    """
    if family != "alpha":
        return None
    obs = (observed or "").strip().lower()
    if not obs or obs not in OCR_LETTER_CONFUSIONS:
        return None
    sibs = {s.lower() for s in sibling_letters if s and len(s) == 1}
    others = {s.lower() for s in other_repr_letters if s and len(s) == 1}
    for cand in OCR_LETTER_CONFUSIONS[obs]:
        if len(cand) != 1 or not cand.isalpha():
            continue
        if cand in others and cand not in sibs:
            # Internal gap that another representation already printed.
            ordered = sorted(sibs | {cand})
            if cand in ordered and ordered[0] <= cand <= ordered[-1]:
                return cand
    return None


def _qid_parent_sub(qid: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"(Q\d+)(?:\(([^)]+)\))?$", str(qid or ""), re.I)
    if not m:
        return None, None
    return m.group(1).upper(), (m.group(2) or "").lower() or None


def build_parent_frontier_audit(
    *,
    accepted: Sequence[Dict[str, Any]],
    marker_candidates: Sequence[Dict[str, Any]],
    missing_genuine: Sequence[str],
    ambiguous_markers: Sequence[Dict[str, Any]],
    rejected: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Per-parent printed-frontier report. Unobserved letters past the frontier
    are omitted — they were never proven to exist.
    """
    by_parent: Dict[str, Dict[str, Any]] = {}

    def _bucket(pid: str) -> Dict[str, Any]:
        key = pid.upper()
        if key not in by_parent:
            by_parent[key] = {
                "parent": key,
                "observed": set(),
                "recovered": set(),
                "inferred": set(),
                "source_proven_missing": set(),
                "ambiguous": [],
                "rejected_noise": [],
                "family": "alpha",
            }
        return by_parent[key]

    for mc in marker_candidates:
        pid, sub = _qid_parent_sub(str(mc.get("marker_id") or ""))
        if pid and sub:
            b = _bucket(pid)
            if (
                len(sub) == 1
                and sub.isalpha()
                and sub not in ALPHABETIC_CHILD_SET
                and sub not in ROMAN_CHILD_SET
            ):
                b["rejected_noise"].append(str(mc.get("marker_id") or f"{pid}({sub})"))
                continue
            b["observed"].add(sub)

    for q in accepted:
        pid = str(q.get("parent_question") or "").upper()
        sub = str(q.get("subquestion") or "").lower()
        if not pid and q.get("question_id"):
            pid, sub = _qid_parent_sub(str(q.get("question_id")))
            pid = pid or ""
            sub = sub or ""
        if not pid:
            continue
        b = _bucket(pid)
        if sub:
            b["recovered"].add(sub)
            if q.get("origin") == "inferred_stem" or q.get("slot_inferred"):
                b["inferred"].add(sub)
            else:
                b["observed"].add(sub)

    for mid in missing_genuine:
        pid, sub = _qid_parent_sub(str(mid))
        if pid and sub:
            _bucket(pid)["source_proven_missing"].add(sub)

    for am in ambiguous_markers:
        qid = str(am.get("question_id") or am.get("marker_id") or "")
        pid, sub = _qid_parent_sub(qid)
        if pid:
            _bucket(pid)["ambiguous"].append({
                "marker": qid,
                "reason": am.get("reason") or am.get("status") or "ambiguous",
            })

    for r in rejected:
        qid = str(r.get("question_id") or r.get("marker") or "")
        pid, sub = _qid_parent_sub(qid)
        reason = str(r.get("reason") or r.get("exact_rejection_reason") or "")
        if not pid:
            continue
        leap = "leap" in reason or "watermark" in reason or "footer" in reason or "noise" in reason
        if leap or (sub and len(sub) == 1 and sub.isalpha() and sub not in ALPHABETIC_CHILD_SET and sub not in ROMAN_CHILD_SET):
            _bucket(pid)["rejected_noise"].append(qid)

    reports: List[Dict[str, Any]] = []
    for pid in sorted(by_parent, key=lambda p: int(re.search(r"\d+", p).group(0)) if re.search(r"\d+", p) else 0):
        b = by_parent[pid]
        family = sibling_family(b["recovered"] | b["observed"])
        b["family"] = family
        # Printed frontier follows admitted children, not rejected OCR leaps.
        proven = set(b["recovered"]) | set(b["source_proven_missing"])
        if family == "alpha":
            proven = {s for s in proven if len(s) == 1 and s.isalpha()}
        if not proven:
            proven = set(b["observed"])
        frontier = printed_frontier_sub(list(proven), family)
        reports.append({
            "parent": pid,
            "printed_frontier": frontier,
            "observed": _sorted_subs(b["observed"], family),
            "observed_children": _sorted_subs(b["observed"], family),
            "recovered": _sorted_subs(b["recovered"], family),
            "recovered_children": _sorted_subs(b["recovered"], family),
            "inferred_candidates": _sorted_subs(b["inferred"], family),
            "source_proven_missing": _sorted_subs(b["source_proven_missing"], family),
            "ambiguous": b["ambiguous"],
            "ambiguous_candidates": b["ambiguous"],
            "rejected_noise": list(dict.fromkeys(b["rejected_noise"])),
            "marker_family": family,
        })
    return reports


def _sorted_subs(subs: Set[str], family: str) -> List[str]:
    if family == "roman":
        return [s for s in ROMAN_CHILD_ORDER if s in subs] + sorted(subs - set(ROMAN_CHILD_ORDER))
    alpha = sorted(s for s in subs if len(s) == 1 and s.isalpha())
    other = sorted(s for s in subs if s not in alpha)
    return alpha + other


def format_parent_frontier_reports(reports: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for r in reports:
        missing = r.get("source_proven_missing") or []
        amb = r.get("ambiguous") or []
        noise = r.get("rejected_noise") or []
        lines.append(
            f"{r.get('parent')} Printed frontier: {r.get('printed_frontier') or 'none'} "
            f"Observed: {','.join(r.get('observed') or []) or 'none'} "
            f"Recovered: {','.join(r.get('recovered') or []) or 'none'} "
            f"Source-proven missing: {','.join(missing) or 'none'} "
            f"Ambiguous: {','.join(str(a.get('marker') if isinstance(a, dict) else a) for a in amb) or 'none'} "
            f"Rejected noise: {','.join(noise) or 'none'}"
        )
    return lines


def parent_scoped_sub_present(region_text: str, sub: str) -> bool:
    """Whether a delimited sibling token appears inside one parent region."""
    if not region_text or not sub:
        return False
    escaped = re.escape(sub)
    q_lead = rf"(?:Q\.?|Question(?:\s+No\.?)?)"
    patterns = [
        re.compile(rf"^\s*\({escaped}\)\s+", re.I),
        re.compile(rf"^\s*{escaped}[\.\):\-]\s+", re.I),
        re.compile(rf"^\s*{escaped}\)\s+", re.I),
        re.compile(rf"^\s*\({escaped}(?=\s|$)", re.I),
        re.compile(rf"^\s*\[{escaped}\]\s+", re.I),
        re.compile(rf"^\s*{q_lead}\s*\d+\s*\({escaped}\)", re.I),
        re.compile(rf"^\s*{q_lead}\s*\d+\s*{escaped}[\.\)]", re.I),
        re.compile(rf"^\s*\d+\s*\({escaped}\)", re.I),
        re.compile(rf"^\s*\d+\s*{escaped}[\.\)]", re.I),
    ]
    lines = region_text.splitlines()
    for i, ln in enumerate(lines):
        if is_protected_letter_context(ln, lines, i):
            continue
        if any(p.search(ln) for p in patterns):
            return True
    return False
