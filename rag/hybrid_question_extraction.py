"""
Hybrid PDF → Question Understanding pipeline.
Universal Question Reconciliation Architecture based on DocumentEvidence.

Never invents wording — all accepted text must ground in source PDF text.
Subject-agnostic. No fixed question counts.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

EXTRACTION_ENGINE_BUILD = "v3.7.1-universal-parent-frontier-20260823"


from rag.evidence_fusion import fuse_candidate_evidence, fusion_bonus
from rag.question_extractor import (
    ACADEMIC_QUESTION_VERBS,
    extract_marks,
    extract_questions_from_page_text,
    hyphen_underscore_is_compound_term,
    is_choice_instruction,
    is_header_or_instruction,
    is_ocr_glued_parent_jump,
    is_valid_question_id,
    iter_unlabelled_stems,
    normalize_question_text,
    prepare_page_text_for_extraction,
    structural_ocr_noise_ratio,
    validate_question_candidate,
    _normalize_subtoken,
    _next_unlabelled_sub,
    _PARENT_NUM,
    _Q_LEAD,
    _SUB_TOKEN,
    _SUB_LETTER,
    _is_marker_only_line,
    _should_skip_line,
)
from rag.subquestion_frontier import (
    build_parent_frontier_audit,
    format_parent_frontier_reports,
    is_protected_letter_context,
    marker_provenance_record,
    parent_region_text,
    parent_scoped_sub_present,
    slot_inference_justified,
    split_text_into_parent_regions,
)


@dataclass
class RecoveryCandidate:
    question_id: str
    source_span: str
    page: int
    representation: str
    layout_evidence: Dict[str, Any] = field(default_factory=dict)
    marker_evidence: str = ""
    body_evidence: str = ""
    grounding_score: float = 0.0
    confidence: float = 0.0
    recovery_reason: str = ""


def crop_and_ocr_suspicious_region(
    doc_or_path: Any,
    page_num: int,
    top_ratio: float = 0.55,
    bottom_ratio: float = 0.98,
    dpi: int = 300,
) -> str:
    """
    Performs targeted high-resolution OCR with multi-pass image preprocessing
    (grayscale, contrast enhancement, binarization thresholding) on a cropped region.
    """
    try:
        import fitz
        import pytesseract
        from PIL import Image, ImageEnhance, ImageOps

        doc = None
        should_close = False
        if isinstance(doc_or_path, str) and os.path.exists(doc_or_path):
            doc = fitz.open(doc_or_path)
            should_close = True
        elif hasattr(doc_or_path, "load_page"):
            doc = doc_or_path

        if not doc or page_num - 1 >= len(doc):
            return ""

        page = doc[page_num - 1]
        rect = page.rect
        crop_box = fitz.Rect(0, rect.height * top_ratio, rect.width, rect.height * bottom_ratio)
        pix = page.get_pixmap(clip=crop_box, dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Pass 1: Grayscale & Contrast enhancement
        gray = ImageOps.grayscale(img)
        enh = ImageEnhance.Contrast(gray).enhance(2.0)
        text_enh = pytesseract.image_to_string(enh)

        # Pass 2: Binarization thresholding
        bw = gray.point(lambda p: 255 if p > 150 else 0)
        text_bw = pytesseract.image_to_string(bw)

        if should_close:
            try:
                doc.close()
            except Exception:
                pass

        combined = f"{text_enh}\n{text_bw}".strip()
        return combined
    except Exception as e:
        print(f"[CROP_OCR_INFO] Targeted crop OCR failed: {e}")
        return ""



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


class DocumentEvidence:
    """
    Intermediate representation storing multi-representation evidence for a document.
    """

    def __init__(self, pages: List[Dict[str, Any]]):
        self.pages = pages
        self.representations: Dict[str, Dict[int, str]] = {
            "native": {},
            "ocr_text": {},
            "ocr_text_hd": {},
            "ocr_layout": {},
        }
        self.marker_candidates: List[Dict[str, Any]] = []
        self.body_candidates: List[Dict[str, Any]] = []
        self.question_candidates: List[Dict[str, Any]] = []
        self.relationships: List[Dict[str, Any]] = []
        self.rejected_candidates: List[Dict[str, Any]] = []
        self.cross_page_links: List[Dict[str, Any]] = []
        self.reconciled_questions: List[Dict[str, Any]] = []
        self.ambiguous_markers: List[Dict[str, Any]] = []
        self.missing_genuine_questions: List[str] = []
        self.representation_sources: Dict[str, str] = {}
        self.parent_frontier_audit: List[Dict[str, Any]] = []


def normalize_marker_id(raw_marker: str) -> Optional[str]:
    """
    Normalize generic OCR marker variations into standard Q<parent>(<sub>) format.
    Supported variations: Q4(b), Q.4(b), Q.4.b), Q4 b), Q4(b)., Q4 (b), 4(b), 4. b), 4 b)
    Rejects duration noise like '2 hours'.
    """
    if not raw_marker:
        return None
    s = raw_marker.strip()
    if re.search(r"\b\d+\s*(?:hours?|hrs?|marks?|mins?|minutes?)\b", s, re.I):
        return None
    s = re.sub(r"^Q[lI](?=\d|\(|\.|\s)", "Q1", s, flags=re.I)

    # 1. Q4(b) / Q.4(b) / Q.4.b) / Q4 b) / Q4(b). / 4(b) / 4. b) / Question No. 4(b)
    m = re.match(
        rf"^(?:{_Q_LEAD})?\.?\s*({_PARENT_NUM})\s*[\.\:\-]?\s*(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})\s*[\.\)]|\(({_SUB_TOKEN})(?=\s|$)|\[({_SUB_TOKEN})\]|\b({_SUB_LETTER})\b)",
        s,
        re.I,
    )
    if m:
        parent = f"Q{m.group(1)}"
        sub_raw = m.group(2) or m.group(3) or m.group(4) or m.group(5) or m.group(6)
        if sub_raw:
            sub = _normalize_subtoken(sub_raw)
            qid = f"{parent}({sub})"
            if is_valid_question_id(qid):
                return qid

    # 2. Parent-only: Q4 / Q.4 / Question 4 / Question No. 4
    m2 = re.match(rf"^(?:{_Q_LEAD})\s*({_PARENT_NUM})\s*[\.\)]?$", s, re.I)
    if m2:
        qid = f"Q{m2.group(1)}"
        if is_valid_question_id(qid):
            return qid

    return None


NOISE_REASONS = {
    "lacks_academic_question_structure",
    "isolated_sequence_leap_ocr_artifact",
    "invented_or_ungrounded_text",
    "invalid_question_id",
    "header_or_instruction",
}

_MARKER_PATTERNS = [
    re.compile(
        rf"(?:^|[\n\r])\s*(?:{_Q_LEAD})\s*({_PARENT_NUM})\s*[\.\):\-]?\s*(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})\s*[^\w\s]{{0,3}}[\.\)]|\(({_SUB_TOKEN})(?=\s|$)|\[({_SUB_TOKEN})\])(?!\s*\d)",
        re.I | re.M,
    ),
    re.compile(
        rf"(?:{_Q_LEAD})\s*({_PARENT_NUM})\(({_SUB_TOKEN})\)",
        re.I,
    ),
    re.compile(
        rf"(?:^|[\n\r])\s*({_PARENT_NUM})\s*[\.\)]\s*(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})\s*[^\w\s]{{0,3}}[\.\)])",
        re.I | re.M,
    ),
    # Bare inline form without any Q prefix or parent delimiter:
    # "1(a) Explain …", "2(b) Derive …" — printed sub in parentheses
    # straight after the parent number.
    re.compile(
        rf"(?:^|[\n\r])\s*({_PARENT_NUM})\s*\(({_SUB_TOKEN})\)\s+[A-Za-z]",
        re.I | re.M,
    ),
]


def detect_markers_in_text(text: str) -> List[str]:
    """
    Find candidate marker IDs in already-normalized text.
    Does not call prepare_page_text_for_extraction (avoids recursion).
    """
    if not text:
        return []
    found: List[str] = []
    seen: Set[str] = set()

    for pat in _MARKER_PATTERNS:
        for m in pat.finditer(text):
            parent = f"Q{m.group(1)}"
            sub_raw = None
            if m.lastindex:
                for gi in range(2, m.lastindex + 1):
                    g = m.group(gi)
                    if g:
                        sub_raw = g
                        break
            if not sub_raw:
                continue
            sub = _normalize_subtoken(sub_raw)
            if sub == "q" and re.fullmatch(r"Q\.?\s*\d+\s*", m.group(0).strip(), re.I):
                continue
            qid = f"{parent}({sub})"
            if is_valid_question_id(qid) and qid not in seen:
                seen.add(qid)
                found.append(qid)

    current_parent = None
    unlabelled_stem_mode = False
    unlabelled_sub: Optional[str] = None
    # Gutter-skeleton handling: OCR often reads a left marker column as a
    # block of consecutive bare parent numbers ("Q2.\nQ3.\nQ4.") detached
    # from their bodies. Orphan letters that follow must NOT be attributed
    # to those parents — the pairing is unknowable from sequence alone and
    # the layout-aware representation owns it.
    skeleton_parents: Set[str] = set()
    prev_nonempty_was_bare_parent = False
    raw_lines = [ln.strip() for ln in text.splitlines()]
    from rag.question_extractor import _LETTER_SIBLINGS, _UNAMBIGUOUS_ROMAN, is_choice_instruction

    def _is_bare_parent_line(ln: str) -> bool:
        return bool(
            re.fullmatch(rf"(?:Q\.?|Question)?\s*{_PARENT_NUM}\s*[\.\)]?", ln, re.I)
        )

    def _next_nonempty(i: int) -> str:
        for j in range(i + 1, len(raw_lines)):
            if raw_lines[j] and raw_lines[j].strip():
                return raw_lines[j].strip()
        return ""

    for idx, line in enumerate(raw_lines):
        if not line:
            continue
        was_prev_bare = prev_nonempty_was_bare_parent
        prev_nonempty_was_bare_parent = False
        if re.search(r"\b\d+\s*(?:hours?|hrs?|marks?|mins?|minutes?)\b", line, re.I):
            if not re.search(rf"(?:Q\.?|Question)\s*{_PARENT_NUM}", line, re.I):
                continue
        pm = re.match(
            rf"^(?:{_Q_LEAD})\s*({_PARENT_NUM})\b|^\s*({_PARENT_NUM})\s*[\.\)]?\s*(?:{_Q_LEAD})\s*(?:No\.?\s*)?{_PARENT_NUM}\b",
            line,
            re.I,
        )
        if not pm:
            bm = re.match(
                rf"^({_PARENT_NUM})\s*[\.\):]?\s*$"
                rf"|^({_PARENT_NUM})\s+(?:Attempt|Solve|any)\b"
                rf"|^({_PARENT_NUM})\s+({_SUB_LETTER})\s+(?:{'|'.join(sorted(ACADEMIC_QUESTION_VERBS))})\b",
                line,
                re.I,
            )
            if bm:
                new_num = bm.group(1) or bm.group(2) or bm.group(3)
                if re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*$", line):
                    # Marks-column / table digit runs ("7 8 9 3") are not
                    # parents: a bare number followed by another bare number
                    # is detached skeleton, never an ownable parent.
                    if _is_bare_parent_line(_next_nonempty(idx)):
                        prev_nonempty_was_bare_parent = True
                        continue
                    follows_letter = False
                    for nxt in raw_lines[idx + 1 : idx + 8]:
                        nxt = (nxt or "").strip()
                        if not nxt:
                            continue
                        if re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*$", nxt):
                            continue
                        if re.fullmatch(r"\[\s*\d+\s*\]", nxt) or re.match(r"^[\d\s]+$", nxt):
                            continue
                        sm = re.match(rf"^\(?({_SUB_TOKEN})\)?[\.\)]\s*", nxt, re.I)
                        if not sm:
                            sm = re.fullmatch(rf"\(?({_SUB_LETTER})\)?", nxt, re.I)
                        if sm and _normalize_subtoken(sm.group(1)) in _LETTER_SIBLINGS:
                            follows_letter = True
                        elif re.match(r"^(?:Attempt|Solve|any)\b", nxt, re.I):
                            follows_letter = True
                        break
                    if not follows_letter:
                        prev_nonempty_was_bare_parent = was_prev_bare  # bare-number runs stay contiguous
                        continue
                current_parent = f"Q{new_num}"
                if was_prev_bare and _is_bare_parent_line(line):
                    skeleton_parents.add(current_parent)
                prev_nonempty_was_bare_parent = _is_bare_parent_line(line)
                unlabelled_stem_mode = bool(re.search(r"(?:Attempt|Solve|any)\b", line, re.I))
                unlabelled_sub = None
                sub_from_bare = bm.group(4) if bm.lastindex and bm.lastindex >= 4 else None
                if sub_from_bare:
                    unlabelled_stem_mode = False
                    qid = f"{current_parent}({sub_from_bare.lower()})"
                    if is_valid_question_id(qid) and qid not in seen:
                        seen.add(qid)
                        found.append(qid)
                elif unlabelled_stem_mode:
                    for _stem in iter_unlabelled_stems(line):
                        unlabelled_sub = _next_unlabelled_sub(unlabelled_sub)
                        if not unlabelled_sub:
                            break
                        qid = f"{current_parent}({unlabelled_sub})"
                        if is_valid_question_id(qid) and qid not in seen:
                            seen.add(qid)
                            found.append(qid)
                continue
        if pm:
            # If Qn is isolated and appears before paper title / duration in header area, skip as top margin noise
            if idx < 15 and re.fullmatch(rf"^(?:{_Q_LEAD})\s*{_PARENT_NUM}\s*[\.\)]?$", line, re.I):
                header_ahead = any(
                    re.search(r"\b(?:hours?|hrs?|marks?|paper\s*/\s*subject|duration|code)\b", l, re.I)
                    for l in raw_lines[idx : idx + 10]
                )
                if header_ahead:
                    continue
            current_parent = f"Q{pm.group(1)}"
            # First member of a consecutive bare-parent run ("Q2.\nQ3.\nQ4."):
            # a detached gutter skeleton. It must never own orphan letters.
            if (
                _is_bare_parent_line(line)
                and _is_bare_parent_line(_next_nonempty(idx))
            ):
                skeleton_parents.add(current_parent)
            rest_pm = re.sub(rf"^(?:Q\.?|Question)\s*{pm.group(1)}\s*[\.\):]?\s*", "", line, flags=re.I)
            if was_prev_bare and _is_bare_parent_line(line):
                skeleton_parents.add(current_parent)
            prev_nonempty_was_bare_parent = _is_bare_parent_line(line)
            unlabelled_stem_mode = bool(rest_pm and is_choice_instruction(rest_pm))
            if unlabelled_stem_mode:
                unlabelled_sub = None
            sub_on_line = re.search(
                rf"(?:Q\.?|Question)\s*{pm.group(1)}\s*[\.\):\-]?\s*(?:\(({_SUB_TOKEN})\)|{_SUB_TOKEN}\s*[.\)])",
                line,
                re.I,
            )
            rest = re.sub(rf"^(?:Q\.?|Question)\s*{pm.group(1)}\s*[\.\):]?\s*", "", line, flags=re.I)
            parent_qid = f"Q{pm.group(1)}"
            if (
                not sub_on_line
                and is_valid_question_id(parent_qid)
                and parent_qid not in seen
                and rest
                and not is_header_or_instruction(line)
                and not is_choice_instruction(rest)
                and any(re.search(rf"\b{v}\b", rest.lower()) for v in ACADEMIC_QUESTION_VERBS)
            ):
                seen.add(parent_qid)
                found.append(parent_qid)
        if not pm:
            # Bare "5(a)" / "10 b)" / "20(a)" must become the active parent so
            # a following "b)" / "(c)" attaches to THAT parent, not the previous
            # Qn. Q-prefixed lines are already handled by `pm`.
            numbered_child = re.match(
                rf"^({_PARENT_NUM})\s*[\.\)]?\s*(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})[\.\)]|({_SUB_TOKEN})\))",
                line,
                re.I,
            )
            if numbered_child:
                current_parent = f"Q{numbered_child.group(1)}"
                unlabelled_stem_mode = False
                unlabelled_sub = None
                prev_nonempty_was_bare_parent = False
                sub_raw = numbered_child.group(2) or numbered_child.group(3) or numbered_child.group(4)
                if sub_raw:
                    qid = f"{current_parent}({_normalize_subtoken(sub_raw)})"
                    if is_valid_question_id(qid) and qid not in seen:
                        seen.add(qid)
                        found.append(qid)
        sm = re.match(rf"^\(?({_SUB_TOKEN})\)?[\.\):\-_]\s*", line, re.I)
        if sm and hyphen_underscore_is_compound_term(line):
            sm = None
        if not sm:
            sm = re.match(rf"^\(({_SUB_TOKEN})(?=\s|$)", line, re.I)
        if not sm:
            sm = re.match(rf"^\[({_SUB_TOKEN})\]\s*", line, re.I)
        if not sm:
            sm = re.match(
                rf"^({_SUB_TOKEN})\s+(?:{'|'.join(sorted(ACADEMIC_QUESTION_VERBS))})\b",
                line,
                re.I,
            )
        if not sm:
            sm_bare = re.fullmatch(rf"\(?({_SUB_LETTER})\)?", line, re.I)
            if sm_bare and current_parent and current_parent not in skeleton_parents:
                ahead = [x.strip() for x in raw_lines[idx + 1 :] if (x or "").strip()]
                if ahead and any(re.search(rf"\b{v}\b", ahead[0].lower()) for v in ACADEMIC_QUESTION_VERBS):
                    sm = sm_bare
            elif sm_bare and re.search(r"[.\)]", line):
                sm = sm_bare
        if sm and is_protected_letter_context(line, raw_lines, idx):
            sm = None
        if (
            sm
            and current_parent
            and current_parent not in skeleton_parents
            and not re.match(rf"^(?:{_Q_LEAD})\s*{_PARENT_NUM}", line, re.I)
        ):
            unlabelled_stem_mode = False
            sub = _normalize_subtoken(sm.group(1))
            parent_subs = {
                qid[len(current_parent) + 1 : -1].lower()
                for qid in found
                if qid.upper().startswith(current_parent.upper() + "(")
            }
            roman_follows = False
            ahead = [x.strip() for x in raw_lines[idx + 1 :] if (x or "").strip()]
            for i, nxt in enumerate(ahead):
                rm = re.match(rf"^\(?({_SUB_TOKEN})\)?[\.\)]\s*", nxt, re.I)
                if rm:
                    tok = _normalize_subtoken(rm.group(1))
                    if tok in _UNAMBIGUOUS_ROMAN:
                        roman_follows = True
                        break
                    if len(tok) == 1 and tok.isalpha() and tok != "i":
                        break
                if re.fullmatch(rf"{_SUB_LETTER}", nxt, re.I):
                    follow = ahead[i + 1] if i + 1 < len(ahead) else ""
                    if not any(re.search(rf"\b{v}\b", follow.lower()) for v in ACADEMIC_QUESTION_VERBS):
                        continue
                if re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*$", nxt):
                    continue
                if re.match(rf"^(?:Q\.?|Question)\s*{_PARENT_NUM}\b", nxt, re.I):
                    break
            if parent_subs & _LETTER_SIBLINGS and (
                sub in _UNAMBIGUOUS_ROMAN or (sub == "i" and roman_follows)
            ):
                continue
            qid = f"{current_parent}({sub})"
            if is_valid_question_id(qid) and qid not in seen:
                seen.add(qid)
                found.append(qid)
            unlabelled_stem_mode = False
        elif unlabelled_stem_mode and current_parent:
            glued = iter_unlabelled_stems(line)
            verb_start = any(re.match(rf"{v}\b", line.lower()) for v in ACADEMIC_QUESTION_VERBS)
            if verb_start or len(glued) > 1:
                if (
                    unlabelled_sub
                    and len(unlabelled_sub) == 1
                    and unlabelled_sub >= "d"
                    and re.search(r"\[\s*10\s*\]", line)
                    and len(glued) <= 1
                ):
                    unlabelled_stem_mode = False
                else:
                    for _stem in glued or [line]:
                        nxt = _next_unlabelled_sub(unlabelled_sub)
                        if not nxt:
                            break
                        unlabelled_sub = nxt
                        qid = f"{current_parent}({unlabelled_sub})"
                        if is_valid_question_id(qid) and qid not in seen:
                            seen.add(qid)
                            found.append(qid)

    return found


def detect_source_question_markers(text: str) -> List[str]:
    """
    Detect candidate marker IDs present in text across all representations.
    No fixed expected count. Strictly rejects noise like '2 hours'.
    """
    if not text:
        return []
    return detect_markers_in_text(prepare_page_text_for_extraction(text))


def drop_nested_roman_question_ids(ids: List[str]) -> List[str]:
    """
    `i.` / `ii.` after lettered siblings (a/b/…) are nested parts of that
    letter, not Qn(i)/Qn(ii). Roman-only papers (no a–h siblings) are kept.
    Letter `i` after `h` on an a–z paper is kept unless `ii` is also present.
    """
    from rag.question_extractor import _ROMAN_SUBS, _UNAMBIGUOUS_ROMAN, _LETTER_SIBLINGS

    by_parent: Dict[str, List[Tuple[str, str]]] = {}
    passthrough: List[str] = []
    for qid in ids:
        m = re.match(r"(Q\d+)\(([^)]+)\)$", qid, re.I)
        if not m:
            passthrough.append(qid)
            continue
        by_parent.setdefault(m.group(1).upper(), []).append((qid, m.group(2).lower()))
    keep: List[str] = list(passthrough)
    for items in by_parent.values():
        subs = {s for _, s in items}
        letters = {s for s in subs if s in _LETTER_SIBLINGS}
        if letters and (subs & _UNAMBIGUOUS_ROMAN):
            drop = {qid for qid, s in items if s in _ROMAN_SUBS}
            keep.extend(qid for qid, _s in items if qid not in drop)
        else:
            keep.extend(qid for qid, _s in items)
    return keep


def drop_leaping_sub_ids(ids: List[str]) -> List[str]:
    """
    When a parent already has a–h letter siblings, drop isolated far-leap
    letters produced by hyphenated terms or wrap fragments (n-gram, v.).
    Roman-only parents and contiguous a–z runs are unchanged.
    """
    from rag.question_extractor import _LETTER_SIBLINGS

    by_parent: Dict[str, List[Tuple[str, str]]] = {}
    passthrough: List[str] = []
    for qid in ids:
        m = re.match(r"(Q\d+)\(([^)]+)\)$", qid, re.I)
        if not m:
            passthrough.append(qid)
            continue
        by_parent.setdefault(m.group(1).upper(), []).append((qid, m.group(2).lower()))
    keep: List[str] = list(passthrough)
    for items in by_parent.values():
        subs = {s for _, s in items}
        if not (subs & _LETTER_SIBLINGS):
            keep.extend(qid for qid, _s in items)
            continue
        multi = [qid for qid, s in items if not (len(s) == 1 and s.isalpha())]
        keep.extend(multi)
        alpha = sorted(
            ((qid, s) for qid, s in items if len(s) == 1 and s.isalpha()),
            key=lambda pair: pair[1],
        )
        prev: Optional[str] = None
        for qid, s in alpha:
            if s in _LETTER_SIBLINGS or ("a" <= s <= "f"):
                keep.append(qid)
                prev = s
            elif prev is None or (ord(s) - ord(prev) <= 3):
                keep.append(qid)
                prev = s
    return keep


def drop_leaping_parent_ids(ids: List[str]) -> List[str]:
    """
    Drop OCR-glued parent numbers (Q5 + '3' → Q53), not legitimate Q10 / Q20.

    Real papers may number parents 1, 2, … 10, 20 independently of any
    particular question (Q6 is not special). Sequential parents (gap ≤ 3)
    are always kept. Non-adjacent round tens (10, 20, 30, …) are kept.
    """

    def _num(qid: str) -> Optional[int]:
        m = re.match(r"Q(\d+)", qid, re.I)
        return int(m.group(1)) if m else None

    nums = sorted({n for n in (_num(q) for q in ids) if n is not None})
    keep_n: Set[int] = set()
    last: Optional[int] = None
    for n in nums:
        if last is None or not is_ocr_glued_parent_jump(last, n):
            keep_n.add(n)
            last = n
    return [q for q in ids if (_num(q) is None or _num(q) in keep_n)]


def classify_genuine_markers(text: str) -> List[Dict[str, Any]]:
    """
    Classify detected markers as genuine source IDs vs OCR/header/duration noise.
    Never invents IDs that are not present in the text.
    """
    if not text:
        return []
    prepared = prepare_page_text_for_extraction(text)
    ids = detect_markers_in_text(prepared)
    lines = [ln.strip() for ln in prepared.splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    parents_with_subs = {
        re.match(r"(Q\d+)", qid, re.I).group(1)
        for qid in ids
        if "(" in qid
    }
    # Missing-marker model: every reported ID must have direct source
    # evidence. Alphabetic ranges are NEVER inferred for absent markers —
    # with ONE structural exception that is observed, not invented: a parent
    # whose own lead body is printed ("Q1. Explain …") immediately before
    # explicit b) c) siblings genuinely IS Q1(a); the body exists in source.
    implied_a: List[str] = []
    for parent in sorted(parents_with_subs):
        subs = {
            qid[len(parent) + 1 : -1].lower()
            for qid in ids
            if qid.upper().startswith(parent.upper() + "(")
        }
        if "a" in subs or "b" not in subs or parent not in ids:
            continue
        num = parent[1:]
        for ln in lines:
            m = re.match(rf"^(?:Q\.?|Question)?\s*{num}\s*([\.\):]?\s+)(.*)$", ln, re.I)
            if not m:
                continue
            remainder = (m.group(2) or "").strip()
            if (
                remainder
                and not is_choice_instruction(remainder)
                and not is_header_or_instruction(remainder)
                and _marker_has_body_evidence(f"{parent}(a)", ln, lines)
            ):
                implied_a.append(f"{parent}(a)")
                break
    ids = ids + [qid for qid in implied_a if qid not in ids]
    ids = drop_nested_roman_question_ids(ids)
    ids = drop_leaping_parent_ids(ids)
    ids = drop_leaping_sub_ids(ids)

    for qid in ids:
        reason = "ok"
        genuine = True
        matching = [ln for ln in lines if qid.lower() in ln.lower() or _line_hosts_marker(ln, qid)]
        line = matching[0] if matching else ""
        if re.search(r"\b\d+\s*(?:hours?|hrs?|mins?|minutes?)\b", line, re.I) and not re.search(
            rf"\bQ\.?\s*{re.search(r'\d+', qid).group(0)}\b", line, re.I
        ):
            genuine, reason = False, "duration_or_admin"
        elif line and is_header_or_instruction(line) and "(" not in qid:
            genuine, reason = False, "header_or_instruction"
        elif "(" not in qid and qid in parents_with_subs:
            genuine, reason = False, "parent_instruction_with_subs"
        elif line and not _marker_has_body_evidence(qid, line, lines):
            if _is_marker_only_line(line) or re.fullmatch(
                rf"Q{_PARENT_NUM}\({_SUB_TOKEN}\)\s*", line, re.I
            ):
                genuine, reason = True, "marker_only_line"
            else:
                genuine, reason = False, "marker_without_body_evidence"
        out.append({"marker_id": qid, "genuine": genuine, "reason": reason})
    return out


def _line_hosts_marker(line: str, qid: str) -> bool:
    m = re.match(rf"Q({_PARENT_NUM})(?:\(({_SUB_TOKEN})\))?$", qid, re.I)
    if not m:
        return False
    parent, sub = m.group(1), m.group(2)
    s = line.strip()
    if sub:
        # Sub must be a real token/delimiter, not the leading letter of
        # "Attempt" / "Answer" on a choice-parent line.
        if re.match(
            rf"^(?:Q\.?|Question)?\s*{parent}\s*[\.\):\-]?\s*"
            rf"(?:\({re.escape(sub)}\)|{re.escape(sub)}\s*[\.\):\-_]|{re.escape(sub)}\b(?![a-z]))",
            s,
            re.I,
        ):
            return True
        return bool(re.fullmatch(rf"\(?{re.escape(sub)}\)?[.\)]?", s, re.I))
    return bool(re.match(rf"^(?:Q\.?|Question)\s*{parent}\b", s, re.I))


def _marker_has_body_evidence(qid: str, line: str, lines: List[str]) -> bool:
    rest = re.sub(rf"^(?:Q\.?|Question)?\s*\d+\s*[\.\):\-]?\s*\(?[A-Za-zivx]+\)?\s*", "", line, flags=re.I)
    if any(re.search(rf"\b{v}\b", rest.lower()) for v in ACADEMIC_QUESTION_VERBS):
        return True
    if "?" in rest or re.search(r"\[\s*\d+\s*\]", rest):
        return True
    if len(rest.split()) >= 4 and not is_header_or_instruction(rest):
        return True
    # Same-line attached content is body evidence even when terse ("Corda",
    # "UTXO model of Bitcoin"): the printed marker and its item share the
    # line, so no lookahead is needed. Pure furniture remainders do not count.
    rest_clean = rest.strip()
    if (
        rest_clean
        and re.search(r"[A-Za-z]{3}", rest_clean)
        and not is_header_or_instruction(rest_clean)
        and not is_choice_instruction(rest_clean)
        and rest_clean.lower() not in {"attempt", "solve", "answer", "marks", "mark"}
    ):
        return True
    try:
        idx = lines.index(line)
    except ValueError:
        return False
    choice_frame = False
    for nxt in lines[idx + 1 : idx + 12]:
        if is_choice_instruction(nxt):
            choice_frame = True
        if is_header_or_instruction(nxt) or re.fullmatch(r"\[\s*\d+\s*\]", nxt):
            continue
        pm = re.match(rf"^(?:Q\.?|Question)\s*({_PARENT_NUM})\s*[\.\):]?\s*(.*)$", nxt, re.I)
        if pm:
            tail_after = (pm.group(2) or "").strip()
            sm2 = (
                re.match(rf"^\(({_SUB_TOKEN})\)\s*(.*)$", tail_after, re.I)
                or re.match(rf"^({_SUB_TOKEN})\s*[.\)]\s*(.*)$", tail_after, re.I)
            )
            if not sm2:
                # A true parent boundary ("Q3." / "Q4 Attempt the…") ends this
                # marker's block.
                break
            # "Qn(b) …" is a sibling list item, not a parent boundary.
            sib_body = (sm2.group(2) or "").strip()
            if not sib_body:
                continue
            if (
                choice_frame
                or len(sib_body.split()) >= 2
                or any(re.search(rf"\b{v}\b", sib_body.lower()) for v in ACADEMIC_QUESTION_VERBS)
            ):
                return True
            continue
        sm = re.match(rf"^\(?({_SUB_LETTER})\)?[.\)]\s*(.*)", nxt, re.I)
        if sm and nxt != line:
            # A sibling sub-marker carrying its own content is evidence of an
            # itemised list ("Write short notes: a. Corda / b. UTXO …"), not a
            # negation of this marker's body.
            sib_body = (sm.group(2) or "").strip()
            if not sib_body:
                continue
            if (
                choice_frame
                or len(sib_body.split()) >= 2
                or any(re.search(rf"\b{v}\b", sib_body.lower()) for v in ACADEMIC_QUESTION_VERBS)
            ):
                return True
            continue
        if any(re.search(rf"\b{v}\b", nxt.lower()) for v in ACADEMIC_QUESTION_VERBS) or len(nxt.split()) >= 4:
            return True
            continue
        if any(re.search(rf"\b{v}\b", nxt.lower()) for v in ACADEMIC_QUESTION_VERBS) or len(nxt.split()) >= 4:
            return True
    return bool(_is_marker_only_line(line) and any(
        (not is_header_or_instruction(nxt) and (
            any(re.search(rf"\b{v}\b", nxt.lower()) for v in ACADEMIC_QUESTION_VERBS)
            or len(nxt.split()) >= 4
        ))
        for nxt in lines[idx + 1 : idx + 4]
    ))


_ROMAN_ORDER = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
_ROMAN_TO_INT = {r: idx + 1 for idx, r in enumerate(_ROMAN_ORDER)}
_INT_TO_ROMAN = {idx + 1: r for idx, r in enumerate(_ROMAN_ORDER)}


def _sub_token_rank(sub: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Returns (style, rank) for a subquestion token string.
    Styles: 'alpha_lower', 'alpha_upper', 'digit', 'roman_lower'
    """
    sub_clean = (sub or "").strip()
    if not sub_clean:
        return None, None
    if len(sub_clean) == 1 and "a" <= sub_clean <= "z":
        return "alpha_lower", ord(sub_clean) - ord("a") + 1
    if len(sub_clean) == 1 and "A" <= sub_clean <= "Z":
        return "alpha_upper", ord(sub_clean) - ord("A") + 1
    if sub_clean.isdigit():
        return "digit", int(sub_clean)
    roman_sub = sub_clean.lower()
    if roman_sub in _ROMAN_TO_INT:
        return "roman_lower", _ROMAN_TO_INT[roman_sub]
    return None, None


def _rank_to_sub_token(style: str, rank: int) -> Optional[str]:
    """Converts rank back to subquestion token string for a given style."""
    if style == "alpha_lower":
        if 1 <= rank <= 26:
            return chr(ord("a") + rank - 1)
    elif style == "alpha_upper":
        if 1 <= rank <= 26:
            return chr(ord("A") + rank - 1)
    elif style == "digit":
        return str(rank)
    elif style == "roman_lower":
        return _INT_TO_ROMAN.get(rank)
    return None


def investigate_subquestion_marker_gaps(
    pages: List[Dict[str, Any]],
    extracted_questions: List[Dict[str, Any]],
    classified_markers: List[Dict[str, Any]],
    marker_candidates: List[Dict[str, Any]],
    rejected_candidates: List[Dict[str, Any]],
    evidence: Any,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Flexible subquestion marker range audit for every discovered parent.
    Allowed subquestion markers (a-i, i-vi, 1-5, A-E) define structure by
    observed markers on THAT parent only. A–I is a recognition range, not a
    required child set and not a Q6-only range.
    Gaps trigger targeted investigation across native text, OCR text, layout coordinates,
    reconstructed text, and cross-page evidence.
    Only source-proven missing items enter missing_questions.
    """
    all_qids: Set[str] = set()
    for q in extracted_questions:
        if q.get("question_id"):
            all_qids.add(str(q["question_id"]))
    for c in classified_markers:
        if c.get("genuine") and c.get("marker_id"):
            all_qids.add(str(c["marker_id"]))

    by_parent_and_style: Dict[Tuple[str, str], Dict[int, str]] = {}

    for qid in all_qids:
        m = re.match(r"(Q\d+)\(([^)]+)\)$", qid, re.I)
        if not m:
            continue
        parent = m.group(1).upper()
        sub = m.group(2).strip()
        style, rank = _sub_token_rank(sub)
        if style and rank is not None:
            by_parent_and_style.setdefault((parent, style), {})[rank] = qid

    proven_missing_ids: List[str] = []
    ambiguous_audits: List[Dict[str, Any]] = []

    source_texts: List[str] = []
    for p in pages:
        for k in ("raw_native_text", "raw_ocr_text", "raw_ocr_text_hd", "reconstructed_text"):
            txt = (p.get(k) or "").strip()
            if txt:
                source_texts.append(txt)

    full_source_blob = "\n".join(source_texts)
    parent_regions = split_text_into_parent_regions(full_source_blob)

    all_candidate_marker_ids = {
        str(mc.get("marker_id")) for mc in marker_candidates if mc.get("marker_id")
    }
    all_classified_marker_ids = {
        str(c.get("marker_id")) for c in classified_markers if c.get("marker_id")
    }
    all_rejected_marker_ids = {
        str(r.get("question_id"))
        for r in rejected_candidates
        if r.get("question_id")
        and str(r.get("reason") or "") not in NOISE_REASONS
        and str(r.get("reason") or "") != "text_too_short"
    }

    for (parent, style), rank_map in by_parent_and_style.items():
        if len(rank_map) < 2:
            continue
        sorted_ranks = sorted(rank_map.keys())
        min_rank = sorted_ranks[0]
        max_rank = sorted_ranks[-1]

        for gap_rank in range(min_rank, max_rank + 1):
            if gap_rank in rank_map:
                continue

            sub_token = _rank_to_sub_token(style, gap_rank)
            if not sub_token:
                continue

            gap_qid = f"{parent}({sub_token})"
            parent_num = re.search(r"\d+", parent).group(0)

            has_evidence = False

            # 1. Candidate / classified marker matches
            if (
                gap_qid in all_candidate_marker_ids
                or gap_qid in all_classified_marker_ids
            ):
                has_evidence = True

            # 2. Text evidence INSIDE this parent region only.
            # Unanchored "(c)" elsewhere in the paper (another parent, a table,
            # a diagram legend) is not proof that THIS parent printed (c).
            if not has_evidence:
                region = parent_regions.get(parent.upper()) or parent_region_text(
                    full_source_blob, parent
                )
                escaped_sub = re.escape(sub_token)
                parent_num = re.search(r"\d+", parent).group(0)
                if region and (
                    parent_scoped_sub_present(region, sub_token)
                    or re.search(
                        rf"\bQ\.?\s*{parent_num}\s*\({escaped_sub}\)", region, re.I
                    )
                    or re.search(
                        rf"\bQ\.?\s*{parent_num}\s*{escaped_sub}[\.\)\:\-]", region, re.I
                    )
                    or re.search(rf"\b{parent_num}\s*\({escaped_sub}\)", region, re.I)
                ):
                    has_evidence = True
                elif full_source_blob:
                    # Parent-qualified forms remain valid even if region split failed.
                    if re.search(
                        rf"\bQ\.?\s*{parent_num}\s*\({escaped_sub}\)", full_source_blob, re.I
                    ) or re.search(
                        rf"\b{parent_num}\s*\({escaped_sub}\)", full_source_blob, re.I
                    ):
                        has_evidence = True

            # 3. Layout coordinate evidence
            if not has_evidence:
                for p in pages:
                    layout_data = p.get("raw_ocr_layout")
                    if not layout_data:
                        continue
                    layout_str = str(layout_data)
                    escaped_sub = re.escape(sub_token)
                    if re.search(rf"\bQ\.?\s*{parent_num}\s*\({escaped_sub}\)", layout_str, re.I) or re.search(
                        rf"\b{parent_num}\s*\({escaped_sub}\)", layout_str, re.I
                    ):
                        has_evidence = True
                        break

            if has_evidence:
                proven_missing_ids.append(gap_qid)
            else:
                audit_entry = {
                    "question_id": gap_qid,
                    "status": "NOT-OBSERVED",
                    "reason": "alphabetical_or_numerical_gap_without_source_evidence",
                }
                ambiguous_audits.append(audit_entry)
                if evidence and hasattr(evidence, "ambiguous_markers"):
                    evidence.ambiguous_markers.append(audit_entry)

    return proven_missing_ids, ambiguous_audits


def compute_extraction_quality(
    extracted_ids: List[str],
    genuine_marker_ids: List[str],
    rejected: Optional[List[Dict[str, Any]]] = None,
    recovered_count: int = 0,
) -> Dict[str, Any]:
    """
    Completeness = genuine source markers vs extracted canonical IDs.
    Statuses: COMPLETE, RECOVERED, PARTIAL, FAILED.
    PARTIAL means genuine question boundaries were detected but not recovered.
    """
    extracted = list(dict.fromkeys(extracted_ids or []))
    noise = {
        str(r.get("question_id"))
        for r in (rejected or [])
        if r.get("question_id") and str(r.get("reason") or "") in NOISE_REASONS
    }
    genuine = [m for m in dict.fromkeys(genuine_marker_ids or []) if m not in noise]
    missing = [m for m in genuine if m not in set(extracted)]
    if not extracted:
        quality = "FAILED"
        confidence = 0.0
    elif not missing:
        if recovered_count > 0:
            quality = "RECOVERED"
        else:
            quality = "COMPLETE"
        confidence = min(0.99, 0.85 + 0.01 * len(extracted))
    else:
        quality = "PARTIAL"
        confidence = round(len(extracted) / (len(extracted) + len(missing)), 3)
    return {
        "questions_extracted": len(extracted),
        "source_markers_detected": len(genuine_marker_ids or []),
        "missing_questions": missing,
        "extraction_quality": quality,
        "confidence": round(confidence, 3),
    }


def validate_grounded_questions(
    candidates: List[Dict[str, Any]],
    source_blob: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for cand in candidates or []:
        exact = (cand.get("exact_text") or "").strip()
        ok, ratio, reason = text_grounded_in_source(exact, source_blob)
        if not ok:
            rejected.append({**cand, "reason": reason or "invented_or_ungrounded_text"})
            continue
        accepted.append({**cand, "grounding_score": round(ratio, 3)})
    return accepted, rejected


def score_page_representation(
    source_text: str,
    accepted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    *,
    word_quality_ratio: float = 0.0,
    geometry_support: float = 0.0,
) -> Dict[str, Any]:
    """Evidence scorecard. Character count and raw regex counts are not winners."""
    genuine_ids = [c["marker_id"] for c in classify_genuine_markers(source_text) if c.get("genuine")]
    genuine_set = set(genuine_ids)
    grounded_accepted = 0
    associated = 0
    inferred = 0
    header_hits = 0
    verb_hits = 0
    for q in accepted or []:
        exact = q.get("exact_text") or ""
        qid = q.get("question_id") or ""
        ok, _, _ = text_grounded_in_source(exact, source_text)
        if ok:
            grounded_accepted += 1
        if qid in genuine_set:
            associated += 1
        else:
            inferred += 1
        if is_header_or_instruction(exact):
            header_hits += 1
        if any(re.search(rf"\b{v}\b", exact.lower()) for v in ACADEMIC_QUESTION_VERBS):
            verb_hits += 1
    n_acc = max(1, len(accepted or []))
    association_rate = associated / max(1, len(genuine_set)) if genuine_set else (1.0 if not accepted else 0.0)
    noise_ratio = structural_ocr_noise_ratio(source_text)
    extracted_ids = [q.get("question_id") for q in (accepted or []) if q.get("question_id")]
    duplicate_ids = len(extracted_ids) - len(set(extracted_ids))
    score = (
        4.0 * grounded_accepted
        + 2.0 * associated
        + 1.5 * association_rate
        + 1.0 * float(word_quality_ratio or 0.0)
        + 0.5 * (verb_hits / n_acc)
        + 0.5 * float(geometry_support or 0.0)
        - 3.0 * inferred
        - 1.5 * (header_hits / n_acc)
        - 0.25 * len(rejected or [])
        - 2.0 * noise_ratio
        - 1.5 * duplicate_ids
    )
    return {
        "genuine_marker_ids": genuine_ids,
        "genuine_marker_count": len(genuine_set),
        "grounded_accepted": grounded_accepted,
        "associated_count": associated,
        "association_rate": round(association_rate, 4),
        "inferred_id_count": inferred,
        "rejected_count": len(rejected or []),
        "word_quality_ratio": float(word_quality_ratio or 0.0),
        "academic_verb_rate": round(verb_hits / n_acc, 4),
        "header_hit_rate": round(header_hits / n_acc, 4),
        "geometry_support": float(geometry_support or 0.0),
        "noise_ratio": noise_ratio,
        "duplicate_id_count": duplicate_ids,
        "score": round(score, 4),
    }


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
    Reject LLM/rule inventions: majority of significant tokens must appear in source.
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
    novel = [t for t in q_toks if t not in src and len(t) > 3]
    if novel and ratio < 0.85:
        return False, ratio, "invented_or_ungrounded_text"
    if ratio < min_token_overlap:
        return False, ratio, "invented_or_ungrounded_text"
    return True, ratio, "token_overlap"


def text_not_truncated_vs_span(exact_text: str, source_span: str) -> Tuple[bool, str]:
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
        marks = 0

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
    return []


def _strip_trailing_contamination(text: str) -> str:
    t = (text or "").strip()
    contamination_tail = re.compile(
        r"(?:\s+(?:Engineering|University|Paper\s*/\s*Subject|QP\s*CODE|Page\s+\d+|B\.E\.|Sem\s+[IVX]+).*)+$"
        r"|(?:\s+\d+\]?\s*\d{3,}\s*)+$"
        r"|(?:\s+10\]\s*\d+\s*)+$",
        re.I,
    )
    t2 = contamination_tail.sub("", t).strip()
    return t2 if len(t2) >= 12 else t


def continuation_score(prev_tail: str, next_line: str) -> float:
    """Layout-agnostic score: higher means next_line continues prev_tail."""
    tail = (prev_tail or "").rstrip()
    nxt = (next_line or "").strip()
    if not nxt:
        return 0.0
    if re.match(rf"^Q{_PARENT_NUM}(?:\(({_SUB_TOKEN})\))?\b", nxt, re.I):
        return 0.0
    if re.match(rf"^\(?{_SUB_TOKEN}\)?[\.\)]\s+\S+", nxt, re.I):
        return 0.0
    if is_header_or_instruction(nxt):
        return 0.0
    score = 0.0
    if tail and not tail.endswith((".", "?", "!")):
        score += 2.0
    if tail.endswith((",", ";", ":", "-", "(")):
        score += 1.5
    if re.match(r"^(and|or|the remaining|its|their|with|without|to|of|in|on)\b", nxt, re.I):
        score += 1.5
    if re.match(r"^[-•]\s+", nxt) or re.match(rf"^\({_SUB_TOKEN}\)\s+", nxt, re.I):
        score += 1.0
    if re.match(r"^(explain|what|discuss|describe|define|differentiate|compare|derive)\b", nxt, re.I):
        score -= 3.0
    if tail.endswith(("=", "+", "−", "×")):
        score += 1.5
    return score


def first_question_boundary_is_orphan_sub(prepared_text: str) -> bool:
    """True when the first exam marker on a page is a standalone letter/roman sub.

    Used so a page-break `b)` inherits the previous page's last parent instead of
    being dropped. Does not invent a parent when none exists.
    """
    if not prepared_text:
        return False
    parent_pat = re.compile(
        rf"^(?:Q\.?|Question)\s*{_PARENT_NUM}\b"
        rf"|^{_PARENT_NUM}\s*[\.\):]?\s*$"
        rf"|^{_PARENT_NUM}\s+(?:Attempt|Solve|any)\b"
        rf"|^{_PARENT_NUM}\s+{_SUB_LETTER}\b",
        re.I,
    )
    sub_pat = re.compile(rf"^\(?({_SUB_TOKEN})\)?[\.\)]\s*", re.I)
    sub_loose = re.compile(
        rf"^({_SUB_TOKEN})\s+(?:{'|'.join(sorted(ACADEMIC_QUESTION_VERBS))})\b",
        re.I,
    )
    letter_only = re.compile(rf"^\(?({_SUB_LETTER})\)?$", re.I)
    for ln in prepared_text.splitlines():
        ln = ln.strip()
        if not ln or _should_skip_line(ln) or is_header_or_instruction(ln):
            continue
        if parent_pat.match(ln):
            return False
        if sub_pat.match(ln) or sub_loose.match(ln) or letter_only.match(ln):
            return True
        if re.search(r"[A-Za-z]{4,}", ln):
            return False
    return False


def _last_parent_id(questions: List[Dict[str, Any]]) -> Optional[str]:
    for q in reversed(questions or []):
        parent = q.get("parent_question")
        if parent and re.fullmatch(rf"Q{_PARENT_NUM}", str(parent), re.I):
            return str(parent).upper()
        m = re.match(rf"(Q{_PARENT_NUM})", str(q.get("question_id") or ""), re.I)
        if m:
            return m.group(1).upper()
    return None


def leading_continuation_text(prepared_text: str) -> str:
    if not prepared_text:
        return ""
    lines = [ln.strip() for ln in prepared_text.splitlines() if ln.strip()]
    collected: List[str] = []
    for ln in lines:
        if re.match(rf"^Q{_PARENT_NUM}(?:\(({_SUB_TOKEN})\))?\b", ln, re.I):
            break
        if re.match(rf"^\(?{_SUB_TOKEN}\)?[\.\)]\s+\S+", ln, re.I):
            break
        if is_header_or_instruction(ln):
            continue
        if re.match(
            r"^(explain|what|discuss|describe|define|differentiate|compare|derive)\b",
            ln,
            re.I,
        ) and (not collected or continuation_score(" ".join(collected), ln) < 1.0):
            break
        if collected and continuation_score(" ".join(collected), ln) < 0:
            break
        collected.append(ln)
    if not collected:
        return ""
    return " ".join(collected).strip()


def run_universal_reconciliation_pipeline(
    pages_payload: List[Dict[str, Any]],
    *,
    filename: str,
    workspace_id: str,
    subject: str = "Subject",
    year: int = 0,
    syllabus_topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes the 9-stage Universal Question Reconciliation Architecture on DocumentEvidence:

    1. Representation Extraction (native, ocr_text, ocr_layout)
    2. Marker Candidate Detection
    3. Body Candidate Detection
    4. Marker ↔ Body Association
    5. Cross-Page Continuation Stitching
    6. Cross-Representation Reconciliation
    7. Grounding Verification
    8. Structural Validation
    9. Final Canonical Question Set & Genuine Boundary Reconciliation
    """
    evidence = DocumentEvidence(pages_payload)

    # -------------------------------------------------------------
    # Stage 1: Representation Extraction
    # -------------------------------------------------------------
    for p in pages_payload:
        page_num = int(p.get("page", 1))
        evidence.representations["native"][page_num] = p.get("raw_native_text") or ""
        evidence.representations["ocr_text"][page_num] = p.get("raw_ocr_text") or ""
        evidence.representations["ocr_text_hd"][page_num] = p.get("raw_ocr_hd_text") or ""
        evidence.representations["ocr_layout"][page_num] = p.get("reconstructed_text") or ""

    # Build combined source blob for global grounding
    all_blobs = []
    for p in pages_payload:
        for k in ("reconstructed_text", "raw_ocr_text", "raw_ocr_hd_text", "raw_native_text"):
            val = p.get(k) or ""
            if val.strip():
                all_blobs.append(val)
    source_blob = "\n\n".join(all_blobs)

    # -------------------------------------------------------------
    # Stage 2 & 3 & 4: Per-representation extraction & association
    # -------------------------------------------------------------
    candidates_by_repr: Dict[str, List[Dict[str, Any]]] = {
        "native": [],
        "ocr_text": [],
        "ocr_text_hd": [],
        "ocr_layout": [],
    }
    rejected_by_repr: Dict[str, List[Dict[str, Any]]] = {
        "native": [],
        "ocr_text": [],
        "ocr_text_hd": [],
        "ocr_layout": [],
    }

    _repr_field = {
        "native": "raw_native_text",
        "ocr_layout": "reconstructed_text",
        "ocr_text": "raw_ocr_text",
        "ocr_text_hd": "raw_ocr_hd_text",
    }
    _slot_state: Dict[str, Dict[str, Any]] = {}

    # Document-level evidence that this paper letter-splits its parents. Used
    # to open sequential slot inference under childless "marks each" parents
    # and to carry that context across page breaks. Computed from source
    # markers only — never from counts or subjects.
    _doc_conv_parents: Set[str] = set()
    for _p in pages_payload:
        for _field in _repr_field.values():
            _t = _p.get(_field)
            if not _t or not _t.strip():
                continue
            for _mid in detect_source_question_markers(_t):
                if "(" in _mid:
                    _doc_conv_parents.add(_mid.split("(", 1)[0].upper())
    _doc_lettered_convention = len(_doc_conv_parents) >= 2

    def _begins_with_marker(text: str) -> bool:
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            return bool(
                re.match(rf"^(?:{_Q_LEAD})\s*{_PARENT_NUM}\b", s, re.I)
                or re.match(rf"^{_PARENT_NUM}\s*[\.\)]?\s*\(", s, re.I)
                or re.match(rf"^\(?{_SUB_TOKEN}\)?[\.\):\-_]", s, re.I)
                or re.match(rf"^\(({_SUB_TOKEN})(?=\s|$)", s, re.I)
                or re.match(rf"^\[{_SUB_TOKEN}\]", s, re.I)
            )
        return True
    for repr_name in ("native", "ocr_layout", "ocr_text", "ocr_text_hd"):
        last_parent: Optional[str] = None
        last_sub: Optional[str] = None
        last_parent_page: int = 0
        last_body_text = ""
        for p in pages_payload:
            page_num = int(p.get("page", 1))
            text = p.get(_repr_field[repr_name])
            if not text or not text.strip():
                continue
            prepared = prepare_page_text_for_extraction(text)
            if not prepared.strip():
                continue

            # Detect markers for candidate auditing
            detected_markers = detect_source_question_markers(prepared)
            for m_id in detected_markers:
                evidence.marker_candidates.append(
                    marker_provenance_record(
                        marker_id=m_id,
                        raw_marker=m_id,
                        page=page_num,
                        representation=repr_name,
                        provenance=repr_name,
                    )
                )

            infer_ctx = None
            orphan_sub = bool(last_parent and first_question_boundary_is_orphan_sub(prepared))
            markerless_page = bool(
                _doc_lettered_convention
                and last_parent
                and page_num == last_parent_page + 1
                and not _begins_with_marker(prepared)
            )
            inherit = None
            if orphan_sub:
                inherit = last_parent
            elif markerless_page:
                inherit = last_parent
            if inherit and _doc_lettered_convention and last_sub and not markerless_page:
                nxt_letter = chr(ord(last_sub) + 1)
                if "a" <= nxt_letter <= "i":
                    infer_ctx = {
                        "enabled": True,
                        "start_sub": nxt_letter,
                        "anchor_parent": last_parent,
                    }
            elif (
                inherit
                and markerless_page
                and last_sub
                and slot_inference_justified(last_body_text)
            ):
                nxt_letter = chr(ord(last_sub) + 1)
                if "a" <= nxt_letter <= "i":
                    infer_ctx = {
                        "enabled": True,
                        "start_sub": nxt_letter,
                        "anchor_parent": last_parent,
                    }
            elif _doc_lettered_convention and not inherit:
                # Same-page opening: a childless "marks each" parent followed
                # by unlabelled bodies occupies sequential slots when this
                # document letter-splits other parents.
                infer_ctx = {"enabled": True}
            acc, rej = extract_questions_from_page_text(
                page_text=prepared,
                page_num=page_num,
                source_file=filename,
                workspace_id=workspace_id,
                subject=subject,
                year=year,
                syllabus_topics=syllabus_topics,
                inherit_parent=inherit,
                infer_slots=infer_ctx,
            )
            if acc:
                last_parent = _last_parent_id(acc) or last_parent
                tail_q = acc[-1]
                if tail_q.get("parent_question"):
                    last_parent = str(tail_q["parent_question"]).upper()
                    last_sub = (tail_q.get("subquestion") or "").lower() or None
                    last_parent_page = page_num
                    last_body_text = tail_q.get("exact_text") or ""
            for q in acc:
                q["source_pages"] = [page_num]
                q["extraction_method"] = repr_name
            candidates_by_repr[repr_name].extend(acc)
            rejected_by_repr[repr_name].extend(rej)
        _slot_state[repr_name] = {
            "parent": last_parent,
            "sub": last_sub,
            "page": last_parent_page,
        }

    # -------------------------------------------------------------
    # Stage 4.5: Cross-page sibling-slot inference
    # A page break can orphan a markerless imperative body that structurally
    # continues a parent's lettered subquestion list ("…(a) grid on page 1;
    # 'Describe …' on page 2"). When the document convention shows lettered
    # subs on other parents, the next sequential slot is recovered with
    # origin="inferred_stem" — never presented as a printed marker.
    # -------------------------------------------------------------
    doc_has_lettered_subs = any(
        q.get("subquestion") and len(str(q.get("subquestion"))) == 1
        for qs in candidates_by_repr.values()
        for q in qs
    )
    if doc_has_lettered_subs and len(pages_payload) > 1:
        for repr_name in ("native", "ocr_text", "ocr_text_hd"):
            state = _slot_state.get(repr_name) or {}
            parent = state.get("parent")
            sub = state.get("sub")
            from_page = int(state.get("page") or 0)
            if not parent or not sub or from_page < 1 or from_page >= len(pages_payload):
                continue
            nxt = chr(ord(sub) + 1)
            if not ("a" <= nxt <= "i"):
                continue
            last_body = ""
            for q in candidates_by_repr.get(repr_name, []):
                if (
                    str(q.get("parent_question") or "").upper() == str(parent).upper()
                    and str(q.get("subquestion") or "").lower() == str(sub).lower()
                ):
                    last_body = q.get("exact_text") or last_body
            if not slot_inference_justified(last_body):
                continue
            next_payload = pages_payload[from_page]
            raw = (
                next_payload.get(_repr_field[repr_name])
                or next_payload.get("raw_ocr_text")
                or ""
            )
            if not raw.strip():
                continue
            # Anchor the lead-region boundary at the earliest occurrence of any
            # accepted candidate's opening words on this page — flat OCR emits
            # margin parent labels ("Q.5") out of visual order, so a naive
            # line-scan boundary is unreliable here.
            page_acc = [
                q for q in candidates_by_repr.get(repr_name, [])
                if int((q.get("source_pages") or [0])[0]) == from_page + 1
            ]
            boundary = len(raw)
            for q in page_acc:
                words = re.findall(r"[A-Za-z]+", (q.get("exact_text") or ""))[:5]
                if len(words) < 4:
                    continue
                pat = re.compile(r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b", re.I)
                m = pat.search(raw)
                if m:
                    boundary = min(boundary, m.start())
            if boundary <= 0:
                continue
            lead_region = raw[:boundary]
            frag = leading_continuation_text(lead_region)
            remainder = lead_region[len(frag):].strip() if frag else lead_region.strip()
            keep_lines = []
            for ln in remainder.splitlines():
                s = ln.strip()
                if not s:
                    continue
                if re.match(r"^paper\s*/\s*subject\s*code", s, re.I):
                    continue
                if re.fullmatch(rf"(?:Q\.?\s*)?{_PARENT_NUM}", s, re.I):
                    continue
                if re.fullmatch(r"[\d\s]+", s):
                    continue
                keep_lines.append(s)
            remainder = "\n".join(keep_lines).strip()
            first_word = (remainder.split() or [""])[0].lower().strip(":.,;()-'\"")
            if not remainder or first_word not in ACADEMIC_QUESTION_VERBS:
                continue
            extra, _extra_rej = extract_questions_from_page_text(
                page_text=remainder,
                page_num=from_page + 1,
                source_file=filename,
                workspace_id=workspace_id,
                subject=subject,
                year=year,
                syllabus_topics=syllabus_topics,
                inherit_parent=parent,
                infer_slots={"enabled": True, "start_sub": nxt, "anchor_parent": parent},
            )
            kept = []
            for q in extra:
                if q.get("parent_question", "").upper() != parent:
                    continue
                q["source_pages"] = [from_page + 1]
                q["extraction_method"] = repr_name
                q["slot_inferred"] = True
                kept.append(q)
            if kept:
                candidates_by_repr[repr_name].extend(kept)
                evidence.cross_page_links.append(
                    {
                        "representation": repr_name,
                        "slot_inference_parent": parent,
                        "inferred_ids": [q.get("question_id") for q in kept],
                    }
                )

    # -------------------------------------------------------------
    # Stage 5: Cross-Page Continuation Stitching
    # -------------------------------------------------------------
    for repr_name in ("ocr_layout", "native", "ocr_text"):
        qs = candidates_by_repr[repr_name]
        if len(pages_payload) > 1 and qs:
            rejs = rejected_by_repr[repr_name]
            recovered, rejs, consumed, tail_count = recover_truncated_page_tails(
                pages_payload,
                rejs,
                filename=filename,
                workspace_id=workspace_id,
                subject=subject,
                year=year,
                syllabus_topics=syllabus_topics,
            )
            qs.extend(recovered)
            qs, merges = merge_cross_page_continuations(qs, pages_payload, skip_pages=consumed)
            merges += tail_count
            if merges > 0:
                evidence.cross_page_links.append(
                    {"representation": repr_name, "merges_count": merges}
                )
            candidates_by_repr[repr_name] = qs
            rejected_by_repr[repr_name] = rejs

    # Optional LLM Candidate Enhancement (Deterministic pipeline active)
    llm_raw: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # Stage 6: Cross-Representation Reconciliation
    # Synthesize the best grounded candidate per question ID across representations
    # -------------------------------------------------------------
    all_cands: Dict[str, List[Dict[str, Any]]] = {}
    for repr_name, qs in candidates_by_repr.items():
        for q in qs:
            qid = q.get("question_id")
            if qid:
                all_cands.setdefault(qid, []).append(q)

    for q in llm_raw:
        qid = q.get("question_id")
        if qid:
            all_cands.setdefault(qid, []).append(q)

    reconciled_set: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []

    def sort_key(q: Dict[str, Any]):
        m = re.match(r"Q(\d+)(?:\((\w+)\))?", q.get("question_id", ""), re.I)
        if not m:
            return (999, "z")
        return (int(m.group(1)), (m.group(2) or "").lower())

    for qid in sorted(all_cands.keys(), key=lambda k: sort_key({"question_id": k})):
        cands = all_cands[qid]
        known_parents = {
            str(c.get("parent_question") or "").upper()
            for cs in all_cands.values()
            for c in cs
            if c.get("parent_question")
        }
        all_ids_for_fusion = list(all_cands.keys())

        def _candidate_score(cand: Dict[str, Any]) -> float:
            exact = cand.get("exact_text") or ""
            score = 0.0
            if any(re.search(rf"\b{v}\b", exact.lower()) for v in ACADEMIC_QUESTION_VERBS):
                score += 100.0
            grounded, ratio, _ = text_grounded_in_source(exact, source_blob)
            if grounded:
                score += ratio * 200.0
            else:
                score -= 500.0
            # Layout reconstruction binds marker to body by GEOMETRY (marker
            # column / body column), so it is immune to the plain-OCR
            # reading-order artifact that re-attributes orphaned bodies to
            # whichever gutter parent was read last. Prefer it over flat-text
            # passes whenever both produced candidates for the same ID.
            if cand.get("extraction_method") == "ocr_layout":
                score += 120.0
            elif cand.get("extraction_method") == "native":
                score += 20.0
            # Length is a cap, never the rank key.
            score += min(len(exact), 800) / 800.0
            # A parent's own choice-instruction frame ("Write short notes on…",
            # "Attempt any four…") is not item content and must lose to any
            # real subquestion body for the same ID.
            from rag.question_extractor import is_instruction_frame_text as _is_frame

            if _is_frame(exact):
                score -= 400.0
            # Multi-signal evidence fusion: bounded tie-break only. Grounding
            # and frame penalties above always dominate the ranking.
            fused = fuse_candidate_evidence(
                cand,
                source_blob=source_blob,
                all_ids=all_ids_for_fusion,
                known_parents=known_parents,
            )
            score += fusion_bonus(fused["confidence"])
            cand["_fusion"] = fused
            return score

        best_cand = max(cands, key=_candidate_score)
        exact = (best_cand.get("exact_text") or "").strip()

        # -------------------------------------------------------------
        # Stage 7 & 8: Grounding Verification & Structural Validation
        # -------------------------------------------------------------
        grounded, ratio, g_reason = text_grounded_in_source(exact, source_blob)
        # Items under an explicit instruction frame ("Write short notes on…")
        # may legitimately be terse topic names ("Ripple", "Corda"); only the
        # instruction-frame flag from structural parsing can unlock this.
        under_instruction = bool(best_cand.get("under_instruction_parent"))
        min_len = 3 if under_instruction else 8
        if not grounded or len(exact) < min_len:
            ambiguous.append(
                {
                    "question_id": qid,
                    "reason": g_reason or "ungrounded_candidate",
                    "exact_text": exact,
                    "candidates_count": len(cands),
                }
            )
            continue

        exact = _strip_trailing_contamination(exact)
        best_cand["exact_text"] = exact
        best_cand["grounding_score"] = round(ratio, 3)
        best_cand["evidence_fusion"] = best_cand.pop("_fusion", None) or fuse_candidate_evidence(
            {
                **best_cand,
                "question_id": qid,
                "source_pages": best_cand.get("source_pages"),
            },
            source_blob=source_blob,
            all_ids=list(all_cands.keys()),
            known_parents=known_parents,
        )
        evidence.representation_sources[qid] = best_cand.get("extraction_method", "hybrid")
        reconciled_set.append(best_cand)

    # -------------------------------------------------------------
    # Stage 9: Final Canonical Question Set & Boundary Reconciliation
    # -------------------------------------------------------------
    evidence.reconciled_questions = reconciled_set
    evidence.ambiguous_markers = ambiguous

    # Collect all rejected candidates across representations
    all_rejected: List[Dict[str, Any]] = []
    for repr_name, rejs in rejected_by_repr.items():
        for r in rejs:
            all_rejected.append({**r, "representation": repr_name})

    # Drop OCR-leap artifacts and parent instruction rows. Do not invent letter gaps.
    cleaned_reconciled = []
    reconciled_parents = {q["parent_question"] for q in reconciled_set if q.get("parent_question")}

    from rag.question_extractor import _LETTER_SIBLINGS

    for parent in sorted(reconciled_parents):
        parent_qs = [q for q in reconciled_set if q.get("parent_question") == parent]
        subs = [q.get("subquestion", "").lower() for q in parent_qs if q.get("subquestion")]
        alpha_subs = sorted([s for s in subs if len(s) == 1 and s.isalpha()])

        valid_subs = set()
        if alpha_subs:
            current_max = ord(alpha_subs[0])
            valid_subs.add(alpha_subs[0])
            for s in alpha_subs[1:]:
                code = ord(s)
                if s == "q" and code - current_max > 1:
                    evidence.ambiguous_markers.append({
                        "question_id": f"{parent}({s})",
                        "reason": "isolated_sequence_leap_ocr_artifact",
                    })
                    all_rejected.append({
                        "question_id": f"{parent}({s})",
                        "reason": "isolated_sequence_leap_ocr_artifact",
                    })
                    continue
                if s in _LETTER_SIBLINGS or ("a" <= s <= "f") or (code - current_max <= 3):
                    valid_subs.add(s)
                    current_max = max(current_max, code)
                else:
                    evidence.ambiguous_markers.append({
                        "question_id": f"{parent}({s})",
                        "reason": "isolated_sequence_leap_ocr_artifact",
                    })
                    all_rejected.append({
                        "question_id": f"{parent}({s})",
                        "reason": "isolated_sequence_leap_ocr_artifact",
                    })

        has_subs = len(alpha_subs) > 0 or any(q.get("subquestion") for q in parent_qs)
        for q in parent_qs:
            sub = (q.get("subquestion") or "").lower()
            text = (q.get("exact_text") or "").lower()
            if not sub and has_subs:
                if is_header_or_instruction(text) or any(
                    w in text for w in ("attempt", "solve", "compulsory", "marks each", "following", "out of", "short notes")
                ):
                    continue
            if not sub or sub in valid_subs or not sub.isalpha() or len(sub) > 1:
                cleaned_reconciled.append(q)

    # Fabricated-slot protection: a subquestion slot ABOVE a parent's highest
    # printed sibling letter, carrying a data-corruption signature (tag-like
    # tokens, corpus/pipe/brace debris) instead of question prose, is an OCR
    # reading-order leak — demote it to the audit trail.
    def _data_signature(text: str) -> bool:
        hits = sum((text or "").count(c) for c in "<>{}|")
        hits += len(re.findall(r"</?[a-z]{1,3}>", text or ""))
        return hits >= 3

    def _data_leading(text: str) -> bool:
        head = (text or "").lstrip()[:48]
        if not head:
            return False
        hits = sum(head.count(c) for c in "<>{}|") + len(
            re.findall(r"</?[a-z]{1,3}>", head)
        )
        return hits >= 2

    # Trusted printed frontier per parent: highest letter in a leap-bounded
    # run of marker-origin records. Isolated far letters (h after b) do not
    # raise the frontier.
    letters_by_parent: Dict[str, List[str]] = {}
    for q in cleaned_reconciled:
        par = str(q.get("parent_question") or "").upper()
        sb = (q.get("subquestion") or "").lower()
        if (
            q.get("origin") == "marker"
            and len(sb) == 1
            and sb.isalpha()
            and not _data_signature(q.get("exact_text") or "")
        ):
            letters_by_parent.setdefault(par, []).append(sb)
    explicit_max_letter: Dict[str, str] = {}
    for par, letters in letters_by_parent.items():
        ordered = sorted(set(letters))
        run_max = ordered[0]
        for s in ordered[1:]:
            if ord(s) - ord(run_max) <= 3:
                run_max = s
        explicit_max_letter[par] = run_max

    demoted: List[Dict[str, Any]] = []
    for q in list(cleaned_reconciled):
        par = str(q.get("parent_question") or "").upper()
        sb = (q.get("subquestion") or "").lower()
        frontier = explicit_max_letter.get(par)
        if not (len(sb) == 1 and sb.isalpha()):
            continue
        demote_reason: Optional[str] = None
        if frontier and sb > frontier and _data_leading(q.get("exact_text") or ""):
            demote_reason = "data_leak_slot_beyond_printed_siblings"
        elif frontier:
            # Beyond the printed frontier only ONE adjacent continuation slot
            # is structurally plausible without independent corroboration.
            allowed_adjacent = chr(ord(frontier) + 1)
            qid_s = str(q.get("question_id") or "")
            sup = {
                str(mc.get("representation"))
                for mc in evidence.marker_candidates
                if mc.get("marker_id") == qid_s
            }
            if sb > allowed_adjacent and len(sup) < 2:
                demote_reason = "slot_beyond_printed_siblings_uncorroborated"
            elif ord(sb) - ord(frontier) > 4:
                demote_reason = "isolated_sequence_leap_ocr_artifact"
        if demote_reason:
            demoted.append({**q, "_reason": demote_reason})
            cleaned_reconciled.remove(q)
    for q in demoted:
        evidence.ambiguous_markers.append({
            "question_id": q.get("question_id"),
            "reason": q["_reason"],
            "exact_text": (q.get("exact_text") or "")[:160],
        })
        all_rejected.append({
            "question_id": q.get("question_id"),
            "reason": q["_reason"],
        })

    # Genuine markers = union across representations. Classify each
    # representation as one document so a page-leading "b)" still belongs
    # to the previous page's parent. Per-page classification would drop it.
    classified: List[Dict[str, Any]] = []
    for k in ("reconstructed_text", "raw_ocr_text", "raw_ocr_hd_text", "raw_native_text"):
        parts = [
            (p.get(k) or "")
            for p in sorted(pages_payload, key=lambda x: int(x.get("page", 1)))
            if (p.get(k) or "").strip()
        ]
        if parts:
            classified.extend(classify_genuine_markers("\n\n".join(parts)))
    genuine_ids = list(
        dict.fromkeys(c["marker_id"] for c in classified if c.get("genuine"))
    )
    genuine_ids = drop_nested_roman_question_ids(genuine_ids)
    genuine_ids = drop_leaping_parent_ids(genuine_ids)
    genuine_ids = drop_leaping_sub_ids(genuine_ids)
    child_parents = {
        re.match(r"(Q\d+)", qid, re.I).group(1).upper()
        for qid in genuine_ids
        if "(" in qid and re.match(r"(Q\d+)", qid, re.I)
    }
    genuine_ids = [
        qid for qid in genuine_ids
        if "(" in qid or qid.upper() not in child_parents
    ]
    # -------------------------------------------------------------
    # Identity corroboration: how many independent representations saw each
    # marker ID. OCR reading order can attribute orphaned bodies to the wrong
    # gutter parent in ONE representation; cross-representation evidence and
    # duplicate-body analysis decide identity, never a single noisy pass.
    # -------------------------------------------------------------
    rep_support: Dict[str, Set[str]] = {}
    for mc in evidence.marker_candidates:
        mid = mc.get("marker_id")
        if mid:
            rep_support.setdefault(str(mid), set()).add(str(mc.get("representation")))

    genuine_set = set(genuine_ids)
    candidate_seen = set(all_cands.keys())
    # Classifier-blindness fallback: when the marker classifier found NO
    # genuine labels at all (unusual numbering the classifier cannot see),
    # grounded reconciled IDs are kept rather than wiping a valid extraction.
    # Fabrication risk stays bounded: every kept record already passed
    # structural parsing, grounding and validation gates.
    classifier_blind = not genuine_ids

    kept_reconciled: List[Dict[str, Any]] = []
    fabricated: List[Dict[str, Any]] = []
    # Slot-inference admission: reconstructed sibling slots are single-
    # representation BY DESIGN (OCR lost the printed marker). They are
    # admitted one adjacent letter at a time past the parent's printed
    # frontier, must carry question prose (never data debris), and keep
    # id_status="unverified_label" so downstream consumers see provenance.
    _inferred_frontier: Dict[str, str] = {}
    for _par, _letter in (explicit_max_letter or {}).items():
        _nxt = chr(ord(_letter) + 1)
        _inferred_frontier[_par] = _nxt if "a" <= _nxt <= "z" else ""
    for q in cleaned_reconciled:
        qid = str(q.get("question_id") or "")
        if qid in genuine_set:
            q["id_status"] = "labelled"
            kept_reconciled.append(q)
            continue
        if len(rep_support.get(qid, set())) >= 2:
            # Multiple independent representations produced this exact
            # grounded question; the classifier simply missed the label.
            q["id_status"] = "unverified_label"
            kept_reconciled.append(q)
            continue
        if classifier_blind:
            q["id_status"] = "unverified_label"
            kept_reconciled.append(q)
            continue
        _sb = (q.get("subquestion") or "").lower()
        _par = str(q.get("parent_question") or "").upper()
        _allowed = _inferred_frontier.get(_par)
        if _allowed is None:
            _allowed = "a"
            _inferred_frontier[_par] = "a"
        if (
            (q.get("origin") == "inferred_stem" or q.get("slot_inferred"))
            and len(_sb) == 1
            and _sb == _allowed
            and not _data_leading(q.get("exact_text") or "")
        ):
            q["id_status"] = "unverified_label"
            kept_reconciled.append(q)
            _nxt2 = chr(ord(_sb) + 1)
            _inferred_frontier[_par] = _nxt2 if "a" <= _nxt2 <= "z" else ""
            continue
        fabricated.append(q)
        evidence.ambiguous_markers.append({
            "question_id": qid,
            "reason": "inferred_id_not_in_genuine_markers",
        })
    cleaned_reconciled = kept_reconciled

    # -------------------------------------------------------------
    # Cross-ID duplicate-body attribution removal: the same body text under
    # two different IDs means one physical question was attributed twice
    # (gutter-first OCR reuses bodies under whichever parent was read last).
    # Comparison uses CONTENT words only — academic verbs and domain-generic
    # vocabulary are shared by genuinely different questions and must never
    # look like duplication. The better-supported attribution wins; the loser
    # falls back to any non-colliding candidate of its own before its ID may
    # be reported missing.
    # -------------------------------------------------------------
    from rag.question_extractor import STOPWORDS as _Q_STOP, GENERIC_DOMAIN_TERMS as _Q_DOMAIN

    def _toks(text: str) -> Set[str]:
        stop = set(ACADEMIC_QUESTION_VERBS) | set(_Q_STOP) | set(_Q_DOMAIN)
        # Short DIGIT tokens survive filtering: they are exactly what
        # distinguishes templated subquestions ("topic number 1" vs
        # "number 2") from a second OCR read of the same line.
        return {
            t
            for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if (len(t) > 2 or t.isdigit()) and t not in stop
        }

    def _fuzzy_same(ta: Set[str], tb: Set[str]) -> bool:
        if len(ta) < 3 or len(tb) < 3:
            # Too little content to judge; only near-exact wording counts.
            return bool(ta and tb and ta == tb)
        inter = len(ta & tb)
        if inter < 4:
            return False
        # Two attributions of ONE physical question agree on nearly all
        # content words from BOTH sides, even when OCR garbles them
        # differently. Templated siblings that differ in any distinguishing
        # token ("topic number 1" vs "number 2") drop below this bar.
        return inter / len(ta) >= 0.7 and inter / len(tb) >= 0.7

    def _anchor_rank(q: Dict[str, Any]) -> Tuple[int, int, int]:
        qid = str(q.get("question_id") or "")
        return (
            len(rep_support.get(qid, set())),
            1 if qid in genuine_set else 0,
            len(q.get("exact_text") or ""),
        )

    def _grounded_len(c: Dict[str, Any]) -> Tuple[int, int]:
        exact = c.get("exact_text") or ""
        grounded, _ratio, _r = text_grounded_in_source(exact, source_blob)
        return (1 if grounded else 0, len(exact))

    ordered = sorted(cleaned_reconciled, key=_anchor_rank, reverse=True)
    resolved: List[Dict[str, Any]] = []
    taken: List[Tuple[str, Set[str]]] = []
    for q in ordered:
        qid = str(q.get("question_id") or "")
        toks = _toks(q.get("exact_text") or "")
        collide_with = next((oid for oid, ot in taken if _fuzzy_same(toks, ot)), None)
        if collide_with is None:
            resolved.append(q)
            taken.append((qid, toks))
            continue

        other_sets = [ot for oid, ot in taken if oid != qid]
        alts = [
            c for c in (all_cands.get(qid) or [])
            if (c.get("exact_text") or "").strip()
            and not any(_fuzzy_same(_toks(c.get("exact_text") or ""), ot) for ot in other_sets)
        ]
        alt = max(alts, key=_grounded_len) if alts else None
        alt_exact = (alt.get("exact_text") or "").strip() if alt else ""
        min_len = 3 if alt and alt.get("under_instruction_parent") else 8
        if (
            alt is not None
            and alt_exact
            and _grounded_len(alt)[0]
            and len(alt_exact) >= min_len
            and not _fuzzy_same(_toks(alt_exact), toks)
        ):
            replacement = dict(alt)
            replacement["question_id"] = qid
            replacement["id_status"] = "labelled" if qid in genuine_set else "unverified_label"
            resolved.append(replacement)
            taken.append((qid, _toks(alt_exact)))
            evidence.ambiguous_markers.append({
                "question_id": qid,
                "reason": "duplicate_body_attribution_resolved",
                "collided_with": collide_with,
            })
        else:
            # No disambiguating alternative: keep the grounded body and make
            # the conflict visible in the audit instead of dropping the ID.
            resolved.append(q)
            taken.append((qid, toks))
            evidence.ambiguous_markers.append({
                "question_id": qid,
                "reason": "duplicate_body_attribution",
                "collided_with": collide_with,
            })

    resolved.sort(key=lambda q: sort_key(q))
    cleaned_reconciled = resolved

    # -------------------------------------------------------------
    # Raw-source parent anchoring: a question's PARENT number must have
    # evidence in the raw source text (native / OCR passes). An ID that
    # exists ONLY inside a single layout reconstruction — e.g. born from a
    # marks-column digit run or a mis-read gutter glyph — is a phantom and
    # is demoted to the ambiguous audit instead of being accepted.
    # -------------------------------------------------------------
    raw_blob = "\n".join(
        (p.get("raw_native_text") or "") + "\n" + (p.get("raw_ocr_text") or "") + "\n"
        + (p.get("raw_ocr_hd_text") or "")
        for p in pages_payload
    )
    raw_parent_anchors = {
        f"Q{m.group(1)}"
        for m in re.finditer(r"(?:Q\.?\s*)?([1-9]\d?)\s*[\.\):]?\s*(?:[A-Za-z(\[]|$)", raw_blob)
        if len(m.group(1)) <= 2
    }
    anchored: List[Dict[str, Any]] = []
    for q in cleaned_reconciled:
        qid = str(q.get("question_id") or "")
        pm = re.match(r"(Q\d+)", qid, re.I)
        parent_id = pm.group(1).upper() if pm else ""
        multi_rep = len(rep_support.get(qid, set())) >= 2
        if (
            parent_id
            and parent_id not in raw_parent_anchors
            and not multi_rep
            and qid not in genuine_set
            and not classifier_blind
        ):
            evidence.ambiguous_markers.append({
                "question_id": qid,
                "reason": "layout_only_parent_unanchored_in_raw_source",
            })
            continue
        anchored.append(q)
    cleaned_reconciled = anchored

    # -------------------------------------------------------------
    # Evidence-gated missing set: a never-extracted marker may only drive
    # PARTIAL when the evidence is corroborated — multiple representations
    # saw it, or an extraction candidate for it existed and that candidate
    # was not merely a duplicate-body attribution of another question.
    # Uncorroborated single-pass ghosts stay visible in the audit but must
    # not fail an otherwise complete paper.
    # -------------------------------------------------------------
    final_ids = {str(q.get("question_id") or "") for q in cleaned_reconciled}
    corroborated_genuine: List[str] = []
    for m in genuine_ids:
        if m in final_ids:
            corroborated_genuine.append(m)
            continue
        if len(rep_support.get(m, set())) >= 2 or m in candidate_seen:
            corroborated_genuine.append(m)
        else:
            evidence.ambiguous_markers.append({
                "question_id": m,
                "reason": "uncorroborated_single_representation_marker",
            })
    genuine_ids = corroborated_genuine

    reconciled_set = cleaned_reconciled
    reconciled_ids = [q["question_id"] for q in reconciled_set]

    # Targeted crop OCR recovery pass for missing genuine questions
    recovered_count = 0
    missing_targets = [m for m in genuine_ids if m not in set(reconciled_ids)]
    if missing_targets:
        for m_qid in list(missing_targets):
            m_match = re.match(r"(Q\d+)(?:\((.+)\))?", m_qid, re.I)
            if not m_match:
                continue
            parent = m_match.group(1).upper()
            parent_qs = [q for q in cleaned_reconciled if (q.get("parent_question") or "").upper() == parent]
            target_page = 1
            if parent_qs:
                target_page = (parent_qs[0].get("source_pages") or [1])[0]

            crop_text = crop_and_ocr_suspicious_region(
                filename, page_num=target_page, top_ratio=0.5, bottom_ratio=0.98
            )
            if not crop_text and "doc" in locals() and locals()["doc"]:
                crop_text = crop_and_ocr_suspicious_region(
                    locals()["doc"], page_num=target_page, top_ratio=0.5, bottom_ratio=0.98
                )

            if crop_text:
                prepared_crop = prepare_page_text_for_extraction(crop_text)
                crop_acc, _ = extract_questions_from_page_text(
                    page_text=prepared_crop,
                    page_num=target_page,
                    source_file=filename,
                    workspace_id=workspace_id,
                    subject=subject,
                    year=year,
                    syllabus_topics=syllabus_topics,
                )
                for c_q in crop_acc:
                    c_qid = c_q.get("question_id")
                    if c_qid and c_qid not in set(reconciled_ids):
                        c_exact = c_q.get("exact_text") or ""
                        grounded, ratio, _ = text_grounded_in_source(c_exact, crop_text + "\n" + source_blob)
                        # Instruction-frame items may legitimately be terse
                        # topic names ("Corda", "Quorum").
                        min_len = 3 if c_q.get("under_instruction_parent") else 8
                        if grounded and len(c_exact) >= min_len:
                            c_q["recovery_candidate"] = True
                            c_q["extraction_method"] = "crop_ocr_recovery"
                            c_q["id_status"] = "labelled"
                            cleaned_reconciled.append(c_q)
                            reconciled_ids.append(c_qid)
                            recovered_count += 1

    proven_missing_gaps, _ = investigate_subquestion_marker_gaps(
        pages=pages_payload,
        extracted_questions=cleaned_reconciled,
        classified_markers=classified,
        marker_candidates=evidence.marker_candidates,
        rejected_candidates=all_rejected,
        evidence=evidence,
    )
    for g_id in proven_missing_gaps:
        if g_id not in set(genuine_ids):
            genuine_ids.append(g_id)

    # Universal Missing Sibling Recovery Pass (Requirement 1-9)
    cleaned_reconciled, recovered_siblings = recover_missing_sibling_questions(
        pages_payload,
        cleaned_reconciled,
        filename=filename,
        workspace_id=workspace_id,
        subject=subject,
        year=year,
        source_blob=source_blob,
        syllabus_topics=syllabus_topics,
    )
    if recovered_siblings:
        for s_q in recovered_siblings:
            s_qid = s_q.get("question_id")
            if s_qid and s_qid not in set(genuine_ids):
                genuine_ids.append(s_qid)
            if s_q not in cleaned_reconciled:
                cleaned_reconciled.append(s_q)
        recovered_count += len(recovered_siblings)

    # Adaptive Failure Recovery Pass: if 0 questions were recovered so far,
    # but source text / candidates exist, trigger recovery before returning FAILED.
    if not cleaned_reconciled and (all_cands or source_blob.strip()):
        adaptive_recovered = run_adaptive_failure_recovery_pass(
            pages_payload,
            all_cands=all_cands,
            source_blob=source_blob,
            filename=filename,
            workspace_id=workspace_id,
            subject=subject,
            year=year,
            syllabus_topics=syllabus_topics,
        )
        if adaptive_recovered:
            cleaned_reconciled.extend(adaptive_recovered)
            recovered_count += len(adaptive_recovered)

    # Completeness corroboration gate (additive): a SUBQUESTION id drives
    # missing/PARTIAL status only when more than one representation detected
    # its marker, or the pipeline itself reconstructed it. A lone noisy
    # "c)"/"e)" fragment in one OCR pass is audit evidence, never proof that
    # a printed question existed.
    reconciled_id_set = {str(q.get("question_id") or "") for q in cleaned_reconciled}
    quality_genuine_ids: List[str] = []
    for g_id in genuine_ids:
        if "(" not in g_id:
            quality_genuine_ids.append(g_id)
            continue
        if len(rep_support.get(g_id, set())) >= 2 or g_id in reconciled_id_set:
            quality_genuine_ids.append(g_id)
            continue
        evidence.ambiguous_markers.append({
            "question_id": g_id,
            "reason": "single_representation_marker_not_proven",
        })

    reconciled_set = cleaned_reconciled
    reconciled_ids = [q["question_id"] for q in reconciled_set]
    quality_info = compute_extraction_quality(
        reconciled_ids, quality_genuine_ids or reconciled_ids, rejected=all_rejected, recovered_count=recovered_count
    )
    missing_genuine = quality_info["missing_questions"]
    evidence.missing_genuine_questions = missing_genuine
    quality = quality_info["extraction_quality"]
    confidence = quality_info["confidence"]
    n_rec = quality_info["questions_extracted"]

    # Enrich accepted questions with canonical fields
    enriched_questions = []
    for q in reconciled_set:
        exact = q["exact_text"]
        qid = q["question_id"]
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
            q["parent_marks"] = q.get("parent_marks") or 0
            q["marks_total"] = q.get("marks") or 0
            q.setdefault("id_status", "labelled")
            q.setdefault("visual_appendix", [])
            q["question_number"] = qid
            q["question_type_struct"] = "SINGLE"
            q["quality_score"] = float(q.get("confidence") or 0.9)
            q["syllabus_mapping"] = {
                "module": "Unmapped",
                "chapter": "Unmapped",
                "topic": concepts[0] if concepts else "Unmapped",
            }
        q.setdefault("source_span", exact)
        q.setdefault("parent_id", q.get("parent_question"))
        q.setdefault("grounding_status", "grounded")
        q.setdefault("candidate_status", "ADMITTED")
        q.setdefault("extraction_method", q.get("extraction_method") or "hybrid")
        fusion = q.get("evidence_fusion") or {}
        signals = fusion.get("signals") or {}
        q["marker_confidence"] = round(float(signals.get("marker", 0.9)), 3)
        q["body_confidence"] = round(float(signals.get("semantic", 0.9)), 3)
        q["layout_confidence"] = round(float(signals.get("layout", 0.9)), 3)
        q["grounding_confidence"] = round(float(q.get("grounding_score", 0.95)), 3)
        q["overall_confidence"] = round(float(q.get("quality_score") or fusion.get("confidence", 0.95)), 3)
        q.setdefault("bounding_region", q.get("bounding_region") or {"page": spans[0], "crop": "full_width"})
        q.setdefault("marker_provenance", q.get("extraction_method") or "hybrid")
        enriched_questions.append(q)

    # Optional LLM Candidate Enhancement (Deterministic pipeline active)
    llm_raw: List[Dict[str, Any]] = []

    # Format debug report audit records for per-rejected candidate details
    formatted_rejected = []
    for r in all_rejected:
        formatted_rejected.append({
            "marker": r.get("question_id") or r.get("marker_id") or "UNKNOWN",
            "source_span": r.get("source_span") or r.get("raw_text") or r.get("exact_text") or "",
            "body_span": r.get("exact_text") or r.get("body_evidence") or "",
            "representation": r.get("representation") or r.get("extraction_method") or "unknown",
            "confidence": float(r.get("confidence") or 0.0),
            "rejection_gate": r.get("rejection_gate") or r.get("gate") or "validation_gate",
            "exact_rejection_reason": r.get("reason") or r.get("exact_rejection_reason") or "unsupported_or_noisy_candidate",
        })

    parent_frontier_audit = build_parent_frontier_audit(
        accepted=enriched_questions,
        marker_candidates=evidence.marker_candidates,
        missing_genuine=missing_genuine,
        ambiguous_markers=evidence.ambiguous_markers,
        rejected=all_rejected,
    )
    evidence.parent_frontier_audit = parent_frontier_audit
    inferred_ids = [
        q["question_id"]
        for q in enriched_questions
        if q.get("origin") == "inferred_stem" or q.get("slot_inferred")
    ]
    observed_ids = list(dict.fromkeys(
        str(mc.get("marker_id"))
        for mc in evidence.marker_candidates
        if mc.get("marker_id")
    ))
    recovered_ids = [q["question_id"] for q in enriched_questions]
    rejected_noise_ids = [
        str(r.get("question_id") or r.get("marker") or "")
        for r in all_rejected
        if "leap" in str(r.get("reason") or "").lower()
        or "watermark" in str(r.get("reason") or "").lower()
        or "footer" in str(r.get("reason") or "").lower()
        or "noise" in str(r.get("reason") or "").lower()
        or "uncorroborated" in str(r.get("reason") or "").lower()
        or "beyond_printed" in str(r.get("reason") or "").lower()
    ]
    rejected_noise_ids = [x for x in dict.fromkeys(rejected_noise_ids) if x]

    debug_report = {
        "FILE": filename,
        "STATUS": quality,
        "RECOVERED_QUESTIONS": recovered_ids,
        "AMBIGUOUS_MARKERS": evidence.ambiguous_markers,
        "REJECTED_MARKERS": formatted_rejected,
        "MISSING_SOURCE_PROVEN_QUESTIONS": missing_genuine,
        "FABRICATED_IDS": [q.get("question_id") for q in fabricated],
        "DUPLICATES": quality_info.get("duplicate_id_count", 0),
        "PARENT_FRONTIER": format_parent_frontier_reports(parent_frontier_audit),
    }

    return {
        "accepted_questions": enriched_questions,
        "rejected_candidates": all_rejected,
        "evidence": evidence,
        "quality": {
            "questions_extracted": n_rec,
            "source_markers_detected": quality_info.get("source_markers_detected", len(genuine_ids)),
            "missing_questions": missing_genuine,
            "extraction_quality": quality,
            "confidence": round(float(confidence), 3),
            "genuine_markers": genuine_ids,
            "fabricated_ids": [q.get("question_id") for q in fabricated],
        },
        "extraction_audit": {
            "representations": {
                "native": {"pages": len(evidence.representations["native"])},
                "ocr_text": {"pages": len(evidence.representations["ocr_text"])},
                "ocr_text_hd": {"pages": len(evidence.representations["ocr_text_hd"])},
                "ocr_layout": {"pages": len(evidence.representations["ocr_layout"])},
            },
            "marker_candidates": evidence.marker_candidates,
            "reconciled_questions": [q["question_id"] for q in enriched_questions],
            "ambiguous_markers": evidence.ambiguous_markers,
            "rejected_markers": formatted_rejected,
            "missing_genuine_questions": missing_genuine,
            "observed_markers": observed_ids,
            "recovered_markers": recovered_ids,
            "inferred_candidates": inferred_ids,
            "source_proven_missing": missing_genuine,
            "rejected_noise": rejected_noise_ids,
            "parent_frontier_audit": parent_frontier_audit,
            "cross_page_merges": evidence.cross_page_links,
            "representation_sources": evidence.representation_sources,
            "debug_report": debug_report,
        },
        "debug_report": debug_report,
        "llm_raw": llm_raw,
    }


def run_adaptive_failure_recovery_pass(
    pages_payload: List[Dict[str, Any]],
    all_cands: Dict[str, List[Dict[str, Any]]],
    source_blob: str,
    filename: str,
    workspace_id: str,
    subject: str,
    year: int,
    syllabus_topics: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Adaptive Failure Recovery Pass:
    Triggered when initial extraction produces 0 questions, but candidate markers or
    academic question bodies exist in the source document.
    """
    recovered: List[Dict[str, Any]] = []

    # Pass 1: Admitting grounded candidates from all_cands
    for qid, cands in all_cands.items():
        for cand in cands:
            exact = (cand.get("exact_text") or "").strip()
            if not exact or len(exact) < 3:
                continue
            ok, ratio, _ = text_grounded_in_source(exact, source_blob)
            if ok:
                cand_copy = dict(cand)
                cand_copy["question_id"] = qid
                cand_copy["grounding_score"] = round(ratio, 3)
                cand_copy["id_status"] = "unverified_label"
                cand_copy["extraction_method"] = cand.get("extraction_method") or "adaptive_recovery"
                recovered.append(cand_copy)
                break

    if recovered:
        return recovered

    # Pass 2: Relaxed stem extraction across all representations
    for p in pages_payload:
        page_num = int(p.get("page", 1))
        text = p.get("reconstructed_text") or p.get("raw_ocr_text") or p.get("raw_native_text") or ""
        if not text or not text.strip():
            continue
        prepared = prepare_page_text_for_extraction(text)
        acc, _ = extract_questions_from_page_text(
            page_text=prepared,
            page_num=page_num,
            source_file=filename,
            workspace_id=workspace_id,
            subject=subject,
            year=year,
            syllabus_topics=syllabus_topics,
            infer_slots={"enabled": True},
        )
        for q in acc:
            q_exact = (q.get("exact_text") or "").strip()
            ok, ratio, _ = text_grounded_in_source(q_exact, source_blob)
            if ok and len(q_exact) >= 3:
                q["source_pages"] = [page_num]
                q["extraction_method"] = "adaptive_recovery_pass"
                q["id_status"] = "unverified_label"
                recovered.append(q)

    if recovered:
        return recovered

    # Pass 3: High-DPI crop OCR recovery
    for p in pages_payload:
        page_num = int(p.get("page", 1))
        crop_text = crop_and_ocr_suspicious_region(
            filename, page_num=page_num, top_ratio=0.05, bottom_ratio=0.95, dpi=300
        )
        if crop_text:
            prepared = prepare_page_text_for_extraction(crop_text)
            crop_acc, _ = extract_questions_from_page_text(
                page_text=prepared,
                page_num=page_num,
                source_file=filename,
                workspace_id=workspace_id,
                subject=subject,
                year=year,
                syllabus_topics=syllabus_topics,
            )
            for q in crop_acc:
                q_exact = (q.get("exact_text") or "").strip()
                ok, ratio, _ = text_grounded_in_source(q_exact, crop_text + "\n" + source_blob)
                if ok and len(q_exact) >= 3:
                    q["source_pages"] = [page_num]
                    q["extraction_method"] = "crop_ocr_adaptive_recovery"
                    q["id_status"] = "unverified_label"
                    recovered.append(q)

    return recovered


def recover_missing_sibling_questions(
    pages_payload: List[Dict[str, Any]],
    cleaned_reconciled: List[Dict[str, Any]],
    *,
    filename: str,
    workspace_id: str,
    subject: str,
    year: int,
    source_blob: str,
    syllabus_topics: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Universal Missing Sibling Recovery Mechanism (Requirement 1-9):
    Detects missing sibling gaps between valid subquestions under ANY parent
    (Q1, Q2, … Q10, Q20, …) — e.g. Qn(a), Qn(b), Qn(d), Qn(e) where Qn(c)
    has source text but a garbled marker.

    Searches the source layout lines between the preceding and succeeding sibling.
    If an unlabelled/garbled-marker line exists that carries genuine question body
    text or marks notation, recovers it under the missing sibling ID.

    Guards:
      - Never invents a question if no source text exists in the gap region.
      - Never extends past that parent's printed sibling frontier.
      - Never turns isolated OCR noise letters into questions without question prose.
      - Never uses the parent number or "attempt any N" to guess child count.
    """
    if not cleaned_reconciled or not pages_payload:
        return cleaned_reconciled, []

    existing_ids = {str(q.get("question_id") or "") for q in cleaned_reconciled}
    reconciled_parents = {str(q.get("parent_question") or "").upper() for q in cleaned_reconciled if q.get("parent_question")}

    blob_parts: List[str] = []
    for p in sorted(pages_payload, key=lambda x: int(x.get("page", 1))):
        blob_parts.append(
            p.get("reconstructed_text")
            or p.get("raw_native_text")
            or p.get("raw_ocr_text")
            or p.get("raw_ocr_hd_text")
            or ""
        )
    parent_regions = split_text_into_parent_regions("\n".join(blob_parts))

    recovered_siblings = []
    
    for par in sorted(reconciled_parents):
        if not par.startswith("Q"):
            continue
        parent_qs = [q for q in cleaned_reconciled if str(q.get("parent_question") or "").upper() == par]
        subs_map = {}
        for q in parent_qs:
            sb = (q.get("subquestion") or "").lower()
            if len(sb) == 1 and sb.isalpha():
                subs_map[sb] = q
        
        if len(subs_map) < 2:
            continue
            
        sorted_letters = sorted(subs_map.keys())
        min_ord = ord(sorted_letters[0])
        max_ord = ord(sorted_letters[-1])

        parent_n = re.search(r"\d+", par)
        parent_n = parent_n.group(0) if parent_n else ""
        
        # Check alphabetical gaps between min_ord and max_ord ONLY (never beyond max_ord)
        for code in range(min_ord + 1, max_ord):
            gap_letter = chr(code)
            if gap_letter in subs_map:
                continue
            gap_qid = f"{par}({gap_letter})"
            if gap_qid in existing_ids:
                continue

            prev_letter = chr(code - 1)
            next_letter = chr(code + 1)
            
            prev_q = subs_map.get(prev_letter)
            next_q = subs_map.get(next_letter)
            
            if not prev_q or not next_q:
                prev_q = max((subs_map[k] for k in subs_map if k < gap_letter), key=lambda q: (q.get("subquestion") or "").lower(), default=None)
                next_q = min((subs_map[k] for k in subs_map if k > gap_letter), key=lambda q: (q.get("subquestion") or "").lower(), default=None)

            if not prev_q or not next_q:
                continue

            prev_page = (prev_q.get("source_pages") or [1])[0]
            target_page = prev_page
            
            p_payload = next((p for p in pages_payload if int(p.get("page", 1)) == target_page), None)
            if not p_payload:
                continue
                
            page_text = (
                p_payload.get("reconstructed_text")
                or p_payload.get("raw_native_text")
                or p_payload.get("raw_ocr_text")
                or p_payload.get("raw_ocr_hd_text")
                or ""
            )
            search_text = parent_regions.get(par) or page_text
            
            prev_text_snippet = (prev_q.get("exact_text") or "")[:35].lower().strip()
            next_text_snippet = (next_q.get("exact_text") or "")[:35].lower().strip()
            
            lines = [ln.strip() for ln in search_text.splitlines() if ln.strip()]
            
            prev_idx = -1
            next_idx = -1
            for idx, ln in enumerate(lines):
                ln_low = ln.lower()
                if prev_idx < 0 and (prev_text_snippet in ln_low or (prev_q.get("subquestion") and ln_low.startswith(prev_q.get("subquestion")))):
                    prev_idx = idx
                if prev_idx >= 0 and next_idx < 0 and (next_text_snippet in ln_low or (next_q.get("subquestion") and ln_low.startswith(next_q.get("subquestion")))):
                    next_idx = idx
                    break
                    
            if prev_idx >= 0 and next_idx > prev_idx:
                gap_lines = lines[prev_idx + 1 : next_idx]
                scoped_gap = []
                for gl in gap_lines:
                    other = re.match(
                        rf"^(?:{_Q_LEAD})\s*({_PARENT_NUM})\b",
                        gl,
                        re.I,
                    )
                    if other and str(int(other.group(1))) != str(int(parent_n or 0)):
                        continue
                    scoped_gap.append(gl)
                candidate_text = " ".join(scoped_gap).strip()
                candidate_text = re.sub(r"^[a-z0-9\.\)\s_|\-]{1,5}\s*", "", candidate_text, flags=re.I)
                
                prev_exact = (prev_q.get("exact_text") or "").strip()
                if (not candidate_text or len(candidate_text) < 4) and prev_exact:
                    parts = re.split(r"(?:\b05\b|\b10\b|\[\d+\]|\bmarks?\b|\bM\])\s+", prev_exact, flags=re.I)
                    if len(parts) >= 2 and len(parts[-1].strip()) >= 4:
                        candidate_text = parts[-1].strip()
                        prev_q["exact_text"] = " ".join(parts[:-1]).strip()
                    else:
                        m_split = re.search(r"\s(\b(?:Role|Triggers|Types|Conversion|Describe|Explain|What|Why|How|Discuss|State|Define|List)\b.*$)", prev_exact)
                        if m_split and m_split.start() > 5:
                            candidate_text = m_split.group(1).strip()
                            prev_q["exact_text"] = prev_exact[: m_split.start()].strip()
                
                if candidate_text and len(candidate_text) >= 4:
                    grounded, ratio, _ = text_grounded_in_source(candidate_text, page_text + "\n" + source_blob)
                    is_valid_cand, reason, _ = validate_question_candidate(
                        candidate_text, under_instruction_parent=True
                    )
                    if grounded and (is_valid_cand or re.search(r"\[?\s*\d{1,2}\s*M?\s*\]?", candidate_text)):
                        rec_item = {
                            "question_id": gap_qid,
                            "parent_question": par,
                            "subquestion": gap_letter,
                            "exact_text": candidate_text,
                            "source_pages": [target_page],
                            "source_span": candidate_text,
                            "grounding_score": round(ratio, 3),
                            "id_status": "unverified_label",
                            "extraction_method": "missing_sibling_recovery",
                            "under_instruction_parent": True,
                            "marks": extract_marks(candidate_text),
                            "origin": "missing_sibling_recovery",
                        }
                        recovered_siblings.append(rec_item)
                        existing_ids.add(gap_qid)

    return cleaned_reconciled, recovered_siblings



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
    if len(pages) < 2 or not rejected:
        return [], rejected, set(), 0

    text_by_page = {
        int(p.get("page", 1)): (
            p.get("reconstructed_text") or p.get("raw_ocr_text") or p.get("raw_native_text") or ""
        )
        for p in pages
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
        fragment = ""
        for frag_source in (
            p.get("reconstructed_text"),
            p.get("raw_ocr_text"),
            p.get("raw_native_text"),
            p.get("raw_ocr_hd_text"),
        ):
            if not (frag_source or "").strip():
                continue
            fragment = leading_continuation_text(frag_source)
            if fragment:
                break
        if not fragment:
            continue
        target = prev_qs[-1]
        tail = (target.get("exact_text") or "").rstrip()
        first_frag = (fragment.split() or [""])[0]
        if continuation_score(tail, fragment) < 0.5 and tail.endswith((".", "?", "!", ":")):
            continue
        if first_frag and continuation_score(tail, fragment) < 0:
            continue
        target["exact_text"] = f"{tail} {fragment}".strip()
        spans = sorted({*(target.get("source_pages") or [page_no - 1]), page_no})
        target["source_pages"] = spans
        target["source_page_end"] = spans[-1]
        target["cross_page_merged"] = True
        merges += 1

    return questions, merges


def merge_prefer_longer(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for q in a + b:
        qid = q.get("question_id")
        if not qid:
            continue
        if qid not in by_id or len(q.get("exact_text") or "") > len(by_id[qid].get("exact_text") or ""):
            by_id[qid] = q

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
    Main document extraction entrypoint wrapper calling universal reconciliation pipeline.
    """
    res = run_universal_reconciliation_pipeline(
        pages,
        filename=filename,
        workspace_id=workspace_id,
        subject=subject,
        year=year,
        syllabus_topics=syllabus_topics,
    )

    accepted = res.get("accepted_questions") or []
    rejected = res.get("rejected_candidates") or []
    quality = res.get("quality") or {}
    evidence = res.get("evidence")
    audit = res.get("extraction_audit") or {}

    markers = detect_source_question_markers(
        "\n\n".join(
            (p.get("reconstructed_text") or p.get("raw_ocr_text") or p.get("raw_native_text") or "")
            for p in pages
        )
    )

    return {
        "accepted_questions": accepted,
        "rejected_candidates": rejected,
        "source_markers": markers,
        "quality": quality,
        "extraction_audit": audit,
        "llm_used": bool(res.get("llm_raw")),
        "llm_candidates": len(accepted),
        "llm_rejected_for_truncation": 0,
        "cross_page_merges": max(
            (int(x.get("merges_count") or 0) for x in (evidence.cross_page_links if evidence else [])),
            default=0,
        ),
        "grounding_coverage": round(
            sum(q.get("grounding_score", 0.0) for q in accepted) / max(1, len(accepted)), 3
        ) if accepted else 0.0,
        "source_blob_chars": sum(
            len(p.get("reconstructed_text") or p.get("raw_ocr_text") or p.get("raw_native_text") or "")
            for p in pages
        ),
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
    return


RECURRENCE_SYSTEM = """Classify the relationship between two university exam questions.
You receive the original texts AND deterministic semantic signatures.
Labels ONLY: EXACT_REPEAT | SEMANTIC_REPEAT | RELATED_TOPIC | DIFFERENT

Rules:
- EXACT_REPEAT: same question (minor wording/OCR differences only). You cannot invent EXACT_REPEAT.
- SEMANTIC_REPEAT: a student preparing one is essentially preparing the other (same entity, intent, requested output, constraints). Paraphrase OK.
- RELATED_TOPIC: same broad concept but a DIFFERENT exam ask (CNN vs RNN, prevention vs detection, applications vs explanation).
- DIFFERENT: no meaningful conceptual relationship.

Generic shared words (analytics, social, media, system, explain) are NOT enough.
Contradictory technical entities MUST NOT be SEMANTIC_REPEAT.
You cannot invent source content. You cannot create a repeat without evidence in the questions.
Prefer RELATED_TOPIC or DIFFERENT over a false SEMANTIC_REPEAT.
Return STRICT JSON: {"label": "...", "confidence": 0.0, "same_underlying_question": false, "same_entity": false, "same_intent": false, "same_requested_output": false, "same_constraints": false, "contradictory_entities": false, "reason": "..."}
"""


def llm_judge_question_pair(
    text_a: str,
    text_b: str,
    signature_a: Optional[Dict[str, Any]] = None,
    signature_b: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    return None
