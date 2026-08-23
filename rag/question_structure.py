"""
Generic Question-Structure Identification & Subquestion Segmentation Layer.

Provides universal identification of parent questions, instructions, subquestion
markers, and subquestion body boundaries without subject-specific or filename-specific rules.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_PARENT_NUM = r"(?:[1-9]\d?|100)"
_SUB_LETTER = r"[a-z]"
_SUB_UPPER = r"[A-Z]"
_SUB_ROMAN = r"(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)"
_SUB_DIGIT = r"(?:[1-9]|1\d|20)"
_SUB_TOKEN = rf"(?:{_SUB_LETTER}|{_SUB_ROMAN}|{_SUB_UPPER}|{_SUB_DIGIT})"

ACADEMIC_QUESTION_VERBS = {
    "explain", "describe", "discuss", "define", "compare", "differentiate",
    "distinguish", "write", "state", "list", "enumerate", "evaluate",
    "illustrate", "calculate", "compute", "derive", "prove", "design",
    "construct", "analyze", "analyse", "implement", "detail", "sketch",
    "draw", "show", "what", "why", "how", "where", "when", "which", "who",
    "give", "briefly", "elaborate", "formulate", "solve", "identify",
    "outline", "summarize", "summarise", "classify", "demonstrate",
}

INSTRUCTION_KEYWORDS = {
    "attempt", "solve", "answer", "write", "explain", "choose", "note",
    "following", "any", "four", "three", "two", "one", "five", "six",
}


def is_parent_instruction_line(text: str) -> bool:
    """Detect if line is a parent instruction header like 'Q6. Solve any Four'."""
    s = text.strip()
    m = re.match(rf"^(?:Q\.?|Question)?\s*({_PARENT_NUM})\s*[\.\):]?\s*(.*)$", s, re.I)
    if not m:
        return False
    rest = m.group(2).strip().lower()
    if not rest:
        return False
    words = set(re.findall(r"\b[a-z]+\b", rest))
    return len(words & INSTRUCTION_KEYWORDS) >= 2 or re.search(r"\b(?:attempt|solve|answer)\s+any\b", rest, re.I) is not None


def split_embedded_subquestions(parent_id: str, text: str) -> List[Dict[str, Any]]:
    """
    Split a block of text into subquestion records if embedded subquestion markers
    (e.g., 'a. Explain... b. Explain... c. Difference...') are present.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    # Regex for subquestion boundary: (a), a., a), Q1(a), 1(a), (i), i.
    sub_pattern = re.compile(
        rf"(?:^|[\n\r]|(?<=\s))(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})[\.\)]|{_PARENT_NUM}\(({_SUB_TOKEN})\))\s+",
        re.I,
    )

    matches = list(sub_pattern.finditer(text))
    if len(matches) < 2:
        return []

    sub_records: List[Dict[str, Any]] = []
    parent_num = parent_id.replace("Q", "").strip()

    for idx, match in enumerate(matches):
        raw_sub = match.group(1) or match.group(2) or match.group(3)
        if not raw_sub:
            continue
        sub_tok = raw_sub.lower()

        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment_text = text[start_pos:end_pos].strip()

        # Clean trailing marks or parent numbers
        segment_text = re.sub(r"\[\s*\d+\s*\]$", "", segment_text).strip()

        qid = f"Q{parent_num}({sub_tok})"
        if segment_text:
            sub_records.append({
                "question_id": qid,
                "question_number": qid,
                "parent_question": f"Q{parent_num}",
                "subquestion": sub_tok,
                "exact_text": segment_text,
                "origin": "subquestion_segmentation",
            })

    return sub_records
