"""
Layout-aware OCR reconstruction.

Plain `image_to_string` OCR discards geometry, so exam papers whose parent
question numbers sit in a left gutter lose their structure entirely (markers
and bodies end up in unrelated blocks). This module keeps word coordinates
and rebuilds `Qn(sub) text` lines from *positional evidence* only:

  - the marker column (where short sub-markers like "a)" live)
  - the body column (indented statement text)
  - letter-sequence progression
  - per-question marks tags

No subject, university, year, filename or question-count assumptions.
When the evidence is insufficient the functions return nothing so the caller
falls back to plain-text handling rather than inventing structure.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

from rag.question_extractor import (
    ACADEMIC_QUESTION_VERBS,
    _PARENT_NUM,
    _SUB_DELIMITED,
    _SUB_LETTER,
    _SUB_TOKEN,
    _normalize_subtoken,
    fix_ocr_question_glyphs,
    is_header_or_instruction,
)

# Optional junk between letter and delimiter covers "c')" / "c|)".
# A sub-marker is only recognised when it is delimited, so
# "1 Attempt any four" never parses as parent 1 + sub "a".
_MARKER_ONLY = re.compile(rf"^[qQ]?\(?({_SUB_TOKEN})\)?\s*[\.\)]?$", re.I)
_MARKER_LEAD = re.compile(rf"^[qQ]?{_SUB_DELIMITED}[\.\)]*\s+(.*)$", re.I)
_PARENT_SUB_LEAD = re.compile(
    rf"^({_PARENT_NUM})[\s\.\):]*{_SUB_DELIMITED}[\.\)]*\s*(.*)$", re.I
)
_PARENT_ONLY = re.compile(rf"^({_PARENT_NUM})\s*[\.\):]?$")
_PARENT_LEAD = re.compile(rf"^({_PARENT_NUM})[\s\.\):]+(\S.*)$")
# Marks tags sit at the end of a line OR after the first wrapped clause
# ("configuration: [10] -The input...", "5M]", "[10M]", "20[M]").
_MARKS_TAG = re.compile(
    r"\[\s*\d{1,2}\s*M?\s*\]|\b\d{1,2}\s*M\s*\]|\d{1,2}\s*\[M\]|(?<=\s)\d{1,2}\s*\]",
    re.I,
)
_MARKS_ONLY = re.compile(r"^\[?\s*\d{1,2}\s*M?\s*\]?$", re.I)
_SUB_ORDER = "abcdefghijklmnopqrstuvwxyz"

def ocr_page_lines(page, dpi: int = 150) -> List[Dict[str, Any]]:
    """
    OCR one PyMuPDF page keeping geometry.

    Returns visual lines: {text, x0, x1, top, bottom, height}.
    Empty list when tesseract/PIL are unavailable.
    """
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on local install
        print(f"[OCR_LAYOUT_INFO] pytesseract/PIL unavailable: {exc}")
        return []

    try:
        pix = page.get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        print(f"[OCR_LAYOUT_INFO] OCR failed: {exc}")
        return []

    words: List[Dict[str, Any]] = []
    count = len(data.get("text") or [])
    for i in range(count):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        words.append(
            {
                "text": txt,
                "x0": int(data["left"][i]),
                "x1": int(data["left"][i]) + int(data["width"][i]),
                "top": int(data["top"][i]),
                "bottom": int(data["top"][i]) + int(data["height"][i]),
                "height": int(data["height"][i]),
            }
        )
    if not words:
        return []

    heights = sorted(w["height"] for w in words)
    median_h = heights[len(heights) // 2] or 10
    tol = max(4, int(median_h * 0.6))

    words.sort(key=lambda w: (w["top"], w["x0"]))
    lines: List[Dict[str, Any]] = []
    for w in words:
        placed = False
        for ln in reversed(lines[-3:]):
            centre = (ln["top"] + ln["bottom"]) / 2
            w_centre = (w["top"] + w["bottom"]) / 2
            if abs(w_centre - centre) <= tol:
                ln["words"].append(w)
                ln["top"] = min(ln["top"], w["top"])
                ln["bottom"] = max(ln["bottom"], w["bottom"])
                placed = True
                break
        if not placed:
            lines.append({"words": [w], "top": w["top"], "bottom": w["bottom"]})

    out: List[Dict[str, Any]] = []
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws).strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "x0": ws[0]["x0"],
                "x1": ws[-1]["x1"],
                "top": ln["top"],
                "bottom": ln["bottom"],
                "height": max(1, ln["bottom"] - ln["top"]),
            }
        )
    out.sort(key=lambda l: l["top"])
    return out


def _modal_x(values: List[int], tol: int = 12) -> Optional[int]:
    """Most common left edge within a tolerance band."""
    if not values:
        return None
    best_x, best_n = None, 0
    for v in values:
        n = sum(1 for o in values if abs(o - v) <= tol)
        if n > best_n:
            best_x, best_n = v, n
    return best_x


def _first_group(m: "re.Match[str]", *idx: int) -> str:
    for i in idx:
        if m.group(i):
            return m.group(i)
    return ""


def _classify(text: str) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Classify a visual line.

    Returns (kind, parent, sub, remainder) where kind is one of
    parent_sub | parent_only | parent_lead | marker_only | marker_lead | marks | text.
    """
    t = fix_ocr_question_glyphs(text).strip()
    # "Q.2.a)" / "Q. 3 b)" / "Q.1. Any four" — the dot after Q is not a sub-marker.
    t = re.sub(r"^Q\s*[\.\s]+(?=\d)", "Q", t, flags=re.I)
    t = re.sub(r"^Q\s+(?=\d)", "Q", t)
    # Leading OCR junk before a letter marker: "�c Explain..." → "c Explain..."
    if not re.match(r"^Q\d", t, re.I):
        t = re.sub(r"^[^A-Za-z0-9(]+", "", t)

    if _MARKS_ONLY.match(t):
        return "marks", None, None, ""

    m = re.match(rf"^Q\s*({_PARENT_NUM})[\s\.\):]*{_SUB_DELIMITED}[\.\)]*\s*(.*)$", t, re.I)
    if m:
        sub = _first_group(m, 2, 3)
        rest = re.sub(r"^[\s|]+", "", (m.group(4) or "")).strip()
        return "parent_sub", f"Q{m.group(1)}", _normalize_subtoken(sub), rest

    m = re.match(rf"^Q\s*({_PARENT_NUM})\s*[\.\):]?\s*$", t, re.I)
    if m:
        return "parent_only", f"Q{m.group(1)}", None, ""

    m = re.match(rf"^Q\s*({_PARENT_NUM})[\s\.\):]+(\S.*)$", t, re.I)
    if m and not re.match(rf"^{_SUB_DELIMITED}", m.group(2) or "", re.I):
        return "parent_lead", f"Q{m.group(1)}", None, (m.group(2) or "").strip()

    if not re.match(r"^\d+\s*[x*×]\s*\d+", t):
        m = _PARENT_SUB_LEAD.match(t)
        if m:
            sub = _first_group(m, 2, 3)
            return "parent_sub", f"Q{m.group(1)}", _normalize_subtoken(sub), (m.group(4) or "").strip()

    m = _PARENT_ONLY.match(t)
    if m:
        return "parent_only", f"Q{m.group(1)}", None, ""

    m = _MARKER_ONLY.match(t)
    if m:
        return "marker_only", None, _normalize_subtoken(m.group(1)), ""

    m = _MARKER_LEAD.match(t)
    if m:
        sub = _first_group(m, 1, 2)
        return "marker_lead", None, _normalize_subtoken(sub), (m.group(3) or "").strip()

    # Bare "b Consider ..." — marker letter lost its bracket during OCR
    m = re.match(rf"^({_SUB_LETTER})\s+([A-Z].*)$", t)
    if m:
        return "marker_lead", None, _normalize_subtoken(m.group(1)), m.group(2).strip()

    # Gutter parent line carrying instruction/statement text ("1 Attempt any four")
    m = _PARENT_LEAD.match(t)
    if m:
        return "parent_lead", f"Q{m.group(1)}", None, (m.group(2) or "").strip()

    return "text", None, None, t


def _is_continuation(text: str) -> bool:
    """Wrapped line rather than a new exam item."""
    t = text.strip()
    if not t:
        return True
    if t.startswith("-") or t.startswith("\u2022"):
        return True
    if t[0].islower():
        return True
    return False


def _is_page_artifact(text: str) -> bool:
    """Footer/QP-code style lines: mostly digits or too short to be a question."""
    t = text.strip()
    if not t:
        return True
    if re.search(r"page\s+\d+\s+of\s+\d+", t, re.I):
        return True
    letters = sum(1 for c in t if c.isalpha())
    digits = sum(1 for c in t if c.isdigit())
    if digits and letters <= 2:
        return True
    if len(t) <= 3:
        return True
    return False


def _next_letter(sub: Optional[str]) -> str:
    if not sub or sub not in _SUB_ORDER:
        return "a"
    idx = _SUB_ORDER.index(sub)
    return _SUB_ORDER[min(idx + 1, len(_SUB_ORDER) - 1)]


def _retain_marks(text: str) -> str:
    """Keep a canonical [n] tag so later extract_marks can read it."""
    m = _MARKS_TAG.search(text or "")
    n = None
    if m:
        digits = re.search(r"\d{1,2}", m.group(0) or "")
        if digits:
            n = digits.group(0)
    cleaned = _MARKS_TAG.sub("", text or "").strip()
    if n and f"[{n}]" not in cleaned:
        cleaned = f"{cleaned} [{n}]".strip()
    return cleaned


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def reconstruct_questions_from_layout(lines: List[Dict[str, Any]]) -> str:
    """
    Rebuild "Qn(sub) text" lines from OCR geometry.

    Returns "" when positional evidence is too weak (caller keeps plain text).
    """
    marked = _reconstruct_from_markers(lines)
    if marked:
        return marked
    return _reconstruct_unlabelled_bodies(lines)


_STRONG_ITEM_START = re.compile(
    r"^(explain|what|discuss|describe|define|differentiate|compare|derive|outline)\b",
    re.I,
)


def _is_question_like(text: str) -> bool:
    t = (text or "").strip()
    if not t or is_header_or_instruction(t) or _is_page_artifact(t):
        return False
    low = t.lower()
    if any(re.search(rf"\b{re.escape(v)}\b", low) for v in ACADEMIC_QUESTION_VERBS):
        return True
    if "?" in t:
        return True
    return False


def _remaining_parent_count(lines: List[Dict[str, Any]]) -> Optional[int]:
    blob = " ".join(ln.get("text") or "" for ln in lines)
    m = re.search(r"remaining\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)", blob, re.I)
    if not m:
        return None
    raw = m.group(1).lower()
    if raw.isdigit():
        n = int(raw)
        return n if 2 <= n <= 20 else None
    return _NUM_WORDS.get(raw)


def _reconstruct_unlabelled_bodies(lines: List[Dict[str, Any]]) -> str:
    """
    OCR sometimes drops every Q/a) glyph but leaves academic question bodies
    and marks tags. Recover items from instruction + marks + verb evidence.
    Do not invent wording; IDs come from document order and remaining-N pairing.
    """
    if len(lines) < 5:
        return ""

    started = False
    items: List[Tuple[str, bool]] = []
    buf: List[str] = []
    in_marked = False

    def flush_item(marked: bool):
        nonlocal buf
        if buf:
            joined = _retain_marks(" ".join(x.strip() for x in buf if x.strip()))
            if joined and not is_header_or_instruction(joined):
                items.append((joined, marked or bool(_MARKS_TAG.search(joined))))
        buf = []

    for ln in lines:
        raw = (ln.get("text") or "").strip()
        if not raw or _is_page_artifact(raw) or re.fullmatch(r"[\W_]+", raw):
            continue
        if is_header_or_instruction(raw):
            if re.search(r"attempt\s+any", raw, re.I) or re.search(
                r"question\s+no\.?\s*1\b", raw, re.I
            ):
                started = True
            continue
        marked = bool(_MARKS_TAG.search(raw))
        like = _is_question_like(raw)
        if not started:
            if like:
                started = True
            else:
                continue
        # Left-gutter OCR junk ("acre", lone glyphs) sitting between questions
        if (
            not marked
            and not like
            and len(raw.split()) <= 2
            and len(raw) <= 12
            and (not buf or buf[-1].rstrip().endswith(("?", ".", ":", "]")))
        ):
            continue
        if marked:
            flush_item(in_marked)
            in_marked = True
            buf = [raw]
            continue
        if in_marked:
            buf.append(raw)
            continue
        if like and not _is_continuation(raw):
            flush_item(False)
            buf = [raw]
            continue
        if buf:
            buf.append(raw)

    flush_item(in_marked)
    if len(items) < 3:
        return ""

    shorts = [t for t, m in items if not m]
    longs = [t for t, m in items if m]
    if not longs and len(shorts) < 3:
        return ""

    slots: List[Tuple[str, str, str]] = []
    if shorts:
        for i, text in enumerate(shorts):
            slots.append(("Q1", _SUB_ORDER[i] if i < len(_SUB_ORDER) else "a", text))
        parent_start = 2
    else:
        parent_start = 1

    remaining = _remaining_parent_count(lines)
    pair = bool(remaining and remaining >= 2 and len(longs) == remaining * 2)
    if pair:
        for i, text in enumerate(longs):
            parent = parent_start + (i // 2)
            sub = "a" if i % 2 == 0 else "b"
            slots.append((f"Q{parent}", sub, text))
    else:
        for i, text in enumerate(longs):
            slots.append((f"Q{parent_start + i}", "a", text))

    parents = {s[0] for s in slots}
    if len(slots) < 3 or len(parents) < 2:
        return ""
    ids = [f"{p}({s})" for p, s, _t in slots]
    if len(set(ids)) != len(ids):
        return ""
    return "\n".join(f"{p}({s}) {t}" for p, s, t in slots)


def _reconstruct_from_markers(lines: List[Dict[str, Any]]) -> str:
    """
    Rebuild "Qn(sub) text" lines from OCR geometry.

    Returns "" when positional evidence is too weak (caller keeps plain text).
    """
    if len(lines) < 5:
        return ""

    body = [
        ln
        for ln in lines
        if ln["text"].strip()
        and not is_header_or_instruction(ln["text"])
        and not re.fullmatch(r"[\W_]+", ln["text"])
    ]
    if len(body) < 5:
        return ""

    # The sub-marker column is defined by markers that stand alone or lead a
    # line ("a)", "b. Explain ..."). Lines that carry their own parent number
    # sit further left in the gutter and must not skew this measurement.
    marker_x = _modal_x(
        [
            ln["x0"]
            for ln in body
            if _classify(ln["text"])[0] in ("marker_only", "marker_lead")
        ]
    )
    if marker_x is None:
        marker_x = _modal_x(
            [ln["x0"] for ln in body if _classify(ln["text"])[0] == "parent_sub"]
        )
    if marker_x is None:
        marker_x = _modal_x(
            [
                ln["x0"]
                for ln in body
                if _classify(ln["text"])[0] in ("parent_lead", "parent_only")
            ]
        )
    if marker_x is None:
        return ""

    text_x_values = [
        ln["x0"]
        for ln in body
        if _classify(ln["text"])[0] == "text" and ln["x0"] > marker_x + 8
    ]
    body_x = _modal_x(text_x_values)

    classified = [(ln, _classify(ln["text"])) for ln in body]

    slots: List[Tuple[str, str, str]] = []
    parent_num = 0
    current_sub: Optional[str] = None
    buf: List[str] = []
    seen_any_marker = False
    parent_uses_letter_markers = False

    def flush():
        nonlocal buf
        if parent_num >= 1 and current_sub and buf:
            joined = " ".join(x.strip() for x in buf if x.strip()).strip()
            joined = _retain_marks(joined)
            if joined:
                slots.append((f"Q{parent_num}", current_sub, joined))
        buf = []

    def next_marker_sub(start: int) -> Optional[str]:
        for ln2, (kind2, _p2, sub2, _r2) in classified[start + 1 : start + 6]:
            if kind2 in ("marker_only", "marker_lead") and abs(ln2["x0"] - marker_x) <= 24:
                return sub2
            if kind2 in ("parent_sub", "parent_only", "parent_lead"):
                return None
        return None

    for idx, (ln, (kind, parent, sub, rest)) in enumerate(classified):
        raw = ln["text"].strip()

        if kind == "marks":
            continue

        if kind == "parent_sub":
            flush()
            parent_num = int(re.search(r"\d+", parent).group(0))
            current_sub = sub
            seen_any_marker = True
            parent_uses_letter_markers = True
            buf = [rest] if rest else []
            continue

        if kind in ("parent_only", "parent_lead"):
            flush()
            parent_num = int(re.search(r"\d+", parent).group(0))
            current_sub = None
            seen_any_marker = True
            parent_uses_letter_markers = False
            buf = []
            continue

        if kind in ("marker_only", "marker_lead") and abs(ln["x0"] - marker_x) <= 24:
            if parent_num == 0:
                if sub not in ("a", "i", "l"):
                    continue
                parent_num = 1
                sub = "a"
            elif current_sub is None and sub in ("i", "l") and not _is_question_like(rest or ""):
                sub = "a"
            elif current_sub and sub == "a":
                parent_num += 1
                parent_uses_letter_markers = True
            else:
                parent_uses_letter_markers = True
            flush()
            current_sub = sub
            seen_any_marker = True
            buf = [rest] if rest else []
            continue

        # First body after "Q.n 10 marks each" with no a) glyph is (a).
        if (
            parent_num >= 1
            and current_sub is None
            and kind == "text"
            and not _is_page_artifact(raw)
            and not is_header_or_instruction(raw)
        ):
            current_sub = "a"
            seen_any_marker = True
            buf = [raw]
            continue

        # New academic-verb line at the body column is the next sub, not a wrap.
        # New academic-verb line is the next sub only when OCR dropped a)/b)
        # glyphs entirely. If letter markers exist, those own the boundaries.
        if (
            (not parent_uses_letter_markers)
            and parent_num >= 1
            and current_sub
            and kind == "text"
            and _STRONG_ITEM_START.match(raw)
            and not _is_continuation(raw)
            and not _is_page_artifact(raw)
            and ln["x0"] >= marker_x + 8
        ):
            flush()
            current_sub = _next_letter(current_sub)
            buf = [raw]
            continue

        # Unlabelled statement indented past the marker column, carrying its own
        # marks tag: OCR dropped this item's marker. Whether it belongs to the
        # next parent or continues the current one is decided by the letter that
        # follows it, never by assumption.
        starts_new_item = (
            seen_any_marker
            and ln["x0"] >= marker_x + 8
            and not _is_continuation(raw)
            and not _is_page_artifact(raw)
            and bool(_MARKS_TAG.search(raw))
        )
        if starts_new_item:
            following = next_marker_sub(idx)
            flush()
            if following == "b" or parent_num == 0:
                parent_num += 1
                current_sub = "a"
            else:
                current_sub = _next_letter(current_sub)
            buf = [raw]
            continue

        if parent_num >= 1 and current_sub and not _is_page_artifact(raw):
            if is_header_or_instruction(raw):
                continue
            buf.append(raw)

    flush()

    if len(slots) < 3:
        return ""
    parents = {s[0] for s in slots}
    if len(parents) < 2:
        return ""

    # Duplicate ids mean the positional evidence contradicted itself; refuse
    # rather than emit a structure we cannot justify.
    ids = [f"{p}({s})" for p, s, _t in slots]
    if len(set(ids)) != len(ids):
        return ""

    return "\n".join(f"{p}({s}) {t}" for p, s, t in slots)


def ocr_layout_text(page, dpi: int = 150) -> str:
    """Convenience: OCR a page with geometry and return reconstructed text."""
    lines = ocr_page_lines(page, dpi=dpi)
    if not lines:
        return ""
    return reconstruct_questions_from_layout(lines)
