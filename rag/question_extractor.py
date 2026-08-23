"""
Canonical PYQ question extraction, validation, intent modeling, and recurrence helpers.

Generic for ANY subject / ANY number of uploaded papers.
Does NOT hardcode question counts, years, IDs, or subject-specific topic lists.
"""

from __future__ import annotations

import math
import re
from rag.config import current_academic_year
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Administrative / OCR garbage patterns
# ---------------------------------------------------------------------------

INSTRUCTION_PATTERNS = [
    r"attempt\s+any\s+(?:four|five|three|two|one|\d+)",
    r"attempt\s+all\s+questions?",
    r"answer\s+any\s+(?:four|five|three|two|one|\d+)",
    r"solve\s+any\s+(?:four|five|three|two|one|\d+)",
    r"any\s+(?:two|three|four|five|six)\s+out\s+of",
    r"all\s+questions?\s+are\s+compulsory",
    r"question\s+no\.?\s*\d+\s+is\s+compulsory",
    r"remaining\s+five",
    r"^nb\s*:",
    r"figures?\s+to\s+(?:the\s+)?right\s+indicate\s+(?:full\s+)?marks",
    r"use\s+of\s+calculator\s+is\s+allowed",
    r"draw\s+neat\s+diagrams?",
    r"assume\s+suitable\s+data",
    r"write\s+short\s+notes\s+on\s+any\s+(?:two|three|\d+)",
    r"instructions?\s+to\s+candidates",
    r"instructions?\s*:",
    r"section\s+-[a-z0-9]+",
    r"part\s+-[a-z0-9]+",
    r"bachelor\s+of",
    r"master\s+of",
    r"department\s+of",
    r"college\s+of",
    r"\bb\.e\.(?:[\s/]+|$)",
    r"max(?:imum)?\s+marks",
    r"time\s*:\s*\d+\s*hours?",
    r"duration\s*:\s*\d+",
    r"qp\s+code",
    r"seat\s+no",
    r"paper\s*/\s*subject\s*code",
    r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b",
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-\s]*20\d{2}",
]

FOOTER_HEADER_TOKENS = [
    "qp code",
    "b.e.",
    "b.e. ",
    "paper / subject code",
    "seat no",
    "***",
    "page no",
    "end of paper",
]

ACADEMIC_QUESTION_VERBS = {
    "explain", "what", "discuss", "derive", "calculate", "compare", "describe",
    "outline", "define", "list", "show", "state", "evaluate", "differentiate",
    "illustrate", "design", "implement", "analyze", "analyse", "solve", "why", "how",
    "distinguish", "elaborate", "contrast", "briefly", "prove", "find", "determine",
    "comment", "consider", "draw", "write", "give", "mention", "justify", "obtain",
    "create", "apply", "parse", "compute", "construct", "represent",
    "demonstrate",
}

QUESTION_TYPE_MAP = {
    "explain": "explain",
    "describe": "describe",
    "discuss": "explain",
    "elaborate": "explain",
    "define": "define",
    "what": "define",
    "compare": "compare",
    "differentiate": "compare",
    "distinguish": "compare",
    "contrast": "compare",
    "calculate": "calculate",
    "find": "calculate",
    "determine": "calculate",
    "obtain": "calculate",
    "derive": "derive",
    "prove": "derive",
    "design": "design",
    "implement": "design",
    "analyze": "analyze",
    "analyse": "analyze",
    "evaluate": "analyze",
    "illustrate": "describe",
    "outline": "describe",
    "list": "describe",
    "state": "define",
    "show": "describe",
    "draw": "describe",
    "write": "describe",
    "comment": "analyze",
    "justify": "analyze",
    "how": "explain",
    "why": "analyze",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at", "to",
    "for", "of", "with", "its", "their", "this", "that", "these", "those", "any", "all",
    "using", "used", "use", "into", "from", "by", "as", "be", "been", "being", "do",
    "does", "did", "your", "you", "we", "our", "also", "detail", "details", "suitable",
    "following", "various", "different", "types", "type", "short", "note", "notes",
    "examples", "example", "briefly", "main", "important", "give", "mention", "process",
    "system", "systems", "method", "methods", "mean", "each", "every", "such", "than",
    "then", "them", "they", "have", "has", "had", "can", "could", "should", "would",
    "may", "might", "will", "shall", "about", "over", "under", "between", "among",
    "through", "during", "before", "after", "above", "below", "again", "further",
    "once", "here", "there", "when", "where", "which", "who", "whom", "whose",
    "marks", "mark", "question", "questions", "paper", "exam", "attempt",
    "including", "include", "includes", "upto", "within", "without", "via",
}

GENERIC_ACTION_VERBS = STOPWORDS | ACADEMIC_QUESTION_VERBS | {
    "significance", "advantages", "disadvantages", "applications", "application",
    "architecture", "architectures", "working", "operation", "operations",
    "properties", "property", "features", "feature", "algorithm", "algorithms",
}

# Subquestion letters / roman numerals only — NEVER bare integers like (2)/(6).
# Roman alternatives are listed longest-first so "iii" is not consumed as "i".
# The letter class is a-z: papers use a-f, a-j, … and must not be capped at a-e.
_SUB_TOKEN = r"(?:qd|x|ix|viii|vii|vi|iv|iii|ii|i|[a-z])"
_SUB_LETTER = r"[a-z]"
_PARENT_NUM = r"[1-9]\d?"
# Sub-markers must be delimited. "2 hours" is duration, not Q2(h).
# Optional non-word junk covers OCR "c')" / "c|)".
_SUB_DELIMITED = rf"(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})\s*[^\w\s]{{0,3}}\s*[\.\)])"
_UNAMBIGUOUS_ROMAN = {"ii", "iii", "iv", "vi", "vii", "viii", "ix"}
_ROMAN_SUBS = _UNAMBIGUOUS_ROMAN | {"i", "v", "x"}
_LETTER_SIBLINGS = set("abcdefgh")


def is_choice_instruction(text: str) -> bool:
    """Parent headers like 'Attempt/Solve any four' are not questions."""
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        re.search(
            r"\b(?:attempt|solve|answer)\s+any\b"
            r"|\bany\s+(?:two|three|four|five|six|\d+)\b"
            r"|\bcompulsory\b"
            r"|\bmarks each\b"
            r"|\b(?:write\s+)?short\s+notes?\s*(?:on\s+)?(?:any\b|\()",
            t,
        )
    )


def is_instruction_frame_text(text: str) -> bool:
    """
    Strict parent-frame test used when dropping duplicate records: the text
    must BE an instruction header, not merely contain "any three" inside a
    genuine question ("…explain any three of them.").
    """
    t = (text or "").strip()
    if not t:
        return False
    if not is_choice_instruction(t):
        return False
    if "?" in t:
        return False
    if len(t.split()) > 8:
        return False
    # Real item bodies carry their own ask; frames only command.
    return not any(
        re.search(rf"\b{re.escape(v)}\b", t.lower())
        for v in ACADEMIC_QUESTION_VERBS - {"write", "give"}
    )


def _academic_verb_alt() -> str:
    return "|".join(sorted(ACADEMIC_QUESTION_VERBS, key=len, reverse=True))


def hyphen_underscore_is_compound_term(line: str) -> bool:
    """
    Letter + hyphen/underscore continuing a lowercase word is a compound
    (n-gram, k-means, v-structure), not an OCR gutter marker (b- Explain).
    """
    s = (line or "").strip()
    m = re.match(rf"^\(?({_SUB_TOKEN})\)?([\-_])(.*)$", s, re.I)
    if not m:
        return False
    rest = (m.group(3) or "").strip()
    if not rest:
        return False
    if rest[:1].isupper() or rest[:1] in "([":
        return False
    if re.match(rf"(?:{_academic_verb_alt()})\b", rest, re.I):
        return False
    return bool(re.match(r"^[a-z0-9]", rest))


def iter_unlabelled_stems(blob: str) -> List[str]:
    """
    Verb-initial exam stems, including several glued onto one OCR line after
    a choice instruction. Does not mint IDs; the caller assigns letters.
    """
    text = (blob or "").strip()
    if not text:
        return []
    verb = _academic_verb_alt()
    text = re.sub(
        rf"^(?:Q\.?|Question)?\s*{_PARENT_NUM}\s*[\.\):]?\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(
        r"^(?:Attempt|Solve|Answer)\s+any\b[^A-Za-z]*"
        r"(?:two|three|four|five|six|seven|eight)?\b\s*",
        "",
        text,
        flags=re.I,
    ).strip()
    text = re.sub(r"^\[?\s*\d+\s*\]\s*", "", text).strip()
    if not text:
        return []
    pat = re.compile(rf"(?:^|(?<=[.!?]\s))({verb})\b", re.I)
    starts = [m.start() for m in pat.finditer(text)]
    if not starts:
        return []
    out: List[str] = []
    for i, s in enumerate(starts):
        chunk = text[s : starts[i + 1] if i + 1 < len(starts) else len(text)].strip()
        if chunk:
            out.append(chunk)
    return out


def _next_unlabelled_sub(current_sub: Optional[str]) -> Optional[str]:
    if not current_sub:
        return "a"
    if len(current_sub) == 1 and "a" <= current_sub < "z":
        return chr(ord(current_sub) + 1)
    if current_sub in _LETTER_SIBLINGS:
        return chr(ord(current_sub) + 1)
    return None

# Subject-independent academic vocabulary. These words appear in questions from
# every discipline, so sharing one of them is not evidence of repetition.
GENERIC_DOMAIN_TERMS = {
    "algorithm", "algorithms", "application", "applications", "architecture",
    "concept", "concepts", "design", "diagram", "example", "examples",
    "function", "functions", "learning", "mechanism", "mechanisms", "method",
    "methods", "model", "models", "network", "networks", "operation",
    "operations", "principle", "principles", "problem", "problems", "procedure",
    "process", "processes", "property", "properties", "structure", "structures",
    "system", "systems", "technique", "techniques", "theory", "type", "types",
}


_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](20\d{2})(?!\d)")
# Markers that identify a syllabus revision / scheme year rather than the year
# the examination was held.
# Matched against the few characters immediately before a year, so an unrelated
# mention of "scheme" earlier in the header cannot suppress a real exam year.
_REVISION_CONTEXT = re.compile(
    r"(?:\b(?:rev|revised|revision|scheme|syllabus|pattern|cbcs|regulation|"
    r"autonomy|structure)|w\.?e\.?f|effect\s+from|\bR)\s*[-–:.]?\s*$",
    re.I,
)
_MONTH_SESSIONS = (
    (r"jan(?:uary)?", "Jan/Feb"),
    (r"feb(?:ruary)?", "Jan/Feb"),
    (r"mar(?:ch)?", "Mar/Apr"),
    (r"apr(?:il)?", "Mar/Apr"),
    (r"may", "May/June"),
    (r"jun(?:e)?", "May/June"),
    (r"jul(?:y)?", "Jul/Aug"),
    (r"aug(?:ust)?", "Jul/Aug"),
    (r"sep(?:t(?:ember)?)?", "Sep/Oct"),
    (r"oct(?:ober)?", "Sep/Oct"),
    (r"nov(?:ember)?", "Nov/Dec"),
    (r"dec(?:ember)?", "Nov/Dec"),
    (r"winter", "Nov/Dec"),
    (r"summer", "May/June"),
    (r"spring", "Mar/Apr"),
    (r"autumn|fall", "Sep/Oct"),
)
_MONTH_NUM_SESSION = {
    1: "Jan/Feb", 2: "Jan/Feb", 3: "Mar/Apr", 4: "Mar/Apr",
    5: "May/June", 6: "May/June", 7: "Jul/Aug", 8: "Jul/Aug",
    9: "Sep/Oct", 10: "Sep/Oct", 11: "Nov/Dec", 12: "Nov/Dec",
}


def _session_near(text: str, position: int, window: int = 14) -> Optional[str]:
    start = max(0, position - window)
    context = text[start : position + window].lower()
    for pattern, session in _MONTH_SESSIONS:
        if re.search(rf"\b{pattern}\b", context):
            return session
    return None


def detect_exam_year_and_session(
    filename: str, header_text: str = ""
) -> Tuple[int, str]:
    """
    Determine when an exam was held, from the filename and page header.

    Papers routinely print a syllabus revision year ("Rev-2019 'C' Scheme")
    alongside the exam date, so candidate years are scored by context instead of
    taking the first match. Returns (year, session); year is 0 when unknown —
    never guessed.
    """
    # Separators such as "_2024_" defeat \b, so normalise before scanning.
    fname = re.sub(r"[^A-Za-z0-9]+", " ", filename or "")
    blob = f"{fname}\n{header_text or ''}"

    best_year, best_score, best_session = 0, -1, ""

    for m in _DATE_RE.finditer(blob):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            continue
        if 100 > best_score:
            best_year, best_score = year, 100
            best_session = _MONTH_NUM_SESSION.get(month, "")

    for m in _YEAR_RE.finditer(blob):
        year = int(m.group(1))
        lead = blob[max(0, m.start() - 14) : m.start()]
        if _REVISION_CONTEXT.search(lead):
            continue
        session = _session_near(blob, m.start())
        if session:
            score = 80
        elif m.start() < len(fname):
            score = 60
        else:
            score = 20
        if score > best_score or (score == best_score and year > best_year):
            best_year, best_score = year, score
            best_session = session or best_session

    if not best_session:
        fname_session = _session_near(fname, 0, window=len(fname) + 1)
        best_session = fname_session or _session_near(blob, 0, window=400) or ""

    return best_year, best_session or "Unknown session"


def _meta_field(value: Optional[str], confidence: float, source: str) -> Dict[str, Any]:
    val = (value or "").strip() or None
    uncertain = val is None or confidence < 0.6
    return {
        "value": val,
        "confidence": round(confidence, 3),
        "source": source if val else "unknown",
        "uncertain": uncertain,
    }


def extract_paper_metadata(header_text: str, filename: str = "") -> Dict[str, Any]:
    """
    Generic header/filename metadata. No university or subject allowlists.
    Filename may corroborate; it never overrides a stronger PDF value.
    """
    header = header_text or ""
    first = "\n".join(header.splitlines()[:25])
    fname = re.sub(r"[^A-Za-z0-9]+", " ", filename or "").strip()

    uni = None
    uni_conf, uni_src = 0.0, "unknown"
    m = re.search(
        r"((?:[A-Z][A-Za-z.& ]{2,40}\s+)?(?:University|Institute of Technology|Institute of Science)(?:\s+of\s+[A-Z][A-Za-z ]{2,30})?)",
        first,
    )
    if m:
        uni, uni_conf, uni_src = m.group(1).strip(" .:-"), 0.85, "pdf"
    else:
        m2 = re.search(r"\bUniversity of ([A-Z][A-Za-z ]{2,30})", first)
        if m2:
            uni, uni_conf, uni_src = f"University of {m2.group(1).strip()}", 0.85, "pdf"

    subj = None
    subj_conf, subj_src = 0.0, "unknown"
    sm = re.search(
        r"(?:paper\s*/\s*subject|subject(?:\s*code)?|course(?:\s*name)?)\s*[:\-]\s*([A-Za-z][A-Za-z0-9 &/\-]{3,60})",
        first,
        re.I,
    )
    if sm:
        cand = sm.group(1).strip(" .:-")
        if not is_header_or_instruction(cand):
            subj, subj_conf, subj_src = cand, 0.8, "pdf"

    code = None
    code_conf, code_src = 0.0, "unknown"
    cm = re.search(r"(?:QP\s*CODE|subject\s*code|paper\s*code)\s*[:\-]?\s*([A-Za-z0-9]{3,12})", first, re.I)
    if cm:
        code, code_conf, code_src = cm.group(1).strip(), 0.85, "pdf"

    sem = None
    sem_conf, sem_src = 0.0, "unknown"
    sem_m = re.search(r"\bSem(?:ester)?\s*[:\-]?\s*([IVX]+|\d{1,2})\b", first, re.I)
    if sem_m:
        sem, sem_conf, sem_src = f"Semester {sem_m.group(1)}", 0.8, "pdf"

    year, session = detect_exam_year_and_session(filename, header)
    year_field = _meta_field(str(year) if year else None, 0.8 if year else 0.0, "pdf" if year else "unknown")
    session_field = _meta_field(
        session if session and session != "Unknown session" else None,
        0.7 if session and session != "Unknown session" else 0.0,
        "pdf",
    )

    fname_tokens = {t.lower() for t in fname.split() if len(t) > 3}
    if uni and any(t in uni.lower() for t in fname_tokens):
        uni_conf = min(0.95, uni_conf + 0.1)
    if subj and any(t in subj.lower() for t in fname_tokens):
        subj_conf = min(0.95, subj_conf + 0.1)

    return {
        "university": _meta_field(uni, uni_conf, uni_src),
        "subject": _meta_field(subj, subj_conf, subj_src),
        "paper_code": _meta_field(code, code_conf, code_src),
        "semester": _meta_field(sem, sem_conf, sem_src),
        "year": year_field,
        "exam_session": session_field,
    }


def merge_paper_metadata(
    pdf_meta: Dict[str, Any],
    workspace_info: Dict[str, Any],
) -> Dict[str, Any]:
    """PDF wins at confidence >= 0.6; else workspace; else Unknown."""
    mapping = {
        "university": ("university", "Academic Institution"),
        "subject": ("subject", "Academic Subject"),
        "paper_code": ("subjectCode", "COURSE"),
        "semester": ("semester", "Unknown"),
    }
    merged = {}
    for field, (ws_key, fallback) in mapping.items():
        rec = dict(pdf_meta.get(field) or _meta_field(None, 0.0, "unknown"))
        if rec.get("uncertain"):
            ws_val = workspace_info.get(ws_key) or workspace_info.get("subject_code") or ""
            if field == "paper_code":
                ws_val = workspace_info.get("subjectCode") or workspace_info.get("subject_code") or ""
            if ws_val and str(ws_val).strip() and str(ws_val) not in (
                "Academic Institution", "Academic Subject", "COURSE",
            ):
                rec = _meta_field(str(ws_val).strip(), 0.5, "workspace")
            elif not rec.get("value"):
                rec = _meta_field(fallback, 0.0, "unknown")
        merged[field] = rec
    merged["year"] = pdf_meta.get("year") or _meta_field(None, 0.0, "unknown")
    merged["exam_session"] = pdf_meta.get("exam_session") or _meta_field(None, 0.0, "unknown")
    return merged


def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def detect_suspicious_alphanumeric_noise(text: str) -> bool:
    if not text:
        return False
    suspicious = re.findall(
        r"(?i)(?:\d+[A-Za-z]+\d+){2,}|(?:[A-Za-z]+\d+[A-Za-z]+\d+){2,}|\d+[A-Za-z]{3,}\d+[A-Za-z]{3,}\d+",
        text,
    )
    if suspicious:
        return True

    tokens = re.findall(r"\b[A-Za-z0-9]{4,}\b", text)
    if not tokens:
        return False

    noise_count = 0
    for tok in tokens:
        has_digits = any(c.isdigit() for c in tok)
        has_upper = any(c.isupper() for c in tok)
        has_lower = any(c.islower() for c in tok)
        if has_digits and (has_upper or has_lower) and len(tok) >= 6:
            interleaved = re.search(r"\d+[A-Za-z]+\d+", tok) or re.search(r"[A-Za-z]+\d+[A-Za-z]+", tok)
            if interleaved:
                noise_count += 1
        half = len(tok) // 2
        if len(tok) >= 8 and tok[:half] == tok[half : half * 2]:
            noise_count += 1
    return (noise_count / max(1, len(tokens))) > 0.15


def is_header_or_instruction(text: str) -> bool:
    text_lower = text.strip().lower()
    if not text_lower:
        return True
    # N.B.-style instruction blocks ("N.B.: 1. Question No 1 is compulsory.")
    # are exam furniture even though they start with a single letter + dot,
    # which would otherwise look like a sub-marker "n.".
    if re.match(r"^n\.?\s*b\.?\s*[\.\:\-]", text_lower):
        return True
    # Marker-looking lines (Q1 / 2 a) / bare parents) stay visible for boundary
    # detection — but numbered instruction-list items ("2. Answer any three out
    # of the remaining questions.") are furniture, not parents. Choice-parent
    # leads ("1 Attempt any four") remain non-headers.
    marker_led = bool(
        re.match(rf"^(?:q\.?\s*)?\d{{1,2}}(?:\s*[\.\):]|\s+{_SUB_LETTER}\b|\s+attempt\b|\s*$)", text_lower)
        or re.match(rf"^\(?{_SUB_LETTER}\)?\s*[.\)]", text_lower)
    )
    if marker_led and not re.search(
        r"(?:figures?\s+to\s+(?:the\s+)?right|assume\s+suitable|out\s+of\s+the\s+remaining"
        r"|is\s+compulsory|carry\s+equal\s+marks|all\s+questions\s+carry)",
        text_lower,
    ):
        return False
    if detect_suspicious_alphanumeric_noise(text_lower):
        return True
    if "***" in text_lower:
        return True
    for token in FOOTER_HEADER_TOKENS:
        if token in text_lower and len(text_lower.split()) <= 18:
            return True
    for pattern in INSTRUCTION_PATTERNS:
        if re.search(pattern, text_lower):
            if "write short notes" in text_lower and len(text_lower.split()) > 7:
                continue
            return True
    # Institution names belong to page headers, not to questions. Matched
    # generically for any institution rather than by name, and guarded by
    # length plus the absence of an instruction verb so that a question which
    # merely mentions a university ("Design a schema for a university library")
    # is never discarded.
    if re.search(
        r"\b(?:universit(?:y|ies)|institute\s+of\s+technology|college\s+of\s+engineering)\b",
        text_lower,
    ):
        if len(text_lower.split()) <= 18 and not any(
            re.search(rf"\b{v}\b", text_lower) for v in ACADEMIC_QUESTION_VERBS
        ):
            return True
    if text_lower.startswith(("page ", "duration:", "code:", "paper / subject code")):
        return True
    if re.search(r"questions?\s+carr(?:y|ies)\s+equal\s+marks", text_lower):
        return True
    if re.match(r"^\(\d+\)\s+all\s+questions", text_lower):
        return True
    # Course-document furniture that OCR can splice onto a candidate. These are
    # structural exam/curriculum phrases, not subject vocabulary.
    if re.search(
        r"\b(?:useful links|nptel courses?|reference books?|course outcomes?|"
        r"term work|as mention(?:ed)? in the syllabus|figures? to (?:the )?right)\b",
        text_lower,
    ):
        return True
    return False


def alphabetic_quality(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / max(1, len(text))


def looks_like_ocr_garbage_topic(
    label: str, *, known_terms: Optional[Set[str]] = None
) -> bool:
    """
    Reject labels that are not defensible topic names.

    `known_terms` carries evidence from the uploaded syllabus. Bare acronyms are
    indistinguishable from OCR shreds ("ACT", "POS", "DOE") without such
    evidence, so they are only accepted when the syllabus corroborates them.
    A missed topic is preferable to a fabricated one.

    Quality checks are structural (duplication, concatenation, low information)
    — not a blacklist of subject-specific phrases.
    """
    if not label:
        return True
    low = label.lower().strip()
    if low in {
        "core academic topic", "core academic concept", "core concept", "core topic",
        "architecture", "carefully", "related widget", "topics hours", "content",
        "module", "hours", "marks", "page", "semester",
    }:
        return True
    # Syllabus-table / OCR shreds. An all-caps short token needs corroboration.
    if re.fullmatch(r"[A-Z]{2,4}", label.strip()):
        if not known_terms or low not in known_terms:
            return True
    if re.search(r"\d+s\b", low) or "topics hours" in low or "content " in low:
        return True
    if low.endswith(" concepts") or low.endswith(" fundamentals"):
        words = [w for w in re.findall(r"[A-Za-z]+", label) if len(w) > 2]
        if len(words) < 2:
            return True
        vowel_ok = sum(1 for w in words if any(ch in w.lower() for ch in "aeiouy"))
        if vowel_ok / max(1, len(words)) < 0.6:
            return True
    if detect_suspicious_alphanumeric_noise(label):
        return True
    if alphabetic_quality(label) < 0.55:
        return True

    words = [w for w in re.findall(r"[a-z]{2,}", low)]
    if len(words) >= 3:
        unique = set(words)
        if len(unique) / len(words) < 0.5:
            return True
        from collections import Counter
        counts = Counter(words)
        if any(v >= 3 for v in counts.values()):
            return True
    # Isolated-letter concatenation / sentence-like shreds
    if re.search(r"(?:\b[a-z]\b\s*){3,}", low):
        return True
    if len(low) > 48 and low.count(" ") >= 6 and not re.search(r"[A-Z]", label):
        # Long uncapitalized run is usually a question fragment, not a topic.
        return True
    return False


def structural_ocr_noise_ratio(text: str) -> float:
    """
    Generic 0–1 OCR corruption score. No hardcoded watermark strings.

    Signals: repeated characters, duplicated words, isolated letters,
    header/footer-like lines, decorative runs, impossible marker density.
    """
    if not text or not text.strip():
        return 1.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    n = len(lines)
    noise_hits = 0.0
    for ln in lines:
        if re.search(r"(.)\1{4,}", ln):
            noise_hits += 1.0
            continue
        if re.fullmatch(r"[\W_.=-]{4,}", ln):
            noise_hits += 1.0
            continue
        if re.search(r"page\s+\d+\s+of\s+\d+", ln, re.I) or re.fullmatch(r"\d{1,3}", ln):
            noise_hits += 0.6
            continue
        words = re.findall(r"[A-Za-z]{2,}", ln)
        if len(words) >= 3:
            from collections import Counter
            c = Counter(w.lower() for w in words)
            if any(v >= 3 for v in c.values()):
                noise_hits += 0.8
        isolated = len(re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", ln))
        if isolated >= 4 and isolated >= len(words):
            noise_hits += 0.7
        if detect_suspicious_alphanumeric_noise(ln):
            noise_hits += 0.8
    ratio = min(1.0, noise_hits / max(1, n))
    return round(ratio, 4)


def validate_question_candidate(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    under_instruction_parent: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Hard validation gate. Invalid candidates are ALWAYS rejected.

    under_instruction_parent: the item sits directly under an explicit exam
    instruction frame ("Write short notes on …", "Attempt any …"). The parent
    supplies the interrogative frame, so terse topic-style items are genuine
    questions even without their own academic verb.
    """
    if not text or not text.strip():
        return False, "text_too_short", {"length": 0}

    clean_text = text.strip()
    words = clean_text.split()
    has_academic_verb = any(re.search(rf"\b{re.escape(v)}\b", clean_text.lower()) for v in ACADEMIC_QUESTION_VERBS)
    has_question_mark = "?" in clean_text
    if under_instruction_parent:
        min_chars = 3
        min_words = 1
    else:
        # Short legitimate questions are allowed when structure is clear
        min_chars = 8 if (has_academic_verb or has_question_mark) else 15
    if len(clean_text) < min_chars:
        return False, "text_too_short", {"length": len(clean_text)}

    if not under_instruction_parent:
        min_words = 2 if has_academic_verb or has_question_mark else 4
        if len(words) < min_words:
            return False, "insufficient_word_count", {"word_count": len(words)}

    text_lower = clean_text.lower()

    if is_header_or_instruction(clean_text):
        return False, "administrative_instruction_or_header", {}

    # Footer-like date/time lines
    if re.search(r"\b(?:dec|nov|may|june?|jan|feb|mar|apr|jul|aug|sep|oct)[a-z]*[-\s/]*20\d{2}\b", text_lower):
        if not any(v in text_lower for v in ACADEMIC_QUESTION_VERBS):
            return False, "exam_datetime_footer", {}

    if "***" in clean_text or "qp code" in text_lower:
        return False, "header_footer_marker", {}

    if detect_suspicious_alphanumeric_noise(clean_text):
        return False, "garbled_ocr_alphanumeric_noise", {}

    academic_words = re.findall(r"[A-Za-z]{3,}", clean_text)
    if (
        not under_instruction_parent
        and len(academic_words) < 4
        and structural_ocr_noise_ratio(clean_text) >= 0.55
    ):
        return False, "garbled_ocr_alphanumeric_noise", {
            "academic_words": len(academic_words),
            "noise_ratio": structural_ocr_noise_ratio(clean_text),
        }

    unique_words = set(w.lower() for w in words)
    unique_token_ratio = len(unique_words) / len(words)
    if len(words) >= 6 and unique_token_ratio < 0.35:
        return False, "excessive_repeated_token_noise", {"unique_token_ratio": round(unique_token_ratio, 2)}

    if not has_academic_verb and not has_question_mark:
        if not under_instruction_parent:
            return False, "lacks_academic_question_structure", {}

    last_word = words[-1].lower().strip(".,;:?!")
    suspicious_ending_words = {"have", "suppose", "the", "and", "or", "with", "in", "on", "at", "to", "for", "of", "is", "are"}
    if (
        last_word in suspicious_ending_words
        and not has_question_mark
        and not clean_text.endswith(".")
        and not under_instruction_parent
    ):
        return False, "truncated_question_fragment", {"last_word": last_word}

    printable_chars = sum(1 for c in clean_text if c.isprintable())
    alpha_chars = sum(1 for c in clean_text if c.isalpha())
    printable_ratio = printable_chars / len(clean_text)
    alpha_ratio = alpha_chars / len(clean_text)
    if printable_ratio < 0.70 or alpha_ratio < 0.30:
        return False, "low_character_quality", {
            "printable_ratio": round(printable_ratio, 2),
            "alpha_ratio": round(alpha_ratio, 2),
        }

    metrics = {
        "word_count": len(words),
        "unique_token_ratio": round(unique_token_ratio, 2),
        "printable_ratio": round(printable_ratio, 2),
        "alpha_ratio": round(alpha_ratio, 2),
        "has_academic_verb": has_academic_verb,
    }
    return True, "valid_question", metrics


def extract_marks(text: str) -> int:
    """Read marks from the source line. 0 when the document does not state any."""
    if not text:
        return 0
    patterns = [
        r"\[\s*(\d{1,2})\s*M?\s*\]",
        r"(\d{1,2})\s*M\s*\]",
        r"(\d{1,2})\s*\[M\]",
        r"\[?\b(\d+)\s*(?:Marks?|marks?|mark|M)\b\]?",
    ]
    for pat in patterns:
        marks_match = re.search(pat, text, re.IGNORECASE)
        if marks_match:
            try:
                val = int(marks_match.group(1))
                if 1 <= val <= 50:
                    return val
            except ValueError:
                pass
    return 0


def normalize_question_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = re.sub(r"^\s*(?:Q\.?\s*\d+|Question\s+\d+|\d+\.|\([a-z0-9]+\))\s*:?\s*", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+\s*(?:Marks?|mark|M)\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\d+\s*(?:Marks?|mark|M)\s*\]", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Safe OCR spacing: restore a missing "of" after "types" only.
    text = re.sub(r"\btypes\s+(?!of\b)([a-z]{3,})", r"types of \1", text)
    return text


def _normalize_subtoken(tok: str) -> str:
    t = tok.lower().strip()
    if t == "qd":
        return "d"
    return t


def is_valid_question_id(q_id: str) -> bool:
    """Canonical IDs: Q1, Q1(a), Q2(b), Q10(a) — never Q2(6)/Q3(2)."""
    if not q_id:
        return False
    if re.fullmatch(rf"Q{_PARENT_NUM}", q_id, re.I):
        return True
    return bool(re.fullmatch(rf"Q{_PARENT_NUM}\({_SUB_TOKEN}\)", q_id, re.I))


def detect_question_type(text: str) -> str:
    low = text.lower()
    for verb, qtype in QUESTION_TYPE_MAP.items():
        if re.search(rf"\b{re.escape(verb)}\b", low):
            return qtype
    if "?" in text:
        return "define"
    return "explain"


def extract_entities(text: str) -> List[str]:
    """
    Subject-agnostic entity extraction.
    Uses content bigrams/trigrams + dynamic acronyms — no hardcoded subject topic lists.
    """
    if not text:
        return []
    entities: List[str] = []

    # Dynamic acronyms (2–6 caps) appearing in the question itself
    for ac in re.findall(r"\b[A-Z]{2,6}\b", text):
        if ac.lower() in {"pdf", "http", "https", "marks", "page", "sem"}:
            continue
        if ac not in entities:
            entities.append(ac)

    # Content-word bigrams / trigrams (generic academic phrasing)
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9+\-]{2,}\b", text)
    content = []
    for tok in tokens:
        tl = tok.lower()
        if tl in STOPWORDS or tl in ACADEMIC_QUESTION_VERBS or tl in GENERIC_ACTION_VERBS:
            continue
        if looks_like_ocr_garbage_topic(tok):
            continue
        content.append(tok)

    for n in (3, 2):
        for i in range(len(content) - n + 1):
            phrase_toks = content[i : i + n]
            phrase = " ".join(phrase_toks)
            if len(phrase) < 6:
                continue
            title = " ".join(
                w.upper() if w.isupper() and len(w) <= 5 else w.title()
                for w in phrase.split()
            )
            if title not in entities and not looks_like_ocr_garbage_topic(title):
                entities.append(title)
            if len(entities) >= 8:
                break
        if len(entities) >= 8:
            break

    weak_singleton = {
        "suppose", "consider", "following", "detail", "suitable", "early", "data",
        "number", "layer", "size", "input", "volume", "filters", "filter",
        "parameters", "parameter", "architecture", "network", "networks", "model",
        "method", "methods", "problem", "system", "process", "approach",
        "carefully", "clearly", "briefly", "related", "complete", "academic",
    }
    for tok in content:
        tl = tok.lower()
        if tl in weak_singleton:
            continue
        cand = tok.upper() if tok.isupper() and len(tok) <= 5 else tok.title()
        if cand not in entities and len(entities) < 8:
            entities.append(cand)

    return entities[:8]


def extract_constraints(text: str) -> List[str]:
    constraints: List[str] = []
    low = text.lower()
    if re.search(r"\badvantages?\b|\bdisadvantages?\b|\bmerits?\b|\bdemerits?\b", low):
        constraints.append("advantages_disadvantages")
    if re.search(r"\btypes?\b|\bvariants?\b|\bkinds?\b|\bforms?\b", low):
        constraints.append("enumerate_types")
    if re.search(r"\bcompare\b|\bdifferentiate\b|\bdistinguish\b|\bvs\b|\bversus\b|\bdifference\b", low):
        constraints.append("comparison")
    if re.search(r"\bcalculate\b|\bcompute\b|\bfind\b|\bdetermine\b|\bnumber of\b", low):
        constraints.append("numerical_calculation")
    if re.search(r"\barchitecture\b|\bstructure\b|\bworking\b|\bmechanism\b", low):
        constraints.append("architecture_explanation")
    if re.search(r"\bapplications?\b|\buse cases?\b|\buses?\b", low):
        constraints.append("applications")
    if re.search(r"\bderive\b|\bproof\b|\bprove\b", low):
        constraints.append("derivation")
    if re.search(r"\bdiagram\b|\bdraw\b|\bsketch\b|\billustrate\b", low):
        constraints.append("diagram_required")
    if re.search(r"\d+\s*[x\*]\s*\d+", low):
        constraints.append("numeric_dimensions")
    return constraints


def extract_operations(text: str) -> List[str]:
    ops = []
    low = text.lower()
    for verb in ("explain", "compare", "calculate", "derive", "define", "design", "analyze", "list", "prove"):
        if re.search(rf"\b{verb}\b", low):
            ops.append(verb)
    return ops


def extract_comparison_targets(text: str) -> List[str]:
    low = text.lower()
    targets = []
    m = re.search(
        r"(?:differentiate|distinguish|compare|difference)\s+(?:between\s+)?(.+?)\s+and\s+(.+?)(?:\.|$|\?)",
        low,
    )
    if m:
        for g in m.groups():
            t = g.strip(" .,;:")
            if t:
                targets.append(t.title() if len(t) > 2 else t)
    m2 = re.search(r"\b([A-Za-z][A-Za-z0-9+\-]{1,12})\s+(?:vs\.?|versus)\s+([A-Za-z][A-Za-z0-9+\-]{1,12})\b", text, re.I)
    if m2:
        targets.extend([m2.group(1), m2.group(2)])
    return targets[:4]


def build_question_intent(text: str) -> str:
    qtype = detect_question_type(text)
    entities = extract_entities(text)
    constraints = extract_constraints(text)
    ent_key = "+".join(e.lower() for e in entities[:4]) or "general"
    cons_key = "+".join(constraints) or "none"
    return f"{qtype}::{ent_key}::{cons_key}"


def build_question_representation(
    question_id: str,
    original_text: str,
    syllabus_module: str = "Unmapped",
    syllabus_chapter: str = "Unmapped",
) -> Dict[str, Any]:
    entities = extract_entities(original_text)
    constraints = extract_constraints(original_text)
    ops = extract_operations(original_text)
    comps = extract_comparison_targets(original_text)
    calc = "numerical_calculation" in constraints or "numeric_dimensions" in constraints
    diagram = "diagram_required" in constraints
    depth = "detailed" if len(original_text.split()) > 25 or "detail" in original_text.lower() else "standard"
    return {
        "question_id": question_id,
        "original_text": original_text,
        "normalized_text": normalize_question_text(original_text),
        "normalized_question": normalize_question_text(original_text),
        "question_intent": build_question_intent(original_text),
        "intent": build_question_intent(original_text),
        "question_type": detect_question_type(original_text),
        "entities": entities,
        "concepts": entities[:],
        "constraints": constraints,
        "operations": ops,
        "comparison_targets": comps,
        "calculation_required": calc,
        "diagram_required": diagram,
        "depth": depth,
        "syllabus_module": syllabus_module or "Unmapped",
        "syllabus_chapter": syllabus_chapter or "Unmapped",
    }


class CanonicalConceptExtractor:
    """
    Source-grounded concept labels from question text.
    Does NOT invent syllabus modules. Does NOT emit OCR garbage topics.
    Optional syllabus topic list may be supplied for grounding.
    """

    @classmethod
    def extract_canonical_concepts(
        cls, question_text: str, syllabus_topics: Optional[List[str]] = None
    ) -> List[str]:
        # Syllabus terms are the only admissible evidence for short acronyms.
        known_terms: Set[str] = set()
        for topic in syllabus_topics or []:
            t = str(topic).lower().strip()
            if t:
                known_terms.add(t)
                known_terms.update(re.findall(r"[a-z0-9]{2,}", t))

        if not question_text or looks_like_ocr_garbage_topic(
            question_text[:40], known_terms=known_terms
        ):
            return []

        entities = extract_entities(question_text)
        concepts: List[str] = []

        # Prefer syllabus topic hits when provided
        if syllabus_topics:
            q_lower = question_text.lower()
            for topic in syllabus_topics:
                t_low = topic.lower()
                if len(t_low) >= 3 and t_low in q_lower:
                    if topic not in concepts:
                        concepts.append(topic)

        if concepts:
            return concepts[:5]

        # Build readable labels from entities + question type signals
        constraints = extract_constraints(question_text)
        qtype = detect_question_type(question_text)

        if entities:
            primary = entities[0]
            if "comparison" in constraints and len(entities) >= 2:
                label = f"{entities[0]} vs {entities[1]}"
            elif "numerical_calculation" in constraints and "architecture_explanation" not in constraints:
                label = f"{primary} Parameter Calculation" if "parameter" in question_text.lower() else f"{primary} Calculation"
            elif "architecture_explanation" in constraints and "Architecture" not in primary:
                label = f"{primary} Architecture"
            elif "applications" in constraints:
                label = f"{primary} Applications"
            else:
                label = primary
            if not looks_like_ocr_garbage_topic(label, known_terms=known_terms):
                concepts.append(label)
            for ent in entities[1:4]:
                if (
                    ent not in concepts
                    and not looks_like_ocr_garbage_topic(ent, known_terms=known_terms)
                    and len(ent) > 3
                ):
                    concepts.append(ent)

        return [
            c for c in concepts
            if not looks_like_ocr_garbage_topic(c, known_terms=known_terms)
        ][:5]


def validate_analyzed_pyq_paper(
    questions: List[Dict[str, Any]], syllabus_index: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    seen_ids: Set[str] = set()
    valid_modules: Set[str] = {"Unmapped", "Unmapped Module"}

    if syllabus_index and "modules" in syllabus_index:
        for m in syllabus_index["modules"]:
            if isinstance(m, dict):
                if "module" in m:
                    valid_modules.add(m["module"])
                if "chapter" in m:
                    valid_modules.add(m["chapter"])

    for q in questions:
        q_id = q.get("question_id")
        q_num = q.get("question_number", "")
        if not q_id or q_id in seen_ids:
            errors.append(f"Duplicate or missing question_id: {q_id}")
        if q_id:
            seen_ids.add(q_id)
        if q_id and not is_valid_question_id(str(q_id)):
            errors.append(f"Invalid question identifier format: {q_id}")

        parent = q.get("parent_question")
        sub = q.get("subquestion")
        if sub and f"{parent}({sub})" != q_num:
            errors.append(f"Mismatched question identifier: {q_num} vs {parent}({sub})")

        topics = q.get("detected_topics") or q.get("canonical_concepts") or []
        for t in topics:
            if re.match(r"^Q\d+$", str(t), re.IGNORECASE):
                errors.append(f"Parent question container {t} invalidly used as topic for {q_num}")
            if looks_like_ocr_garbage_topic(str(t)):
                errors.append(f"Garbage topic label '{t}' for {q_num}")

        syl = q.get("syllabus_mapping") or {}
        mod = syl.get("module")
        if valid_modules and mod and mod not in valid_modules and mod not in ("Unmapped", "Unmapped Module"):
            errors.append(f"Invented syllabus module '{mod}' not found in subject syllabus index")

    return len(errors) == 0, errors


def _strip_trailing_marks(text: str) -> str:
    return re.sub(r"\[?\s*\d+\s*(?:Marks?|mark|M)?\s*\]?\s*$", "", text, flags=re.I).strip()


# Trailing runs of marks-column tokens ("… (10)", "… [10] [5]", "… 10 M]")
# are margin furniture, never question content.
_TRAILING_MARKS_TAIL = re.compile(
    r"(?:\s*(?:\[\s*\d{1,2}\s*M?\s*\]|\(\s*\d{1,2}\s*\)|\b\d{1,2}\s*M\s*\]))+\s*$",
    re.I,
)


def strip_trailing_marks_tail(text: str) -> str:
    return _TRAILING_MARKS_TAIL.sub("", (text or "").rstrip()).strip()


def _should_skip_line(line: str) -> bool:
    low = line.lower().strip()
    if not low:
        return True
    if detect_suspicious_alphanumeric_noise(line):
        return True
    if any(
        h in low
        for h in [
            "paper / subject code",
            "duration:",
            "max marks:",
            "is compulsory",
            "attempt any three",
            "attempt any four",
            "carry equal marks",
            "assume suitable data",
            "qp code",
            "seat no",
        ]
    ):
        if not re.match(rf"^(?:q\.?\s*)?{_PARENT_NUM}\b", low):
            return True
    if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", low) or re.match(r"^page\s+\d+\b", low):
        return True
    if low.startswith("university of") or re.match(r"^university\s+of\b", low):
        return True
    if re.fullmatch(r"\*+", low) or "***" in low:
        return True
    # Bare dimension fragments that are not full questions
    if re.match(r"^[\d\*xX\s]+$", line) and len(line) < 20:
        return False  # keep as continuation candidate, handled by caller
    return False


def fix_ocr_question_glyphs(text: str) -> str:
    """Fix common OCR confusions in question markers (Ql→Q1, etc.). Subject-agnostic."""
    if not text:
        return text
    # Ql / QI / Q| misread as Q1
    text = re.sub(r"\bQ[\|Il]\b", "Q1", text)
    text = re.sub(r"\bQ[\|Il]([\.\):\-])", r"Q1\1", text)
    # Gutter digit confusions: Qs/QS → Q5 (5↔s is a classic OCR swap).
    # Line-anchored and delimiter-anchored so prose ("Qs" plural) is untouched.
    text = re.sub(r"(?im)^[qQ][sS]([\.\):])", r"Q5\1", text)
    text = re.sub(r"(?im)^[qQ][sS](?=\s+[A-Z(])", "Q5", text)
    # "Qu." is the OCR rendering of a gutter "Qn." marker. When a digit
    # follows ("Qu. 3"), keep that number; a lone "Qu." heads the first block.
    text = re.sub(r"(?im)^[qQ][uU]\.?\s*(?=\d)", "Q", text)
    text = re.sub(r"(?im)^[qQ][uU]\.(?=\s|$)", "Q1", text)
    # Ql.a / Q1.a glued forms already handled by patterns
    return text


def _is_marker_only_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 24:
        return False
    # Q1. / Q2 / Q3. a. / a. / b) / (c) / e. / bare a|b|c
    if re.fullmatch(r"Q\.?\s*[0-9lI\|]+\s*[\.\):]?\s*", s, re.I):
        return True
    if re.fullmatch(rf"Q\.?\s*[0-9lI\|]+\s*[\.\):]?\s*\(?{_SUB_TOKEN}\)?[\.\)]?\s*", s, re.I):
        return True
    if re.fullmatch(rf"\(?{_SUB_TOKEN}\)?[\.\)]\s*", s, re.I):
        return True
    if re.fullmatch(rf"{_SUB_LETTER}\s*", s, re.I):
        return True
    return False


def _parse_marker_line(line: str, current_parent: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (parent, sub, rest) for a marker-ish line.
    parent like Q1; sub like a; rest leftover text.
    """
    s = fix_ocr_question_glyphs(line.strip())
    # 1. Parent + Sub on same line: Q1(a), 1 a), 1. a., 1 a .
    m = re.match(
        rf"^(?:Q\.?|Question)?\s*({_PARENT_NUM})\s*[\.\):\-]?\s*(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})[\.\):\-])\s*(.*)$",
        s,
        re.I,
    )
    if m and (s.upper().startswith("Q") or s[0].isdigit()):
        parent = f"Q{m.group(1)}"
        sub = _normalize_subtoken(m.group(2) or m.group(3))
        rest = (m.group(4) or "").strip()
        return parent, sub, rest or None

    # 2. Sub-only line when current_parent is set: a) Comment..., (a) Explain..., a. What...
    if current_parent and not hyphen_underscore_is_compound_term(s):
        m2 = re.match(rf"^(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})[\.\):\-])\s*(.*)$", s, re.I)
        if m2:
            sub = _normalize_subtoken(m2.group(1) or m2.group(2))
            rest = (m2.group(3) or "").strip()
            return current_parent, sub, rest or None

    # 3. Parent-only line: 1 Attempt any four [20], Q1. Answer all
    m3 = re.match(rf"^(?:Q\.?|Question)?\s*({_PARENT_NUM})\b\s*[\.\):]?\s*(.*)$", s, re.I)
    if m3 and (s.upper().startswith("Q") or s[0].isdigit()):
        parent = f"Q{m3.group(1)}"
        rest = (m3.group(2) or "").strip()
        return parent, None, rest or None

    return None, None, None


def _is_new_statement_line(line: str) -> bool:
    """True when a line likely starts a NEW exam question (not mid-sentence continuation)."""
    low = line.lower().strip()
    if not low:
        return False
    if is_header_or_instruction(line):
        return False
    # Must start with an academic cue — do NOT match verbs mid-line (OCR wrap)
    verb_alt = "|".join(re.escape(v) for v in sorted(ACADEMIC_QUESTION_VERBS, key=len, reverse=True))
    if re.match(rf"^(?:{verb_alt})\b", low):
        return True
    if re.match(r"^(suppose|consider|given that|let us|let\b)", low):
        return True
    return False


def _should_start_new_statement(buf: List[str], line: str) -> bool:
    if not buf or not _is_new_statement_line(line):
        return False
    prev = " ".join(buf).rstrip()
    # Soft verbs that often continue prior sentence after line wrap
    soft = re.match(
        r"^(calculate|derive|find|determine|obtain|compute|using|also|and|or|with)\b",
        line.lower().strip(),
    )
    if soft and not prev.endswith((".", "?", "!", "]", '"', "'")):
        return False
    return True


def normalize_ocr_split_layout(page_text: str) -> str:
    """
    Rebuild OCR pages where markers are listed first and statements follow:

        Q1.
        a.
        b.
        Q2. a.
        Design AND gate...
        Explain dropout...

    into canonical lines:

        Q1(a) Design AND gate...
        Q1(b) Explain dropout...

    Subject-agnostic; does not assume a fixed question count.
    """
    if not page_text or len(page_text.strip()) < 20:
        return page_text

    raw_lines = [ln.strip() for ln in fix_ocr_question_glyphs(page_text).splitlines() if ln.strip()]
    if len(raw_lines) < 6:
        return "\n".join(raw_lines)

    # Refuse detached gutter skeletons: >=3 consecutive bare parent numbers
    # ("Q2.\nQ3.\nQ4.") mean OCR split the marker column from the bodies.
    # Pairing those parents with the following statements would fabricate
    # IDs (Q2(c)…) for another question's content; the geometry-aware
    # representation owns such pages instead.
    bare_run = 0
    max_bare_run = 0
    for ln in raw_lines:
        if re.fullmatch(r"(?:Q\.?\s*)?[1-9]\d?\s*[\.\)]?", ln):
            bare_run += 1
            max_bare_run = max(max_bare_run, bare_run)
        elif ln:
            bare_run = 0
    if max_bare_run >= 3:
        return "\n".join(raw_lines)

    # Find the densest marker-only run (OCR skeleton region)
    best_start, best_end, best_len = 0, 0, 0
    i = 0
    while i < len(raw_lines):
        if not _is_marker_only_line(raw_lines[i]):
            i += 1
            continue
        j = i
        while j < len(raw_lines) and (
            _is_marker_only_line(raw_lines[j])
            or (
                re.match(r"^Q\.?\s*\d+", raw_lines[j], re.I)
                and len(raw_lines[j]) < 24
                and not _is_new_statement_line(raw_lines[j])
            )
        ):
            j += 1
        if (j - i) > best_len:
            best_start, best_end, best_len = i, j, j - i
        i = max(j, i + 1)

    if best_len < 3:
        return "\n".join(raw_lines)

    marker_lines = raw_lines[best_start:best_end]
    body_lines = raw_lines[best_end:]

    slots: List[Tuple[str, str]] = []
    current_parent: Optional[str] = None
    for ln in marker_lines:
        parent, sub, rest = _parse_marker_line(ln, current_parent)
        if parent:
            current_parent = parent
        if current_parent and sub:
            slots.append((current_parent, sub))
        # If marker line secretly had statement text, push into body
        if rest and _is_new_statement_line(rest):
            body_lines.insert(0, rest)

    if len(slots) < 2 or current_parent is None:
        return "\n".join(raw_lines)

    cleaned_body = []
    for ln in body_lines:
        if re.fullmatch(r"\d{4,}", ln):
            continue
        if is_header_or_instruction(ln):
            continue
        if _is_marker_only_line(ln):
            continue
        # Skip leftover header fragments before first academic statement
        if not cleaned_body and not _is_new_statement_line(ln) and len(ln) < 40:
            continue
        cleaned_body.append(ln)

    if not cleaned_body:
        return "\n".join(raw_lines)

    statements: List[str] = []
    buf: List[str] = []
    for ln in cleaned_body:
        if _should_start_new_statement(buf, ln):
            statements.append(" ".join(buf).strip())
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        statements.append(" ".join(buf).strip())

    statements = [s for s in statements if len(s) >= 12]
    if len(statements) < 3:
        return "\n".join(raw_lines)

    # Extra statements are continuations of the last genuine slot — never mint IDs.
    n = min(len(slots), len(statements))
    rebuilt = [f"{slots[i][0]}({slots[i][1]}) {statements[i]}" for i in range(n)]
    if len(statements) > n and rebuilt:
        rebuilt[-1] = rebuilt[-1] + " " + " ".join(statements[n:])

    return "\n".join(rebuilt)


def unglue_parent_letter_markers(page_text: str) -> str:
    """
    Line-start glued forms without inventing IDs from mid-sentence years:

        Q3a) Explain...   → Q3(a) Explain...
        Q3a Explain...    → Q3(a) Explain...
        4a) Explain...    → Q4(a) Explain...
        3 a_i) Why...     → 3 a) Why...   (OCR-merged letter + leftover)
    """
    if not page_text:
        return page_text
    verb = "|".join(sorted(ACADEMIC_QUESTION_VERBS))
    text = re.sub(
        rf"(?im)^(?:Q\.?\s*)?({_PARENT_NUM})({_SUB_LETTER})\)\s*",
        lambda m: f"Q{m.group(1)}({m.group(2).lower()}) ",
        page_text,
    )
    text = re.sub(
        rf"(?im)^(?:Q\.?\s*)?({_PARENT_NUM})({_SUB_LETTER})\s+(?=({verb})\b)",
        lambda m: f"Q{m.group(1)}({m.group(2).lower()}) ",
        text,
    )
    text = re.sub(
        rf"(?im)^({_PARENT_NUM})\s+({_SUB_LETTER})_[A-Za-z]\)\s*",
        r"\1 \2) ",
        text,
    )
    return text


def normalize_bare_numbered_exam_layout(page_text: str) -> str:
    """
    Normalize exam layouts that use bare parent numbers without a 'Q' prefix:

        1
        a)
        What are Feed Forward Neural Networks?
        b) Explain Gradient Descent...
        2 a)
        What are the Three Classes...

    into canonical lines:

        Q1(a) What are Feed Forward Neural Networks?
        Q1(b) Explain Gradient Descent...
        Q2(a) What are the Three Classes...

    Subject-agnostic; question count comes only from detected markers.
    """
    if not page_text or len(page_text.strip()) < 20:
        return page_text

    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    verb_alt = "|".join(sorted(ACADEMIC_QUESTION_VERBS))
    letter_verb = re.compile(rf"^({_SUB_LETTER})\s+((?:{verb_alt})\b.*)$", re.I)
    num_letter_verb = re.compile(
        rf"^({_PARENT_NUM})\s+({_SUB_LETTER})\s+((?:{verb_alt})\b.*)$", re.I
    )
    # Need evidence of bare numbering (digit parent + letter subs)
    has_bare_parent = any(re.fullmatch(r"[1-9]\d?", ln) for ln in lines)
    has_letter_sub = any(re.match(rf"^\(?{_SUB_LETTER}\)?[.\)]\s*", ln, re.I) for ln in lines)
    has_letter_verb = any(letter_verb.match(ln) for ln in lines)
    has_num_letter = any(
        re.match(rf"^[1-9]\d?\s+{_SUB_DELIMITED}", ln, re.I) for ln in lines
    )
    has_num_letter_verb = any(num_letter_verb.match(ln) for ln in lines)
    if not (
        (has_bare_parent and (has_letter_sub or has_letter_verb))
        or has_num_letter
        or has_num_letter_verb
    ):
        return page_text

    out: List[str] = []
    current_parent: Optional[str] = None
    pending_sub: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal buf, pending_sub
        if current_parent and pending_sub and buf:
            text = " ".join(buf).strip()
            text = re.sub(r"^\[\d+\]\s*", "", text)
            if text and not is_header_or_instruction(text):
                out.append(f"{current_parent}({pending_sub}) {text}")
        buf = []
        pending_sub = None

    for ln in lines:
        if is_header_or_instruction(ln):
            continue
        if re.fullmatch(r"\[\d+\]", ln):
            continue
        if re.fullmatch(r"\*{3,}", ln):
            continue

        # Bare parent: "1" / "1." / "1 Attempt any four" / "1 Solve any four"
        if re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*", ln) or re.match(
            rf"^{_PARENT_NUM}\s+(Attempt|Solve|any)\b", ln, re.I
        ):
            flush()
            num = re.match(rf"^({_PARENT_NUM})", ln).group(1)
            current_parent = f"Q{num}"
            pending_sub = None
            continue

        m_nlv = num_letter_verb.match(ln)
        if m_nlv:
            flush()
            current_parent = f"Q{m_nlv.group(1)}"
            pending_sub = m_nlv.group(2).lower()
            buf = [m_nlv.group(3).strip()]
            continue

        # "2 a)" / "2 a) What..." / "3.a) What..." / "4 a)."
        # The sub-letter MUST be delimited; "2 hours" is not Q2(h).
        m_combo = re.match(
            rf"^({_PARENT_NUM})\s*[\.\):]*\s*{_SUB_DELIMITED}\s*(.*)$",
            ln,
            re.I,
        )
        if m_combo:
            flush()
            sub = m_combo.group(2) or m_combo.group(3)
            current_parent = f"Q{m_combo.group(1)}"
            pending_sub = _normalize_subtoken(sub)
            rest = (m_combo.group(4) or "").strip()
            rest = re.sub(r"^\[\d+\]\s*", "", rest)
            rest = re.sub(r"\s*\[?\d+\]?\s*$", "", rest).strip()
            buf = [rest] if rest else []
            continue

        # Standalone "a)" / "a." / "(a)" with optional rest
        m_sub = re.match(rf"^\(?({_SUB_TOKEN})\)?\s*[\.\)]\s*(.*)$", ln, re.I)
        if m_sub:
            sub_tok = _normalize_subtoken(m_sub.group(1))
            if current_parent is None and sub_tok == "a":
                # Only infer Q1 when later bare parents (2, 3, …) prove a Q1 block existed.
                # Never blindly map every orphan a) → Q1.
                later_parents = any(
                    re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*", x)
                    or re.match(rf"^{_PARENT_NUM}\s+\(?{_SUB_LETTER}", x, re.I)
                    or re.match(rf"^{_PARENT_NUM}\s+(Attempt|Solve|any)\b", x, re.I)
                    for x in lines
                )
                if later_parents:
                    current_parent = "Q1"
                else:
                    continue
            if current_parent:
                flush()
                pending_sub = sub_tok
                rest = (m_sub.group(2) or "").strip()
                buf = [rest] if rest else []
                continue

        if re.fullmatch(rf"{_SUB_LETTER}\)?" , ln, re.I) and current_parent:
            flush()
            pending_sub = ln[0].lower()
            buf = []
            continue

        m_lv = letter_verb.match(ln)
        if m_lv:
            sub_tok = m_lv.group(1).lower()
            if current_parent is None and sub_tok == "a":
                later_parents = any(
                    re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*", x)
                    or re.match(rf"^{_PARENT_NUM}\s+\(?{_SUB_LETTER}", x, re.I)
                    or re.match(rf"^{_PARENT_NUM}\s+(Attempt|Solve|any)\b", x, re.I)
                    for x in lines
                )
                if later_parents:
                    current_parent = "Q1"
            if current_parent:
                flush()
                pending_sub = sub_tok
                buf = [m_lv.group(2).strip()]
                continue

        # Continuation / body
        if current_parent and pending_sub is not None:
            if re.fullmatch(r"\[?\d+\]?", ln):
                continue
            buf.append(ln)
        elif current_parent and pending_sub is None:
            # body before first sub — ignore instruction leftovers
            continue

    flush()
    return "\n".join(out) if out else page_text


def normalize_letter_only_skeleton(page_text: str) -> str:
    """
    When OCR drops parent numbers but leaves letter markers + bodies:

        a)
        b)
        Explain A...
        Explain B...
        a)
        Explain C...

    Reconstruct Qn(letter) by letter-run restarts (a after prior letter).
    If letter markers are corrupted (repeated OCR junk), refuse rather than
    guess sequential Q1(a)..Q1(f), Q2(a)... from statement order.
    """
    if not page_text:
        return page_text
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    letter_idxs = []
    for i, ln in enumerate(lines):
        if re.fullmatch(rf"\(?{_SUB_LETTER}\)?\s*[\.\)]\s*", ln, re.I) or re.fullmatch(
            r"q?d\)?\s*", ln, re.I
        ):
            letter_idxs.append(i)
    if len(letter_idxs) < 3:
        return page_text
    if any(re.match(r"^Q\d+", ln, re.I) for ln in lines):
        return page_text
    if any(re.fullmatch(r"[1-9]\d?", ln) for ln in lines):
        return page_text

    letters_raw: List[str] = []
    for idx in letter_idxs:
        ln = lines[idx]
        m = re.match(rf"^\(?({_SUB_LETTER}|qd)\)?\s*[\.\)]?\s*$", ln, re.I)
        if not m:
            continue
        raw = m.group(1).lower()
        letters_raw.append("d" if raw == "qd" else raw)

    body_start = letter_idxs[-1] + 1
    statements: List[str] = []
    buf: List[str] = []
    for ln in lines[body_start:]:
        if is_header_or_instruction(ln):
            continue
        if re.fullmatch(rf"\(?{_SUB_LETTER}\)?\s*[\.\)]\s*", ln, re.I):
            continue
        if re.fullmatch(r"\[\d+\]", ln):
            continue
        if buf and _is_new_statement_line(ln):
            statements.append(" ".join(buf).strip())
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        statements.append(" ".join(buf).strip())
    statements = [
        s
        for s in statements
        if len(s) >= 12 and not is_header_or_instruction(s)
        and "compulsory" not in s.lower()
        and not re.match(r"^\(?\d+\)?\s*question\s+no", s, re.I)
    ]
    if len(statements) < 3:
        return page_text

    # Only letter-run evidence may drive reconstruction. A corrupted marker
    # stream is refused outright: guessing parent numbers from statement order
    # would fabricate question identity. Layout-aware OCR handles those pages.
    unique_ratio = len(set(letters_raw)) / max(1, len(letters_raw))
    dup_runs = sum(
        1 for i in range(1, len(letters_raw)) if letters_raw[i] == letters_raw[i - 1]
    )
    if unique_ratio < 0.55 or dup_runs >= 2:
        return page_text

    slots: List[Tuple[str, str]] = []
    parent_num = 1
    prev = None
    for letter in letters_raw:
        if prev is not None and letter == "a" and prev >= "a":
            parent_num += 1
        slots.append((f"Q{parent_num}", letter))
        prev = letter

    n = min(len(slots), len(statements))
    if n < 3:
        return page_text
    return "\n".join(f"{slots[i][0]}({slots[i][1]}) {statements[i]}" for i in range(n))


def question_structure_score(question_ids: List[str]) -> float:
    """
    How internally consistent a set of extracted IDs is, in [0, 1].

    Real papers number parents from 1 upwards without gaps and letter their
    subquestions from 'a' upwards without gaps. A representation that produces
    gaps or stray parents has almost certainly mis-split the page.
    """
    if not question_ids:
        return 0.0
    parents: Dict[int, List[str]] = {}
    for qid in question_ids:
        m = re.match(rf"Q({_PARENT_NUM})(?:\(({_SUB_TOKEN})\))?$", qid, re.I)
        if not m:
            continue
        parents.setdefault(int(m.group(1)), []).append((m.group(2) or "").lower())
    if not parents:
        return 0.0

    nums = sorted(parents)
    parents_contiguous = bool(nums) and nums[0] in (1, 2) and (
        nums == list(range(nums[0], nums[0] + len(nums)))
    )

    compliant = 0
    for _num, subs in parents.items():
        letters = sorted({s for s in subs if s and s.isalpha() and len(s) == 1})
        if not letters:
            compliant += 1
            continue
        # Unique valid letters are enough. a,c without b is not a failure
        # when b was never a source marker.
        if len(letters) == len(set(letters)):
            compliant += 1
    sub_ratio = compliant / len(parents)
    return round(0.5 * sub_ratio + (0.5 if parents_contiguous else 0.0), 4)


def prepare_page_text_for_extraction(page_text: str) -> str:
    """Glyph fix + layout normalize. Winner is evidence, not regex marker count."""
    text = unglue_parent_letter_markers(fix_ocr_question_glyphs(page_text or ""))
    # OCR junk directly in front of a marker ("�Q. 1 Solve…", "£2 a)")
    # breaks the parent chain for everything that follows. Strip up to three
    # leading junk characters ONLY when a marker form immediately follows.
    if text:
        text = re.sub(
            rf"(?m)^[^\w\n(]{{1,3}}(?=(?:Q\.?\s*)?{_PARENT_NUM}\s*[\.\)\(:]|Q\.?\s*{_PARENT_NUM}\b)",
            "",
            text,
        )
    if not text.strip():
        return text
    from rag.hybrid_question_extraction import detect_markers_in_text

    original_ids = set(detect_markers_in_text(text))
    bare = normalize_bare_numbered_exam_layout(text)
    text2 = normalize_ocr_split_layout(text)
    bare2 = normalize_bare_numbered_exam_layout(text2)
    letter = normalize_letter_only_skeleton(text2)
    pool = [bare, text2, bare2, letter, text]

    def _norm_score(candidate: str) -> Tuple[int, int, int, int]:
        ids = detect_markers_in_text(candidate)
        associated = sum(1 for i in ids if i in original_ids)
        inferred = sum(1 for i in ids if i not in original_ids)
        recovered = 0 if original_ids else len(ids)
        paired = sum(
            1
            for ln in candidate.splitlines()
            if re.match(rf"Q{_PARENT_NUM}\({_SUB_TOKEN}\)\s+\S", ln, re.I)
        )
        return (associated, paired, recovered, -inferred)

    best = max(pool, key=_norm_score)
    if original_ids:
        best_ids = detect_markers_in_text(best)
        inferred = [i for i in best_ids if i not in original_ids]
        invalid_inferred = sum(1 for i in inferred if not is_valid_question_id(i))
        if invalid_inferred > 2:
            return text
    return best


def extract_questions_from_page_text(
    page_text: str,
    page_num: int,
    source_file: str,
    workspace_id: str,
    subject: str = "Subject",
    year: int | None = None,
    syllabus_topics: Optional[List[str]] = None,
    inherit_parent: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split page text into discrete canonical subquestions.
    Question count is determined solely from detected boundaries in the PDF text.

    inherit_parent: last parent from the previous page (e.g. "Q5") so a leading
    orphan "b)" / "c)" after a page break stays attached to that parent.
    """
    if year is None:
        year = current_academic_year()
    page_text = prepare_page_text_for_extraction(page_text)
    if not page_text or len(page_text.strip()) < 15:
        return [], []

    # Page-level instruction-frame evidence. Layout-reconstructed pages lose
    # the standalone "Write short notes on… / Attempt any…" lead line (it is
    # consumed as the parent boundary), so its presence anywhere on the page
    # still marks terse printed sub items as instruction-frame questions.
    # OCR-garbled forms ("Write short no} y 2):") must match too, hence the
    # loose stems rather than exact words.
    page_has_instruction_frame = any(
        re.search(
            r"writ\w*\s+short|short\s+n[o0]t\w*|attempt\s+(?:the\s+)?follow|"
            r"\bany\s+\d{1,2}\b|\bany\s+(?:two|three|four|five|six|seven)\b",
            ln or "",
            re.I,
        )
        for ln in page_text.splitlines()
    )

    raw_lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    clean_lines: List[str] = []
    for line in raw_lines:
        line_clean = re.sub(r"\s*(?:QP\s*CODE:?\s*\d+|\d{4,}\s+)?Page\s*[|\d\s]*of\s*\d+.*$", "", line, flags=re.I).strip()
        if _should_skip_line(line_clean):
            # Dimension-only lines still merge as continuations
            if re.match(r"^[\d\*xX.\s/]+$", line_clean) and clean_lines:
                clean_lines.append(line_clean)
            continue
        # Reject B.E. / degree lines masquerading as subquestions
        if re.match(r"^B\.?\s*E\.?\b", line_clean, re.I):
            continue
        clean_lines.append(line_clean)

    questions: List[Dict[str, Any]] = []
    unlabelled_stem_mode = False
    # True while the active parent is an explicit instruction frame
    # ("Attempt any four", "Write short notes on…"): its printed sub items are
    # genuine even when they carry no academic verb of their own.
    choice_frame = False
    # Some OCR layouts lose the parent header line entirely but the paper-wide
    # instruction phrase survives elsewhere on the page ("Write short notes",
    # "Attempt any …"). Terse printed sub items are still genuine there.
    doc_has_frame = bool(
        re.search(
            r"(?:write\s+short\s+notes?|attempt\s+(?:any|the\s+following)|solve\s+any|answer\s+any)",
            page_text or "",
            re.I,
        )
    )
    inherited = (
        inherit_parent.upper()
        if inherit_parent and re.fullmatch(rf"Q{_PARENT_NUM}", inherit_parent, re.I)
        else None
    )
    current_parent = inherited
    current_sub = None
    current_marks = 0
    current_text: List[str] = []
    current_origin = "marker"

    # STRICT + flexible academic layouts (subject-agnostic; no fixed Q count)
    # Q1(a) / Q.1(a) / Question 1(a) / Q1.a / Q1 a) / Q1-a
    combined_pattern = re.compile(
        rf"^(?:Q\.?|Question)\s*({_PARENT_NUM})\s*[\.\):\-]?\s*"
        rf"(?:\(({_SUB_TOKEN})\)|({_SUB_TOKEN})\)|({_SUB_TOKEN})\.(?:\s|$)|({_SUB_TOKEN})\s+)"
        rf"\s*(.*)$",
        re.IGNORECASE,
    )
    # 1(a) / 1.a / 1) a) without Q prefix
    numbered_sub_pattern = re.compile(
        rf"^({_PARENT_NUM})\s*[\.\)]?\s*{_SUB_DELIMITED}\s*(.*)$",
        re.IGNORECASE,
    )
    # Standalone subquestion: (a) / a) / a. / (i) / i)  — rest may be on following lines
    sub_pattern = re.compile(
        rf"^\(?({_SUB_TOKEN})\)?[\.\):\-_]\s*(.*)$",
        re.IGNORECASE,
    )
    # "a Explain..." / "b Explain..." when letter followed by academic verb
    sub_loose_pattern = re.compile(
        rf"^({_SUB_TOKEN})\s+((?:{'|'.join(sorted(ACADEMIC_QUESTION_VERBS))})\b.*)$",
        re.IGNORECASE,
    )
    # Q3-A / Q3-B style
    dashed_sub_pattern = re.compile(
        rf"^(?:Q\.?|Question)?\s*({_PARENT_NUM})\s*[-–]\s*([A-Za-z])\b[\.\):]?\s*(.*)$",
        re.IGNORECASE,
    )
    parent_only_pattern = re.compile(
        rf"^(?:Q\.?|Question)\s*({_PARENT_NUM})\b(?:\s*[\.\):]?\s*)(.*)$",
        re.IGNORECASE,
    )
    bare_parent_pattern = re.compile(
        rf"^({_PARENT_NUM})\s*[\.\):]?\s*$"
        rf"|^({_PARENT_NUM})\s+(?=Attempt|Solve|any\b)",
        re.IGNORECASE,
    )
    numbered_letter_verb_pattern = re.compile(
        rf"^({_PARENT_NUM})\s+({_SUB_LETTER})\s+((?:{'|'.join(sorted(ACADEMIC_QUESTION_VERBS))})\b.*)$",
        re.IGNORECASE,
    )

    def _roman_list_follows(start: int) -> bool:
        # Scan through tables/boxes until the next parent or letter sibling.
        # Lone digits and wrap letters (Thursda / y) are not boundaries.
        lines_after = clean_lines[start + 1 :]
        for i, nxt in enumerate(lines_after):
            m = re.match(rf"^\(?({_SUB_TOKEN})\)?[\.\)]\s*", nxt, re.I)
            if m:
                tok = _normalize_subtoken(m.group(1))
                if tok in _UNAMBIGUOUS_ROMAN:
                    return True
                if len(tok) == 1 and tok.isalpha() and tok != "i":
                    return False
            if re.fullmatch(rf"{_SUB_LETTER}", nxt, re.I):
                follow = lines_after[i + 1] if i + 1 < len(lines_after) else ""
                if not any(re.search(rf"\b{v}\b", follow.lower()) for v in ACADEMIC_QUESTION_VERBS):
                    continue
            if re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*$", nxt):
                continue
            if (
                combined_pattern.match(nxt)
                or parent_only_pattern.match(nxt)
                or numbered_sub_pattern.match(nxt)
                or numbered_letter_verb_pattern.match(nxt)
                or bare_parent_pattern.match(nxt)
            ):
                return False
        return False

    def _is_nested_roman(new_sub: str, idx: int) -> bool:
        if not current_sub or current_sub not in _LETTER_SIBLINGS:
            return False
        if new_sub in _UNAMBIGUOUS_ROMAN:
            return True
        return new_sub == "i" and _roman_list_follows(idx)

    def flush_current():
        nonlocal current_text, current_sub, current_parent, current_marks, current_origin
        # Inherited parent from the previous page is already emitted there.
        # Leading junk / diagram residue must not become a new parent-only ID.
        if inherited and current_parent == inherited and current_sub is None:
            current_text = []
            return
        if current_parent and current_sub and current_text:
            q_id = f"{current_parent}({current_sub})"
            if is_valid_question_id(q_id):
                joined_text = " ".join(current_text).strip()
                frame_item = choice_frame
                if not frame_item and doc_has_frame:
                    # Page-wide frame only relaxes validation for genuinely
                    # terse topic-style items, never for long bodies.
                    frame_item = len(joined_text) <= 60 and len(joined_text.split()) <= 8
                questions.append(
                    {
                        "question_id": q_id,
                        "question_number": q_id,
                        "parent_question": current_parent,
                        "subquestion": current_sub,
                        "marks": current_marks,
                        "exact_text": joined_text,
                        "origin": current_origin,
                        "under_instruction_parent": bool(
                            frame_item and current_origin == "marker"
                        ),
                    }
                )
        elif current_parent and not current_sub and current_text:
            q_id = current_parent
            if is_valid_question_id(q_id):
                questions.append(
                    {
                        "question_id": q_id,
                        "question_number": q_id,
                        "parent_question": current_parent,
                        "subquestion": None,
                        "marks": current_marks,
                        "exact_text": " ".join(current_text).strip(),
                        "origin": current_origin,
                        "under_instruction_parent": False,
                    }
                )
        current_text = []

    def parent_has_explicit_subs(start: int) -> bool:
        """
        Does a standalone sub-marker follow this parent before the next parent?

        Decides whether "Q3. Explain ..." is the (a) part of a subdivided
        question or a flat question in its own right. Papers use both layouts,
        so this is read from the document rather than assumed.
        """
        for nxt in clean_lines[start + 1:]:
            if (
                combined_pattern.match(nxt)
                or dashed_sub_pattern.match(nxt)
                or parent_only_pattern.match(nxt)
                or numbered_sub_pattern.match(nxt)
            ):
                return False
            if (
                sub_pattern.match(nxt) or sub_loose_pattern.match(nxt)
            ) and not hyphen_underscore_is_compound_term(nxt):
                return True
        return False

    for idx, line in enumerate(clean_lines):
        if line.isdigit() and len(line) > 3:
            continue

        def _parent_leap(new_num: str) -> bool:
            if not current_parent:
                return False
            try:
                old_n = int(re.search(r"\d+", current_parent).group(0))
                new_n = int(new_num)
            except (TypeError, ValueError, AttributeError):
                return False
            return new_n >= 10 and new_n > old_n + 3

        m_nlv = numbered_letter_verb_pattern.match(line)
        if m_nlv:
            if _parent_leap(m_nlv.group(1)):
                if m_nlv.group(3):
                    current_text.append(m_nlv.group(3).strip())
                continue
            flush_current()
            current_parent = f"Q{m_nlv.group(1)}"
            current_sub = m_nlv.group(2).lower()
            current_marks = extract_marks(line)
            current_text = [m_nlv.group(3).strip()] if m_nlv.group(3).strip() else []
            current_origin = "marker"
            choice_frame = False
            unlabelled_stem_mode = False
            continue

        m_comb = combined_pattern.match(line) or dashed_sub_pattern.match(line) or numbered_sub_pattern.match(line)
        if m_comb:
            rest = _strip_trailing_marks(m_comb.group(m_comb.lastindex) or "")
            sub_raw = None
            for gi in range(2, m_comb.lastindex):
                if m_comb.group(gi):
                    sub_raw = m_comb.group(gi)
                    break
            # numbered_sub without Q: require academic content to avoid dates/codes
            if numbered_sub_pattern.match(line) and not combined_pattern.match(line) and not dashed_sub_pattern.match(line):
                if not any(re.search(rf"\b{v}\b", rest.lower()) for v in ACADEMIC_QUESTION_VERBS) and "?" not in rest:
                    # might be a false positive (page numbers etc.) — skip as boundary
                    if current_parent:
                        cleaned_line = _strip_trailing_marks(line)
                        if cleaned_line and not is_header_or_instruction(cleaned_line):
                            current_text.append(cleaned_line)
                    continue
            if _parent_leap(m_comb.group(1)):
                if rest:
                    current_text.append(rest)
                continue
            flush_current()
            current_parent = f"Q{m_comb.group(1)}"
            current_sub = _normalize_subtoken(sub_raw or "")
            current_marks = extract_marks(line)
            current_text = [rest] if rest else []
            current_origin = "marker"
            choice_frame = False
            unlabelled_stem_mode = False
            continue

        m_sub = sub_pattern.match(line) or sub_loose_pattern.match(line)
        if m_sub and current_parent and hyphen_underscore_is_compound_term(line):
            m_sub = None
        # "Q. 6 Solve any Four" / "Q6. Attempt…" — the leading Q is a
        # question prefix, never a sub-question letter of the previous
        # parent. A lone q/Q token followed by a number is that prefix.
        if (
            m_sub
            and _normalize_subtoken(m_sub.group(1)) == "q"
            and re.match(r"\s*\.?\s*\d", m_sub.group(2) or "")
        ):
            m_sub = None
        if m_sub and current_parent:
            rest = _strip_trailing_marks(m_sub.group(2) or "")
            new_sub = _normalize_subtoken(m_sub.group(1))
            if _is_nested_roman(new_sub, idx):
                if rest:
                    current_text.append(rest)
                elif line not in current_text:
                    current_text.append(line)
                continue
            flush_current()
            current_sub = new_sub
            if re.search(r"\b\d+\s*(?:Marks?|mark|M)\b", line, re.I):
                current_marks = extract_marks(line)
            current_text = [rest] if rest else []
            current_origin = "marker"
            unlabelled_stem_mode = False
            choice_frame = False
            continue

        m_letter_only = re.fullmatch(rf"({_SUB_LETTER})\)?", line, re.I)
        if m_letter_only and current_parent:
            new_sub = m_letter_only.group(1).lower()
            nxt = clean_lines[idx + 1] if idx + 1 < len(clean_lines) else ""
            nxt_has_verb = any(
                re.search(rf"\b{v}\b", nxt.lower()) for v in ACADEMIC_QUESTION_VERBS
            )
            if not re.search(r"[.\)]", line) and not nxt_has_verb:
                if current_text:
                    current_text.append(line)
                continue
            if not _is_nested_roman(new_sub, idx):
                flush_current()
                current_sub = new_sub
                current_text = []
                current_origin = "marker"
                continue

        m_bare = bare_parent_pattern.match(line)
        if m_bare:
            new_num = m_bare.group(1) or m_bare.group(2)
            # Lone digits are table cells unless a lettered sub follows.
            if re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*$", line):
                follows_sub = False
                for nxt in clean_lines[idx + 1 : idx + 8]:
                    if re.fullmatch(rf"{_PARENT_NUM}\s*[\.\):]?\s*$", nxt):
                        continue
                    if re.fullmatch(r"\[\s*\d+\s*\]", nxt) or re.match(r"^[\d\s]+$", nxt):
                        continue
                    sm = sub_pattern.match(nxt) or sub_loose_pattern.match(nxt) or re.fullmatch(rf"({_SUB_LETTER})\)?", nxt, re.I)
                    if sm and hyphen_underscore_is_compound_term(nxt):
                        sm = None
                    if sm and _normalize_subtoken(sm.group(1)) in _LETTER_SIBLINGS:
                        follows_sub = True
                    elif re.match(r"^(?:Attempt|Solve|any)\b", nxt, re.I):
                        follows_sub = True
                    break
                if not follows_sub:
                    if current_parent and current_text:
                        current_text.append(line)
                    continue
            if _parent_leap(new_num):
                continue
            flush_current()
            current_parent = f"Q{new_num}"
            current_sub = None
            current_text = []
            unlabelled_stem_mode = bool(re.search(r"(?:Attempt|Solve|any)\b", line, re.I))
            choice_frame = unlabelled_stem_mode
            if unlabelled_stem_mode:
                for stem in iter_unlabelled_stems(line):
                    nxt = _next_unlabelled_sub(current_sub)
                    if not nxt:
                        break
                    flush_current()
                    current_sub = nxt
                    current_text = [stem]
                    current_origin = "inferred_stem"
            else:
                current_origin = "marker"
            continue

        m_parent = parent_only_pattern.match(line)
        if m_parent:
            rest = _strip_trailing_marks(m_parent.group(2) or "")
            # Reject numeric-only pseudo-subs like Q2(6) / Q3(2) — not canonical IDs
            if re.match(r"^\(\d+\)", rest.strip()):
                continue
            # Parent header with instruction text only
            if rest and (is_choice_instruction(rest) or is_instruction_frame_text(rest)):
                if _parent_leap(m_parent.group(1)):
                    continue
                flush_current()
                current_parent = f"Q{m_parent.group(1)}"
                current_sub = None
                current_text = []
                unlabelled_stem_mode = True
                choice_frame = True
                for stem in iter_unlabelled_stems(line):
                    nxt = _next_unlabelled_sub(current_sub)
                    if not nxt:
                        break
                    flush_current()
                    current_sub = nxt
                    current_text = [stem]
                    current_origin = "inferred_stem"
                continue
            if rest and any(x in rest.lower() for x in ("attempt", "solve", "answer", "following", "compulsory", "marks")) and not any(
                re.search(rf"\b{v}\b", rest.lower()) for v in ACADEMIC_QUESTION_VERBS
            ):
                flush_current()
                current_parent = f"Q{m_parent.group(1)}"
                current_sub = None
                current_text = []
                choice_frame = True
                unlabelled_stem_mode = False
                continue
            # Q3. Explain ...  (no explicit sub-letter). Only call it (a) when
            # the paper actually subdivides this parent; otherwise it is a flat
            # question and inventing a subquestion would falsify its identity.
            if rest and any(re.search(rf"\b{v}\b", rest.lower()) for v in ACADEMIC_QUESTION_VERBS):
                flush_current()
                current_parent = f"Q{m_parent.group(1)}"
                current_sub = "a" if parent_has_explicit_subs(idx) else None
                if re.search(r"\b\d+\s*(?:Marks?|mark|M)\b", line, re.I):
                    current_marks = extract_marks(line)
                current_text = [rest]
                current_origin = "marker"
                choice_frame = False
                unlabelled_stem_mode = False
                continue
            if not rest:
                flush_current()
                current_parent = f"Q{m_parent.group(1)}"
                current_sub = None
                current_text = []
                choice_frame = False
                unlabelled_stem_mode = False
                continue
            # Non-question residue after Qn — ignore rather than invent a record
            continue

        # Continuation — append to current question (never invent new IDs from digits)
        if current_parent:
            cleaned_line = _strip_trailing_marks(line)
            if not cleaned_line:
                continue
            low = cleaned_line.lower()
            glued_stems = (
                iter_unlabelled_stems(cleaned_line) if unlabelled_stem_mode else []
            )
            if unlabelled_stem_mode and (
                any(re.match(rf"{v}\b", low) for v in ACADEMIC_QUESTION_VERBS)
                or len(glued_stems) > 1
            ):
                prev = " ".join(current_text).strip()
                prev_done = (
                    not prev
                    or prev.endswith((".", "?", "!", ":", "]"))
                    or len(prev.split()) >= 6
                )
                if prev_done:
                    if (
                        current_sub
                        and len(current_sub) == 1
                        and current_sub >= "d"
                        and re.search(r"\[\s*10\s*\]", cleaned_line)
                        and len(glued_stems) <= 1
                    ):
                        unlabelled_stem_mode = False
                    else:
                        stems = glued_stems or [cleaned_line]
                        for stem in stems:
                            nxt = _next_unlabelled_sub(current_sub)
                            if not nxt:
                                break
                            flush_current()
                            current_sub = nxt
                            current_text = [stem]
                            current_origin = "inferred_stem"
                        continue
            if any(h in low for h in ("attempt any", "compulsory")):
                continue
            # Never treat "Q2(6)" style numeric garbage as a new question — already blocked by patterns
            if re.match(rf"^(?:Q\.?)?\d+\(\d+\)", cleaned_line, re.I):
                continue
            # Do not append header/footer/datetime residue into the active question
            if is_header_or_instruction(cleaned_line):
                continue
            if re.search(
                r"\b(?:dec|nov|may|june?|jan|feb|mar|apr|jul|aug|sep|oct)[a-z]*[-\s/]*20\d{2}\b",
                low,
            ) and not any(re.search(rf"\b{v}\b", low) for v in ACADEMIC_QUESTION_VERBS):
                continue
            current_text.append(cleaned_line)

    flush_current()

    # Deduplicate by ID within page. Printed markers outrank inferred stems,
    # and a parent's own instruction frame never wins over real item content.
    def _record_rank(q: Dict[str, Any]) -> Tuple[int, int]:
        text = q.get("exact_text") or ""
        is_frame = 1 if is_instruction_frame_text(text) else 0
        inferred = 1 if q.get("origin") == "inferred_stem" else 0
        return (-is_frame, -inferred, len(text))

    by_id: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        qid = q["question_id"]
        prev = by_id.get(qid)
        if prev is None or _record_rank(q) > _record_rank(prev):
            by_id[qid] = q
    unique_qs = list(by_id.values())

    # A record that merely repeats the parent instruction frame while the same
    # parent produced genuine item content is dropped entirely.
    if any(is_instruction_frame_text(q.get("exact_text") or "") for q in unique_qs):
        content_siblings = {
            str(q.get("parent_question") or "")
            for q in unique_qs
            if not is_instruction_frame_text(q.get("exact_text") or "")
        }
        unique_qs = [
            q for q in unique_qs
            if not (
                is_instruction_frame_text(q.get("exact_text") or "")
                and str(q.get("parent_question") or "") in content_siblings
            )
        ]

    accepted_questions: List[Dict[str, Any]] = []
    rejected_candidates: List[Dict[str, Any]] = []

    for q in unique_qs:
        exact_text = q["exact_text"]
        q_id = q["question_id"]

        if not is_valid_question_id(q_id):
            rejected_candidates.append(
                {"raw_text": exact_text, "reason": "invalid_question_id", "page": page_num, "question_id": q_id, "metrics": {}}
            )
            continue

        under_instruction = bool(q.get("under_instruction_parent")) or (
            page_has_instruction_frame
            and q.get("origin") == "marker"
            and len(exact_text) <= 80
            and not is_instruction_frame_text(exact_text)
        )
        is_valid, reason, metrics = validate_question_candidate(
            exact_text, under_instruction_parent=under_instruction
        )
        if not is_valid:
            rejected_candidates.append(
                {"raw_text": exact_text, "reason": reason, "page": page_num, "question_id": q_id, "metrics": metrics}
            )
            continue

        normalized_text = normalize_question_text(exact_text)
        canonical_concepts = CanonicalConceptExtractor.extract_canonical_concepts(
            exact_text, syllabus_topics=syllabus_topics
        )
        intent_rep = build_question_representation(q_id, exact_text)

        question_obj = {
            "question_id": q_id,
            "question_number": q_id,
            "parent_question": q["parent_question"],
            "subquestion": q.get("subquestion"),
            "exact_text": exact_text,
            "normalized_text": normalized_text,
            "detected_topics": canonical_concepts,
            "canonical_concepts": canonical_concepts,
            "question_intent": intent_rep["question_intent"],
            "question_type": intent_rep["question_type"],
            "entities": intent_rep["entities"],
            "constraints": intent_rep["constraints"],
            "syllabus_mapping": {
                "module": "Unmapped",
                "chapter": "Unmapped",
                "topic": canonical_concepts[0] if canonical_concepts else "Unmapped",
            },
            "confidence": 0.95,
            "year": year,
            "marks": q["marks"],
            "parent_marks": 0,
            "marks_total": q["marks"],
            "question_type_struct": "SINGLE",
            "source_file": source_file,
            "source_page": page_num,
            "subject": subject,
            "workspace_id": workspace_id,
            "extraction_method": "canonical_boundary_parse",
            "origin": q.get("origin", "marker"),
            "under_instruction_parent": under_instruction,
            "quality_score": 1.0,
            "rejected_question": False,
        }
        accepted_questions.append(question_obj)

    return accepted_questions, rejected_candidates


def compute_text_similarity(text1: str, text2: str) -> float:
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())
    if not tokens1 or not tokens2:
        return 0.0
    jaccard_all = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    core1 = {w for w in tokens1 if w not in GENERIC_ACTION_VERBS and len(w) > 2}
    core2 = {w for w in tokens2 if w not in GENERIC_ACTION_VERBS and len(w) > 2}
    if core1 and core2:
        core_intersection = core1.intersection(core2)
        core_union = core1.union(core2)
        core_jaccard = len(core_intersection) / len(core_union)
        core_overlap = len(core_intersection) / min(len(core1), len(core2))
        core_sim = 0.5 * core_jaccard + 0.5 * core_overlap
    else:
        core_sim = 0.0
    return round(0.75 * core_sim + 0.25 * jaccard_all, 3)


def _entity_overlap(e1: List[str], e2: List[str]) -> float:
    if not e1 or not e2:
        return 0.0
    s1 = {x.lower() for x in e1}
    s2 = {x.lower() for x in e2}
    exact = len(s1 & s2) / len(s1 | s2)

    # Token containment: "Dropout" vs "Dropout Problem Overfitting"
    def toks(s: set) -> set:
        out = set()
        for phrase in s:
            out.update(re.findall(r"[a-z0-9]{3,}", phrase.lower()))
        return out

    t1, t2 = toks(s1), toks(s2)
    if not t1 or not t2:
        return exact
    token_j = len(t1 & t2) / len(t1 | t2)
    # Bonus when a short entity is fully contained in the other set's tokens
    containment = 0.0
    for phrase in s1:
        pt = set(re.findall(r"[a-z0-9]{3,}", phrase))
        if pt and pt <= t2:
            containment = max(containment, 0.55)
    for phrase in s2:
        pt = set(re.findall(r"[a-z0-9]{3,}", phrase))
        if pt and pt <= t1:
            containment = max(containment, 0.55)
    return max(exact, token_j, containment)


def _constraint_overlap(c1: List[str], c2: List[str]) -> float:
    if not c1 and not c2:
        return 1.0
    if not c1 or not c2:
        # One unconstrained descriptive question vs lightly constrained — soft match
        return 0.55
    s1, s2 = set(c1), set(c2)
    if s1 & s2:
        return len(s1 & s2) / len(s1 | s2)
    # Compatible families (paraphrase of same intent)
    families = [
        {"advantages_disadvantages", "architecture_explanation"},
        {"architecture_explanation"},
        {"enumerate_types", "architecture_explanation"},
        {"advantages_disadvantages"},
    ]
    for fam in families:
        if (s1 & fam) and (s2 & fam):
            return 0.5
    # advantages vs plain explain of same topic — soft compatible
    soft_a = {"advantages_disadvantages"}
    if (s1 <= soft_a and not s2) or (s2 <= soft_a and not s1):
        return 0.55
    return len(s1 & s2) / len(s1 | s2)


def classify_repeat_relationship(
    similarity: float,
    norm1: str,
    norm2: str,
    q1_text: str = "",
    q2_text: str = "",
    intent1: Optional[Dict[str, Any]] = None,
    intent2: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Backward-compatible classifier returning (relationship, concept_label).

    Relationships:
      EXACT_REPEAT | SEMANTIC_REPEAT | PARAPHRASED_REPEAT (alias) |
      RELATED_TOPIC | RELATED (alias) | SAME_TOPIC | DIFFERENT
    """
    rel, concept, _conf, _reason = classify_repeat_relationship_full(
        similarity, norm1, norm2, q1_text, q2_text, intent1, intent2
    )
    # Legacy aliases expected by older tests/UI
    if rel == "SEMANTIC_REPEAT":
        return "PARAPHRASED_REPEAT", concept
    if rel == "RELATED_TOPIC":
        return "RELATED", concept
    return rel, concept


def classify_repeat_relationship_full(
    similarity: float,
    norm1: str,
    norm2: str,
    q1_text: str = "",
    q2_text: str = "",
    intent1: Optional[Dict[str, Any]] = None,
    intent2: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, float, str]:
    """
    Full classifier: (relationship, concept, confidence, reason).
    relationship in EXACT_REPEAT | SEMANTIC_REPEAT | RELATED_TOPIC | DIFFERENT
    """
    i1 = intent1 or build_question_representation("Q?", q1_text or norm1)
    i2 = intent2 or build_question_representation("Q?", q2_text or norm2)

    concepts = CanonicalConceptExtractor.extract_canonical_concepts(q1_text or norm1)
    concept = concepts[0] if concepts else (i1.get("entities") or ["Unmapped"])[0]

    n1 = norm1 or normalize_question_text(q1_text)
    n2 = norm2 or normalize_question_text(q2_text)

    if n1 and n2 and n1 == n2:
        return "EXACT_REPEAT", concept, 1.0, "Normalized text identical after OCR-safe cleanup"

    exam_acronym_stop = {
        "QB", "QP", "ID", "OR", "AND", "THE", "FOR", "NOT", "ANY", "ALL",
        "MAY", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "JUN", "JUL", "AUG",
        "SEP", "OCT", "PDF", "OCR", "BE", "ME", "II", "III", "IV", "VI",
    }

    def _source_acronyms(text: str) -> Set[str]:
        return {
            m.lower()
            for m in re.findall(r"\b[A-Z]{2,6}\b", text or "")
            if m not in exam_acronym_stop
        }

    named1 = _source_acronyms(q1_text)
    named2 = _source_acronyms(q2_text)
    disjoint_named = bool(named1 and named2 and named1.isdisjoint(named2))

    # Near-exact: tiny edit distance via high token similarity AND same intent signature
    if similarity >= 0.92 and i1.get("question_type") == i2.get("question_type"):
        if _entity_overlap(i1.get("entities", []), i2.get("entities", [])) >= 0.8:
            return (
                "EXACT_REPEAT",
                concept,
                round(0.9 + 0.1 * similarity, 3),
                "Near-identical wording with same question type and entities",
            )

    ent_ov = _entity_overlap(i1.get("entities", []), i2.get("entities", []))
    cons_ov = _constraint_overlap(i1.get("constraints", []), i2.get("constraints", []))
    same_type = i1.get("question_type") == i2.get("question_type")

    c1 = set(i1.get("constraints", []))
    c2 = set(i2.get("constraints", []))
    conflicting = (
        (("numerical_calculation" in c1) != ("numerical_calculation" in c2) and ("numerical_calculation" in c1 or "numerical_calculation" in c2))
        or (("comparison" in c1) != ("comparison" in c2) and ("comparison" in c1 or "comparison" in c2))
        or ("applications" in c1 and "applications" not in c2 and "architecture_explanation" in c2)
        or ("applications" in c2 and "applications" not in c1 and "architecture_explanation" in c1)
    )

    e1 = {x.lower() for x in i1.get("entities", [])}
    e2 = {x.lower() for x in i2.get("entities", [])}
    shared_core = e1 & e2
    # Generic academic vocabulary is shared by unrelated questions in every
    # subject, so it can never be the evidence that two questions are the same.
    shared_core -= GENERIC_DOMAIN_TERMS

    soft_focus = {
        "advantages", "disadvantages", "overfitting", "underfitting", "significance",
        "types", "type", "method", "methods", "problem", "detail", "details",
        "technique", "techniques", "including", "include", "process", "operation",
        "operations",
    }
    # Specific named concepts stay in `focus`: they are exactly what distinguishes
    # "explain X architecture" from "explain Y architecture".
    focus1 = {x for x in (e1 - shared_core) if x not in soft_focus}
    focus2 = {x for x in (e2 - shared_core) if x not in soft_focus}
    # Ignore compound phrases that only add soft words around a shared core entity
    def _harden(focus: set, core: set) -> set:
        hardened = set()
        for phrase in focus:
            toks = set(re.findall(r"[a-z0-9]{3,}", phrase)) - soft_focus
            if toks - core:
                hardened.add(phrase)
        return hardened

    focus1 = _harden(focus1, shared_core)
    focus2 = _harden(focus2, shared_core)
    divergent_focus = bool(focus1 and focus2 and focus1.isdisjoint(focus2))
    # Asymmetric facet: one question names an extra specific concept the other lacks
    # (e.g. CNN architecture vs pooling in CNN) → RELATED, not semantic repeat.
    asymmetric_focus = bool((focus1 - focus2) or (focus2 - focus1)) and not (focus1 & focus2)

    strong_shared = bool(shared_core) and any(len(x) >= 4 for x in shared_core)
    ent_threshold = 0.22 if strong_shared else 0.35
    if (
        same_type
        and ent_ov >= ent_threshold
        and cons_ov >= 0.34
        and not conflicting
        and not divergent_focus
        and not disjoint_named
        and not (asymmetric_focus and similarity < 0.72)
        and similarity >= 0.20
        and shared_core
    ):
        conf = round(0.55 * ent_ov + 0.25 * cons_ov + 0.20 * min(1.0, similarity / 0.7), 3)
        if strong_shared:
            conf = max(conf, 0.62) if conf >= 0.40 else conf
        if conf >= 0.45:
            return (
                "SEMANTIC_REPEAT",
                concept,
                max(conf, 0.62),
                f"Same question intent ({i1.get('question_type')}); entity overlap={ent_ov:.2f}, constraint overlap={cons_ov:.2f}",
            )

    if ent_ov >= 0.25 or similarity >= 0.22:
        if conflicting or not same_type or cons_ov < 0.34 or divergent_focus or asymmetric_focus or disjoint_named:
            return (
                "RELATED_TOPIC",
                concept,
                round(max(ent_ov, similarity * 0.5), 3),
                "Shared topic vocabulary but different question intent/constraints — not a repeat",
            )
        # Broad topical overlap without enough intent evidence
        return (
            "RELATED_TOPIC",
            concept,
            round(max(ent_ov, similarity * 0.4), 3),
            "Topic overlap only — not enough evidence for question-level repeat",
        )

    return "DIFFERENT", concept, 0.0, "Insufficient evidence of recurrence"


def analyze_single_paper_patterns(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    patterns = []
    n = len(questions)
    if n < 2:
        return patterns

    for i in range(n):
        for j in range(i + 1, n):
            q1, q2 = questions[i], questions[j]
            n1 = q1.get("normalized_text") or normalize_question_text(q1.get("exact_text", ""))
            n2 = q2.get("normalized_text") or normalize_question_text(q2.get("exact_text", ""))
            sim = compute_text_similarity(n1, n2)
            i1 = {
                "question_type": q1.get("question_type") or detect_question_type(q1.get("exact_text", "")),
                "entities": q1.get("entities") or extract_entities(q1.get("exact_text", "")),
                "constraints": q1.get("constraints") or extract_constraints(q1.get("exact_text", "")),
            }
            i2 = {
                "question_type": q2.get("question_type") or detect_question_type(q2.get("exact_text", "")),
                "entities": q2.get("entities") or extract_entities(q2.get("exact_text", "")),
                "constraints": q2.get("constraints") or extract_constraints(q2.get("exact_text", "")),
            }
            relationship, concept, confidence, reason = classify_repeat_relationship_full(
                sim, n1, n2, q1.get("exact_text", ""), q2.get("exact_text", ""), i1, i2
            )
            if relationship in {"EXACT_REPEAT", "SEMANTIC_REPEAT", "RELATED_TOPIC"}:
                # Preserve legacy alias in within-paper output for older UI
                display_rel = (
                    "PARAPHRASED_REPEAT" if relationship == "SEMANTIC_REPEAT"
                    else ("RELATED" if relationship == "RELATED_TOPIC" else relationship)
                )
                patterns.append(
                    {
                        "q1_id": q1["question_id"],
                        "q1_number": q1.get("question_number", q1["question_id"]),
                        "q1_text": q1.get("exact_text", ""),
                        "q2_id": q2["question_id"],
                        "q2_number": q2.get("question_number", q2["question_id"]),
                        "q2_text": q2.get("exact_text", ""),
                        "concept": concept,
                        "relationship": display_rel,
                        "relationship_canonical": relationship,
                        "similarity": sim,
                        "confidence": confidence,
                        "reason": reason,
                    }
                )

    patterns.sort(key=lambda x: (x.get("confidence", 0), x.get("similarity", 0)), reverse=True)
    return patterns
