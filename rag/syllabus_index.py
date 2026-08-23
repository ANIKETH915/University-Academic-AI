"""
Subject-agnostic syllabus index built ONLY from uploaded syllabus documents
in the active workspace. Never invents modules from hardcoded subject catalogs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def empty_syllabus_index(subject: str = "Academic Subject") -> Dict[str, Any]:
    return {
        "subject": subject or "Academic Subject",
        "modules": [],
        "source": "none",
    }


def parse_unit_label(unit_str: str) -> Dict[str, str]:
    """Parse 'Module 4: Convolutional Networks' / 'Unit 2 Normalization' into parts with OCR cleanup."""
    if not unit_str:
        return {"module": "Unmapped", "chapter": "Unmapped"}

    u = re.sub(r"\bModule C:\s*ontent\b", "Module Content", unit_str, flags=re.I)
    u = re.sub(r"\s*\(Answer Any\s+[A-Za-z0-9]+\)\s*|\s*\(Any Two\)\s*", "", u, flags=re.I).strip(" -:,.()")

    m = re.match(
        r"^\s*(Module|Unit|Chapter|Section|Block)\s*([0-9IVX]+|[A-Z])\s*:?\s*(.*)$",
        u.strip(),
        re.IGNORECASE,
    )
    if m:
        kind = m.group(1).title()
        num = m.group(2)
        rest = (m.group(3) or "").strip()
        rest = re.sub(r"\bontent\b", "Content", rest, flags=re.I).strip(" -:,.()")
        return {"module": f"{kind} {num}", "chapter": rest if rest else f"{kind} {num}"}
    return {"module": "Unmapped", "chapter": u.strip()[:80]}


def extract_topics_from_text(text: str, limit: int = 40) -> List[str]:
    """Pull candidate topic phrases from syllabus chunk text (generic)."""
    if not text:
        return []
    # Strip header prefixes
    body = re.sub(r"^Syllabus Document\s*\[[^\]]+\]\s*", "", text).strip()
    topics: List[str] = []

    def _add(candidate: str) -> None:
        cleaned = re.sub(r"^[0-9]+[\.\)]\s*", "", candidate).strip(" -•*\t")
        if cleaned and 3 <= len(cleaned) <= 80 and cleaned not in topics:
            if not re.search(r"https?://|@|\.com\b", cleaned, re.I):
                topics.append(cleaned)

    # Bullet / numbered topic lines
    for line in body.splitlines():
        line = line.strip(" -•*\t")
        if not line or len(line) < 3:
            continue
        if re.match(r"^(Module|Unit|Chapter|Section|Block)\b", line, re.I):
            # Also capture title after the module label
            rest = re.sub(r"^(Module|Unit|Chapter|Section|Block)\s*[0-9IVX]+\s*:?\s*", "", line, flags=re.I).strip()
            if rest:
                _add(rest)
            continue
        _add(line)

    # Space-joined syllabus chunks: recover "- Topic" fragments
    for m in re.finditer(r"(?:^|[\s])[-•*]\s*([A-Za-z][A-Za-z0-9+\- /&()]{2,60})", body):
        _add(m.group(1).strip())

    # Acronym tokens
    for ac in re.findall(r"\b[A-Z]{2,6}\b", body):
        if ac.lower() not in {"pdf", "http", "https", "page"}:
            _add(ac)
    return topics[:limit]


def build_syllabus_index_from_chunks(
    chunks: List[Dict[str, Any]], subject: str = "Academic Subject"
) -> Dict[str, Any]:
    """
    Build syllabus index from ingested syllabus chunk metadatas/texts.
    Expected chunk shape: {"text": str, "metadata": dict} or metadata-only dicts.
    """
    modules_map: Dict[str, Dict[str, Any]] = {}

    for ch in chunks:
        meta = ch.get("metadata") if isinstance(ch, dict) and "metadata" in ch else ch
        text = ch.get("text", "") if isinstance(ch, dict) else ""
        if not isinstance(meta, dict):
            meta = {}

        unit = meta.get("unit") or meta.get("block") or ""
        parsed = parse_unit_label(str(unit))
        mod_key = parsed["module"]
        chap = parsed["chapter"]

        if mod_key == "Unmapped" and not text:
            continue

        if mod_key not in modules_map:
            modules_map[mod_key] = {
                "module": mod_key,
                "chapter": chap if chap != "Unmapped" else mod_key,
                "topics": [],
            }
        elif chap != "Unmapped" and modules_map[mod_key]["chapter"] in (mod_key, "Unmapped"):
            modules_map[mod_key]["chapter"] = chap

        topics = extract_topics_from_text(text)
        # Also split chapter title into topic seeds
        if chap and chap != "Unmapped":
            topics = [chap] + topics
        existing = modules_map[mod_key]["topics"]
        for t in topics:
            if t and t not in existing:
                existing.append(t)

    modules = [modules_map[k] for k in sorted(modules_map.keys(), key=lambda x: x)]
    return {
        "subject": subject or "Academic Subject",
        "modules": modules,
        "source": "uploaded_syllabus" if modules else "none",
    }


def build_syllabus_index_from_workspace(
    vector_store: Any, workspace_id: str, subject: str = "Academic Subject"
) -> Dict[str, Any]:
    """Load syllabus chunks from vector store for workspace_id and build syllabus index."""
    if not workspace_id:
        return empty_syllabus_index(subject)

    syllabus_chunks = []
    try:
        res = vector_store.collection.get(
            where={
                "$and": [
                    {"workspace_id": {"$eq": workspace_id}},
                    {"doc_type": {"$eq": "syllabus"}},
                ]
            }
        )
        if res and res.get("documents"):
            for doc_text, meta in zip(res["documents"], res["metadatas"]):
                syllabus_chunks.append({"text": doc_text, "metadata": meta})
    except Exception as ex:
        print(f"[SYLLABUS_INDEX] collection.get failed: {ex}")

    if not syllabus_chunks:
        return empty_syllabus_index(subject)

    return build_syllabus_index_from_chunks(syllabus_chunks, subject=subject)


def map_question_to_syllabus_index(
    question_text: str,
    detected_topics: List[str],
    syllabus_index: Dict[str, Any],
) -> Tuple[Dict[str, str], float]:
    """
    Map question to syllabus index modules.
    Returns ({module, chapter, topic}, confidence).
    Unmapped when confidence < 0.4.
    """
    if not syllabus_index or not syllabus_index.get("modules"):
        return {"module": "Unmapped", "chapter": "Unmapped", "topic": "Unmapped"}, 0.0

    q_lower = (question_text or "").lower()
    topics_lower = [t.lower() for t in (detected_topics or []) if t]

    best_module = "Unmapped"
    best_chapter = "Unmapped"
    best_topic = "Unmapped"
    best_score = 0.0

    for mod_info in syllabus_index["modules"]:
        mod_name = mod_info.get("module", "Unmapped")
        chap_name = mod_info.get("chapter", "Unmapped")
        mod_topics = mod_info.get("topics", [])

        # Match against module topics
        for t in mod_topics:
            t_low = t.lower()
            if len(t_low) >= 3:
                score = 0.0
                if t_low in q_lower:
                    score = 0.85
                elif any(dt in t_low or t_low in dt for dt in topics_lower):
                    score = 0.75
                elif len(t_low.split()) > 1:
                    matched_words = sum(1 for w in t_low.split() if len(w) > 3 and w in q_lower)
                    if matched_words >= 2:
                        score = 0.65

                if score > best_score:
                    best_score = score
                    best_module = mod_name
                    best_chapter = chap_name
                    best_topic = t

    if best_score >= 0.40:
        return {
            "module": best_module,
            "chapter": best_chapter,
            "topic": best_topic,
        }, best_score

    return {"module": "Unmapped", "chapter": "Unmapped", "topic": "Unmapped"}, 0.0
