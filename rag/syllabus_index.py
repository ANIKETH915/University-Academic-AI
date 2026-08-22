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
    """Parse 'Module 4: Convolutional Networks' / 'Unit 2 Normalization' into parts."""
    if not unit_str:
        return {"module": "Unmapped", "chapter": "Unmapped"}
    m = re.match(
        r"^\s*(Module|Unit|Chapter|Section|Block)\s*([0-9IVX]+|[A-Z])\s*:?\s*(.*)$",
        unit_str.strip(),
        re.IGNORECASE,
    )
    if m:
        kind = m.group(1).title()
        num = m.group(2)
        rest = (m.group(3) or "").strip() or "Unmapped"
        return {"module": f"{kind} {num}", "chapter": rest}
    return {"module": "Unmapped", "chapter": unit_str.strip()[:80]}


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
    """Load syllabus vectors for workspace_id and build a subject-scoped index."""
    if not workspace_id or not vector_store:
        return empty_syllabus_index(subject)
    try:
        res = vector_store.collection.get(
            where={
                "$and": [
                    {"workspace_id": {"$eq": workspace_id}},
                    {"doc_type": {"$eq": "syllabus"}},
                ]
            }
        )
    except Exception:
        return empty_syllabus_index(subject)

    if not res or not res.get("documents"):
        return empty_syllabus_index(subject)

    chunks = []
    for doc, meta in zip(res["documents"], res.get("metadatas") or []):
        chunks.append({"text": doc, "metadata": meta or {}})
    return build_syllabus_index_from_chunks(chunks, subject=subject)


def map_question_to_syllabus_index(
    question_text: str,
    detected_topics: List[str],
    syllabus_index: Optional[Dict[str, Any]],
) -> tuple:
    """
    Map a question against an uploaded syllabus index only.
    Returns (mapping_dict, confidence). Unmapped when insufficient evidence.
    """
    best_topic = (detected_topics[0] if detected_topics else None) or "Unmapped"
    if not syllabus_index or not syllabus_index.get("modules"):
        return {"module": "Unmapped", "chapter": "Unmapped", "topic": best_topic}, 0.0

    q_lower = (question_text or "").lower()
    topics_lower = [t.lower() for t in (detected_topics or []) if t]
    # Prefer precise tokens (acronyms / multi-word topics) over vague words
    q_tokens = set(re.findall(r"[a-z0-9]{2,}", q_lower))

    def _norm_stem(s: str) -> str:
        s = s.lower().strip()
        if len(s) > 4 and s.endswith("s"):
            return s[:-1]
        return s

    q_stems = {_norm_stem(t) for t in q_tokens}

    best_module = None
    best_chapter = None
    best_score = 0
    best_hit = ""

    for m in syllabus_index["modules"]:
        mod_name = m.get("module", "")
        chap_name = m.get("chapter", "")
        mod_topics = m.get("topics", []) or []
        score = 0
        hit = ""
        for top in mod_topics:
            top_lower = str(top).lower().strip()
            if len(top_lower) < 3:
                continue
            top_stem = _norm_stem(top_lower)
            # Exact / contained topic phrase in question — strong signal
            if top_lower in q_lower or (len(top_stem) >= 5 and top_stem in q_lower):
                bonus = 6 if len(top_lower) >= 6 else 4
                if bonus >= 4:
                    hit = top_lower
                score += bonus
            elif any(
                t_low == top_lower
                or _norm_stem(t_low) == top_stem
                or (len(top_lower) >= 5 and (t_low in top_lower or top_lower in t_low))
                for t_low in topics_lower
            ):
                score += 3
                hit = hit or top_lower
            else:
                # Acronym / token hit (e.g. CNN)
                top_toks = set(re.findall(r"[a-z0-9]{2,}", top_lower))
                strong = set()
                for t in top_toks:
                    if len(t) < 3:
                        continue
                    if t in q_tokens or _norm_stem(t) in q_stems:
                        strong.add(t)
                # Ignore ultra-generic chapter words
                strong -= {"neural", "networks", "network", "models", "model", "learning", "system", "systems"}
                if strong:
                    score += 2 + min(3, len(strong))
                    hit = hit or next(iter(strong))
        for cw in [w.lower() for w in re.findall(r"[A-Za-z0-9]{4,}", str(chap_name))]:
            if cw in {"neural", "networks", "network", "models", "model", "learning", "system", "systems"}:
                continue
            if cw in q_lower or _norm_stem(cw) in q_stems:
                score += 2
                hit = hit or cw
        if score > best_score:
            best_score = score
            best_module = mod_name
            best_chapter = chap_name
            best_hit = hit

    if best_score >= 3 and best_module and best_module != "Unmapped":
        confidence = min(0.96, round(0.70 + (best_score * 0.04), 2))
        return {
            "module": best_module,
            "chapter": best_chapter or "Unmapped",
            "topic": best_topic if best_topic != "Unmapped" else (best_hit.title() if best_hit else "Unmapped"),
        }, confidence

    return {"module": "Unmapped", "chapter": "Unmapped", "topic": best_topic}, 0.0
