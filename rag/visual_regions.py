"""
Geometry-only table / box / image attachment.

No subject rules. A region belongs to the question whose vertical span
contains the region's midpoint. Cue words never create a question.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_visual_regions(page) -> List[Dict[str, Any]]:
    regions: List[Dict[str, Any]] = []
    page_no = getattr(page, "number", 0) + 1
    try:
        tables = page.find_tables()
        for tbl in getattr(tables, "tables", []) or []:
            bbox = getattr(tbl, "bbox", None)
            if not bbox:
                continue
            try:
                data = tbl.extract()
            except Exception:
                data = []
            rows = []
            for row in data or []:
                cells = [str(c).strip() for c in (row or []) if c]
                if cells:
                    rows.append(" | ".join(cells))
            text = "\n".join(rows).strip()
            if text:
                regions.append(_region("table", page_no, bbox, text, native_ok=True))
    except Exception:
        pass

    try:
        for d in page.get_drawings() or []:
            rect = d.get("rect")
            if rect is None:
                continue
            width = float(rect.x1 - rect.x0)
            height = float(rect.y1 - rect.y0)
            if width < page.rect.width * 0.15 or height < 28:
                continue
            try:
                clip = page.get_text("text", clip=rect) or ""
            except Exception:
                clip = ""
            clip = clip.strip()
            if clip:
                regions.append(_region("box", page_no, rect, clip, native_ok=True))
    except Exception:
        pass

    try:
        for img in page.get_images(full=True) or []:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for rect in rects or []:
                regions.append(
                    _region("image", page_no, rect, f"[diagram on page {page_no}]", native_ok=False)
                )
    except Exception:
        pass
    return regions


def attach_regions_to_questions(
    questions: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    marker_spans: Optional[List[Dict[str, Any]]] = None,
    page_height: float = 842.0,
) -> List[Dict[str, Any]]:
    if not questions or not regions:
        return questions
    spans = marker_spans or _spans_from_questions(questions, page_height=page_height)
    for region in regions:
        owner = _owner_for_region(region, spans)
        if not owner:
            continue
        q = next((x for x in questions if x.get("question_id") == owner), None)
        if not q:
            continue
        block = f"\n[{region['kind'].upper()}]\n{region['text']}"
        exact = q.get("exact_text") or ""
        if region["text"] and region["text"] not in exact:
            q["exact_text"] = (exact + block).strip()
        appendix = q.setdefault("visual_appendix", [])
        appendix.append({
            "kind": region["kind"],
            "page": region["page"],
            "text": region["text"],
            "y0": region.get("y0"),
            "y1": region.get("y1"),
        })
    return questions


def _region(kind: str, page: int, bbox, text: str, *, native_ok: bool) -> Dict[str, Any]:
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return {
        "kind": kind,
        "page": page,
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "mid_y": (y0 + y1) / 2.0,
        "text": text,
        "native_ok": native_ok,
    }


def _spans_from_questions(
    questions: List[Dict[str, Any]],
    page_height: float = 842.0,
) -> List[Dict[str, Any]]:
    """
    Vertical ownership bands.

    Prefer explicit layout_y0/y1 / source_span geometry when present.
    Otherwise assign sequential bands in document order so a table in the
    lower half of a multi-question page is not given to every question.
    """
    from collections import defaultdict

    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for q in questions:
        pages = q.get("source_pages") or [q.get("source_page") or 1]
        try:
            page = int(pages[0] if pages else 1)
        except (TypeError, ValueError):
            page = 1
        by_page[page].append(q)

    spans: List[Dict[str, Any]] = []
    height = float(page_height or 842.0)
    for page, items in by_page.items():
        n = max(1, len(items))
        for idx, q in enumerate(items):
            y0 = q.get("layout_y0")
            y1 = q.get("layout_y1")
            if y0 is None:
                y0 = (q.get("source_span") or {}).get("y0") if isinstance(q.get("source_span"), dict) else None
            if y1 is None:
                y1 = (q.get("source_span") or {}).get("y1") if isinstance(q.get("source_span"), dict) else None
            if y0 is not None and y1 is not None:
                try:
                    spans.append({
                        "question_id": q.get("question_id"),
                        "page": page,
                        "y0": float(y0),
                        "y1": float(y1),
                    })
                    continue
                except (TypeError, ValueError):
                    pass
            band0 = (idx / n) * height
            band1 = ((idx + 1) / n) * height
            pad = height * 0.02
            spans.append({
                "question_id": q.get("question_id"),
                "page": page,
                "y0": max(0.0, band0 - pad),
                "y1": min(height, band1 + pad),
            })
    return spans


def _owner_for_region(region: Dict[str, Any], spans: List[Dict[str, Any]]) -> Optional[str]:
    page = region["page"]
    mid = region["mid_y"]
    candidates: List[tuple] = []
    for span in spans:
        if int(span.get("page") or 0) != page:
            continue
        y0 = float(span.get("y0") or 0)
        y1 = float(span.get("y1") or 99999)
        if y0 <= mid <= y1:
            center = (y0 + y1) / 2.0
            candidates.append((abs(mid - center), span.get("question_id")))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    same_page = [s for s in spans if int(s.get("page") or 0) == page]
    if len(same_page) == 1:
        return same_page[0].get("question_id")
    return None
