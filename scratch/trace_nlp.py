"""End-to-end stage dump for one real PDF. No subject assumptions."""
from __future__ import annotations

import json
import os
import sys

import fitz

sys.path.insert(0, r"D:\pyqrag")

from rag.dynamic_ingest import (
    DynamicIngestPipeline,
    filter_noise_lines,
    perform_ocr_page,
)
from rag.hybrid_question_extraction import (
    compute_extraction_quality,
    detect_source_question_markers,
    hybrid_extract_document,
)
from rag.ocr_layout import _classify, ocr_layout_text, ocr_page_lines, reconstruct_questions_from_layout
from rag.question_extractor import (
    extract_questions_from_page_text,
    prepare_page_text_for_extraction,
    question_structure_score,
)

PDF = (
    r"D:\pyqrag\data\uploads\ws-nlp-31567ba5"
    r"\be_computer-engineering_semester-7_2022_december_"
    r"dloc-iii-natural-language-processing-rev-2019-c-scheme.pdf"
)
WS = "ws-nlp-trace"
OUT = r"D:\pyqrag\scratch\extraction_audits\nlp_stage_trace.json"


def clip(s, n=400):
    s = (s or "").replace("\x00", "")
    return s if len(s) <= n else s[:n] + " …"


def eval_repr(name, text, page_no, filename):
    if not (text or "").strip():
        return {
            "name": name,
            "empty": True,
            "chars": 0,
            "prepared": "",
            "markers": [],
            "accepted": [],
            "rejected": [],
        }
    prepared = prepare_page_text_for_extraction(text)
    acc, rej = extract_questions_from_page_text(prepared, page_no, filename, WS, year=2022)
    marks = detect_source_question_markers(prepared)
    return {
        "name": name,
        "empty": False,
        "chars": len(text),
        "prepared_chars": len(prepared),
        "prepared_head": clip(prepared, 800),
        "markers": marks,
        "accepted_ids": [q["question_id"] for q in acc],
        "accepted": [
            {
                "id": q["question_id"],
                "marks": q.get("marks"),
                "text": q.get("exact_text", "")[:240],
                "len": len(q.get("exact_text") or ""),
            }
            for q in acc
        ],
        "rejected": [
            {
                "id": r.get("question_id"),
                "reason": r.get("reason") or r.get("rejection_reason"),
                "text": (r.get("raw_text") or r.get("text") or "")[:160],
            }
            for r in rej
        ],
        "structure_score": question_structure_score([q["question_id"] for q in acc]),
        "prepared": prepared,
        "source": text,
    }


def main():
    pipe = DynamicIngestPipeline()
    doc = fitz.open(PDF)
    filename = os.path.basename(PDF)
    report = {"pdf": filename, "pages": doc.page_count, "page_traces": []}
    pages_payload = []

    print(f"PDF {filename} pages={doc.page_count}")

    for i, page in enumerate(doc):
        page_no = i + 1
        raw_native = page.get_text() or ""
        filtered = filter_noise_lines(raw_native)
        valid, reason, metrics = pipe.validate_text_quality(filtered)
        ocr_raw = perform_ocr_page(page, dpi=150) or ""
        ocr_filtered = filter_noise_lines(ocr_raw)
        layout_lines = ocr_page_lines(page, dpi=150)
        layout_text = reconstruct_questions_from_layout(layout_lines)

        classified = []
        for ln in layout_lines:
            kind, parent, sub, rest = _classify(ln["text"])
            classified.append(
                {
                    "x0": ln["x0"],
                    "top": ln["top"],
                    "kind": kind,
                    "parent": parent,
                    "sub": sub,
                    "text": ln["text"][:140],
                    "rest": (rest or "")[:80],
                }
            )

        reps = {
            "native": eval_repr("native", raw_native, page_no, filename),
            "filtered_native": eval_repr("filtered_native", filtered, page_no, filename),
            "ocr_text": eval_repr("ocr_text", ocr_filtered, page_no, filename),
            "ocr_layout": eval_repr("ocr_layout", layout_text, page_no, filename),
        }

        print(f"\n=== PAGE {page_no} native_valid={valid} reason={reason} ===")
        print(f"native chars={len(raw_native)} filtered={len(filtered)} ocr={len(ocr_raw)} layout={len(layout_text)}")
        print(f"quality {metrics}")
        for name, r in reps.items():
            print(
                f"  {name:16s} markers={len(r.get('markers') or [])} "
                f"acc={len(r.get('accepted_ids') or [])} rej={len(r.get('rejected') or [])} "
                f"struct={r.get('structure_score')} ids={r.get('accepted_ids')}"
            )
            if r.get("markers") and set(r.get("markers") or []) - set(r.get("accepted_ids") or []):
                print(f"      marker-not-accepted: {sorted(set(r['markers']) - set(r.get('accepted_ids') or []))}")
            for rej in (r.get("rejected") or [])[:8]:
                print(f"      rejected {rej.get('id')} <- {rej.get('reason')} | {rej.get('text')}")

        print("--- layout classified ---")
        for c in classified:
            print(
                f"  x0={c['x0']:4d} {c['kind']:12s} p={str(c['parent']):4s} "
                f"s={str(c['sub']):4s} {c['text']}"
            )
        print("--- layout reconstructed ---")
        print(layout_text or "<empty>")
        print("--- native head ---")
        print(clip(raw_native, 600))
        print("--- ocr head ---")
        print(clip(ocr_raw, 600))

        best = max(
            [r for r in reps.values() if not r.get("empty")],
            key=lambda r: (
                len(r.get("accepted_ids") or []),
                r.get("structure_score") or 0,
            ),
            default=None,
        )
        pages_payload.append(
            {
                "page": page_no,
                "raw_native_text": raw_native,
                "raw_ocr_text": ocr_raw,
                "reconstructed_text": (best or {}).get("prepared") or "",
                "ocr_used": True,
            }
        )
        report["page_traces"].append(
            {
                "page": page_no,
                "native_valid": valid,
                "quality_reason": reason,
                "quality_metrics": metrics,
                "native_chars": len(raw_native),
                "ocr_chars": len(ocr_raw),
                "layout_chars": len(layout_text),
                "classified": classified,
                "layout_reconstructed": layout_text,
                "reps": {
                    k: {kk: vv for kk, vv in v.items() if kk not in ("prepared", "source")}
                    for k, v in reps.items()
                },
            }
        )

    doc.close()

    hybrid = hybrid_extract_document(
        pages_payload, filename=filename, workspace_id=WS, subject="Subject", year=2022
    )
    acc = hybrid["accepted_questions"]
    quality = hybrid["quality"]
    report["hybrid"] = {
        "accepted_ids": [q["question_id"] for q in acc],
        "accepted": [
            {
                "id": q["question_id"],
                "marks": q.get("marks"),
                "pages": q.get("source_pages"),
                "text": q.get("exact_text"),
            }
            for q in acc
        ],
        "rejected": [
            {
                "id": r.get("question_id"),
                "reason": r.get("reason") or r.get("rejection_reason"),
                "text": (r.get("raw_text") or "")[:200],
            }
            for r in (hybrid.get("rejected_candidates") or [])
        ],
        "markers": hybrid.get("source_markers"),
        "quality": quality,
        "cross_page_merges": hybrid.get("cross_page_merges"),
        "grounding_coverage": hybrid.get("grounding_coverage"),
    }
    print("\n=== DOCUMENT RECONCILIATION ===")
    print("markers", hybrid.get("source_markers"))
    print("accepted", [q["question_id"] for q in acc])
    print("quality", quality)
    print("rejected", [(r.get("question_id"), r.get("reason")) for r in (hybrid.get("rejected_candidates") or [])])
    missing = quality.get("missing_questions") or []
    print("MISSING", missing)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
