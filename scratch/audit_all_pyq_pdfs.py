"""Full-stage extraction audit for real university PYQ PDFs."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz
from rag.dynamic_ingest import DynamicIngestPipeline, filter_noise_lines, perform_ocr_page
from rag.question_extractor import prepare_page_text_for_extraction
from rag.hybrid_question_extraction import (
    detect_source_question_markers,
    hybrid_extract_document,
)


OUT_DIR = os.path.join(os.path.dirname(__file__), "extraction_audits")
os.makedirs(OUT_DIR, exist_ok=True)


def audit_pdf(pdf_path: str) -> dict:
    pipe = DynamicIngestPipeline.__new__(DynamicIngestPipeline)
    doc = fitz.open(pdf_path)
    pages_payload = []
    page_reports = []

    for i in range(len(doc)):
        page = doc[i]
        raw_native = page.get_text() or ""
        filtered = filter_noise_lines(raw_native)
        ok, reason, metrics = DynamicIngestPipeline.validate_text_quality(pipe, filtered)

        if ok and len(raw_native) >= 800 and len(filtered) < max(120, int(0.25 * len(raw_native))):
            ok = False
            reason = "filtered_text_collapsed_from_garbled_page"

        ocr_raw = ""
        ocr_used = False
        final = filtered
        if not ok:
            ocr_raw = perform_ocr_page(page, dpi=150) or ""
            if ocr_raw:
                ocr_filt = filter_noise_lines(ocr_raw)
                ocr_ok, ocr_reason, ocr_metrics = DynamicIngestPipeline.validate_text_quality(pipe, ocr_filt)
                if ocr_ok:
                    final = ocr_filt
                    ok = True
                    reason = "valid_via_ocr"
                    metrics = ocr_metrics
                    ocr_used = True
                else:
                    reason = ocr_reason or reason

        reconstructed = prepare_page_text_for_extraction(final) if ok else ""
        markers = detect_source_question_markers(reconstructed or final or ocr_raw or raw_native)

        report = {
            "page": i + 1,
            "native_len": len(raw_native),
            "filtered_native_len": len(filtered),
            "native_quality_ok": ok if not ocr_used else False,
            "quality_reason": reason,
            "quality_metrics": metrics,
            "ocr_triggered": ocr_used or (not ok and bool(ocr_raw)),
            "ocr_used_as_final": ocr_used,
            "ocr_len": len(ocr_raw),
            "reconstructed_len": len(reconstructed),
            "markers": markers,
            "native_head": raw_native[:400],
            "native_tail": raw_native[-300:] if raw_native else "",
            "ocr_head": ocr_raw[:400] if ocr_raw else "",
            "reconstructed_head": reconstructed[:600],
            "reconstructed_full": reconstructed,
            "raw_native_full": raw_native,
            "raw_ocr_full": ocr_raw,
            "filtered_full": filtered,
        }
        page_reports.append(report)
        pages_payload.append(
            {
                "page": i + 1,
                "raw_native_text": raw_native,
                "raw_ocr_text": ocr_raw,
                "reconstructed_text": reconstructed,
                "ocr_used": ocr_used,
            }
        )
    doc.close()

    hybrid = hybrid_extract_document(
        pages_payload,
        filename=os.path.basename(pdf_path),
        workspace_id="ws-audit",
        subject="Deep Learning",
        year=2023,
    )

    return {
        "pdf": os.path.basename(pdf_path),
        "path": pdf_path,
        "pages": page_reports,
        "hybrid": {
            "accepted": [
                {"question_id": q["question_id"], "exact_text": q["exact_text"], "marks": q.get("marks")}
                for q in (hybrid.get("accepted_questions") or [])
            ],
            "rejected": [
                {
                    "question_id": r.get("question_id"),
                    "reason": r.get("reason"),
                    "page": r.get("page"),
                    "text": (r.get("raw_text") or "")[:200],
                }
                for r in (hybrid.get("rejected_candidates") or [])
            ],
            "markers": hybrid.get("source_markers"),
            "quality": hybrid.get("quality"),
            "llm_used": hybrid.get("llm_used"),
        },
    }


def main():
    paths = sorted(
        p
        for p in glob.glob(r"D:\pyqrag\data\pyq\deep-learning\*.pdf")
        if "Copy" not in p and "compressed" not in p.lower()
    )
    # Also include syllabus-sized? No — only PYQs
    summary = []
    for pdf in paths:
        print("=" * 70)
        print("AUDITING", os.path.basename(pdf))
        try:
            result = audit_pdf(pdf)
        except Exception as e:
            print("ERROR", e)
            summary.append({"pdf": os.path.basename(pdf), "error": str(e)})
            continue
        q = result["hybrid"]["quality"]
        print(
            "accepted",
            len(result["hybrid"]["accepted"]),
            "rejected",
            len(result["hybrid"]["rejected"]),
            "quality",
            q,
        )
        for p in result["pages"]:
            print(
                f"  P{p['page']} native={p['native_len']} ocr={p['ocr_used_as_final']}/{p['ocr_len']} "
                f"recon={p['reconstructed_len']} markers={p['markers']} reason={p['quality_reason']}"
            )
        for a in result["hybrid"]["accepted"]:
            print(f"  OK {a['question_id']}: {a['exact_text'][:90]}")
        for r in result["hybrid"]["rejected"][:12]:
            print(f"  REJ {r.get('question_id')} {r.get('reason')}: {r.get('text')}")

        out = os.path.join(OUT_DIR, os.path.basename(pdf).replace(".pdf", "")[:80] + "_audit.json")
        # Shrink full texts for JSON size but keep heads
        slim = {
            "pdf": result["pdf"],
            "path": result["path"],
            "pages": [
                {
                    **{k: v for k, v in p.items() if not k.endswith("_full")},
                }
                for p in result["pages"]
            ],
            "full_texts": {
                f"page_{p['page']}": {
                    "native": p["raw_native_full"],
                    "ocr": p["raw_ocr_full"],
                    "filtered": p["filtered_full"],
                    "reconstructed": p["reconstructed_full"],
                }
                for p in result["pages"]
            },
            "hybrid": result["hybrid"],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=2)
        print("wrote", out)
        summary.append(
            {
                "pdf": result["pdf"],
                "accepted": len(result["hybrid"]["accepted"]),
                "rejected": len(result["hybrid"]["rejected"]),
                "quality": q.get("extraction_quality") if q else None,
                "markers": q.get("source_markers_detected") if q else None,
            }
        )

    with open(os.path.join(OUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
