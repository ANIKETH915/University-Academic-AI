"""Real PDF extraction matrix — prints COMPLETE/PARTIAL/FAILED for every PYQ PDF."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.vector_store import VectorStore

OUT = os.path.join(os.path.dirname(__file__), "extraction_audits", "real_pdf_matrix.json")


def main():
    paths = sorted(
        p
        for p in glob.glob(r"D:\pyqrag\data\pyq\**\*.pdf", recursive=True)
        if "Copy" not in p and "compressed" not in p.lower() and "syllabus" not in p.lower()
    )
    # Prefer deep-learning + any other subject folders
    store = VectorStore()
    pipe = DynamicIngestPipeline(vector_store=store)
    rows = []
    for i, pdf in enumerate(paths):
        wid = f"ws-matrix-{i}"
        store.delete_by_workspace(wid)
        ws = {
            "id": wid,
            "subject": "Audit Subject",
            "university": "Audit U",
            "semester": "Semester 1",
            "subject_code": "AUD",
        }
        metas = pipe.parse_pyq_pdf(pdf, ws)
        audit = pipe.last_pyq_questions_audit or {}
        q = audit.get("quality_summary") or {}
        pages = audit.get("page_extraction_audit") or []
        row = {
            "file": os.path.basename(pdf),
            "pages": q.get("total_pages") or len(pages),
            "ocr_pages": q.get("ocr_pages"),
            "native_chars": sum(p.get("pymupdf_chars", 0) for p in pages),
            "final_questions": len(metas),
            "markers": q.get("source_markers_detected"),
            "missing": q.get("missing_questions"),
            "rejected": q.get("rejected_count"),
            "extraction_quality": audit.get("extraction_quality") or q.get("extraction_quality"),
            "ingestion_status": audit.get("ingestion_status"),
            "vectors_inserted": len(metas),
            "llm_used": q.get("llm_used"),
            "page_reasons": [p.get("quality_reason") for p in pages],
        }
        rows.append(row)
        print(
            f"{row['file'][:70]:70} q={row['final_questions']:3} "
            f"qual={row['extraction_quality']} vec={row['vectors_inserted']} "
            f"ocr_pages={row['ocr_pages']}"
        )

    summary = {
        "total_pdfs": len(rows),
        "complete": sum(1 for r in rows if r["extraction_quality"] == "COMPLETE"),
        "partial": sum(1 for r in rows if r["extraction_quality"] == "PARTIAL"),
        "failed": sum(1 for r in rows if r["extraction_quality"] == "FAILED"),
        "total_questions": sum(r["final_questions"] for r in rows),
        "total_vectors": sum(r["vectors_inserted"] for r in rows),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
