"""
Audit every distinct real PYQ PDF found under data/ and scratch/.

Writes scratch/restore_points/20260823/universe_pdf_matrix.json
Does not encode subject-specific expectations. Reports honestly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_universe_matrix")

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.pyq_intelligence import PYQIntelligenceEngine
from rag.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [ROOT / "data" / "pyq", ROOT / "data" / "uploads", ROOT / "scratch"]
OUT = ROOT / "scratch" / "restore_points" / "20260823" / "universe_pdf_matrix.json"


def _iter_pdfs():
    seen = set()
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.pdf"):
            if "Copy" in path.name or "corrupted" in str(path).lower():
                continue
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            yield path


def main():
    store = VectorStore()
    pipe = DynamicIngestPipeline(vector_store=store)
    rows = []
    for pdf in _iter_pdfs():
        ws_id = "ws-univ-" + hashlib.sha1(pdf.name.encode("utf-8")).hexdigest()[:10]
        try:
            store.delete_by_workspace(ws_id)
        except Exception:
            pass
        row = {
            "pdf": pdf.name,
            "path": str(pdf),
            "subject": "unspecified",
            "pages": None,
            "genuine_questions": None,
            "extracted_questions": [],
            "missing_genuine_ids": [],
            "fabricated_ids": [],
            "duplicates": [],
            "status": "ERROR",
            "vectors": 0,
            "representation": None,
            "confidence": None,
        }
        try:
            import fitz
            doc = fitz.open(str(pdf))
            row["pages"] = len(doc)
            doc.close()
        except Exception:
            pass
        try:
            metas = pipe.parse_pyq_pdf(
                str(pdf),
                {
                    "id": ws_id,
                    "subject": "Audit Subject",
                    "university": "Audit Institution",
                    "semester": "Unknown",
                },
            )
            audit = pipe.last_pyq_questions_audit or {}
            accepted = audit.get("accepted_questions") or []
            ids = [q.get("question_id") for q in accepted if q.get("question_id")]
            quality = (audit.get("quality_summary") or {}).get("extraction_quality") or audit.get("extraction_quality")
            genuine = ((audit.get("extraction_audit") or {}).get("missing_genuine_questions") and None)
            missing = (audit.get("quality_summary") or {}).get("missing_questions") or audit.get("missing_questions") or []
            if isinstance(missing, dict):
                missing = missing.get("missing_questions") or []
            row.update({
                "extracted_questions": ids,
                "missing_genuine_ids": missing if isinstance(missing, list) else [],
                "fabricated_ids": (audit.get("quality_summary") or {}).get("fabricated_ids") or [],
                "duplicates": sorted({i for i in ids if ids.count(i) > 1}),
                "status": quality or "UNKNOWN",
                "vectors": len(metas or []),
                "representation": (audit.get("extraction_audit") or {}).get("representation_sources"),
                "confidence": (audit.get("quality_summary") or {}).get("confidence") or audit.get("question_extraction_confidence"),
                "genuine_questions": (audit.get("quality_summary") or {}).get("genuine_markers")
                or (audit.get("extraction_audit") or {}).get("reconciled_questions"),
            })
            _ = genuine
        except Exception as exc:
            row["status"] = f"ERROR: {exc}"
        rows.append(row)
        print(f"{pdf.name}\t{row['status']}\textracted={len(row['extracted_questions'])}\tmissing={row['missing_genuine_ids']}")

    intel_ws = "ws-univ-intel"
    try:
        store.delete_by_workspace(intel_ws)
    except Exception:
        pass
    intelligence = {}
    try:
        engine = PYQIntelligenceEngine(vector_store=store)
        intelligence = engine.get_pyq_analysis(intel_ws)
    except Exception as exc:
        intelligence = {"error": str(exc)}

    report = {
        "pdf_count": len(rows),
        "by_status": defaultdict(int),
        "rows": rows,
        "intelligence_on_empty_universe_workspace": {
            "available": intelligence.get("available"),
            "total_questions": intelligence.get("total_valid_questions"),
        },
    }
    for r in rows:
        report["by_status"][str(r["status"]).split(":")[0]] += 1
    report["by_status"] = dict(report["by_status"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT} ({len(rows)} PDFs)")
    return report


if __name__ == "__main__":
    main()
