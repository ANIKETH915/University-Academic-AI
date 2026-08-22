"""
Adversarial real-document check.

AIDSAIML-SEM-7_2024_compressed.pdf is a real, scanned, 80-page multi-subject
SYLLABUS full of numbered lab-experiment lists ("1. Install Hadoop ...") that
superficially resemble exam questions. Feeding it through the PYQ path must not
produce a paper's worth of fabricated questions.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\pyqrag")
os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.vector_store import VectorStore

PDF = r"D:\pyqrag\data\pyq\deep-learning\AIDSAIML-SEM-7_2024_compressed.pdf"
WS = "ws-probe-syllabus-as-pyq"

store = VectorStore()
store.delete_by_workspace(WS)
pipe = DynamicIngestPipeline(vector_store=store)

metas = pipe.parse_pyq_pdf(PDF, {"id": WS, "subject": "Unknown Subject"})
audit = pipe.last_pyq_questions_audit or {}
summary = audit.get("quality_summary") or {}

res = store.collection.get(where={"workspace_id": {"$eq": WS}})
vectors = len(res.get("ids") or [])

report = {
    "file": os.path.basename(PDF),
    "pages": summary.get("total_pages"),
    "extraction_quality": audit.get("extraction_quality"),
    "ingestion_status": audit.get("ingestion_status"),
    "questions_accepted": len(audit.get("accepted_questions") or []),
    "source_markers_detected": summary.get("source_markers_detected"),
    "rejected_candidates": len(audit.get("rejected_candidates") or []),
    "rejection_reasons": summary.get("rejection_reasons", [])[:12],
    "vectors_inserted": vectors,
    "grounding_coverage": summary.get("grounding_coverage"),
    "ocr_pages": summary.get("ocr_pages"),
}
print(json.dumps(report, indent=2))

sample = [
    {"id": q.get("question_id"), "text": (q.get("exact_text") or "")[:110]}
    for q in (audit.get("accepted_questions") or [])[:15]
]
print("SAMPLE ACCEPTED:")
print(json.dumps(sample, indent=2))

out_dir = r"D:\pyqrag\scratch\extraction_audits"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "syllabus_as_pyq_probe.json"), "w", encoding="utf-8") as fh:
    json.dump({"report": report, "sample_accepted": sample}, fh, indent=2)

store.delete_by_workspace(WS)
print("cleaned probe workspace")
