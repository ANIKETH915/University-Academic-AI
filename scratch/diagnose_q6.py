import glob
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.workspace_db import WorkspaceDB

class DummyVectorStore:
    def add_documents(self, *args, **kwargs):
        pass
    def delete_by_workspace(self, *args, **kwargs):
        pass
    def count(self):
        return 0

pipe = DynamicIngestPipeline(vector_store=DummyVectorStore())
ws_db = WorkspaceDB()
ws = ws_db.get_or_create("ws-audit-diag", subject="Audit Subject")

pdf_files = []
for root in ["data/pyq", "data/uploads"]:
    if os.path.exists(root):
        for path in glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True):
            if "Copy" not in path:
                pdf_files.append(path)

print(f"Auditing {len(pdf_files)} PDFs...")
found = []

for pdf_path in pdf_files:
    fname = os.path.basename(pdf_path)
    try:
        metas = pipe.parse_pyq_pdf(pdf_path, ws)
        audit = pipe.last_pyq_questions_audit or {}
        quality = audit.get("quality_summary") or {}
        extracted = quality.get("questions_extracted")
        missing = quality.get("missing_questions") or []
        print(f"PDF: {fname} | Extracted: {extracted} | Missing: {missing}")
        if missing:
            found.append((pdf_path, extracted, missing, audit))
    except Exception as e:
        print(f"Error reading {fname}: {e}")

print("="*60)
print(f"TOTAL PDFs WITH MISSING MARKERS: {len(found)}")
for pdf, ext, miss, audit in found:
    print(f"\nPDF: {pdf}")
    print(f"  Extracted count: {ext}")
    print(f"  Missing: {miss}")
    print(f"  Quality summary: {audit.get('quality_summary')}")
