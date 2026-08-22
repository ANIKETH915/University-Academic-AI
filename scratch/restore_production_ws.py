"""Safely re-ingest production default workspace PYQs from on-disk PDFs."""
import os
from rag.workspace_db import WorkspaceDB
from rag.vector_store import VectorStore
from rag.dynamic_ingest import DynamicIngestPipeline

WS = "ws-default-workspace"
DL = r"d:/pyqrag/data/pyq/deep-learning"

ws_db = WorkspaceDB()
ws = ws_db.get_by_id(WS)
store = VectorStore()
ingest = DynamicIngestPipeline(vector_store=store)

print("Workspace:", ws.get("subject") if ws else None)
before = store.collection.get(where={"workspace_id": {"$eq": WS}})
print("Before vectors:", len(before.get("ids") or []))

if not ws:
    raise SystemExit("workspace missing")

pdfs = [
    os.path.join(DL, f)
    for f in os.listdir(DL)
    if f.endswith(".pdf") and "Copy" not in f and "compressed" not in f
]
print("PDFs:", len(pdfs))
total = 0
for p in pdfs:
    metas = ingest.parse_pyq_pdf(p, ws)
    print(os.path.basename(p), "->", len(metas))
    total += len(metas)

after = store.collection.get(where={"workspace_id": {"$eq": WS}})
print("After vectors:", len(after.get("ids") or []), "questions ingested:", total)
