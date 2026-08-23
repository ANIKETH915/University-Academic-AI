import os
import sys
import json
import fitz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.dynamic_ingest import DynamicIngestPipeline
from rag.workspace_db import WorkspaceDB
from rag.ocr_layout import ocr_page_lines

pdf_path = os.path.abspath("data/uploads/ws-computer-network-b1c007df/be_computer-engineering_semester-7_2023_may_big-data-analysis-rev-2019-c-scheme.pdf")
print("Target PDF Path:", pdf_path)
print("Exists:", os.path.exists(pdf_path))

class DummyVectorStore:
    def add_documents(self, *args, **kwargs): pass
    def delete_by_workspace(self, *args, **kwargs): pass
    def replace_documents_for_source(self, *args, **kwargs): pass
    def count(self): return 0

pipe = DynamicIngestPipeline(vector_store=DummyVectorStore())
ws_db = WorkspaceDB()
ws = ws_db.get_or_create("ws-se-2023-may-diag", subject="Software Engineering")

metas = pipe.parse_pyq_pdf(pdf_path, ws)
audit = pipe.last_pyq_questions_audit or {}

doc = fitz.open(pdf_path)
print("\n" + "="*80)
print("1. NATIVE PDF TEXT AROUND Q6:")
print("="*80)
for i, page in enumerate(doc):
    txt = page.get_text() or ""
    if "Q6" in txt or "Q.6" in txt or "6." in txt or "XP" in txt or "Six Sigma" in txt:
        print(f"--- Page {i+1} Native Text ---")
        print(txt)

print("\n" + "="*80)
print("2. PLAIN OCR TEXT AROUND Q6:")
print("="*80)
page_audits = audit.get("page_extraction_audit") or []
for pa in page_audits:
    print(f"--- Page {pa.get('page')} Audit ---")
    print(f"  OCR used: {pa.get('ocr_used')}, status: {pa.get('status')}")

print("\n" + "="*80)
print("3. QUALITY SUMMARY AROUND Q6:")
print("="*80)
quality_summary = audit.get("quality_summary") or {}
print("Quality Summary:", json.dumps(quality_summary, indent=2))

print("\n" + "="*80)
print("4. MARKER CANDIDATES DETECTED AROUND Q6:")
print("="*80)
ext_audit = audit.get("extraction_audit") or {}
marker_candidates = ext_audit.get("marker_candidates") or []
q6_mcs = [m for m in marker_candidates if str(m.get("marker_id")).startswith("Q6") or "Q6" in str(m.get("marker_id"))]
print("Q6 Marker Candidates:", json.dumps(q6_mcs, indent=2))

print("\n" + "="*80)
print("5. BODY CANDIDATES / ACCEPTED QUESTIONS AROUND Q6:")
print("="*80)
accepted = audit.get("accepted_questions") or []
for q in accepted:
    qid = q.get("question_id")
    if str(qid).startswith("Q6") or str(qid).startswith("Q.6"):
        print(f"ACCEPTED QID: {qid}")
        print(f"  Parent Marks: {q.get('parent_marks')}, Marks: {q.get('marks')}")
        print(f"  Method: {q.get('extraction_method')}")
        print(f"  Exact Text:\n{q.get('exact_text')}\n")

doc.close()

print("\n" + "="*80)
print("8. REJECTED CANDIDATES AROUND Q6:")
print("="*80)
rejected = audit.get("rejected_candidates") or []
for r in rejected:
    rqid = str(r.get("question_id") or r.get("marker_id") or "")
    if "Q6" in rqid or "6" in rqid:
        print(f"REJECTED QID: {rqid} | Reason: {r.get('reason')} | Text: {str(r.get('exact_text'))[:120]}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
