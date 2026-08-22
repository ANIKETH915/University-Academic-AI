"""
Live end-to-end proof against a running backend (no TestClient, no mocks).

Flow: health -> create workspace -> upload syllabus -> upload 4 real PYQ PDFs
-> verify extraction + vectors -> PYQ intelligence -> study priority
-> source questions -> ask a grounded question. Every request uses the same
canonical workspace_id returned by POST /workspaces.
"""
import glob
import json
import os
import sys

import requests

BASE = os.environ.get("PYQRAG_BASE_URL", "http://127.0.0.1:8000")
PYQ_DIR = r"D:\pyqrag\data\pyq\deep-learning"
SYL_DIR = r"D:\pyqrag\data"

report = {"base_url": BASE, "steps": []}


def step(name, ok, **extra):
    entry = {"step": name, "ok": bool(ok), **extra}
    report["steps"].append(entry)
    flag = "OK  " if ok else "FAIL"
    print(f"[{flag}] {name} {json.dumps(extra, default=str)[:300]}")
    return ok


health = requests.get(f"{BASE}/health", timeout=30).json()
baseline_vectors = health.get("vector_store_stats", {}).get("total_vectors")
step("health", health.get("status") == "ok",
     version=health.get("version"),
     vectors=health.get("vector_store_stats", {}).get("total_vectors"),
     llm_configured=health.get("llm", {}).get("configured"),
     llm_chain=health.get("llm", {}).get("provider_chain"))

created = requests.post(f"{BASE}/workspaces", json={
    "university": "Live E2E University",
    "branch": "Computer Engineering",
    "semester": "Semester 7",
    "subject": "Deep Learning",
    "subject_code": "CSC701",
}, timeout=60).json()
WS = created.get("id")
step("create_workspace", bool(WS), workspace_id=WS)
report["workspace_id"] = WS


def upload(path, doc_type):
    with open(path, "rb") as fh:
        res = requests.post(
            f"{BASE}/workspaces/{WS}/ingest",
            files={"file": (os.path.basename(path), fh, "application/pdf")},
            data={"doc_type": doc_type},
            timeout=900,
        )
    try:
        body = res.json()
    except Exception:
        body = {"raw": res.text[:400]}
    return res.status_code, body


syllabus = sorted(glob.glob(os.path.join(SYL_DIR, "syllabus", "**", "*.pdf"), recursive=True))
if syllabus:
    code, body = upload(syllabus[0], "syllabus")
    step("upload_syllabus", code == 200,
         file=os.path.basename(syllabus[0]), status=code,
         vectors=body.get("vectors_inserted"))
else:
    step("upload_syllabus", True, note="no syllabus PDF in repo; skipped")

pyqs = sorted(
    p for p in glob.glob(os.path.join(PYQ_DIR, "*.pdf"))
    if "aidsaiml" not in os.path.basename(p).lower()
    and "copy" not in os.path.basename(p).lower()
)
total_q = 0
per_file = []
for pdf in pyqs:
    code, body = upload(pdf, "pyq")
    ok = code == 200 and body.get("ingestion_status") == "ready"
    q = body.get("questions_extracted", 0)
    total_q += q if ok else 0
    per_file.append({
        "file": os.path.basename(pdf)[:60],
        "http": code,
        "questions": q,
        "vectors": body.get("vectors_inserted"),
        "quality": body.get("extraction_quality"),
        "status": body.get("ingestion_status"),
        "workspace_echo": body.get("workspace_id"),
    })
    step(f"upload_pyq:{os.path.basename(pdf)[:34]}", ok,
         http=code, questions=q, quality=body.get("extraction_quality"))
report["pyq_uploads"] = per_file

audit = requests.get(f"{BASE}/workspaces/{WS}/audit", timeout=120).json()
step("vector_audit", audit.get("pyq_vectors", 0) == total_q,
     pyq_vectors=audit.get("pyq_vectors"),
     syllabus_vectors=audit.get("syllabus_vectors"),
     expected_pyq=total_q)
report["vector_audit"] = {
    "pyq_vectors": audit.get("pyq_vectors"),
    "syllabus_vectors": audit.get("syllabus_vectors"),
    "documents": audit.get("documents"),
}

analysis = requests.post(f"{BASE}/workspaces/{WS}/analyze-pyq",
                         json={"workspace_id": WS}, timeout=300).json()
step("pyq_intelligence",
     (analysis.get("total_valid_questions") or analysis.get("total_questions_analyzed", 0)) > 0,
     questions=analysis.get("total_valid_questions") or analysis.get("total_questions_analyzed"),
     papers=analysis.get("total_papers"),
     years=analysis.get("years_covered"),
     exact_repeats=len(analysis.get("exact_repeats") or []),
     semantic_repeats=len(analysis.get("semantic_repeats") or []),
     related_topics=len(analysis.get("related_topics") or []),
     topics=len(analysis.get("topics") or []),
     raw_dump_in_dashboard=len(analysis.get("extracted_questions") or []))
report["intelligence"] = {
    "questions": analysis.get("total_valid_questions") or analysis.get("total_questions_analyzed"),
    "papers": analysis.get("total_papers"),
    "years_covered": analysis.get("years_covered"),
    "exact_repeats": len(analysis.get("exact_repeats") or []),
    "semantic_repeats": len(analysis.get("semantic_repeats") or []),
    "related_topics": len(analysis.get("related_topics") or []),
    "topic_count": len(analysis.get("topics") or []),
    "top_topics": [t.get("topic") or t.get("name") for t in (analysis.get("topics") or [])[:8]],
    "dashboard_raw_question_dump": len(analysis.get("extracted_questions") or []),
}

priority = requests.post(f"{BASE}/workspaces/{WS}/study-priority",
                         json={"workspace_id": WS, "top_n": 5}, timeout=300).json()
prio_list = priority.get("top_high_priority_topics") or []
step("study_priority", all(p.get("topic_name") for p in prio_list), count=len(prio_list),
     top=[(p.get("topic_name"), p.get("priority_score"), p.get("evidence_label")) for p in prio_list[:5]])
report["study_priority"] = prio_list[:5]

sources = requests.get(f"{BASE}/workspaces/{WS}/pyq-questions", timeout=120).json()
acc = sources.get("accepted_questions") or []
step("source_questions_view", len(acc) == total_q, count=len(acc), expected=total_q)
report["source_questions"] = {
    "count": len(acc),
    "sample": [
        {
            "question_id": q.get("question_id"),
            "marks": q.get("marks"),
            "year": q.get("year"),
            "source_file": (q.get("source_file") or "")[:44],
            "page": q.get("source_page"),
            "text": (q.get("exact_text") or "")[:90],
        }
        for q in acc[:5]
    ],
}

ask = requests.post(f"{BASE}/ask", json={
    "question": "Explain the dropout method and its advantages.",
    "workspace_id": WS,
    "mode": "10_marks",
}, timeout=300).json()
citations = ask.get("citations") or ask.get("sources") or []
step("ask_grounded", bool(ask.get("answer")),
     mode=ask.get("answer_mode"), citations=len(citations),
     answer_chars=len(ask.get("answer") or ""))
report["ask"] = {
    "answer_mode": ask.get("answer_mode"),
    "citations": len(citations),
    "answer_preview": (ask.get("answer") or "")[:220],
}

unknown = requests.get(f"{BASE}/workspaces/ws-not-real-e2e/pyq-questions", timeout=60)
step("ghost_workspace_404", unknown.status_code == 404, status=unknown.status_code)

# The E2E workspace is a real backend workspace, so it must be removed again or
# every run leaves 60 stale vectors behind in the live store. Set
# PYQRAG_E2E_KEEP=1 to keep it for manual UI inspection.
if os.environ.get("PYQRAG_E2E_KEEP") != "1":
    purged = requests.delete(f"{BASE}/workspaces/{WS}", timeout=120)
    after = requests.get(f"{BASE}/health", timeout=60).json()
    after_vectors = after.get("vector_store_stats", {}).get("total_vectors")
    step("cleanup_e2e_workspace",
         purged.status_code == 200 and after_vectors == baseline_vectors,
         status=purged.status_code,
         vectors_before=baseline_vectors,
         vectors_after=after_vectors)

report["all_ok"] = all(s["ok"] for s in report["steps"])
out = r"D:\pyqrag\scratch\extraction_audits\live_e2e_report.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, default=str)
print("\nALL_OK:", report["all_ok"])
print("workspace_id:", WS)
print("wrote", out)
sys.exit(0 if report["all_ok"] else 1)
