"""Exercise POST /workspaces/{id}/ingest via FastAPI TestClient (same route as the UI)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, r"D:\pyqrag")

from fastapi.testclient import TestClient

from rag.api import app

PDF = (
    r"D:\pyqrag\data\uploads\ws-nlp-31567ba5"
    r"\be_computer-engineering_semester-7_2022_december_"
    r"dloc-iii-natural-language-processing-rev-2019-c-scheme.pdf"
)
OUT = r"D:\pyqrag\scratch\extraction_audits\nlp_http_ingest.json"


def main():
    client = TestClient(app)
    health = client.get("/health")
    print("HEALTH", health.status_code, health.json().get("vector_store_stats"))

    created = client.post(
        "/workspaces",
        json={
            "university": "mumbai",
            "branch": "computer-engineering",
            "semester": "Semester 7",
            "subject": "natural language processing",
            "subject_code": "42175",
        },
    )
    print("CREATE", created.status_code, created.json())
    ws_id = created.json()["id"]

    with open(PDF, "rb") as fh:
        ingest = client.post(
            f"/workspaces/{ws_id}/ingest",
            files={"file": (os.path.basename(PDF), fh, "application/pdf")},
            data={"doc_type": "pyq"},
        )
    print("INGEST", ingest.status_code)
    payload = ingest.json()
    print(json.dumps({k: payload.get(k) for k in (
        "status", "extraction_quality", "questions_extracted", "vectors_inserted",
        "ingestion_status", "workspace_id",
    )}, indent=2))
    audit = payload.get("extraction_audit") or {}
    print("accepted", audit.get("accepted_question_ids"))
    print("selected", audit.get("selected_representation"))
    print("missing", audit.get("missing_questions") or payload.get("missing_questions"))

    analysis = client.get(f"/workspaces/{ws_id}/analyze-pyq")
    print("ANALYZE GET", analysis.status_code)
    ajson = analysis.json()
    print("questions analyzed", ajson.get("total_valid_questions") or ajson.get("total_questions_analyzed"))

    questions = client.get(f"/workspaces/{ws_id}/pyq-questions")
    qjson = questions.json()
    print("QUESTIONS", questions.status_code, "accepted", qjson.get("accepted_count"))

    report = {
        "http_status": ingest.status_code,
        "workspace_id": ws_id,
        "ingest": payload,
        "analyze_pyq": ajson,
        "pyq_questions": qjson,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("wrote", OUT)
    if ingest.status_code != 200:
        sys.exit(1)


if __name__ == "__main__":
    main()
