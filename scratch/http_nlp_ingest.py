"""HTTP ingest of the real NLP PDF through the frontend endpoint."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

import requests

BASE = os.environ.get("PYQRAG_HTTP", "http://127.0.0.1:8000")
WS = "ws-nlp-31567ba5"
PDF = (
    r"D:\pyqrag\data\uploads\ws-nlp-31567ba5"
    r"\be_computer-engineering_semester-7_2022_december_"
    r"dloc-iii-natural-language-processing-rev-2019-c-scheme.pdf"
)
OUT = r"D:\pyqrag\scratch\extraction_audits\nlp_http_ingest.json"


def ensure_workspace():
    path = r"D:\pyqrag\workspaces.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if any(w.get("id") == WS for w in data):
        print("workspace already registered")
        return
    data.insert(
        0,
        {
            "id": WS,
            "university": "mumbai",
            "branch": "computer-engineering",
            "program": "computer-engineering",
            "semester": "Semester 7",
            "subject": "natural language processing",
            "subject_code": "42175",
            "is_demo": False,
            "created_at": "2026-08-22T00:00:00+00:00",
            "syllabus_files": [],
            "pyq_files": [],
        },
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print("registered workspace", WS)


def wait_health(timeout=60):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                print("health", r.json())
                return
            last = r.status_code
        except Exception as e:
            last = str(e)
        time.sleep(1)
    raise SystemExit(f"server not healthy: {last}")


def main():
    ensure_workspace()
    wait_health()
    with open(PDF, "rb") as fh:
        r = requests.post(
            f"{BASE}/workspaces/{WS}/ingest",
            files={"file": (os.path.basename(PDF), fh, "application/pdf")},
            data={"doc_type": "pyq"},
            timeout=180,
        )
    print("INGEST", r.status_code)
    try:
        payload = r.json()
    except Exception:
        payload = {"text": r.text[:2000]}
    print(json.dumps(payload, indent=2, default=str)[:4000])

    analysis = None
    if r.status_code == 200:
        g = requests.get(f"{BASE}/workspaces/{WS}/analyze-pyq", timeout=60)
        print("ANALYZE GET", g.status_code)
        analysis = g.json() if g.ok else {"error": g.text[:1000]}
        q = requests.get(f"{BASE}/workspaces/{WS}/pyq-questions", timeout=60)
        print("QUESTIONS GET", q.status_code)
        questions = q.json() if q.ok else {"error": q.text[:1000]}
    else:
        questions = None

    report = {
        "http_status": r.status_code,
        "ingest": payload,
        "analyze_pyq": analysis,
        "pyq_questions": questions,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("wrote", OUT)
    if r.status_code != 200:
        sys.exit(1)


if __name__ == "__main__":
    main()
