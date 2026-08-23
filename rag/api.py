import os
import shutil
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag.vector_store import VectorStore
from rag.answer_engine import GroundedAnswerEngine
from rag.pyq_intelligence import PYQIntelligenceEngine
from rag.workspace_db import WorkspaceDB
from rag.dynamic_ingest import DynamicIngestPipeline

app = FastAPI(
    title="University Academic AI RAG API",
    description="Dynamic Multi-Workspace RAG System & Universal Intelligence Engine",
    version="3.5.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
store = VectorStore()
answer_engine = GroundedAnswerEngine(vector_store=store)
pyq_intel = PYQIntelligenceEngine(vector_store=store)
workspace_db = WorkspaceDB()
dynamic_ingest = DynamicIngestPipeline(vector_store=store)

class CreateWorkspaceRequest(BaseModel):
    university: str = Field(..., example="Anna University")
    branch: str = Field(..., example="Computer Science and Engineering")
    semester: str = Field(..., example="Semester 5")
    subject: str = Field(..., example="Computer Networks")
    subject_code: Optional[str] = Field("CS3591", example="CS3591")

class SearchRequest(BaseModel):
    query: str = Field(..., example="Explain count to infinity problem in detail")
    workspace_id: str = Field(..., example="mu-cmpn-sem5-cn", description="Required active workspace ID")
    doc_type: str = Field("both", example="both")
    top_k: int = Field(5, ge=1, le=50, example=5)
    semester: Optional[str] = Field(None)
    subject: Optional[str] = Field(None)
    university: Optional[str] = Field(None)

class AskRequest(BaseModel):
    question: str = Field(..., example="Explain count to infinity problem in detail")
    workspace_id: str = Field(..., example="mu-cmpn-sem5-cn", description="Required active workspace ID")
    mode: str = Field("general", example="5_marks")
    doc_type: str = Field("both", example="both")
    semester: Optional[str] = Field(None)
    subject: Optional[str] = Field(None)
    university: Optional[str] = Field(None)

class PYQAnalysisRequest(BaseModel):
    workspace_id: str = Field(..., description="Required active workspace ID")
    # Left unset so the workspace's own subject/semester are used. A concrete
    # default here would silently relabel every workspace.
    subject: Optional[str] = Field(None)
    semester: Optional[str] = Field(None)

class StudyPriorityRequest(BaseModel):
    workspace_id: str = Field(..., description="Required active workspace ID")
    subject: Optional[str] = Field(None)
    semester: Optional[str] = Field(None)
    top_n: int = Field(5, ge=1, le=20)

def require_workspace(workspace_id: str) -> Dict[str, Any]:
    """
    Resolve a workspace or fail. Never creates one implicitly: a request for an
    unknown id is a bug in the caller, not a reason to invent a workspace.
    """
    if not workspace_id or workspace_id.strip() in {"", "/", "undefined", "null", "ws-default-workspace"}:
        raise HTTPException(status_code=400, detail="A valid workspace_id is required.")
    ws = workspace_db.get_by_id(workspace_id)
    if not ws:
        raise HTTPException(
            status_code=404,
            detail=f"Workspace '{workspace_id}' not found. Create it via POST /workspaces.",
        )
    return ws


import time
START_TIME = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

@app.get("/health", summary="Check system & vector store status")
def health_check():
    try:
        stats = store.get_stats()
    except Exception as e:
        stats = {"total_vectors": 0, "status": "active", "warning": str(e)}
    from rag.llm_client import llm_status

    return {
        "status": "ok",
        "service": "University Academic AI RAG Backend",
        "version": "3.6.0-universal-reconciliation",
        "extraction_pipeline": "universal_9_stage_reconciliation_v2",
        "git_commit": "914f07d3e54714e78d0dbb93a190e2a6ae8baeca",
        "started_at": START_TIME,
        "vector_store_stats": stats,
        # Provider names and models only — credentials never leave the server.
        "llm": llm_status(),
    }

@app.get("/workspaces", summary="List all academic workspaces")
def list_workspaces():
    return workspace_db.get_all()

@app.post("/workspaces", summary="Create a new persistent academic workspace")
def create_workspace(req: CreateWorkspaceRequest):
    return workspace_db.create(
        university=req.university,
        branch=req.branch,
        semester=req.semester,
        subject=req.subject,
        subject_code=req.subject_code or ""
    )

@app.delete("/workspaces/{workspace_id}", summary="Delete workspace and purge all its vectors")
def delete_workspace(workspace_id: str):
    store.delete_by_workspace(workspace_id)
    success = workspace_db.delete_workspace(workspace_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"status": "success", "message": f"Workspace {workspace_id} and all its vectors deleted."}

@app.get("/workspace/{workspace_id}/audit", summary="Development diagnostic audit endpoint")
@app.get("/workspaces/{workspace_id}/audit", summary="Development diagnostic audit endpoint")
@app.get("/debug/pyq-extraction/{workspace_id}", summary="Development diagnostic extraction audit endpoint")
def workspace_audit(workspace_id: str):
    ws = require_workspace(workspace_id)
    try:
        c_res = store.collection.get(where={"workspace_id": {"$eq": workspace_id}})
        all_metas = c_res.get("metadatas", []) if c_res else []
    except Exception:
        all_metas = []

    syl_vectors = sum(1 for m in all_metas if m.get("doc_type") == "syllabus")
    pyq_vectors = sum(1 for m in all_metas if m.get("doc_type") == "pyq")

    doc_summary = {}
    for m in all_metas:
        fname = m.get("source_file", "unknown.pdf")
        dtype = m.get("doc_type", "syllabus")
        if fname not in doc_summary:
            doc_summary[fname] = {"filename": fname, "doc_type": dtype, "vector_count": 0}
        doc_summary[fname]["vector_count"] += 1

    return {
        "workspace_id": workspace_id,
        "workspace_exists": ws is not None,
        "workspace_metadata": ws or {},
        "documents": list(doc_summary.values()),
        "syllabus_vectors": syl_vectors,
        "pyq_vectors": pyq_vectors,
        "total_vectors": len(all_metas),
        "last_ingest_audit_log": dynamic_ingest.last_audit_log
    }

@app.post("/workspaces/{workspace_id}/ingest", summary="Dynamically ingest PDF document into workspace")
async def ingest_document(workspace_id: str, file: UploadFile = File(...), doc_type: str = Form("syllabus")):
    if not workspace_id or workspace_id in {"/", "undefined", "null", "ws-default-workspace"}:
        raise HTTPException(
            status_code=400,
            detail="Valid workspace_id is required. Create a workspace from the UI first.",
        )
    ws = workspace_db.get_by_id(workspace_id)
    if not ws:
        raise HTTPException(
            status_code=404,
            detail=f"Workspace '{workspace_id}' not found. Create it via POST /workspaces before ingesting.",
        )

    import tempfile
    from rag.config import BASE_DIR

    # Persist original upload for re-audit (real frontend flow debugging)
    persist_dir = os.path.join(BASE_DIR, "data", "uploads", workspace_id)
    os.makedirs(persist_dir, exist_ok=True)
    safe_name = os.path.basename(file.filename or "upload.pdf")
    persist_path = os.path.join(persist_dir, safe_name)

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{safe_name}")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        shutil.copy2(temp_path, persist_path)
    except Exception as e:
        print(f"[INGEST] persist copy warning: {e}")

    pages_count = 1
    try:
        import fitz
        doc_tmp = fitz.open(temp_path)
        pages_count = len(doc_tmp)
        doc_tmp.close()
    except Exception:
        pass

    rejected_count = 0
    ingestion_status = "ready"
    extraction_audit = {}
    if doc_type == "syllabus":
        metas = dynamic_ingest.parse_syllabus_pdf(temp_path, ws)
    elif doc_type == "textbook":
        metas = dynamic_ingest.parse_textbook_pdf(temp_path, ws)
    else:
        metas = dynamic_ingest.parse_pyq_pdf(temp_path, ws)
        audit = dynamic_ingest.last_pyq_questions_audit or {}
        if metas:
            rejected_count = metas[0].get("rejected_count", 0)
        ingestion_status = audit.get("ingestion_status", "ready")
        raw_audit = audit.get("extraction_audit") or {}
        extraction_audit = {
            "representations": raw_audit.get("representations", {}),
            "marker_candidates": raw_audit.get("marker_candidates", []),
            "reconciled_questions": raw_audit.get("reconciled_questions", [q.get("question_id") for q in (audit.get("accepted_questions") or [])]),
            "ambiguous_markers": raw_audit.get("ambiguous_markers", []),
            "rejected_markers": raw_audit.get("rejected_markers", [
                {
                    "question_id": r.get("question_id"),
                    "reason": r.get("reason") or r.get("rejection_reason"),
                    "page": r.get("page"),
                    "candidate_text": (r.get("raw_text") or r.get("exact_text") or "")[:240],
                }
                for r in (audit.get("rejected_candidates") or [])[:80]
            ]),
            "missing_genuine_questions": raw_audit.get("missing_genuine_questions", (audit.get("quality_summary") or {}).get("missing_questions") or []),
            "cross_page_merges": raw_audit.get("cross_page_merges", []),
            "representation_sources": raw_audit.get("representation_sources", {}),
            "page_extraction_audit": audit.get("page_extraction_audit", []),
            "accepted_question_ids": [q.get("question_id") for q in (audit.get("accepted_questions") or [])],
            "detected_markers": audit.get("source_markers") or [],
            "reconciled_markers": [q.get("question_id") for q in (audit.get("accepted_questions") or [])],
            "missing_questions": (audit.get("quality_summary") or {}).get("missing_questions") or [],
            "accepted_count": len(audit.get("accepted_questions") or []),
            "rejected_count": len(audit.get("rejected_candidates") or []),
            "candidate_count": len(audit.get("accepted_questions") or [])
            + len(audit.get("rejected_candidates") or []),
            "selected_representation": (
                ((audit.get("quality_summary") or {}).get("selected_representations") or [{}])[0].get("selected")
            ),
            "extraction_incomplete": audit.get("extraction_incomplete", False),
            "incomplete_reason": audit.get("incomplete_reason"),
            "persisted_pdf": persist_path,
        }
        if not metas:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            qsum = audit.get("quality_summary") or {}
            raise HTTPException(
                status_code=422,
                detail={
                    "status": "ingestion_failed",
                    "document_id": f"doc-{safe_name}",
                    "source_file": safe_name,
                    "workspace_id": workspace_id,
                    "pages": pages_count,
                    "questions_extracted": qsum.get("questions_extracted", len(audit.get("accepted_questions") or [])),
                    "source_markers_detected": qsum.get("source_markers_detected", 0),
                    "missing_questions": qsum.get("missing_questions", []),
                    "extraction_quality": audit.get("extraction_quality") or qsum.get("extraction_quality"),
                    "confidence": audit.get("question_extraction_confidence"),
                    "chunks_created": 0,
                    "embeddings_created": 0,
                    "vectors_inserted": 0,
                    "rejected_questions": len(audit.get("rejected_candidates", [])),
                    "ingestion_status": ingestion_status,
                    "extraction_audit": extraction_audit,
                    "error": audit.get("incomplete_reason")
                    or "Question extraction failed. Please review the extraction audit.",
                },
            )

    size_mb = f"{round(os.path.getsize(temp_path) / (1024*1024), 1)} MB"
    file_info = {
        "id": f"doc-{safe_name}",
        "name": safe_name,
        "size": size_mb,
        "metadata": ws.get("semester", "Semester 1"),
        "status": "VERIFIED" if metas else "FAILED",
        "persisted_path": persist_path,
    }
    workspace_db.add_file(workspace_id, file_info, doc_type)

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    status_out = "success"
    if doc_type == "pyq" and ingestion_status == "INGESTION_FAILED":
        status_out = "ingestion_failed"

    # Surface quality block at top level for frontend
    quality_block = {}
    if doc_type == "pyq":
        audit = dynamic_ingest.last_pyq_questions_audit or {}
        quality_block = audit.get("quality_summary") or {}
        if not metas and ingestion_status == "INGESTION_FAILED":
            # Return structured failure instead of empty success
            raise HTTPException(
                status_code=422,
                detail={
                    "status": status_out,
                    "document_id": f"doc-{safe_name}",
                    "source_file": safe_name,
                    "workspace_id": workspace_id,
                    "pages": pages_count,
                    "questions_extracted": quality_block.get("questions_extracted", 0),
                    "source_markers_detected": quality_block.get("source_markers_detected", 0),
                    "missing_questions": quality_block.get("missing_questions", []),
                    "extraction_quality": audit.get("extraction_quality") or quality_block.get("extraction_quality"),
                    "confidence": audit.get("question_extraction_confidence"),
                    "chunks_created": 0,
                    "embeddings_created": 0,
                    "vectors_inserted": 0,
                    "rejected_questions": len(extraction_audit.get("rejected_candidates") or []),
                    "ingestion_status": ingestion_status,
                    "extraction_audit": extraction_audit,
                    "error": audit.get("incomplete_reason")
                    or "Question extraction failed. Please review the extraction audit.",
                },
            )

    return {
        "status": status_out,
        "document_id": f"doc-{safe_name}",
        "source_file": safe_name,
        "workspace_id": workspace_id,
        "pages": pages_count,
        "questions_extracted": len(metas),
        "chunks_created": len(metas),
        "embeddings_created": len(metas),
        "vectors_inserted": len(metas),
        "rejected_questions": rejected_count if doc_type != "pyq" else len(extraction_audit.get("rejected_candidates") or []),
        "ingestion_status": ingestion_status,
        "extraction_quality": (dynamic_ingest.last_pyq_questions_audit or {}).get("extraction_quality") if doc_type == "pyq" else None,
        "extraction_audit": extraction_audit,
        "quality": quality_block if doc_type == "pyq" else None,
        "persisted_pdf": persist_path if os.path.exists(persist_path) else None,
    }

@app.delete("/workspaces/{workspace_id}/documents/{file_id}", summary="Delete document from workspace and purge vectors")
def delete_document(workspace_id: str, file_id: str, doc_type: str = "syllabus", filename: Optional[str] = None):
    if filename:
        store.delete_by_source_file(filename, workspace_id=workspace_id)

    updated_ws = workspace_db.remove_file(workspace_id, file_id, doc_type)
    return {"status": "success", "workspace": updated_ws}

@app.post("/search", summary="Hybrid vector search strictly scoped by workspace_id")
def search_endpoint(req: SearchRequest):
    if not req.workspace_id or not req.workspace_id.strip():
        raise HTTPException(status_code=400, detail="Academic workspace_id is required.")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    filters = {"workspace_id": req.workspace_id}

    results = store.search(
        query=req.query,
        doc_type=req.doc_type,
        top_k=req.top_k,
        filters=filters
    )

    return {
        "query": req.query,
        "workspace_id": req.workspace_id,
        "doc_type": req.doc_type,
        "total_retrieved": len(results),
        "results": results
    }

@app.post("/ask", summary="Generate grounded answer strictly for workspace_id")
def ask_endpoint(req: AskRequest):
    if not req.workspace_id or not req.workspace_id.strip():
        raise HTTPException(status_code=400, detail="Academic workspace_id is required.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    filters = {"workspace_id": req.workspace_id}

    result = answer_engine.generate_grounded_answer(
        question=req.question,
        mode=req.mode,
        doc_type=req.doc_type,
        filters=filters,
        debug=False,
    )

    return result

@app.post("/ask/debug", summary="Generate grounded answer with full retrieval debug trace")
def ask_debug_endpoint(req: AskRequest):
    """
    Same as /ask but returns the complete retrieval debug payload:
    retrieved chunks, scores, rerank scores, context sent to synthesis,
    answer_mode (rag_llm / retrieval_only / insufficient_evidence), filters.
    """
    if not req.workspace_id or not req.workspace_id.strip():
        raise HTTPException(status_code=400, detail="Academic workspace_id is required.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    filters = {"workspace_id": req.workspace_id}
    return answer_engine.generate_grounded_answer(
        question=req.question,
        mode=req.mode,
        doc_type=req.doc_type,
        filters=filters,
        debug=True,
    )

@app.post("/pyq-analysis", summary="Analyze PYQ topic frequency dynamically for workspace_id")
@app.post("/workspaces/{workspace_id}/analyze-pyq", summary="Analyze PYQ topic frequency dynamically for workspace_id")
def pyq_analysis_endpoint(req: PYQAnalysisRequest, workspace_id: Optional[str] = None):
    ws_id = workspace_id or req.workspace_id
    ws = require_workspace(ws_id)
    subject = req.subject or (ws.get("subject") if ws else "Subject")
    semester = req.semester or (ws.get("semester") if ws else "Semester")
    return _pyq_analysis_payload(ws_id, subject, semester)


@app.get("/workspaces/{workspace_id}/analyze-pyq", summary="Analyze PYQ topic frequency for a workspace")
def pyq_analysis_get(workspace_id: str, subject: Optional[str] = None, semester: Optional[str] = None):
    ws = require_workspace(workspace_id)
    return _pyq_analysis_payload(
        workspace_id,
        subject or ws.get("subject") or "Subject",
        semester or ws.get("semester") or "Semester",
    )


def _pyq_analysis_payload(ws_id: str, subject: str, semester: str):
    analysis = pyq_intel.get_pyq_analysis(workspace_id=ws_id, subject=subject, semester=semester)
    qcount = analysis.get("total_valid_questions") or analysis.get("total_questions_analyzed") or 0
    if analysis.get("available") and qcount > 0 and qcount < 3 and (analysis.get("total_papers") or 0) <= 1:
        return {
            **analysis,
            "extraction_incomplete": True,
            "prediction_notice": (
                "Priority analysis limited because question extraction is incomplete. "
                + str(analysis.get("prediction_notice") or "")
            ).strip(),
            "message": (
                f"PYQ extraction incomplete: {qcount} questions extracted. "
                "Valid grounded questions remain visible with reduced confidence."
            ),
        }
    return analysis

@app.get("/workspaces/{workspace_id}/pyq-questions", summary="Get diagnostic accepted questions and rejected candidates for workspace_id")
def get_pyq_questions(workspace_id: str):
    require_workspace(workspace_id)

    accepted = pyq_intel.get_source_questions(workspace_id)
    rejected = dynamic_ingest.last_pyq_questions_audit.get("rejected_candidates", [])
    summary = {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected)
    }

    return {
        "workspace_id": workspace_id,
        "accepted_questions": accepted,
        "rejected_candidates": rejected,
        "quality_summary": summary
    }

@app.get("/workspaces/{workspace_id}/pyq-patterns", summary="Get within-paper or multi-year pattern analysis for workspace_id")
def get_pyq_patterns(workspace_id: str):
    ws = require_workspace(workspace_id)
    analysis = pyq_intel.get_pyq_analysis(workspace_id=workspace_id, subject=ws.get("subject"), semester=ws.get("semester"))
    return {
        "workspace_id": workspace_id,
        "single_paper_mode": analysis.get("single_paper_mode", False),
        "within_paper_patterns": analysis.get("within_paper_patterns", []),
        "exact_repeats": analysis.get("exact_repeats", []),
        "semantic_repeats": analysis.get("semantic_repeats", []),
        "related_topics": analysis.get("related_topics", []),
        "topics": analysis.get("topics", [])
    }

@app.get("/workspaces/{workspace_id}/pyq-audit", summary="Get full workspace PYQ audit including accepted and rejected candidates")
def get_pyq_audit(workspace_id: str):
    ws = require_workspace(workspace_id)
    analysis = pyq_intel.get_pyq_analysis(
        workspace_id=workspace_id,
        subject=ws.get("subject"),
        semester=ws.get("semester"),
        include_source_questions=True,
    )
    rejected = dynamic_ingest.last_pyq_questions_audit.get("rejected_candidates", [])
    return {
        "workspace_id": workspace_id,
        "accepted_questions": analysis.get("extracted_questions", []),
        "rejected_candidates": rejected,
        "quality_summary": {
            "accepted_count": len(analysis.get("extracted_questions", [])),
            "rejected_count": len(rejected),
            "total_pages": dynamic_ingest.last_pyq_questions_audit.get("quality_summary", {}).get("total_pages", 0),
            "single_paper_mode": analysis.get("single_paper_mode", False),
            "prediction_notice": analysis.get("prediction_notice", "Analysis based on recurring patterns across papers.")
        }
    }

@app.get("/workspaces/{workspace_id}/study-priority", summary="Get the workspace study priority ranking")
def get_study_priority_get(workspace_id: str, top_n: int = 5):
    ws = require_workspace(workspace_id)
    return pyq_intel.get_study_priority(workspace_id=workspace_id, subject=ws.get("subject"), semester=ws.get("semester"), top_n=top_n)

@app.post("/study-priority", summary="Rank high-priority exam topics dynamically for workspace_id")
@app.post("/workspaces/{workspace_id}/study-priority", summary="Rank high-priority exam topics dynamically for workspace_id")
def study_priority_endpoint(req: StudyPriorityRequest, workspace_id: Optional[str] = None):
    ws_id = workspace_id or req.workspace_id
    ws = require_workspace(ws_id)
    subject = req.subject or (ws.get("subject") if ws else "Subject")
    semester = req.semester or (ws.get("semester") if ws else "Semester")
    return pyq_intel.get_study_priority(workspace_id=ws_id, subject=subject, semester=semester, top_n=req.top_n)
