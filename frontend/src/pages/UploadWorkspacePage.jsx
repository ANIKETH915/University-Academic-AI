import React, { useState } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import PageHeader from '../components/PageHeader';
import UploadDropzone from '../components/UploadDropzone';
import FileCard from '../components/FileCard';
import { 
  Database, 
  BookOpen, 
  Layers, 
  Sparkles, 
  CheckCircle2, 
  Loader2,
  ArrowRight,
  FileText,
  AlertCircle
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/** Carries the backend's structured extraction diagnostics alongside the message. */
class IngestError extends Error {
  constructor(message, filename, payload) {
    super(message);
    this.name = 'IngestError';
    this.filename = filename;
    this.payload = payload || null;
  }
}

/**
 * Flatten a backend ingest/extraction payload into labelled rows so a rejected
 * upload explains itself instead of showing a bare sentence.
 */
function readDiagnostics(payload, filename) {
  if (!payload || typeof payload !== 'object') return null;
  const audit = payload.extraction_audit || {};
  const rows = [];
  const push = (label, value) => {
    if (value === undefined || value === null || value === '') return;
    rows.push({ label, value: String(value) });
  };

  push('File', filename);
  push('Extraction quality', payload.extraction_quality || payload.ingestion_status || payload.status);
  push('Pages', payload.pages ?? audit.pages);
  push('Questions extracted', payload.questions_extracted ?? audit.accepted_count);
  push('Candidates detected', payload.candidates ?? audit.candidate_count);
  push('Accepted', audit.accepted_count);
  push('Rejected', payload.rejected_questions ?? audit.rejected_count);
  push('Representation selected', audit.selected_representation);
  push('Text source used', payload.extraction_method || audit.selected_representation);
  push('OCR used', audit.ocr_used === undefined ? undefined : audit.ocr_used ? 'yes' : 'no');
  push('Vectors inserted', payload.vectors_inserted);

  const detected = audit.detected_markers || payload.detected_markers;
  if (Array.isArray(detected) && detected.length) {
    push('Detected markers', detected.join(', '));
  }
  const reconciled = audit.reconciled_markers || audit.accepted_question_ids;
  if (Array.isArray(reconciled) && reconciled.length) {
    push('Reconciled markers', reconciled.join(', '));
  }

  const missing = payload.missing_questions || audit.missing_questions;
  if (Array.isArray(missing) && missing.length) {
    push('Missing markers', missing.slice(0, 12).join(', '));
  }

  const reasons = payload.rejection_reasons || audit.rejection_reasons;
  if (Array.isArray(reasons) && reasons.length) {
    push('Rejection reasons', reasons.slice(0, 6).join('; '));
  } else if (reasons && typeof reasons === 'object') {
    const pairs = Object.entries(reasons).slice(0, 6).map(([k, v]) => `${k}: ${v}`);
    if (pairs.length) push('Rejection reasons', pairs.join('; '));
  }

  return rows.length ? rows : null;
}

export default function UploadWorkspacePage({ setActiveTab }) {
  const {
    activeWorkspace,
    activeWorkspaceId,
    removeFileFromWorkspace,
    refreshWorkspaces,
    setIsSelectorOpen
  } = useWorkspace();

  const [pendingSyllabusFiles, setPendingSyllabusFiles] = useState([]);
  const [pendingPyqFiles, setPendingPyqFiles] = useState([]);

  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [isReady, setIsReady] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [errorDiagnostics, setErrorDiagnostics] = useState(null);
  const [lastIngestSummary, setLastIngestSummary] = useState(null);

  const syllabusFiles = activeWorkspace.syllabusFiles || [];
  const pyqFiles = activeWorkspace.pyqFiles || [];
  const totalFiles = syllabusFiles.length + pyqFiles.length + pendingSyllabusFiles.length + pendingPyqFiles.length;
  const canIngest = Boolean(activeWorkspaceId) && (pendingSyllabusFiles.length + pendingPyqFiles.length) > 0;

  const handleSyllabusSelected = (files) => {
    setPendingSyllabusFiles(prev => [...prev, ...files]);
    setIsReady(false);
    setErrorMsg(null);
  };

  const handlePyqSelected = (files) => {
    setPendingPyqFiles(prev => [...prev, ...files]);
    setIsReady(false);
    setErrorMsg(null);
  };

  const ingestOne = async (file, docType) => {
    const wsId = activeWorkspaceId;
    if (!wsId) {
      throw new Error('No active workspace. Create or select a workspace first.');
    }

    console.log('[INGEST] workspace_id=', wsId, 'filename=', file.name, 'doc_type=', docType);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_type', docType);

    const res = await fetch(`${API_BASE_URL}/workspaces/${wsId}/ingest`, {
      method: 'POST',
      body: formData
    });

    let data = null;
    try {
      data = await res.json();
    } catch {
      data = null;
    }

    if (!res.ok) {
      const detail = data?.detail;
      const errMsg =
        (typeof detail === 'object' && detail?.error) ||
        (typeof detail === 'string' && detail) ||
        data?.error ||
        `Ingest failed (${res.status}) for ${file.name}`;
      throw new IngestError(errMsg, file.name, typeof detail === 'object' ? detail : data);
    }

    if (data?.status === 'ingestion_failed' || data?.ingestion_status === 'ingestion_failed_no_valid_questions' || data?.ingestion_status === 'INGESTION_FAILED') {
      throw new IngestError(data?.error || `No valid content extracted from ${file.name}`, file.name, data);
    }
    if (
      data?.status === 'extraction_incomplete'
      || data?.ingestion_status === 'extraction_incomplete'
      || data?.ingestion_status === 'INGESTION_PARTIAL'
      || data?.extraction_quality === 'PARTIAL'
      || data?.extraction_quality === 'FAILED'
    ) {
      const reason = data?.extraction_audit?.incomplete_reason
        || data?.incomplete_reason
        || data?.error
        || data?.detail?.error
        || `Question extraction is incomplete. Please review the extraction audit.`;
      throw new IngestError(
        typeof reason === 'string' ? reason : 'Question extraction is incomplete.',
        file.name,
        data
      );
    }

    console.log(
      '[INGEST COMPLETE] workspace_id=', data.workspace_id,
      'document_id=', data.document_id,
      'source_file=', data.source_file,
      'questions_extracted=', data.questions_extracted,
      'vectors_inserted=', data.vectors_inserted,
      'status=', data.ingestion_status
    );

    if (data.workspace_id && data.workspace_id !== wsId) {
      throw new Error(`Workspace ID mismatch: frontend=${wsId} backend=${data.workspace_id}`);
    }

    return data;
  };

  const handleBuildAssistant = async () => {
    if (!canIngest) {
      if (!activeWorkspaceId) {
        setErrorMsg('Create or select a workspace before uploading documents.');
      }
      return;
    }

    setIsProcessing(true);
    setErrorMsg(null);
    setErrorDiagnostics(null);
    setLastIngestSummary(null);
    setProcessingStep(1);

    const results = [];
    try {
      for (const file of pendingSyllabusFiles) {
        setProcessingStep(1);
        results.push(await ingestOne(file, 'syllabus'));
      }

      for (const file of pendingPyqFiles) {
        setProcessingStep(2);
        results.push(await ingestOne(file, 'pyq'));
      }

      setProcessingStep(3);
      // Backend is source of truth for file lists
      await refreshWorkspaces();
      setPendingSyllabusFiles([]);
      setPendingPyqFiles([]);
      setLastIngestSummary(results);
      setIsReady(true);
    } catch (err) {
      console.error('[INGEST ERROR]', err);
      setErrorMsg(err.message || 'Ingestion failed. No fake local registration was applied.');
      setErrorDiagnostics(readDiagnostics(err?.payload, err?.filename));
      setIsReady(false);
      try {
        await refreshWorkspaces();
      } catch (_) {}
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        breadcrumb="Knowledge Base"
        title="Build Your Academic Assistant"
        subtitle="Upload your syllabus and previous-year question papers to get started."
        icon={Database}
        badge="Document Ingestion"
      />

      <div className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] text-slate-300 text-xs flex items-center justify-between gap-3">
        <div className="flex items-center space-x-2">
          <BookOpen className="w-4 h-4 text-purple-400" />
          <span>
            Active Target:{' '}
            <strong className="text-white">
              {activeWorkspace.university || 'Workspace'} / {activeWorkspace.subject || 'Subject'} ({activeWorkspace.semester || 'Sem'})
            </strong>
            {activeWorkspaceId ? (
              <span className="ml-2 text-slate-500 font-mono">[{activeWorkspaceId}]</span>
            ) : (
              <span className="ml-2 text-rose-400">No workspace selected</span>
            )}
          </span>
        </div>

        <button
          onClick={() => setIsSelectorOpen(true)}
          className="px-3 py-1.5 rounded-lg bg-[#111827] hover:bg-[#1F2937] border border-[#1F2937] text-purple-300 text-xs font-semibold flex items-center space-x-1 transition-colors flex-shrink-0"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Switch Workspace</span>
        </button>
      </div>

      {errorMsg && (
        <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-200 text-xs space-y-2">
          <div className="flex items-center space-x-2 font-semibold text-rose-300">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Extraction Rejected — No Vectors Were Written</span>
          </div>
          <p className="pl-6 text-slate-300">{errorMsg}</p>

          {errorDiagnostics && (
            <div className="pl-6 pt-1 space-y-1">
              <div className="text-[11px] font-semibold text-rose-300/90 uppercase tracking-wide">
                Extraction diagnostics
              </div>
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1">
                {errorDiagnostics.map((row) => (
                  <div key={row.label} className="flex justify-between gap-3 border-b border-rose-500/10 py-0.5">
                    <dt className="text-slate-400 shrink-0">{row.label}</dt>
                    <dd className="text-slate-200 font-mono text-right break-all">{row.value}</dd>
                  </div>
                ))}
              </dl>
              <p className="text-[11px] text-slate-400 pt-1">
                Existing vectors for this workspace were left untouched.
              </p>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="saas-card p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 font-bold">
                  📚
                </div>
                <div>
                  <h3 className="font-heading font-bold text-sm text-white">Upload Syllabus</h3>
                  <p className="text-[11px] text-slate-400">Official course breakdown & topics (PDF format)</p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded-md bg-purple-500/20 text-purple-300 text-[10px] font-bold uppercase">Optional</span>
            </div>

            <UploadDropzone
              onFilesSelected={handleSyllabusSelected}
              accept=".pdf,application/pdf"
              multiple={false}
              title="Upload Official Syllabus PDF"
              subtitle="or drag & drop your PDF file here"
            />
          </div>

          {(syllabusFiles.length > 0 || pendingSyllabusFiles.length > 0) && (
            <div className="space-y-2 pt-2 border-t border-[#1F2937]">
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Indexed Syllabus ({syllabusFiles.length + pendingSyllabusFiles.length})
              </div>
              {syllabusFiles.map((f, idx) => (
                <FileCard
                  key={`${f.id || f.name}-${idx}`}
                  file={f}
                  onDelete={() => removeFileFromWorkspace(activeWorkspaceId, f.id, 'syllabus')}
                />
              ))}
              {pendingSyllabusFiles.map((f, i) => (
                <div key={`pending-syl-${f.name}-${i}`} className="p-2.5 rounded-xl bg-[#0B1020] border border-amber-500/30 text-amber-300 text-xs flex items-center justify-between">
                  <div className="flex items-center space-x-2 truncate">
                    <FileText className="w-4 h-4 text-amber-400" />
                    <span className="truncate">{f.name}</span>
                  </div>
                  <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-amber-500/20">Pending Ingestion</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="saas-card p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 font-bold">
                  📝
                </div>
                <div>
                  <h3 className="font-heading font-bold text-sm text-white">Upload Previous-Year Papers</h3>
                  <p className="text-[11px] text-slate-400">Past exam question papers for topic frequency analysis</p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded-md bg-indigo-500/20 text-indigo-300 text-[10px] font-bold uppercase">Optional Multi-PDF</span>
            </div>

            <UploadDropzone
              onFilesSelected={handlePyqSelected}
              accept=".pdf,application/pdf"
              multiple={true}
              title="Upload PYQ Question PDFs"
              subtitle="or drag & drop your PDF files here"
            />
          </div>

          {(pyqFiles.length > 0 || pendingPyqFiles.length > 0) && (
            <div className="space-y-2 pt-2 border-t border-[#1F2937]">
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Indexed PYQ Papers ({pyqFiles.length + pendingPyqFiles.length})
              </div>
              {pyqFiles.map((f, idx) => (
                <FileCard
                  key={`${f.id || f.name}-${idx}`}
                  file={f}
                  onDelete={() => removeFileFromWorkspace(activeWorkspaceId, f.id, 'pyq')}
                />
              ))}
              {pendingPyqFiles.map((f, i) => (
                <div key={`pending-pyq-${f.name}-${i}`} className="p-2.5 rounded-xl bg-[#0B1020] border border-amber-500/30 text-amber-300 text-xs flex items-center justify-between">
                  <div className="flex items-center space-x-2 truncate">
                    <FileText className="w-4 h-4 text-amber-400" />
                    <span className="truncate">{f.name}</span>
                  </div>
                  <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-amber-500/20">Pending Ingestion</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="saas-card p-6 text-center space-y-4">
        {!isProcessing && !isReady && (
          <div className="space-y-3">
            <div className="space-y-1">
              <h3 className="font-heading font-bold text-base text-white">Ready to Build Your Assistant?</h3>
              <p className="text-xs text-slate-400">
                {!activeWorkspaceId
                  ? 'Create a workspace first, then select PDFs to ingest.'
                  : canIngest
                    ? `${pendingSyllabusFiles.length + pendingPyqFiles.length} pending document(s) for ${activeWorkspace.subject || 'your workspace'}.`
                    : 'Select at least one syllabus or PYQ PDF above to start.'}
              </p>
            </div>

            <button
              onClick={handleBuildAssistant}
              disabled={!canIngest}
              className={`px-8 py-3.5 rounded-xl font-bold text-xs shadow-xl transition-all inline-flex items-center space-x-2 ${
                canIngest
                  ? 'bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-purple-500/25 cursor-pointer'
                  : 'bg-[#111827] text-slate-600 border border-[#1F2937] cursor-not-allowed'
              }`}
            >
              <Sparkles className="w-4 h-4" />
              <span>Build My Academic Assistant</span>
            </button>
          </div>
        )}

        {isProcessing && (
          <div className="py-6 max-w-md mx-auto space-y-4">
            <Loader2 className="w-8 h-8 animate-spin mx-auto text-purple-400" />
            <h3 className="font-heading font-bold text-sm text-white">Preparing your academic assistant...</h3>
            <div className="space-y-2 text-left text-xs bg-[#0B1020] p-4 rounded-xl border border-[#1F2937]">
              <div className={`flex items-center space-x-2 ${processingStep >= 1 ? 'text-emerald-400' : 'text-slate-500'}`}>
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                <span>Reading syllabus structure and course units</span>
              </div>
              <div className={`flex items-center space-x-2 ${processingStep >= 2 ? 'text-emerald-400' : 'text-slate-500'}`}>
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                <span>Processing previous-year question papers</span>
              </div>
              <div className={`flex items-center space-x-2 ${processingStep >= 3 ? 'text-emerald-400' : 'text-slate-500'}`}>
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                <span>Indexing vectors for workspace {activeWorkspaceId}</span>
              </div>
            </div>
          </div>
        )}

        {isReady && (
          <div className="py-4 space-y-4">
            <div className="w-12 h-12 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <h3 className="font-heading font-extrabold text-lg text-white">Your Academic Assistant is Ready</h3>
              <p className="text-xs text-slate-400">
                Knowledge base built for {activeWorkspace.subject || 'your workspace'} [{activeWorkspaceId}].
              </p>
              {lastIngestSummary?.length > 0 && (
                <p className="text-[11px] text-slate-500">
                  Inserted {lastIngestSummary.reduce((s, r) => s + (r.vectors_inserted || 0), 0)} vectors
                  across {lastIngestSummary.length} file(s).
                </p>
              )}
            </div>
            <button
              onClick={() => setActiveTab('ask')}
              className="px-8 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs shadow-lg shadow-purple-500/20 inline-flex items-center space-x-2 transition-all"
            >
              <span>Ask Your First Question</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
