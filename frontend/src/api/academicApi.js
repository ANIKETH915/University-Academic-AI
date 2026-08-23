const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Read a structured backend error. FastAPI puts diagnostics in `detail`,
 * which may itself be an object (extraction audits) or a plain string.
 */
async function describeFailure(res, fallback) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  const detail = body?.detail;
  if (detail && typeof detail === 'object') {
    return { message: detail.error || fallback, detail, status: res.status };
  }
  if (typeof detail === 'string' && detail) {
    return { message: detail, detail: null, status: res.status };
  }
  return { message: `${fallback} (HTTP ${res.status})`, detail: null, status: res.status };
}

const EMPTY_ANALYSIS = {
  total_questions_analyzed: 0,
  total_valid_questions: 0,
  unique_topic_clusters: 0,
  topics: [],
  most_repeated_questions: [],
  exact_repeats: [],
  semantic_repeats: [],
  related_topics: [],
  topic_recurrence: [],
  study_priorities: [],
  extracted_questions: [],
  within_paper_patterns: []
};

export async function fetchHealthStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    console.warn('API offline or unreachable', err);
    return {
      status: 'offline',
      service: 'University Academic AI Backend',
      version: 'unknown',
      offline: true
    };
  }
}

export async function askQuestion(payload) {
  try {
    const res = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: payload.question,
        workspace_id: payload.workspace_id || payload.workspaceId,
        mode: payload.mode || 'general',
        doc_type: payload.doc_type || 'both',
        subject: payload.subject,
        semester: payload.semester
      })
    });
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Ask API error', err);
    throw err;
  }
}

export async function fetchPYQAnalysis(subject = 'Subject', semester = 'Semester', hasPyqs = true, workspace_id = '') {
  if (!hasPyqs || !workspace_id) {
    return { workspace_id, subject, semester, ...EMPTY_ANALYSIS };
  }

  try {
    const res = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/analyze-pyq`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id, subject, semester })
    });
    if (!res.ok) {
      const failure = await describeFailure(res, 'PYQ analysis failed');
      console.warn('PYQ Analysis API error', failure);
      // An empty result and a failed request are different things; say which.
      return {
        workspace_id,
        subject,
        semester,
        ...EMPTY_ANALYSIS,
        unavailable: true,
        error: failure.message,
        error_status: failure.status
      };
    }
    return await res.json();
  } catch (err) {
    console.warn('PYQ Analysis API error', err);
    return {
      workspace_id,
      subject,
      semester,
      ...EMPTY_ANALYSIS,
      unavailable: true,
      error: err?.message || 'Backend unreachable'
    };
  }
}

export async function fetchPYQSourceQuestions(workspace_id = '') {
  if (!workspace_id) {
    return { workspace_id: '', accepted_questions: [], rejected_candidates: [] };
  }
  try {
    const res = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/pyq-questions`);
    if (!res.ok) {
      const failure = await describeFailure(res, 'Source questions unavailable');
      return {
        workspace_id,
        accepted_questions: [],
        rejected_candidates: [],
        unavailable: true,
        error: failure.message
      };
    }
    return await res.json();
  } catch (err) {
    console.warn('PYQ source questions API error', err);
    return {
      workspace_id,
      accepted_questions: [],
      rejected_candidates: [],
      unavailable: true,
      error: err?.message || 'Backend unreachable'
    };
  }
}

export async function fetchStudyPriority(subject = 'Subject', semester = 'Semester', top_n = 5, hasPyqs = true, workspace_id = '') {
  if (!hasPyqs || !workspace_id) {
    return {
      workspace_id: workspace_id,
      subject: subject,
      semester: semester,
      top_high_priority_topics: []
    };
  }

  try {
    const res = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/study-priority`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id, subject, semester, top_n })
    });
    if (!res.ok) {
      const failure = await describeFailure(res, 'Study priority unavailable');
      return {
        workspace_id,
        subject,
        semester,
        top_high_priority_topics: [],
        unavailable: true,
        error: failure.message
      };
    }
    return await res.json();
  } catch (err) {
    console.warn('Study Priority API error', err);
    return {
      workspace_id,
      subject,
      semester,
      top_high_priority_topics: [],
      unavailable: true,
      error: err?.message || 'Backend unreachable'
    };
  }
}

export async function fetchPYQQuestions(workspace_id) {
  if (!workspace_id) return { extracted_questions: [] };
  try {
    const res = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/pyq-questions`);
    if (!res.ok) {
      const failure = await describeFailure(res, 'Extracted questions unavailable');
      return { extracted_questions: [], unavailable: true, error: failure.message };
    }
    return await res.json();
  } catch (err) {
    console.warn('Fetch PYQ questions error', err);
    return { extracted_questions: [], unavailable: true, error: err?.message || 'Backend unreachable' };
  }
}

export async function fetchPYQPatterns(workspace_id) {
  if (!workspace_id) return { within_paper_patterns: [], topics: [] };
  try {
    const res = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/pyq-patterns`);
    if (!res.ok) {
      const failure = await describeFailure(res, 'PYQ patterns unavailable');
      return { within_paper_patterns: [], topics: [], unavailable: true, error: failure.message };
    }
    return await res.json();
  } catch (err) {
    console.warn('Fetch PYQ patterns error', err);
    return { within_paper_patterns: [], topics: [], unavailable: true, error: err?.message || 'Backend unreachable' };
  }
}

export async function fetchPYQAudit(workspace_id) {
  if (!workspace_id) return {};
  try {
    const res = await fetch(`${API_BASE_URL}/workspaces/${workspace_id}/audit`);
    if (!res.ok) {
      const failure = await describeFailure(res, 'Audit log unavailable');
      return { unavailable: true, error: failure.message };
    }
    return await res.json();
  } catch (err) {
    console.warn('Fetch audit log error', err);
    return { unavailable: true, error: err?.message || 'Backend unreachable' };
  }
}

export async function deleteDocument(workspaceId, fileId, docType = 'pyq', filename = '') {
  if (!workspaceId || !fileId) return null;
  try {
    const url = new URL(`${API_BASE_URL}/workspaces/${workspaceId}/documents/${fileId}`);
    url.searchParams.set('doc_type', docType);
    if (filename) url.searchParams.set('filename', filename);
    const res = await fetch(url.toString(), { method: 'DELETE' });
    if (!res.ok) {
      const failure = await describeFailure(res, 'Delete document failed');
      throw new Error(failure.message);
    }
    return await res.json();
  } catch (err) {
    console.error('Delete document error', err);
    throw err;
  }
}

