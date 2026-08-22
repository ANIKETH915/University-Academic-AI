import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const WorkspaceContext = createContext();

const API_BASE = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000';

/** Placeholder only when backend list is empty — never used as a silent ingest target. */
const EMPTY_WORKSPACE = {
  id: '',
  university: '',
  branch: '',
  program: '',
  semester: '',
  subject: '',
  subjectCode: '',
  isDemo: false,
  syllabusFiles: [],
  pyqFiles: []
};

function formatWorkspace(w) {
  return {
    id: w.id,
    university: w.university,
    branch: w.branch,
    program: w.program || w.branch,
    semester: w.semester,
    subject: w.subject,
    subjectCode: w.subject_code || w.subjectCode || '',
    isDemo: w.is_demo ?? w.isDemo ?? false,
    syllabusFiles: w.syllabus_files || w.syllabusFiles || [],
    pyqFiles: w.pyq_files || w.pyqFiles || []
  };
}

function readSavedActiveId() {
  try {
    const savedId = localStorage.getItem('academic_active_workspace_id');
    if (savedId && savedId !== '/' && savedId !== 'null' && savedId !== 'undefined' && savedId.trim() !== '') {
      // Never treat the legacy fake default as a real selection
      if (savedId === 'ws-default-workspace') return '';
      return savedId;
    }
  } catch (e) {
    console.warn('Failed to load activeWorkspaceId from localStorage', e);
  }
  return '';
}

export function WorkspaceProvider({ children }) {
  const [workspaces, setWorkspaces] = useState([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState(readSavedActiveId);
  const [isSelectorOpen, setIsSelectorOpen] = useState(false);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [workspacesLoaded, setWorkspacesLoaded] = useState(false);

  const persistActiveId = useCallback((id) => {
    if (!id || id === '/' || id === 'undefined' || id === 'null' || id === 'ws-default-workspace') return;
    try {
      localStorage.setItem('academic_active_workspace_id', id);
    } catch (e) {
      console.warn('Failed to save activeWorkspaceId', e);
    }
  }, []);

  const refreshWorkspaces = useCallback(async () => {
    const res = await fetch(`${API_BASE}/workspaces`);
    if (!res.ok) throw new Error(`Status ${res.status}`);
    const backendWorkspaces = await res.json();
    const rawFormatted = Array.isArray(backendWorkspaces)
      ? backendWorkspaces.map(formatWorkspace)
      : [];

    const seenIds = new Set();
    const backendFormatted = rawFormatted.filter((w) => {
      if (!w || !w.id || seenIds.has(w.id)) return false;
      seenIds.add(w.id);
      return true;
    });

    setWorkspaces(backendFormatted);
    setWorkspacesLoaded(true);

    setActiveWorkspaceId((current) => {
      const saved = readSavedActiveId() || current;
      if (saved && backendFormatted.some((w) => w.id === saved)) {
        persistActiveId(saved);
        return saved;
      }
      if (current && backendFormatted.some((w) => w.id === current)) {
        persistActiveId(current);
        return current;
      }
      const fallbackId = backendFormatted[0]?.id || '';
      if (fallbackId) persistActiveId(fallbackId);
      return fallbackId;
    });

    return backendFormatted;
  }, [persistActiveId]);

  // Fetch & sync workspaces from backend on mount (single source of truth)
  useEffect(() => {
    let cancelled = false;
    refreshWorkspaces()
      .catch((err) => {
        if (!cancelled) {
          console.warn('[WORKSPACE BACKEND FETCH WARNING]', err);
          setWorkspacesLoaded(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [refreshWorkspaces]);

  useEffect(() => {
    persistActiveId(activeWorkspaceId);
  }, [activeWorkspaceId, persistActiveId]);

  // A saved id that the backend does not know about is stale, not a workspace.
  useEffect(() => {
    if (!workspacesLoaded || !activeWorkspaceId) return;
    if (!workspaces.some((w) => w && w.id === activeWorkspaceId)) {
      try {
        localStorage.removeItem('academic_active_workspace_id');
      } catch (e) {
        console.warn('Failed to clear stale activeWorkspaceId', e);
      }
      setActiveWorkspaceId('');
    }
  }, [workspacesLoaded, workspaces, activeWorkspaceId]);

  // Before the backend list arrives an id may legitimately be unresolved; once it
  // has arrived, an id with no backing workspace is stale and must not look real.
  const activeWorkspace =
    workspaces.find((w) => w && w.id === activeWorkspaceId) ||
    (activeWorkspaceId && !workspacesLoaded
      ? { ...EMPTY_WORKSPACE, id: activeWorkspaceId }
      : EMPTY_WORKSPACE);

  const switchWorkspace = (id) => {
    if (!id || id === '/' || id === 'undefined' || id === 'null' || id === 'ws-default-workspace') return;
    // The backend owns the workspace list. Activating an id it does not know
    // about would point every later ingest/analysis call at a 404.
    if (workspacesLoaded && !workspaces.some((w) => w && w.id === id)) {
      console.warn('[WORKSPACE] refusing to activate unknown workspace id', id);
      return;
    }
    setActiveWorkspaceId(id);
    persistActiveId(id);
    setIsSelectorOpen(false);
  };

  const setWorkspaceById = (id) => {
    switchWorkspace(id);
  };

  const createWorkspace = async (params = {}) => {
    const res = await fetch(`${API_BASE}/workspaces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        university: params.university || 'Academic Institution',
        branch: params.branch || params.program || 'Computer Engineering',
        semester: params.semester || 'Semester 1',
        subject: params.subject || 'Academic Subject',
        subject_code: params.subjectCode || params.subject_code || 'SUB'
      })
    });
    if (!res.ok) {
      throw new Error(`Workspace create failed: ${res.status}`);
    }
    const data = await res.json();
    if (!data?.id) {
      throw new Error('Backend did not return a workspace id');
    }

    const canonical = formatWorkspace(data);
    console.log('[WORKSPACE] backend_returned_id=', canonical.id, 'subject=', canonical.subject);

    setWorkspaces((prev) => [canonical, ...prev.filter((w) => w.id !== canonical.id)]);
    setActiveWorkspaceId(canonical.id);
    persistActiveId(canonical.id);
    setIsCreateOpen(false);
    setIsSelectorOpen(false);

    // Re-sync from backend so file lists match server truth
    try {
      await refreshWorkspaces();
      setActiveWorkspaceId(canonical.id);
      persistActiveId(canonical.id);
    } catch (e) {
      console.warn('[WORKSPACE] refresh after create failed', e);
    }

    console.log('[WORKSPACE] frontend_active_id=', canonical.id);
    return canonical;
  };

  const updateWorkspaceMetadata = (workspaceId, metadata) => {
    setWorkspaces((prev) =>
      prev.map((w) => (w.id === workspaceId ? { ...w, ...metadata } : w))
    );
  };

  const addFileToWorkspace = (workspaceId, fileInfo, docType = 'syllabus') => {
    setWorkspaces((prev) =>
      prev.map((w) => {
        if (w.id !== workspaceId) return w;
        const targetId = fileInfo.id || fileInfo.name;
        if (docType === 'syllabus') {
          const cleaned = (w.syllabusFiles || []).filter(f => (f.id || f.name) !== targetId);
          return { ...w, syllabusFiles: [fileInfo, ...cleaned] };
        }
        const cleaned = (w.pyqFiles || []).filter(f => (f.id || f.name) !== targetId);
        return { ...w, pyqFiles: [...cleaned, fileInfo] };
      })
    );
  };

  const removeFileFromWorkspace = (workspaceId, fileId, docType = 'syllabus') => {
    setWorkspaces((prev) =>
      prev.map((w) => {
        if (w.id !== workspaceId) return w;
        if (docType === 'syllabus') {
          return { ...w, syllabusFiles: w.syllabusFiles.filter((f) => f.id !== fileId) };
        }
        return { ...w, pyqFiles: w.pyqFiles.filter((f) => f.id !== fileId) };
      })
    );
  };

  return (
    <WorkspaceContext.Provider
      value={{
        workspaces,
        activeWorkspace,
        activeWorkspaceId,
        workspacesLoaded,
        switchWorkspace,
        setWorkspaceById,
        createWorkspace,
        createNewWorkspace: createWorkspace,
        refreshWorkspaces,
        updateWorkspaceMetadata,
        addFileToWorkspace,
        removeFileFromWorkspace,
        isSelectorOpen,
        setIsSelectorOpen,
        isCreateOpen,
        setIsCreateOpen
      }}
    >
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace() {
  return useContext(WorkspaceContext);
}
