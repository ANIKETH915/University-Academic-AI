import React from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { Layers, Plus, Check, X, GraduationCap, BookOpen, Sparkles } from 'lucide-react';

export default function WorkspaceSelectorModal() {
  const {
    workspaces,
    activeWorkspaceId,
    switchWorkspace,
    isSelectorOpen,
    setIsSelectorOpen,
    setIsCreateOpen
  } = useWorkspace();

  if (!isSelectorOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="saas-card w-full max-w-xl p-6 space-y-5 animate-in fade-in zoom-in-95 duration-150">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-[#1F2937] pb-4">
          <div className="flex items-center space-x-2 text-purple-400">
            <Layers className="w-5 h-5" />
            <div>
              <h3 className="font-heading font-extrabold text-base text-white">MY ACADEMIC WORKSPACES</h3>
              <p className="text-xs text-slate-400">Select an isolated subject workspace or create a new one.</p>
            </div>
          </div>
          <button
            onClick={() => setIsSelectorOpen(false)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-[#1F2937] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Workspaces List */}
        <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
          {workspaces.map((ws, idx) => {
            const isActive = ws.id === activeWorkspaceId;
            return (
              <div
                key={`${ws.id || 'ws'}-${idx}`}
                onClick={() => switchWorkspace(ws.id)}
                className={`p-4 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                  isActive
                    ? 'bg-purple-950/30 border-purple-500 text-white shadow-lg glow-purple'
                    : 'bg-[#0B1020] border-[#1F2937] text-slate-300 hover:border-slate-700 hover:bg-[#0B1020]/80'
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-heading font-bold text-sm text-white">{ws.subject}</span>
                    {ws.isDemo && (
                      <span className="text-[9px] font-extrabold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        PRELOADED DEMO
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-400 flex items-center space-x-2">
                    <span className="text-purple-300 font-medium">{ws.university}</span>
                    <span>•</span>
                    <span>{ws.branch}</span>
                    <span>•</span>
                    <span className="text-emerald-400">{ws.semester}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  {isActive ? (
                    <span className="text-xs font-bold px-2.5 py-1 rounded bg-purple-600 text-white flex items-center space-x-1">
                      <Check className="w-3.5 h-3.5" />
                      <span>Active</span>
                    </span>
                  ) : (
                    <span className="text-xs font-semibold text-slate-400 hover:text-white">
                      Select
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Modal Actions */}
        <div className="pt-2 border-t border-[#1F2937] flex items-center justify-between">
          <span className="text-xs text-slate-400">
            {workspaces.length} workspace{workspaces.length > 1 ? 's' : ''} available
          </span>

          <button
            onClick={() => {
              setIsSelectorOpen(false);
              setIsCreateOpen(true);
            }}
            className="px-4 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-md shadow-purple-500/20 flex items-center space-x-2 transition-all"
          >
            <Plus className="w-4 h-4" />
            <span>Create New Workspace</span>
          </button>
        </div>

      </div>
    </div>
  );
}
