import React from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { Menu, ChevronRight, Layers, Plus } from 'lucide-react';

export default function Header({ activeTab, onOpenMobileSidebar }) {
  const { activeWorkspace, setIsSelectorOpen, setIsCreateOpen } = useWorkspace();

  const getTabTitles = () => {
    switch (activeTab) {
      case 'landing':
        return { breadcrumb: 'Overview', title: 'System Overview', subtitle: 'Universal AI platform for any university syllabus and previous-year papers.' };
      case 'dashboard':
        return { breadcrumb: 'Dashboard', title: 'Academic Dashboard', subtitle: `AI insights for ${activeWorkspace?.subject || 'your subject'} (${activeWorkspace?.semester || 'Sem'}).` };
      case 'workspace':
        return { breadcrumb: 'Knowledge Base', title: 'Knowledge Base & Uploads', subtitle: `Isolated document storage for ${activeWorkspace?.subject || 'your workspace'}.` };
      case 'ask':
        return { breadcrumb: 'Ask AI', title: 'Ask Academic AI', subtitle: `Grounded answer generation strictly scoped to ${activeWorkspace?.subject || 'your documents'}.` };
      case 'pyq-analysis':
        return { breadcrumb: 'PYQ Intelligence', title: 'PYQ Pattern Intelligence', subtitle: `Historical question analysis for ${activeWorkspace?.subject || 'your subject'}.` };
      case 'study-priority':
        return { breadcrumb: 'Study Priority', title: 'Study Priority Planner', subtitle: `Time-budgeted topic priority for ${activeWorkspace?.subject || 'your subject'}.` };
      case 'demo':
        return { breadcrumb: 'Hackathon Demo', title: '3-Minute Live Demo Script', subtitle: 'Evaluator presentation script pre-loaded with workspace documents.' };
      default:
        return { breadcrumb: 'Dashboard', title: 'Academic Dashboard', subtitle: 'AI-powered academic insights.' };
    }
  };

  const { breadcrumb, title, subtitle } = getTabTitles();

  return (
    <header className="sticky top-0 z-30 bg-[#080B14]/95 backdrop-blur-md border-b border-[#1F2937] px-4 sm:px-8 py-3.5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        
        {/* Left: Mobile Drawer Trigger + Breadcrumb + Heading */}
        <div className="flex items-center space-x-3">
          <button
            onClick={onOpenMobileSidebar}
            className="p-2 rounded-lg bg-[#111827] border border-[#1F2937] text-slate-300 hover:text-white lg:hidden"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div>
            <div className="flex items-center space-x-1.5 text-xs text-slate-400">
              <span className="font-medium text-slate-400">University Academic AI</span>
              <ChevronRight className="w-3 h-3 text-slate-600" />
              <span className="text-purple-400 font-semibold">{breadcrumb}</span>
            </div>
            <h2 className="font-heading font-extrabold text-lg sm:text-xl text-white tracking-tight leading-tight mt-0.5">
              {title}
            </h2>
          </div>
        </div>

        {/* Right: Academic Workspace Selector Badge & Button */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          
          {/* Workspace Switcher Trigger Card */}
          <button
            type="button"
            onClick={() => setIsSelectorOpen(true)}
            className="bg-[#0B1020] hover:bg-[#111827] border border-purple-500/40 hover:border-purple-500 rounded-xl px-3.5 py-2 text-xs flex items-center space-x-2.5 transition-all text-left group shadow-sm"
          >
            <div className="p-1 rounded-lg bg-purple-500/20 text-purple-300 group-hover:scale-105 transition-transform">
              <Layers className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider flex items-center space-x-1">
                <span>Current Workspace</span>
                <span className="text-purple-400">▼</span>
              </div>
              <div className="font-heading font-bold text-white text-xs truncate max-w-[180px] sm:max-w-[220px]">
                {activeWorkspace?.university || 'No workspace'} / {activeWorkspace?.subject || 'Select or create'}
              </div>
            </div>
          </button>

          {/* Quick Create Workspace CTA */}
          <button
            type="button"
            onClick={() => setIsCreateOpen(true)}
            className="p-2.5 rounded-xl bg-purple-600/20 hover:bg-purple-600 text-purple-300 hover:text-white border border-purple-500/30 transition-all flex items-center justify-center"
            title="Create New Workspace"
          >
            <Plus className="w-4 h-4" />
          </button>

        </div>

      </div>
    </header>
  );
}
