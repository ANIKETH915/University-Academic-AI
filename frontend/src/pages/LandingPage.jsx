import React, { useState } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { 
  GraduationCap, 
  Upload, 
  Cpu, 
  Flame, 
  ShieldCheck, 
  Sparkles, 
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Database
} from 'lucide-react';

export default function LandingPage({ setActiveTab }) {
  const { createNewWorkspace, setIsSelectorOpen, workspaces } = useWorkspace();
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState(null);

  // Navigate only after the backend has returned a canonical id, otherwise the
  // upload page can fire an ingest before it knows where to send it.
  const handleStartNewWorkspace = async () => {
    setCreating(true);
    setCreateError(null);
    try {
      await createNewWorkspace({
        university: 'My University',
        branch: 'Computer Science',
        semester: 'Semester 5',
        subject: 'My Academic Subject',
        subject_code: 'SUB501'
      });
      setActiveTab('workspace');
    } catch (err) {
      setCreateError(err?.message || 'Could not create a workspace. Is the backend running?');
    } finally {
      setCreating(false);
    }
  };

  const handleBrowseWorkspaces = () => {
    if (!workspaces.length) {
      setActiveTab('workspace');
      return;
    }
    setIsSelectorOpen(true);
  };

  return (
    <div className="space-y-12 py-4">
      
      {/* Hero Section */}
      <div className="saas-card p-8 sm:p-12 text-center space-y-6 relative overflow-hidden bg-gradient-to-b from-purple-100/60 via-slate-50 to-slate-100 dark:from-purple-950/30 dark:via-[#0B1020] dark:to-[#0B1020] border-slate-200 dark:border-purple-500/20">
        
        {/* Top Product Tag Pill */}
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-700 dark:text-purple-300 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
          <span>Grounded Academic AI Assistant</span>
        </div>

        {/* Main Headline */}
        <div className="max-w-3xl mx-auto space-y-3">
          <h1 className="font-heading font-extrabold text-3xl sm:text-5xl text-slate-900 dark:text-white tracking-tight leading-tight">
            Your Syllabus. Your PYQs.<br />
            <span className="text-gradient">Your Exam Intelligence.</span>
          </h1>
          <p className="text-sm sm:text-base text-slate-600 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Upload your official syllabus and previous-year question papers. Ask questions and get structured, exam-ready answers grounded strictly in your uploaded documents.
          </p>
        </div>

        {/* Dual CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
          <button
            onClick={handleStartNewWorkspace}
            disabled={creating}
            className="w-full sm:w-auto px-7 py-3.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-bold text-sm shadow-xl shadow-purple-500/25 flex items-center justify-center space-x-2 transition-all transform hover:-translate-y-0.5"
          >
            <Upload className="w-4 h-4" />
            <span>{creating ? 'Creating workspace…' : 'Start New Academic Workspace'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={handleBrowseWorkspaces}
            className="w-full sm:w-auto px-6 py-3.5 rounded-xl bg-white dark:bg-[#111827] hover:bg-slate-100 dark:hover:bg-[#1F2937] border border-slate-200 dark:border-[#1F2937] text-slate-700 dark:text-slate-300 font-semibold text-sm flex items-center justify-center space-x-2 transition-colors"
          >
            <Database className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <span>{workspaces.length ? 'Open Existing Workspace' : 'Upload Your Documents'}</span>
          </button>
        </div>

        {createError && (
          <p className="text-xs text-rose-600 dark:text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-xl px-4 py-2.5 max-w-md mx-auto">
            {createError}
          </p>
        )}

        {/* Trust Badges */}
        <div className="pt-6 border-t border-slate-200 dark:border-[#1F2937]/60 grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto text-xs text-slate-600 dark:text-slate-400">
          <div className="flex items-center justify-center space-x-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            <span>100% Grounded Answers</span>
          </div>
          <div className="flex items-center justify-center space-x-1.5">
            <BookOpen className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <span>Exact Page Citations</span>
          </div>
          <div className="flex items-center justify-center space-x-1.5">
            <Flame className="w-4 h-4 text-amber-600 dark:text-amber-400" />
            <span>PYQ Recurrence Analysis</span>
          </div>
        </div>

      </div>

      {/* 3-Step Simple User Workflow */}
      <div className="space-y-6">
        <div className="text-center space-y-1">
          <h2 className="font-heading font-extrabold text-xl text-slate-900 dark:text-white">How It Works</h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">Build your private academic assistant in 3 simple steps</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
          
          <div className="saas-card p-6 space-y-3 relative group hover:border-purple-500/30 transition-all">
            <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-600 dark:text-purple-400 font-extrabold flex items-center justify-center text-sm">
              1
            </div>
            <h3 className="font-heading font-bold text-sm text-slate-900 dark:text-white">Upload Documents</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Upload your official syllabus PDF and past question papers. The assistant reads and understands your course structure.
            </p>
          </div>

          <div className="saas-card p-6 space-y-3 relative group hover:border-purple-500/30 transition-all">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-400 font-extrabold flex items-center justify-center text-sm">
              2
            </div>
            <h3 className="font-heading font-bold text-sm text-slate-900 dark:text-white">Ask Questions</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Ask any question in 2-Mark, 5-Mark, or 10-Mark exam formats. Select your desired answer detail level.
            </p>
          </div>

          <div className="saas-card p-6 space-y-3 relative group hover:border-purple-500/30 transition-all">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-extrabold flex items-center justify-center text-sm">
              3
            </div>
            <h3 className="font-heading font-bold text-sm text-slate-900 dark:text-white">Get Grounded Answers</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Receive structured exam explanations, HTML comparison tables, PYQ recurrence indicators, and page citations.
            </p>
          </div>

        </div>
      </div>

    </div>
  );
}
