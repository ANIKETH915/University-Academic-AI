import React from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import PageHeader from '../components/PageHeader';
import { Rocket, Cpu, Flame, GraduationCap, ArrowRight, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react';

export default function DemoPage({ setActiveTab, setSelectedTopicForAsk }) {
  const { activeWorkspace, activeWorkspaceId, setIsSelectorOpen } = useWorkspace();

  const steps = [
    {
      step: '1',
      title: 'Ask a Grounded Exam Question',
      desc: 'Test 10-mark answer generation with strict citations back to your uploaded syllabus and PYQs.',
      action: 'Open Ask →',
      target: 'ask',
      accent: 'border-blue-500/30 text-blue-400 bg-blue-500/10'
    },
    {
      step: '2',
      title: 'Analyze PYQ Pattern Recurrence',
      desc: 'Inspect semantic question clustering, recurrence frequency across your papers, and mark distributions.',
      action: 'Explore PYQ Intelligence →',
      target: 'pyq-analysis',
      accent: 'border-purple-500/30 text-purple-400 bg-purple-500/10'
    },
    {
      step: '3',
      title: 'Discover Study Priorities',
      desc: 'View exam topics ranked by evidence-based priority scoring over your uploaded papers.',
      action: 'View Study Priority →',
      target: 'study-priority',
      accent: 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10'
    },
    {
      step: '4',
      title: 'Explain Priority Rationale',
      desc: 'Open the "Why this priority?" explainability modal to break down frequency, recency, and mark weights.',
      action: 'Test Explainability →',
      target: 'pyq-analysis',
      accent: 'border-amber-500/30 text-amber-400 bg-amber-500/10'
    }
  ];

  const handleStepClick = (target, prompt) => {
    if (prompt && setSelectedTopicForAsk) {
      setSelectedTopicForAsk(prompt);
    }
    setActiveTab(target);
  };

  return (
    <div className="space-y-6">
      
      {/* Standardized Page Header */}
      <PageHeader
        breadcrumb="Guided Walkthrough"
        title="3-Minute Live Presentation Script"
        subtitle="A guided tour of the assistant. Every step runs against your currently selected workspace."
        icon={Rocket}
        badge="Walkthrough"
      />

      {/* The walkthrough operates on real uploaded data, never a preloaded fixture. */}
      {activeWorkspaceId ? (
        <div className="p-4 rounded-xl bg-white dark:bg-[#0B1020] border border-slate-200 dark:border-[#1F2937] text-slate-700 dark:text-slate-300 text-xs flex items-center justify-between gap-3 shadow-xs">
          <span>
            Running against:{' '}
            <strong className="text-slate-900 dark:text-white">
              {activeWorkspace.university || 'Workspace'} / {activeWorkspace.subject || 'Subject'} ({activeWorkspace.semester || 'Sem'})
            </strong>
            <span className="ml-2 text-slate-500 font-mono">[{activeWorkspaceId}]</span>
          </span>
          <button
            onClick={() => setIsSelectorOpen(true)}
            className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-[#111827] hover:bg-slate-200 dark:hover:bg-[#1F2937] border border-slate-200 dark:border-[#1F2937] text-purple-600 dark:text-purple-300 text-xs font-semibold flex-shrink-0 transition-colors"
          >
            Switch Workspace
          </button>
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex items-center justify-between gap-3">
          <span className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>No workspace selected. Create one and upload your documents to run this walkthrough.</span>
          </span>
          <button
            onClick={() => setActiveTab('workspace')}
            className="px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs flex-shrink-0 transition-colors"
          >
            Upload Documents
          </button>
        </div>
      )}

      {/* Demo Workflow Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {steps.map((s, idx) => (
          <div key={idx} className="saas-card saas-card-hover p-6 flex flex-col justify-between space-y-4">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="w-8 h-8 rounded-xl bg-purple-600/20 text-purple-300 font-extrabold text-sm flex items-center justify-center border border-purple-500/30">
                  #{s.step}
                </span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                  GUIDED STEP
                </span>
              </div>

              <h3 className="font-heading font-bold text-base text-white">{s.title}</h3>
              <p className="text-xs text-slate-400 mt-1 leading-normal">{s.desc}</p>
            </div>

            <button
              onClick={() => handleStepClick(s.target, s.prompt)}
              className="w-full px-4 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-md shadow-purple-500/20 flex items-center justify-center space-x-1.5 transition-all"
            >
              <span>{s.action}</span>
            </button>
          </div>
        ))}
      </div>

    </div>
  );
}
