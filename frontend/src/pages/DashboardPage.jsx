import React from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import { 
  GraduationCap, 
  BookOpen, 
  Database, 
  Cpu, 
  Flame, 
  CheckCircle2, 
  ArrowRight,
  Plus,
  Layers,
  Upload
} from 'lucide-react';

export default function DashboardPage({ stats, setActiveTab }) {
  const { activeWorkspace, setIsSelectorOpen, setIsCreateOpen } = useWorkspace();

  const sylChunks = activeWorkspace.syllabusFiles.length;
  const pyqChunks = activeWorkspace.pyqFiles.length;
  const totalFiles = sylChunks + pyqChunks;
  const hasDocuments = totalFiles > 0;

  const quickActions = [
    {
      id: 'ask',
      title: 'Ask Academic AI',
      desc: `Get grounded answers for ${activeWorkspace.subject || 'your subject'}.`,
      icon: Cpu,
      accent: 'border-blue-500/30 text-blue-400 bg-blue-500/10'
    },
    {
      id: 'pyq-analysis',
      title: 'PYQ Intelligence',
      desc: `Explore question frequency for ${activeWorkspace.subject || 'your subject'}.`,
      icon: Flame,
      accent: 'border-purple-500/30 text-purple-400 bg-purple-500/10'
    },
    {
      id: 'study-priority',
      title: 'Study Priority',
      desc: `Build a focused study plan for ${activeWorkspace.subject || 'your subject'}.`,
      icon: GraduationCap,
      accent: 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10'
    },
    {
      id: 'workspace',
      title: 'Knowledge Base',
      desc: `Upload syllabus and PYQs for ${activeWorkspace.subject || 'your subject'}.`,
      icon: Database,
      accent: 'border-amber-500/30 text-amber-400 bg-amber-500/10'
    }
  ];

  return (
    <div className="space-y-6">
      
      {/* Standardized Page Header */}
      <PageHeader
        breadcrumb="Dashboard"
        title="Academic Dashboard"
        subtitle={`AI-powered insights for ${activeWorkspace.university || 'Academic Workspace'} - ${activeWorkspace.subject || 'Subject'}.`}
        icon={Layers}
        badge="Active Workspace"
      />

      {/* Active Workspace Banner */}
      <div className="saas-card p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-gradient-to-r from-purple-100/80 via-indigo-50/80 to-slate-100 dark:from-purple-950/30 dark:via-indigo-950/20 dark:to-[#111827] border-slate-200 dark:border-[#1F2937]">
        <div>
          <div className="text-[10px] text-purple-600 dark:text-purple-400 font-extrabold uppercase tracking-wider">
            Current Workspace Scope
          </div>
          <h2 className="font-heading font-extrabold text-xl text-slate-900 dark:text-white mt-0.5">
            {activeWorkspace.subject || 'My Academic Subject'} ({activeWorkspace.subjectCode || 'SUB'})
          </h2>
          <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
            {activeWorkspace.university || 'University'} • {activeWorkspace.branch || 'Branch'} • {activeWorkspace.semester || 'Semester'}
          </p>
        </div>

        <div className="flex items-center space-x-2 flex-shrink-0">
          <button
            onClick={() => setIsSelectorOpen(true)}
            className="px-4 py-2 rounded-xl bg-white dark:bg-[#0B1020] hover:bg-slate-100 dark:hover:bg-[#1F2937] border border-slate-200 dark:border-[#1F2937] text-slate-700 dark:text-slate-200 text-xs font-semibold flex items-center space-x-1.5 transition-colors"
          >
            <Layers className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
            <span>Switch Workspace</span>
          </button>

          <button
            onClick={() => setIsCreateOpen(true)}
            className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold shadow-md shadow-purple-500/20 flex items-center space-x-1.5 transition-all"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Create Workspace</span>
          </button>
        </div>
      </div>

      {/* EMPTY STATE BANNER IF ZERO FILES */}
      {!hasDocuments && (
        <div className="p-4 rounded-xl bg-purple-500/10 border border-purple-500/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-purple-500/20 text-purple-700 dark:text-purple-300">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <div className="font-bold text-slate-900 dark:text-white">Your Knowledge Base is Empty</div>
              <div className="text-slate-600 dark:text-slate-400">Upload a syllabus or previous-year papers for {activeWorkspace.subject || 'your subject'} to start building academic intelligence.</div>
            </div>
          </div>
          <button
            onClick={() => setActiveTab('workspace')}
            className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-md flex items-center space-x-1.5 flex-shrink-0"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Files Now</span>
          </button>
        </div>
      )}

      {/* 4 Statistics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="SYLLABUS DOCUMENTS"
          value={`${sylChunks} ${sylChunks === 1 ? 'File' : 'Files'}`}
          subtext={sylChunks > 0 ? 'Official Course Breakdown' : 'No Syllabus Uploaded'}
          icon={GraduationCap}
          accentColor={sylChunks > 0 ? 'emerald' : 'purple'}
        />
        <StatCard
          label="PYQ QUESTION PAPERS"
          value={`${pyqChunks} ${pyqChunks === 1 ? 'Paper' : 'Papers'}`}
          subtext={pyqChunks > 0 ? 'Verified Exam Papers' : 'No PYQs Uploaded'}
          icon={BookOpen}
          accentColor={pyqChunks > 0 ? 'blue' : 'purple'}
        />
        <StatCard
          label="INDEXED DOCUMENTS"
          value={hasDocuments ? `${totalFiles} Uploaded` : '0 Documents'}
          subtext="Workspace Knowledge Base"
          icon={Database}
          accentColor="purple"
        />
        <StatCard
          label="RAG STATUS"
          value={hasDocuments ? 'Active' : 'Empty'}
          subtext={hasDocuments ? 'Grounded Retrieval Active' : 'Awaiting Document Ingestion'}
          icon={Cpu}
          accentColor={hasDocuments ? 'amber' : 'purple'}
        />
      </div>

      {/* Quick Actions Grid */}
      <div className="saas-card p-6 space-y-4">
        <div>
          <h3 className="font-heading font-bold text-sm text-slate-900 dark:text-white">Academic Actions for {activeWorkspace.subject || 'Your Subject'}</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">Select an isolated workflow tool.</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {quickActions.map((act) => {
            const Icon = act.icon;
            return (
              <div
                key={act.id}
                onClick={() => setActiveTab(act.id)}
                className="saas-card saas-card-hover p-4 cursor-pointer flex flex-col justify-between group"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className={`p-2 rounded-xl border ${act.accent}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors" />
                  </div>
                  <h4 className="font-heading font-bold text-xs text-slate-900 dark:text-white group-hover:text-purple-600 dark:group-hover:text-purple-300 transition-colors">
                    {act.title}
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 leading-normal">
                    {act.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}
