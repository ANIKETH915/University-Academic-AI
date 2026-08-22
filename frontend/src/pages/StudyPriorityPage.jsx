import React, { useState, useEffect } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { fetchStudyPriority } from '../api/academicApi';
import PageHeader from '../components/PageHeader';
import PriorityCard from '../components/PriorityCard';
import { GraduationCap, ShieldAlert, Loader2, Layers, Upload, Database, AlertCircle } from 'lucide-react';

export default function StudyPriorityPage({ setActiveTab, setSelectedTopicForAsk }) {
  const { activeWorkspace, setIsSelectorOpen } = useWorkspace();
  const [timeBudget, setTimeBudget] = useState('2 Days');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const isInvalidWorkspace = !activeWorkspace || !activeWorkspace.id || activeWorkspace.id === '/' || activeWorkspace.id === 'undefined' || activeWorkspace.id === 'null';

  useEffect(() => {
    async function loadPriority() {
      setData(null);

      if (isInvalidWorkspace) {
        setLoading(false);
        return;
      }

      setLoading(true);
      console.log('[PRIORITY] workspace_id=', activeWorkspace.id);

      const res = await fetchStudyPriority(
        activeWorkspace.subject || 'Subject',
        activeWorkspace.semester || 'Semester',
        5,
        true,
        activeWorkspace.id
      );

      setData(res);
      setLoading(false);
    }

    loadPriority();
  }, [activeWorkspace?.id, activeWorkspace?.subject, activeWorkspace?.semester, activeWorkspace?.pyqFiles?.length]);

  const handleStudyTopic = (topicName) => {
    if (setSelectedTopicForAsk) {
      setSelectedTopicForAsk(topicName);
    }
    setActiveTab('ask');
  };

  return (
    <div className="space-y-6">
      
      {/* Standardized Page Header */}
      <PageHeader
        breadcrumb="Study Priority"
        title={`Study Priority Planner (${activeWorkspace?.subject || 'Your Subject'})`}
        subtitle={`Focus your limited study time on historically important topics for ${activeWorkspace?.subject || 'your workspace'}.`}
        icon={GraduationCap}
        badge="Priority Scoring"
      />

      {/* Target Scope Banner */}
      <div className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] text-slate-300 text-xs flex items-center justify-between gap-3">
        <div>
          <span className="font-semibold text-white">Target Workspace:</span> Topic priorities ranked for <strong className="text-purple-300">{activeWorkspace?.university || 'Workspace'} / {activeWorkspace?.subject || 'Subject'} ({activeWorkspace?.semester || 'Sem'})</strong>.
        </div>

        <button
          onClick={() => setIsSelectorOpen(true)}
          className="px-3 py-1.5 rounded-lg bg-[#111827] hover:bg-[#1F2937] border border-[#1F2937] text-purple-300 text-xs font-semibold flex items-center space-x-1 transition-colors flex-shrink-0"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Switch Workspace</span>
        </button>
      </div>

      {/* INVALID WORKSPACE STATE */}
      {isInvalidWorkspace ? (
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center mx-auto">
            <AlertCircle className="w-8 h-8" />
          </div>

          <div className="space-y-1">
            <h3 className="font-heading font-extrabold text-lg text-white">
              No active workspace selected
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Please create or select an active academic workspace to calculate topic priorities.
            </p>
          </div>

          <button
            onClick={() => setIsSelectorOpen(true)}
            className="px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/20 inline-flex items-center space-x-2 transition-all"
          >
            <Layers className="w-4 h-4" />
            <span>Select Workspace</span>
          </button>
        </div>
      ) : loading ? (
        <div className="p-12 text-center text-slate-400 saas-card">
          <Loader2 className="w-8 h-8 animate-spin mx-auto text-emerald-400 mb-2" />
          <p className="text-xs">Computing study priority for {activeWorkspace.subject || 'uploaded papers'}...</p>
        </div>
      ) : data?.unavailable ? (
        /* A failed request is not the same as an empty workspace. */
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center mx-auto">
            <ShieldAlert className="w-8 h-8" />
          </div>
          <div className="space-y-1">
            <h3 className="font-heading font-extrabold text-lg text-white">
              Study priority could not be calculated
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              {data.error || 'The backend rejected the request.'}
            </p>
            <p className="text-[11px] text-slate-500 pt-1">
              This is a backend error, not an empty workspace.
            </p>
          </div>
        </div>
      ) : !(data?.top_high_priority_topics || []).length ? (
        /* EMPTY STATE: NO PYQS IN WORKSPACE */
        <div className="saas-card p-10 text-center space-y-4 max-w-2xl mx-auto my-6">
          <div className="w-16 h-16 rounded-3xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto">
            <GraduationCap className="w-8 h-8" />
          </div>

          <div className="space-y-1">
            <h3 className="font-heading font-extrabold text-lg text-white">
              No PYQ data available for Study Priority
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Add previous-year question papers for <strong className="text-white">{activeWorkspace.subject || 'your subject'}</strong> to generate time-budgeted study priority rankings.
            </p>
          </div>

          <button
            onClick={() => setActiveTab('workspace')}
            className="px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/20 inline-flex items-center space-x-2 transition-all"
          >
            <Upload className="w-4 h-4" />
            <span>Upload PYQs</span>
          </button>
        </div>
      ) : (
        <>
          {/* Time Budget Selection Bar */}
          <div className="saas-card p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h3 className="font-heading font-bold text-sm text-white">Select Study Time Budget</h3>
              <p className="text-xs text-slate-400">Ranks topics dynamically based on your available preparation window.</p>
            </div>

            <div className="flex space-x-2">
              {['1 Day', '2 Days', '3 Days', '7 Days'].map(t => (
                <button
                  key={t}
                  onClick={() => setTimeBudget(t)}
                  className={`px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
                    timeBudget === t
                      ? 'bg-purple-600 border-purple-500 text-white shadow-md glow-purple'
                      : 'bg-[#0B1020] border-[#1F2937] text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Disclaimer Banner */}
          <div className="p-4 rounded-xl bg-[#0B1020] border border-[#1F2937] text-slate-300 text-xs flex items-center space-x-3">
            <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>
              <strong>Disclaimer:</strong> Historical priority is based on past examination patterns and does not predict the next paper.
            </span>
          </div>

          {/* Ranked Priority Cards List */}
          {loading ? (
            <div className="p-12 text-center text-slate-400 saas-card">
              <Loader2 className="w-8 h-8 animate-spin mx-auto text-emerald-400 mb-2" />
              <p className="text-xs">Computing optimal study priority for {activeWorkspace.subject || 'uploaded papers'} ({timeBudget} budget)...</p>
            </div>
          ) : (
            <div className="space-y-4">
              <h3 className="font-heading font-bold text-xs text-white uppercase tracking-wider">
                Ranked Topics in {activeWorkspace.subject || 'Uploaded Papers'} ({timeBudget} Budget)
              </h3>

              <div className="space-y-3">
                {data?.top_high_priority_topics?.map((topic, idx) => (
                  <PriorityCard
                    key={idx}
                    rank={idx + 1}
                    topic={topic}
                    onStudy={(topicName) => handleStudyTopic(topicName)}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}

    </div>
  );
}
